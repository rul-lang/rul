//! Milestone 2c (`v0.6`): Tests for Section 5b (`link_meta`) codec & `.rls` round-trip,
//! `verify_module` vs `verify_program` fail-closed `UnresolvedImport`, and `link_modules`
//! verified multi-module linking, relocation, capability enforcement, proof ceiling, and determinism.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::aot::{
    compile_rl_to_executable, compile_to_elf64_bytes, run_compiled_structured_executable,
};
use crate::bytecode::{
    ArenaDescriptor, LinkDirection, LinkMetadata, LinkSymbol, LinkSymbolKind, NodeFrame, Opcode,
    RulFormatError, RulProgram, ScheduleHint, TokenOp, TypeLane, CAP_DIVERGE, CAP_NET_RING,
    CAP_NONE, LINK_META_MAGIC, LINK_META_VERSION, RL_HEADER_SIZE,
};
use crate::engine::{execute_tier0_interpreter, RuntimeValue};
use crate::linker::{link_modules, LinkError, LinkOptions};
use crate::transpiler::{assemble_rls, program_to_rls};
use crate::verifier::{verify_module, verify_program, VerifyErrorKind};

fn test_exe_path(tag: &str) -> std::path::PathBuf {
    static NEXT_EXE_ID: AtomicU64 = AtomicU64::new(0);
    let seq = NEXT_EXE_ID.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    std::env::temp_dir().join(format!(
        "rul_v06_link_{tag}_{}_{seq}_{nanos}.elf",
        std::process::id()
    ))
}

fn scalar_i64_from_report(val: &RuntimeValue) -> Result<i64, String> {
    match val {
        RuntimeValue::TypedScalar(ts) => ts
            .as_i64()
            .ok_or_else(|| format!("expected I64 scalar, got {ts:?}")),
        other => Err(format!("expected I64 scalar, got {other:?}")),
    }
}

fn run_tier0_eq_aot_i64(program: &RulProgram, tag: &str) -> Result<i64, String> {
    let report =
        execute_tier0_interpreter(program, &[]).map_err(|e| format!("tier0 failed: {e}"))?;
    let expected = scalar_i64_from_report(&report.output)?;

    let exe_path = test_exe_path(tag);
    compile_rl_to_executable(program, &exe_path).map_err(|e| format!("aot compile failed: {e}"))?;
    let stdout_res = run_compiled_structured_executable(&exe_path, &[]);
    let _ = std::fs::remove_file(&exe_path);
    let stdout = stdout_res.map_err(|e| format!("aot exec failed: {e}"))?;

    let bytes: [u8; 8] = stdout
        .as_slice()
        .try_into()
        .map_err(|_e| format!("expected 8-byte stdout, got {} bytes", stdout.len()))?;
    let actual = i64::from_le_bytes(bytes);
    if actual != expected {
        return Err(format!(
            "Tier 0 ({expected}) != ELF64 AOT ({actual}) for {tag}"
        ));
    }
    Ok(expected)
}

#[test]
fn test_v06_link_metadata_codec_and_rls_roundtrip() -> Result<(), String> {
    // 1. Assemble `.rls` containing all 4 export/import directives.
    let rls = "\
cap 0x0
export func std.vec:push-1 1
import func ext/math.add_2 0
export region app.buf 0
import region net.rx 1
proof 1 rule=region_decl target=0 lo=0 hi=16 witness=i64:plain
proof 2 rule=region_decl target=1 lo=0 hi=16 witness=i64:plain
proof 3 rule=func_sig target=0 lo=1 hi=i64 witness=i64
proof 4 rule=func_sig target=1 lo=1 hi=i64 witness=i64
proof 5 rule=func_region_effect func=0 slot=1 pre=owned post=owned
proof 6 rule=func_region_effect func=1 slot=0 pre=owned post=owned
proof 7 rule=func_region_effect func=1 slot=1 pre=owned post=owned
node $f1 = Func i64x1 a=1 b=0
node $p0 = LocalGet i64x1 a=0 b=0
node $res = Call i64x1 a=0 b=$p0 c=none
node $e1 = End i64x1 a=$res b=1
output $f1
";
    let prog = assemble_rls(rls).map_err(|e| format!("assemble failed: {e}"))?;
    assert!(!prog.link_meta.is_empty());

    // 2. Verify canonical Section 5b header: RLMK + version 1 + entry_count 4.
    assert_eq!(prog.link_meta.get(0..4), Some(LINK_META_MAGIC.as_slice()));
    assert_eq!(
        prog.link_meta.get(4..6),
        Some(LINK_META_VERSION.to_le_bytes().as_slice())
    );
    assert_eq!(
        prog.link_meta.get(6..8),
        Some(4u16.to_le_bytes().as_slice())
    );

    // 3. Round-trip through `.rl` binary image header[44..48] and decode_link_meta().
    let rl_bytes = prog.to_rl_bytes();
    assert_eq!(
        rl_bytes.get(44..48),
        Some(
            u32::try_from(prog.link_meta.len())
                .unwrap_or(0)
                .to_le_bytes()
                .as_slice()
        )
    );
    assert!(rl_bytes.len() > RL_HEADER_SIZE);
    let decoded_prog =
        RulProgram::from_rl_bytes(&rl_bytes).map_err(|e| format!("from_rl_bytes failed: {e}"))?;
    assert_eq!(decoded_prog, prog);

    let meta = decoded_prog
        .decode_link_meta()
        .map_err(|e| format!("decode_link_meta failed: {e}"))?;
    assert_eq!(
        meta,
        LinkMetadata {
            entries: vec![
                LinkSymbol {
                    direction: LinkDirection::Export,
                    kind: LinkSymbolKind::Func,
                    local_id: 1,
                    name: "std.vec:push-1".to_owned(),
                },
                LinkSymbol {
                    direction: LinkDirection::Import,
                    kind: LinkSymbolKind::Func,
                    local_id: 0,
                    name: "ext/math.add_2".to_owned(),
                },
                LinkSymbol {
                    direction: LinkDirection::Export,
                    kind: LinkSymbolKind::Region,
                    local_id: 0,
                    name: "app.buf".to_owned(),
                },
                LinkSymbol {
                    direction: LinkDirection::Import,
                    kind: LinkSymbolKind::Region,
                    local_id: 1,
                    name: "net.rx".to_owned(),
                },
            ],
        }
    );

    // 4. Lossless `.rls` decompile -> reassemble round-trip and identical content hash.
    let decompiled = program_to_rls(&decoded_prog);
    assert!(decompiled.contains("export func std.vec:push-1 1"));
    assert!(decompiled.contains("import func ext/math.add_2 0"));
    assert!(decompiled.contains("export region app.buf 0"));
    assert!(decompiled.contains("import region net.rx 1"));
    let reassembled =
        assemble_rls(&decompiled).map_err(|e| format!("reassemble failed: {e}\n{decompiled}"))?;
    assert_eq!(reassembled, prog);
    assert_eq!(reassembled.compute_graph_hash(), prog.compute_graph_hash());

    // 5. Empty LinkMetadata encodes to empty Vec<u8> and decodes cleanly.
    let empty_bytes = LinkMetadata::default()
        .to_bytes()
        .map_err(|e| format!("empty to_bytes failed: {e}"))?;
    assert!(empty_bytes.is_empty());
    assert_eq!(LinkMetadata::from_bytes(&[]), Ok(LinkMetadata::default()));

    // 6. Malformed Section 5b inputs fail closed with RulFormatError::InvalidLinkMetadata.
    let bad_magic = [b'B', b'A', b'D', b'!', 1, 0, 0, 0];
    assert!(matches!(
        LinkMetadata::from_bytes(&bad_magic),
        Err(RulFormatError::InvalidLinkMetadata(_))
    ));

    let bad_version = [b'R', b'L', b'M', b'K', 2, 0, 0, 0];
    assert!(matches!(
        LinkMetadata::from_bytes(&bad_version),
        Err(RulFormatError::InvalidLinkMetadata(_))
    ));

    let truncated_hdr = [b'R', b'L', b'M', b'K', 1, 0];
    assert!(matches!(
        LinkMetadata::from_bytes(&truncated_hdr),
        Err(RulFormatError::InvalidLinkMetadata(_))
    ));

    let too_many_entries = [b'R', b'L', b'M', b'K', 1, 0, 0x01, 0x10]; // 4097
    assert!(matches!(
        LinkMetadata::from_bytes(&too_many_entries),
        Err(RulFormatError::InvalidLinkMetadata(_))
    ));

    let bad_dir = [b'R', b'L', b'M', b'K', 1, 0, 1, 0, 2, 0, 0, 0, 1, 0, b'a'];
    assert!(matches!(
        LinkMetadata::from_bytes(&bad_dir),
        Err(RulFormatError::InvalidLinkMetadata(_))
    ));

    let bad_kind = [b'R', b'L', b'M', b'K', 1, 0, 1, 0, 0, 5, 0, 0, 1, 0, b'a'];
    assert!(matches!(
        LinkMetadata::from_bytes(&bad_kind),
        Err(RulFormatError::InvalidLinkMetadata(_))
    ));

    let empty_name = [b'R', b'L', b'M', b'K', 1, 0, 1, 0, 0, 0, 0, 0, 0, 0];
    assert!(matches!(
        LinkMetadata::from_bytes(&empty_name),
        Err(RulFormatError::InvalidLinkMetadata(_))
    ));

    let mut oversized_name = vec![b'R', b'L', b'M', b'K', 1, 0, 1, 0, 0, 0, 0, 0, 129, 0];
    oversized_name.extend(std::iter::repeat_n(b'a', 129));
    assert!(matches!(
        LinkMetadata::from_bytes(&oversized_name),
        Err(RulFormatError::InvalidLinkMetadata(_))
    ));

    let space_name = [
        b'R', b'L', b'M', b'K', 1, 0, 1, 0, 0, 0, 0, 0, 3, 0, b'a', b' ', b'b',
    ];
    assert!(matches!(
        LinkMetadata::from_bytes(&space_name),
        Err(RulFormatError::InvalidLinkMetadata(_))
    ));

    let non_ascii_name = [
        b'R', b'L', b'M', b'K', 1, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0xC3, 0xA9,
    ];
    assert!(matches!(
        LinkMetadata::from_bytes(&non_ascii_name),
        Err(RulFormatError::InvalidLinkMetadata(_))
    ));

    let trailing_bytes = [
        b'R', b'L', b'M', b'K', 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, b'x', 0xFF,
    ];
    assert!(matches!(
        LinkMetadata::from_bytes(&trailing_bytes),
        Err(RulFormatError::InvalidLinkMetadata(_))
    ));

    // 7. `.rls` rejects invalid symbol syntax and mixing raw `link_meta` with `export`/`import`.
    let long_name = "a".repeat(129);
    let bad_rls_long = format!(
        "cap 0x0\nexport func {long_name} 0\nnode $c = ConstI64 i64x1 const_i64=1\noutput $c\n"
    );
    assert!(assemble_rls(&bad_rls_long).is_err());

    let bad_rls_chars =
        "cap 0x0\nexport func bad$name 0\nnode $c = ConstI64 i64x1 const_i64=1\noutput $c\n";
    assert!(assemble_rls(bad_rls_chars).is_err());

    let bad_rls_mix = "cap 0x0\nlink_meta hex \"0102\"\nexport func foo 0\nnode $c = ConstI64 i64x1 const_i64=1\noutput $c\n";
    assert!(assemble_rls(bad_rls_mix).is_err());

    Ok(())
}

#[test]
fn test_v06_verify_module_and_reject_unlinked_exec() -> Result<(), String> {
    // 1. Valid partial library module importing a function (func 0) and a region (slot 1),
    //    having a local private region (slot 0), calling imported func 0 directly AND via CallIndirect
    //    while holding a live lease on private region slot 0 (frame rule!).
    let valid_mod_rls = "\
cap 0x0
import func ext.bump 0
import region shared.buf 1
export func mid.compute 2
proof 1 rule=region_decl target=0 lo=1 hi=4 witness=i64:plain
proof 2 rule=region_decl target=1 lo=1 hi=4 witness=i64:plain
proof 3 rule=func_sig target=0 lo=1 hi=i64 witness=i64
proof 4 rule=func_sig target=1 lo=1 hi=i64 witness=i64
proof 5 rule=func_sig target=2 lo=1 hi=i64 witness=i64
proof 6 rule=func_region_effect func=0 slot=1 pre=owned post=owned
proof 7 rule=func_region_effect func=1 slot=1 pre=owned post=owned
proof 8 rule=func_region_effect func=2 slot=0 pre=owned post=owned
proof 9 rule=func_region_effect func=2 slot=1 pre=owned post=owned
call_table 0 funcs=0,1
node $f1 = Func i64x1 a=1 b=0
node $x1 = LocalGet i64x1 a=0 b=0
node $c1 = ConstI64 i64x1 const_i64=10
node $r1 = Add i64x1 a=$x1 b=$c1
node $e1 = End i64x1 a=$r1 b=1
node $f2 = Func i64x1 a=2 b=0
node $arg = LocalGet i64x1 a=0 b=0
node $z = ConstI64 i64x1 const_i64=0
node $l0 = RegionLease i64x1 a=0 b=0
node $direct = Call i64x1 a=0 b=$arg c=none
node $ind = CallIndirect i64x1 table=0 idx=$z b=$direct c=none
node $priv = RegionLoad i64x1 a=0 b=$z
node $el0 = RegionEndLease i64x1 a=0 b=0
node $sum = Add i64x1 a=$ind b=$priv
node $e2 = End i64x1 a=$sum b=1
output $f2
";
    let mod_prog = assemble_rls(valid_mod_rls).map_err(|e| format!("assemble mod: {e}"))?;

    // `verify_module` succeeds and returns `static_cost == None` because unresolved imports exist.
    let mod_summary = verify_module(&mod_prog).map_err(|e| format!("verify_module failed: {e}"))?;
    assert_eq!(mod_summary.static_cost, None);

    // `verify_program` fails closed with `VerifyErrorKind::UnresolvedImport`.
    let exec_err = verify_program(&mod_prog);
    assert!(
        matches!(
            exec_err,
            Err(ref d) if matches!(d.kind, VerifyErrorKind::UnresolvedImport { .. })
        ),
        "expected UnresolvedImport from verify_program, got {exec_err:?}"
    );

    // Tier 0 interpreter and ELF64 AOT compiler MUST also fail closed on unlinked modules.
    assert!(execute_tier0_interpreter(&mod_prog, &[]).is_err());
    assert!(compile_to_elf64_bytes(&mod_prog).is_err());

    // 2. Frame rule violation on imported callee: leasing slot 1 (in callee's dom) before calling imported func 0 MUST fail.
    let lease_violation_rls = "\
cap 0x0
import func ext.bump 0
import region shared.buf 1
proof 1 rule=region_decl target=1 lo=1 hi=4 witness=i64:plain
proof 2 rule=func_sig target=0 lo=1 hi=i64 witness=i64
proof 3 rule=func_sig target=1 lo=1 hi=i64 witness=i64
proof 4 rule=func_region_effect func=0 slot=1 pre=owned post=owned
proof 5 rule=func_region_effect func=1 slot=1 pre=owned post=owned
node $f1 = Func i64x1 a=1 b=0
node $arg = LocalGet i64x1 a=0 b=0
node $l1 = RegionLease i64x1 a=1 b=0
node $bad = Call i64x1 a=0 b=$arg c=none
node $el1 = RegionEndLease i64x1 a=1 b=0
node $e1 = End i64x1 a=$bad b=1
output $f1
";
    let lease_violation = assemble_rls(lease_violation_rls).map_err(|e| e.to_string())?;
    assert!(matches!(
        verify_module(&lease_violation),
        Err(d) if matches!(d.kind, VerifyErrorKind::RegionTypestateMismatch { .. })
    ));

    // 3. Imported function defining a local Func body MUST be rejected.
    let imported_with_body_rls = "\
cap 0x0
import func ext.foo 0
proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64
node $f0 = Func i64x1 a=0 b=0
node $c = ConstI64 i64x1 const_i64=1
node $e = End i64x1 a=$c b=1
output $f0
";
    let imported_with_body = assemble_rls(imported_with_body_rls).map_err(|e| e.to_string())?;
    assert!(matches!(
        verify_module(&imported_with_body),
        Err(d) if matches!(d.kind, VerifyErrorKind::InvalidFuncSig { .. })
    ));

    // 4. Missing Func body for a non-imported FuncSig MUST be rejected.
    let missing_body_rls = "\
cap 0x0
import func ext.foo 0
proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64
proof 2 rule=func_sig target=1 lo=0 hi=0 witness=i64
proof 3 rule=func_sig target=2 lo=0 hi=0 witness=i64
node $f1 = Func i64x1 a=1 b=0
node $c = ConstI64 i64x1 const_i64=1
node $e = End i64x1 a=$c b=1
output $f1
";
    let missing_body = assemble_rls(missing_body_rls).map_err(|e| e.to_string())?;
    assert!(matches!(
        verify_module(&missing_body),
        Err(d) if matches!(d.kind, VerifyErrorKind::InvalidFuncSig { .. })
    ));

    // 5. Duplicate export name, duplicate import name, non-existent func/region exports,
    //    and crafted invalid link_meta bytes MUST be rejected.
    let dup_export_rls = "\
cap 0x0
export func dup.name 0
export func dup.name 1
proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64
proof 2 rule=func_sig target=1 lo=0 hi=0 witness=i64
node $f0 = Func i64x1 a=0 b=0
node $c0 = ConstI64 i64x1 const_i64=1
node $e0 = End i64x1 a=$c0 b=1
node $f1 = Func i64x1 a=1 b=0
node $c1 = ConstI64 i64x1 const_i64=2
node $e1 = End i64x1 a=$c1 b=1
output $f1
";
    let dup_export = assemble_rls(dup_export_rls).map_err(|e| e.to_string())?;
    assert!(matches!(
        verify_module(&dup_export),
        Err(d) if matches!(d.kind, VerifyErrorKind::InvalidLinkMetadata { .. })
    ));

    let dup_import_rls = "\
cap 0x0
import func dup.imp 0
import func dup.imp 1
proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64
proof 2 rule=func_sig target=1 lo=0 hi=0 witness=i64
proof 3 rule=func_sig target=2 lo=0 hi=0 witness=i64
node $f2 = Func i64x1 a=2 b=0
node $c = ConstI64 i64x1 const_i64=3
node $e = End i64x1 a=$c b=1
output $f2
";
    let dup_import = assemble_rls(dup_import_rls).map_err(|e| e.to_string())?;
    assert!(matches!(
        verify_module(&dup_import),
        Err(d) if matches!(d.kind, VerifyErrorKind::InvalidLinkMetadata { .. })
    ));

    let bad_export_func_rls = "\
cap 0x0
export func ghost.fn 5
proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64
node $f0 = Func i64x1 a=0 b=0
node $c = ConstI64 i64x1 const_i64=1
node $e = End i64x1 a=$c b=1
output $f0
";
    let bad_export_func = assemble_rls(bad_export_func_rls).map_err(|e| e.to_string())?;
    assert!(matches!(
        verify_module(&bad_export_func),
        Err(d) if matches!(d.kind, VerifyErrorKind::InvalidLinkMetadata { .. })
    ));

    let bad_export_reg_rls = "\
cap 0x0
export region ghost.reg 7
proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64
node $f0 = Func i64x1 a=0 b=0
node $c = ConstI64 i64x1 const_i64=1
node $e = End i64x1 a=$c b=1
output $f0
";
    let bad_export_reg = assemble_rls(bad_export_reg_rls).map_err(|e| e.to_string())?;
    assert!(matches!(
        verify_module(&bad_export_reg),
        Err(d) if matches!(d.kind, VerifyErrorKind::InvalidLinkMetadata { .. })
    ));

    // Crafted invalid binary link_meta (e.g., whitespace name) rejected by verify_module.
    let mut bad_meta_prog = assemble_rls(
        "cap 0x0\nproof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64\nnode $f0 = Func i64x1 a=0 b=0\nnode $c = ConstI64 i64x1 const_i64=1\nnode $e = End i64x1 a=$c b=1\noutput $f0\n",
    )
    .map_err(|e| e.to_string())?;
    bad_meta_prog.link_meta = vec![
        b'R', b'L', b'M', b'K', 1, 0, 1, 0, 0, 0, 0, 0, 3, 0, b'a', b' ', b'b',
    ];
    assert!(matches!(
        verify_module(&bad_meta_prog),
        Err(d) if matches!(d.kind, VerifyErrorKind::InvalidLinkMetadata { .. })
    ));

    Ok(())
}

#[test]
fn test_v06_linker_multi_module_relocation_and_errors() -> Result<(), String> {
    // =========================================================================
    // 1. 3-module chain exercising:
    //    - Private regions in mod0, mod1, and mod2 + shared exported/imported region `shared.acc`
    //    - Legacy effect-less 0-region function (`leaf.pure_legacy`) materialized with 0xFFFF sentinel
    //    - Live-lease frame-rule calls across linked module boundaries
    //    - Relocated ConstDataSeg + ConstDataLen + ConstDataLoad + ConstDataInBounds across all modules
    //    - Relocated BrTable + BrTableSpec across multiple modules
    //    - Relocated CallIndirect + CallTable referencing both local and imported functions
    //    - Tier 0 == ELF64 AOT byte-identical output + StaticCostSummary + deterministic output equality
    // =========================================================================

    // Module 0: Pure legacy library (0 regions, no FuncRegionEffect -> materialized to 0xFFFF!)
    // + own ConstDataSeg 0 ("ABC" len=3, byte[1]='B'=66) + own BrTable 0.
    // Exports: `leaf.pure_legacy` (func 0): takes x: i64, uses BrTable on x:
    //   case 0 -> x + len("ABC") (3)
    //   case 1 -> x + byte("ABC"[1]) (66)
    //   default -> x + 100
    let mod0_pure_rls = "\
cap 0x0
export func leaf.pure_legacy 0
data seg 0 \"ABC\"
proof 2 rule=const_data_in_bounds target=0 lo=0 hi=3 witness=0
br_table 0 default=2 targets=0,1
proof 4 rule=func_sig target=0 lo=1 hi=i64 witness=i64
node $f0 = Func i64x1 a=0 b=0
node $x = LocalGet i64x1 a=0 b=0
node $b2 = Block i64x1 a=0 b=0
node $b1 = Block i64x1 a=0 b=0
node $b0 = Block i64x1 a=0 b=0
node $bt = BrTable i64x1 a=$x b=0
node $e_b0 = End i64x1 a=0 b=0
node $len = ConstDataLen i64x1 a=0 b=0
node $r0 = Add i64x1 a=$x b=$len
node $ret0 = Return i64x1 a=$r0 b=0
node $e_b1 = End i64x1 a=0 b=0
node $c1 = ConstI64 i64x1 const_i64=1
node $ch = ConstDataLoad u8x1 proof=2 a=0 b=$c1
node $ch_i = Cast i64x1 a=$ch b=0
node $r1 = Add i64x1 a=$x b=$ch_i
node $ret1 = Return i64x1 a=$r1 b=0
node $e_b2 = End i64x1 a=0 b=0
node $c100 = ConstI64 i64x1 const_i64=100
node $r2 = Add i64x1 a=$x b=$c100
node $e0 = End i64x1 a=$r2 b=1
output $f0
";

    // Module 1: Shared-state library
    // - Declares private region slot 0 (priv_mid) AND exports shared region slot 1 (`shared.acc`, min=2 max=4 i64:plain)
    // - Imports `leaf.pure_legacy` as func 0 (with empty footprint slot=none!)
    // - Defines local func 1 (`local_double`, empty footprint)
    // - Defines CallTable 0 = [0 (imported leaf.pure_legacy), 1 (local_double)]
    // - Exports `mid.accumulate` as func 2 (footprint: slot 1 `shared.acc` owned->owned):
    //     reads ConstDataSeg 0 ("WXYZ" len=4, byte[2]='Y'=89), calls CallIndirect(table 0, idx=0, x=1) -> 1+66=67,
    //     calls CallIndirect(table 0, idx=1, x=10) -> 20, adds len(4) + 'Y'(89) -> 180,
    //     stores 180 into shared.acc[0] using RegionIndexInBounds proof, returns shared.acc[0].
    let mod1_mid_rls = "\
cap 0x0
import func leaf.pure_legacy 0
export region shared.acc 1
export func mid.accumulate 2
data seg 0 \"WXYZ\"
proof 2 rule=const_data_in_bounds target=0 lo=0 hi=4 witness=0
proof 3 rule=region_decl target=0 lo=1 hi=4 witness=i64:plain
proof 4 rule=region_decl target=1 lo=2 hi=4 witness=i64:plain
proof 5 rule=region_index_in_bounds target=1 lo=0 hi=2 witness=0
proof 6 rule=func_sig target=0 lo=1 hi=i64 witness=i64
proof 7 rule=func_sig target=1 lo=1 hi=i64 witness=i64
proof 8 rule=func_sig target=2 lo=1 hi=i64 witness=i64
proof 9 rule=func_region_effect func=0 slot=none pre=owned post=owned
proof 10 rule=func_region_effect func=1 slot=none pre=owned post=owned
proof 11 rule=func_region_effect func=2 slot=1 pre=owned post=owned
call_table 0 funcs=0,1
node $f1 = Func i64x1 a=1 b=0
node $a1 = LocalGet i64x1 a=0 b=0
node $dbl = Add i64x1 a=$a1 b=$a1
node $e1 = End i64x1 a=$dbl b=1
node $f2 = Func i64x1 a=2 b=0
node $sel = LocalGet i64x1 a=0 b=0
node $c0 = ConstI64 i64x1 const_i64=0
node $c1 = ConstI64 i64x1 const_i64=1
node $c2 = ConstI64 i64x1 const_i64=2
node $c10 = ConstI64 i64x1 const_i64=10
node $v0 = CallIndirect i64x1 table=0 idx=$c0 b=$sel c=none
node $v1 = CallIndirect i64x1 table=0 idx=$c1 b=$c10 c=none
node $slen = ConstDataLen i64x1 a=0 b=0
node $ch = ConstDataLoad u8x1 proof=2 a=0 b=$c2
node $ch_i = Cast i64x1 a=$ch b=0
node $s1 = Add i64x1 a=$v0 b=$v1
node $s2 = Add i64x1 a=$s1 b=$slen
node $total = Add i64x1 a=$s2 b=$ch_i
node $st = RegionStore i64x1 proof=5 a=1 b=$c0 c=$total
node $ld = RegionLoad i64x1 proof=5 a=1 b=$c0
node $e2 = End i64x1 a=$ld b=1
output $f2
";

    // Module 2: Entry application
    // - Declares private region slot 0 (`priv_main`, min=2 max=4 i64:plain)
    // - Imports `shared.acc` at slot 1
    // - Imports `mid.accumulate` as func 0 (dom={slot 1}) and `leaf.pure_legacy` as func 1 (dom=empty)
    // - Has its own ConstDataSeg 0 ("PQ" len=2) and its own BrTable 0!
    // - Exports `app.main` as func 2:
    //     stores 500 into priv_main[0], LEASES priv_main (slot 0),
    //     calls mid.accumulate(1) -> 180 (frame rule preserves live lease on priv_main!),
    //     calls leaf.pure_legacy(0) -> 0 + 3 = 3 (legacy empty footprint preserves live lease!),
    //     reads priv_main[0] (500) under live lease, ends lease on priv_main,
    //     uses own BrTable 0 on 0 -> adds ConstDataLen("PQ") (2) + shared.acc[0] (180),
    //     returns 500 + 180 + 3 + 2 + 180 = 865!
    let mod2_entry_rls = "\
cap 0x0
import region shared.acc 1
import func mid.accumulate 0
import func leaf.pure_legacy 1
export func app.main 2
data seg 0 \"PQ\"
br_table 0 default=1 targets=0
proof 3 rule=region_decl target=0 lo=2 hi=4 witness=i64:plain
proof 4 rule=region_decl target=1 lo=2 hi=4 witness=i64:plain
proof 5 rule=region_index_in_bounds target=0 lo=0 hi=2 witness=0
proof 6 rule=region_index_in_bounds target=1 lo=0 hi=2 witness=0
proof 7 rule=func_sig target=0 lo=1 hi=i64 witness=i64
proof 8 rule=func_sig target=1 lo=1 hi=i64 witness=i64
proof 9 rule=func_sig target=2 lo=0 hi=0 witness=i64
proof 10 rule=func_region_effect func=0 slot=1 pre=owned post=owned
proof 11 rule=func_region_effect func=1 slot=none pre=owned post=owned
proof 12 rule=func_region_effect func=2 slot=0 pre=owned post=owned
proof 13 rule=func_region_effect func=2 slot=1 pre=owned post=owned
node $f2 = Func i64x1 a=2 b=0
node $c0 = ConstI64 i64x1 const_i64=0
node $c1 = ConstI64 i64x1 const_i64=1
node $c500 = ConstI64 i64x1 const_i64=500
node $st0 = RegionStore i64x1 proof=5 a=0 b=$c0 c=$c500
node $lease = RegionLease i64x1 a=0 b=0
node $m = Call i64x1 a=0 b=$c1 c=none
node $p = Call i64x1 a=1 b=$c0 c=none
node $priv_v = RegionLoad i64x1 proof=5 a=0 b=$c0
node $end_lease = RegionEndLease i64x1 a=0 b=0
node $shared_v = RegionLoad i64x1 proof=6 a=1 b=$c0
node $b1 = Block i64x1 a=0 b=0
node $b0 = Block i64x1 a=0 b=0
node $bt = BrTable i64x1 a=$c0 b=0
node $eb0 = End i64x1 a=0 b=0
node $pq_len = ConstDataLen i64x1 a=0 b=0
node $acc1 = Add i64x1 a=$priv_v b=$m
node $acc2 = Add i64x1 a=$acc1 b=$p
node $acc3 = Add i64x1 a=$acc2 b=$pq_len
node $acc4 = Add i64x1 a=$acc3 b=$shared_v
node $ret = Return i64x1 a=$acc4 b=0
node $eb1 = End i64x1 a=0 b=0
node $e2 = End i64x1 a=$c0 b=1
output $f2
";

    let m0 = assemble_rls(mod0_pure_rls).map_err(|e| format!("m0: {e}"))?;
    let m1 = assemble_rls(mod1_mid_rls).map_err(|e| format!("m1: {e}"))?;
    let m2 = assemble_rls(mod2_entry_rls).map_err(|e| format!("m2: {e}"))?;

    // Link with default entry (last module's last local func).
    let rep1 = link_modules(
        &[m0.clone(), m1.clone(), m2.clone()],
        &LinkOptions::default(),
    )
    .map_err(|e| format!("link 3-module failed: {e}"))?;
    assert_eq!(rep1.module_count, 3);
    assert_eq!(rep1.func_count, 4); // leaf.pure_legacy, local_double, mid.accumulate, app.main
    assert_eq!(rep1.region_count, 3); // priv_mid, shared.acc, priv_main
    assert_eq!(rep1.capability_mask, CAP_NONE);
    assert!(rep1.program.link_meta.is_empty());
    assert!(rep1.summary.static_cost.is_some());
    assert!(rep1.summary.zero_runtime_bounds_checks);

    // Deterministic byte-identical output on repeated links AND when passing explicit `--entry app.main` in different module order.
    let rep2 = link_modules(
        &[m0.clone(), m1.clone(), m2.clone()],
        &LinkOptions::default(),
    )
    .map_err(|e| format!("re-link failed: {e}"))?;
    assert_eq!(rep1.program.to_rl_bytes(), rep2.program.to_rl_bytes());

    let rep_explicit_entry = link_modules(
        &[m2.clone(), m0.clone(), m1.clone()],
        &LinkOptions {
            entry_export: Some("app.main".to_owned()),
            capability_mask: None,
        },
    )
    .map_err(|e| format!("explicit entry link failed: {e}"))?;

    // Execute both linked binaries on Tier 0 AND ELF64 AOT and verify exact expected result = 865!
    let val1 = run_tier0_eq_aot_i64(&rep1.program, "3mod_default")?;
    assert_eq!(val1, 865);
    let val_exp = run_tier0_eq_aot_i64(&rep_explicit_entry.program, "3mod_explicit")?;
    assert_eq!(val_exp, 865);

    // =========================================================================
    // 2. LinkError::CapabilityEscalation (ZERO silent capability escalation!)
    // =========================================================================
    let cap_diverge_lib = assemble_rls(
        "\
cap 0x8
export func lib.diverge_fn 0
proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64
node $f0 = Func i64x1 a=0 b=0
node $c = ConstI64 i64x1 const_i64=7
node $e = End i64x1 a=$c b=1
output $f0
",
    )
    .map_err(|e| e.to_string())?;
    let cap_none_entry = assemble_rls(
        "\
cap 0x0
import func lib.diverge_fn 0
proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64
proof 2 rule=func_sig target=1 lo=0 hi=0 witness=i64
node $f1 = Func i64x1 a=1 b=0
node $r = Call i64x1 a=0 b=none c=none
node $e = End i64x1 a=$r b=1
output $f1
",
    )
    .map_err(|e| e.to_string())?;

    let cap_err = link_modules(
        &[cap_diverge_lib.clone(), cap_none_entry.clone()],
        &LinkOptions::default(),
    );
    assert_eq!(
        cap_err,
        Err(LinkError::CapabilityEscalation {
            module_index: 0,
            module_cap: CAP_DIVERGE,
            allowed_cap: CAP_NONE,
        })
    );

    // Also test non-CAP_DIVERGE capability bit escalation (e.g. CAP_NET_RING).
    let mut cap_net_lib = cap_diverge_lib.clone();
    cap_net_lib.capability_mask = CAP_NET_RING;
    assert_eq!(
        link_modules(
            &[cap_net_lib, cap_none_entry.clone()],
            &LinkOptions::default()
        ),
        Err(LinkError::CapabilityEscalation {
            module_index: 0,
            module_cap: CAP_NET_RING,
            allowed_cap: CAP_NONE,
        })
    );

    // Succeeds when `--cap 0x8` is explicitly granted!
    let cap_ok = link_modules(
        &[cap_diverge_lib, cap_none_entry],
        &LinkOptions {
            entry_export: None,
            capability_mask: Some(CAP_DIVERGE),
        },
    )
    .map_err(|e| format!("explicit --cap 0x8 should succeed: {e}"))?;
    assert_eq!(cap_ok.capability_mask, CAP_DIVERGE);
    assert_eq!(cap_ok.summary.static_cost, None);
    assert_eq!(run_tier0_eq_aot_i64(&cap_ok.program, "cap_explicit")?, 7);

    // =========================================================================
    // 3. LinkError::ProofIndexOverflow (> 255 unique node-referenced proofs) + dedup success
    // =========================================================================
    let build_loop_bound_mod = |export_name: &str, start_bound: u32, count: u32, is_entry: bool| {
        let mut s = String::from("cap 0x0\n");
        if is_entry {
            s.push_str("import func lib.loops 0\n");
            s.push_str("proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64\n");
            s.push_str("proof 2 rule=func_sig target=1 lo=0 hi=0 witness=i64\n");
            for i in 0..count {
                let p_idx = 3 + i;
                let bound = start_bound + i;
                s.push_str(&format!(
                    "proof {p_idx} rule=loop_bound target=0 lo=0 hi={bound} witness=0\n"
                ));
            }
            s.push_str("node $f1 = Func i64x1 a=1 b=0\n");
            for i in 0..count {
                let p_idx = 3 + i;
                s.push_str(&format!("node $lp{i} = Loop i64x1 proof={p_idx} a=0 b=0\n"));
                s.push_str(&format!("node $le{i} = End i64x1 a=0 b=0\n"));
            }
            s.push_str("node $c = Call i64x1 a=0 b=none c=none\n");
            s.push_str("node $e1 = End i64x1 a=$c b=1\n");
            s.push_str("output $f1\n");
        } else {
            s.push_str(&format!("export func {export_name} 0\n"));
            s.push_str("proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64\n");
            for i in 0..count {
                let p_idx = 2 + i;
                let bound = start_bound + i;
                s.push_str(&format!(
                    "proof {p_idx} rule=loop_bound target=0 lo=0 hi={bound} witness=0\n"
                ));
            }
            s.push_str("node $f0 = Func i64x1 a=0 b=0\n");
            for i in 0..count {
                let p_idx = 2 + i;
                s.push_str(&format!("node $lp{i} = Loop i64x1 proof={p_idx} a=0 b=0\n"));
                s.push_str(&format!("node $le{i} = End i64x1 a=0 b=0\n"));
            }
            s.push_str("node $v = ConstI64 i64x1 const_i64=99\n");
            s.push_str("node $e0 = End i64x1 a=$v b=1\n");
            s.push_str("output $f0\n");
        }
        assemble_rls(&s)
    };

    // 130 distinct bounds in modA (1..=130) + 130 distinct bounds in modB (1001..=1130) => 260 > 255 -> ProofIndexOverflow!
    let p_mod_a = build_loop_bound_mod("lib.loops", 1, 130, false).map_err(|e| e.to_string())?;
    let p_mod_b = build_loop_bound_mod("", 1001, 130, true).map_err(|e| e.to_string())?;
    assert_eq!(
        link_modules(&[p_mod_a.clone(), p_mod_b], &LinkOptions::default()),
        Err(LinkError::ProofIndexOverflow {
            needed: 260,
            max: 255,
        })
    );

    // 130 IDENTICAL bounds in modA (1..=130) + 130 IDENTICAL bounds in modB (1..=130) => deduplicates to 130 <= 255 -> OK!
    let p_mod_b_dedup = build_loop_bound_mod("", 1, 130, true).map_err(|e| e.to_string())?;
    let dedup_rep = link_modules(&[p_mod_a, p_mod_b_dedup], &LinkOptions::default())
        .map_err(|e| format!("dedup proof link failed: {e}"))?;
    assert_eq!(run_tier0_eq_aot_i64(&dedup_rep.program, "proof_dedup")?, 99);

    // =========================================================================
    // 4. LinkError::CycleWithoutDiverge (cross-module cycle rejected without CAP_DIVERGE, allowed with CAP_DIVERGE)
    // =========================================================================
    let cyc_a_rls = "\
cap 0x0
import func cyc.b 0
export func cyc.a 1
proof 1 rule=func_sig target=0 lo=1 hi=i64 witness=i64
proof 2 rule=func_sig target=1 lo=1 hi=i64 witness=i64
node $f1 = Func i64x1 a=1 b=0
node $n = LocalGet i64x1 a=0 b=0
node $z = ConstI64 i64x1 const_i64=0
node $le = CmpEq maskx1 a=$n b=$z
node $if = If i64x1 a=$le b=0
node $ret0 = Return i64x1 a=$z b=0
node $eif = End i64x1 a=0 b=0
node $one = ConstI64 i64x1 const_i64=1
node $n1 = Sub i64x1 a=$n b=$one
node $sub = Call i64x1 a=0 b=$n1 c=none
node $ans = Add i64x1 a=$sub b=$one
node $e1 = End i64x1 a=$ans b=1
output $f1
";
    let cyc_b_rls = "\
cap 0x0
import func cyc.a 0
export func cyc.b 1
proof 1 rule=func_sig target=0 lo=1 hi=i64 witness=i64
proof 2 rule=func_sig target=1 lo=1 hi=i64 witness=i64
proof 3 rule=func_sig target=2 lo=0 hi=0 witness=i64
node $f1 = Func i64x1 a=1 b=0
node $n = LocalGet i64x1 a=0 b=0
node $z = ConstI64 i64x1 const_i64=0
node $le = CmpEq maskx1 a=$n b=$z
node $if = If i64x1 a=$le b=0
node $ret0 = Return i64x1 a=$z b=0
node $eif = End i64x1 a=0 b=0
node $one = ConstI64 i64x1 const_i64=1
node $n1 = Sub i64x1 a=$n b=$one
node $sub = Call i64x1 a=0 b=$n1 c=none
node $ans = Add i64x1 a=$sub b=$one
node $e1 = End i64x1 a=$ans b=1
node $f2 = Func i64x1 a=2 b=0
node $c6 = ConstI64 i64x1 const_i64=6
node $r = Call i64x1 a=1 b=$c6 c=none
node $e2 = End i64x1 a=$r b=1
output $f2
";
    let cyc_a = assemble_rls(cyc_a_rls).map_err(|e| e.to_string())?;
    let cyc_b = assemble_rls(cyc_b_rls).map_err(|e| e.to_string())?;
    assert_eq!(
        link_modules(&[cyc_a.clone(), cyc_b.clone()], &LinkOptions::default()),
        Err(LinkError::CycleWithoutDiverge)
    );

    // Same cross-module mutual recursion succeeds when CAP_DIVERGE is granted explicitly!
    let cyc_ok = link_modules(
        &[cyc_a, cyc_b],
        &LinkOptions {
            entry_export: None,
            capability_mask: Some(CAP_DIVERGE),
        },
    )
    .map_err(|e| format!("cycle with CAP_DIVERGE failed: {e}"))?;
    assert_eq!(run_tier0_eq_aot_i64(&cyc_ok.program, "mutual_rec")?, 6);

    // =========================================================================
    // 5. Remaining typed LinkError diagnostics
    // =========================================================================

    // UnresolvedSymbol (func)
    assert!(matches!(
        link_modules(std::slice::from_ref(&m2), &LinkOptions::default()),
        Err(LinkError::UnresolvedSymbol { .. })
    ));

    // DuplicateExport
    assert!(matches!(
        link_modules(&[m0.clone(), m0.clone()], &LinkOptions::default()),
        Err(LinkError::DuplicateExport {
            kind: LinkSymbolKind::Func,
            ..
        })
    ));

    // FuncSigMismatch
    let sig_mismatch_entry = assemble_rls(
        "\
cap 0x0
import func leaf.pure_legacy 0
proof 1 rule=func_sig target=0 lo=2 hi=i64,i64 witness=i64
proof 2 rule=func_sig target=1 lo=0 hi=0 witness=i64
node $f1 = Func i64x1 a=1 b=0
node $c = ConstI64 i64x1 const_i64=1
node $r = Call i64x1 a=0 b=$c c=$c
node $e = End i64x1 a=$r b=1
output $f1
",
    )
    .map_err(|e| e.to_string())?;
    assert!(matches!(
        link_modules(&[m0.clone(), sig_mismatch_entry], &LinkOptions::default()),
        Err(LinkError::FuncSigMismatch { .. })
    ));

    // FuncRegionEffectMismatch
    let eff_mismatch_entry = assemble_rls(
        "\
cap 0x0
import region shared.acc 0
import func mid.accumulate 0
proof 1 rule=region_decl target=0 lo=2 hi=4 witness=i64:plain
proof 2 rule=func_sig target=0 lo=1 hi=i64 witness=i64
proof 3 rule=func_sig target=1 lo=0 hi=0 witness=i64
proof 4 rule=func_region_effect func=0 slot=none pre=owned post=owned
proof 5 rule=func_region_effect func=1 slot=0 pre=owned post=owned
node $f1 = Func i64x1 a=1 b=0
node $c = ConstI64 i64x1 const_i64=1
node $r = Call i64x1 a=0 b=$c c=none
node $e = End i64x1 a=$r b=1
output $f1
",
    )
    .map_err(|e| e.to_string())?;
    assert!(matches!(
        link_modules(
            &[m0.clone(), m1.clone(), eff_mismatch_entry],
            &LinkOptions::default()
        ),
        Err(LinkError::FuncRegionEffectMismatch { .. })
    ));

    // RegionDeclMismatch (different max_len)
    let reg_mismatch_entry = assemble_rls(
        "\
cap 0x0
import region shared.acc 0
proof 1 rule=region_decl target=0 lo=2 hi=8 witness=i64:plain
proof 2 rule=func_sig target=0 lo=0 hi=0 witness=i64
node $f0 = Func i64x1 a=0 b=0
node $c = ConstI64 i64x1 const_i64=1
node $e = End i64x1 a=$c b=1
output $f0
",
    )
    .map_err(|e| e.to_string())?;
    assert!(matches!(
        link_modules(&[m0, m1, reg_mismatch_entry], &LinkOptions::default()),
        Err(LinkError::RegionDeclMismatch { .. })
    ));

    // TooManyRegions (> 64 unique regions across modules)
    let build_many_regions_mod = |count: u16| {
        let mut s = String::from("cap 0x0\n");
        let mut pid = 1u32;
        for slot in 0..count {
            s.push_str(&format!(
                "proof {pid} rule=region_decl target={slot} lo=0 hi=4 witness=i64:plain\n"
            ));
            pid += 1;
        }
        s.push_str(&format!(
            "proof {pid} rule=func_sig target=0 lo=0 hi=0 witness=i64\n"
        ));
        s.push_str("node $f0 = Func i64x1 a=0 b=0\n");
        s.push_str("node $c = ConstI64 i64x1 const_i64=0\n");
        s.push_str("node $e = End i64x1 a=$c b=1\n");
        s.push_str("output $f0\n");
        assemble_rls(&s)
    };
    let r33 = build_many_regions_mod(33).map_err(|e| e.to_string())?;
    let r32 = build_many_regions_mod(32).map_err(|e| e.to_string())?;
    assert_eq!(
        link_modules(&[r33, r32], &LinkOptions::default()),
        Err(LinkError::TooManyRegions {
            needed: 65,
            max: 64
        })
    );

    // UnsupportedColumnarModule
    let columnar_mod = RulProgram {
        capability_mask: CAP_NONE,
        arenas: vec![ArenaDescriptor {
            arena_id: 0,
            element_count: 8,
            type_lane: TypeLane::F64X8,
            mutable: false,
            alignment: 64,
            initial_version: 1,
        }],
        proofs: Vec::new(),
        nodes: vec![
            NodeFrame {
                opcode: Opcode::ArenaInput,
                type_lane: TypeLane::F64X8,
                schedule: ScheduleHint::UNFUSED,
                proof_idx: 0,
                token_op: TokenOp::read_lease(0, 1),
                in_a: 0,
                in_b: 0,
            },
            NodeFrame {
                opcode: Opcode::ReduceSum,
                type_lane: TypeLane::F64_SCALAR,
                schedule: ScheduleHint::UNFUSED,
                proof_idx: 0,
                token_op: TokenOp::PURE,
                in_a: 0,
                in_b: 0,
            },
        ],
        output_node: 1,
        const_data: Vec::new(),
        link_meta: Vec::new(),
    };
    assert_eq!(
        link_modules(&[columnar_mod], &LinkOptions::default()),
        Err(LinkError::UnsupportedColumnarModule { module_index: 0 })
    );

    Ok(())
}
