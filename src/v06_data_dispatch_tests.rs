//! Milestone 2b (`v0.6`): Tests for Section 5a/5b `.rl` binary format, domain-separated hashing,
//! CAS collision rejection, segment-addressed `ConstDataLen`/`ConstDataLoad`, dense `BrTable`,
//! and typed `CallIndirect` across verifier, `Tier 0`, and standalone Linux `x86_64` `ELF64` AOT.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::aot::{
    compile_rl_to_executable, compile_to_elf64_bytes, run_compiled_structured_executable, AotError,
    AOT_EXIT_GATHER_OOB, AOT_EXIT_INDIRECT_CALL_OOB, AOT_EXIT_STACK_OVERFLOW,
};
use crate::bytecode::{
    ArenaDescriptor, ElemType, NodeFrame, Opcode, ProofEntry, RulFormatError, RulProgram,
    ScheduleHint, TokenOp, TypeLane, CALL_ARG_SENTINEL, CAP_DIVERGE, CAP_NONE,
    MAX_CONST_DATA_BYTES, MAX_LINK_META_BYTES, RL_HEADER_SIZE,
};
use crate::cas::{CasError, CasHash, RulCasStore};
use crate::engine::{
    execute_tier0_interpreter, execute_tier1_stencil, execute_tier2_fused, ExecError, RuntimeValue,
};
use crate::transpiler::{assemble_rls, disassemble_rl, program_to_rls};
use crate::verifier::{verify_program, verify_program_with_join_work, VerifyErrorKind};

fn test_exe_path(tag: &str) -> std::path::PathBuf {
    static NEXT_EXE_ID: AtomicU64 = AtomicU64::new(0);
    let seq = NEXT_EXE_ID.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    std::env::temp_dir().join(format!(
        "rul_v06_dd_{tag}_{}_{seq}_{nanos}.elf",
        std::process::id()
    ))
}

fn scalar_i64_from_report(val: &RuntimeValue) -> Result<i64, String> {
    match val {
        RuntimeValue::TypedScalar(ts) => {
            if let Some(v) = ts.as_i64() {
                Ok(v)
            } else if let Some(b) = ts.as_u8() {
                Ok(i64::from(b))
            } else {
                Err(format!("expected I64/U8 scalar, got {ts:?}"))
            }
        }
        other => Err(format!("expected I64/U8 scalar, got {other:?}")),
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

fn run_aot_expect_err(program: &RulProgram, tag: &str) -> Result<AotError, String> {
    let exe_path = test_exe_path(tag);
    compile_rl_to_executable(program, &exe_path).map_err(|e| format!("aot compile failed: {e}"))?;
    let res = run_compiled_structured_executable(&exe_path, &[]);
    let _ = std::fs::remove_file(&exe_path);
    match res {
        Ok(out) => Err(format!("expected AOT trap for {tag}, got stdout {out:?}")),
        Err(err) => Ok(err),
    }
}

fn run_aot_expect_trap_zero_stdout(
    program: &RulProgram,
    tag: &str,
    expected_exit: i32,
) -> Result<(), String> {
    let exe_path = test_exe_path(tag);
    compile_rl_to_executable(program, &exe_path).map_err(|e| format!("aot compile failed: {e}"))?;
    // Retry transient ETXTBSY (os error 26) caused by concurrent test threads forking child
    // processes that briefly inherit open descriptors to freshly written test binaries.
    let mut attempt = 0u32;
    let out_res = loop {
        match std::process::Command::new(&exe_path).output() {
            Ok(out) => break Ok(out),
            Err(e) if e.raw_os_error() == Some(26) && attempt < 64 => {
                attempt += 1;
                std::thread::sleep(Duration::from_micros(500));
            }
            Err(e) => break Err(format!("spawn {tag} failed: {e}")),
        }
    };
    let _ = std::fs::remove_file(&exe_path);
    let out = out_res?;
    if out.status.code() != Some(expected_exit) {
        return Err(format!(
            "expected exit {expected_exit} for {tag}, got {:?}",
            out.status.code()
        ));
    }
    if !out.stdout.is_empty() {
        return Err(format!(
            "expected zero stdout on trap for {tag}, got {} bytes",
            out.stdout.len()
        ));
    }
    Ok(())
}

fn legacy_v05_hash(program: &RulProgram) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let prime: u64 = 0x0100_0000_01b3;
    for arena in &program.arenas {
        for b in arena.to_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(prime);
        }
    }
    for proof in &program.proofs {
        for b in proof.to_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(prime);
        }
    }
    for node in &program.nodes {
        for b in node.to_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(prime);
        }
    }
    hash
}

#[test]
fn test_v06_const_data_format_hash_cas_and_rls_roundtrip() -> Result<(), String> {
    // 1. v0.5.0 byte compatibility when const_data and link_meta are both empty.
    let base_prog = RulProgram {
        capability_mask: CAP_NONE,
        arenas: Vec::new(),
        proofs: Vec::new(),
        nodes: vec![NodeFrame::const_i64(42, ScheduleHint::UNFUSED)],
        output_node: 0,
        const_data: Vec::new(),
        link_meta: Vec::new(),
    };
    assert_eq!(base_prog.compute_graph_hash(), legacy_v05_hash(&base_prog));
    let base_bytes = base_prog.to_rl_bytes();
    assert_eq!(base_bytes.len(), RL_HEADER_SIZE + 16);
    assert_eq!(base_bytes.get(40..64), Some([0u8; 24].as_slice()));
    let base_decoded =
        RulProgram::from_rl_bytes(&base_bytes).map_err(|e| format!("decode base: {e}"))?;
    assert_eq!(base_decoded, base_prog);

    // 2. Non-empty Section 5a (const_data) + Section 5b (link_meta) header encoding & round-trip.
    let mut sec5_prog = base_prog.clone();
    sec5_prog.const_data = b"hello # world!\n".to_vec();
    sec5_prog.link_meta = vec![0xDE, 0xAD, 0xBE, 0xEF];
    sec5_prog.proofs = vec![ProofEntry::const_data_seg(
        0,
        0,
        u32::try_from(sec5_prog.const_data.len()).unwrap_or(0),
    )];

    let sec5_bytes = sec5_prog.to_rl_bytes();
    assert_eq!(
        sec5_bytes.get(40..44),
        Some(
            u32::try_from(sec5_prog.const_data.len())
                .unwrap_or(0)
                .to_le_bytes()
                .as_slice()
        )
    );
    assert_eq!(
        sec5_bytes.get(44..48),
        Some(
            u32::try_from(sec5_prog.link_meta.len())
                .unwrap_or(0)
                .to_le_bytes()
                .as_slice()
        )
    );
    assert_eq!(sec5_bytes.get(48..64), Some([0u8; 16].as_slice()));
    let sec5_decoded =
        RulProgram::from_rl_bytes(&sec5_bytes).map_err(|e| format!("decode sec5: {e}"))?;
    assert_eq!(sec5_decoded, sec5_prog);

    // 3. Reject non-zero trailing reserved bytes in header[48..64].
    let mut bad_reserved = sec5_bytes.clone();
    if let Some(b) = bad_reserved.get_mut(48) {
        *b = 1;
    }
    assert_eq!(
        RulProgram::from_rl_bytes(&bad_reserved),
        Err(RulFormatError::InvalidReservedHeader)
    );

    // 4. Reject oversized const_data (> 8 MiB), oversized link_meta (> 2 MiB), and truncated image.
    let mut bad_const_size = sec5_bytes.clone();
    if let Some(dst) = bad_const_size.get_mut(40..44) {
        dst.copy_from_slice(&(MAX_CONST_DATA_BYTES + 1).to_le_bytes());
    }
    assert_eq!(
        RulProgram::from_rl_bytes(&bad_const_size),
        Err(RulFormatError::SectionTooLarge)
    );

    let mut bad_link_size = sec5_bytes.clone();
    if let Some(dst) = bad_link_size.get_mut(44..48) {
        dst.copy_from_slice(&(MAX_LINK_META_BYTES + 1).to_le_bytes());
    }
    assert_eq!(
        RulProgram::from_rl_bytes(&bad_link_size),
        Err(RulFormatError::SectionTooLarge)
    );

    let truncated = sec5_bytes
        .get(..sec5_bytes.len().saturating_sub(2))
        .ok_or_else(|| "slice error".to_owned())?;
    assert!(matches!(
        RulProgram::from_rl_bytes(truncated),
        Err(RulFormatError::LengthMismatch { .. })
    ));

    // 5. Domain separation: shifting boundary between Section 5a and Section 5b produces distinct hashes.
    let mut p_ab_c = base_prog.clone();
    p_ab_c.const_data = b"AB".to_vec();
    p_ab_c.link_meta = b"C".to_vec();
    let mut p_a_bc = base_prog.clone();
    p_a_bc.const_data = b"A".to_vec();
    p_a_bc.link_meta = b"BC".to_vec();
    let mut p_abc_empty = base_prog.clone();
    p_abc_empty.const_data = b"ABC".to_vec();
    let mut p_empty_abc = base_prog.clone();
    p_empty_abc.link_meta = b"ABC".to_vec();

    let hashes = [
        base_prog.compute_graph_hash(),
        p_ab_c.compute_graph_hash(),
        p_a_bc.compute_graph_hash(),
        p_abc_empty.compute_graph_hash(),
        p_empty_abc.compute_graph_hash(),
    ];
    for i in 0..hashes.len() {
        for j in (i + 1)..hashes.len() {
            assert_ne!(
                hashes.get(i),
                hashes.get(j),
                "domain separation collision between index {i} and {j}"
            );
        }
    }

    // 6. RulCasStore::insert_verified idempotency and hash-collision rejection.
    let mut store = RulCasStore::new();
    let prog_a = base_prog.clone();
    let h_a = store
        .insert_verified(prog_a.clone())
        .map_err(|e| format!("insert prog_a: {e}"))?;
    // Identical re-insert must be idempotent.
    let h_a2 = store
        .insert_verified(prog_a.clone())
        .map_err(|e| format!("re-insert prog_a: {e}"))?;
    assert_eq!(h_a, h_a2);
    assert_eq!(store.len(), 1);

    // Distinct program colliding at the same 64-bit hash must be rejected without overwrite.
    let prog_b = RulProgram {
        capability_mask: CAP_NONE,
        arenas: Vec::new(),
        proofs: Vec::new(),
        nodes: vec![NodeFrame::const_i64(99, ScheduleHint::UNFUSED)],
        output_node: 0,
        const_data: Vec::new(),
        link_meta: Vec::new(),
    };
    let h_b = CasHash::from_program(&prog_b);
    let summary_a = verify_program(&prog_a).map_err(|e| format!("verify a: {e}"))?;
    store.insert_with_forced_hash_for_test(h_b, prog_a.clone(), summary_a);
    let coll_res = store.insert_verified(prog_b);
    assert_eq!(coll_res, Err(CasError::HashCollision { hash: h_b }));
    assert_eq!(store.get(h_b).map(|e| &e.program), Some(&prog_a));

    // 7. Lossless .rls round-trip for named & indexed `data seg` string & hex declarations.
    let rls_named = r#"
        cap 0x0
        data seg $greeting "Hi #1\n"
        data seg $raw hex "0102ff0010203040"
        proof 3 rule=const_data_in_bounds target=$greeting lo=0 hi=6 witness=0
        node $len0 = ConstDataLen i64x1 fg=0 pol=l1 proof=0 tok=pure a=$greeting b=0
        node $idx0 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=3
        node $b3 = ConstDataLoad u8x1 fg=0 pol=l1 proof=3 tok=pure a=$greeting b=$idx0
        node $b3_i64 = Cast i64x1 fg=0 pol=l1 proof=0 tok=pure a=$b3 b=0
        node $zero = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
        node $q0 = ConstDataLoad i64x1 fg=0 pol=l1 proof=0 tok=pure a=$raw b=$zero
        node $sum = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$len0 b=$b3_i64
        node $out = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$sum b=$q0
        output $out
    "#;
    let prog_rls = assemble_rls(rls_named).map_err(|e| format!("assemble named rls: {e}"))?;
    verify_program(&prog_rls).map_err(|e| format!("verify named rls: {e}"))?;
    let decompiled = program_to_rls(&prog_rls);
    assert!(decompiled.contains("data seg 0 \"Hi #1\\n\""));
    assert!(decompiled.contains("data seg 1 hex \"0102ff0010203040\""));
    let reassembled =
        assemble_rls(&decompiled).map_err(|e| format!("reassemble rls: {e}\n{decompiled}"))?;
    assert_eq!(reassembled, prog_rls);
    assert_eq!(
        reassembled.compute_graph_hash(),
        prog_rls.compute_graph_hash()
    );
    let dis = disassemble_rl(&prog_rls);
    assert!(dis.contains("ConstData=14B"));

    // 8. Reject unescaped quote inside string literal in `data seg`.
    let bad_quote_rls = r#"
        cap 0x0
        data seg 0 "foo" "bar"
        node $c = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
        output $c
    "#;
    assert!(assemble_rls(bad_quote_rls).is_err());

    Ok(())
}

#[test]
fn test_v06_const_data_segments_tier0_vs_aot() -> Result<(), String> {
    // 1. Multi-segment program testing ConstDataLen, u8x1 and i64x1 loads (proven via ConstI64 and BitAnd, plus dynamic unchecked).
    let mut const_data = Vec::new();
    let seg0 = b"Rul-v0.6!"; // 9 bytes at offset 0
    const_data.extend_from_slice(seg0);
    let seg1_off = u32::try_from(const_data.len()).unwrap_or(0);
    let val_a: i64 = 0x1122_3344_5566_7788;
    let val_b: i64 = -0x0102_0304_0506_0708;
    const_data.extend_from_slice(&val_a.to_le_bytes());
    const_data.extend_from_slice(&val_b.to_le_bytes()); // 16 bytes at offset 9

    let rls_proven = format!(
        r#"
        cap 0x0
        data seg 0 "Rul-v0.6!"
        data seg 1 hex "{}"
        proof 3 rule=const_data_in_bounds target=0 lo=0 hi=9 witness=0
        proof 4 rule=const_data_in_bounds target=1 lo=0 hi=16 witness=0
        node $len0 = ConstDataLen i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $len1 = ConstDataLen i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
        node $c4 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=4
        node $ch = ConstDataLoad u8x1 fg=0 pol=l1 proof=3 tok=pure a=0 b=$c4
        node $ch_i = Cast i64x1 fg=0 pol=l1 proof=0 tok=pure a=$ch b=0
        node $mask7 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=7
        node $masked = BitAnd i64x1 fg=0 pol=l1 proof=0 tok=pure a=$len1 b=$mask7
        node $ch0 = ConstDataLoad u8x1 fg=0 pol=l1 proof=3 tok=pure a=0 b=$masked
        node $ch0_i = Cast i64x1 fg=0 pol=l1 proof=0 tok=pure a=$ch0 b=0
        node $c8 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=8
        node $q1 = ConstDataLoad i64x1 fg=0 pol=l1 proof=4 tok=pure a=1 b=$c8
        node $s1 = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$len0 b=$len1
        node $s2 = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$ch_i b=$ch0_i
        node $s3 = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$s1 b=$s2
        node $out = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$s3 b=$q1
        output $out
        "#,
        {
            let mut hex = String::new();
            for &b in const_data.get(seg1_off as usize..).unwrap_or(&[]) {
                use std::fmt::Write as _;
                let _ = write!(hex, "{b:02x}");
            }
            hex
        }
    );
    let prog_proven = assemble_rls(&rls_proven).map_err(|e| format!("assemble proven: {e}"))?;
    let summary_proven = verify_program(&prog_proven).map_err(|e| format!("verify proven: {e}"))?;
    assert!(
        summary_proven.zero_runtime_bounds_checks,
        "all ConstDataLoad nodes are statically proven in bounds"
    );

    let expected = 9i64 + 16i64 + i64::from(b'v') + i64::from(b'R') + val_b;
    let got = run_tier0_eq_aot_i64(&prog_proven, "const_proven")?;
    assert_eq!(got, expected);

    // 2. Verify ELF64 W^X layout: 2 PT_LOAD headers, Phdr #0 = PF_R|PF_X (5), Phdr #1 = PF_R (4, non-exec), page-aligned.
    let (elf_bytes, _) =
        compile_to_elf64_bytes(&prog_proven).map_err(|e| format!("compile_to_elf64: {e}"))?;
    let phnum = u16::from_le_bytes([
        elf_bytes.get(56).copied().unwrap_or(0),
        elf_bytes.get(57).copied().unwrap_or(0),
    ]);
    assert_eq!(phnum, 2, "const_data must emit a dedicated 2nd PT_LOAD");
    let phdr0_flags = u32::from_le_bytes([
        elf_bytes.get(64 + 4).copied().unwrap_or(0),
        elf_bytes.get(64 + 5).copied().unwrap_or(0),
        elf_bytes.get(64 + 6).copied().unwrap_or(0),
        elf_bytes.get(64 + 7).copied().unwrap_or(0),
    ]);
    let phdr1_off_base = 64 + 56;
    let phdr1_flags = u32::from_le_bytes([
        elf_bytes.get(phdr1_off_base + 4).copied().unwrap_or(0),
        elf_bytes.get(phdr1_off_base + 5).copied().unwrap_or(0),
        elf_bytes.get(phdr1_off_base + 6).copied().unwrap_or(0),
        elf_bytes.get(phdr1_off_base + 7).copied().unwrap_or(0),
    ]);
    let phdr1_offset = u64::from_le_bytes([
        elf_bytes.get(phdr1_off_base + 8).copied().unwrap_or(0),
        elf_bytes.get(phdr1_off_base + 9).copied().unwrap_or(0),
        elf_bytes.get(phdr1_off_base + 10).copied().unwrap_or(0),
        elf_bytes.get(phdr1_off_base + 11).copied().unwrap_or(0),
        elf_bytes.get(phdr1_off_base + 12).copied().unwrap_or(0),
        elf_bytes.get(phdr1_off_base + 13).copied().unwrap_or(0),
        elf_bytes.get(phdr1_off_base + 14).copied().unwrap_or(0),
        elf_bytes.get(phdr1_off_base + 15).copied().unwrap_or(0),
    ]);
    let phdr1_vaddr = u64::from_le_bytes([
        elf_bytes.get(phdr1_off_base + 16).copied().unwrap_or(0),
        elf_bytes.get(phdr1_off_base + 17).copied().unwrap_or(0),
        elf_bytes.get(phdr1_off_base + 18).copied().unwrap_or(0),
        elf_bytes.get(phdr1_off_base + 19).copied().unwrap_or(0),
        elf_bytes.get(phdr1_off_base + 20).copied().unwrap_or(0),
        elf_bytes.get(phdr1_off_base + 21).copied().unwrap_or(0),
        elf_bytes.get(phdr1_off_base + 22).copied().unwrap_or(0),
        elf_bytes.get(phdr1_off_base + 23).copied().unwrap_or(0),
    ]);
    let phdr1_filesz = u64::from_le_bytes([
        elf_bytes.get(phdr1_off_base + 32).copied().unwrap_or(0),
        elf_bytes.get(phdr1_off_base + 33).copied().unwrap_or(0),
        elf_bytes.get(phdr1_off_base + 34).copied().unwrap_or(0),
        elf_bytes.get(phdr1_off_base + 35).copied().unwrap_or(0),
        elf_bytes.get(phdr1_off_base + 36).copied().unwrap_or(0),
        elf_bytes.get(phdr1_off_base + 37).copied().unwrap_or(0),
        elf_bytes.get(phdr1_off_base + 38).copied().unwrap_or(0),
        elf_bytes.get(phdr1_off_base + 39).copied().unwrap_or(0),
    ]);
    assert_eq!(phdr0_flags, 5, "Phdr #0 must be PF_R | PF_X");
    assert_eq!(phdr1_flags, 4, "Phdr #1 must be PF_R only (non-executable)");
    assert_eq!(
        phdr1_offset % 4096,
        0,
        "rodata p_offset must be page-aligned"
    );
    assert_eq!(phdr1_vaddr % 4096, 0, "rodata p_vaddr must be page-aligned");
    assert_eq!(phdr1_filesz, prog_proven.const_data.len() as u64);
    assert_eq!(
        elf_bytes
            .get((phdr1_offset as usize)..(phdr1_offset as usize + prog_proven.const_data.len())),
        Some(prog_proven.const_data.as_slice())
    );

    // 3. Tier 1 and Tier 2 must fail closed with UnsupportedInColumnarTier.
    assert_eq!(
        execute_tier1_stencil(&prog_proven, &[]),
        Err(ExecError::UnsupportedInColumnarTier)
    );
    assert_eq!(
        execute_tier2_fused(&prog_proven, &[]),
        Err(ExecError::UnsupportedInColumnarTier)
    );

    // 4. Per-segment isolation: reading offset 4 from seg 0 (len=4) MUST trap even though seg 1 follows at offset 4!
    let cross_seg_rls = r#"
        cap 0x0
        data seg 0 "ABCD"
        data seg 1 "SECRET_DATA"
        node $idx = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=4
        node $b = ConstDataLoad u8x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$idx
        node $out = Cast i64x1 fg=0 pol=l1 proof=0 tok=pure a=$b b=0
        output $out
    "#;
    let prog_cross = assemble_rls(cross_seg_rls).map_err(|e| format!("assemble cross: {e}"))?;
    let summary_cross = verify_program(&prog_cross).map_err(|e| format!("verify cross: {e}"))?;
    assert!(!summary_cross.zero_runtime_bounds_checks);
    assert!(matches!(
        execute_tier0_interpreter(&prog_cross, &[]),
        Err(ExecError::ConstDataOutOfBounds {
            seg_id: 0,
            index: 4,
            seg_len: 4,
            ..
        })
    ));
    let aot_cross = run_aot_expect_err(&prog_cross, "const_cross_seg")?;
    assert!(
        matches!(aot_cross, AotError::GatherOutOfBounds),
        "expected exit {AOT_EXIT_GATHER_OOB}, got {aot_cross:?}"
    );
    run_aot_expect_trap_zero_stdout(&prog_cross, "const_cross_seg_z", AOT_EXIT_GATHER_OOB)?;

    // 5. Partial i64 OOB (idx=4 in 8-byte seg: 4 + 8 = 12 > 8), negative (-1), i64::MIN, and i64::MAX must trap with zero stdout.
    for (tag, idx_val) in [
        ("const_i64_partial_oob", 4i64),
        ("const_neg_oob", -1i64),
        ("const_i64_min_oob", i64::MIN),
        ("const_i64_max_oob", i64::MAX),
    ] {
        let oob_rls = format!(
            r#"
            cap 0x0
            data seg 0 hex "0102030405060708"
            data seg 1 hex "aabbccddeeff0011"
            node $idx = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64={idx_val}
            node $v = ConstDataLoad i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$idx
            output $v
            "#
        );
        let prog_oob = assemble_rls(&oob_rls).map_err(|e| format!("assemble {tag}: {e}"))?;
        assert!(matches!(
            execute_tier0_interpreter(&prog_oob, &[]),
            Err(ExecError::ConstDataOutOfBounds {
                seg_id: 0,
                index,
                seg_len: 8,
                ..
            }) if index == idx_val
        ));
        let err = run_aot_expect_err(&prog_oob, tag)?;
        assert!(matches!(err, AotError::GatherOutOfBounds));
        run_aot_expect_trap_zero_stdout(&prog_oob, &format!("{tag}_z"), AOT_EXIT_GATHER_OOB)?;
    }

    // 6. Reject unprovable ConstDataInBounds witness (never silently fall back).
    let bad_witness_rls = r#"
        cap 0x0
        data seg 0 "ABCDEFGH"
        proof 2 rule=const_data_in_bounds target=0 lo=0 hi=8 witness=0
        node $c = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1
        node $s = LocalSet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$c
        node $g = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $v = ConstDataLoad u8x1 fg=0 pol=l1 proof=2 tok=pure a=0 b=$g
        node $out = Cast i64x1 fg=0 pol=l1 proof=0 tok=pure a=$v b=0
        output $out
    "#;
    let prog_bad_wit = assemble_rls(bad_witness_rls).map_err(|e| format!("{e}"))?;
    assert!(matches!(
        verify_program(&prog_bad_wit).map_err(|d| d.kind),
        Err(VerifyErrorKind::InvalidRefinementProof { .. })
    ));

    // 7. Reject ConstDataInBounds for i64x1 when idx + 8 > bound_hi (even though idx < bound_hi!).
    let bad_i64_room_rls = r#"
        cap 0x0
        data seg 0 "0123456789abcdef"
        proof 2 rule=const_data_in_bounds target=0 lo=0 hi=16 witness=0
        node $c9 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=9
        node $v = ConstDataLoad i64x1 fg=0 pol=l1 proof=2 tok=pure a=0 b=$c9
        output $v
    "#;
    let prog_bad_room = assemble_rls(bad_i64_room_rls).map_err(|e| format!("{e}"))?;
    assert!(matches!(
        verify_program(&prog_bad_room).map_err(|d| d.kind),
        Err(VerifyErrorKind::InvalidRefinementProof { .. })
    ));

    // 8. Reject overlapping or out-of-order ConstDataSeg ranges in pre-scan.
    let prog_overlap = RulProgram {
        capability_mask: CAP_NONE,
        arenas: Vec::new(),
        proofs: vec![
            ProofEntry::const_data_seg(0, 0, 8),
            ProofEntry::const_data_seg(1, 4, 8),
        ],
        nodes: vec![NodeFrame::const_i64(0, ScheduleHint::UNFUSED)],
        output_node: 0,
        const_data: vec![0u8; 16],
        link_meta: Vec::new(),
    };
    assert!(matches!(
        verify_program(&prog_overlap).map_err(|d| d.kind),
        Err(VerifyErrorKind::InvalidConstDataTable { .. })
    ));

    // 9. Function-mode program reading const_data in both helper and entry functions.
    let func_const_rls = r#"
        cap 0x0
        data seg 0 "HELLO"
        proof 2 rule=func_sig target=0 lo=1 hi=i64 witness=i64
        proof 3 rule=func_sig target=1 lo=0 hi=0 witness=i64
        node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $idx = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $ch = ConstDataLoad u8x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$idx
        node $ch_i = Cast i64x1 fg=0 pol=l1 proof=0 tok=pure a=$ch b=0
        node $f0_end = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$ch_i b=1
        node $f1 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
        node $len = ConstDataLen i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $c1 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1
        node $last_idx = Sub i64x1 fg=0 pol=l1 proof=0 tok=pure a=$len b=$c1
        node $last_ch = Call i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$last_idx c=none
        node $f1_end = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$last_ch b=1
        output $f1
    "#;
    let prog_fn_const = assemble_rls(func_const_rls).map_err(|e| format!("{e}"))?;
    assert_eq!(
        run_tier0_eq_aot_i64(&prog_fn_const, "const_func_mode")?,
        i64::from(b'O')
    );

    Ok(())
}

#[test]
fn test_v06_br_table_verifier_dos_tier0_and_aot() -> Result<(), String> {
    // 1. Dense multi-way BrTable (I64 selector) across nested blocks: test negative, 0, boundary, and large selectors.
    let build_switch_i64 = |sel: i64| -> Result<RulProgram, String> {
        // Nested blocks:
        // Block $done (depth 3 at BrTable)
        //   Block $b2 (depth 2)
        //     Block $b1 (depth 1)
        //       Block $b0 (depth 0)
        //         BrTable targets=[0, 1, 2, 0] default=3 (falls out to default=999? Wait: init local 0 = 999 first!)
        //       End $b0 -> local0 = 10; Br 2 ($done)
        //     End $b1 -> local0 = 20; Br 1 ($done)
        //   End $b2 -> local0 = 30; Br 0 ($done)
        // End $done -> return local0
        let rls = format!(
            r#"
            cap 0x0
            br_table 0 default=3 targets=0,1,2,1
            node $def = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=999
            node $s0 = LocalSet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$def
            node $sel = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64={sel}
            node $bdone = Block i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
            node $bb2 = Block i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
            node $bb1 = Block i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
            node $bb0 = Block i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
            node $sw = BrTable i64x1 fg=0 pol=l1 proof=0 tok=pure a=$sel b=0
            node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
            node $c10 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=10
            node $set10 = LocalSet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$c10
            node $brk0 = Br i64x1 fg=0 pol=l1 proof=0 tok=pure a=2 b=0
            node $e1 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
            node $c20 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=20
            node $set20 = LocalSet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$c20
            node $brk1 = Br i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
            node $e2 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
            node $c30 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=30
            node $set30 = LocalSet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$c30
            node $edone = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
            node $out = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
            output $out
            "#
        );
        assemble_rls(&rls).map_err(|e| format!("assemble switch: {e}"))
    };

    for (sel, want) in [
        (i64::MIN, 999),
        (-5, 999),
        (-1, 999),
        (0, 10),
        (1, 20),
        (2, 30),
        (3, 20),
        (4, 999),
        (1000, 999),
        (i64::MAX, 999),
    ] {
        let prog = build_switch_i64(sel)?;
        let got = run_tier0_eq_aot_i64(&prog, &format!("br_table_i64_{}", sel.unsigned_abs()))?;
        assert_eq!(got, want, "BrTable mismatch for selector {sel}");
        if sel == 0 {
            // Lossless .rls round-trip check
            let rls_text = program_to_rls(&prog);
            assert!(rls_text.contains("br_table 0 default=3 targets=0,1,2,1"));
            let round = assemble_rls(&rls_text).map_err(|e| format!("roundtrip br_table: {e}"))?;
            assert_eq!(round, prog);
            assert_eq!(
                execute_tier1_stencil(&prog, &[]),
                Err(ExecError::UnsupportedInColumnarTier)
            );
            assert_eq!(
                execute_tier2_fused(&prog, &[]),
                Err(ExecError::UnsupportedInColumnarTier)
            );
        }
    }

    // 2. BrTable with U8 selector + unbounded Loop back-edge under CAP_DIVERGE.
    // Loop increments local0 from 0 -> 3 using BrTable on U8: 0,1,2 -> continue loop (depth 0), default -> exit block (depth 1).
    let loop_brtable_rls = r#"
        cap diverge
        br_table 0 default=1 targets=0,0,0
        node $z = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
        node $init = LocalSet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$z
        node $outer = Block i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $lp = Loop i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $cur = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $one = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1
        node $nxt = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$cur b=$one
        node $upd = LocalSet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$nxt
        node $sel_u8 = Cast u8x1 fg=0 pol=l1 proof=0 tok=pure a=$cur b=0
        node $sw = BrTable i64x1 fg=0 pol=l1 proof=0 tok=pure a=$sel_u8 b=0
        node $end_lp = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $end_outer = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $res = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        output $res
    "#;
    let prog_loop_sw = assemble_rls(loop_brtable_rls).map_err(|e| format!("{e}"))?;
    assert_eq!(run_tier0_eq_aot_i64(&prog_loop_sw, "br_table_u8_loop")?, 4);

    // 3. Adversarial: Reject BrTableSpec reuse across two BrTable sites (DuplicateBrTableUse).
    let dup_rls = r#"
        cap 0x0
        br_table 0 default=0 targets=0
        node $c = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
        node $b1 = Block i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $sw1 = BrTable i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c b=0
        node $e1 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $b2 = Block i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $sw2 = BrTable i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c b=0
        node $e2 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        output $c
    "#;
    let prog_dup = assemble_rls(dup_rls).map_err(|e| format!("{e}"))?;
    assert_eq!(
        verify_program(&prog_dup).map_err(|d| d.kind),
        Err(VerifyErrorKind::DuplicateBrTableUse { br_table_id: 0 })
    );

    // 4. Adversarial: Reject overlapping BrTableSpec payloads in const_data.
    let prog_overlap_bt = RulProgram {
        capability_mask: CAP_NONE,
        arenas: Vec::new(),
        proofs: vec![
            ProofEntry::br_table_spec(0, 0, 0, 2),
            ProofEntry::br_table_spec(1, 0, 4, 2),
        ],
        nodes: vec![NodeFrame::const_i64(0, ScheduleHint::UNFUSED)],
        output_node: 0,
        const_data: vec![0u8; 12],
        link_meta: Vec::new(),
    };
    assert!(matches!(
        verify_program(&prog_overlap_bt).map_err(|d| d.kind),
        Err(VerifyErrorKind::InvalidConstDataTable { .. })
    ));

    // 5. Adversarial: Reject BrTable targeting a bounded Loop BOTH without CAP_DIVERGE and with CAP_DIVERGE
    //    (whether targeted via both, case-only, or default-only)!
    for cap in ["0x0", "diverge"] {
        for (def_d, tgt) in [(0u32, "0"), (1u32, "0"), (0u32, "1")] {
            let bounded_loop_rls = format!(
                r#"
                cap {cap}
                proof 1 rule=loop_bound target=0 lo=0 hi=10 witness=0
                br_table 0 default={def_d} targets={tgt}
                node $c = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
                node $b = Block i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
                node $lp = Loop i64x1 fg=0 pol=l1 proof=1 tok=pure a=0 b=0
                node $sw = BrTable i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c b=0
                node $e = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
                node $eb = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
                output $c
                "#
            );
            let prog_bl = assemble_rls(&bounded_loop_rls).map_err(|e| format!("{e}"))?;
            assert!(
                matches!(
                    verify_program(&prog_bl).map_err(|d| d.kind),
                    Err(VerifyErrorKind::InvalidControlFlow(_))
                ),
                "BrTable targeting bounded Loop must be rejected even with cap={cap}, def={def_d}, tgt={tgt}"
            );
        }
    }

    // 6. Adversarial: Reject BrTable targeting a Block with mismatched region typestate on another exit.
    let bad_region_bt_rls = r#"
        cap 0x0
        proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
        br_table 0 default=0 targets=0
        node $c0 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
        node $b = Block i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $cond = CmpEq maskx1 fg=0 pol=l1 proof=0 tok=pure a=$c0 b=$c0
        node $if0 = If i64x1 fg=0 pol=l1 proof=0 tok=pure a=$cond b=0
        node $lease = RegionLease i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $sw = BrTable i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c0 b=0
        node $eif = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $eb = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        output $c0
    "#;
    let prog_bad_reg = assemble_rls(bad_region_bt_rls).map_err(|e| format!("{e}"))?;
    assert!(matches!(
        verify_program(&prog_bad_reg).map_err(|d| d.kind),
        Err(VerifyErrorKind::RegionTypestateMismatch { .. })
    ));

    // 6b. Adversarial: Reject reading a local not initialized on all BrTable paths.
    let uninit_bt_rls = r#"
        cap 0x0
        br_table 0 default=1 targets=0
        node $c0 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
        node $b1 = Block i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $b0 = Block i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $sw = BrTable i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c0 b=0
        node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $s0 = LocalSet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$c0
        node $e1 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $g0 = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        output $g0
    "#;
    let prog_uninit_bt = assemble_rls(uninit_bt_rls).map_err(|e| format!("{e}"))?;
    assert_eq!(
        verify_program(&prog_uninit_bt).map_err(|d| d.kind),
        Err(VerifyErrorKind::UninitializedLocalRead { slot: 0 })
    );

    // 7. Max-size BrTable (4096 cases) verifies in linear time and matches Tier 0 == AOT.
    let mut big_const = Vec::with_capacity(4096 * 4);
    for i in 0..4096u32 {
        let depth: u32 = if i == 4095 { 0 } else { 1 };
        big_const.extend_from_slice(&depth.to_le_bytes());
    }
    let big_bt_prog = RulProgram {
        capability_mask: CAP_NONE,
        arenas: Vec::new(),
        proofs: vec![ProofEntry::br_table_spec(0, 1, 0, 4096)],
        nodes: vec![
            NodeFrame::const_i64(77, ScheduleHint::UNFUSED), // 0
            NodeFrame {
                opcode: Opcode::LocalSet,
                type_lane: TypeLane::I64X1,
                schedule: ScheduleHint::UNFUSED,
                proof_idx: 0,
                token_op: TokenOp::PURE,
                in_a: 0,
                in_b: 0,
            }, // 1
            NodeFrame::const_i64(4095, ScheduleHint::UNFUSED), // 2
            NodeFrame {
                opcode: Opcode::Block,
                type_lane: TypeLane::I64X1,
                schedule: ScheduleHint::UNFUSED,
                proof_idx: 0,
                token_op: TokenOp::PURE,
                in_a: 0,
                in_b: 0,
            }, // 3 (depth 1)
            NodeFrame {
                opcode: Opcode::Block,
                type_lane: TypeLane::I64X1,
                schedule: ScheduleHint::UNFUSED,
                proof_idx: 0,
                token_op: TokenOp::PURE,
                in_a: 0,
                in_b: 0,
            }, // 4 (depth 0)
            NodeFrame {
                opcode: Opcode::BrTable,
                type_lane: TypeLane::I64X1,
                schedule: ScheduleHint::UNFUSED,
                proof_idx: 0,
                token_op: TokenOp::PURE,
                in_a: 2,
                in_b: 0,
            }, // 5
            NodeFrame {
                opcode: Opcode::End,
                type_lane: TypeLane::I64X1,
                schedule: ScheduleHint::UNFUSED,
                proof_idx: 0,
                token_op: TokenOp::PURE,
                in_a: 0,
                in_b: 0,
            }, // 6
            NodeFrame::const_i64(12345, ScheduleHint::UNFUSED), // 7
            NodeFrame {
                opcode: Opcode::LocalSet,
                type_lane: TypeLane::I64X1,
                schedule: ScheduleHint::UNFUSED,
                proof_idx: 0,
                token_op: TokenOp::PURE,
                in_a: 0,
                in_b: 7,
            }, // 8
            NodeFrame {
                opcode: Opcode::End,
                type_lane: TypeLane::I64X1,
                schedule: ScheduleHint::UNFUSED,
                proof_idx: 0,
                token_op: TokenOp::PURE,
                in_a: 0,
                in_b: 0,
            }, // 9
            NodeFrame {
                opcode: Opcode::LocalGet,
                type_lane: TypeLane::I64X1,
                schedule: ScheduleHint::UNFUSED,
                proof_idx: 0,
                token_op: TokenOp::PURE,
                in_a: 0,
                in_b: 0,
            }, // 10
        ],
        output_node: 10,
        const_data: big_const,
        link_meta: Vec::new(),
    };
    assert_eq!(run_tier0_eq_aot_i64(&big_bt_prog, "br_table_4096")?, 12345);

    // 8. Multi-arena ArenaVersionTrie linearity check: 4096-case BrTable with 2000 arenas must stay O(cases + log A), not O(cases * A).
    let mut many_arenas = Vec::with_capacity(2000);
    for id in 0..2000u32 {
        many_arenas.push(ArenaDescriptor {
            arena_id: id,
            element_count: 1,
            type_lane: TypeLane::F64X8,
            mutable: true,
            alignment: 64,
            initial_version: 1,
        });
    }
    let mut multi_arena_bt = big_bt_prog;
    multi_arena_bt.arenas = many_arenas;
    let (_, join_work) = verify_program_with_join_work(&multi_arena_bt)
        .map_err(|e| format!("multi-arena BrTable verify: {e}"))?;
    assert!(
        join_work <= 5_000,
        "expected O(case_count) join work on 4096-case BrTable x 2000 arenas, got {join_work}"
    );

    // 9. BrTable inside Func (testing func_mode AOT relocation + eval_functions_tier0) and rejecting BrTable targeting Func scope.
    let func_bt_rls = r#"
        cap diverge
        proof 1 rule=func_sig target=0 lo=1 hi=i64 witness=i64
        proof 2 rule=func_sig target=1 lo=0 hi=0 witness=i64
        br_table 0 default=2 targets=0,1
        br_table 1 default=1 targets=0,0

        # func 0: switch(x): 0 -> 100, 1 -> 200, default -> 300
        node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $x = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $c300 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=300
        node $sdef = LocalSet i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=$c300
        node $bd = Block i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $b1 = Block i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $b0 = Block i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $sw = BrTable i64x1 fg=0 pol=l1 proof=0 tok=pure a=$x b=0
        node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $c100 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=100
        node $s100 = LocalSet i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=$c100
        node $brk0 = Br i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
        node $e1 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $c200 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=200
        node $s200 = LocalSet i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=$c200
        node $ed = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $ret0 = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
        node $f0_end = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$ret0 b=1

        # func 1 (entry): loops i=0,1,2 calling f0(i) and summing 100 + 200 + 300 = 600
        node $f1 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
        node $z = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
        node $si = LocalSet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$z
        node $sa = LocalSet i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=$z
        node $outer = Block i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $lp = Loop i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $ci = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $ca = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
        node $v = Call i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$ci c=none
        node $na = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$ca b=$v
        node $ua = LocalSet i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=$na
        node $one = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1
        node $ni = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$ci b=$one
        node $ui = LocalSet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$ni
        node $swl = BrTable i64x1 fg=0 pol=l1 proof=0 tok=pure a=$ci b=1
        node $elp = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $eo = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $ans = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
        node $f1_end = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$ans b=1
        output $f1
    "#;
    let prog_func_bt = assemble_rls(func_bt_rls).map_err(|e| format!("{e}"))?;
    assert_eq!(
        run_tier0_eq_aot_i64(&prog_func_bt, "br_table_in_func")?,
        600
    );

    let bad_func_target_rls = r#"
        cap 0x0
        proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64
        br_table 0 default=0 targets=0
        node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $z = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
        node $sw = BrTable i64x1 fg=0 pol=l1 proof=0 tok=pure a=$z b=0
        node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        output $f0
    "#;
    let prog_bad_fn_bt = assemble_rls(bad_func_target_rls).map_err(|e| format!("{e}"))?;
    assert!(matches!(
        verify_program(&prog_bad_fn_bt).map_err(|d| d.kind),
        Err(VerifyErrorKind::InvalidControlFlow(_))
    ));

    Ok(())
}

#[test]
fn test_v06_call_indirect_verifier_dos_tier0_and_aot() -> Result<(), String> {
    // 1. Arity-4 CallIndirect (testing rdi, rsi, rdx, rcx register preservation when index & live values are in registers)
    //    plus arity-0 and arity-2 CallTables, plus lossless .rls round-trip!
    let arity4_rls = r#"
        cap 0x0
        proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64
        proof 2 rule=func_sig target=1 lo=4 hi=i64,i64,i64,i64 witness=i64
        proof 3 rule=func_sig target=2 lo=4 hi=i64,i64,i64,i64 witness=i64
        proof 4 rule=func_sig target=3 lo=0 hi=0 witness=i64
        call_table 0 funcs=0
        call_table 1 funcs=1,2

        # func 0: () -> 7
        node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $f0_c = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=7
        node $f0_end = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f0_c b=1

        # func 1: (a, b, c, d) -> a*1000 + b*100 + c*10 + d
        node $f1 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
        node $f1_a = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $f1_b = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
        node $f1_c = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=2 b=0
        node $f1_d = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=3 b=0
        node $k1000 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1000
        node $k100 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=100
        node $k10 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=10
        node $t1 = Mul i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f1_a b=$k1000
        node $t2 = Mul i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f1_b b=$k100
        node $t3 = Mul i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f1_c b=$k10
        node $s1 = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$t1 b=$t2
        node $s2 = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$t3 b=$f1_d
        node $f1_res = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$s1 b=$s2
        node $f1_end = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f1_res b=1

        # func 2: (a, b, c, d) -> d*1000 + c*100 + b*10 + a
        node $f2 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=2 b=0
        node $f2_a = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $f2_b = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
        node $f2_c = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=2 b=0
        node $f2_d = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=3 b=0
        node $m1000 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1000
        node $m100 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=100
        node $m10 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=10
        node $u1 = Mul i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f2_d b=$m1000
        node $u2 = Mul i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f2_c b=$m100
        node $u3 = Mul i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f2_b b=$m10
        node $v1 = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$u1 b=$u2
        node $v2 = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$u3 b=$f2_a
        node $f2_res = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$v1 b=$v2
        node $f2_end = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f2_res b=1

        # func 3 (entry): calls table 0 [0], then table 1 [0] and table 1 [1] with arity 4 while keeping live values across calls
        node $f3 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=3 b=0
        node $i0 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
        node $i1 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1
        node $a1 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1
        node $a2 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=2
        node $a3 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=3
        node $a4 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=4
        node $dyn_a1 = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$a1 b=$i0
        node $dyn_a2 = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$a2 b=$i0
        node $dyn_a3 = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$a3 b=$i0
        node $dyn_a4 = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$a4 b=$i0
        node $dyn_i0 = Sub i64x1 fg=0 pol=l1 proof=0 tok=pure a=$a1 b=$a1
        node $dyn_i1 = Sub i64x1 fg=0 pol=l1 proof=0 tok=pure a=$a2 b=$a1
        node $r0 = CallIndirect i64x1 fg=0 pol=l1 proof=0 tok=pure table=0 idx=$dyn_i0 b=none c=none
        node $ca1 = CallArg i64x1 fg=0 pol=l1 proof=0 tok=pure a=$dyn_a3 b=$dyn_a4
        node $r1 = CallIndirect i64x1 fg=0 pol=l1 proof=0 tok=pure table=1 idx=$dyn_i0 b=$dyn_a1 c=$dyn_a2
        node $ca2 = CallArg i64x1 fg=0 pol=l1 proof=0 tok=pure a=$dyn_a3 b=$dyn_a4
        node $r2 = CallIndirect i64x1 fg=0 pol=l1 proof=0 tok=pure table=1 idx=$dyn_i1 b=$dyn_a1 c=$dyn_a2
        node $sum01 = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$r0 b=$r1
        node $sum012 = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$sum01 b=$r2
        node $total = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$sum012 b=$dyn_a4
        node $f3_end = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$total b=1
        output $f3
    "#;
    let prog_a4 = assemble_rls(arity4_rls).map_err(|e| format!("assemble arity4: {e}"))?;
    let summary_a4 = verify_program(&prog_a4).map_err(|e| format!("verify arity4: {e}"))?;
    let cost = summary_a4
        .static_cost
        .ok_or_else(|| "expected StaticCostSummary in sandbox mode".to_owned())?;
    assert_eq!(cost.max_call_depth, 2);
    assert!(cost.worst_case_steps > 20);
    // 7 + 1234 + 4321 + 4 = 5566
    assert_eq!(
        run_tier0_eq_aot_i64(&prog_a4, "call_indirect_arity4")?,
        5566
    );

    // Lossless .rls round-trip & columnar fail-closed checks
    let rls_out = program_to_rls(&prog_a4);
    assert!(rls_out.contains("call_table 1 funcs=1,2"));
    assert!(rls_out.contains("CallIndirect"));
    let round_a4 = assemble_rls(&rls_out).map_err(|e| format!("roundtrip arity4: {e}"))?;
    assert_eq!(round_a4, prog_a4);
    assert_eq!(
        execute_tier1_stencil(&prog_a4, &[]),
        Err(ExecError::UnsupportedInColumnarTier)
    );
    assert_eq!(
        execute_tier2_fused(&prog_a4, &[]),
        Err(ExecError::UnsupportedInColumnarTier)
    );

    // 2. Out-of-bounds CallIndirect index (-1 and 2 for 2-entry table) -> Tier 0 IndirectCallOutOfBounds, AOT exit 76.
    for (tag, bad_idx) in [
        ("call_indirect_oob_neg", -1i64),
        ("call_indirect_oob_hi", 2i64),
    ] {
        let oob_rls = format!(
            r#"
            cap 0x0
            proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64
            proof 2 rule=func_sig target=1 lo=0 hi=0 witness=i64
            proof 3 rule=func_sig target=2 lo=0 hi=0 witness=i64
            call_table 0 funcs=0,1
            node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
            node $c0 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=11
            node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c0 b=1
            node $f1 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
            node $c1 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=22
            node $e1 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c1 b=1
            node $f2 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=2 b=0
            node $idx = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64={bad_idx}
            node $r = CallIndirect i64x1 fg=0 pol=l1 proof=0 tok=pure table=0 idx=$idx b=none c=none
            node $e2 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$r b=1
            output $f2
            "#
        );
        let prog_oob = assemble_rls(&oob_rls).map_err(|e| format!("{e}"))?;
        assert!(matches!(
            execute_tier0_interpreter(&prog_oob, &[]),
            Err(ExecError::IndirectCallOutOfBounds {
                table_id: 0,
                index,
                entry_count: 2,
                ..
            }) if index == bad_idx
        ));
        let aot_err = run_aot_expect_err(&prog_oob, tag)?;
        assert!(
            matches!(aot_err, AotError::IndirectCallOutOfBounds),
            "expected exit {AOT_EXIT_INDIRECT_CALL_OOB}, got {aot_err:?}"
        );
        run_aot_expect_trap_zero_stdout(
            &prog_oob,
            &format!("{tag}_z"),
            AOT_EXIT_INDIRECT_CALL_OOB,
        )?;
    }

    // 2b. Verify StaticCostSummary takes the worst-case max across heterogeneous-cost callees in a CallTable.
    let hetero_cost_rls = r#"
        cap 0x0
        proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64
        proof 2 rule=func_sig target=1 lo=0 hi=0 witness=i64
        proof 3 rule=func_sig target=2 lo=0 hi=0 witness=i64
        proof 4 rule=loop_bound target=0 lo=0 hi=20 witness=0
        call_table 0 funcs=0,1

        # func 0: cheap (1 step)
        node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $c0 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1
        node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c0 b=1

        # func 1: expensive (bounded loop of 20 iterations)
        node $f1 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
        node $z1 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
        node $s1 = LocalSet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$z1
        node $lp1 = Loop i64x1 fg=0 pol=l1 proof=4 tok=pure a=0 b=0
        node $g1 = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $o1 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1
        node $n1 = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$g1 b=$o1
        node $u1 = LocalSet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$n1
        node $k5 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=5
        node $cmp = CmpLt maskx1 fg=0 pol=l1 proof=0 tok=pure a=$n1 b=$k5
        node $br = BrIf i64x1 fg=0 pol=l1 proof=0 tok=pure a=$cmp b=0
        node $el1 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $r1 = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $e1 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$r1 b=1

        # func 2 (entry): invokes table 0 at index 0 (cheap at runtime), but static cost must reflect max(f0, f1)
        node $f2 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=2 b=0
        node $idx = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
        node $res = CallIndirect i64x1 fg=0 pol=l1 proof=0 tok=pure table=0 idx=$idx b=none c=none
        node $e2 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$res b=1
        output $f2
    "#;
    let prog_hetero = assemble_rls(hetero_cost_rls).map_err(|e| format!("{e}"))?;
    let hetero_sum = verify_program(&prog_hetero).map_err(|e| format!("{e}"))?;
    let hetero_cost = hetero_sum
        .static_cost
        .ok_or_else(|| "expected StaticCostSummary for hetero table".to_owned())?;
    assert!(
        hetero_cost.worst_case_steps > 100,
        "CallIndirect worst_case_steps must reflect expensive f1 in table, got {}",
        hetero_cost.worst_case_steps
    );
    assert_eq!(
        run_tier0_eq_aot_i64(&prog_hetero, "call_indirect_hetero_cost")?,
        1
    );

    // 3. Recursive CallIndirect under CAP_DIVERGE exceeding MAX_CALL_DEPTH -> Tier 0 CallDepthExceeded, AOT exit 75.
    let rec_rls = r#"
        cap diverge
        proof 1 rule=func_sig target=0 lo=1 hi=i64 witness=i64
        proof 2 rule=func_sig target=1 lo=0 hi=0 witness=i64
        call_table 0 funcs=0
        node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $n = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $z = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
        node $one = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1
        node $next = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$n b=$one
        node $r = CallIndirect i64x1 fg=0 pol=l1 proof=0 tok=pure table=0 idx=$z b=$next c=none
        node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$r b=1
        node $f1 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
        node $z1 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
        node $r1 = CallIndirect i64x1 fg=0 pol=l1 proof=0 tok=pure table=0 idx=$z1 b=$z1 c=none
        node $e1 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$r1 b=1
        output $f1
    "#;
    let prog_rec = assemble_rls(rec_rls).map_err(|e| format!("{e}"))?;
    assert!(matches!(
        execute_tier0_interpreter(&prog_rec, &[]),
        Err(ExecError::CallDepthExceeded { .. })
    ));
    let aot_rec = run_aot_expect_err(&prog_rec, "call_indirect_stack_overflow")?;
    assert!(
        matches!(aot_rec, AotError::StackOverflow),
        "expected exit {AOT_EXIT_STACK_OVERFLOW}, got {aot_rec:?}"
    );

    // 4. Frame-rule lease preservation across CallIndirect:
    //    slot 0 is mutated by table 0 (`dom={0}, owned->owned`), while caller holds a live ReadLease on slot 1!
    let frame_rule_rls = r#"
        cap 0x0
        proof 1 rule=region_decl target=0 lo=1 hi=4 witness=i64:plain
        proof 2 rule=region_decl target=1 lo=1 hi=4 witness=i64:plain
        proof 3 rule=func_sig target=0 lo=1 hi=i64 witness=i64
        proof 4 rule=func_sig target=1 lo=1 hi=i64 witness=i64
        proof 5 rule=func_sig target=2 lo=0 hi=0 witness=i64
        proof 6 rule=func_region_effect func=0 slot=0 pre=owned post=owned
        proof 7 rule=func_region_effect func=1 slot=0 pre=owned post=owned
        proof 8 rule=func_region_effect func=2 slot=0 pre=owned post=owned
        proof 9 rule=func_region_effect func=2 slot=1 pre=owned post=owned
        call_table 0 funcs=0,1

        node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $f0_x = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $f0_z = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
        node $f0_st = RegionStore i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$f0_z c=$f0_x
        node $f0_ld = RegionLoad i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$f0_z
        node $f0_end = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f0_ld b=1

        node $f1 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
        node $f1_x = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $f1_k = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=10
        node $f1_v = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f1_x b=$f1_k
        node $f1_z = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
        node $f1_st = RegionStore i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$f1_z c=$f1_v
        node $f1_ld = RegionLoad i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$f1_z
        node $f1_end = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f1_ld b=1

        node $f2 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=2 b=0
        node $z = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
        node $one = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1
        node $v42 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=42
        node $st1 = RegionStore i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=$z c=$v42
        node $ls1 = RegionLease i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
        node $c1 = CallIndirect i64x1 fg=0 pol=l1 proof=0 tok=pure table=0 idx=$one b=$v42 c=none
        node $r1 = RegionLoad i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=$z
        node $el1 = RegionEndLease i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
        node $out = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c1 b=$r1
        node $f2_end = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$out b=1
        output $f2
    "#;
    let prog_frame = assemble_rls(frame_rule_rls).map_err(|e| format!("{e}"))?;
    // f1(42) = 52, region[1][0] = 42 -> 94
    assert_eq!(
        run_tier0_eq_aot_i64(&prog_frame, "call_indirect_frame_rule")?,
        94
    );

    // 4b. Arities 1, 2, 3, 4 (with incoming arg3 in `rcx` used as BOTH CallIndirect index AND permuted arg!) + void CallIndirect.
    let all_arities_rls = r#"
        cap 0x0
        proof 1 rule=region_decl target=0 lo=1 hi=4 witness=i64:plain
        proof 2 rule=func_sig target=0 lo=1 hi=i64 witness=void
        proof 3 rule=func_sig target=1 lo=2 hi=i64,i64 witness=i64
        proof 4 rule=func_sig target=2 lo=3 hi=i64,i64,i64 witness=i64
        proof 5 rule=func_sig target=3 lo=4 hi=i64,i64,i64,i64 witness=i64
        proof 6 rule=func_sig target=4 lo=4 hi=i64,i64,i64,i64 witness=i64
        proof 7 rule=func_sig target=5 lo=4 hi=i64,i64,i64,i64 witness=i64
        proof 8 rule=func_sig target=6 lo=0 hi=0 witness=i64
        call_table 0 funcs=0
        call_table 1 funcs=1
        call_table 2 funcs=2
        call_table 3 funcs=3,4

        # func 0: void (x) -> region[0][0] = x
        node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $f0_x = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $f0_z = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
        node $f0_st = RegionStore i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$f0_z c=$f0_x
        node $f0_end = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0

        # func 1: (a, b) -> a * 10 + b
        node $f1 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
        node $f1_a = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $f1_b = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
        node $f1_k = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=10
        node $f1_m = Mul i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f1_a b=$f1_k
        node $f1_r = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f1_m b=$f1_b
        node $f1_end = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f1_r b=1

        # func 2: (a, b, c) -> a * 100 + b * 10 + c
        node $f2 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=2 b=0
        node $f2_a = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $f2_b = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
        node $f2_c = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=2 b=0
        node $f2_k100 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=100
        node $f2_k10 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=10
        node $f2_m1 = Mul i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f2_a b=$f2_k100
        node $f2_m2 = Mul i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f2_b b=$f2_k10
        node $f2_s = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f2_m1 b=$f2_m2
        node $f2_r = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f2_s b=$f2_c
        node $f2_end = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f2_r b=1

        # func 3: (a, b, c, d) -> a + b*2 + c*3 + d*4
        node $f3 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=3 b=0
        node $f3_a = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $f3_b = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
        node $f3_c = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=2 b=0
        node $f3_d = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=3 b=0
        node $f3_k2 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=2
        node $f3_k3 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=3
        node $f3_k4 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=4
        node $f3_t1 = Mul i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f3_b b=$f3_k2
        node $f3_t2 = Mul i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f3_c b=$f3_k3
        node $f3_t3 = Mul i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f3_d b=$f3_k4
        node $f3_s1 = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f3_a b=$f3_t1
        node $f3_s2 = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f3_t2 b=$f3_t3
        node $f3_r = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f3_s1 b=$f3_s2
        node $f3_end = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f3_r b=1

        # func 4: (a, b, c, d) -> a*1000 + b*100 + c*10 + d
        node $f4 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=4 b=0
        node $f4_a = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $f4_b = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
        node $f4_c = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=2 b=0
        node $f4_d = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=3 b=0
        node $f4_k1000 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1000
        node $f4_k100 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=100
        node $f4_k10 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=10
        node $f4_t1 = Mul i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f4_a b=$f4_k1000
        node $f4_t2 = Mul i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f4_b b=$f4_k100
        node $f4_t3 = Mul i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f4_c b=$f4_k10
        node $f4_s1 = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f4_t1 b=$f4_t2
        node $f4_s2 = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f4_t3 b=$f4_d
        node $f4_r = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f4_s1 b=$f4_s2
        node $f4_end = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f4_r b=1

        # func 5: forwards & rotates incoming (p0=rdi, p1=rsi, p2=rdx, p3=rcx) using p3 (rcx) as CallIndirect index!
        node $f5 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=5 b=0
        node $p0 = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $p1 = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
        node $p2 = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=2 b=0
        node $p3 = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=3 b=0
        node $f5_ca = CallArg i64x1 fg=0 pol=l1 proof=0 tok=pure a=$p1 b=$p0
        node $f5_r = CallIndirect i64x1 fg=0 pol=l1 proof=0 tok=pure table=3 idx=$p3 b=$p3 c=$p2
        node $f5_end = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$f5_r b=1

        # func 6 (entry): invokes table 0 (void, arity 1), table 1 (arity 2), table 2 (arity 3), and func 5 (arity 4 -> table 3)
        node $f6 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=6 b=0
        node $z = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
        node $c1 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1
        node $c2 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=2
        node $c3 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=3
        node $c5 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=5
        node $c7 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=7
        node $v0 = CallIndirect i64x1 fg=0 pol=l1 proof=0 tok=pure table=0 idx=$z b=$c7 c=none
        node $r_reg = RegionLoad i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$z
        node $r_ar2 = CallIndirect i64x1 fg=0 pol=l1 proof=0 tok=pure table=1 idx=$z b=$c3 c=$c5
        node $ca_ar3 = CallArg i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c7 b=none
        node $r_ar3 = CallIndirect i64x1 fg=0 pol=l1 proof=0 tok=pure table=2 idx=$z b=$c2 c=$c3
        node $ca_f5 = CallArg i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c3 b=$c1
        node $r_ar4 = Call i64x1 fg=0 pol=l1 proof=0 tok=pure a=5 b=$c5 c=$c2
        node $s1 = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$r_reg b=$r_ar2
        node $s2 = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$r_ar3 b=$r_ar4
        node $out = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$s1 b=$s2
        node $f6_end = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$out b=1
        output $f6
    "#;
    let prog_arities = assemble_rls(all_arities_rls).map_err(|e| format!("{e}"))?;
    // r_reg = 7; r_ar2 = 35; r_ar3 = 237; r_ar4 = f4(1, 3, 2, 5) = 1325 -> total = 1604
    assert_eq!(
        run_tier0_eq_aot_i64(&prog_arities, "call_indirect_all_arities")?,
        1604
    );

    // 4c. Negative frame-rule checks on CallIndirect: leased slot violation + callee footprint exceeding caller.
    let bad_lease_call_rls = r#"
        cap 0x0
        proof 1 rule=region_decl target=0 lo=1 hi=4 witness=i64:plain
        proof 2 rule=func_sig target=0 lo=0 hi=0 witness=i64
        proof 3 rule=func_sig target=1 lo=0 hi=0 witness=i64
        proof 4 rule=func_region_effect func=0 slot=0 pre=owned post=owned
        proof 5 rule=func_region_effect func=1 slot=0 pre=owned post=owned
        call_table 0 funcs=0
        node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $c0 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1
        node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c0 b=1
        node $f1 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
        node $z = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
        node $ls = RegionLease i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $ci = CallIndirect i64x1 fg=0 pol=l1 proof=0 tok=pure table=0 idx=$z b=none c=none
        node $el = RegionEndLease i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $e1 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$ci b=1
        output $f1
    "#;
    let prog_bad_lease = assemble_rls(bad_lease_call_rls).map_err(|e| format!("{e}"))?;
    assert!(matches!(
        verify_program(&prog_bad_lease).map_err(|d| d.kind),
        Err(VerifyErrorKind::RegionTypestateMismatch { .. })
    ));

    let bad_dom_call_rls = r#"
        cap 0x0
        proof 1 rule=region_decl target=0 lo=1 hi=4 witness=i64:plain
        proof 2 rule=func_sig target=0 lo=0 hi=0 witness=i64
        proof 3 rule=func_sig target=1 lo=0 hi=0 witness=i64
        proof 4 rule=func_region_effect func=0 slot=0 pre=owned post=owned
        proof 5 rule=func_region_effect func=1 slot=none
        call_table 0 funcs=0
        node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $c0 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1
        node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c0 b=1
        node $f1 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
        node $z = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
        node $ci = CallIndirect i64x1 fg=0 pol=l1 proof=0 tok=pure table=0 idx=$z b=none c=none
        node $e1 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$ci b=1
        output $f1
    "#;
    let prog_bad_dom = assemble_rls(bad_dom_call_rls).map_err(|e| format!("{e}"))?;
    assert!(matches!(
        verify_program(&prog_bad_dom).map_err(|d| d.kind),
        Err(VerifyErrorKind::CalleeFootprintExceedsCaller {
            callee_func_id: 0,
            extra_mask: 1
        })
    ));

    // 5. Reject CallTable whose entries disagree on FuncSig or FuncRegionContract.
    let bad_sig_table_rls = r#"
        cap 0x0
        proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64
        proof 2 rule=func_sig target=1 lo=1 hi=i64 witness=i64
        proof 3 rule=func_sig target=2 lo=0 hi=0 witness=i64
        call_table 0 funcs=0,1
        node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $c0 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1
        node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c0 b=1
        node $f1 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
        node $p = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $e1 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$p b=1
        node $f2 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=2 b=0
        node $c2 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
        node $e2 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c2 b=1
        output $f2
    "#;
    let prog_bad_sig = assemble_rls(bad_sig_table_rls).map_err(|e| format!("{e}"))?;
    assert!(matches!(
        verify_program(&prog_bad_sig).map_err(|d| d.kind),
        Err(VerifyErrorKind::InvalidConstDataTable { .. })
    ));

    let bad_contract_table_rls = r#"
        cap 0x0
        proof 1 rule=region_decl target=0 lo=1 hi=4 witness=i64:plain
        proof 2 rule=func_sig target=0 lo=0 hi=0 witness=i64
        proof 3 rule=func_sig target=1 lo=0 hi=0 witness=i64
        proof 4 rule=func_sig target=2 lo=0 hi=0 witness=i64
        proof 5 rule=func_region_effect func=0 slot=0 pre=owned post=owned
        proof 6 rule=func_region_effect func=1 slot=none
        call_table 0 funcs=0,1
        node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $c0 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1
        node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c0 b=1
        node $f1 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
        node $c1 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=2
        node $e1 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c1 b=1
        node $f2 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=2 b=0
        node $c2 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
        node $e2 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c2 b=1
        output $f2
    "#;
    let prog_bad_contract = assemble_rls(bad_contract_table_rls).map_err(|e| format!("{e}"))?;
    assert!(matches!(
        verify_program(&prog_bad_contract).map_err(|d| d.kind),
        Err(VerifyErrorKind::InvalidConstDataTable { .. })
    ));

    // 6. Reject sandbox CallIndirect when CallTable contains max_func_id >= caller_func_id without CAP_DIVERGE.
    let bad_dag_rls = r#"
        cap 0x0
        proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64
        proof 2 rule=func_sig target=1 lo=0 hi=0 witness=i64
        call_table 0 funcs=0,1
        node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
        node $c0 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
        node $r0 = CallIndirect i64x1 fg=0 pol=l1 proof=0 tok=pure table=0 idx=$c0 b=none c=none
        node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$r0 b=1
        node $f1 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
        node $c1 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1
        node $e1 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c1 b=1
        output $f1
    "#;
    let prog_bad_dag = assemble_rls(bad_dag_rls).map_err(|e| format!("{e}"))?;
    assert!(matches!(
        verify_program(&prog_bad_dag).map_err(|d| d.kind),
        Err(VerifyErrorKind::MissingCapability {
            required: CAP_DIVERGE,
            ..
        })
    ));

    // 7. Adversarial multi-site CallIndirect linearity check:
    //    1 CallTable with 4096 entries invoked from 4096 CallIndirect sites must verify in O(N + table_len), not O(N * table_len)!
    let mut const_data = Vec::with_capacity(4096 * 4);
    for _ in 0..4096u32 {
        const_data.extend_from_slice(&0u32.to_le_bytes());
    }
    let proofs = vec![
        ProofEntry::func_sig(0, &[], Some(ElemType::I64)),
        ProofEntry::func_sig(1, &[], Some(ElemType::I64)),
        ProofEntry::call_table(0, 0, 4096),
    ];
    let mut nodes: Vec<NodeFrame> = Vec::with_capacity(4 + 4096 * 2 + 4);
    // func 0: returns 1
    nodes.push(NodeFrame {
        opcode: Opcode::Func,
        type_lane: TypeLane::I64X1,
        schedule: ScheduleHint::UNFUSED,
        proof_idx: 0,
        token_op: TokenOp::PURE,
        in_a: 0,
        in_b: 0,
    });
    nodes.push(NodeFrame::const_i64(1, ScheduleHint::UNFUSED));
    nodes.push(NodeFrame {
        opcode: Opcode::End,
        type_lane: TypeLane::I64X1,
        schedule: ScheduleHint::UNFUSED,
        proof_idx: 0,
        token_op: TokenOp::PURE,
        in_a: 1,
        in_b: 1,
    });
    let entry_start = u32::try_from(nodes.len()).unwrap_or(0);
    // func 1 (entry): 4096 CallIndirect sites summing results
    nodes.push(NodeFrame {
        opcode: Opcode::Func,
        type_lane: TypeLane::I64X1,
        schedule: ScheduleHint::UNFUSED,
        proof_idx: 0,
        token_op: TokenOp::PURE,
        in_a: 1,
        in_b: 0,
    });
    nodes.push(NodeFrame::const_i64(0, ScheduleHint::UNFUSED)); // rel 1 = idx 0 & initial acc
    let mut acc_rel: u32 = 1;
    let empty_args = NodeFrame::encode_u16_pair(CALL_ARG_SENTINEL, CALL_ARG_SENTINEL);
    let call_in_a = NodeFrame::encode_u16_pair(0, 1); // table 0, idx rel 1
    for _ in 0..4096usize {
        let call_rel = u32::try_from(nodes.len()).unwrap_or(0) - entry_start;
        nodes.push(NodeFrame {
            opcode: Opcode::CallIndirect,
            type_lane: TypeLane::I64X1,
            schedule: ScheduleHint::UNFUSED,
            proof_idx: 0,
            token_op: TokenOp::PURE,
            in_a: call_in_a,
            in_b: empty_args,
        });
        let add_rel = u32::try_from(nodes.len()).unwrap_or(0) - entry_start;
        nodes.push(NodeFrame {
            opcode: Opcode::Add,
            type_lane: TypeLane::I64X1,
            schedule: ScheduleHint::UNFUSED,
            proof_idx: 0,
            token_op: TokenOp::PURE,
            in_a: acc_rel,
            in_b: call_rel,
        });
        acc_rel = add_rel;
    }
    nodes.push(NodeFrame {
        opcode: Opcode::End,
        type_lane: TypeLane::I64X1,
        schedule: ScheduleHint::UNFUSED,
        proof_idx: 0,
        token_op: TokenOp::PURE,
        in_a: acc_rel,
        in_b: 1,
    });

    let dos_prog = RulProgram {
        capability_mask: CAP_NONE,
        arenas: Vec::new(),
        proofs,
        nodes,
        output_node: entry_start,
        const_data,
        link_meta: Vec::new(),
    };
    let t0 = Instant::now();
    let dos_summary = verify_program(&dos_prog).map_err(|e| format!("verify dos_prog: {e}"))?;
    let elapsed_ms = t0.elapsed().as_millis();
    assert!(
        elapsed_ms < 500,
        "4096-site x 4096-entry CallIndirect verification took {elapsed_ms} ms (expected O(N) linear)"
    );
    assert!(dos_summary.static_cost.is_some());
    assert_eq!(
        run_tier0_eq_aot_i64(&dos_prog, "call_indirect_4096_sites")?,
        4096
    );

    Ok(())
}
