//! End-to-end CLI tests for `rul link` (AC-4): mixed `.rl`/`.rls` inputs,
//! capability escalation rejection, `rul check`, `rul build`, and `rul run`
//! (`Tier 0 == ELF64 AOT`) on a 3-module program with shared regions, constant
//! segments, `BrTable`, and `CallIndirect`.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![deny(clippy::indexing_slicing)]

use std::process::Command;

use rul::suite10::save_rl_file;
use rul::transpiler::assemble_rls;

#[test]
fn test_cli_rul_link_end_to_end() -> Result<(), String> {
    let rul_bin = env!("CARGO_BIN_EXE_rul");
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let work_dir =
        std::env::temp_dir().join(format!("rul_cli_link_v06_{}_{nanos}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work_dir);
    std::fs::create_dir_all(&work_dir).map_err(|e| e.to_string())?;

    // Module 0 (saved as binary `.rl` to verify mixed `.rl` + `.rls` inputs!):
    // Pure leaf with ConstDataSeg ("XYZ" len=3, byte[0]='X'=88) + BrTable.
    let mod0_rls = "\
cap 0x0
export func leaf.eval 0
data seg 0 \"XYZ\"
proof 2 rule=const_data_in_bounds target=0 lo=0 hi=3 witness=0
br_table 0 default=1 targets=0
proof 4 rule=func_sig target=0 lo=1 hi=i64 witness=i64
node $f0 = Func i64x1 a=0 b=0
node $x = LocalGet i64x1 a=0 b=0
node $b1 = Block i64x1 a=0 b=0
node $b0 = Block i64x1 a=0 b=0
node $bt = BrTable i64x1 a=$x b=0
node $eb0 = End i64x1 a=0 b=0
node $z = ConstI64 i64x1 const_i64=0
node $ch = ConstDataLoad u8x1 proof=2 a=0 b=$z
node $ch_i = Cast i64x1 a=$ch b=0
node $ret0 = Return i64x1 a=$ch_i b=0
node $eb1 = End i64x1 a=0 b=0
node $len = ConstDataLen i64x1 a=0 b=0
node $r = Add i64x1 a=$x b=$len
node $e0 = End i64x1 a=$r b=1
output $f0
";
    let mod0_prog = assemble_rls(mod0_rls).map_err(|e| e.to_string())?;
    let mod0_rl_path = work_dir.join("mod0_leaf.rl");
    let mod0_rl_str = mod0_rl_path
        .to_str()
        .ok_or_else(|| "non-UTF8 temp path".to_owned())?;
    save_rl_file(mod0_rl_str, &mod0_prog).map_err(|e| e.to_string())?;

    // Module 1 (`.rls`): exports shared region `shared.slot`, imports `leaf.eval`,
    // dispatches via CallIndirect (CallTable with imported + local function), writes result to `shared.slot`.
    let mod1_rls = "\
cap 0x0
import func leaf.eval 0
export region shared.slot 0
export func mid.step 2
data seg 0 \"HI\"
proof 2 rule=region_decl target=0 lo=1 hi=4 witness=i64:plain
proof 3 rule=region_index_in_bounds target=0 lo=0 hi=1 witness=0
proof 4 rule=func_sig target=0 lo=1 hi=i64 witness=i64
proof 5 rule=func_sig target=1 lo=1 hi=i64 witness=i64
proof 6 rule=func_sig target=2 lo=1 hi=i64 witness=i64
proof 7 rule=func_region_effect func=0 slot=none pre=owned post=owned
proof 8 rule=func_region_effect func=1 slot=none pre=owned post=owned
proof 9 rule=func_region_effect func=2 slot=0 pre=owned post=owned
call_table 0 funcs=0,1
node $f1 = Func i64x1 a=1 b=0
node $v = LocalGet i64x1 a=0 b=0
node $c10 = ConstI64 i64x1 const_i64=10
node $m = Mul i64x1 a=$v b=$c10
node $e1 = End i64x1 a=$m b=1
node $f2 = Func i64x1 a=2 b=0
node $x = LocalGet i64x1 a=0 b=0
node $c0 = ConstI64 i64x1 const_i64=0
node $c1 = ConstI64 i64x1 const_i64=1
node $a = CallIndirect i64x1 table=0 idx=$c0 b=$c0 c=none
node $b = CallIndirect i64x1 table=0 idx=$c1 b=$x c=none
node $slen = ConstDataLen i64x1 a=0 b=0
node $sum1 = Add i64x1 a=$a b=$b
node $sum2 = Add i64x1 a=$sum1 b=$slen
node $st = RegionStore i64x1 proof=3 a=0 b=$c0 c=$sum2
node $e2 = End i64x1 a=$sum2 b=1
output $f2
";
    let mod1_path = work_dir.join("mod1_mid.rls");
    std::fs::write(&mod1_path, mod1_rls).map_err(|e| e.to_string())?;

    // Module 2 (`.rls`): entry module importing `shared.slot` and `mid.step`,
    // has its own private region slot 0, holds a live lease on slot 0 across `mid.step(5)`
    // -> mid.step(5) = 'X'(88) + 5*10(50) + len("HI")(2) = 140; + priv(100) + shared(140) = 380!
    let mod2_rls = "\
cap 0x0
import region shared.slot 1
import func mid.step 0
export func app.entry 1
proof 1 rule=region_decl target=0 lo=1 hi=4 witness=i64:plain
proof 2 rule=region_decl target=1 lo=1 hi=4 witness=i64:plain
proof 3 rule=region_index_in_bounds target=0 lo=0 hi=1 witness=0
proof 4 rule=region_index_in_bounds target=1 lo=0 hi=1 witness=0
proof 5 rule=func_sig target=0 lo=1 hi=i64 witness=i64
proof 6 rule=func_sig target=1 lo=0 hi=0 witness=i64
proof 7 rule=func_region_effect func=0 slot=1 pre=owned post=owned
proof 8 rule=func_region_effect func=1 slot=0 pre=owned post=owned
proof 9 rule=func_region_effect func=1 slot=1 pre=owned post=owned
node $f1 = Func i64x1 a=1 b=0
node $c0 = ConstI64 i64x1 const_i64=0
node $c5 = ConstI64 i64x1 const_i64=5
node $c100 = ConstI64 i64x1 const_i64=100
node $st0 = RegionStore i64x1 proof=3 a=0 b=$c0 c=$c100
node $l0 = RegionLease i64x1 a=0 b=0
node $res = Call i64x1 a=0 b=$c5 c=none
node $priv = RegionLoad i64x1 proof=3 a=0 b=$c0
node $el0 = RegionEndLease i64x1 a=0 b=0
node $sh = RegionLoad i64x1 proof=4 a=1 b=$c0
node $s1 = Add i64x1 a=$res b=$priv
node $s2 = Add i64x1 a=$s1 b=$sh
node $e1 = End i64x1 a=$s2 b=1
output $f1
";
    let mod2_path = work_dir.join("mod2_entry.rls");
    std::fs::write(&mod2_path, mod2_rls).map_err(|e| e.to_string())?;

    // 1. Run `rul link` on mixed `.rl` + `.rls` inputs with `--entry app.entry`.
    let out_rl = work_dir.join("linked_app.rl");
    let link_out = Command::new(rul_bin)
        .current_dir(&work_dir)
        .arg("link")
        .arg(&mod0_rl_path)
        .arg(&mod1_path)
        .arg(&mod2_path)
        .arg("-o")
        .arg(&out_rl)
        .arg("--entry")
        .arg("app.entry")
        .output()
        .map_err(|e| e.to_string())?;
    assert!(
        link_out.status.success(),
        "rul link failed: stderr={}",
        String::from_utf8_lossy(&link_out.stderr)
    );
    let link_stdout = String::from_utf8_lossy(&link_out.stdout);
    assert!(
        link_stdout.contains("[RUL LINK OK]"),
        "missing [RUL LINK OK]: {link_stdout}"
    );
    assert!(
        link_stdout.contains("Cap=0x0"),
        "missing Cap=0x0 in output: {link_stdout}"
    );
    assert!(
        link_stdout.contains("Modules=3"),
        "missing Modules=3 in output: {link_stdout}"
    );
    assert!(out_rl.exists(), "linked output file was not written");

    // 2. Run `rul check` on linked `.rl` -> passes with `StaticCostSummary` (`Cost(...)`).
    let check_out = Command::new(rul_bin)
        .current_dir(&work_dir)
        .arg("check")
        .arg(&out_rl)
        .output()
        .map_err(|e| e.to_string())?;
    assert!(
        check_out.status.success(),
        "rul check failed: stderr={}",
        String::from_utf8_lossy(&check_out.stderr)
    );
    let check_stdout = String::from_utf8_lossy(&check_out.stdout);
    assert!(check_stdout.contains("[RUL CHECK OK]"), "{check_stdout}");
    assert!(check_stdout.contains("Cost(steps="), "{check_stdout}");

    // 3. Run `rul build` on linked `.rl` -> compiles standalone ELF64 binary cleanly.
    let build_out = Command::new(rul_bin)
        .current_dir(&work_dir)
        .arg("build")
        .arg(&out_rl)
        .output()
        .map_err(|e| e.to_string())?;
    assert!(
        build_out.status.success(),
        "rul build failed: stderr={}",
        String::from_utf8_lossy(&build_out.stderr)
    );
    let build_stdout = String::from_utf8_lossy(&build_out.stdout);
    assert!(build_stdout.contains("[RUL BUILD OK]"), "{build_stdout}");

    // 4. Run `rul run` on linked `.rl` -> verifies Tier 0 == ELF64 AOT byte-identical (`Output_i64=380`).
    let run_out = Command::new(rul_bin)
        .current_dir(&work_dir)
        .arg("run")
        .arg(&out_rl)
        .output()
        .map_err(|e| e.to_string())?;
    assert!(
        run_out.status.success(),
        "rul run failed: stderr={}",
        String::from_utf8_lossy(&run_out.stderr)
    );
    let run_stdout = String::from_utf8_lossy(&run_out.stdout);
    assert!(run_stdout.contains("[RUL RUN OK]"), "{run_stdout}");
    assert!(
        run_stdout.contains("Output_i64=380"),
        "expected Output_i64=380, got: {run_stdout}"
    );

    // 5. Capability escalation failure via CLI: library with `cap 0x8` linked into `cap 0x0` entry fails closed!
    let esc_lib_rls = "\
cap 0x8
export func esc.fn 0
proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64
node $f0 = Func i64x1 a=0 b=0
node $c = ConstI64 i64x1 const_i64=11
node $e = End i64x1 a=$c b=1
output $f0
";
    let esc_entry_rls = "\
cap 0x0
import func esc.fn 0
proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64
proof 2 rule=func_sig target=1 lo=0 hi=0 witness=i64
node $f1 = Func i64x1 a=1 b=0
node $r = Call i64x1 a=0 b=none c=none
node $e = End i64x1 a=$r b=1
output $f1
";
    let esc_lib_path = work_dir.join("esc_lib.rls");
    let esc_entry_path = work_dir.join("esc_entry.rls");
    let esc_out_rl = work_dir.join("esc_out.rl");
    std::fs::write(&esc_lib_path, esc_lib_rls).map_err(|e| e.to_string())?;
    std::fs::write(&esc_entry_path, esc_entry_rls).map_err(|e| e.to_string())?;

    let esc_fail = Command::new(rul_bin)
        .current_dir(&work_dir)
        .arg("link")
        .arg(&esc_lib_path)
        .arg(&esc_entry_path)
        .arg("-o")
        .arg(&esc_out_rl)
        .output()
        .map_err(|e| e.to_string())?;
    assert!(
        !esc_fail.status.success(),
        "expected rul link to fail on CapabilityEscalation"
    );
    let esc_stderr = String::from_utf8_lossy(&esc_fail.stderr);
    assert!(
        esc_stderr.contains("capability escalation rejected"),
        "expected capability escalation diagnostic, got: {esc_stderr}"
    );
    assert!(
        !esc_out_rl.exists(),
        "failed link must not write output file"
    );

    // Explicitly passing `--cap 0x8` allows linking cleanly!
    let esc_ok = Command::new(rul_bin)
        .current_dir(&work_dir)
        .arg("link")
        .arg(&esc_lib_path)
        .arg(&esc_entry_path)
        .arg("-o")
        .arg(&esc_out_rl)
        .arg("--cap")
        .arg("0x8")
        .output()
        .map_err(|e| e.to_string())?;
    assert!(
        esc_ok.status.success(),
        "expected rul link --cap 0x8 to succeed, stderr={}",
        String::from_utf8_lossy(&esc_ok.stderr)
    );
    let esc_ok_stdout = String::from_utf8_lossy(&esc_ok.stdout);
    assert!(esc_ok_stdout.contains("[RUL LINK OK]"), "{esc_ok_stdout}");
    assert!(esc_ok_stdout.contains("Cap=0x8"), "{esc_ok_stdout}");

    let _ = std::fs::remove_dir_all(&work_dir);
    Ok(())
}
