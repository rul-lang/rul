//! Milestone 2d (`v0.6`) test suite:
//! - AC-1: Independent `verify_module` and unit/differential tests for `stdlib/vec.rls`,
//!   `stdlib/string.rls` (exhaustive RFC 3629 UTF-8), and `stdlib/hashmap.rls` (collisions,
//!   tombstones, rehash via `RegionDrop`/`RegionMove`), plus `CAP_NONE` sandbox `StaticCostSummary`
//!   and live `RegionLease` calls across `Tier 0 == ELF64 AOT`.
//! - AC-2: End-to-end linked exit programs (`json_parser`, `kv_store`, `wc_grep`) verified via
//!   `verify_program`, round-tripped through `.rl`, and matched byte-for-byte across
//!   `Tier 0 == ELF64 AOT == rustc -O` on valid and adversarial inputs.
//! - AC-3 (`#[ignore]`): Release performance gate (`<= 2.0x` vs `rustc -O`, `>= 100 ms` reference,
//!   min of 5 alternating single-threaded runs, byte-identical stdout).
//! - AC-4 (`#[ignore]`): Release linearity gate (`verify_module`, `link_modules`, and
//!   `compile_to_elf64_bytes` independently `<= 15.0x` for 60k/6k multi-module inputs, and
//!   `<= 50 ns/node` AOT codegen on `stress_60k`).

use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::aot::{compile_rl_to_executable, compile_to_elf64_bytes};
use crate::bytecode::{RulProgram, CAP_DIVERGE, CAP_NONE};
use crate::engine::{execute_tier0_interpreter_with_input, RuntimeValue};
use crate::linker::{link_modules, LinkOptions};
use crate::transpiler::{assemble_rls, program_to_rls};
use crate::verifier::{verify_module, verify_program};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn unique_tmp_path(tag: &str, ext: &str) -> PathBuf {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    let seq = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    std::env::temp_dir().join(format!(
        "rul_v06_stdlib_{tag}_{}_{seq}_{nanos}.{ext}",
        std::process::id()
    ))
}

fn load_rls(rel_path: &str) -> Result<RulProgram, String> {
    let full = repo_root().join(rel_path);
    let src = fs::read_to_string(&full).map_err(|e| format!("read {}: {e}", full.display()))?;
    assemble_rls(&src).map_err(|e| format!("assemble {}: {e}", full.display()))
}

/// Spawns `exe` with `stdin_bytes` piped to stdin, retrying transient Linux `ETXTBSY` (os error 26)
/// caused by concurrent test threads forking child processes while freshly written ELF files close.
fn run_exe_with_stdin(exe: &Path, stdin_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let mut attempt = 0u32;
    loop {
        let spawn_res = Command::new(exe)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();
        match spawn_res {
            Ok(mut child) => {
                if let Some(mut sin) = child.stdin.take() {
                    let _ = sin.write_all(stdin_bytes);
                }
                let out = child
                    .wait_with_output()
                    .map_err(|e| format!("wait {}: {e}", exe.display()))?;
                if !out.status.success() {
                    return Err(format!(
                        "{} exited with {:?}, stderr={}",
                        exe.display(),
                        out.status,
                        String::from_utf8_lossy(&out.stderr)
                    ));
                }
                return Ok(out.stdout);
            }
            Err(e) if e.raw_os_error() == Some(26) && attempt < 64 => {
                attempt += 1;
                std::thread::sleep(Duration::from_micros(500));
            }
            Err(e) => return Err(format!("spawn {}: {e}", exe.display())),
        }
    }
}

fn run_exe_timed_from_file(exe: &Path, input_file: &Path) -> Result<(Duration, Vec<u8>), String> {
    let mut attempt = 0u32;
    loop {
        let stdin_file = File::open(input_file)
            .map_err(|e| format!("open input {}: {e}", input_file.display()))?;
        let t0 = Instant::now();
        let res = Command::new(exe)
            .stdin(Stdio::from(stdin_file))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output();
        match res {
            Ok(output) => {
                let elapsed = t0.elapsed();
                if !output.status.success() {
                    return Err(format!(
                        "{} exited with {:?}, stderr={}",
                        exe.display(),
                        output.status,
                        String::from_utf8_lossy(&output.stderr)
                    ));
                }
                return Ok((elapsed, output.stdout));
            }
            Err(e) if e.raw_os_error() == Some(26) && attempt < 64 => {
                attempt += 1;
                std::thread::sleep(Duration::from_micros(500));
            }
            Err(e) => return Err(format!("spawn {}: {e}", exe.display())),
        }
    }
}

fn compile_rust_ref(rel_rs: &str, tag: &str) -> Result<PathBuf, String> {
    let rs_path = repo_root().join(rel_rs);
    let out_exe = unique_tmp_path(tag, "rs_bin");
    let out = Command::new("rustc")
        .arg("-O")
        .arg(&rs_path)
        .arg("-o")
        .arg(&out_exe)
        .output()
        .map_err(|e| format!("invoke rustc -O on {}: {e}", rs_path.display()))?;
    if !out.status.success() {
        let _ = fs::remove_file(&out_exe);
        return Err(format!(
            "rustc -O {} failed: {}",
            rs_path.display(),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(out_exe)
}

fn compile_rul_exe(
    prog: &RulProgram,
    tag: &str,
) -> Result<(PathBuf, crate::aot::AotCompileReport), String> {
    let exe = unique_tmp_path(tag, "elf");
    let rep =
        compile_rl_to_executable(prog, &exe).map_err(|e| format!("aot compile {tag}: {e}"))?;
    if let Ok(f) = File::open(&exe) {
        let _ = f.sync_all();
    }
    Ok((exe, rep))
}

fn scalar_i64(val: &RuntimeValue) -> Result<i64, String> {
    match val {
        RuntimeValue::TypedScalar(ts) => ts
            .as_i64()
            .ok_or_else(|| format!("expected I64 scalar, got {ts:?}")),
        other => Err(format!("expected I64 scalar, got {other:?}")),
    }
}

fn run_tier0_eq_aot_bytes(
    prog: &RulProgram,
    exe: &Path,
    input: &[u8],
    label: &str,
) -> Result<(i64, Vec<u8>), String> {
    let t0 = execute_tier0_interpreter_with_input(prog, &[], input)
        .map_err(|e| format!("tier0 failed on {label}: {e}"))?;
    let ret = scalar_i64(&t0.output)?;
    let aot_out =
        run_exe_with_stdin(exe, input).map_err(|e| format!("aot failed on {label}: {e}"))?;
    let t0_bytes = t0.output_bytes.unwrap_or_default();
    if t0_bytes != aot_out {
        return Err(format!(
            "Tier 0 != ELF64 AOT stdout on {label}:\n  tier0={:?}\n  aot  ={:?}",
            String::from_utf8_lossy(&t0_bytes),
            String::from_utf8_lossy(&aot_out)
        ));
    }
    Ok((ret, aot_out))
}

fn run_tier0_eq_aot_scalar_only(prog: &RulProgram, tag: &str) -> Result<i64, String> {
    let t0 = execute_tier0_interpreter_with_input(prog, &[], &[])
        .map_err(|e| format!("tier0 failed on {tag}: {e}"))?;
    let expected = scalar_i64(&t0.output)?;
    let (exe, _) = compile_rul_exe(prog, tag)?;
    let aot_res = run_exe_with_stdin(&exe, &[]);
    let _ = fs::remove_file(&exe);
    let stdout = aot_res?;
    let bytes: [u8; 8] = stdout
        .as_slice()
        .try_into()
        .map_err(|_e| format!("{tag}: expected 8-byte scalar stdout, got {}", stdout.len()))?;
    let actual = i64::from_le_bytes(bytes);
    if actual != expected {
        return Err(format!(
            "{tag}: Tier 0 ({expected}) != ELF64 AOT ({actual})"
        ));
    }
    Ok(expected)
}

// =========================================================================
// AC-1: Stdlib Independent Verification, Concurrent Vec, Sandbox Cost,
//       Live RegionLease Calls, Exhaustive UTF-8 & HashMap Collisions
// =========================================================================
#[test]
fn test_v06_stdlib_modules_verify_and_unit_suite() -> Result<(), String> {
    // 1. Independently assemble, verify_module, and round-trip all 3 stdlib modules.
    let mod_vec = load_rls("stdlib/vec.rls")?;
    let mod_str = load_rls("stdlib/string.rls")?;
    let mod_map = load_rls("stdlib/hashmap.rls")?;

    for (name, prog) in [
        ("stdlib/vec.rls", &mod_vec),
        ("stdlib/string.rls", &mod_str),
        ("stdlib/hashmap.rls", &mod_map),
    ] {
        assert_eq!(
            prog.capability_mask, CAP_NONE,
            "{name} must be CAP_NONE sandbox-compatible"
        );
        verify_module(prog).map_err(|e| format!("verify_module({name}) failed: {e}"))?;
        let rl_bytes = prog.to_rl_bytes();
        let decoded = RulProgram::from_rl_bytes(&rl_bytes)
            .map_err(|e| format!("from_rl_bytes({name}) failed: {e}"))?;
        assert_eq!(&decoded, prog, "{name} binary round-trip mismatch");
        let rls_roundtrip = program_to_rls(&decoded);
        let reassembled =
            assemble_rls(&rls_roundtrip).map_err(|e| format!("reassemble({name}) failed: {e}"))?;
        assert_eq!(&reassembled, prog, "{name} .rls round-trip mismatch");
    }

    // 2. Concurrent multi-instance Vec (2 I64Vecs + 2 U8Vecs interleaved across growth/relocation)
    //    in CAP_NONE sandbox mode verifying StaticCostSummary + Tier 0 == ELF64 AOT.
    let vec_harness_rls = "\
cap 0x0
import region vec.meta 0
import region vec.i64_data 1
import region vec.u8_data 2
import func vec::new_i64 0
import func vec::new_u8 1
import func vec::len 2
import func vec::clear 3
import func vec::push_i64 4
import func vec::pop_i64 5
import func vec::get_i64 6
import func vec::set_i64 7
import func vec::append_i64 8
import func vec::push_u8 9
import func vec::pop_u8 10
import func vec::get_u8 11
import func vec::set_u8 12
import func vec::append_u8 13
proof 1 rule=loop_bound target=0 lo=0 hi=64 witness=0
proof 2 rule=region_decl target=0 lo=4096 hi=4096 witness=i64:plain
proof 3 rule=region_decl target=1 lo=0 hi=4194304 witness=i64:plain
proof 4 rule=region_decl target=2 lo=0 hi=16777216 witness=u8:plain
proof 5 rule=func_sig target=0 lo=1 hi=i64 witness=i64
proof 6 rule=func_sig target=1 lo=1 hi=i64 witness=i64
proof 7 rule=func_sig target=2 lo=1 hi=i64 witness=i64
proof 8 rule=func_sig target=3 lo=1 hi=i64 witness=i64
proof 9 rule=func_sig target=4 lo=2 hi=i64,i64 witness=i64
proof 10 rule=func_sig target=5 lo=1 hi=i64 witness=i64
proof 11 rule=func_sig target=6 lo=2 hi=i64,i64 witness=i64
proof 12 rule=func_sig target=7 lo=3 hi=i64,i64,i64 witness=i64
proof 13 rule=func_sig target=8 lo=2 hi=i64,i64 witness=i64
proof 14 rule=func_sig target=9 lo=2 hi=i64,u8 witness=i64
proof 15 rule=func_sig target=10 lo=1 hi=i64 witness=u8
proof 16 rule=func_sig target=11 lo=2 hi=i64,i64 witness=u8
proof 17 rule=func_sig target=12 lo=3 hi=i64,i64,u8 witness=i64
proof 18 rule=func_sig target=13 lo=2 hi=i64,i64 witness=i64
proof 19 rule=func_sig target=14 lo=0 hi=0 witness=i64
proof 20 rule=func_region_effect func=0 slot=0 pre=owned post=owned
proof 21 rule=func_region_effect func=0 slot=1 pre=owned post=owned
proof 22 rule=func_region_effect func=1 slot=0 pre=owned post=owned
proof 23 rule=func_region_effect func=1 slot=2 pre=owned post=owned
proof 24 rule=func_region_effect func=2 slot=0 pre=owned post=owned
proof 25 rule=func_region_effect func=3 slot=0 pre=owned post=owned
proof 26 rule=func_region_effect func=4 slot=0 pre=owned post=owned
proof 27 rule=func_region_effect func=4 slot=1 pre=owned post=owned
proof 28 rule=func_region_effect func=5 slot=0 pre=owned post=owned
proof 29 rule=func_region_effect func=5 slot=1 pre=owned post=owned
proof 30 rule=func_region_effect func=6 slot=0 pre=owned post=owned
proof 31 rule=func_region_effect func=6 slot=1 pre=owned post=owned
proof 32 rule=func_region_effect func=7 slot=0 pre=owned post=owned
proof 33 rule=func_region_effect func=7 slot=1 pre=owned post=owned
proof 34 rule=func_region_effect func=8 slot=0 pre=owned post=owned
proof 35 rule=func_region_effect func=8 slot=1 pre=owned post=owned
proof 36 rule=func_region_effect func=9 slot=0 pre=owned post=owned
proof 37 rule=func_region_effect func=9 slot=2 pre=owned post=owned
proof 38 rule=func_region_effect func=10 slot=0 pre=owned post=owned
proof 39 rule=func_region_effect func=10 slot=2 pre=owned post=owned
proof 40 rule=func_region_effect func=11 slot=0 pre=owned post=owned
proof 41 rule=func_region_effect func=11 slot=2 pre=owned post=owned
proof 42 rule=func_region_effect func=12 slot=0 pre=owned post=owned
proof 43 rule=func_region_effect func=12 slot=2 pre=owned post=owned
proof 44 rule=func_region_effect func=13 slot=0 pre=owned post=owned
proof 45 rule=func_region_effect func=13 slot=2 pre=owned post=owned
proof 46 rule=func_region_effect func=14 slot=0 pre=owned post=owned
proof 47 rule=func_region_effect func=14 slot=1 pre=owned post=owned
proof 48 rule=func_region_effect func=14 slot=2 pre=owned post=owned
node $f14 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=14 b=0
node $c8 = ConstI64 i64x1 const_i64=8
node $v0 = Call i64x1 a=0 b=$c8 c=none
node $v1 = Call i64x1 a=0 b=$c8 c=none
node $u0 = Call i64x1 a=1 b=$c8 c=none
node $u1 = Call i64x1 a=1 b=$c8 c=none
node $ls_v0 = LocalSet i64x1 a=0 b=$v0
node $ls_v1 = LocalSet i64x1 a=1 b=$v1
node $ls_u0 = LocalSet i64x1 a=2 b=$u0
node $ls_u1 = LocalSet i64x1 a=3 b=$u1
node $z = ConstI64 i64x1 const_i64=0
node $ls_i = LocalSet i64x1 a=4 b=$z
node $blk1 = Block i64x1 a=0 b=0
node $lp1 = Loop i64x1 proof=1 a=0 b=0
node $i = LocalGet i64x1 a=4 b=0
node $c20 = ConstI64 i64x1 const_i64=20
node $lt = CmpLt maskx1 a=$i b=$c20
node $not = MaskNot maskx1 a=$lt b=0
node $bri = BrIf i64x1 a=$not b=1
node $gv0 = LocalGet i64x1 a=0 b=0
node $gv1 = LocalGet i64x1 a=1 b=0
node $gu0 = LocalGet i64x1 a=2 b=0
node $gu1 = LocalGet i64x1 a=3 b=0
node $c100 = ConstI64 i64x1 const_i64=100
node $v1_val = Add i64x1 a=$i b=$c100
node $p0 = Call i64x1 a=4 b=$gv0 c=$i
node $p1 = Call i64x1 a=4 b=$gv1 c=$v1_val
node $b0 = Cast u8x1 a=$i b=0
node $c50 = ConstI64 i64x1 const_i64=50
node $b1_i = Add i64x1 a=$i b=$c50
node $b1 = Cast u8x1 a=$b1_i b=0
node $pu0 = Call i64x1 a=9 b=$gu0 c=$b0
node $pu1 = Call i64x1 a=9 b=$gu1 c=$b1
node $c1 = ConstI64 i64x1 const_i64=1
node $inext = Add i64x1 a=$i b=$c1
node $ls_i2 = LocalSet i64x1 a=4 b=$inext
node $br = Br i64x1 a=0 b=0
node $end_lp1 = End i64x1 a=0 b=0
node $end_blk1 = End i64x1 a=0 b=0
# Set v0[5] = 999, pop v1 (returns 119), append v1 -> v0 (len = 20 + 19 = 39)
node $gv0b = LocalGet i64x1 a=0 b=0
node $gv1b = LocalGet i64x1 a=1 b=0
node $c5 = ConstI64 i64x1 const_i64=5
node $c999 = ConstI64 i64x1 const_i64=999
node $ca_set = CallArg i64x1 a=$c999 b=none
node $ok_set = Call i64x1 a=7 b=$gv0b c=$c5
node $popped = Call i64x1 a=5 b=$gv1b c=none
node $app_len = Call i64x1 a=8 b=$gv0b c=$gv1b
node $got5 = Call i64x1 a=6 b=$gv0b c=$c5
node $c25 = ConstI64 i64x1 const_i64=25
node $got25 = Call i64x1 a=6 b=$gv0b c=$c25
# Clear v1 (returns kind=1), check len(v1) == 0
node $clr1 = Call i64x1 a=3 b=$gv1b c=none
node $len1 = Call i64x1 a=2 b=$gv1b c=none
# U8Vecs: set u0[3] = 77, pop u1 (69), append u1 -> u0 (len 39), get u0[3] (77), get u0[22] (52), clear u1 (kind=2)
node $gu0b = LocalGet i64x1 a=2 b=0
node $gu1b = LocalGet i64x1 a=3 b=0
node $c3 = ConstI64 i64x1 const_i64=3
node $c77 = ConstI64 i64x1 const_i64=77
node $u77 = Cast u8x1 a=$c77 b=0
node $ca_uset = CallArg i64x1 a=$u77 b=none
node $ok_uset = Call i64x1 a=12 b=$gu0b c=$c3
node $upop = Call u8x1 a=10 b=$gu1b c=none
node $uapp = Call i64x1 a=13 b=$gu0b c=$gu1b
node $ug3 = Call u8x1 a=11 b=$gu0b c=$c3
node $c22 = ConstI64 i64x1 const_i64=22
node $ug22 = Call u8x1 a=11 b=$gu0b c=$c22
node $uclr = Call i64x1 a=3 b=$gu1b c=none
# Fold all observable checks into deterministic sum:
# ok_set(1) + popped(119) + app_len(39) + got5(999) + got25(105) + clr1(1) + len1(0)
# + ok_uset(1) + upop(69) + uapp(39) + ug3(77) + ug22(52) + uclr(2) = 1504
node $s1 = Add i64x1 a=$ok_set b=$popped
node $s2 = Add i64x1 a=$s1 b=$app_len
node $s3 = Add i64x1 a=$s2 b=$got5
node $s4 = Add i64x1 a=$s3 b=$got25
node $s5 = Add i64x1 a=$s4 b=$clr1
node $s6 = Add i64x1 a=$s5 b=$len1
node $s7 = Add i64x1 a=$s6 b=$ok_uset
node $upop_i = Cast i64x1 a=$upop b=0
node $s8 = Add i64x1 a=$s7 b=$upop_i
node $s9 = Add i64x1 a=$s8 b=$uapp
node $ug3_i = Cast i64x1 a=$ug3 b=0
node $s10 = Add i64x1 a=$s9 b=$ug3_i
node $ug22_i = Cast i64x1 a=$ug22 b=0
node $s11 = Add i64x1 a=$s10 b=$ug22_i
node $tot = Add i64x1 a=$s11 b=$uclr
node $e14 = End i64x1 a=$tot b=1
output $f14
";
    let vec_harness = assemble_rls(vec_harness_rls).map_err(|e| format!("vec harness: {e}"))?;
    let vec_link = link_modules(&[mod_vec.clone(), vec_harness], &LinkOptions::default())
        .map_err(|e| format!("link vec harness: {e}"))?;
    assert_eq!(vec_link.capability_mask, CAP_NONE);
    let vec_cost = vec_link
        .summary
        .static_cost
        .ok_or_else(|| "expected StaticCostSummary for CAP_NONE linked Vec harness".to_owned())?;
    assert!(vec_cost.worst_case_steps > 0);
    assert!(vec_cost.max_region_bytes > 0);
    assert_eq!(
        run_tier0_eq_aot_scalar_only(&vec_link.program, "stdlib_vec_concurrent")?,
        1504
    );

    // 3. Exhaustive RFC 3629 UTF-8 validation + live RegionLease(std.io.in) + CAP_NONE StaticCostSummary.
    let utf8_harness_rls = "\
cap 0x0
export region std.io.in 0
export region std.io.out 1
import func str::utf8_validate 0
import func str::fmt_i64 1
proof 1 rule=region_decl target=0 lo=0 hi=134217728 witness=u8:input
proof 2 rule=region_decl target=1 lo=0 hi=268435456 witness=u8:output
proof 3 rule=func_sig target=0 lo=2 hi=i64,i64 witness=i64
proof 4 rule=func_sig target=1 lo=1 hi=i64 witness=i64
proof 5 rule=func_sig target=2 lo=0 hi=0 witness=i64
proof 6 rule=func_region_effect func=0 slot=0 pre=leased post=leased
proof 7 rule=func_region_effect func=1 slot=1 pre=owned post=owned
proof 8 rule=func_region_effect func=2 slot=0 pre=owned post=owned
proof 9 rule=func_region_effect func=2 slot=1 pre=owned post=owned
node $f2 = Func i64x1 a=2 b=0
node $z = ConstI64 i64x1 const_i64=0
node $len = RegionLen i64x1 a=0 b=0
node $l = RegionLease u8x1 a=0 b=0
node $ok = Call i64x1 a=0 b=$z c=$len
node $fmt = Call i64x1 a=1 b=$ok c=none
node $el = RegionEndLease u8x1 a=0 b=0
node $e2 = End i64x1 a=$ok b=1
output $f2
";
    let utf8_harness = assemble_rls(utf8_harness_rls).map_err(|e| format!("utf8 harness: {e}"))?;
    let utf8_link = link_modules(&[mod_str.clone(), utf8_harness], &LinkOptions::default())
        .map_err(|e| format!("link utf8 harness: {e}"))?;
    assert_eq!(utf8_link.capability_mask, CAP_NONE);
    assert!(utf8_link.summary.static_cost.is_some());
    let (utf8_exe, _) = compile_rul_exe(&utf8_link.program, "stdlib_utf8")?;

    let utf8_cases: [(&str, &[u8], i64); 24] = [
        ("empty", b"", 1),
        ("ascii", b"Hello, Rul v0.6! 0123456789\n", 1),
        ("2byte_min_U0080", &[0xC2, 0x80], 1),
        ("2byte_max_U07FF", &[0xDF, 0xBF], 1),
        ("3byte_min_U0800", &[0xE0, 0xA0, 0x80], 1),
        ("3byte_before_surrogate_UD7FF", &[0xED, 0x9F, 0xBF], 1),
        ("3byte_after_surrogate_UE000", &[0xEE, 0x80, 0x80], 1),
        ("3byte_max_UFFFF", &[0xEF, 0xBF, 0xBF], 1),
        ("4byte_min_U10000", &[0xF0, 0x90, 0x80, 0x80], 1),
        ("4byte_max_U10FFFF", &[0xF4, 0x8F, 0xBF, 0xBF], 1),
        ("mixed_valid", "café 日本語 🚀!".as_bytes(), 1),
        ("stray_cont_80", &[0x80], 0),
        ("stray_cont_BF", &[0xBF], 0),
        ("overlong_2byte_C0_80", &[0xC0, 0x80], 0),
        ("overlong_2byte_C1_BF", &[0xC1, 0xBF], 0),
        ("overlong_3byte_E0_80_80", &[0xE0, 0x80, 0x80], 0),
        ("overlong_3byte_E0_9F_BF", &[0xE0, 0x9F, 0xBF], 0),
        ("surrogate_UD800", &[0xED, 0xA0, 0x80], 0),
        ("surrogate_UDFFF", &[0xED, 0xBF, 0xBF], 0),
        ("overlong_4byte_F0_80_80_80", &[0xF0, 0x80, 0x80, 0x80], 0),
        ("overlong_4byte_F0_8F_BF_BF", &[0xF0, 0x8F, 0xBF, 0xBF], 0),
        ("above_U10FFFF_F4_90", &[0xF4, 0x90, 0x80, 0x80], 0),
        ("invalid_lead_F5", &[0xF5, 0x80, 0x80, 0x80], 0),
        ("truncated_4byte", &[0xF1, 0x80, 0x80], 0),
    ];
    for (name, bytes, expected) in utf8_cases {
        let (ret, out) = run_tier0_eq_aot_bytes(&utf8_link.program, &utf8_exe, bytes, name)?;
        assert_eq!(ret, expected, "utf8 case {name} return mismatch");
        let exp_str = format!("{expected}");
        assert_eq!(out, exp_str.as_bytes(), "utf8 case {name} stdout mismatch");
    }
    let _ = fs::remove_file(&utf8_exe);

    // 4. Sandbox CAP_NONE combined String + HashMap harness:
    //    - Tests str::eq_span, str::find_substr, str::parse_i64, str::fmt_i64 (including i64::MIN!)
    //    - Tests map::init(8) tiny capacity forcing collisions, tombstones, lookup past tombstones,
    //      tombstone slot reuse, i64::MIN / -1 / 0 values with map::last_found(), and rehash (>10 keys)
    //      via RegionDrop + RegionMove -- ALL while holding live RegionLease(std.io.in)!
    let map_str_harness_rls = "\
cap 0x0
export region std.io.in 0
export region std.io.out 1
import region map.meta 2
import region map.kmeta 3
import region map.hash 4
import region map.val 5
import region map.keys 6
import region map.kmeta_scratch 7
import region map.hash_scratch 8
import region map.val_scratch 9
import region map.keys_scratch 10
import func str::eq_span 0
import func str::find_substr 1
import func str::parse_i64 2
import func str::fmt_i64 3
import func map::init 4
import func map::set_span 5
import func map::get_span 6
import func map::del_span 7
import func map::count 8
import func map::last_found 9
proof 1 rule=loop_bound target=0 lo=0 hi=64 witness=0
proof 2 rule=region_decl target=0 lo=0 hi=134217728 witness=u8:input
proof 3 rule=region_decl target=1 lo=0 hi=268435456 witness=u8:output
proof 4 rule=region_decl target=2 lo=8 hi=8 witness=i64:plain
proof 5 rule=region_decl target=3 lo=0 hi=33554432 witness=i64:plain
proof 6 rule=region_decl target=4 lo=0 hi=33554432 witness=i64:plain
proof 7 rule=region_decl target=5 lo=0 hi=33554432 witness=i64:plain
proof 8 rule=region_decl target=6 lo=0 hi=134217728 witness=u8:plain
proof 9 rule=region_decl target=7 lo=0 hi=33554432 witness=i64:plain
proof 10 rule=region_decl target=8 lo=0 hi=33554432 witness=i64:plain
proof 11 rule=region_decl target=9 lo=0 hi=33554432 witness=i64:plain
proof 12 rule=region_decl target=10 lo=0 hi=134217728 witness=u8:plain
proof 13 rule=func_sig target=0 lo=3 hi=i64,i64,i64 witness=i64
proof 14 rule=func_sig target=1 lo=4 hi=i64,i64,i64,i64 witness=i64
proof 15 rule=func_sig target=2 lo=2 hi=i64,i64 witness=i64
proof 16 rule=func_sig target=3 lo=1 hi=i64 witness=i64
proof 17 rule=func_sig target=4 lo=1 hi=i64 witness=i64
proof 18 rule=func_sig target=5 lo=3 hi=i64,i64,i64 witness=i64
proof 19 rule=func_sig target=6 lo=2 hi=i64,i64 witness=i64
proof 20 rule=func_sig target=7 lo=2 hi=i64,i64 witness=i64
proof 21 rule=func_sig target=8 lo=0 hi=0 witness=i64
proof 22 rule=func_sig target=9 lo=0 hi=0 witness=i64
proof 23 rule=func_sig target=10 lo=0 hi=0 witness=i64
proof 24 rule=func_region_effect func=0 slot=0 pre=leased post=leased
proof 25 rule=func_region_effect func=1 slot=0 pre=leased post=leased
proof 26 rule=func_region_effect func=2 slot=0 pre=leased post=leased
proof 27 rule=func_region_effect func=3 slot=1 pre=owned post=owned
proof 28 rule=func_region_effect func=4 slot=2 pre=owned post=owned
proof 29 rule=func_region_effect func=4 slot=3 pre=owned post=owned
proof 30 rule=func_region_effect func=4 slot=4 pre=owned post=owned
proof 31 rule=func_region_effect func=4 slot=5 pre=owned post=owned
proof 32 rule=func_region_effect func=4 slot=6 pre=owned post=owned
proof 33 rule=func_region_effect func=4 slot=7 pre=owned post=owned
proof 34 rule=func_region_effect func=4 slot=8 pre=owned post=owned
proof 35 rule=func_region_effect func=4 slot=9 pre=owned post=owned
proof 36 rule=func_region_effect func=4 slot=10 pre=owned post=owned
proof 37 rule=func_region_effect func=5 slot=0 pre=leased post=leased
proof 38 rule=func_region_effect func=5 slot=2 pre=owned post=owned
proof 39 rule=func_region_effect func=5 slot=3 pre=owned post=owned
proof 40 rule=func_region_effect func=5 slot=4 pre=owned post=owned
proof 41 rule=func_region_effect func=5 slot=5 pre=owned post=owned
proof 42 rule=func_region_effect func=5 slot=6 pre=owned post=owned
proof 43 rule=func_region_effect func=5 slot=7 pre=owned post=owned
proof 44 rule=func_region_effect func=5 slot=8 pre=owned post=owned
proof 45 rule=func_region_effect func=5 slot=9 pre=owned post=owned
proof 46 rule=func_region_effect func=5 slot=10 pre=owned post=owned
proof 47 rule=func_region_effect func=6 slot=0 pre=leased post=leased
proof 48 rule=func_region_effect func=6 slot=2 pre=owned post=owned
proof 49 rule=func_region_effect func=6 slot=3 pre=owned post=owned
proof 50 rule=func_region_effect func=6 slot=4 pre=owned post=owned
proof 51 rule=func_region_effect func=6 slot=5 pre=owned post=owned
proof 52 rule=func_region_effect func=6 slot=6 pre=owned post=owned
proof 53 rule=func_region_effect func=7 slot=0 pre=leased post=leased
proof 54 rule=func_region_effect func=7 slot=2 pre=owned post=owned
proof 55 rule=func_region_effect func=7 slot=3 pre=owned post=owned
proof 56 rule=func_region_effect func=7 slot=4 pre=owned post=owned
proof 57 rule=func_region_effect func=7 slot=5 pre=owned post=owned
proof 58 rule=func_region_effect func=7 slot=6 pre=owned post=owned
proof 59 rule=func_region_effect func=8 slot=2 pre=owned post=owned
proof 60 rule=func_region_effect func=9 slot=2 pre=owned post=owned
proof 61 rule=func_region_effect func=10 slot=0 pre=owned post=owned
proof 62 rule=func_region_effect func=10 slot=1 pre=owned post=owned
proof 63 rule=func_region_effect func=10 slot=2 pre=owned post=owned
proof 64 rule=func_region_effect func=10 slot=3 pre=owned post=owned
proof 65 rule=func_region_effect func=10 slot=4 pre=owned post=owned
proof 66 rule=func_region_effect func=10 slot=5 pre=owned post=owned
proof 67 rule=func_region_effect func=10 slot=6 pre=owned post=owned
proof 68 rule=func_region_effect func=10 slot=7 pre=owned post=owned
proof 69 rule=func_region_effect func=10 slot=8 pre=owned post=owned
proof 70 rule=func_region_effect func=10 slot=9 pre=owned post=owned
proof 71 rule=func_region_effect func=10 slot=10 pre=owned post=owned
node $f10 = Func i64x1 a=10 b=0
# Input buffer layout:
#   [0..20):   '-9223372036854775808' (20 bytes, i64::MIN)
#   [20..24):  'abcd'
#   [24..28):  'abcd'
#   [28..30):  'bc'
#   [30..94):  32 distinct 2-byte keys 'k0','k1',...'kV' (offset 30 + i*2, len 2)
node $c8 = ConstI64 i64x1 const_i64=8
node $init = Call i64x1 a=4 b=$c8 c=none
node $ls_in = RegionLease u8x1 a=0 b=0
# 1) str::parse_i64(0, 20) == i64::MIN, fmt_i64(i64::MIN) writes '-9223372036854775808'
node $c0 = ConstI64 i64x1 const_i64=0
node $c20 = ConstI64 i64x1 const_i64=20
node $minv = Call i64x1 a=2 b=$c0 c=$c20
node $fmin = Call i64x1 a=3 b=$minv c=none
# 2) str::eq_span(20, 24, 4) == 1, str::find_substr(20, 4, 28, 2) == 1
node $c24 = ConstI64 i64x1 const_i64=24
node $c4 = ConstI64 i64x1 const_i64=4
node $ca_eq = CallArg i64x1 a=$c4 b=none
node $eq_ok = Call i64x1 a=0 b=$c20 c=$c24
node $c28 = ConstI64 i64x1 const_i64=28
node $c2 = ConstI64 i64x1 const_i64=2
node $ca_find = CallArg i64x1 a=$c28 b=$c2
node $find_ok = Call i64x1 a=1 b=$c20 c=$c4
# 3) Insert 24 keys starting from cap=8 (forces multiple collisions & rehash via RegionDrop+RegionMove!)
node $ls_i = LocalSet i64x1 a=0 b=$c0
node $blk1 = Block i64x1 a=0 b=0
node $lp1 = Loop i64x1 proof=1 a=0 b=0
node $i = LocalGet i64x1 a=0 b=0
node $c24b = ConstI64 i64x1 const_i64=24
node $lt1 = CmpLt maskx1 a=$i b=$c24b
node $nlt1 = MaskNot maskx1 a=$lt1 b=0
node $bri1 = BrIf i64x1 a=$nlt1 b=1
node $c30 = ConstI64 i64x1 const_i64=30
node $off_i = Add i64x1 a=$c30 b=$i
node $off_2 = Add i64x1 a=$off_i b=$i
node $c10 = ConstI64 i64x1 const_i64=10
node $val_i = Mul i64x1 a=$i b=$c10
node $c2b = ConstI64 i64x1 const_i64=2
node $ca_set = CallArg i64x1 a=$val_i b=none
node $sr = Call i64x1 a=5 b=$off_2 c=$c2b
node $c1 = ConstI64 i64x1 const_i64=1
node $inext = Add i64x1 a=$i b=$c1
node $ls_inext = LocalSet i64x1 a=0 b=$inext
node $br1 = Br i64x1 a=0 b=0
node $elp1 = End i64x1 a=0 b=0
node $eblk1 = End i64x1 a=0 b=0
# 4) Delete even keys 0,2,4,...,22 (creating 12 tombstones across probe chains!)
node $ls_i2 = LocalSet i64x1 a=0 b=$c0
node $blk2 = Block i64x1 a=0 b=0
node $lp2 = Loop i64x1 proof=1 a=0 b=0
node $i2 = LocalGet i64x1 a=0 b=0
node $c24c = ConstI64 i64x1 const_i64=24
node $lt2 = CmpLt maskx1 a=$i2 b=$c24c
node $nlt2 = MaskNot maskx1 a=$lt2 b=0
node $bri2 = BrIf i64x1 a=$nlt2 b=1
node $c30b = ConstI64 i64x1 const_i64=30
node $off2 = Add i64x1 a=$c30b b=$i2
node $off2b = Add i64x1 a=$off2 b=$i2
node $c2c = ConstI64 i64x1 const_i64=2
node $dr = Call i64x1 a=7 b=$off2b c=$c2c
node $inext2 = Add i64x1 a=$i2 b=$c2c
node $ls_inext2 = LocalSet i64x1 a=0 b=$inext2
node $br2 = Br i64x1 a=0 b=0
node $elp2 = End i64x1 a=0 b=0
node $eblk2 = End i64x1 a=0 b=0
# 5) Re-insert key 0 with i64::MIN (reclaims tombstone, tests sentinel value with last_found == 1)
node $c30c = ConstI64 i64x1 const_i64=30
node $ca_min = CallArg i64x1 a=$minv b=none
node $sr_min = Call i64x1 a=5 b=$c30c c=$c2
node $g_min = Call i64x1 a=6 b=$c30c c=$c2
node $lf_min = Call i64x1 a=9 b=none c=none
# 6) Lookup deleted key 2 (offset 34): last_found == 0
node $c34 = ConstI64 i64x1 const_i64=34
node $g_miss = Call i64x1 a=6 b=$c34 c=$c2
node $lf_miss = Call i64x1 a=9 b=none c=none
# 7) Lookup surviving odd keys 1..23 (stepping past tombstones) and sum values: sum(10*(2k+1), k=0..11) = 1440
node $c1b = ConstI64 i64x1 const_i64=1
node $ls_i3 = LocalSet i64x1 a=0 b=$c1b
node $ls_sum = LocalSet i64x1 a=1 b=$c0
node $blk3 = Block i64x1 a=0 b=0
node $lp3 = Loop i64x1 proof=1 a=0 b=0
node $i3 = LocalGet i64x1 a=0 b=0
node $c24d = ConstI64 i64x1 const_i64=24
node $lt3 = CmpLt maskx1 a=$i3 b=$c24d
node $nlt3 = MaskNot maskx1 a=$lt3 b=0
node $bri3 = BrIf i64x1 a=$nlt3 b=1
node $c30d = ConstI64 i64x1 const_i64=30
node $off3 = Add i64x1 a=$c30d b=$i3
node $off3b = Add i64x1 a=$off3 b=$i3
node $c2d = ConstI64 i64x1 const_i64=2
node $gv = Call i64x1 a=6 b=$off3b c=$c2d
node $lf = Call i64x1 a=9 b=none c=none
node $cur_s = LocalGet i64x1 a=1 b=0
node $ns1 = Add i64x1 a=$cur_s b=$gv
node $ns2 = Add i64x1 a=$ns1 b=$lf
node $ls_sum2 = LocalSet i64x1 a=1 b=$ns2
node $inext3 = Add i64x1 a=$i3 b=$c2d
node $ls_inext3 = LocalSet i64x1 a=0 b=$inext3
node $br3 = Br i64x1 a=0 b=0
node $elp3 = End i64x1 a=0 b=0
node $eblk3 = End i64x1 a=0 b=0
node $el_in = RegionEndLease u8x1 a=0 b=0
# Final count should be 13 (12 odd keys + re-inserted key 0)
node $cnt = Call i64x1 a=8 b=none c=none
node $min_eq = CmpEq maskx1 a=$g_min b=$minv
node $min_ok = Cast i64x1 a=$min_eq b=0
node $odd_sum = LocalGet i64x1 a=1 b=0
# Expected total = eq_ok(1) + find_ok(1) + lf_min(1) + lf_miss(0) + min_ok(1) + odd_sum(1440+12=1452) + cnt(13) = 1469
node $t1 = Add i64x1 a=$eq_ok b=$find_ok
node $t2 = Add i64x1 a=$t1 b=$lf_min
node $t3 = Add i64x1 a=$t2 b=$lf_miss
node $t4 = Add i64x1 a=$t3 b=$min_ok
node $t5 = Add i64x1 a=$t4 b=$odd_sum
node $tot = Add i64x1 a=$t5 b=$cnt
node $e10 = End i64x1 a=$tot b=1
output $f10
";
    let map_str_harness =
        assemble_rls(map_str_harness_rls).map_err(|e| format!("map_str harness: {e}"))?;
    let map_str_link = link_modules(
        &[mod_str, mod_map, map_str_harness],
        &LinkOptions::default(),
    )
    .map_err(|e| format!("link map_str harness: {e}"))?;
    assert_eq!(map_str_link.capability_mask, CAP_NONE);
    let ms_cost = map_str_link.summary.static_cost.ok_or_else(|| {
        "expected StaticCostSummary for CAP_NONE String+HashMap harness".to_owned()
    })?;
    assert!(ms_cost.worst_case_steps > 0);
    assert!(ms_cost.max_call_depth >= 2);
    assert!(ms_cost.max_stack_units > 0);

    let mut ms_input = Vec::new();
    ms_input.extend_from_slice(b"-9223372036854775808abcdabcdbc");
    for i in 0..32u8 {
        ms_input.push(b'k');
        ms_input.push(b'0' + i);
    }
    let (ms_exe, _) = compile_rul_exe(&map_str_link.program, "stdlib_map_str")?;
    let (ms_ret, ms_out) =
        run_tier0_eq_aot_bytes(&map_str_link.program, &ms_exe, &ms_input, "stdlib_map_str")?;
    let _ = fs::remove_file(&ms_exe);
    assert_eq!(ms_ret, 1469);
    assert_eq!(ms_out, b"-9223372036854775808");

    Ok(())
}

// =========================================================================
// Helper: Link the 3 v0.6 Exit Programs
// =========================================================================
fn link_exit_json_parser() -> Result<RulProgram, String> {
    let m_str = load_rls("stdlib/string.rls")?;
    let m_vec = load_rls("stdlib/vec.rls")?;
    let m_app = load_rls("perf/v0_6_exit/json_parser.rls")?;
    let rep = link_modules(&[m_str, m_vec, m_app], &LinkOptions::default())
        .map_err(|e| format!("link json_parser: {e}"))?;
    assert_eq!(rep.capability_mask, CAP_DIVERGE);
    Ok(rep.program)
}

fn link_exit_kv_store() -> Result<RulProgram, String> {
    let m_map = load_rls("stdlib/hashmap.rls")?;
    let m_str = load_rls("stdlib/string.rls")?;
    let m_app = load_rls("perf/v0_6_exit/kv_store.rls")?;
    let rep = link_modules(&[m_map, m_str, m_app], &LinkOptions::default())
        .map_err(|e| format!("link kv_store: {e}"))?;
    assert_eq!(rep.capability_mask, CAP_DIVERGE);
    Ok(rep.program)
}

fn link_exit_wc_grep() -> Result<RulProgram, String> {
    let m_str = load_rls("stdlib/string.rls")?;
    let m_app = load_rls("perf/v0_6_exit/wc_grep.rls")?;
    let rep = link_modules(&[m_str, m_app], &LinkOptions::default())
        .map_err(|e| format!("link wc_grep: {e}"))?;
    assert_eq!(rep.capability_mask, CAP_DIVERGE);
    Ok(rep.program)
}

// =========================================================================
// AC-2: Linked v0.6 Exit Programs Differential Equivalence
//       (`Tier 0 == ELF64 AOT == rustc -O` on valid & adversarial inputs)
// =========================================================================
#[test]
fn test_v06_exit_programs_linked_tier0_vs_aot() -> Result<(), String> {
    // ---- 1. json_parser ----
    let json_prog = link_exit_json_parser()?;
    verify_program(&json_prog).map_err(|e| format!("verify linked json_parser: {e}"))?;
    let json_bytes = json_prog.to_rl_bytes();
    assert_eq!(
        RulProgram::from_rl_bytes(&json_bytes).map_err(|e| e.to_string())?,
        json_prog
    );
    let (json_rul_exe, _) = compile_rul_exe(&json_prog, "exit_json_rul")?;
    let json_rs_exe = compile_rust_ref("perf/v0_6_exit/json_parser.rs", "exit_json_rs")?;

    let depth_128 = format!("{}{}", "[".repeat(128), "]".repeat(128));
    let depth_129 = format!("{}{}", "[".repeat(129), "]".repeat(129));
    let mut bad_utf8_json = b"{\"k\":\"".to_vec();
    bad_utf8_json.extend_from_slice(&[0xED, 0xA0, 0x80]); // UTF-16 surrogate
    bad_utf8_json.extend_from_slice(b"\"}");
    let mut bad_ctrl_json = b"\"hi".to_vec();
    bad_ctrl_json.push(0x01);
    bad_ctrl_json.push(b'"');

    let json_cases: Vec<(&str, Vec<u8>, &str)> = vec![
        (
            "valid_complex",
            " { \"name\" : \"café 🚀 \\\"ok\\\" \\u0041\" , \"nums\" : [ 0 , -42 , 3.14e-2 , 1E+5 ] , \"flags\" : [ true , false , null ] , \"empty\" : {} } \n"
                .as_bytes()
                .to_vec(),
            "OK obj=2 arr=2 str=5 num=4 bool=2 null=1 max_depth=2\n",
        ),
        ("empty_array", b"[]".to_vec(), "OK obj=0 arr=1 str=0 num=0 bool=0 null=0 max_depth=1\n"),
        ("depth_128_ok", depth_128.into_bytes(), "OK obj=0 arr=128 str=0 num=0 bool=0 null=0 max_depth=128\n"),
        ("depth_129_err", depth_129.into_bytes(), "ERR depth\n"),
        ("bad_utf8_surrogate", bad_utf8_json, "ERR utf8\n"),
        ("bad_trailing_comma", b"[1, 2, ]".to_vec(), "ERR syntax\n"),
        ("bad_leading_zero", b"01".to_vec(), "ERR syntax\n"),
        ("bad_escape", b"\"\\x\"".to_vec(), "ERR syntax\n"),
        ("bad_ctrl_char", bad_ctrl_json, "ERR syntax\n"),
        ("bad_trailing_garbage", b"true 1".to_vec(), "ERR syntax\n"),
        ("empty_input", b"   ".to_vec(), "ERR syntax\n"),
    ];
    for (name, input, expected) in &json_cases {
        let (_, aot_out) = run_tier0_eq_aot_bytes(&json_prog, &json_rul_exe, input, name)?;
        let rs_out = run_exe_with_stdin(&json_rs_exe, input)?;
        assert_eq!(aot_out, rs_out, "json_parser {name}: Rul AOT != rustc -O");
        assert_eq!(
            aot_out,
            expected.as_bytes(),
            "json_parser {name}: unexpected output"
        );
    }
    let _ = fs::remove_file(&json_rul_exe);
    let _ = fs::remove_file(&json_rs_exe);

    // ---- 2. kv_store ----
    let kv_prog = link_exit_kv_store()?;
    verify_program(&kv_prog).map_err(|e| format!("verify linked kv_store: {e}"))?;
    let kv_bytes = kv_prog.to_rl_bytes();
    assert_eq!(
        RulProgram::from_rl_bytes(&kv_bytes).map_err(|e| e.to_string())?,
        kv_prog
    );
    let (kv_rul_exe, _) = compile_rul_exe(&kv_prog, "exit_kv_rul")?;
    let kv_rs_exe = compile_rust_ref("perf/v0_6_exit/kv_store.rs", "exit_kv_rs")?;

    let kv_stream = concat!(
        "COUNT\n",
        "GET missing\n",
        "DEL missing\n",
        "SET alpha 42\r\n",
        "\n",
        "SET beta -9223372036854775808\n",
        "SET gamma 0\n",
        "GET alpha\n",
        "GET beta\n",
        "GET gamma\n",
        "SET alpha 99\n",
        "GET alpha\n",
        "COUNT\n",
        "DEL beta\n",
        "DEL beta\n",
        "GET beta\n",
        "COUNT\n",
        "SET beta 123456789\n",
        "GET beta\n",
        "COUNT\n",
    );
    let kv_stream_expected = concat!(
        "0\n",
        "MISS\n",
        "MISS\n",
        "OK\n",
        "OK\n",
        "OK\n",
        "42\n",
        "-9223372036854775808\n",
        "0\n",
        "OK\n",
        "99\n",
        "3\n",
        "OK\n",
        "MISS\n",
        "MISS\n",
        "2\n",
        "OK\n",
        "123456789\n",
        "3\n",
    );
    let parse_33_digits = format!("SET a 1\nGET a\nSET b {}\nGET b\nCOUNT\n", "1".repeat(33));
    let parse_41_zeros = format!("SET z {}5\nGET z\n", "0".repeat(41));
    let parse_neg_35_nines = format!("SET z -{}\nGET z\n", "9".repeat(35));
    let parse_neg_36_nines = format!("SET z -{}\nGET z\n", "9".repeat(36));

    // >1,000,000-byte key exercising every chunked loop in map::{set,get,del}_span
    // (FNV hash, key-copy, key-compare update, get hit/miss, del hit/miss).
    let long_key_a = "k".repeat(1_000_001);
    let long_key_b = format!("{}x", "k".repeat(1_000_000));
    let kv_long_key_stream = format!(
        "SET {long_key_a} 5\nGET {long_key_a}\nSET {long_key_a} 7\nGET {long_key_a}\nGET {long_key_b}\nDEL {long_key_b}\nCOUNT\nDEL {long_key_a}\nDEL {long_key_a}\nCOUNT\n"
    );

    // Small-scale churn verifying key compaction across rehash in Tier 0 == AOT == rustc -O.
    let mut kv_churn_small = String::with_capacity(64 * 1024);
    let mut kv_churn_small_expected = String::with_capacity(16 * 1024);
    for i in 0..1_200usize {
        let _ = writeln!(kv_churn_small, "SET key_{i:06}_pad_0123456789 {i}");
        let _ = writeln!(kv_churn_small, "DEL key_{i:06}_pad_0123456789");
        kv_churn_small_expected.push_str("OK\nOK\n");
    }
    kv_churn_small.push_str("SET survivor 777\nGET survivor\nCOUNT\n");
    kv_churn_small_expected.push_str("OK\n777\n1\n");

    let kv_cases: Vec<(&str, &[u8], &str)> = vec![
        ("batch_ops", kv_stream.as_bytes(), kv_stream_expected),
        ("empty", b"", ""),
        (
            "parse_33_digits",
            parse_33_digits.as_bytes(),
            "OK\n1\nOK\n4553682020284002759\n2\n",
        ),
        ("parse_41_zeros", parse_41_zeros.as_bytes(), "OK\n5\n"),
        (
            "parse_neg_35_nines",
            parse_neg_35_nines.as_bytes(),
            "OK\n-3136633892082024447\n",
        ),
        (
            "parse_neg_36_nines",
            parse_neg_36_nines.as_bytes(),
            "OK\n5527149226598858753\n",
        ),
        (
            "kv_churn_small_tier0",
            kv_churn_small.as_bytes(),
            &kv_churn_small_expected,
        ),
    ];
    for (name, input, expected) in &kv_cases {
        let (_, aot_out) = run_tier0_eq_aot_bytes(&kv_prog, &kv_rul_exe, input, name)?;
        let rs_out = run_exe_with_stdin(&kv_rs_exe, input)?;
        assert_eq!(aot_out, rs_out, "kv_store {name}: Rul AOT != rustc -O");
        assert_eq!(
            aot_out,
            expected.as_bytes(),
            "kv_store {name}: unexpected golden output"
        );
    }

    // >1,000,000-byte key regression (Tier 0 + AOT in release; AOT enforces identical LoopBound fuel in debug).
    {
        let expected = b"OK\n5\nOK\n7\nMISS\nMISS\n1\nOK\nMISS\n0\n";
        let aot_out = if cfg!(debug_assertions) {
            run_exe_with_stdin(&kv_rul_exe, kv_long_key_stream.as_bytes())?
        } else {
            run_tier0_eq_aot_bytes(
                &kv_prog,
                &kv_rul_exe,
                kv_long_key_stream.as_bytes(),
                "kv_long_key_1000001",
            )?
            .1
        };
        let rs_out = run_exe_with_stdin(&kv_rs_exe, kv_long_key_stream.as_bytes())?;
        assert_eq!(aot_out, rs_out, "kv_store kv_long_key_1000001: AOT != rs");
        assert_eq!(aot_out, expected, "kv_store kv_long_key_1000001 golden");
    }

    // Large-scale hashmap capacity & key-arena compaction regressions (ELF64 AOT == rustc -O):
    // 1. 400k distinct 100-byte keys (~40.8 MB; exceeds prior 32 MiB key arena cap).
    {
        let mut s = String::with_capacity(400_000 * 108 + 16);
        let pad = "a".repeat(92);
        for i in 0..400_000usize {
            let _ = writeln!(s, "SET k{i:07}{pad} 1");
        }
        s.push_str("COUNT\n");
        let aot_out = run_exe_with_stdin(&kv_rul_exe, s.as_bytes())?;
        let rs_out = run_exe_with_stdin(&kv_rs_exe, s.as_bytes())?;
        assert_eq!(aot_out, rs_out, "kv_store kv_arena_400k_x100 mismatch");
        assert!(aot_out.ends_with(b"400000\n"));
    }
    // 2. 400k SET+DEL churn of 100-byte keys (~80.9 MB, max 1 live key; compacts map.keys on rehash).
    {
        let mut s = String::with_capacity(400_000 * 212 + 16);
        let pad = "b".repeat(92);
        for i in 0..400_000usize {
            let _ = writeln!(s, "SET k{i:07}{pad} 1");
            let _ = writeln!(s, "DEL k{i:07}{pad}");
        }
        s.push_str("COUNT\n");
        let aot_out = run_exe_with_stdin(&kv_rul_exe, s.as_bytes())?;
        let rs_out = run_exe_with_stdin(&kv_rs_exe, s.as_bytes())?;
        assert_eq!(aot_out, rs_out, "kv_store kv_churn_400k_x100 mismatch");
        assert!(aot_out.ends_with(b"0\n"));
    }
    // 3. Same-key churn (40,000 SET/DEL cycles of a 1,000-byte key = 40 MB dead key bytes with used_count <= 1).
    {
        let big_k = "c".repeat(1_000);
        let mut s = String::with_capacity(40_000 * 2_016 + 32);
        for i in 0..40_000usize {
            let _ = writeln!(s, "SET {big_k} {i}");
            let _ = writeln!(s, "DEL {big_k}");
        }
        let _ = writeln!(s, "SET {big_k} 99");
        let _ = writeln!(s, "GET {big_k}");
        s.push_str("COUNT\n");
        let aot_out = run_exe_with_stdin(&kv_rul_exe, s.as_bytes())?;
        let rs_out = run_exe_with_stdin(&kv_rs_exe, s.as_bytes())?;
        assert_eq!(aot_out, rs_out, "kv_store kv_same_key_churn_40mb mismatch");
        assert!(aot_out.ends_with(b"OK\n99\n1\n"));
    }
    // 4. 3.1M distinct 8-byte keys (~44.3 MB; exceeds prior 2.94M slot limit).
    {
        let mut s = String::with_capacity(3_100_000 * 15 + 16);
        for i in 0..3_100_000usize {
            let _ = writeln!(s, "SET k{i:07} 1");
        }
        s.push_str("COUNT\n");
        let aot_out = run_exe_with_stdin(&kv_rul_exe, s.as_bytes())?;
        let rs_out = run_exe_with_stdin(&kv_rs_exe, s.as_bytes())?;
        assert_eq!(aot_out, rs_out, "kv_store kv_slots_3100k mismatch");
        assert!(aot_out.ends_with(b"3100000\n"));
    }
    let _ = fs::remove_file(&kv_rul_exe);
    let _ = fs::remove_file(&kv_rs_exe);

    // ---- 3. wc_grep ----
    let wc_prog = link_exit_wc_grep()?;
    verify_program(&wc_prog).map_err(|e| format!("verify linked wc_grep: {e}"))?;
    let wc_bytes = wc_prog.to_rl_bytes();
    assert_eq!(
        RulProgram::from_rl_bytes(&wc_bytes).map_err(|e| e.to_string())?,
        wc_prog
    );
    let (wc_rul_exe, _) = compile_rul_exe(&wc_prog, "exit_wc_rul")?;
    let wc_rs_exe = compile_rust_ref("perf/v0_6_exit/wc_grep.rs", "exit_wc_rs")?;

    // 3,000,000-byte pattern exercising chunked pattern-verify loop in str::find_substr slow path.
    let long_pat = "p".repeat(3_000_000);
    let wc_long_pat = format!("{long_pat}\n{long_pat}\nnomatch\n");

    let wc_cases: Vec<(&str, &[u8], &str)> = vec![
        (
            "normal_lines",
            b"needle\nfirst line\nsecond needle here\n\tthird\x0Bword  needle\nno trailing newline needle",
            "lines=3 words=12 bytes=76 matches=3 digest=6685335052900319498\n",
        ),
        (
            "empty_pattern",
            b"\nalpha beta\ngamma\n",
            "lines=2 words=3 bytes=17 matches=2 digest=2165237401667456574\n",
        ),
        (
            "no_matches",
            b"zzzzzz\nshort\nlines\n",
            "lines=2 words=2 bytes=12 matches=0 digest=5472609002491880229\n",
        ),
        (
            "empty_text",
            b"pattern\n",
            "lines=0 words=0 bytes=0 matches=0 digest=5472609002491880229\n",
        ),
        (
            "no_newline_at_all",
            b"onlypattern",
            "lines=0 words=0 bytes=0 matches=0 digest=5472609002491880229\n",
        ),
    ];
    for (name, input, expected) in &wc_cases {
        let (_, aot_out) = run_tier0_eq_aot_bytes(&wc_prog, &wc_rul_exe, input, name)?;
        let rs_out = run_exe_with_stdin(&wc_rs_exe, input)?;
        assert_eq!(aot_out, rs_out, "wc_grep {name}: Rul AOT != rustc -O");
        assert_eq!(
            aot_out,
            expected.as_bytes(),
            "wc_grep {name}: unexpected golden output"
        );
    }
    // 3,000,000-byte pattern regression (Tier 0 + AOT in release; AOT enforces identical LoopBound fuel in debug).
    {
        let expected = b"lines=2 words=2 bytes=3000009 matches=1 digest=4520812910162906589\n";
        let aot_out = if cfg!(debug_assertions) {
            run_exe_with_stdin(&wc_rul_exe, wc_long_pat.as_bytes())?
        } else {
            run_tier0_eq_aot_bytes(
                &wc_prog,
                &wc_rul_exe,
                wc_long_pat.as_bytes(),
                "wc_long_pattern_3000000",
            )?
            .1
        };
        let rs_out = run_exe_with_stdin(&wc_rs_exe, wc_long_pat.as_bytes())?;
        assert_eq!(
            aot_out, rs_out,
            "wc_grep wc_long_pattern_3000000: AOT != rs"
        );
        assert_eq!(aot_out, expected, "wc_grep wc_long_pattern_3000000 golden");
    }
    let _ = fs::remove_file(&wc_rul_exe);
    let _ = fs::remove_file(&wc_rs_exe);

    Ok(())
}

// =========================================================================
// AC-3 (`#[ignore]`): Release Performance Gate (`<= 2.0x` vs `rustc -O`, `>= 100 ms`)
// =========================================================================
fn generate_json_benchmark_input(records: usize) -> Vec<u8> {
    let mut out = String::with_capacity(records * 150 + 64);
    out.push_str("{\"dataset\":\"v0.6-bench\",\"records\":[\n");
    for i in 0..records {
        if i > 0 {
            out.push_str(",\n");
        }
        let active = if i % 2 == 0 { "true" } else { "false" };
        let opt = if i % 5 == 0 {
            "null"
        } else {
            "\"ok\\t\\u0041\""
        };
        let _ = write!(
            out,
            "{{\"id\":{i},\"name\":\"item_{i}_café\",\"score\":-{}.25e+1,\"active\":{active},\"tags\":[\"a\",\"b_{}\",\"c\"],\"meta\":{opt}}}",
            (i * 17) % 1000,
            i % 97
        );
    }
    out.push_str("\n]}\n");
    out.into_bytes()
}

fn generate_kv_benchmark_input(ops: usize) -> Vec<u8> {
    let mut out = String::with_capacity(ops * 24 + 64);
    let mut state: u64 = 0x1234_5678_9ABC_DEF1;
    let key_space: u64 = 16_384;
    for i in 0..ops {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let k = (state >> 16) % key_space;
        let op = i % 10;
        if op < 4 {
            #[expect(
                clippy::cast_possible_wrap,
                reason = "deterministic benchmark value generation"
            )]
            let val = ((state >> 24) as i64) - 500_000;
            let _ = writeln!(out, "SET k_{k:05} {val}");
        } else if op < 8 {
            let _ = writeln!(out, "GET k_{k:05}");
        } else if op == 8 {
            let _ = writeln!(out, "DEL k_{k:05}");
        } else if i % 10_000 == 9 {
            out.push_str("COUNT\n");
        } else {
            let _ = writeln!(out, "GET k_{k:05}");
        }
    }
    out.push_str("COUNT\n");
    out.into_bytes()
}

fn generate_wc_grep_benchmark_input(lines: usize) -> Vec<u8> {
    let mut out = String::with_capacity(lines * 76 + 32);
    out.push_str("alpha_target_42\n");
    for i in 0..lines {
        if i % 17 == 0 {
            let _ = writeln!(
                out,
                "row {i:07} status ok payload alpha_target_42 checksum {} done",
                (i * 31) % 9973
            );
        } else {
            let _ = writeln!(
                out,
                "row {i:07} status ok payload beta_regular_99 checksum {} done",
                (i * 37) % 9973
            );
        }
    }
    out.into_bytes()
}

#[test]
#[ignore = "AC-3 release performance gate against rustc -O; run with cargo test --release test_v06_performance_gate -- --ignored --nocapture --test-threads=1"]
fn test_v06_performance_gate() -> Result<(), String> {
    let rustc_check = Command::new("rustc")
        .arg("--version")
        .output()
        .map_err(|e| format!("rustc is required for AC-3 performance gate: {e}"))?;
    if !rustc_check.status.success() {
        return Err("rustc --version failed".to_owned());
    }

    let tmp_dir = unique_tmp_path("perf_gate", "dir");
    fs::create_dir_all(&tmp_dir).map_err(|e| format!("create tmp_dir: {e}"))?;

    let json_prog = link_exit_json_parser()?;
    let kv_prog = link_exit_kv_store()?;
    let wc_prog = link_exit_wc_grep()?;

    // Runtime-opaque inputs sized so `rustc -O` runs comfortably >= 100 ms on each benchmark
    // (including at peak CPU turbo clocks), within the declared 128 MiB `std.io.in` cap.
    let json_in = generate_json_benchmark_input(900_000);
    let kv_in = generate_kv_benchmark_input(2_000_000);
    let wc_in = generate_wc_grep_benchmark_input(1_400_000);

    let cases: [(&str, &str, &RulProgram, &[u8]); 3] = [
        (
            "json_parser",
            "perf/v0_6_exit/json_parser.rs",
            &json_prog,
            &json_in,
        ),
        ("kv_store", "perf/v0_6_exit/kv_store.rs", &kv_prog, &kv_in),
        ("wc_grep", "perf/v0_6_exit/wc_grep.rs", &wc_prog, &wc_in),
    ];

    let mut failures: Vec<String> = Vec::new();
    println!(
        "\n{:<14} {:>10} {:>10} {:>8} {:>10} {:>10} {:>10}",
        "Benchmark", "rustc -O", "Rul AOT", "Ratio", "Input MB", "ELF (B)", "Codegen"
    );
    println!("{}", "-".repeat(78));

    for (name, rs_rel, prog, input_bytes) in cases {
        let rul_exe = tmp_dir.join(format!("{name}_rul"));
        let ref_exe = tmp_dir.join(format!("{name}_rs"));
        let in_path = tmp_dir.join(format!("{name}.in"));

        let report = compile_rl_to_executable(prog, &rul_exe)
            .map_err(|e| format!("{name} aot compile: {e}"))?;
        if let Ok(f) = File::open(&rul_exe) {
            let _ = f.sync_all();
        }

        let rustc_out = Command::new("rustc")
            .arg("-O")
            .arg(repo_root().join(rs_rel))
            .arg("-o")
            .arg(&ref_exe)
            .output()
            .map_err(|e| format!("{name} rustc invoke: {e}"))?;
        if !rustc_out.status.success() {
            let _ = fs::remove_dir_all(&tmp_dir);
            return Err(format!(
                "{name} rustc -O failed: {}",
                String::from_utf8_lossy(&rustc_out.stderr)
            ));
        }

        {
            let mut f =
                File::create(&in_path).map_err(|e| format!("create {}: {e}", in_path.display()))?;
            f.write_all(input_bytes)
                .map_err(|e| format!("write {}: {e}", in_path.display()))?;
            f.sync_all()
                .map_err(|e| format!("sync {}: {e}", in_path.display()))?;
        }

        let mut ref_min = Duration::MAX;
        let mut rul_min = Duration::MAX;
        let mut expected_stdout: Option<Vec<u8>> = None;

        for iter in 0..5usize {
            let (d_ref, out_ref) = run_exe_timed_from_file(&ref_exe, &in_path)?;
            if d_ref < ref_min {
                ref_min = d_ref;
            }
            if let Some(ref exp) = expected_stdout {
                if exp != &out_ref {
                    failures.push(format!(
                        "{name}: rustc -O stdout non-deterministic on iter {iter}"
                    ));
                }
            } else {
                expected_stdout = Some(out_ref.clone());
            }

            let (d_rul, out_rul) = run_exe_timed_from_file(&rul_exe, &in_path)?;
            if d_rul < rul_min {
                rul_min = d_rul;
            }
            if out_rul != out_ref {
                failures.push(format!(
                    "{name}: stdout mismatch on iter {iter} (rul={}B vs ref={}B)",
                    out_rul.len(),
                    out_ref.len()
                ));
            }
        }

        let ref_ms = ref_min.as_secs_f64() * 1000.0;
        let rul_ms = rul_min.as_secs_f64() * 1000.0;
        let ratio = rul_ms / ref_ms;
        let in_mb = (input_bytes.len() as f64) / (1024.0 * 1024.0);
        let cg_us = (report.codegen_ns as f64) / 1000.0;

        println!(
            "{name:<14} {ref_ms:>8.2}ms {rul_ms:>8.2}ms {ratio:>7.2}x {in_mb:>8.2}MB {:>10} {cg_us:>8.2}us",
            report.elf_bytes
        );

        if ref_min < Duration::from_millis(100) {
            failures.push(format!(
                "{name}: reference min wall time {ref_ms:.2}ms is below 100ms minimum"
            ));
        }
        if ratio > 2.0 {
            failures.push(format!(
                "{name}: ratio {ratio:.3}x exceeds 2.0x gate (rul={rul_ms:.2}ms, ref={ref_ms:.2}ms)"
            ));
        }

        let _ = fs::remove_file(&in_path);
        let _ = fs::remove_file(&rul_exe);
        let _ = fs::remove_file(&ref_exe);
    }

    let _ = fs::remove_dir_all(&tmp_dir);

    if !failures.is_empty() {
        return Err(format!(
            "AC-3 v0.6 performance gate failed:\n  - {}",
            failures.join("\n  - ")
        ));
    }
    Ok(())
}

// =========================================================================
// AC-4 (`#[ignore]`): Release Linear O(N) Verify + Link + Codegen Gate
// =========================================================================
fn build_synthetic_multi_module(total_nodes: usize) -> Result<Vec<RulProgram>, String> {
    let lib_nodes = total_nodes / 2;
    let app_nodes = total_nodes.saturating_sub(lib_nodes);

    // Module 0: exports `syn::work(x: i64) -> i64` with ~lib_nodes ALU nodes
    let mut s0 = String::with_capacity(lib_nodes * 56);
    s0.push_str("cap 0x0\n");
    s0.push_str("export func syn::work 0\n");
    s0.push_str("proof 1 rule=func_sig target=0 lo=1 hi=i64 witness=i64\n");
    s0.push_str("proof 2 rule=func_region_effect func=0 slot=none pre=owned post=owned\n");
    s0.push_str("node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0\n");
    s0.push_str("node $x = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0\n");
    s0.push_str("node $c1 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1\n");
    let ops0 = lib_nodes.saturating_sub(4);
    let mut prev = "$x".to_owned();
    for k in 0..ops0 {
        let lbl = format!("$a{k}");
        let op = match k % 3 {
            0 => "Add",
            1 => "BitXor",
            _ => "Sub",
        };
        let _ = writeln!(
            s0,
            "node {lbl} = {op} i64x1 fg=0 pol=l1 proof=0 tok=pure a={prev} b=$c1"
        );
        prev = lbl;
    }
    let _ = writeln!(
        s0,
        "node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a={prev} b=1"
    );
    s0.push_str("output $f0\n");

    // Module 1: imports `syn::work`, calls it, and runs ~app_nodes ALU nodes
    let mut s1 = String::with_capacity(app_nodes * 56);
    s1.push_str("cap 0x0\n");
    s1.push_str("import func syn::work 0\n");
    s1.push_str("proof 1 rule=func_sig target=0 lo=1 hi=i64 witness=i64\n");
    s1.push_str("proof 2 rule=func_sig target=1 lo=0 hi=0 witness=i64\n");
    s1.push_str("proof 3 rule=func_region_effect func=0 slot=none pre=owned post=owned\n");
    s1.push_str("proof 4 rule=func_region_effect func=1 slot=none pre=owned post=owned\n");
    s1.push_str("node $f1 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0\n");
    s1.push_str("node $seed = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=7\n");
    s1.push_str("node $c3 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=3\n");
    s1.push_str("node $res0 = Call i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$seed c=none\n");
    let ops1 = app_nodes.saturating_sub(5);
    let mut prev1 = "$res0".to_owned();
    for k in 0..ops1 {
        let lbl = format!("$b{k}");
        let op = match k % 3 {
            0 => "Add",
            1 => "Sub",
            _ => "BitXor",
        };
        let _ = writeln!(
            s1,
            "node {lbl} = {op} i64x1 fg=0 pol=l1 proof=0 tok=pure a={prev1} b=$c3"
        );
        prev1 = lbl;
    }
    let _ = writeln!(
        s1,
        "node $e1 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a={prev1} b=1"
    );
    s1.push_str("output $f1\n");

    let m0 = assemble_rls(&s0).map_err(|e| format!("assemble syn m0 ({total_nodes}): {e}"))?;
    let m1 = assemble_rls(&s1).map_err(|e| format!("assemble syn m1 ({total_nodes}): {e}"))?;
    Ok(vec![m0, m1])
}

#[test]
#[ignore = "AC-4 release linearity gate; run with cargo test --release test_v06_linear_verify_link_codegen -- --ignored --nocapture --test-threads=1"]
fn test_v06_linear_verify_link_codegen() -> Result<(), String> {
    let m_str = load_rls("stdlib/string.rls")?;
    let m_vec = load_rls("stdlib/vec.rls")?;
    let m_map = load_rls("stdlib/hashmap.rls")?;
    let m_json = load_rls("perf/v0_6_exit/json_parser.rls")?;
    let m_kv = load_rls("perf/v0_6_exit/kv_store.rls")?;
    let m_wc = load_rls("perf/v0_6_exit/wc_grep.rls")?;

    let syn_6k = build_synthetic_multi_module(6_000)?;
    let syn_60k = build_synthetic_multi_module(60_000)?;

    let cases: [(&str, Vec<RulProgram>); 5] = [
        ("json_parser", vec![m_str.clone(), m_vec, m_json]),
        ("kv_store", vec![m_map, m_str.clone(), m_kv]),
        ("wc_grep", vec![m_str, m_wc]),
        ("stress_6k", syn_6k),
        ("stress_60k", syn_60k),
    ];

    println!(
        "\n{:<14} {:>8} {:>12} {:>12} {:>12} {:>10}",
        "Program", "Nodes", "Verify (ns)", "Link (ns)", "Codegen (ns)", "cg ns/node"
    );
    println!("{}", "-".repeat(74));

    let mut v_6k: u64 = 0;
    let mut l_6k: u64 = 0;
    let mut c_6k: u64 = 0;
    let mut v_60k: u64 = 0;
    let mut l_60k: u64 = 0;
    let mut c_60k: u64 = 0;

    for (name, mods) in &cases {
        let mut best_verify = u64::MAX;
        let mut best_link = u64::MAX;
        let mut best_cg = u64::MAX;
        let mut linked_prog: Option<RulProgram> = None;

        for _ in 0..5 {
            let tv0 = Instant::now();
            for m in mods {
                verify_module(m).map_err(|e| format!("{name} verify_module: {e}"))?;
            }
            let v_ns = u64::try_from(tv0.elapsed().as_nanos()).unwrap_or(u64::MAX);
            if v_ns < best_verify {
                best_verify = v_ns;
            }

            let tl0 = Instant::now();
            let rep = link_modules(mods, &LinkOptions::default())
                .map_err(|e| format!("{name} link: {e}"))?;
            let l_ns = u64::try_from(tl0.elapsed().as_nanos()).unwrap_or(u64::MAX);
            if l_ns < best_link {
                best_link = l_ns;
            }
            linked_prog = Some(rep.program);
        }

        let prog = linked_prog.ok_or_else(|| format!("{name}: missing linked program"))?;
        for _ in 0..15 {
            let (_, cg_rep) =
                compile_to_elf64_bytes(&prog).map_err(|e| format!("{name} codegen failed: {e}"))?;
            if cg_rep.codegen_ns < best_cg {
                best_cg = cg_rep.codegen_ns;
            }
        }

        let nodes = prog.nodes.len() as u64;
        let cg_ns_per_node = (best_cg as f64) / (nodes.max(1) as f64);
        println!(
            "{name:<14} {nodes:>8} {best_verify:>12} {best_link:>12} {best_cg:>12} {cg_ns_per_node:>9.2}"
        );

        if *name == "stress_6k" {
            v_6k = best_verify;
            l_6k = best_link;
            c_6k = best_cg;
        } else if *name == "stress_60k" {
            v_60k = best_verify;
            l_60k = best_link;
            c_60k = best_cg;
            assert!(
                cg_ns_per_node <= 50.0,
                "stress_60k AOT codegen {cg_ns_per_node:.2} ns/node exceeded 50.0 ns/node limit"
            );
        }
    }

    let rv = (v_60k as f64) / (v_6k.max(1) as f64);
    let rl = (l_60k as f64) / (l_6k.max(1) as f64);
    let rc = (c_60k as f64) / (c_6k.max(1) as f64);
    println!("\n60k / 6k scaling ratios (limit <= 15.0x):");
    println!("  verify_module : {rv:.2}x ({v_60k} ns / {v_6k} ns)");
    println!("  link_modules  : {rl:.2}x ({l_60k} ns / {l_6k} ns)");
    println!("  AOT codegen   : {rc:.2}x ({c_60k} ns / {c_6k} ns)");

    assert!(
        rv <= 15.0,
        "verify_module 60k/6k ratio {rv:.2}x exceeded 15.0x"
    );
    assert!(
        rl <= 15.0,
        "link_modules 60k/6k ratio {rl:.2}x exceeded 15.0x"
    );
    assert!(
        rc <= 15.0,
        "AOT codegen 60k/6k ratio {rc:.2}x exceeded 15.0x"
    );

    Ok(())
}
