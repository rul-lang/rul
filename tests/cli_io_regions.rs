//! End-to-end CLI tests for `rul run` with declarative input/output byte regions.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::indexing_slicing)]

use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn test_cli_rul_run_io_regions_stdin_stdout_forwarding() -> Result<(), String> {
    let rul_bin = env!("CARGO_BIN_EXE_rul");
    let work_dir =
        std::env::temp_dir().join(format!("rul_cli_io_regions_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work_dir);
    std::fs::create_dir_all(&work_dir).map_err(|e| e.to_string())?;

    // Program that copies stdin bytes XOR 0x01 into the declared output region (max_len = 64).
    let prog_rls = "\
cap DIVERGE
proof 1 rule=region_decl target=0 lo=0 hi=64 witness=u8:input
proof 2 rule=region_decl target=1 lo=0 hi=64 witness=u8:output
node $c0 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
node $c1 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1
node $u1 = Cast u8x1 fg=0 pol=l1 proof=0 tok=pure a=$c1 b=0
node $n = RegionLen i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $g = RegionGrow i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=$n
node $si = LocalSet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$c0
node $blk = Block i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $lp = Loop i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $i = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $done = CmpEq maskx1 fg=0 pol=l1 proof=0 tok=pure a=$i b=$n
node $br = BrIf maskx1 fg=0 pol=l1 proof=0 tok=pure a=$done b=1
node $b = RegionLoad u8x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$i
node $xb = BitXor u8x1 fg=0 pol=l1 proof=0 tok=pure a=$b b=$u1
node $st = RegionStore u8x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=$i c=$xb
node $i1 = Add i64x1 fg=0 pol=l1 proof=0 tok=pure a=$i b=$c1
node $ui = LocalSet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$i1
node $back = Br i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $el = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $eb = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
output $g
";
    let rls_path = work_dir.join("echo_xor.rls");
    std::fs::write(&rls_path, prog_rls).map_err(|e| e.to_string())?;

    // 1. Valid payload forwarded via stdin -> exact transformed bytes on stdout (status banner on stderr).
    let payload = b"Rul-v0.5-IO";
    let expected: Vec<u8> = payload.iter().map(|&b| b ^ 1).collect();
    let mut child = Command::new(rul_bin)
        .arg("run")
        .arg(&rls_path)
        .current_dir(&work_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(payload).map_err(|e| e.to_string())?;
    }
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    assert!(
        out.status.success(),
        "rul run failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.stdout, expected);

    // 2. Oversized stdin (> 64 bytes) fails closed cleanly with zero stdout bytes.
    let oversized = vec![b'Z'; 1000];
    let mut child_over = Command::new(rul_bin)
        .arg("run")
        .arg(&rls_path)
        .current_dir(&work_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;
    if let Some(mut stdin) = child_over.stdin.take() {
        let _ = stdin.write_all(&oversized);
    }
    let out_over = child_over.wait_with_output().map_err(|e| e.to_string())?;
    assert!(
        !out_over.status.success(),
        "oversized stdin must fail closed"
    );
    assert!(
        out_over.stdout.is_empty(),
        "oversized stdin must emit zero stdout bytes, got {}",
        out_over.stdout.len()
    );

    let _ = std::fs::remove_dir_all(&work_dir);
    Ok(())
}
