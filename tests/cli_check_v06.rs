//! End-to-end CLI tests for `rul check` inline static cost summary and `--json` output (AC-4).

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![deny(clippy::indexing_slicing)]

use std::process::Command;

#[test]
fn test_cli_rul_check_static_cost_and_json() -> Result<(), String> {
    let rul_bin = env!("CARGO_BIN_EXE_rul");
    let work_dir =
        std::env::temp_dir().join(format!("rul_cli_check_v06_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work_dir);
    std::fs::create_dir_all(&work_dir).map_err(|e| e.to_string())?;

    // 1. Sandbox program (without CAP_DIVERGE) with a declared region and a function.
    let sandbox_rls = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=16 witness=i64:plain
proof 2 rule=func_sig target=0 lo=0 hi=0 witness=i64
node $f0 = Func i64x1 a=0 b=0
node $v = ConstI64 i64x1 const_i64=42
node $e0 = End i64x1 a=$v b=1
output $f0
";
    let sandbox_path = work_dir.join("sandbox.rls");
    std::fs::write(&sandbox_path, sandbox_rls).map_err(|e| e.to_string())?;

    // 1a. Inline text output preserves `[RUL CHECK OK]` and includes `Cost(...)`.
    let out_text = Command::new(rul_bin)
        .arg("check")
        .arg(&sandbox_path)
        .output()
        .map_err(|e| e.to_string())?;
    assert!(
        out_text.status.success(),
        "rul check failed: stderr={}",
        String::from_utf8_lossy(&out_text.stderr)
    );
    let stdout_text = String::from_utf8_lossy(&out_text.stdout);
    assert!(
        stdout_text.contains("[RUL CHECK OK]"),
        "missing [RUL CHECK OK]: {stdout_text}"
    );
    assert!(
        stdout_text.contains(
            "Cost(steps=1, depth=1, stack_units=75, stack_bytes=4696, region_bytes=8192)"
        ),
        "unexpected inline cost output: {stdout_text}"
    );

    // 1b. `--json` output emits a single valid JSON object with all required keys.
    let out_json = Command::new(rul_bin)
        .arg("check")
        .arg("--json")
        .arg(&sandbox_path)
        .output()
        .map_err(|e| e.to_string())?;
    assert!(
        out_json.status.success(),
        "rul check --json failed: stderr={}",
        String::from_utf8_lossy(&out_json.stderr)
    );
    let stdout_json = String::from_utf8_lossy(&out_json.stdout);
    let trimmed = stdout_json.trim();
    assert!(
        trimmed.starts_with('{') && trimmed.ends_with('}') && !trimmed.contains('\n'),
        "expected single-line JSON object, got: {stdout_json}"
    );
    for key in [
        "\"status\":\"ok\"",
        "\"file\":",
        "\"node_count\":3",
        "\"fusion_groups\":1",
        "\"output_shape\":\"Scalar\"",
        "\"output_elem\":\"I64\"",
        "\"zero_runtime_bounds_checks\":true",
        "\"verify_ns\":",
        "\"static_cost\":{\"worst_case_steps\":1,\"max_call_depth\":1,\"max_stack_units\":75,\"max_stack_bytes\":4696,\"max_region_bytes\":8192}",
    ] {
        assert!(
            trimmed.contains(key),
            "JSON output missing `{key}`: {trimmed}"
        );
    }

    // 2. CAP_DIVERGE program: inline shows `Cost=unbounded(CAP_DIVERGE)`, JSON shows `"static_cost":null`.
    let diverge_rls = "\
cap DIVERGE
node $c = ConstI64 i64x1 const_i64=7
output $c
";
    let diverge_path = work_dir.join("diverge.rls");
    std::fs::write(&diverge_path, diverge_rls).map_err(|e| e.to_string())?;

    let div_text = Command::new(rul_bin)
        .arg("check")
        .arg(&diverge_path)
        .output()
        .map_err(|e| e.to_string())?;
    assert!(div_text.status.success());
    let div_stdout = String::from_utf8_lossy(&div_text.stdout);
    assert!(
        div_stdout.contains("Cost=unbounded(CAP_DIVERGE)"),
        "unexpected diverge inline output: {div_stdout}"
    );

    let div_json = Command::new(rul_bin)
        .arg("check")
        .arg(&diverge_path)
        .arg("--json")
        .output()
        .map_err(|e| e.to_string())?;
    assert!(div_json.status.success());
    let div_json_str = String::from_utf8_lossy(&div_json.stdout);
    assert!(
        div_json_str.trim().contains("\"static_cost\":null"),
        "expected null static_cost in JSON: {div_json_str}"
    );

    // 3. Invalid program (MutationUnderLiveLease): `--json` emits JSON error object and exits non-zero.
    let invalid_rls = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
node $z = ConstI64 i64x1 const_i64=0
node $v = ConstI64 i64x1 const_i64=1
node $ls = RegionLease i64x1 a=0 b=0
node $st = RegionStore i64x1 a=0 b=$z c=$v
output $v
";
    let invalid_path = work_dir.join("invalid.rls");
    std::fs::write(&invalid_path, invalid_rls).map_err(|e| e.to_string())?;

    let err_json = Command::new(rul_bin)
        .arg("check")
        .arg("--json")
        .arg(&invalid_path)
        .output()
        .map_err(|e| e.to_string())?;
    assert!(
        !err_json.status.success(),
        "expected non-zero exit status on verification failure"
    );
    let err_stdout = String::from_utf8_lossy(&err_json.stdout);
    let err_trimmed = err_stdout.trim();
    assert!(
        err_trimmed.starts_with('{') && err_trimmed.ends_with('}') && !err_trimmed.contains('\n'),
        "expected single-line JSON error object, got: {err_stdout}"
    );
    assert!(err_trimmed.contains("\"status\":\"error\""));
    assert!(err_trimmed.contains("MutationUnderLiveLease"));

    // 4. Invalid program without `--json` exits non-zero and reports error on stderr.
    let err_text = Command::new(rul_bin)
        .arg("check")
        .arg(&invalid_path)
        .output()
        .map_err(|e| e.to_string())?;
    assert!(!err_text.status.success());
    let err_stderr = String::from_utf8_lossy(&err_text.stderr);
    assert!(err_stderr.contains("MutationUnderLiveLease"));

    // 5. Special characters in file path are properly JSON-escaped via shared `escape_json_string`.
    let quoted_path = work_dir.join("quote\"file.rls");
    std::fs::write(&quoted_path, diverge_rls).map_err(|e| e.to_string())?;
    let quoted_json = Command::new(rul_bin)
        .arg("check")
        .arg("--json")
        .arg(&quoted_path)
        .output()
        .map_err(|e| e.to_string())?;
    assert!(quoted_json.status.success());
    let quoted_out = String::from_utf8_lossy(&quoted_json.stdout);
    assert!(
        quoted_out.contains("quote\\\"file.rls"),
        "expected escaped quote in JSON file field: {quoted_out}"
    );

    // 6. Argument errors with `--json` emit a JSON error object and exit non-zero.
    let bad_flag_json = Command::new(rul_bin)
        .arg("check")
        .arg("--json")
        .arg("--bogus")
        .output()
        .map_err(|e| e.to_string())?;
    assert!(!bad_flag_json.status.success());
    let bad_flag_out = String::from_utf8_lossy(&bad_flag_json.stdout);
    let bad_flag_trimmed = bad_flag_out.trim();
    assert!(
        bad_flag_trimmed.starts_with('{')
            && bad_flag_trimmed.ends_with('}')
            && bad_flag_trimmed.contains("\"status\":\"error\"")
            && bad_flag_trimmed.contains("--bogus"),
        "expected JSON error object for bad flag, got: {bad_flag_out}"
    );

    let _ = std::fs::remove_dir_all(&work_dir);
    Ok(())
}
