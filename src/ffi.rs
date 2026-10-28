//! Zero-Panic Embeddable Bridge & `#[repr(C)]` FFI Surface (`rul-ffi` / `rul-embed`).
//!
//! Allows host applications (Rust web/DB frameworks, Python, Go, Bun/Node, or WASM)
//! to load `.rl` binary graphs in-memory, linearly verify them in $O(N)$, execute them
//! on the Tier 2 Fused L1-Tile / Multi-Core engine, or emit standalone `ELF64` binaries
//! into caller-owned buffers using C-compatible `#[repr(C)]` status pods.

use crate::aot::compile_to_elf64_bytes;
use crate::bytecode::RulProgram;
use crate::engine::{autotune_schedule, execute_tier2_with_zonemap, ArenaZoneMap, RuntimeValue};
use crate::verifier::verify_program;

/// Operation succeeded (`0`).
pub const RUL_OK: i32 = 0;
/// Binary `.rl` header or frame decoding failed (`-1`).
pub const RUL_ERR_FORMAT: i32 = -1;
/// Static $O(N)$ linear verification rejected the `.rl` graph (`-2`).
pub const RUL_ERR_VERIFY: i32 = -2;
/// Runtime execution failed (e.g., `AssertContract` or arena length mismatch) (`-3`).
pub const RUL_ERR_EXEC: i32 = -3;
/// Standalone `ELF64` AOT compilation failed (`-4`).
pub const RUL_ERR_AOT: i32 = -4;
/// Caller-provided output buffer is too small (`-5`).
pub const RUL_ERR_BUFFER_TOO_SMALL: i32 = -5;

/// C-ABI compatible (`#[repr(C)]`) result returned by [`rul_embed_verify_bytes`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RulFfiVerifyResult {
    /// Status code (`RUL_OK` = `0`, or negative error code).
    pub status: i32,
    /// Number of verified 128-bit instruction nodes (`or` offending node index on error).
    pub node_count_or_err_idx: u32,
    /// Number of fused L1-cache loop groups.
    pub fusion_groups: u32,
    /// `1` if zero runtime bounds checks are guaranteed, `0` otherwise.
    pub zero_runtime_bounds_checks: u32,
    ///Deterministic 64-bit graph hash from the `.rl` header.
    pub graph_hash: u64,
}

/// C-ABI compatible (`#[repr(C)]`) result returned by [`rul_embed_exec_tier2_1col`]
/// and [`rul_embed_exec_tier2_2col`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RulFfiExecResult {
    /// Status code (`RUL_OK` = `0`, or negative error code).
    pub status: i32,
    /// Reserved padding for 8-byte alignment.
    pub reserved: i32,
    /// Scalar output value (or first element if output is columnar).
    pub scalar_output: f64,
    /// Time spent in $O(N)$ linear verifier (nanoseconds).
    pub verify_ns: u64,
    /// Time spent in Tier 2 Fused L1-Tile execution (nanoseconds).
    pub exec_ns: u64,
    /// Intermediate heap bytes allocated (`0` for fused L1-tile pipelines).
    pub intermediate_bytes: u64,
    /// Effective memory bandwidth achieved (`GB/s`).
    pub bandwidth_gbps: f64,
    /// Hardware telemetry reward score.
    pub reward_score: f64,
}

/// C-ABI compatible (`#[repr(C)]`) result returned by [`rul_embed_compile_elf64_to_buffer`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RulFfiAotResult {
    /// Status code (`RUL_OK` = `0`, or negative error code).
    pub status: i32,
    /// Total bytes written to `out_elf_buf` (or required size if `RUL_ERR_BUFFER_TOO_SMALL`).
    pub elf_bytes: u32,
    /// Size of the raw `x86_64` machine code payload in bytes.
    pub machine_code_bytes: u32,
    /// Reserved padding for 8-byte alignment.
    pub reserved: u32,
    /// Total compile latency in nanoseconds (`verify_ns + codegen_ns`).
    pub total_compile_ns: u64,
}

/// Linearly verifies a `.rl` binary buffer in $O(N)$ without panicking or allocating
/// unbounded memory, returning a C-compatible [`RulFfiVerifyResult`].
#[must_use]
pub fn rul_embed_verify_bytes(rl_bytes: &[u8]) -> RulFfiVerifyResult {
    let Ok(prog) = RulProgram::from_rl_bytes(rl_bytes) else {
        return RulFfiVerifyResult {
            status: RUL_ERR_FORMAT,
            node_count_or_err_idx: 0,
            fusion_groups: 0,
            zero_runtime_bounds_checks: 0,
            graph_hash: 0,
        };
    };

    match verify_program(&prog) {
        Ok(summary) => RulFfiVerifyResult {
            status: RUL_OK,
            node_count_or_err_idx: u32::try_from(summary.node_count).unwrap_or(u32::MAX),
            fusion_groups: u32::try_from(summary.fusion_groups).unwrap_or(u32::MAX),
            zero_runtime_bounds_checks: u32::from(summary.zero_runtime_bounds_checks),
            graph_hash: prog.compute_graph_hash(),
        },
        Err(diag) => RulFfiVerifyResult {
            status: RUL_ERR_VERIFY,
            node_count_or_err_idx: diag.node_idx,
            fusion_groups: 0,
            zero_runtime_bounds_checks: 0,
            graph_hash: 0,
        },
    }
}

/// Executes a single-arena `.rl` binary buffer on the Tier 2 Fused L1-Tile Engine
/// with automatic Zone-Map acceleration.
#[must_use]
pub fn rul_embed_exec_tier2_1col(rl_bytes: &[u8], col0: &[f64]) -> RulFfiExecResult {
    let Ok(prog) = RulProgram::from_rl_bytes(rl_bytes) else {
        return err_exec_result(RUL_ERR_FORMAT);
    };
    let zmap = ArenaZoneMap::build(col0);
    match execute_tier2_with_zonemap(&prog, &[col0], Some(&zmap)) {
        Ok(rep) => ok_exec_result(&rep.output, &rep.telemetry),
        Err(_) => err_exec_result(RUL_ERR_EXEC),
    }
}

/// Executes a two-arena `.rl` binary buffer on the Tier 2 Fused L1-Tile Engine.
#[must_use]
pub fn rul_embed_exec_tier2_2col(rl_bytes: &[u8], col0: &[f64], col1: &[f64]) -> RulFfiExecResult {
    let Ok(prog) = RulProgram::from_rl_bytes(rl_bytes) else {
        return err_exec_result(RUL_ERR_FORMAT);
    };
    let zmap = ArenaZoneMap::build(col0);
    match execute_tier2_with_zonemap(&prog, &[col0, col1], Some(&zmap)) {
        Ok(rep) => ok_exec_result(&rep.output, &rep.telemetry),
        Err(_) => err_exec_result(RUL_ERR_EXEC),
    }
}

/// Autotunes a `.rl` binary buffer in-memory against `col0`, writes the schedule-optimized
/// `.rl` binary into `out_rl_buf`, and returns the winning [`RulFfiExecResult`].
#[must_use]
pub fn rul_embed_autotune_1col(
    rl_bytes: &[u8],
    col0: &[f64],
    out_rl_buf: &mut [u8],
) -> RulFfiExecResult {
    let Ok(mut prog) = RulProgram::from_rl_bytes(rl_bytes) else {
        return err_exec_result(RUL_ERR_FORMAT);
    };
    let zmap = ArenaZoneMap::build(col0);
    let Ok(rep) = autotune_schedule(&mut prog, &[col0], Some(&zmap)) else {
        return err_exec_result(RUL_ERR_EXEC);
    };

    let tuned_bytes = prog.to_rl_bytes();
    let Some(dst) = out_rl_buf.get_mut(..tuned_bytes.len()) else {
        return err_exec_result(RUL_ERR_BUFFER_TOO_SMALL);
    };
    dst.copy_from_slice(&tuned_bytes);

    ok_exec_result(&rep.output, &rep.best_telemetry)
}

/// Compiles a `.rl` binary buffer directly into a standalone Linux `x86_64` `ELF64` executable
/// image inside `out_elf_buf` in `< 1 ms` with zero external dependencies.
#[must_use]
pub fn rul_embed_compile_elf64_to_buffer(
    rl_bytes: &[u8],
    out_elf_buf: &mut [u8],
) -> RulFfiAotResult {
    let Ok(prog) = RulProgram::from_rl_bytes(rl_bytes) else {
        return RulFfiAotResult {
            status: RUL_ERR_FORMAT,
            elf_bytes: 0,
            machine_code_bytes: 0,
            reserved: 0,
            total_compile_ns: 0,
        };
    };

    let Ok((elf_vec, report)) = compile_to_elf64_bytes(&prog) else {
        return RulFfiAotResult {
            status: RUL_ERR_AOT,
            elf_bytes: 0,
            machine_code_bytes: 0,
            reserved: 0,
            total_compile_ns: 0,
        };
    };

    let elf_len_u32 = u32::try_from(elf_vec.len()).unwrap_or(u32::MAX);
    let mc_len_u32 = u32::try_from(report.machine_code_bytes).unwrap_or(u32::MAX);

    let Some(dst) = out_elf_buf.get_mut(..elf_vec.len()) else {
        return RulFfiAotResult {
            status: RUL_ERR_BUFFER_TOO_SMALL,
            elf_bytes: elf_len_u32,
            machine_code_bytes: mc_len_u32,
            reserved: 0,
            total_compile_ns: report.total_compile_ns,
        };
    };
    dst.copy_from_slice(&elf_vec);

    RulFfiAotResult {
        status: RUL_OK,
        elf_bytes: elf_len_u32,
        machine_code_bytes: mc_len_u32,
        reserved: 0,
        total_compile_ns: report.total_compile_ns,
    }
}

fn ok_exec_result(
    output: &RuntimeValue,
    telemetry: &crate::engine::HardwareTelemetry,
) -> RulFfiExecResult {
    let scalar_output = match output {
        RuntimeValue::Scalar(v) => *v,
        RuntimeValue::ColumnF64(col) => col.first().copied().unwrap_or(0.0),
        RuntimeValue::ColumnMask(mask) => {
            if mask.first().copied().unwrap_or(false) {
                1.0
            } else {
                0.0
            }
        }
    };
    RulFfiExecResult {
        status: RUL_OK,
        reserved: 0,
        scalar_output,
        verify_ns: telemetry.verify_ns,
        exec_ns: telemetry.exec_ns,
        intermediate_bytes: u64::try_from(telemetry.intermediate_bytes_allocated)
            .unwrap_or(u64::MAX),
        bandwidth_gbps: telemetry.effective_bandwidth_gbps,
        reward_score: telemetry.reward_score,
    }
}

const fn err_exec_result(status: i32) -> RulFfiExecResult {
    RulFfiExecResult {
        status,
        reserved: 0,
        scalar_output: f64::NAN,
        verify_ns: 0,
        exec_ns: 0,
        intermediate_bytes: 0,
        bandwidth_gbps: 0.0,
        reward_score: 0.0,
    }
}
