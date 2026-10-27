//! 10-Pipeline Benchmark & Kill-Criteria Suite + Bounded `.rl` File I/O.
//!
//! Validates Rul (`.rl`) across 10 diverse Data Pipeline workloads (Filter, Map,
//! Reduce, Slice with `ProofEntry`, Two-Column Spread/Weighted, Stream Compaction,
//! Indirect `Gather` with bounds proof, and Linear `ArenaMutate` + `RingEnqueue` sandbox).
//! Every pipeline is differentially verified across `Tier 0` (Reference Interpreter),
//! `Tier 1` (Zero-Copy Stencil), `Tier 2` (L1-Cache Tile / Fused Engine), and a
//! ground-truth Rust calculation.

use std::fmt::{self, Write as _};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

use crate::bytecode::{
    Opcode, ProofEntry, ProofRuleKind, RulFormatError, RulProgram, ScheduleHint, TypeLane,
    CAP_NET_RING, CAP_NONE,
};
use crate::engine::{
    execute_tier0_interpreter, execute_tier1_stencil, execute_tier2_fused, ExecError, RuntimeValue,
};
use crate::transpiler::{SymbolicGraphBuilder, TranspileError};
use crate::verifier::{verify_program, VerifyDiagnostic};

/// Maximum permitted `.rl` file size when loading from disk (16 MiB bounded read).
pub const MAX_RL_FILE_BYTES: u64 = 16 * 1024 * 1024;

/// Errors returned by `.rl` file I/O or the 10-pipeline suite.
#[derive(Debug)]
pub enum SuiteError {
    /// Standard I/O error.
    Io(std::io::Error),
    /// File exceeds [`MAX_RL_FILE_BYTES`].
    FileTooLarge(u64),
    /// Binary `.rl` format error.
    Format(RulFormatError),
    /// Symbolic transpiler error.
    Transpile(TranspileError),
    /// Static $O(N)$ verification error.
    Verify(VerifyDiagnostic),
    /// Runtime execution error.
    Exec(ExecError),
    /// Output value did not match expected reference oracle or across execution tiers.
    OracleMismatch {
        /// Pipeline name.
        pipeline: &'static str,
        /// Expected scalar result.
        expected: f64,
        /// Observed scalar result.
        observed: f64,
    },
}

impl fmt::Display for SuiteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::FileTooLarge(sz) => write!(f, ".rl file too large: {sz} bytes"),
            Self::Format(e) => write!(f, "format error: {e}"),
            Self::Transpile(e) => write!(f, "transpile error: {e}"),
            Self::Verify(e) => write!(f, "verify error: {e}"),
            Self::Exec(e) => write!(f, "execution error: {e}"),
            Self::OracleMismatch {
                pipeline,
                expected,
                observed,
            } => write!(
                f,
                "oracle mismatch in {pipeline}: expected {expected}, observed {observed}"
            ),
        }
    }
}

impl std::error::Error for SuiteError {}

/// Saves a [`RulProgram`] to a `.rl` binary file on disk.
///
/// # Errors
/// Returns [`SuiteError::Io`] if creating or writing the file fails.
pub fn save_rl_file(path: impl AsRef<Path>, program: &RulProgram) -> Result<usize, SuiteError> {
    let bytes = program.to_rl_bytes();
    let mut file = File::create(path.as_ref()).map_err(SuiteError::Io)?;
    file.write_all(&bytes).map_err(SuiteError::Io)?;
    Ok(bytes.len())
}

/// Loads and validates a bounded `.rl` binary file from disk.
///
/// # Errors
/// Returns [`SuiteError`] if the file exceeds [`MAX_RL_FILE_BYTES`] or fails `.rl` decoding.
pub fn load_rl_file(path: impl AsRef<Path>) -> Result<RulProgram, SuiteError> {
    let meta = fs::metadata(path.as_ref()).map_err(SuiteError::Io)?;
    if meta.len() > MAX_RL_FILE_BYTES {
        return Err(SuiteError::FileTooLarge(meta.len()));
    }
    let cap = usize::try_from(meta.len()).unwrap_or(0);
    let mut buf = Vec::with_capacity(cap);
    let file = File::open(path.as_ref()).map_err(SuiteError::Io)?;
    file.take(MAX_RL_FILE_BYTES)
        .read_to_end(&mut buf)
        .map_err(SuiteError::Io)?;
    RulProgram::from_rl_bytes(&buf).map_err(SuiteError::Format)
}

/// Metrics recorded for a single pipeline in the 10-problem benchmark suite.
#[derive(Debug, Clone)]
pub struct PipelineCaseReport {
    /// Pipeline identifier.
    pub name: &'static str,
    /// Number of 128-bit nodes in the `.rl` program.
    pub node_count: usize,
    /// Total `.rl` binary size in bytes (including 64-byte header).
    pub rl_bytes: usize,
    /// Intermediate heap bytes allocated by Tier 2 (`0` for L1-tiled/fused pipelines).
    pub interm_bytes_t2: usize,
    /// Single-pass $O(N)$ verification latency in nanoseconds.
    pub verify_ns: u64,
    /// Tier 2 execution time in nanoseconds.
    pub exec_ns: u64,
    /// Verified scalar output.
    pub output_scalar: f64,
    /// True if `Tier 0 == Tier 1 == Tier 2 == Ground-Truth`.
    pub tiers_matched: bool,
    /// Whether sidecar `ProofEntry`, `L1-Tiled`, or `AffineConsume` was exercised.
    pub proof_feature: &'static str,
}

/// Builds and runs all 10 Data Pipeline benchmark cases, verifying 100% $O(N)$ acceptance
/// and 100% differential equivalence (`Tier 0 == Tier 1 == Tier 2 == Ground-Truth`).
///
/// # Errors
/// Returns [`SuiteError`] if any pipeline fails verification or disagrees across tiers.
pub fn run_10_pipeline_suite(elem_count: u32) -> Result<Vec<PipelineCaseReport>, SuiteError> {
    let n = elem_count as usize;
    let col_a: Vec<f64> = (0..elem_count).map(|i| f64::from(i % 2000) + 1.0).collect();
    let idx_mod = elem_count.clamp(1, 500);
    let col_b: Vec<f64> = (0..elem_count).map(|i| f64::from(i % idx_mod)).collect();

    let mut reports = Vec::with_capacity(10);

    // P01: Filter (> 1000) -> Square -> ReduceSum (Fused SIMD Group 1)
    {
        let mut b = SymbolicGraphBuilder::new(CAP_NONE);
        let a0 = b.add_arena(elem_count, TypeLane::F64X8, false);
        let s = ScheduleHint::fused_simd(1);
        b.emit_arena_input("$a", a0, 1, s)
            .map_err(SuiteError::Transpile)?;
        b.emit_const_f64("$thr", 1000.0, s)
            .map_err(SuiteError::Transpile)?;
        b.emit_binary_op("$m", Opcode::CmpGt, TypeLane::MASKX8, "$a", "$thr", s)
            .map_err(SuiteError::Transpile)?;
        b.emit_binary_op("$sq", Opcode::Mul, TypeLane::F64X8, "$a", "$a", s)
            .map_err(SuiteError::Transpile)?;
        b.emit_binary_op("$sel", Opcode::SelectZero, TypeLane::F64X8, "$m", "$sq", s)
            .map_err(SuiteError::Transpile)?;
        b.emit_unary_op("$sum", Opcode::ReduceSum, TypeLane::F64_SCALAR, "$sel", s)
            .map_err(SuiteError::Transpile)?;
        let prog = b.finish("$sum").map_err(SuiteError::Transpile)?;
        let expected: f64 = col_a
            .iter()
            .copied()
            .filter(|&x| x > 1000.0)
            .map(|x| x * x)
            .sum();
        reports.push(eval_case(
            "P01_FilterSquareSum",
            &prog,
            &[&col_a],
            expected,
            "Shape-Trivial (Fused 8-Acc)",
        )?);
    }

    // P02: Filter (> 500) -> Scale (* 0.15) -> ReduceSum (Fused SIMD Group 1)
    {
        let mut b = SymbolicGraphBuilder::new(CAP_NONE);
        let a0 = b.add_arena(elem_count, TypeLane::F64X8, false);
        let s = ScheduleHint::fused_simd(1);
        b.emit_arena_input("$a", a0, 1, s)
            .map_err(SuiteError::Transpile)?;
        b.emit_const_f64("$thr", 500.0, s)
            .map_err(SuiteError::Transpile)?;
        b.emit_const_f64("$rate", 0.15, s)
            .map_err(SuiteError::Transpile)?;
        b.emit_binary_op("$m", Opcode::CmpGt, TypeLane::MASKX8, "$a", "$thr", s)
            .map_err(SuiteError::Transpile)?;
        b.emit_binary_op("$tax", Opcode::Mul, TypeLane::F64X8, "$a", "$rate", s)
            .map_err(SuiteError::Transpile)?;
        b.emit_binary_op("$sel", Opcode::SelectZero, TypeLane::F64X8, "$m", "$tax", s)
            .map_err(SuiteError::Transpile)?;
        b.emit_unary_op("$sum", Opcode::ReduceSum, TypeLane::F64_SCALAR, "$sel", s)
            .map_err(SuiteError::Transpile)?;
        let prog = b.finish("$sum").map_err(SuiteError::Transpile)?;
        let expected: f64 = col_a
            .iter()
            .copied()
            .filter(|&x| x > 500.0)
            .map(|x| x * 0.15)
            .sum();
        reports.push(eval_case(
            "P02_ThresholdTaxSum",
            &prog,
            &[&col_a],
            expected,
            "Shape-Trivial (Fused 8-Acc)",
        )?);
    }

    // P03: Filter (> 1500) -> Identity -> ReduceSum (Fused SIMD Group 1)
    {
        let mut b = SymbolicGraphBuilder::new(CAP_NONE);
        let a0 = b.add_arena(elem_count, TypeLane::F64X8, false);
        let s = ScheduleHint::fused_simd(1);
        b.emit_arena_input("$a", a0, 1, s)
            .map_err(SuiteError::Transpile)?;
        b.emit_const_f64("$thr", 1500.0, s)
            .map_err(SuiteError::Transpile)?;
        b.emit_binary_op("$m", Opcode::CmpGt, TypeLane::MASKX8, "$a", "$thr", s)
            .map_err(SuiteError::Transpile)?;
        b.emit_binary_op("$sel", Opcode::SelectZero, TypeLane::F64X8, "$m", "$a", s)
            .map_err(SuiteError::Transpile)?;
        b.emit_unary_op("$sum", Opcode::ReduceSum, TypeLane::F64_SCALAR, "$sel", s)
            .map_err(SuiteError::Transpile)?;
        let prog = b.finish("$sum").map_err(SuiteError::Transpile)?;
        let expected: f64 = col_a.iter().copied().filter(|&x| x > 1500.0).sum();
        reports.push(eval_case(
            "P03_HighValueFilterSum",
            &prog,
            &[&col_a],
            expected,
            "Shape-Trivial (Fused 8-Acc)",
        )?);
    }

    // P04: Bounds-Proven ArenaSlice [100..900] -> ReduceSum (L1-Cache Tile Engine)
    {
        let mut b = SymbolicGraphBuilder::new(CAP_NONE);
        let a0 = b.add_arena(elem_count, TypeLane::F64X8, false);
        let hi = elem_count.min(900);
        let lo = hi.min(100);
        let p1 = b.add_proof(ProofEntry {
            rule_kind: ProofRuleKind::SliceBounds,
            target_arena: 0,
            bound_lo: lo,
            bound_hi: hi,
            witness_node: 0,
        });
        let s = ScheduleHint::fused_simd(1);
        b.emit_arena_input("$a", a0, 1, s)
            .map_err(SuiteError::Transpile)?;
        b.emit_slice("$win", "$a", p1, s)
            .map_err(SuiteError::Transpile)?;
        b.emit_unary_op("$sum", Opcode::ReduceSum, TypeLane::F64_SCALAR, "$win", s)
            .map_err(SuiteError::Transpile)?;
        let prog = b.finish("$sum").map_err(SuiteError::Transpile)?;
        let expected: f64 = col_a
            .get(lo as usize..hi as usize)
            .unwrap_or(&[])
            .iter()
            .copied()
            .sum();
        reports.push(eval_case(
            "P04_ProvenSliceWindowSum",
            &prog,
            &[&col_a],
            expected,
            "ProofEntry(Slice)+L1Tile",
        )?);
    }

    // P05: Bounds-Proven ArenaSlice [0..500] -> Square -> ReduceMax (L1-Cache Tile Engine)
    {
        let mut b = SymbolicGraphBuilder::new(CAP_NONE);
        let a0 = b.add_arena(elem_count, TypeLane::F64X8, false);
        let hi = elem_count.min(500);
        let p1 = b.add_proof(ProofEntry {
            rule_kind: ProofRuleKind::SliceBounds,
            target_arena: 0,
            bound_lo: 0,
            bound_hi: hi,
            witness_node: 0,
        });
        let s = ScheduleHint::fused_simd(1);
        b.emit_arena_input("$a", a0, 1, s)
            .map_err(SuiteError::Transpile)?;
        b.emit_slice("$win", "$a", p1, s)
            .map_err(SuiteError::Transpile)?;
        b.emit_binary_op("$sq", Opcode::Mul, TypeLane::F64X8, "$win", "$win", s)
            .map_err(SuiteError::Transpile)?;
        b.emit_unary_op("$max", Opcode::ReduceMax, TypeLane::F64_SCALAR, "$sq", s)
            .map_err(SuiteError::Transpile)?;
        let prog = b.finish("$max").map_err(SuiteError::Transpile)?;
        let expected: f64 = col_a
            .get(0..hi as usize)
            .unwrap_or(&[])
            .iter()
            .map(|&x| x * x)
            .fold(f64::NEG_INFINITY, f64::max);
        reports.push(eval_case(
            "P05_SliceSquareMax",
            &prog,
            &[&col_a],
            expected,
            "ProofEntry(Slice)+L1Tile",
        )?);
    }

    // P06: Two-Column SoA Spread (Arena0 - Arena1) -> Filter (> 10) -> ReduceSum (L1-Cache Tile Engine)
    {
        let mut b = SymbolicGraphBuilder::new(CAP_NONE);
        let a0 = b.add_arena(elem_count, TypeLane::F64X8, false);
        let a1 = b.add_arena(elem_count, TypeLane::F64X8, false);
        let s = ScheduleHint::fused_simd(1);
        b.emit_arena_input("$a", a0, 1, s)
            .map_err(SuiteError::Transpile)?;
        b.emit_arena_input("$b", a1, 1, s)
            .map_err(SuiteError::Transpile)?;
        b.emit_binary_op("$diff", Opcode::Sub, TypeLane::F64X8, "$a", "$b", s)
            .map_err(SuiteError::Transpile)?;
        b.emit_const_f64("$thr", 10.0, s)
            .map_err(SuiteError::Transpile)?;
        b.emit_binary_op("$m", Opcode::CmpGt, TypeLane::MASKX8, "$diff", "$thr", s)
            .map_err(SuiteError::Transpile)?;
        b.emit_binary_op(
            "$sel",
            Opcode::SelectZero,
            TypeLane::F64X8,
            "$m",
            "$diff",
            s,
        )
        .map_err(SuiteError::Transpile)?;
        b.emit_unary_op("$sum", Opcode::ReduceSum, TypeLane::F64_SCALAR, "$sel", s)
            .map_err(SuiteError::Transpile)?;
        let prog = b.finish("$sum").map_err(SuiteError::Transpile)?;
        let expected: f64 = col_a
            .iter()
            .zip(col_b.iter())
            .map(|(&x, &y)| x - y)
            .filter(|&d| d > 10.0)
            .sum();
        reports.push(eval_case(
            "P06_TwoColumnSpreadSum",
            &prog,
            &[&col_a, &col_b],
            expected,
            "Dual ReadLease+L1Tile",
        )?);
    }

    // P07: Two-Column SoA Weighted (Arena0 * Arena1 + Arena0) -> ReduceMax (L1-Cache Tile Engine)
    {
        let mut b = SymbolicGraphBuilder::new(CAP_NONE);
        let a0 = b.add_arena(elem_count, TypeLane::F64X8, false);
        let a1 = b.add_arena(elem_count, TypeLane::F64X8, false);
        let s = ScheduleHint::fused_simd(1);
        b.emit_arena_input("$a", a0, 1, s)
            .map_err(SuiteError::Transpile)?;
        b.emit_arena_input("$b", a1, 1, s)
            .map_err(SuiteError::Transpile)?;
        b.emit_binary_op("$prod", Opcode::Mul, TypeLane::F64X8, "$a", "$b", s)
            .map_err(SuiteError::Transpile)?;
        b.emit_binary_op("$w", Opcode::Add, TypeLane::F64X8, "$prod", "$a", s)
            .map_err(SuiteError::Transpile)?;
        b.emit_unary_op("$max", Opcode::ReduceMax, TypeLane::F64_SCALAR, "$w", s)
            .map_err(SuiteError::Transpile)?;
        let prog = b.finish("$max").map_err(SuiteError::Transpile)?;
        let expected: f64 = col_a
            .iter()
            .zip(col_b.iter())
            .map(|(&x, &y)| x * y + x)
            .fold(f64::NEG_INFINITY, f64::max);
        reports.push(eval_case(
            "P07_TwoColumnWeightedMax",
            &prog,
            &[&col_a, &col_b],
            expected,
            "Dual ReadLease+L1Tile",
        )?);
    }

    // P08: Stream Compaction (FilterCompact > 1200) -> Double -> ReduceSum
    {
        let mut b = SymbolicGraphBuilder::new(CAP_NONE);
        let a0 = b.add_arena(elem_count, TypeLane::F64X8, false);
        let s = ScheduleHint::UNFUSED;
        b.emit_arena_input("$a", a0, 1, s)
            .map_err(SuiteError::Transpile)?;
        b.emit_const_f64("$thr", 1200.0, s)
            .map_err(SuiteError::Transpile)?;
        b.emit_binary_op("$m", Opcode::CmpGt, TypeLane::MASKX8, "$a", "$thr", s)
            .map_err(SuiteError::Transpile)?;
        b.emit_binary_op(
            "$comp",
            Opcode::FilterCompact,
            TypeLane::F64X8,
            "$m",
            "$a",
            s,
        )
        .map_err(SuiteError::Transpile)?;
        b.emit_binary_op("$dbl", Opcode::Add, TypeLane::F64X8, "$comp", "$comp", s)
            .map_err(SuiteError::Transpile)?;
        b.emit_unary_op("$sum", Opcode::ReduceSum, TypeLane::F64_SCALAR, "$dbl", s)
            .map_err(SuiteError::Transpile)?;
        let prog = b.finish("$sum").map_err(SuiteError::Transpile)?;
        let expected: f64 = col_a
            .iter()
            .copied()
            .filter(|&x| x > 1200.0)
            .map(|x| x + x)
            .sum();
        reports.push(eval_case(
            "P08_StreamCompactDoubleSum",
            &prog,
            &[&col_a],
            expected,
            "Compacted Subcolumn Shape",
        )?);
    }

    // P09: Indirect Gather with Sidecar ProofEntry (ProofRuleKind::GatherIndexInBounds)
    {
        let mut b = SymbolicGraphBuilder::new(CAP_NONE);
        let a0 = b.add_arena(elem_count, TypeLane::F64X8, false);
        let a1 = b.add_arena(elem_count, TypeLane::F64X8, false);
        let p_gather = b.add_proof(ProofEntry {
            rule_kind: ProofRuleKind::GatherIndexInBounds,
            target_arena: 0,
            bound_lo: 0,
            bound_hi: elem_count,
            witness_node: 1,
        });
        let s = ScheduleHint::UNFUSED;
        b.emit_arena_input("$target", a0, 1, s)
            .map_err(SuiteError::Transpile)?;
        b.emit_arena_input("$idx", a1, 1, s)
            .map_err(SuiteError::Transpile)?;
        b.emit_gather("$gathered", "$target", "$idx", p_gather, s)
            .map_err(SuiteError::Transpile)?;
        b.emit_unary_op(
            "$sum",
            Opcode::ReduceSum,
            TypeLane::F64_SCALAR,
            "$gathered",
            s,
        )
        .map_err(SuiteError::Transpile)?;
        let prog = b.finish("$sum").map_err(SuiteError::Transpile)?;
        let expected: f64 = col_b
            .iter()
            .map(|&idx| {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let pos = idx as usize;
                col_a.get(pos).copied().unwrap_or(0.0)
            })
            .sum();
        reports.push(eval_case(
            "P09_ProvenIndirectGather",
            &prog,
            &[&col_a, &col_b],
            expected,
            "ProofEntry(GatherBounds)",
        )?);
    }

    // P10: Linear AffineConsume Mutation + Zero-Trust RingEnqueue (CAP_NET_RING, L1-Cache Tile Engine)
    {
        let mut b = SymbolicGraphBuilder::new(CAP_NET_RING);
        let a0 = b.add_arena(elem_count, TypeLane::F64X8, true);
        let s = ScheduleHint::fused_simd(1);
        b.emit_arena_input("$a_v1", a0, 1, s)
            .map_err(SuiteError::Transpile)?;
        b.emit_const_f64("$bonus", 5.0, s)
            .map_err(SuiteError::Transpile)?;
        b.emit_binary_op(
            "$updated",
            Opcode::Add,
            TypeLane::F64X8,
            "$a_v1",
            "$bonus",
            s,
        )
        .map_err(SuiteError::Transpile)?;
        b.emit_mutate("$a_v2", a0, 1, "$updated", s)
            .map_err(SuiteError::Transpile)?;
        b.emit_unary_op("$sum", Opcode::ReduceSum, TypeLane::F64_SCALAR, "$a_v2", s)
            .map_err(SuiteError::Transpile)?;
        b.emit_unary_op("$net", Opcode::RingEnqueue, TypeLane::F64_SCALAR, "$sum", s)
            .map_err(SuiteError::Transpile)?;
        let prog = b.finish("$net").map_err(SuiteError::Transpile)?;
        let expected: f64 = col_a.iter().copied().sum::<f64>() + (n as f64) * 5.0;
        reports.push(eval_case(
            "P10_AffineMutateNetRing",
            &prog,
            &[&col_a],
            expected,
            "AffineConsume(v1->v2)+L1Tile",
        )?);
    }

    Ok(reports)
}

fn eval_case(
    name: &'static str,
    prog: &RulProgram,
    inputs: &[&[f64]],
    expected: f64,
    proof_feature: &'static str,
) -> Result<PipelineCaseReport, SuiteError> {
    let _summary = verify_program(prog).map_err(SuiteError::Verify)?;

    // Execute across all 3 tiers and enforce differential equivalence!
    let rep0 = execute_tier0_interpreter(prog, inputs).map_err(SuiteError::Exec)?;
    let rep1 = execute_tier1_stencil(prog, inputs).map_err(SuiteError::Exec)?;
    let rep2 = execute_tier2_fused(prog, inputs).map_err(SuiteError::Exec)?;

    let (RuntimeValue::Scalar(obs0), RuntimeValue::Scalar(obs1), RuntimeValue::Scalar(obs2)) =
        (rep0.output, rep1.output, rep2.output)
    else {
        return Err(SuiteError::OracleMismatch {
            pipeline: name,
            expected,
            observed: f64::NAN,
        });
    };

    let denom = expected.abs().max(1.0);
    let err0 = (obs0 - expected).abs() / denom;
    let err1 = (obs1 - expected).abs() / denom;
    let err2 = (obs2 - expected).abs() / denom;
    let err_t0_t2 = (obs0 - obs2).abs() / denom;

    if err0 > 1e-9 || err1 > 1e-9 || err2 > 1e-9 || err_t0_t2 > 1e-9 {
        return Err(SuiteError::OracleMismatch {
            pipeline: name,
            expected,
            observed: obs2,
        });
    }

    Ok(PipelineCaseReport {
        name,
        node_count: prog.nodes.len(),
        rl_bytes: prog.to_rl_bytes().len(),
        interm_bytes_t2: rep2.telemetry.intermediate_bytes_allocated,
        verify_ns: rep2.telemetry.verify_ns,
        exec_ns: rep2.telemetry.exec_ns,
        output_scalar: obs2,
        tiers_matched: true,
        proof_feature,
    })
}

/// Formats the 10-pipeline suite results as a compact markdown/ASCII table.
#[must_use]
pub fn format_suite10_table(reports: &[PipelineCaseReport]) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "=== 4. 10-PIPELINE DIFFERENTIAL VERIFICATION SUITE (T0 == T1 == T2 == ORACLE) ==="
    );
    let _ = writeln!(
        out,
        "PIPELINE                   | NODES | .RL BYTES | T2 INTERM RAM | VERIFY (ns) | T0==T1==T2 | PROOF / SCHEDULE CERTIFICATE"
    );
    let _ = writeln!(
        out,
        "---------------------------------------------------------------------------------------------------------------------------"
    );
    for r in reports {
        let match_str = if r.tiers_matched { "PASS" } else { "FAIL" };
        let _ = writeln!(
            out,
            "{:<26} | {:>5} | {:>7} B | {:>11} B | {:>9} ns | {:>10} | {}",
            r.name,
            r.node_count,
            r.rl_bytes,
            r.interm_bytes_t2,
            r.verify_ns,
            match_str,
            r.proof_feature
        );
    }
    out
}
