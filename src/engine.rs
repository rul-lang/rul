//! 3-Tier Execution Engine, L1-Cache Tile Vector Evaluator, Zone-Map Skipping,
//! and Hardware Telemetry for Rul (`.rl`) programs.
//!
//! - **Tier 0 (`execute_tier0_interpreter`)**: Zero-Copy Reference Columnar Interpreter (Golden Oracle).
//! - **Tier 1 (`execute_tier1_stencil`)**: Dead-Node-Eliminated 8-Lane SIMD Vector Stencil Engine.
//! - **Tier 2 (`execute_tier2_fused`)**: Exact-DAG L1-Cache Tile Engine (`1,024` elements/tile = `8 KB`),
//!   with **Auto-MultiCore Parallelism (`ExecPolicy::MultiCore`)** and **Zone-Map Block Skipping**.

use std::borrow::Cow;
use std::fmt;
use std::time::Instant;

use crate::bytecode::{ElemType, ExecPolicy, NodeFrame, Opcode, RulProgram, ScheduleHint};
use crate::verifier::{verify_program, VerifyDiagnostic};

/// L1-Cache Tile width (`1,024` `f64` elements = `8 KB`, fitting comfortably inside 32–48 KB L1d).
pub const L1_TILE_LEN: usize = 1024;

/// Precomputed `(min, max)` zone-map summary per [`L1_TILE_LEN`] block of an `Arena` column.
///
/// Allows the Rul engine to skip entire 1,024-element blocks in $O(1)$ when a filter
/// threshold cannot be satisfied, beating sequential C loops that must scan 100% of RAM.
#[derive(Debug, Clone, PartialEq)]
pub struct ArenaZoneMap {
    /// `(min, max)` pair for each `1,024`-element tile.
    pub tile_bounds: Vec<(f64, f64)>,
}

impl ArenaZoneMap {
    /// Builds a zone map for a columnar `f64` slice in one streaming scan.
    #[must_use]
    pub fn build(slice: &[f64]) -> Self {
        let mut tile_bounds = Vec::with_capacity(slice.len().div_ceil(L1_TILE_LEN));
        for tile in slice.chunks(L1_TILE_LEN) {
            let mut min_v = f64::INFINITY;
            let mut max_v = f64::NEG_INFINITY;
            for &x in tile {
                if x < min_v {
                    min_v = x;
                }
                if x > max_v {
                    max_v = x;
                }
            }
            tile_bounds.push((min_v, max_v));
        }
        Self { tile_bounds }
    }
}

/// Runtime value produced by evaluating a node in the `.rl` graph.
#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeValue {
    /// A single scalar `f64` result.
    Scalar(f64),
    /// A columnar `f64` vector.
    ColumnF64(Vec<f64>),
    /// A columnar boolean mask.
    ColumnMask(Vec<bool>),
}

/// Internal zero-copy evaluation value used by Tier 0 and Tier 1.
#[derive(Debug, Clone)]
enum EvalValue<'a> {
    Unused,
    Scalar(f64),
    ColumnF64(Cow<'a, [f64]>),
    ColumnMask(Vec<bool>),
}

impl EvalValue<'_> {
    fn into_runtime_value(self, node_idx: u32) -> Result<RuntimeValue, ExecError> {
        match self {
            Self::Scalar(v) => Ok(RuntimeValue::Scalar(v)),
            Self::ColumnF64(cow) => Ok(RuntimeValue::ColumnF64(cow.into_owned())),
            Self::ColumnMask(m) => Ok(RuntimeValue::ColumnMask(m)),
            Self::Unused => Err(ExecError::InvalidOperandState(node_idx)),
        }
    }
}

/// Hardware performance telemetry returned to the AI Agent as a reward vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HardwareTelemetry {
    /// Time spent in the single-pass $O(N)$ verifier (nanoseconds).
    pub verify_ns: u64,
    /// Time spent executing the graph (nanoseconds).
    pub exec_ns: u64,
    /// Number of input column elements processed.
    pub elements_processed: usize,
    /// Effective memory read bandwidth achieved (`GB/s`).
    pub effective_bandwidth_gbps: f64,
    /// Intermediate heap bytes allocated during pipeline execution (`0` when L1-tiled/fused!).
    pub intermediate_bytes_allocated: usize,
    /// Normalized reward score for the AI Agent's `ScheduleHint` mutation loop.
    pub reward_score: f64,
}

/// Result of executing a verified `.rl` program.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionReport {
    /// Primary scalar or columnar output value.
    pub output: RuntimeValue,
    /// Telemetry feedback for the AI Agent.
    pub telemetry: HardwareTelemetry,
}

/// Errors that can occur during `.rl` program execution.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecError {
    /// Program failed $O(N)$ static verification before execution.
    VerificationFailed(VerifyDiagnostic),
    /// Missing runtime input data for declared `arena_id`.
    MissingArenaInput(u32),
    /// Input column length does not match `ArenaDescriptor.element_count`.
    ArenaLengthMismatch {
        /// Arena ID.
        arena_id: u32,
        /// Expected element count from header.
        expected: usize,
        /// Actual slice length provided.
        actual: usize,
    },
    /// Invalid operand value type at node index.
    InvalidOperandState(u32),
    /// Indirect `Gather` index violated the proven `[0..bound_hi)` refinement range.
    GatherIndexOutOfBounds {
        /// Offending `Gather` node index.
        node_idx: u32,
        /// Observed out-of-bounds index value.
        index: f64,
        /// Proven upper bound (`bound_hi`).
        bound_hi: u32,
    },
    /// Output contract assertion (`AssertContract`) failed (e.g., NaN or infinite output).
    ContractAssertionFailed {
        /// Offending node index.
        node_idx: u32,
        /// Observed scalar value.
        observed: f64,
    },
}

impl fmt::Display for ExecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VerificationFailed(diag) => write!(f, "{diag}"),
            Self::MissingArenaInput(id) => write!(f, "missing input slice for arena #{id}"),
            Self::ArenaLengthMismatch {
                arena_id,
                expected,
                actual,
            } => write!(
                f,
                "arena #{arena_id} length mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidOperandState(idx) => write!(f, "invalid operand state at node #{idx}"),
            Self::GatherIndexOutOfBounds {
                node_idx,
                index,
                bound_hi,
            } => write!(
                f,
                "gather index out of bounds at node #{node_idx}: index={index}, bound_hi={bound_hi}"
            ),
            Self::ContractAssertionFailed { node_idx, observed } => write!(
                f,
                "contract assertion failed at node #{node_idx}: observed {observed}"
            ),
        }
    }
}

impl std::error::Error for ExecError {}

fn validate_inputs(program: &RulProgram, inputs: &[&[f64]]) -> Result<usize, ExecError> {
    let mut total_elements = 0usize;
    for desc in &program.arenas {
        let slice = inputs
            .get(desc.arena_id as usize)
            .copied()
            .ok_or(ExecError::MissingArenaInput(desc.arena_id))?;
        let expected = desc.element_count as usize;
        if slice.len() != expected {
            return Err(ExecError::ArenaLengthMismatch {
                arena_id: desc.arena_id,
                expected,
                actual: slice.len(),
            });
        }
        total_elements = total_elements.saturating_add(slice.len());
    }
    Ok(total_elements)
}

fn build_telemetry(
    verify_ns: u64,
    exec_ns: u64,
    elements_processed: usize,
    intermediate_bytes_allocated: usize,
) -> HardwareTelemetry {
    let bytes_read = (elements_processed as f64) * 8.0;
    let seconds = (exec_ns.max(1) as f64) * 1e-9;
    let effective_bandwidth_gbps = (bytes_read / seconds) / 1e9;
    let alloc_penalty = 1.0 + (intermediate_bytes_allocated as f64) / (bytes_read.max(1.0));
    let reward_score = effective_bandwidth_gbps / alloc_penalty;
    HardwareTelemetry {
        verify_ns,
        exec_ns,
        elements_processed,
        effective_bandwidth_gbps,
        intermediate_bytes_allocated,
        reward_score,
    }
}

/// Computes exact backward DAG reachability starting from `program.output_node`.
///
/// Any dead node not connected to `output_node` is marked `false` and never evaluated.
#[must_use]
pub fn compute_live_nodes(program: &RulProgram) -> Vec<bool> {
    let n = program.nodes.len();
    let mut live = vec![false; n];
    if let Some(slot) = live.get_mut(program.output_node as usize) {
        *slot = true;
    }
    for idx in (0..n).rev() {
        let is_live = live.get(idx).copied().unwrap_or(false);
        if !is_live {
            continue;
        }
        let Some(frame) = program.nodes.get(idx) else {
            continue;
        };
        match frame.opcode {
            Opcode::ArenaInput | Opcode::ConstF64 => {}
            Opcode::ArenaSlice
            | Opcode::Abs
            | Opcode::Sqrt
            | Opcode::Floor
            | Opcode::Ceil
            | Opcode::Ln
            | Opcode::Exp
            | Opcode::ReduceSum
            | Opcode::ReduceMax
            | Opcode::ReduceMin
            | Opcode::RingEnqueue
            | Opcode::AssertContract => {
                if let Some(a) = live.get_mut(frame.in_a as usize) {
                    *a = true;
                }
            }
            Opcode::ArenaMutate => {
                if let Some(b) = live.get_mut(frame.in_b as usize) {
                    *b = true;
                }
            }
            Opcode::Add
            | Opcode::Sub
            | Opcode::Mul
            | Opcode::Div
            | Opcode::Min
            | Opcode::Max
            | Opcode::CmpGt
            | Opcode::CmpLt
            | Opcode::CmpEq
            | Opcode::MaskAnd
            | Opcode::MaskOr
            | Opcode::SelectZero
            | Opcode::FilterCompact
            | Opcode::Gather => {
                if let Some(a) = live.get_mut(frame.in_a as usize) {
                    *a = true;
                }
                if let Some(b) = live.get_mut(frame.in_b as usize) {
                    *b = true;
                }
            }
            Opcode::Fma => {
                let (in_b, in_c) = frame.decode_fma_bc();
                if let Some(a) = live.get_mut(frame.in_a as usize) {
                    *a = true;
                }
                if let Some(b) = live.get_mut(in_b as usize) {
                    *b = true;
                }
                if let Some(c) = live.get_mut(in_c as usize) {
                    *c = true;
                }
            }
        }
    }
    live
}

/// **Tier 0 — Zero-Copy Reference Columnar Interpreter (Golden Oracle)**.
///
/// Verifies the `.rl` program in $O(N)$ and evaluates each node sequentially using
/// zero-copy `Cow<'a, [f64]>` borrows on `ArenaInput` and `ArenaSlice`.
///
/// # Errors
/// Returns [`ExecError`] if verification, input binding, gather bounds, or contract assertions fail.
pub fn execute_tier0_interpreter(
    program: &RulProgram,
    inputs: &[&[f64]],
) -> Result<ExecutionReport, ExecError> {
    let t_verify = Instant::now();
    verify_program(program).map_err(ExecError::VerificationFailed)?;
    let verify_ns = u64::try_from(t_verify.elapsed().as_nanos()).unwrap_or(u64::MAX);

    let elements_processed = validate_inputs(program, inputs)?;

    let t_exec = Instant::now();
    let mut values: Vec<EvalValue<'_>> = Vec::with_capacity(program.nodes.len());
    let mut intermediate_bytes = 0usize;

    for (idx, frame) in program.nodes.iter().enumerate() {
        let node_idx = u32::try_from(idx).unwrap_or(u32::MAX);
        let val = eval_node_tier0(
            program,
            frame,
            node_idx,
            inputs,
            &values,
            &mut intermediate_bytes,
        )?;
        values.push(val);
    }

    let exec_ns = u64::try_from(t_exec.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let out_eval = values
        .get(program.output_node as usize)
        .cloned()
        .ok_or(ExecError::InvalidOperandState(program.output_node))?;
    let output = out_eval.into_runtime_value(program.output_node)?;

    Ok(ExecutionReport {
        output,
        telemetry: build_telemetry(verify_ns, exec_ns, elements_processed, intermediate_bytes),
    })
}

fn eval_node_tier0<'a>(
    program: &RulProgram,
    frame: &NodeFrame,
    node_idx: u32,
    inputs: &[&'a [f64]],
    values: &[EvalValue<'a>],
    intermediate_bytes: &mut usize,
) -> Result<EvalValue<'a>, ExecError> {
    match frame.opcode {
        Opcode::ArenaInput => {
            // Zero-copy borrow of the input arena slice!
            let slice = inputs
                .get(frame.in_a as usize)
                .copied()
                .ok_or(ExecError::MissingArenaInput(frame.in_a))?;
            Ok(EvalValue::ColumnF64(Cow::Borrowed(slice)))
        }
        Opcode::ConstF64 => Ok(EvalValue::Scalar(frame.decode_const_f64())),
        Opcode::ArenaSlice => {
            let proof = program
                .proofs
                .get(usize::from(frame.proof_idx.saturating_sub(1)))
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            let src = values
                .get(frame.in_a as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            match src {
                EvalValue::ColumnF64(Cow::Borrowed(col)) => {
                    let sub = col
                        .get(proof.bound_lo as usize..proof.bound_hi as usize)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    Ok(EvalValue::ColumnF64(Cow::Borrowed(sub)))
                }
                EvalValue::ColumnF64(Cow::Owned(col)) => {
                    let sub = col
                        .get(proof.bound_lo as usize..proof.bound_hi as usize)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    *intermediate_bytes = intermediate_bytes.saturating_add(sub.len() * 8);
                    Ok(EvalValue::ColumnF64(Cow::Owned(sub.to_vec())))
                }
                _ => Err(ExecError::InvalidOperandState(node_idx)),
            }
        }
        Opcode::ArenaMutate => {
            let src = values
                .get(frame.in_b as usize)
                .cloned()
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            Ok(src)
        }
        Opcode::Add | Opcode::Sub | Opcode::Mul | Opcode::Div | Opcode::Min | Opcode::Max => {
            let left = values
                .get(frame.in_a as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            let right = values
                .get(frame.in_b as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            eval_arith_tier0(frame.opcode, left, right, node_idx, intermediate_bytes)
        }
        Opcode::Fma => {
            let (in_b, in_c) = frame.decode_fma_bc();
            let va = values
                .get(frame.in_a as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            let vb = values
                .get(in_b as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            let vc = values
                .get(in_c as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            eval_fma_tier0(va, vb, vc, node_idx, intermediate_bytes)
        }
        Opcode::Abs | Opcode::Sqrt | Opcode::Floor | Opcode::Ceil | Opcode::Ln | Opcode::Exp => {
            let src = values
                .get(frame.in_a as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            eval_unary_math_tier0(frame.opcode, src, node_idx, intermediate_bytes)
        }
        Opcode::CmpGt | Opcode::CmpLt | Opcode::CmpEq => {
            let left = values
                .get(frame.in_a as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            let right = values
                .get(frame.in_b as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            let cmp = |x: f64, y: f64| -> bool {
                match frame.opcode {
                    Opcode::CmpGt => x > y,
                    Opcode::CmpLt => x < y,
                    _ => x == y,
                }
            };
            match (left, right) {
                (EvalValue::ColumnF64(col), EvalValue::Scalar(thr)) => {
                    *intermediate_bytes = intermediate_bytes.saturating_add(col.len());
                    let mask: Vec<bool> = col.iter().map(|&x| cmp(x, *thr)).collect();
                    Ok(EvalValue::ColumnMask(mask))
                }
                (EvalValue::Scalar(thr), EvalValue::ColumnF64(col)) => {
                    *intermediate_bytes = intermediate_bytes.saturating_add(col.len());
                    let mask: Vec<bool> = col.iter().map(|&y| cmp(*thr, y)).collect();
                    Ok(EvalValue::ColumnMask(mask))
                }
                (EvalValue::ColumnF64(a), EvalValue::ColumnF64(b)) => {
                    *intermediate_bytes = intermediate_bytes.saturating_add(a.len());
                    let mask: Vec<bool> =
                        a.iter().zip(b.iter()).map(|(&x, &y)| cmp(x, y)).collect();
                    Ok(EvalValue::ColumnMask(mask))
                }
                _ => Err(ExecError::InvalidOperandState(node_idx)),
            }
        }
        Opcode::MaskAnd | Opcode::MaskOr => {
            let left = values
                .get(frame.in_a as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            let right = values
                .get(frame.in_b as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            if let (EvalValue::ColumnMask(a), EvalValue::ColumnMask(b)) = (left, right) {
                *intermediate_bytes = intermediate_bytes.saturating_add(a.len());
                let out: Vec<bool> = a
                    .iter()
                    .zip(b.iter())
                    .map(|(&x, &y)| match frame.opcode {
                        Opcode::MaskAnd => x && y,
                        _ => x || y,
                    })
                    .collect();
                Ok(EvalValue::ColumnMask(out))
            } else {
                Err(ExecError::InvalidOperandState(node_idx))
            }
        }
        Opcode::SelectZero => {
            let mask_val = values
                .get(frame.in_a as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            let col_val = values
                .get(frame.in_b as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            match (mask_val, col_val) {
                (EvalValue::ColumnMask(mask), EvalValue::ColumnF64(col)) => {
                    *intermediate_bytes = intermediate_bytes.saturating_add(col.len() * 8);
                    let out: Vec<f64> = mask
                        .iter()
                        .zip(col.iter())
                        .map(|(&m, &v)| if m { v } else { 0.0 })
                        .collect();
                    Ok(EvalValue::ColumnF64(Cow::Owned(out)))
                }
                (EvalValue::ColumnMask(mask), EvalValue::Scalar(s)) => {
                    *intermediate_bytes = intermediate_bytes.saturating_add(mask.len() * 8);
                    let out: Vec<f64> = mask.iter().map(|&m| if m { *s } else { 0.0 }).collect();
                    Ok(EvalValue::ColumnF64(Cow::Owned(out)))
                }
                _ => Err(ExecError::InvalidOperandState(node_idx)),
            }
        }
        Opcode::FilterCompact => {
            let mask_val = values
                .get(frame.in_a as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            let col_val = values
                .get(frame.in_b as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            if let (EvalValue::ColumnMask(mask), EvalValue::ColumnF64(col)) = (mask_val, col_val) {
                let out: Vec<f64> = mask
                    .iter()
                    .zip(col.iter())
                    .filter_map(|(&m, &v)| if m { Some(v) } else { None })
                    .collect();
                *intermediate_bytes = intermediate_bytes.saturating_add(out.len() * 8);
                Ok(EvalValue::ColumnF64(Cow::Owned(out)))
            } else {
                Err(ExecError::InvalidOperandState(node_idx))
            }
        }
        Opcode::ReduceSum => {
            let src = values
                .get(frame.in_a as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            if let EvalValue::ColumnF64(col) = src {
                let sum = stencil_reduce_sum_f64x8(col);
                Ok(EvalValue::Scalar(sum))
            } else {
                Err(ExecError::InvalidOperandState(node_idx))
            }
        }
        Opcode::ReduceMax => {
            let src = values
                .get(frame.in_a as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            if let EvalValue::ColumnF64(col) = src {
                let max = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                Ok(EvalValue::Scalar(max))
            } else {
                Err(ExecError::InvalidOperandState(node_idx))
            }
        }
        Opcode::ReduceMin => {
            let src = values
                .get(frame.in_a as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            if let EvalValue::ColumnF64(col) = src {
                let min = col.iter().copied().fold(f64::INFINITY, f64::min);
                Ok(EvalValue::Scalar(min))
            } else {
                Err(ExecError::InvalidOperandState(node_idx))
            }
        }
        Opcode::Gather => {
            let proof = program
                .proofs
                .get(usize::from(frame.proof_idx.saturating_sub(1)))
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            let target_val = values
                .get(frame.in_a as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            let idx_val = values
                .get(frame.in_b as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            if let (EvalValue::ColumnF64(target), EvalValue::ColumnF64(indices)) =
                (target_val, idx_val)
            {
                let max_bound = proof.bound_hi as usize;
                let mut gathered = Vec::with_capacity(indices.len());
                for &raw_idx in indices.iter() {
                    if !raw_idx.is_finite()
                        || raw_idx < 0.0
                        || raw_idx >= (max_bound as f64)
                        || raw_idx.fract() != 0.0
                    {
                        return Err(ExecError::GatherIndexOutOfBounds {
                            node_idx,
                            index: raw_idx,
                            bound_hi: proof.bound_hi,
                        });
                    }
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "checked finite, >= 0.0, and < max_bound above"
                    )]
                    let idx_usize = raw_idx as usize;
                    let elem = target.get(idx_usize).copied().ok_or(
                        ExecError::GatherIndexOutOfBounds {
                            node_idx,
                            index: raw_idx,
                            bound_hi: proof.bound_hi,
                        },
                    )?;
                    gathered.push(elem);
                }
                *intermediate_bytes = intermediate_bytes.saturating_add(gathered.len() * 8);
                Ok(EvalValue::ColumnF64(Cow::Owned(gathered)))
            } else {
                Err(ExecError::InvalidOperandState(node_idx))
            }
        }
        Opcode::RingEnqueue => {
            let src = values
                .get(frame.in_a as usize)
                .cloned()
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            Ok(src)
        }
        Opcode::AssertContract => {
            let src = values
                .get(frame.in_a as usize)
                .cloned()
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            if let EvalValue::Scalar(val) = src {
                if !val.is_finite() {
                    return Err(ExecError::ContractAssertionFailed {
                        node_idx,
                        observed: val,
                    });
                }
            }
            Ok(src)
        }
    }
}

fn eval_arith_tier0<'a>(
    opcode: Opcode,
    left: &EvalValue<'a>,
    right: &EvalValue<'a>,
    node_idx: u32,
    intermediate_bytes: &mut usize,
) -> Result<EvalValue<'a>, ExecError> {
    let apply = |a: f64, b: f64| -> f64 {
        match opcode {
            Opcode::Add => a + b,
            Opcode::Sub => a - b,
            Opcode::Mul => a * b,
            Opcode::Div => a / b,
            Opcode::Min => {
                if a < b {
                    a
                } else {
                    b
                }
            }
            _ => {
                if a > b {
                    a
                } else {
                    b
                }
            }
        }
    };
    match (left, right) {
        (EvalValue::ColumnF64(a), EvalValue::ColumnF64(b)) => {
            *intermediate_bytes = intermediate_bytes.saturating_add(a.len() * 8);
            let out: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| apply(x, y)).collect();
            Ok(EvalValue::ColumnF64(Cow::Owned(out)))
        }
        (EvalValue::ColumnF64(a), EvalValue::Scalar(s)) => {
            *intermediate_bytes = intermediate_bytes.saturating_add(a.len() * 8);
            let out: Vec<f64> = a.iter().map(|&x| apply(x, *s)).collect();
            Ok(EvalValue::ColumnF64(Cow::Owned(out)))
        }
        (EvalValue::Scalar(s), EvalValue::ColumnF64(b)) => {
            *intermediate_bytes = intermediate_bytes.saturating_add(b.len() * 8);
            let out: Vec<f64> = b.iter().map(|&y| apply(*s, y)).collect();
            Ok(EvalValue::ColumnF64(Cow::Owned(out)))
        }
        (EvalValue::Scalar(a), EvalValue::Scalar(b)) => Ok(EvalValue::Scalar(apply(*a, *b))),
        _ => Err(ExecError::InvalidOperandState(node_idx)),
    }
}

fn eval_fma_tier0<'a>(
    va: &EvalValue<'a>,
    vb: &EvalValue<'a>,
    vc: &EvalValue<'a>,
    node_idx: u32,
    intermediate_bytes: &mut usize,
) -> Result<EvalValue<'a>, ExecError> {
    if let (EvalValue::Scalar(a), EvalValue::Scalar(b), EvalValue::Scalar(c)) = (va, vb, vc) {
        return Ok(EvalValue::Scalar(a.mul_add(*b, *c)));
    }
    let col_len = match (va, vb, vc) {
        (EvalValue::ColumnF64(col), _, _)
        | (_, EvalValue::ColumnF64(col), _)
        | (_, _, EvalValue::ColumnF64(col)) => col.len(),
        _ => return Err(ExecError::InvalidOperandState(node_idx)),
    };
    *intermediate_bytes = intermediate_bytes.saturating_add(col_len * 8);
    let mut out = Vec::with_capacity(col_len);
    for i in 0..col_len {
        let a = match va {
            EvalValue::ColumnF64(col) => {
                *col.get(i).ok_or(ExecError::InvalidOperandState(node_idx))?
            }
            EvalValue::Scalar(s) => *s,
            _ => return Err(ExecError::InvalidOperandState(node_idx)),
        };
        let b = match vb {
            EvalValue::ColumnF64(col) => {
                *col.get(i).ok_or(ExecError::InvalidOperandState(node_idx))?
            }
            EvalValue::Scalar(s) => *s,
            _ => return Err(ExecError::InvalidOperandState(node_idx)),
        };
        let c = match vc {
            EvalValue::ColumnF64(col) => {
                *col.get(i).ok_or(ExecError::InvalidOperandState(node_idx))?
            }
            EvalValue::Scalar(s) => *s,
            _ => return Err(ExecError::InvalidOperandState(node_idx)),
        };
        out.push(a.mul_add(b, c));
    }
    Ok(EvalValue::ColumnF64(Cow::Owned(out)))
}

fn eval_unary_math_tier0<'a>(
    opcode: Opcode,
    src: &EvalValue<'a>,
    node_idx: u32,
    intermediate_bytes: &mut usize,
) -> Result<EvalValue<'a>, ExecError> {
    let apply = |x: f64| -> f64 {
        match opcode {
            Opcode::Abs => x.abs(),
            Opcode::Sqrt => x.sqrt(),
            Opcode::Floor => x.floor(),
            Opcode::Ceil => x.ceil(),
            Opcode::Ln => x.ln(),
            _ => x.exp(),
        }
    };
    match src {
        EvalValue::ColumnF64(col) => {
            *intermediate_bytes = intermediate_bytes.saturating_add(col.len() * 8);
            let out: Vec<f64> = col.iter().map(|&x| apply(x)).collect();
            Ok(EvalValue::ColumnF64(Cow::Owned(out)))
        }
        EvalValue::Scalar(s) => Ok(EvalValue::Scalar(apply(*s))),
        _ => Err(ExecError::InvalidOperandState(node_idx)),
    }
}

/// Pre-baked 8-lane SIMD reduction stencil (`f64x8` unrolled accumulator to break
/// floating-point latency chains on AVX2 / AVX-512 / NEON).
#[must_use]
pub fn stencil_reduce_sum_f64x8(slice: &[f64]) -> f64 {
    let mut acc = [0.0f64; 8];
    let (chunks, remainder) = slice.as_chunks::<8>();
    for chunk in chunks {
        for (lane, &val) in acc.iter_mut().zip(chunk.iter()) {
            *lane += val;
        }
    }
    let mut total = (acc[0] + acc[1]) + (acc[2] + acc[3]) + (acc[4] + acc[5]) + (acc[6] + acc[7]);
    for &val in remainder {
        total += val;
    }
    total
}

/// **Tier 1 — Dead-Node-Eliminated Vector Stencil Engine**.
///
/// Computes exact backward reachability from `output_node` so dead nodes are never
/// executed, then evaluates live nodes using zero-copy borrows and 8-lane stencils.
///
/// # Errors
/// Returns [`ExecError`] if verification or execution fails.
pub fn execute_tier1_stencil(
    program: &RulProgram,
    inputs: &[&[f64]],
) -> Result<ExecutionReport, ExecError> {
    let t_verify = Instant::now();
    verify_program(program).map_err(ExecError::VerificationFailed)?;
    let verify_ns = u64::try_from(t_verify.elapsed().as_nanos()).unwrap_or(u64::MAX);

    let elements_processed = validate_inputs(program, inputs)?;

    let t_exec = Instant::now();
    let live = compute_live_nodes(program);
    let mut values: Vec<EvalValue<'_>> = Vec::with_capacity(program.nodes.len());
    let mut intermediate_bytes = 0usize;

    for (idx, frame) in program.nodes.iter().enumerate() {
        if !live.get(idx).copied().unwrap_or(false) {
            values.push(EvalValue::Unused);
            continue;
        }
        let node_idx = u32::try_from(idx).unwrap_or(u32::MAX);
        let val = eval_node_tier0(
            program,
            frame,
            node_idx,
            inputs,
            &values,
            &mut intermediate_bytes,
        )?;
        values.push(val);
    }

    let exec_ns = u64::try_from(t_exec.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let out_eval = values
        .get(program.output_node as usize)
        .cloned()
        .ok_or(ExecError::InvalidOperandState(program.output_node))?;
    let output = out_eval.into_runtime_value(program.output_node)?;

    Ok(ExecutionReport {
        output,
        telemetry: build_telemetry(verify_ns, exec_ns, elements_processed, intermediate_bytes),
    })
}

/// **Tier 2 — Exact-DAG L1-Cache Tile Engine + Auto-MultiCore**.
///
/// Evaluates arbitrary elementwise-to-reduction `.rl` DAGs by partitioning the iteration
/// space into `1,024`-element (`8 KB`) L1-cache tiles with **0 bytes of intermediate heap
/// allocation**, tracing exact `(in_a, in_b)` edges from `output_node` so constant order
/// and dead nodes never cause miscompilation. When `ScheduleHint.policy == ExecPolicy::MultiCore`,
/// tiles are evaluated in parallel across CPU cores via `std::thread::scope`.
///
/// # Errors
/// Returns [`ExecError`] if static verification or execution fails.
pub fn execute_tier2_fused(
    program: &RulProgram,
    inputs: &[&[f64]],
) -> Result<ExecutionReport, ExecError> {
    execute_tier2_with_zonemap(program, inputs, None)
}

/// Executes **Tier 2** with an optional precomputed [`ArenaZoneMap`] on `Arena(0)` for
/// sub-arena block skipping.
///
/// # Errors
/// Returns [`ExecError`] if static verification or execution fails.
pub fn execute_tier2_with_zonemap(
    program: &RulProgram,
    inputs: &[&[f64]],
    zonemap_a0: Option<&ArenaZoneMap>,
) -> Result<ExecutionReport, ExecError> {
    let t_verify = Instant::now();
    verify_program(program).map_err(ExecError::VerificationFailed)?;
    let verify_ns = u64::try_from(t_verify.elapsed().as_nanos()).unwrap_or(u64::MAX);

    let elements_processed = validate_inputs(program, inputs)?;

    let t_exec = Instant::now();
    let live = compute_live_nodes(program);

    // Check if the live DAG is eligible for Zero-Allocation L1-Cache Tile Fusion.
    if let Some(plan) = build_l1_tile_plan(program, &live) {
        let scalar_out = execute_l1_tile_plan(program, inputs, &plan, zonemap_a0)?;
        let exec_ns = u64::try_from(t_exec.elapsed().as_nanos()).unwrap_or(u64::MAX);

        return Ok(ExecutionReport {
            output: RuntimeValue::Scalar(scalar_out),
            telemetry: build_telemetry(verify_ns, exec_ns, elements_processed, 0),
        });
    }

    // Fallback to dead-node-eliminated Tier 1 for pipelines containing FilterCompact or Gather.
    execute_tier1_stencil(program, inputs)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReductionKind {
    Sum,
    Max,
    Min,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TileNodePhase {
    Dead,
    Const,
    Column,
    Reduce,
    PostReduceScalar,
}

#[derive(Debug, Clone, Copy)]
struct TileReductionSpec {
    reduce_node_idx: usize,
    reduce_input_idx: usize,
    kind: ReductionKind,
}

/// Exact-DAG verified plan for L1-Cache Tile Execution (supporting single or multiple reductions
/// plus post-reduction scalar DAG operations in a single zero-allocation pass).
#[derive(Debug, Clone)]
struct L1TilePlan {
    phases: Vec<TileNodePhase>,
    reductions: Vec<TileReductionSpec>,
    post_reduce_nodes: Vec<usize>,
    domain_start: usize,
    domain_len: usize,
    multicore: bool,
    /// Optional exact-edge fast path for single-column `SelectZero(CmpGt(col, thr), expr)`.
    fast_predicated: Option<ExactPredicatedKernel>,
}

#[derive(Debug, Clone, Copy)]
enum ExactValueExpr {
    Identity,
    Square,
    MulConst(f64),
    DivConst(f64),
    AddConst(f64),
}

#[derive(Debug, Clone, Copy)]
struct ExactPredicatedKernel {
    arena_id: usize,
    threshold: f64,
    expr: ExactValueExpr,
}

fn build_l1_tile_plan(program: &RulProgram, live: &[bool]) -> Option<L1TilePlan> {
    let out_idx = program.output_node as usize;
    if !live.get(out_idx).copied().unwrap_or(false) {
        return None;
    }
    let out_frame = program.nodes.get(out_idx)?;
    if out_frame.type_lane.elem != ElemType::F64 {
        return None;
    }

    let mut phases = vec![TileNodePhase::Dead; program.nodes.len()];
    let mut reductions = Vec::new();
    let mut post_reduce_nodes = Vec::new();
    let mut arena_len: Option<usize> = None;
    let mut slice_window: Option<(usize, usize)> = None;
    let mut multicore = false;

    for (idx, frame) in program.nodes.iter().enumerate() {
        if !live.get(idx).copied().unwrap_or(false) {
            continue;
        }
        if frame.schedule.policy == ExecPolicy::MultiCore {
            multicore = true;
        }
        let phase = match frame.opcode {
            Opcode::ConstF64 => TileNodePhase::Const,
            Opcode::ArenaInput => {
                let desc = program.arenas.get(frame.in_a as usize)?;
                let len = desc.element_count as usize;
                if let Some(prev_len) = arena_len {
                    if prev_len != len {
                        return None;
                    }
                } else {
                    arena_len = Some(len);
                }
                TileNodePhase::Column
            }
            Opcode::ArenaSlice => {
                if phases.get(frame.in_a as usize).copied()? != TileNodePhase::Column {
                    return None;
                }
                let proof = program
                    .proofs
                    .get(usize::from(frame.proof_idx.saturating_sub(1)))?;
                let start = proof.bound_lo as usize;
                let len = (proof.bound_hi - proof.bound_lo) as usize;
                if let Some((prev_start, prev_len)) = slice_window {
                    if prev_start != start || prev_len != len {
                        return None;
                    }
                } else {
                    slice_window = Some((start, len));
                }
                TileNodePhase::Column
            }
            Opcode::ArenaMutate => {
                let pb = phases.get(frame.in_b as usize).copied()?;
                if !matches!(pb, TileNodePhase::Const | TileNodePhase::Column) {
                    return None;
                }
                TileNodePhase::Column
            }
            Opcode::Add | Opcode::Sub | Opcode::Mul | Opcode::Div | Opcode::Min | Opcode::Max => {
                let pa = phases.get(frame.in_a as usize).copied()?;
                let pb = phases.get(frame.in_b as usize).copied()?;
                if pa == TileNodePhase::Column || pb == TileNodePhase::Column {
                    if !matches!(pa, TileNodePhase::Const | TileNodePhase::Column)
                        || !matches!(pb, TileNodePhase::Const | TileNodePhase::Column)
                    {
                        return None;
                    }
                    TileNodePhase::Column
                } else if matches!(
                    pa,
                    TileNodePhase::Const | TileNodePhase::Reduce | TileNodePhase::PostReduceScalar
                ) && matches!(
                    pb,
                    TileNodePhase::Const | TileNodePhase::Reduce | TileNodePhase::PostReduceScalar
                ) {
                    post_reduce_nodes.push(idx);
                    TileNodePhase::PostReduceScalar
                } else {
                    return None;
                }
            }
            Opcode::Fma => {
                let (in_b, in_c) = frame.decode_fma_bc();
                let pa = phases.get(frame.in_a as usize).copied()?;
                let pb = phases.get(in_b as usize).copied()?;
                let pc = phases.get(in_c as usize).copied()?;
                if pa == TileNodePhase::Column
                    || pb == TileNodePhase::Column
                    || pc == TileNodePhase::Column
                {
                    if !matches!(pa, TileNodePhase::Const | TileNodePhase::Column)
                        || !matches!(pb, TileNodePhase::Const | TileNodePhase::Column)
                        || !matches!(pc, TileNodePhase::Const | TileNodePhase::Column)
                    {
                        return None;
                    }
                    TileNodePhase::Column
                } else if matches!(
                    pa,
                    TileNodePhase::Const | TileNodePhase::Reduce | TileNodePhase::PostReduceScalar
                ) && matches!(
                    pb,
                    TileNodePhase::Const | TileNodePhase::Reduce | TileNodePhase::PostReduceScalar
                ) && matches!(
                    pc,
                    TileNodePhase::Const | TileNodePhase::Reduce | TileNodePhase::PostReduceScalar
                ) {
                    post_reduce_nodes.push(idx);
                    TileNodePhase::PostReduceScalar
                } else {
                    return None;
                }
            }
            Opcode::Abs
            | Opcode::Sqrt
            | Opcode::Floor
            | Opcode::Ceil
            | Opcode::Ln
            | Opcode::Exp => {
                let pa = phases.get(frame.in_a as usize).copied()?;
                match pa {
                    TileNodePhase::Column => TileNodePhase::Column,
                    TileNodePhase::Const
                    | TileNodePhase::Reduce
                    | TileNodePhase::PostReduceScalar => {
                        post_reduce_nodes.push(idx);
                        TileNodePhase::PostReduceScalar
                    }
                    TileNodePhase::Dead => return None,
                }
            }
            Opcode::CmpGt
            | Opcode::CmpLt
            | Opcode::CmpEq
            | Opcode::MaskAnd
            | Opcode::MaskOr
            | Opcode::SelectZero => {
                let pa = phases.get(frame.in_a as usize).copied()?;
                let pb = phases.get(frame.in_b as usize).copied()?;
                if (pa != TileNodePhase::Column && pb != TileNodePhase::Column)
                    || !matches!(pa, TileNodePhase::Const | TileNodePhase::Column)
                    || !matches!(pb, TileNodePhase::Const | TileNodePhase::Column)
                {
                    return None;
                }
                TileNodePhase::Column
            }
            Opcode::ReduceSum | Opcode::ReduceMax | Opcode::ReduceMin => {
                let pa = phases.get(frame.in_a as usize).copied()?;
                if pa != TileNodePhase::Column {
                    return None;
                }
                let kind = match frame.opcode {
                    Opcode::ReduceSum => ReductionKind::Sum,
                    Opcode::ReduceMax => ReductionKind::Max,
                    _ => ReductionKind::Min,
                };
                reductions.push(TileReductionSpec {
                    reduce_node_idx: idx,
                    reduce_input_idx: frame.in_a as usize,
                    kind,
                });
                TileNodePhase::Reduce
            }
            Opcode::AssertContract | Opcode::RingEnqueue => {
                let pa = phases.get(frame.in_a as usize).copied()?;
                if !matches!(
                    pa,
                    TileNodePhase::Const | TileNodePhase::Reduce | TileNodePhase::PostReduceScalar
                ) {
                    return None;
                }
                post_reduce_nodes.push(idx);
                TileNodePhase::PostReduceScalar
            }
            Opcode::FilterCompact | Opcode::Gather => return None,
        };
        if let Some(slot) = phases.get_mut(idx) {
            *slot = phase;
        }
    }

    if reductions.is_empty()
        || !matches!(
            phases.get(out_idx).copied()?,
            TileNodePhase::Reduce | TileNodePhase::PostReduceScalar
        )
    {
        return None;
    }

    let (domain_start, domain_len) = slice_window.unwrap_or((0, arena_len?));

    let only_wrapper_epilogue = post_reduce_nodes.iter().all(|&i| {
        matches!(
            program.nodes.get(i).map(|f| f.opcode),
            Some(Opcode::AssertContract | Opcode::RingEnqueue)
        )
    });
    let fast_predicated = if reductions.len() == 1 && domain_start == 0 && only_wrapper_epilogue {
        let first_red = reductions.first()?;
        if matches!(first_red.kind, ReductionKind::Sum) {
            trace_exact_predicated_dag(program, first_red.reduce_input_idx)
        } else {
            None
        }
    } else {
        None
    };

    Some(L1TilePlan {
        phases,
        reductions,
        post_reduce_nodes,
        domain_start,
        domain_len,
        multicore,
        fast_predicated,
    })
}

/// Traces exact `(in_a, in_b)` edges from `reduce_input_idx` to check if the live sub-DAG
/// is `SelectZero(CmpGt(ArenaInput(a), ConstF64(thr)), expr(ArenaInput(a)))`.
///
/// Unlike a flat list scan, this follows exact SSA edges, so dead nodes or reordered
/// constants cannot affect the matched kernel.
fn trace_exact_predicated_dag(
    program: &RulProgram,
    reduce_input_idx: usize,
) -> Option<ExactPredicatedKernel> {
    let sel_node = program.nodes.get(reduce_input_idx)?;
    if sel_node.opcode != Opcode::SelectZero {
        return None;
    }

    let cmp_node = program.nodes.get(sel_node.in_a as usize)?;
    if cmp_node.opcode != Opcode::CmpGt {
        return None;
    }

    let col_idx = cmp_node.in_a as usize;
    let col_node = program.nodes.get(col_idx)?;
    if col_node.opcode != Opcode::ArenaInput {
        return None;
    }
    let arena_id = col_node.in_a as usize;

    let thr_node = program.nodes.get(cmp_node.in_b as usize)?;
    if thr_node.opcode != Opcode::ConstF64 {
        return None;
    }
    let threshold = thr_node.decode_const_f64();

    let val_idx = sel_node.in_b as usize;
    let expr = if val_idx == col_idx {
        ExactValueExpr::Identity
    } else {
        let val_node = program.nodes.get(val_idx)?;
        match val_node.opcode {
            Opcode::Mul
                if val_node.in_a as usize == col_idx && val_node.in_b as usize == col_idx =>
            {
                ExactValueExpr::Square
            }
            Opcode::Mul if val_node.in_a as usize == col_idx => {
                let c = program.nodes.get(val_node.in_b as usize)?;
                if c.opcode != Opcode::ConstF64 {
                    return None;
                }
                ExactValueExpr::MulConst(c.decode_const_f64())
            }
            Opcode::Mul if val_node.in_b as usize == col_idx => {
                let c = program.nodes.get(val_node.in_a as usize)?;
                if c.opcode != Opcode::ConstF64 {
                    return None;
                }
                ExactValueExpr::MulConst(c.decode_const_f64())
            }
            Opcode::Div if val_node.in_a as usize == col_idx => {
                let c = program.nodes.get(val_node.in_b as usize)?;
                if c.opcode != Opcode::ConstF64 {
                    return None;
                }
                ExactValueExpr::DivConst(c.decode_const_f64())
            }
            Opcode::Add if val_node.in_a as usize == col_idx => {
                let c = program.nodes.get(val_node.in_b as usize)?;
                if c.opcode != Opcode::ConstF64 {
                    return None;
                }
                ExactValueExpr::AddConst(c.decode_const_f64())
            }
            _ => return None,
        }
    };

    Some(ExactPredicatedKernel {
        arena_id,
        threshold,
        expr,
    })
}

fn execute_l1_tile_plan(
    program: &RulProgram,
    inputs: &[&[f64]],
    plan: &L1TilePlan,
    zonemap_a0: Option<&ArenaZoneMap>,
) -> Result<f64, ExecError> {
    // 1. If exact-edge predicated kernel matches, run the specialized 8-lane SIMD kernel
    //    (with optional Zone-Map skipping and MultiCore parallel dispatch).
    let reduced_vals = if let Some(fast) = plan.fast_predicated {
        let slice = inputs
            .get(fast.arena_id)
            .copied()
            .ok_or(ExecError::MissingArenaInput(
                u32::try_from(fast.arena_id).unwrap_or(0),
            ))?;
        let zm = if fast.arena_id == 0 { zonemap_a0 } else { None };
        let val = if plan.multicore && slice.len() >= L1_TILE_LEN * 16 {
            run_fast_predicated_multicore(slice, fast, zm)
        } else {
            run_fast_predicated_range(slice, 0, fast, zm)
        };
        vec![val]
    } else if plan.multicore && plan.domain_len >= L1_TILE_LEN * 16 {
        // 2. General L1-Cache Tile Evaluator (`[f64; 1024]` stack/L1 buffers, 0 intermediate column bytes)
        eval_general_tiles_multicore(program, inputs, plan)?
    } else {
        eval_general_tiles_range(
            program,
            inputs,
            plan,
            plan.domain_start,
            plan.domain_start + plan.domain_len,
        )?
    };

    eval_post_reduce_scalars(program, plan, &reduced_vals)
}

fn eval_post_reduce_scalars(
    program: &RulProgram,
    plan: &L1TilePlan,
    reduced_vals: &[f64],
) -> Result<f64, ExecError> {
    let mut scalar_vals = vec![0.0f64; program.nodes.len()];
    for (idx, frame) in program.nodes.iter().enumerate() {
        if plan.phases.get(idx).copied() == Some(TileNodePhase::Const) {
            if let Some(slot) = scalar_vals.get_mut(idx) {
                *slot = frame.decode_const_f64();
            }
        }
    }
    for (spec, &val) in plan.reductions.iter().zip(reduced_vals.iter()) {
        if let Some(slot) = scalar_vals.get_mut(spec.reduce_node_idx) {
            *slot = val;
        }
    }

    for &idx in &plan.post_reduce_nodes {
        let node_idx = u32::try_from(idx).unwrap_or(0);
        let frame = program
            .nodes
            .get(idx)
            .ok_or(ExecError::InvalidOperandState(node_idx))?;
        let a = *scalar_vals
            .get(frame.in_a as usize)
            .ok_or(ExecError::InvalidOperandState(node_idx))?;
        let b = scalar_vals.get(frame.in_b as usize).copied().unwrap_or(0.0);
        let val = match frame.opcode {
            Opcode::Add => a + b,
            Opcode::Sub => a - b,
            Opcode::Mul => a * b,
            Opcode::Div => a / b,
            Opcode::Min => a.min(b),
            Opcode::Max => a.max(b),
            Opcode::Fma => {
                let (in_b, in_c) = frame.decode_fma_bc();
                let b_val = *scalar_vals
                    .get(in_b as usize)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                let c_val = *scalar_vals
                    .get(in_c as usize)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                a.mul_add(b_val, c_val)
            }
            Opcode::Abs => a.abs(),
            Opcode::Sqrt => a.sqrt(),
            Opcode::Floor => a.floor(),
            Opcode::Ceil => a.ceil(),
            Opcode::Ln => a.ln(),
            Opcode::Exp => a.exp(),
            Opcode::RingEnqueue => a,
            Opcode::AssertContract => {
                if !a.is_finite() {
                    return Err(ExecError::ContractAssertionFailed {
                        node_idx,
                        observed: a,
                    });
                }
                a
            }
            _ => return Err(ExecError::InvalidOperandState(node_idx)),
        };
        if let Some(slot) = scalar_vals.get_mut(idx) {
            *slot = val;
        }
    }

    scalar_vals
        .get(program.output_node as usize)
        .copied()
        .ok_or(ExecError::InvalidOperandState(program.output_node))
}

fn run_fast_predicated_multicore(
    slice: &[f64],
    fast: ExactPredicatedKernel,
    zonemap: Option<&ArenaZoneMap>,
) -> f64 {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, 8))
        .unwrap_or(4);
    let total_tiles = slice.len().div_ceil(L1_TILE_LEN);
    let tiles_per_thread = total_tiles.div_ceil(threads);
    let chunk_elems = tiles_per_thread * L1_TILE_LEN;

    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(threads);
        for (thread_idx, sub_slice) in slice.chunks(chunk_elems).enumerate() {
            let base_tile = thread_idx * tiles_per_thread;
            handles.push(
                s.spawn(move || run_fast_predicated_range(sub_slice, base_tile, fast, zonemap)),
            );
        }
        let mut total = 0.0f64;
        for h in handles {
            if let Ok(part) = h.join() {
                total += part;
            }
        }
        total
    })
}

#[inline(never)]
fn run_fast_predicated_range(
    slice: &[f64],
    base_tile_idx: usize,
    fast: ExactPredicatedKernel,
    zonemap: Option<&ArenaZoneMap>,
) -> f64 {
    if let Some(zm) = zonemap {
        let mut total = 0.0f64;
        for (local_tile_idx, tile) in slice.chunks(L1_TILE_LEN).enumerate() {
            let global_tile_idx = base_tile_idx + local_tile_idx;
            if let Some(&(min_v, max_v)) = zm.tile_bounds.get(global_tile_idx) {
                // Zone-Map Skip: entire 1,024-element block is <= threshold!
                if max_v <= fast.threshold {
                    continue;
                }
                // Zone-Map All-Pass: entire 1,024-element block is > threshold (no comparison needed!)
                if min_v > fast.threshold {
                    total += eval_unconditional_tile_f64x8(tile, fast.expr);
                    continue;
                }
            }
            total += eval_predicated_slice_f64x8(tile, fast.threshold, fast.expr);
        }
        return total;
    }

    eval_predicated_slice_f64x8(slice, fast.threshold, fast.expr)
}

#[inline(always)]
fn eval_unconditional_tile_f64x8(tile: &[f64], expr: ExactValueExpr) -> f64 {
    let mut acc = [0.0f64; 8];
    let (chunks, remainder) = tile.as_chunks::<8>();
    match expr {
        ExactValueExpr::Square => {
            for chunk in chunks {
                for (lane, &x) in acc.iter_mut().zip(chunk.iter()) {
                    *lane += x * x;
                }
            }
            let mut sum =
                (acc[0] + acc[1]) + (acc[2] + acc[3]) + (acc[4] + acc[5]) + (acc[6] + acc[7]);
            for &x in remainder {
                sum += x * x;
            }
            sum
        }
        ExactValueExpr::MulConst(k) => {
            for chunk in chunks {
                for (lane, &x) in acc.iter_mut().zip(chunk.iter()) {
                    *lane += x * k;
                }
            }
            let mut sum =
                (acc[0] + acc[1]) + (acc[2] + acc[3]) + (acc[4] + acc[5]) + (acc[6] + acc[7]);
            for &x in remainder {
                sum += x * k;
            }
            sum
        }
        ExactValueExpr::DivConst(k) => {
            for chunk in chunks {
                for (lane, &x) in acc.iter_mut().zip(chunk.iter()) {
                    *lane += x / k;
                }
            }
            let mut sum =
                (acc[0] + acc[1]) + (acc[2] + acc[3]) + (acc[4] + acc[5]) + (acc[6] + acc[7]);
            for &x in remainder {
                sum += x / k;
            }
            sum
        }
        ExactValueExpr::AddConst(k) => {
            for chunk in chunks {
                for (lane, &x) in acc.iter_mut().zip(chunk.iter()) {
                    *lane += x + k;
                }
            }
            let mut sum =
                (acc[0] + acc[1]) + (acc[2] + acc[3]) + (acc[4] + acc[5]) + (acc[6] + acc[7]);
            for &x in remainder {
                sum += x + k;
            }
            sum
        }
        ExactValueExpr::Identity => stencil_reduce_sum_f64x8(tile),
    }
}

#[inline(always)]
fn eval_predicated_slice_f64x8(slice: &[f64], threshold: f64, expr: ExactValueExpr) -> f64 {
    let mut acc = [0.0f64; 8];
    let (chunks, remainder) = slice.as_chunks::<8>();
    match expr {
        ExactValueExpr::Square => {
            for chunk in chunks {
                for (lane, &x) in acc.iter_mut().zip(chunk.iter()) {
                    *lane += if x > threshold { x * x } else { 0.0 };
                }
            }
            let mut sum =
                (acc[0] + acc[1]) + (acc[2] + acc[3]) + (acc[4] + acc[5]) + (acc[6] + acc[7]);
            for &x in remainder {
                if x > threshold {
                    sum += x * x;
                }
            }
            sum
        }
        ExactValueExpr::MulConst(k) => {
            for chunk in chunks {
                for (lane, &x) in acc.iter_mut().zip(chunk.iter()) {
                    *lane += if x > threshold { x * k } else { 0.0 };
                }
            }
            let mut sum =
                (acc[0] + acc[1]) + (acc[2] + acc[3]) + (acc[4] + acc[5]) + (acc[6] + acc[7]);
            for &x in remainder {
                if x > threshold {
                    sum += x * k;
                }
            }
            sum
        }
        ExactValueExpr::DivConst(k) => {
            for chunk in chunks {
                for (lane, &x) in acc.iter_mut().zip(chunk.iter()) {
                    *lane += if x > threshold { x / k } else { 0.0 };
                }
            }
            let mut sum =
                (acc[0] + acc[1]) + (acc[2] + acc[3]) + (acc[4] + acc[5]) + (acc[6] + acc[7]);
            for &x in remainder {
                if x > threshold {
                    sum += x / k;
                }
            }
            sum
        }
        ExactValueExpr::AddConst(k) => {
            for chunk in chunks {
                for (lane, &x) in acc.iter_mut().zip(chunk.iter()) {
                    *lane += if x > threshold { x + k } else { 0.0 };
                }
            }
            let mut sum =
                (acc[0] + acc[1]) + (acc[2] + acc[3]) + (acc[4] + acc[5]) + (acc[6] + acc[7]);
            for &x in remainder {
                if x > threshold {
                    sum += x + k;
                }
            }
            sum
        }
        ExactValueExpr::Identity => {
            for chunk in chunks {
                for (lane, &x) in acc.iter_mut().zip(chunk.iter()) {
                    *lane += if x > threshold { x } else { 0.0 };
                }
            }
            let mut sum =
                (acc[0] + acc[1]) + (acc[2] + acc[3]) + (acc[4] + acc[5]) + (acc[6] + acc[7]);
            for &x in remainder {
                if x > threshold {
                    sum += x;
                }
            }
            sum
        }
    }
}

fn eval_general_tiles_multicore(
    program: &RulProgram,
    inputs: &[&[f64]],
    plan: &L1TilePlan,
) -> Result<Vec<f64>, ExecError> {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, 8))
        .unwrap_or(4);
    let total_tiles = plan.domain_len.div_ceil(L1_TILE_LEN);
    let tiles_per_thread = total_tiles.div_ceil(threads);
    let chunk_elems = tiles_per_thread * L1_TILE_LEN;
    let domain_end = plan.domain_start + plan.domain_len;

    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(threads);
        let mut start = plan.domain_start;
        while start < domain_end {
            let end = (start + chunk_elems).min(domain_end);
            handles
                .push(s.spawn(move || eval_general_tiles_range(program, inputs, plan, start, end)));
            start = end;
        }
        let mut accs: Vec<f64> = plan
            .reductions
            .iter()
            .map(|r| match r.kind {
                ReductionKind::Sum => 0.0f64,
                ReductionKind::Max => f64::NEG_INFINITY,
                ReductionKind::Min => f64::INFINITY,
            })
            .collect();
        for h in handles {
            let parts = h
                .join()
                .map_err(|_panic| ExecError::InvalidOperandState(program.output_node))??;
            for ((acc, spec), &part) in accs
                .iter_mut()
                .zip(plan.reductions.iter())
                .zip(parts.iter())
            {
                *acc = match spec.kind {
                    ReductionKind::Sum => *acc + part,
                    ReductionKind::Max => acc.max(part),
                    ReductionKind::Min => acc.min(part),
                };
            }
        }
        Ok(accs)
    })
}

/// Evaluates an arbitrary elementwise DAG over `[range_start..range_end)` in `1,024`-element
/// L1-cache tiles using reusable tile registers (`0` per-tile heap allocations).
fn eval_general_tiles_range(
    program: &RulProgram,
    inputs: &[&[f64]],
    plan: &L1TilePlan,
    range_start: usize,
    range_end: usize,
) -> Result<Vec<f64>, ExecError> {
    let node_count = program.nodes.len();
    let mut f64_regs: Vec<[f64; L1_TILE_LEN]> = vec![[0.0; L1_TILE_LEN]; node_count];
    let mut mask_regs: Vec<[bool; L1_TILE_LEN]> = vec![[false; L1_TILE_LEN]; node_count];
    let mut scalars: Vec<f64> = vec![0.0; node_count];
    let mut is_scalar: Vec<bool> = vec![false; node_count];

    for (idx, frame) in program.nodes.iter().enumerate() {
        if plan.phases.get(idx).copied() == Some(TileNodePhase::Const) {
            if let Some(s) = scalars.get_mut(idx) {
                *s = frame.decode_const_f64();
            }
            if let Some(flag) = is_scalar.get_mut(idx) {
                *flag = true;
            }
        }
    }

    let mut global_accs: Vec<f64> = plan
        .reductions
        .iter()
        .map(|r| match r.kind {
            ReductionKind::Sum => 0.0f64,
            ReductionKind::Max => f64::NEG_INFINITY,
            ReductionKind::Min => f64::INFINITY,
        })
        .collect();

    let mut tile_start = range_start;
    while tile_start < range_end {
        let tile_end = (tile_start + L1_TILE_LEN).min(range_end);
        let tile_len = tile_end - tile_start;

        for (idx, frame) in program.nodes.iter().enumerate() {
            if plan.phases.get(idx).copied() != Some(TileNodePhase::Column) {
                continue;
            }
            let node_idx = u32::try_from(idx).unwrap_or(0);
            match frame.opcode {
                Opcode::ConstF64 => {}
                Opcode::ArenaInput => {
                    let col = inputs
                        .get(frame.in_a as usize)
                        .copied()
                        .ok_or(ExecError::MissingArenaInput(frame.in_a))?;
                    let sub = col
                        .get(tile_start..tile_end)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let dst = f64_regs
                        .get_mut(idx)
                        .and_then(|r| r.get_mut(0..tile_len))
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    dst.copy_from_slice(sub);
                }
                Opcode::ArenaSlice => {
                    let src_idx = frame.in_a as usize;
                    if src_idx >= idx {
                        return Err(ExecError::InvalidOperandState(node_idx));
                    }
                    let (left, right) = f64_regs.split_at_mut(idx);
                    let src = left
                        .get(src_idx)
                        .and_then(|r| r.get(0..tile_len))
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let dst = right
                        .first_mut()
                        .and_then(|r| r.get_mut(0..tile_len))
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    dst.copy_from_slice(src);
                }
                Opcode::ArenaMutate => {
                    let src_idx = frame.in_b as usize;
                    if src_idx >= idx {
                        return Err(ExecError::InvalidOperandState(node_idx));
                    }
                    let (left, right) = f64_regs.split_at_mut(idx);
                    let src = left
                        .get(src_idx)
                        .and_then(|r| r.get(0..tile_len))
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let dst = right
                        .first_mut()
                        .and_then(|r| r.get_mut(0..tile_len))
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    dst.copy_from_slice(src);
                }
                Opcode::Add
                | Opcode::Sub
                | Opcode::Mul
                | Opcode::Div
                | Opcode::Min
                | Opcode::Max => {
                    eval_tile_arith(
                        frame.opcode,
                        idx,
                        frame.in_a as usize,
                        frame.in_b as usize,
                        tile_len,
                        &mut f64_regs,
                        &scalars,
                        &is_scalar,
                    )
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                }
                Opcode::Fma => {
                    let (in_b, in_c) = frame.decode_fma_bc();
                    eval_tile_fma(
                        idx,
                        frame.in_a as usize,
                        in_b as usize,
                        in_c as usize,
                        tile_len,
                        &mut f64_regs,
                        &scalars,
                        &is_scalar,
                    )
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                }
                Opcode::Abs
                | Opcode::Sqrt
                | Opcode::Floor
                | Opcode::Ceil
                | Opcode::Ln
                | Opcode::Exp => {
                    eval_tile_unary_math(
                        frame.opcode,
                        idx,
                        frame.in_a as usize,
                        tile_len,
                        &mut f64_regs,
                        &scalars,
                        &is_scalar,
                    )
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                }
                Opcode::CmpGt | Opcode::CmpLt | Opcode::CmpEq => {
                    eval_tile_cmp(
                        frame.opcode,
                        idx,
                        frame.in_a as usize,
                        frame.in_b as usize,
                        tile_len,
                        &f64_regs,
                        &mut mask_regs,
                        &scalars,
                        &is_scalar,
                    )
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                }
                Opcode::MaskAnd | Opcode::MaskOr => {
                    eval_tile_mask_logic(
                        frame.opcode,
                        idx,
                        frame.in_a as usize,
                        frame.in_b as usize,
                        tile_len,
                        &mut mask_regs,
                    )
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                }
                Opcode::SelectZero => {
                    let m_idx = frame.in_a as usize;
                    let v_idx = frame.in_b as usize;
                    if v_idx >= idx {
                        return Err(ExecError::InvalidOperandState(node_idx));
                    }
                    let mask = mask_regs
                        .get(m_idx)
                        .and_then(|r| r.get(0..tile_len))
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let v_is_scalar = is_scalar.get(v_idx).copied().unwrap_or(false);
                    if v_is_scalar {
                        let s = scalars
                            .get(v_idx)
                            .copied()
                            .ok_or(ExecError::InvalidOperandState(node_idx))?;
                        let dst = f64_regs
                            .get_mut(idx)
                            .and_then(|r| r.get_mut(0..tile_len))
                            .ok_or(ExecError::InvalidOperandState(node_idx))?;
                        for (d, &m) in dst.iter_mut().zip(mask.iter()) {
                            *d = if m { s } else { 0.0 };
                        }
                    } else {
                        let (left, right) = f64_regs.split_at_mut(idx);
                        let val = left
                            .get(v_idx)
                            .and_then(|r| r.get(0..tile_len))
                            .ok_or(ExecError::InvalidOperandState(node_idx))?;
                        let dst = right
                            .first_mut()
                            .and_then(|r| r.get_mut(0..tile_len))
                            .ok_or(ExecError::InvalidOperandState(node_idx))?;
                        for ((d, &m), &v) in dst.iter_mut().zip(mask.iter()).zip(val.iter()) {
                            *d = if m { v } else { 0.0 };
                        }
                    }
                }
                _ => return Err(ExecError::InvalidOperandState(node_idx)),
            }
        }

        for (acc, spec) in global_accs.iter_mut().zip(plan.reductions.iter()) {
            let final_tile = f64_regs
                .get(spec.reduce_input_idx)
                .and_then(|r| r.get(0..tile_len))
                .ok_or(ExecError::InvalidOperandState(program.output_node))?;
            match spec.kind {
                ReductionKind::Sum => {
                    *acc += stencil_reduce_sum_f64x8(final_tile);
                }
                ReductionKind::Max => {
                    for &x in final_tile {
                        if x > *acc {
                            *acc = x;
                        }
                    }
                }
                ReductionKind::Min => {
                    for &x in final_tile {
                        if x < *acc {
                            *acc = x;
                        }
                    }
                }
            }
        }

        tile_start = tile_end;
    }

    Ok(global_accs)
}

fn eval_tile_unary_math(
    opcode: Opcode,
    dst_idx: usize,
    a_idx: usize,
    tile_len: usize,
    f64_regs: &mut [[f64; L1_TILE_LEN]],
    scalars: &[f64],
    is_scalar: &[bool],
) -> Option<()> {
    if a_idx >= dst_idx {
        return None;
    }
    let a_is_scalar = *is_scalar.get(a_idx)?;
    let (left, right) = f64_regs.split_at_mut(dst_idx);
    let dst = right.first_mut()?.get_mut(0..tile_len)?;
    let apply = |x: f64| -> f64 {
        match opcode {
            Opcode::Abs => x.abs(),
            Opcode::Sqrt => x.sqrt(),
            Opcode::Floor => x.floor(),
            Opcode::Ceil => x.ceil(),
            Opcode::Ln => x.ln(),
            _ => x.exp(),
        }
    };
    if a_is_scalar {
        let val = apply(*scalars.get(a_idx)?);
        dst.fill(val);
    } else {
        let a = left.get(a_idx)?.get(0..tile_len)?;
        for (d, &x) in dst.iter_mut().zip(a.iter()) {
            *d = apply(x);
        }
    }
    Some(())
}

fn eval_tile_mask_logic(
    opcode: Opcode,
    dst_idx: usize,
    a_idx: usize,
    b_idx: usize,
    tile_len: usize,
    mask_regs: &mut [[bool; L1_TILE_LEN]],
) -> Option<()> {
    if a_idx >= dst_idx || b_idx >= dst_idx {
        return None;
    }
    let (left, right) = mask_regs.split_at_mut(dst_idx);
    let a = left.get(a_idx)?.get(0..tile_len)?;
    let b = left.get(b_idx)?.get(0..tile_len)?;
    let dst = right.first_mut()?.get_mut(0..tile_len)?;
    for ((d, &x), &y) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
        *d = match opcode {
            Opcode::MaskAnd => x && y,
            _ => x || y,
        };
    }
    Some(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "internal L1-tile register file evaluator"
)]
fn eval_tile_arith(
    opcode: Opcode,
    dst_idx: usize,
    a_idx: usize,
    b_idx: usize,
    tile_len: usize,
    f64_regs: &mut [[f64; L1_TILE_LEN]],
    scalars: &[f64],
    is_scalar: &[bool],
) -> Option<()> {
    if a_idx >= dst_idx || b_idx >= dst_idx {
        return None;
    }
    let a_is_scalar = *is_scalar.get(a_idx)?;
    let b_is_scalar = *is_scalar.get(b_idx)?;
    let (left, right) = f64_regs.split_at_mut(dst_idx);
    let dst = right.first_mut()?.get_mut(0..tile_len)?;

    let apply = |x: f64, y: f64| -> f64 {
        match opcode {
            Opcode::Add => x + y,
            Opcode::Sub => x - y,
            Opcode::Mul => x * y,
            Opcode::Div => x / y,
            Opcode::Min => {
                if x < y {
                    x
                } else {
                    y
                }
            }
            _ => {
                if x > y {
                    x
                } else {
                    y
                }
            }
        }
    };

    match (a_is_scalar, b_is_scalar) {
        (false, false) => {
            let a = left.get(a_idx)?.get(0..tile_len)?;
            let b = left.get(b_idx)?.get(0..tile_len)?;
            for ((d, &x), &y) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
                *d = apply(x, y);
            }
        }
        (false, true) => {
            let a = left.get(a_idx)?.get(0..tile_len)?;
            let s = *scalars.get(b_idx)?;
            for (d, &x) in dst.iter_mut().zip(a.iter()) {
                *d = apply(x, s);
            }
        }
        (true, false) => {
            let s = *scalars.get(a_idx)?;
            let b = left.get(b_idx)?.get(0..tile_len)?;
            for (d, &y) in dst.iter_mut().zip(b.iter()) {
                *d = apply(s, y);
            }
        }
        (true, true) => {
            let val = apply(*scalars.get(a_idx)?, *scalars.get(b_idx)?);
            dst.fill(val);
        }
    }
    Some(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "internal L1-tile ternary FMA evaluator"
)]
fn eval_tile_fma(
    dst_idx: usize,
    a_idx: usize,
    b_idx: usize,
    c_idx: usize,
    tile_len: usize,
    f64_regs: &mut [[f64; L1_TILE_LEN]],
    scalars: &[f64],
    is_scalar: &[bool],
) -> Option<()> {
    if a_idx >= dst_idx || b_idx >= dst_idx || c_idx >= dst_idx {
        return None;
    }
    let a_sc = *is_scalar.get(a_idx)?;
    let b_sc = *is_scalar.get(b_idx)?;
    let c_sc = *is_scalar.get(c_idx)?;
    let sa = *scalars.get(a_idx)?;
    let sb = *scalars.get(b_idx)?;
    let sc = *scalars.get(c_idx)?;
    let (left, right) = f64_regs.split_at_mut(dst_idx);
    let dst = right.first_mut()?.get_mut(0..tile_len)?;

    for (i, d) in dst.iter_mut().enumerate() {
        let va = if a_sc { sa } else { *left.get(a_idx)?.get(i)? };
        let vb = if b_sc { sb } else { *left.get(b_idx)?.get(i)? };
        let vc = if c_sc { sc } else { *left.get(c_idx)?.get(i)? };
        *d = va.mul_add(vb, vc);
    }
    Some(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "internal L1-tile predicate evaluator"
)]
fn eval_tile_cmp(
    opcode: Opcode,
    dst_idx: usize,
    a_idx: usize,
    b_idx: usize,
    tile_len: usize,
    f64_regs: &[[f64; L1_TILE_LEN]],
    mask_regs: &mut [[bool; L1_TILE_LEN]],
    scalars: &[f64],
    is_scalar: &[bool],
) -> Option<()> {
    let dst = mask_regs.get_mut(dst_idx)?.get_mut(0..tile_len)?;
    let cmp = |x: f64, y: f64| -> bool {
        match opcode {
            Opcode::CmpGt => x > y,
            Opcode::CmpLt => x < y,
            _ => x == y,
        }
    };
    let a_is_scalar = *is_scalar.get(a_idx)?;
    let b_is_scalar = *is_scalar.get(b_idx)?;
    match (a_is_scalar, b_is_scalar) {
        (false, true) => {
            let a = f64_regs.get(a_idx)?.get(0..tile_len)?;
            let thr = *scalars.get(b_idx)?;
            for (m, &x) in dst.iter_mut().zip(a.iter()) {
                *m = cmp(x, thr);
            }
        }
        (true, false) => {
            let thr = *scalars.get(a_idx)?;
            let b = f64_regs.get(b_idx)?.get(0..tile_len)?;
            for (m, &y) in dst.iter_mut().zip(b.iter()) {
                *m = cmp(thr, y);
            }
        }
        _ => {
            let a = f64_regs.get(a_idx)?.get(0..tile_len)?;
            let b = f64_regs.get(b_idx)?.get(0..tile_len)?;
            for ((m, &x), &y) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
                *m = cmp(x, y);
            }
        }
    }
    Some(())
}

/// Telemetry and schedule configuration for a single evaluated schedule candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct AutotuneCandidate {
    /// Execution policy tested (`AutoL1Tile`, `SimdStream`, `MultiCore`).
    pub policy: ExecPolicy,
    /// Fusion group ID applied (`0` for unfused, `1` for fused).
    pub fusion_group: u8,
    /// Measured hardware telemetry for this candidate.
    pub telemetry: HardwareTelemetry,
}

/// Result of running the automatic hardware schedule explorer (`autotune_schedule`).
#[derive(Debug, Clone, PartialEq)]
pub struct AutotuneReport {
    /// Winning `ScheduleHint` that achieved the highest `reward_score`.
    pub best_schedule: ScheduleHint,
    /// Hardware telemetry of the winning schedule.
    pub best_telemetry: HardwareTelemetry,
    /// Verified output value (guaranteed equivalent to Tier 0 Interpreter).
    pub output: RuntimeValue,
    /// All evaluated schedule candidates in exploration order.
    pub candidates: Vec<AutotuneCandidate>,
}

fn values_approx_equal(a: &RuntimeValue, b: &RuntimeValue) -> bool {
    match (a, b) {
        (RuntimeValue::Scalar(x), RuntimeValue::Scalar(y)) => {
            let diff = (x - y).abs();
            let scale = x.abs().max(y.abs()).max(1.0);
            diff / scale <= 1e-9
        }
        (RuntimeValue::ColumnF64(xs), RuntimeValue::ColumnF64(ys)) => {
            if xs.len() != ys.len() {
                return false;
            }
            xs.iter().zip(ys.iter()).all(|(&x, &y)| {
                let diff = (x - y).abs();
                let scale = x.abs().max(y.abs()).max(1.0);
                diff / scale <= 1e-9
            })
        }
        (RuntimeValue::ColumnMask(xs), RuntimeValue::ColumnMask(ys)) => xs == ys,
        _ => false,
    }
}

/// Automatically sweeps `ScheduleHint` policies (`AutoL1Tile`, `SimdStream`, `MultiCore`)
/// against `execute_tier2_with_zonemap`, verifies differential equivalence against
/// `execute_tier0_interpreter`, mutates `program` in-place with the winning schedule,
/// and returns the [`AutotuneReport`].
pub fn autotune_schedule(
    program: &mut RulProgram,
    inputs: &[&[f64]],
    zonemap_a0: Option<&ArenaZoneMap>,
) -> Result<AutotuneReport, ExecError> {
    let baseline = execute_tier0_interpreter(program, inputs)?;

    let schedule_space = [
        ScheduleHint {
            fusion_group: 0,
            policy: ExecPolicy::AutoL1Tile,
        },
        ScheduleHint {
            fusion_group: 1,
            policy: ExecPolicy::AutoL1Tile,
        },
        ScheduleHint {
            fusion_group: 1,
            policy: ExecPolicy::SimdStream,
        },
        ScheduleHint {
            fusion_group: 1,
            policy: ExecPolicy::MultiCore,
        },
    ];

    let mut candidates = Vec::with_capacity(schedule_space.len());
    let mut best_schedule = ScheduleHint::fused_simd(1);
    let mut best_telemetry: Option<HardwareTelemetry> = None;

    for sched in schedule_space {
        for node in &mut program.nodes {
            node.schedule = sched;
        }
        let rep = execute_tier2_with_zonemap(program, inputs, zonemap_a0)?;
        if !values_approx_equal(&baseline.output, &rep.output) {
            return Err(ExecError::ContractAssertionFailed {
                node_idx: program.output_node,
                observed: f64::NAN,
            });
        }

        let is_better = match &best_telemetry {
            None => true,
            Some(prev) => {
                rep.telemetry.reward_score > prev.reward_score
                    || ((rep.telemetry.reward_score - prev.reward_score).abs() <= f64::EPSILON
                        && rep.telemetry.exec_ns < prev.exec_ns)
            }
        };

        if is_better {
            best_schedule = sched;
            best_telemetry = Some(rep.telemetry);
        }

        candidates.push(AutotuneCandidate {
            policy: sched.policy,
            fusion_group: sched.fusion_group,
            telemetry: rep.telemetry,
        });
    }

    for node in &mut program.nodes {
        node.schedule = best_schedule;
    }

    Ok(AutotuneReport {
        best_schedule,
        best_telemetry: best_telemetry.unwrap_or(baseline.telemetry),
        output: baseline.output,
        candidates,
    })
}

/// Deterministic white-box execution trace entry for a single 128-bit [`NodeFrame`].
#[derive(Debug, Clone, PartialEq)]
pub struct NodeTraceEntry {
    /// 0-based index of the SSA node.
    pub node_idx: u32,
    /// Executed opcode.
    pub opcode: Opcode,
    /// True if this node is backward-reachable from `output_node`.
    pub is_live: bool,
    /// Runtime value category (`Scalar`, `ColumnF64`, `ColumnMask`).
    pub value_kind: &'static str,
    /// Number of elements produced (`1` for scalars).
    pub element_count: usize,
    /// Minimum finite value (or `0.0`/`1.0` for boolean masks).
    pub min_val: f64,
    /// Maximum finite value (or `0.0`/`1.0` for boolean masks).
    pub max_val: f64,
    /// Sum of values (or `true` count for boolean masks).
    pub sum_or_true_count: f64,
    /// Count of `NaN` or `±Infinity` values observed in this node's output.
    pub nan_or_inf_count: usize,
    /// Up to 4 sample values from the head of the node's output for human/agent inspection.
    pub sample_head: Vec<f64>,
}

/// Complete white-box execution trace report for a `.rl` program.
#[derive(Debug, Clone, PartialEq)]
pub struct ProgramTraceReport {
    /// Per-node execution trace entries in topological order `#0..#N-1`.
    pub entries: Vec<NodeTraceEntry>,
    /// Final verified output of the program.
    pub output: RuntimeValue,
    /// Execution telemetry from the trace run.
    pub telemetry: HardwareTelemetry,
}

/// Executes a `.rl` program in **White-Box Trace Mode**, recording deterministic per-node
/// statistics (`min`, `max`, `sum`, `nan_or_inf_count`, and sample head values) at every
/// SSA instruction `#0..#N-1`.
///
/// This eliminates any "black-box" opacity when auditing AI-generated `.rl` modules:
/// engineers and agents can pinpoint the exact node where a predicate, scaling factor,
/// or IEEE-754 `NaN`/`Inf` anomaly originates.
///
/// # Errors
/// Returns [`ExecError`] if static verification or node evaluation fails.
pub fn trace_program(
    program: &RulProgram,
    inputs: &[&[f64]],
) -> Result<ProgramTraceReport, ExecError> {
    let t_verify = Instant::now();
    verify_program(program).map_err(ExecError::VerificationFailed)?;
    let verify_ns = u64::try_from(t_verify.elapsed().as_nanos()).unwrap_or(u64::MAX);

    let elements_processed = validate_inputs(program, inputs)?;
    let live = compute_live_nodes(program);

    let t_exec = Instant::now();
    let mut values: Vec<EvalValue<'_>> = Vec::with_capacity(program.nodes.len());
    let mut entries: Vec<NodeTraceEntry> = Vec::with_capacity(program.nodes.len());
    let mut intermediate_bytes = 0usize;

    for (idx, frame) in program.nodes.iter().enumerate() {
        let node_idx = u32::try_from(idx).unwrap_or(u32::MAX);
        let is_live = live.get(idx).copied().unwrap_or(false);
        let val = eval_node_tier0(
            program,
            frame,
            node_idx,
            inputs,
            &values,
            &mut intermediate_bytes,
        )?;
        let entry = summarize_eval_value(node_idx, frame.opcode, is_live, &val);
        entries.push(entry);
        values.push(val);
    }

    let exec_ns = u64::try_from(t_exec.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let out_eval = values
        .get(program.output_node as usize)
        .cloned()
        .ok_or(ExecError::InvalidOperandState(program.output_node))?;
    let output = out_eval.into_runtime_value(program.output_node)?;

    Ok(ProgramTraceReport {
        entries,
        output,
        telemetry: build_telemetry(verify_ns, exec_ns, elements_processed, intermediate_bytes),
    })
}

fn summarize_eval_value(
    node_idx: u32,
    opcode: Opcode,
    is_live: bool,
    val: &EvalValue<'_>,
) -> NodeTraceEntry {
    match val {
        EvalValue::Scalar(x) => NodeTraceEntry {
            node_idx,
            opcode,
            is_live,
            value_kind: "Scalar",
            element_count: 1,
            min_val: *x,
            max_val: *x,
            sum_or_true_count: *x,
            nan_or_inf_count: usize::from(!x.is_finite()),
            sample_head: vec![*x],
        },
        EvalValue::ColumnF64(col) => {
            let mut min_v = f64::INFINITY;
            let mut max_v = f64::NEG_INFINITY;
            let mut sum_v = 0.0f64;
            let mut non_finite = 0usize;
            for &x in col.iter() {
                if !x.is_finite() {
                    non_finite = non_finite.saturating_add(1);
                } else {
                    if x < min_v {
                        min_v = x;
                    }
                    if x > max_v {
                        max_v = x;
                    }
                }
                sum_v += x;
            }
            if min_v == f64::INFINITY {
                min_v = f64::NAN;
                max_v = f64::NAN;
            }
            let sample_head: Vec<f64> = col.iter().take(4).copied().collect();
            NodeTraceEntry {
                node_idx,
                opcode,
                is_live,
                value_kind: "ColumnF64",
                element_count: col.len(),
                min_val: min_v,
                max_val: max_v,
                sum_or_true_count: sum_v,
                nan_or_inf_count: non_finite,
                sample_head,
            }
        }
        EvalValue::ColumnMask(mask) => {
            let trues = mask.iter().filter(|&&b| b).count();
            let min_v = if trues == mask.len() && !mask.is_empty() {
                1.0
            } else {
                0.0
            };
            let max_v = if trues > 0 { 1.0 } else { 0.0 };
            let sample_head: Vec<f64> = mask
                .iter()
                .take(4)
                .map(|&b| if b { 1.0 } else { 0.0 })
                .collect();
            NodeTraceEntry {
                node_idx,
                opcode,
                is_live,
                value_kind: "ColumnMask",
                element_count: mask.len(),
                min_val: min_v,
                max_val: max_v,
                sum_or_true_count: trues as f64,
                nan_or_inf_count: 0,
                sample_head,
            }
        }
        EvalValue::Unused => NodeTraceEntry {
            node_idx,
            opcode,
            is_live,
            value_kind: "Unused",
            element_count: 0,
            min_val: 0.0,
            max_val: 0.0,
            sum_or_true_count: 0.0,
            nan_or_inf_count: 0,
            sample_head: Vec::new(),
        },
    }
}

/// Formats a [`ProgramTraceReport`] as a deterministic human- and agent-readable
/// white-box inspection table.
#[must_use]
pub fn format_trace_table(report: &ProgramTraceReport) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(512 + report.entries.len() * 128);
    let _ = writeln!(
        out,
        "=== RUL (.rl) WHITE-BOX EXECUTION TRACE ({} SSA Nodes) ===",
        report.entries.len()
    );
    let _ = writeln!(
        out,
        "{:<6} {:<15} {:<5} {:<11} {:>8} {:>12} {:>12} {:>14} {:>7}  SAMPLE[0..4]",
        "NODE", "OPCODE", "LIVE", "KIND", "LEN", "MIN", "MAX", "SUM/TRUES", "NAN/INF"
    );
    for e in &report.entries {
        let preview: Vec<String> = e.sample_head.iter().map(|v| format!("{v:.3}")).collect();
        let _ = writeln!(
            out,
            "#{:<5} {:<15} {:<5} {:<11} {:>8} {:>12.4} {:>12.4} {:>14.4} {:>7}  [{}]",
            e.node_idx,
            format!("{:?}", e.opcode),
            if e.is_live { "yes" } else { "dead" },
            e.value_kind,
            e.element_count,
            e.min_val,
            e.max_val,
            e.sum_or_true_count,
            e.nan_or_inf_count,
            preview.join(", ")
        );
    }
    out
}
