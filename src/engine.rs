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

use crate::bytecode::{
    func_frame_units, ElemType, ExecPolicy, FuncSigInfo, NodeFrame, Opcode, ProofEntry,
    ProofRuleKind, RegionRole, RulProgram, ScheduleHint, CALL_ARG_SENTINEL, MAX_CALL_DEPTH,
    MAX_REGION_SLOTS, MAX_STACK_UNITS,
};
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

/// Typed scalar output pairing its static [`ElemType`] with its exact 64-bit little-endian bit pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypedScalar {
    /// Element primitive type (`F64`, `I64`, `BoolMask`, or `U8`).
    pub elem: ElemType,
    /// Raw 64-bit little-endian bit pattern.
    pub bits: u64,
}

impl TypedScalar {
    /// Constructs a [`TypedScalar`] holding an `f64`.
    #[must_use]
    pub const fn f64(val: f64) -> Self {
        Self {
            elem: ElemType::F64,
            bits: val.to_bits(),
        }
    }

    /// Constructs a [`TypedScalar`] holding an `i64`.
    #[must_use]
    pub const fn i64(val: i64) -> Self {
        #[expect(
            clippy::cast_sign_loss,
            reason = "preserving exact two's-complement 64-bit pattern of i64"
        )]
        let bits = val as u64;
        Self {
            elem: ElemType::I64,
            bits,
        }
    }

    /// Constructs a [`TypedScalar`] holding a zero-extended `u8`.
    #[must_use]
    pub const fn u8(val: u8) -> Self {
        Self {
            elem: ElemType::U8,
            bits: val as u64,
        }
    }

    /// Constructs a [`TypedScalar`] holding a `BoolMask` (`0` or `1`).
    #[must_use]
    pub const fn bool_mask(val: bool) -> Self {
        Self {
            elem: ElemType::BoolMask,
            bits: val as u64,
        }
    }

    /// Returns the value as `f64` if `self.elem == ElemType::F64`.
    #[must_use]
    pub const fn as_f64(self) -> Option<f64> {
        if matches!(self.elem, ElemType::F64) {
            Some(f64::from_bits(self.bits))
        } else {
            None
        }
    }

    /// Returns the value as `i64` if `self.elem == ElemType::I64`.
    #[must_use]
    pub const fn as_i64(self) -> Option<i64> {
        if matches!(self.elem, ElemType::I64) {
            #[expect(
                clippy::cast_possible_wrap,
                reason = "restoring exact two's-complement i64 from its 64-bit pattern"
            )]
            Some(self.bits as i64)
        } else {
            None
        }
    }

    /// Returns the value as `u8` if `self.elem == ElemType::U8`.
    #[must_use]
    pub const fn as_u8(self) -> Option<u8> {
        if matches!(self.elem, ElemType::U8) {
            Some((self.bits & 0xFF) as u8)
        } else {
            None
        }
    }

    /// Returns the value as `bool` if `self.elem == ElemType::BoolMask`.
    #[must_use]
    pub const fn as_bool(self) -> Option<bool> {
        if matches!(self.elem, ElemType::BoolMask) {
            Some(self.bits != 0)
        } else {
            None
        }
    }
}

/// Runtime value produced by evaluating a node in the `.rl` graph.
#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeValue {
    /// A single scalar `f64` result.
    Scalar(f64),
    /// A typed non-`f64` scalar result (`I64`, `U8`, or `BoolMask`) with exact 64-bit pattern.
    TypedScalar(TypedScalar),
    /// A columnar `f64` vector.
    ColumnF64(Vec<f64>),
    /// A columnar boolean mask.
    ColumnMask(Vec<bool>),
}

impl RuntimeValue {
    /// Returns this runtime value as a [`TypedScalar`] if it is any scalar variant (`F64`, `I64`, `U8`, or `BoolMask`).
    #[must_use]
    pub const fn as_typed_scalar(&self) -> Option<TypedScalar> {
        match self {
            Self::Scalar(v) => Some(TypedScalar::f64(*v)),
            Self::TypedScalar(ts) => Some(*ts),
            Self::ColumnF64(_) | Self::ColumnMask(_) => None,
        }
    }
}

/// Internal zero-copy evaluation value used by Tier 0 and Tier 1.
#[derive(Debug, Clone)]
enum EvalValue<'a> {
    Unused,
    Scalar(f64),
    ScalarI64(i64),
    ScalarU8(u8),
    ScalarMask(bool),
    ColumnF64(Cow<'a, [f64]>),
    ColumnMask(Vec<bool>),
}

impl EvalValue<'_> {
    fn into_runtime_value(self, node_idx: u32) -> Result<RuntimeValue, ExecError> {
        match self {
            Self::Scalar(v) => Ok(RuntimeValue::Scalar(v)),
            Self::ScalarI64(v) => Ok(RuntimeValue::TypedScalar(TypedScalar::i64(v))),
            Self::ScalarU8(v) => Ok(RuntimeValue::TypedScalar(TypedScalar::u8(v))),
            Self::ScalarMask(b) => Ok(RuntimeValue::TypedScalar(TypedScalar::bool_mask(b))),
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
    /// Byte output collected from the declared output region, if any.
    pub output_bytes: Option<Vec<u8>>,
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
    /// Structured control-flow or typed scalar local opcode is unsupported in columnar Tier 1 / Tier 2.
    UnsupportedInColumnarTier,
    /// Bounded `Loop` (`ProofRuleKind::LoopBound`) exceeded its declared `bound_hi` back-edge limit.
    LoopBoundExceeded {
        /// Offending `Loop` header node index.
        loop_node: u32,
        /// Maximum permitted back-edge iterations (`bound_hi`).
        limit: u32,
    },
    /// Checked integer division or remainder by zero or `i64::MIN / -1` (`AOT_EXIT_INT_DIV_ERR = 72`).
    IntegerDivisionError {
        /// Offending `Div` or `Rem` node index.
        node_idx: u32,
    },
    /// Float-to-integer `Cast` received `NaN`, infinity, or an out-of-range value (`AOT_EXIT_CAST_ERR = 73`).
    CastOutOfRange {
        /// Offending `Cast` node index.
        node_idx: u32,
    },
    /// Region allocation, reset, growth, or input pre-fill exceeded `[min_len, max_len]` or failed (`AOT_EXIT_REGION_OOM = 74`).
    RegionAllocationFailed {
        /// Target region slot.
        slot: u32,
    },
    /// `RegionLoad` or `RegionStore` index fell outside `0..len` (`AOT_EXIT_GATHER_OOB = 71`).
    RegionBoundsOutOfRange {
        /// Offending `RegionLoad` or `RegionStore` node index.
        node_idx: u32,
        /// Target region slot.
        slot: u32,
        /// Observed index value.
        index: i64,
        /// Current region length.
        len: usize,
    },
    /// Function call exceeded [`MAX_CALL_DEPTH`] (4096) or [`MAX_STACK_UNITS`] (`1 << 20`) (`AOT_EXIT_STACK_OVERFLOW = 75`).
    CallDepthExceeded {
        /// Attempted live activation count (entry function included).
        call_depth: u32,
        /// Attempted live ISA stack frame units sum.
        stack_units: u32,
    },
    /// `ConstDataLoad` byte offset fell outside `[0, seg_len - elem_bytes]` (`AOT_EXIT_GATHER_OOB = 71`).
    ConstDataOutOfBounds {
        /// Offending `ConstDataLoad` node index.
        node_idx: u32,
        /// Target constant-data segment identifier.
        seg_id: u32,
        /// Observed byte offset index.
        index: i64,
        /// Byte length of the target constant-data segment.
        seg_len: u32,
    },
    /// `CallIndirect` index fell outside `0..entry_count` (`AOT_EXIT_INDIRECT_CALL_OOB = 76`).
    IndirectCallOutOfBounds {
        /// Offending `CallIndirect` node index.
        node_idx: u32,
        /// Target `CallTable` identifier.
        table_id: u32,
        /// Observed table index.
        index: i64,
        /// Entry count of the target `CallTable`.
        entry_count: u32,
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
            Self::UnsupportedInColumnarTier => write!(
                f,
                "structured control flow and typed scalar locals are unsupported in columnar Tier 1 / Tier 2"
            ),
            Self::LoopBoundExceeded { loop_node, limit } => write!(
                f,
                "loop bound exceeded at Loop node #{loop_node}: back-edge limit {limit} exhausted"
            ),
            Self::IntegerDivisionError { node_idx } => write!(
                f,
                "integer division or remainder error at node #{node_idx}"
            ),
            Self::CastOutOfRange { node_idx } => {
                write!(f, "float-to-integer cast out of range at node #{node_idx}")
            }
            Self::RegionAllocationFailed { slot } => {
                write!(f, "region allocation or length constraint failed for slot #{slot}")
            }
            Self::RegionBoundsOutOfRange {
                node_idx,
                slot,
                index,
                len,
            } => write!(
                f,
                "region bounds out of range at node #{node_idx} (slot #{slot}): index={index}, len={len}"
            ),
            Self::CallDepthExceeded {
                call_depth,
                stack_units,
            } => write!(
                f,
                "call stack guard exceeded (depth={call_depth}/{MAX_CALL_DEPTH}, stack_units={stack_units}/{MAX_STACK_UNITS})"
            ),
            Self::ConstDataOutOfBounds {
                node_idx,
                seg_id,
                index,
                seg_len,
            } => write!(
                f,
                "const data load out of bounds at node #{node_idx} (seg #{seg_id}): index={index}, seg_len={seg_len}"
            ),
            Self::IndirectCallOutOfBounds {
                node_idx,
                table_id,
                index,
                entry_count,
            } => write!(
                f,
                "indirect call index out of bounds at node #{node_idx} (table #{table_id}): index={index}, entry_count={entry_count}"
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
    if program.uses_structured_control_or_locals() {
        return vec![true; n];
    }
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
            Opcode::ArenaInput
            | Opcode::ConstF64
            | Opcode::ConstI64
            | Opcode::LocalGet
            | Opcode::Block
            | Opcode::Loop
            | Opcode::Br
            | Opcode::RegionLen
            | Opcode::RegionFree
            | Opcode::RegionLease
            | Opcode::RegionEndLease
            | Opcode::RegionMove
            | Opcode::RegionDrop
            | Opcode::Func
            | Opcode::Return
            | Opcode::Call
            | Opcode::CallArg
            | Opcode::ConstDataLen
            | Opcode::CallIndirect => {}
            Opcode::ArenaSlice
            | Opcode::Abs
            | Opcode::Sqrt
            | Opcode::Floor
            | Opcode::Ceil
            | Opcode::Ln
            | Opcode::Exp
            | Opcode::MaskNot
            | Opcode::Cast
            | Opcode::ReduceSum
            | Opcode::ReduceMax
            | Opcode::ReduceMin
            | Opcode::RingEnqueue
            | Opcode::AssertContract
            | Opcode::If
            | Opcode::BrIf
            | Opcode::BrTable => {
                if let Some(a) = live.get_mut(frame.in_a as usize) {
                    *a = true;
                }
            }
            Opcode::ArenaMutate
            | Opcode::LocalSet
            | Opcode::ConstDataLoad
            | Opcode::RegionNew
            | Opcode::RegionGrow
            | Opcode::RegionLoad => {
                if let Some(b) = live.get_mut(frame.in_b as usize) {
                    *b = true;
                }
            }
            Opcode::RegionStore => {
                let (idx_node, val_node) = frame.decode_u16_pair();
                if let Some(a) = live.get_mut(idx_node as usize) {
                    *a = true;
                }
                if let Some(b) = live.get_mut(val_node as usize) {
                    *b = true;
                }
            }
            Opcode::Else | Opcode::End => {
                if frame.in_b != 0 {
                    if let Some(a) = live.get_mut(frame.in_a as usize) {
                        *a = true;
                    }
                }
            }
            Opcode::Add
            | Opcode::Sub
            | Opcode::Mul
            | Opcode::Div
            | Opcode::Rem
            | Opcode::BitAnd
            | Opcode::BitOr
            | Opcode::BitXor
            | Opcode::Shl
            | Opcode::ShrU
            | Opcode::ShrS
            | Opcode::Min
            | Opcode::Max
            | Opcode::CmpGt
            | Opcode::CmpLt
            | Opcode::CmpEq
            | Opcode::CmpLtU
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

/// **Tier 0 — Zero-Copy Reference Columnar & Structured Control-Flow Interpreter (Golden Oracle)**.
///
/// Verifies the `.rl` program in $O(N)$ and evaluates each node sequentially using
/// zero-copy `Cow<'a, [f64]>` borrows on `ArenaInput` and `ArenaSlice`, or executes
/// structured WASM-style control flow, 64 typed scalar local slots, and declared regions.
///
/// # Errors
/// Returns [`ExecError`] if verification, input binding, loop bounds, region bounds/allocation, integer division, casts, or contract assertions fail.
pub fn execute_tier0_interpreter(
    program: &RulProgram,
    inputs: &[&[f64]],
) -> Result<ExecutionReport, ExecError> {
    execute_tier0_interpreter_with_input(program, inputs, &[])
}

/// **Tier 0 — Reference Interpreter with Caller-Supplied Byte Input Region**.
///
/// Pre-fills the declared `RegionRole::Input` region (if any) with `input_bytes` before node `#0`,
/// and returns any declared `RegionRole::Output` region's bytes in [`ExecutionReport::output_bytes`].
///
/// # Errors
/// Returns [`ExecError`] if verification or execution fails, including [`ExecError::RegionAllocationFailed`]
/// if `input_bytes.len()` exceeds the input region's declared `max_len`.
pub fn execute_tier0_interpreter_with_input(
    program: &RulProgram,
    inputs: &[&[f64]],
    input_bytes: &[u8],
) -> Result<ExecutionReport, ExecError> {
    let t_verify = Instant::now();
    verify_program(program).map_err(ExecError::VerificationFailed)?;
    let verify_ns = u64::try_from(t_verify.elapsed().as_nanos()).unwrap_or(u64::MAX);

    let elements_processed = validate_inputs(program, inputs)?;

    let t_exec = Instant::now();
    let (values, intermediate_bytes, output_bytes) = if program.uses_structured_control_or_locals()
    {
        eval_structured_tier0(program, inputs, input_bytes)?
    } else {
        let mut vals: Vec<EvalValue<'_>> = Vec::with_capacity(program.nodes.len());
        let mut inter_bytes = 0usize;
        for (idx, frame) in program.nodes.iter().enumerate() {
            let node_idx = u32::try_from(idx).unwrap_or(u32::MAX);
            let val = eval_node_tier0(program, frame, node_idx, inputs, &vals, &mut inter_bytes)?;
            vals.push(val);
        }
        (vals, inter_bytes, None)
    };

    let exec_ns = u64::try_from(t_exec.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let out_eval = values
        .get(program.output_node as usize)
        .cloned()
        .ok_or(ExecError::InvalidOperandState(program.output_node))?;
    let output = out_eval.into_runtime_value(program.output_node)?;

    Ok(ExecutionReport {
        output,
        output_bytes,
        telemetry: build_telemetry(verify_ns, exec_ns, elements_processed, intermediate_bytes),
    })
}

#[derive(Debug, Clone)]
enum RegionStorage {
    U8(Vec<u8>),
    I64(Vec<i64>),
    F64(Vec<f64>),
}

impl RegionStorage {
    fn alloc_zeroed(elem: ElemType, len: usize, slot: u32) -> Result<Self, ExecError> {
        match elem {
            ElemType::U8 => {
                let mut v = Vec::new();
                v.try_reserve_exact(len)
                    .map_err(|_e| ExecError::RegionAllocationFailed { slot })?;
                v.resize(len, 0);
                Ok(Self::U8(v))
            }
            ElemType::I64 => {
                let mut v = Vec::new();
                v.try_reserve_exact(len)
                    .map_err(|_e| ExecError::RegionAllocationFailed { slot })?;
                v.resize(len, 0);
                Ok(Self::I64(v))
            }
            ElemType::F64 => {
                let mut v = Vec::new();
                v.try_reserve_exact(len)
                    .map_err(|_e| ExecError::RegionAllocationFailed { slot })?;
                v.resize(len, 0.0);
                Ok(Self::F64(v))
            }
            ElemType::BoolMask => Err(ExecError::RegionAllocationFailed { slot }),
        }
    }

    fn alloc_u8_from_slice(bytes: &[u8], slot: u32) -> Result<Self, ExecError> {
        let mut v = Vec::new();
        v.try_reserve_exact(bytes.len())
            .map_err(|_e| ExecError::RegionAllocationFailed { slot })?;
        v.extend_from_slice(bytes);
        Ok(Self::U8(v))
    }

    fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::F64(v) => v.len(),
        }
    }

    fn reset_zeroed(&mut self, new_len: usize, slot: u32) -> Result<(), ExecError> {
        match self {
            Self::U8(v) => {
                if new_len > v.capacity() {
                    *v = Vec::new();
                    v.try_reserve_exact(new_len)
                        .map_err(|_e| ExecError::RegionAllocationFailed { slot })?;
                    v.resize(new_len, 0);
                } else {
                    v.clear();
                    v.resize(new_len, 0);
                }
            }
            Self::I64(v) => {
                if new_len > v.capacity() {
                    *v = Vec::new();
                    v.try_reserve_exact(new_len)
                        .map_err(|_e| ExecError::RegionAllocationFailed { slot })?;
                    v.resize(new_len, 0);
                } else {
                    v.clear();
                    v.resize(new_len, 0);
                }
            }
            Self::F64(v) => {
                if new_len > v.capacity() {
                    *v = Vec::new();
                    v.try_reserve_exact(new_len)
                        .map_err(|_e| ExecError::RegionAllocationFailed { slot })?;
                    v.resize(new_len, 0.0);
                } else {
                    v.clear();
                    v.resize(new_len, 0.0);
                }
            }
        }
        Ok(())
    }

    fn grow_zeroed(&mut self, add_len: usize, slot: u32) -> Result<(), ExecError> {
        match self {
            Self::U8(v) => {
                v.try_reserve(add_len)
                    .map_err(|_e| ExecError::RegionAllocationFailed { slot })?;
                v.resize(v.len() + add_len, 0);
            }
            Self::I64(v) => {
                v.try_reserve(add_len)
                    .map_err(|_e| ExecError::RegionAllocationFailed { slot })?;
                v.resize(v.len() + add_len, 0);
            }
            Self::F64(v) => {
                v.try_reserve(add_len)
                    .map_err(|_e| ExecError::RegionAllocationFailed { slot })?;
                v.resize(v.len() + add_len, 0.0);
            }
        }
        Ok(())
    }

    fn free_storage(&mut self) {
        match self {
            Self::U8(v) => {
                v.clear();
                v.shrink_to_fit();
            }
            Self::I64(v) => {
                v.clear();
                v.shrink_to_fit();
            }
            Self::F64(v) => {
                v.clear();
                v.shrink_to_fit();
            }
        }
    }
}

#[derive(Debug, Clone)]
struct Tier0Region {
    min_len: usize,
    max_len: usize,
    storage: RegionStorage,
}

fn move_tier0_region_storage(
    regions: &mut [Option<Tier0Region>; MAX_REGION_SLOTS],
    src_slot: u32,
    dst_slot: u32,
) -> Result<(), ExecError> {
    let src_idx = src_slot as usize;
    let dst_idx = dst_slot as usize;
    if src_idx == dst_idx {
        return Err(ExecError::RegionAllocationFailed { slot: src_slot });
    }
    let moved_storage = {
        let src_reg = regions
            .get_mut(src_idx)
            .and_then(Option::as_mut)
            .ok_or(ExecError::RegionAllocationFailed { slot: src_slot })?;
        let empty = match &src_reg.storage {
            RegionStorage::U8(_) => RegionStorage::U8(Vec::new()),
            RegionStorage::I64(_) => RegionStorage::I64(Vec::new()),
            RegionStorage::F64(_) => RegionStorage::F64(Vec::new()),
        };
        std::mem::replace(&mut src_reg.storage, empty)
    };
    let dst_reg = regions
        .get_mut(dst_idx)
        .and_then(Option::as_mut)
        .ok_or(ExecError::RegionAllocationFailed { slot: dst_slot })?;
    dst_reg.storage = moved_storage;
    Ok(())
}

type StructuredTier0Output<'a> = (Vec<EvalValue<'a>>, usize, Option<Vec<u8>>);

fn init_tier0_regions(
    program: &RulProgram,
    input_bytes: &[u8],
) -> Result<([Option<Tier0Region>; MAX_REGION_SLOTS], Option<usize>), ExecError> {
    let mut regions: [Option<Tier0Region>; MAX_REGION_SLOTS] = std::array::from_fn(|_i| None);
    let mut output_region_slot: Option<usize> = None;
    for proof in &program.proofs {
        if proof.rule_kind != ProofRuleKind::RegionDecl {
            continue;
        }
        let slot = usize::from(proof.target_arena);
        let Some((elem, role)) = ProofEntry::decode_region_decl_witness(proof.witness_node) else {
            return Err(ExecError::RegionAllocationFailed {
                slot: u32::from(proof.target_arena),
            });
        };
        let min_len = proof.bound_lo as usize;
        let max_len = proof.bound_hi as usize;
        let slot_u32 = u32::from(proof.target_arena);
        let storage = if role == RegionRole::Input {
            if input_bytes.len() > max_len {
                return Err(ExecError::RegionAllocationFailed { slot: slot_u32 });
            }
            RegionStorage::alloc_u8_from_slice(input_bytes, slot_u32)?
        } else {
            if role == RegionRole::Output {
                output_region_slot = Some(slot);
            }
            RegionStorage::alloc_zeroed(elem, min_len, slot_u32)?
        };
        if let Some(entry) = regions.get_mut(slot) {
            *entry = Some(Tier0Region {
                min_len,
                max_len,
                storage,
            });
        }
    }
    Ok((regions, output_region_slot))
}

fn take_output_region_bytes(
    regions: &mut [Option<Tier0Region>; MAX_REGION_SLOTS],
    output_region_slot: Option<usize>,
) -> Option<Vec<u8>> {
    output_region_slot.and_then(|slot| {
        regions
            .get_mut(slot)
            .and_then(Option::as_mut)
            .and_then(|r| match &mut r.storage {
                RegionStorage::U8(v) => Some(std::mem::take(v)),
                RegionStorage::I64(_) | RegionStorage::F64(_) => None,
            })
    })
}

#[derive(Debug, Clone, Copy)]
struct Tier0ConstSeg {
    offset: usize,
    len: usize,
}

#[derive(Debug, Clone)]
struct Tier0BrTable {
    default_depth: usize,
    case_depths: Vec<usize>,
}

#[derive(Debug, Clone)]
struct Tier0CallTable {
    func_ids: Vec<usize>,
}

struct Tier0ConstTables {
    const_segs: Vec<Tier0ConstSeg>,
    br_tables: Vec<Tier0BrTable>,
    call_tables: Vec<Tier0CallTable>,
}

fn scan_tier0_const_tables(program: &RulProgram) -> Result<Tier0ConstTables, ExecError> {
    let mut raw_segs: Vec<(usize, Tier0ConstSeg)> = Vec::new();
    let mut raw_br: Vec<(usize, Tier0BrTable)> = Vec::new();
    let mut raw_call: Vec<(usize, Tier0CallTable)> = Vec::new();

    for proof in &program.proofs {
        match proof.rule_kind {
            ProofRuleKind::ConstDataSeg => {
                raw_segs.push((
                    usize::from(proof.target_arena),
                    Tier0ConstSeg {
                        offset: proof.bound_lo as usize,
                        len: proof.bound_hi as usize,
                    },
                ));
            }
            ProofRuleKind::BrTableSpec => {
                let lo = proof.bound_lo as usize;
                let count = proof.bound_hi as usize;
                let slice = program
                    .const_data
                    .get(lo..lo.saturating_add(count.saturating_mul(4)))
                    .ok_or(ExecError::InvalidOperandState(0))?;
                let mut case_depths = Vec::with_capacity(count);
                for chunk in slice.as_chunks::<4>().0 {
                    let d = u32::from_le_bytes(*chunk);
                    case_depths.push(d as usize);
                }
                raw_br.push((
                    proof.witness_node as usize,
                    Tier0BrTable {
                        default_depth: usize::from(proof.target_arena),
                        case_depths,
                    },
                ));
            }
            ProofRuleKind::CallTable => {
                let lo = proof.bound_lo as usize;
                let count = proof.bound_hi as usize;
                let slice = program
                    .const_data
                    .get(lo..lo.saturating_add(count.saturating_mul(4)))
                    .ok_or(ExecError::InvalidOperandState(0))?;
                let mut func_ids = Vec::with_capacity(count);
                for chunk in slice.as_chunks::<4>().0 {
                    let fid = u32::from_le_bytes(*chunk);
                    func_ids.push(fid as usize);
                }
                raw_call.push((usize::from(proof.target_arena), Tier0CallTable { func_ids }));
            }
            _ => {}
        }
    }

    let mut segs: Vec<Tier0ConstSeg> = vec![Tier0ConstSeg { offset: 0, len: 0 }; raw_segs.len()];
    for (id, seg) in raw_segs {
        if let Some(slot) = segs.get_mut(id) {
            *slot = seg;
        }
    }
    let mut br_tables: Vec<Tier0BrTable> = vec![
        Tier0BrTable {
            default_depth: 0,
            case_depths: Vec::new(),
        };
        raw_br.len()
    ];
    for (id, bt) in raw_br {
        if let Some(slot) = br_tables.get_mut(id) {
            *slot = bt;
        }
    }
    let mut call_tables: Vec<Tier0CallTable> = vec![
        Tier0CallTable {
            func_ids: Vec::new(),
        };
        raw_call.len()
    ];
    for (id, ct) in raw_call {
        if let Some(slot) = call_tables.get_mut(id) {
            *slot = ct;
        }
    }
    Ok(Tier0ConstTables {
        const_segs: segs,
        br_tables,
        call_tables,
    })
}

fn eval_const_data_load_tier0<'a>(
    program: &RulProgram,
    const_segs: &[Tier0ConstSeg],
    frame: &NodeFrame,
    node_idx: u32,
    idx_i64: i64,
) -> Result<EvalValue<'a>, ExecError> {
    let seg = const_segs
        .get(frame.in_a as usize)
        .copied()
        .ok_or(ExecError::InvalidOperandState(node_idx))?;
    let seg_len_u32 = u32::try_from(seg.len).unwrap_or(u32::MAX);
    let elem_bytes = if frame.type_lane.elem == ElemType::I64 {
        8usize
    } else {
        1usize
    };
    let Ok(idx_usize) = usize::try_from(idx_i64) else {
        return Err(ExecError::ConstDataOutOfBounds {
            node_idx,
            seg_id: frame.in_a,
            index: idx_i64,
            seg_len: seg_len_u32,
        });
    };
    let Some(end_rel) = idx_usize.checked_add(elem_bytes) else {
        return Err(ExecError::ConstDataOutOfBounds {
            node_idx,
            seg_id: frame.in_a,
            index: idx_i64,
            seg_len: seg_len_u32,
        });
    };
    if end_rel > seg.len {
        return Err(ExecError::ConstDataOutOfBounds {
            node_idx,
            seg_id: frame.in_a,
            index: idx_i64,
            seg_len: seg_len_u32,
        });
    }
    let abs_lo = seg.offset.saturating_add(idx_usize);
    let abs_hi = abs_lo.saturating_add(elem_bytes);
    let slice = program
        .const_data
        .get(abs_lo..abs_hi)
        .ok_or(ExecError::ConstDataOutOfBounds {
            node_idx,
            seg_id: frame.in_a,
            index: idx_i64,
            seg_len: seg_len_u32,
        })?;
    if frame.type_lane.elem == ElemType::I64 {
        let arr: [u8; 8] = slice
            .try_into()
            .map_err(|_e| ExecError::InvalidOperandState(node_idx))?;
        Ok(EvalValue::ScalarI64(i64::from_le_bytes(arr)))
    } else {
        let byte = slice
            .first()
            .copied()
            .ok_or(ExecError::InvalidOperandState(node_idx))?;
        Ok(EvalValue::ScalarU8(byte))
    }
}

fn select_br_table_header(
    sel_val: &EvalValue<'_>,
    default_hdr: usize,
    case_hdrs: &[usize],
    node_idx: u32,
) -> Result<usize, ExecError> {
    let idx_opt = match *sel_val {
        EvalValue::ScalarI64(v) => usize::try_from(v).ok(),
        EvalValue::ScalarU8(b) => Some(usize::from(b)),
        _ => return Err(ExecError::InvalidOperandState(node_idx)),
    };
    Ok(idx_opt
        .and_then(|i| case_hdrs.get(i).copied())
        .unwrap_or(default_hdr))
}

#[derive(Debug, Clone, Copy)]
struct Tier0FuncMeta {
    start: usize,
    end: usize,
    node_count: usize,
    frame_units: u32,
    sig: FuncSigInfo,
}

struct Tier0Activation<'a> {
    func_start: usize,
    func_end: usize,
    frame_units: u32,
    pc: usize,
    return_dest_rel: Option<usize>,
    values: Vec<EvalValue<'a>>,
    locals: [EvalValue<'a>; 64],
    loop_fuel: Vec<u32>,
    pending_call_args: Option<(EvalValue<'a>, Option<EvalValue<'a>>)>,
}

enum FuncStepAction<'a> {
    Continue,
    PushCall {
        target_func_id: usize,
        args: [EvalValue<'a>; 4],
        arity: usize,
        ret_dest_rel: Option<usize>,
    },
    ReturnVal(EvalValue<'a>),
}

#[expect(
    clippy::too_many_lines,
    reason = "heap-stack function-table reference interpreter with explicit depth and stack-unit guards"
)]
fn eval_functions_tier0<'a>(
    program: &RulProgram,
    inputs: &[&'a [f64]],
    input_bytes: &[u8],
) -> Result<StructuredTier0Output<'a>, ExecError> {
    let (mut regions, output_region_slot) = init_tier0_regions(program, input_bytes)?;
    let Tier0ConstTables {
        const_segs,
        br_tables,
        call_tables,
    } = scan_tier0_const_tables(program)?;
    let n = program.nodes.len();

    let mut sigs_raw: Vec<FuncSigInfo> = Vec::new();
    for proof in &program.proofs {
        if proof.rule_kind == ProofRuleKind::FuncSig {
            let sig = proof
                .decode_func_sig()
                .ok_or(ExecError::InvalidOperandState(0))?;
            sigs_raw.push(sig);
        }
    }
    let func_count = sigs_raw.len();
    let mut sig_table: Vec<Option<FuncSigInfo>> = vec![None; func_count];
    for sig in sigs_raw {
        if let Some(slot) = sig_table.get_mut(usize::from(sig.func_id)) {
            *slot = Some(sig);
        }
    }

    let mut if_to_else: Vec<Option<usize>> = vec![None; n];
    let mut else_to_if: Vec<usize> = vec![0; n];
    let mut header_to_end: Vec<usize> = vec![0; n];
    let mut br_target_header: Vec<usize> = vec![0; n];
    let mut br_table_targets: Vec<Option<(usize, Vec<usize>)>> = vec![None; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut func_starts: Vec<usize> = vec![0; func_count];
    let mut func_ends: Vec<usize> = vec![0; func_count];

    for (idx, frame) in program.nodes.iter().enumerate() {
        let node_idx = u32::try_from(idx).unwrap_or(u32::MAX);
        match frame.opcode {
            Opcode::Func => {
                let fid = frame.in_a as usize;
                if let Some(slot) = func_starts.get_mut(fid) {
                    *slot = idx;
                }
                stack.push(idx);
            }
            Opcode::Block | Opcode::Loop | Opcode::If => {
                stack.push(idx);
            }
            Opcode::Else => {
                let if_hdr = stack
                    .last()
                    .copied()
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                if let Some(slot) = if_to_else.get_mut(if_hdr) {
                    *slot = Some(idx);
                }
                if let Some(slot) = else_to_if.get_mut(idx) {
                    *slot = if_hdr;
                }
            }
            Opcode::End => {
                let hdr = stack
                    .pop()
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                if let Some(slot) = header_to_end.get_mut(hdr) {
                    *slot = idx;
                }
                if let Some(hdr_frame) = program.nodes.get(hdr) {
                    if hdr_frame.opcode == Opcode::Func {
                        let fid = hdr_frame.in_a as usize;
                        if let Some(slot) = func_ends.get_mut(fid) {
                            *slot = idx;
                        }
                    }
                }
            }
            Opcode::Br => {
                let depth = frame.in_a as usize;
                let target_pos = stack
                    .len()
                    .checked_sub(1 + depth)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                let hdr = stack
                    .get(target_pos)
                    .copied()
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                if let Some(slot) = br_target_header.get_mut(idx) {
                    *slot = hdr;
                }
            }
            Opcode::BrIf => {
                let depth = frame.in_b as usize;
                let target_pos = stack
                    .len()
                    .checked_sub(1 + depth)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                let hdr = stack
                    .get(target_pos)
                    .copied()
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                if let Some(slot) = br_target_header.get_mut(idx) {
                    *slot = hdr;
                }
            }
            Opcode::BrTable => {
                let spec = br_tables
                    .get(frame.in_b as usize)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                let def_pos = stack
                    .len()
                    .checked_sub(1 + spec.default_depth)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                let default_hdr = stack
                    .get(def_pos)
                    .copied()
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                let mut case_hdrs = Vec::with_capacity(spec.case_depths.len());
                for &d in &spec.case_depths {
                    let pos = stack
                        .len()
                        .checked_sub(1 + d)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let hdr = stack
                        .get(pos)
                        .copied()
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    case_hdrs.push(hdr);
                }
                if let Some(slot) = br_table_targets.get_mut(idx) {
                    *slot = Some((default_hdr, case_hdrs));
                }
            }
            _ => {}
        }
    }

    let mut funcs: Vec<Tier0FuncMeta> = Vec::with_capacity(func_count);
    for fid in 0..func_count {
        let start = func_starts
            .get(fid)
            .copied()
            .ok_or(ExecError::InvalidOperandState(0))?;
        let end = func_ends
            .get(fid)
            .copied()
            .ok_or(ExecError::InvalidOperandState(0))?;
        let sig = sig_table
            .get(fid)
            .copied()
            .flatten()
            .ok_or(ExecError::InvalidOperandState(0))?;
        let node_count = end.saturating_sub(start).saturating_add(1);
        funcs.push(Tier0FuncMeta {
            start,
            end,
            node_count,
            frame_units: func_frame_units(node_count),
            sig,
        });
    }

    let entry_id = func_count
        .checked_sub(1)
        .ok_or(ExecError::InvalidOperandState(0))?;
    let entry_meta = funcs
        .get(entry_id)
        .copied()
        .ok_or(ExecError::InvalidOperandState(0))?;

    let mut call_stack: Vec<Tier0Activation<'a>> = Vec::new();
    let mut live_call_depth: u32 = 0;
    let mut live_stack_units: u32 = 0;

    let push_frame = |meta: Tier0FuncMeta,
                      args: [EvalValue<'a>; 4],
                      arity: usize,
                      ret_dest_rel: Option<usize>,
                      call_stack: &mut Vec<Tier0Activation<'a>>,
                      live_call_depth: &mut u32,
                      live_stack_units: &mut u32|
     -> Result<(), ExecError> {
        let next_depth = live_call_depth.saturating_add(1);
        let next_units = live_stack_units.saturating_add(meta.frame_units);
        if next_depth > MAX_CALL_DEPTH || next_units > MAX_STACK_UNITS {
            return Err(ExecError::CallDepthExceeded {
                call_depth: next_depth,
                stack_units: next_units,
            });
        }
        let mut values: Vec<EvalValue<'a>> = Vec::new();
        values
            .try_reserve_exact(meta.node_count)
            .map_err(|_e| ExecError::CallDepthExceeded {
                call_depth: next_depth,
                stack_units: next_units,
            })?;
        values.resize(meta.node_count, EvalValue::Unused);

        let mut loop_fuel: Vec<u32> = Vec::new();
        loop_fuel.try_reserve_exact(meta.node_count).map_err(|_e| {
            ExecError::CallDepthExceeded {
                call_depth: next_depth,
                stack_units: next_units,
            }
        })?;
        loop_fuel.resize(meta.node_count, 0);

        *live_call_depth = next_depth;
        *live_stack_units = next_units;

        let mut locals: [EvalValue<'a>; 64] = std::array::from_fn(|_i| EvalValue::Unused);
        for (i, arg) in args.into_iter().take(arity).enumerate() {
            if let Some(slot) = locals.get_mut(i) {
                *slot = arg;
            }
        }
        call_stack.push(Tier0Activation {
            func_start: meta.start,
            func_end: meta.end,
            frame_units: meta.frame_units,
            pc: meta.start + 1,
            return_dest_rel: ret_dest_rel,
            values,
            locals,
            loop_fuel,
            pending_call_args: None,
        });
        Ok(())
    };

    push_frame(
        entry_meta,
        [
            EvalValue::Unused,
            EvalValue::Unused,
            EvalValue::Unused,
            EvalValue::Unused,
        ],
        0,
        None,
        &mut call_stack,
        &mut live_call_depth,
        &mut live_stack_units,
    )?;

    let mut intermediate_bytes = 0usize;
    let mut entry_result: Option<EvalValue<'a>> = None;

    while !call_stack.is_empty() {
        let action = {
            let act = call_stack
                .last_mut()
                .ok_or(ExecError::InvalidOperandState(0))?;
            let pc = act.pc;
            let node_idx = u32::try_from(pc).unwrap_or(u32::MAX);
            let rel_pc = pc
                .checked_sub(act.func_start)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            let frame = program
                .nodes
                .get(pc)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;

            match frame.opcode {
                Opcode::ConstI64 => {
                    let slot = act
                        .values
                        .get_mut(rel_pc)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    *slot = EvalValue::ScalarI64(frame.decode_const_i64());
                    act.pc += 1;
                    FuncStepAction::Continue
                }
                Opcode::LocalGet => {
                    let val = act
                        .locals
                        .get(frame.in_a as usize)
                        .cloned()
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    if matches!(val, EvalValue::Unused) {
                        return Err(ExecError::InvalidOperandState(node_idx));
                    }
                    let slot = act
                        .values
                        .get_mut(rel_pc)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    *slot = val;
                    act.pc += 1;
                    FuncStepAction::Continue
                }
                Opcode::LocalSet => {
                    let val = act
                        .values
                        .get(frame.in_b as usize)
                        .cloned()
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let local_slot = act
                        .locals
                        .get_mut(frame.in_a as usize)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    *local_slot = val.clone();
                    let node_slot = act
                        .values
                        .get_mut(rel_pc)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    *node_slot = val;
                    act.pc += 1;
                    FuncStepAction::Continue
                }
                Opcode::Block => {
                    act.pc += 1;
                    FuncStepAction::Continue
                }
                Opcode::Loop => {
                    if frame.proof_idx > 0 {
                        let proof = program
                            .proofs
                            .get(usize::from(frame.proof_idx - 1))
                            .ok_or(ExecError::InvalidOperandState(node_idx))?;
                        let fuel_slot = act
                            .loop_fuel
                            .get_mut(rel_pc)
                            .ok_or(ExecError::InvalidOperandState(node_idx))?;
                        *fuel_slot = proof.bound_hi;
                    }
                    act.pc += 1;
                    FuncStepAction::Continue
                }
                Opcode::If => {
                    let cond_val = act
                        .values
                        .get(frame.in_a as usize)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let EvalValue::ScalarMask(cond_bool) = *cond_val else {
                        return Err(ExecError::InvalidOperandState(node_idx));
                    };
                    if cond_bool {
                        act.pc += 1;
                    } else if let Some(else_pc) = if_to_else.get(pc).copied().flatten() {
                        act.pc = else_pc + 1;
                    } else {
                        let end_pc = header_to_end
                            .get(pc)
                            .copied()
                            .ok_or(ExecError::InvalidOperandState(node_idx))?;
                        act.pc = end_pc + 1;
                    }
                    FuncStepAction::Continue
                }
                Opcode::Else => {
                    let if_hdr = else_to_if
                        .get(pc)
                        .copied()
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let end_pc = header_to_end
                        .get(if_hdr)
                        .copied()
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    if frame.in_b != 0 {
                        let then_val = act
                            .values
                            .get(frame.in_a as usize)
                            .cloned()
                            .ok_or(ExecError::InvalidOperandState(node_idx))?;
                        let end_rel = end_pc
                            .checked_sub(act.func_start)
                            .ok_or(ExecError::InvalidOperandState(node_idx))?;
                        let end_slot = act
                            .values
                            .get_mut(end_rel)
                            .ok_or(ExecError::InvalidOperandState(node_idx))?;
                        *end_slot = then_val;
                    }
                    act.pc = end_pc + 1;
                    FuncStepAction::Continue
                }
                Opcode::End => {
                    if pc == act.func_end {
                        let ret_val = if frame.in_b != 0 {
                            act.values
                                .get(frame.in_a as usize)
                                .cloned()
                                .ok_or(ExecError::InvalidOperandState(node_idx))?
                        } else {
                            EvalValue::Unused
                        };
                        FuncStepAction::ReturnVal(ret_val)
                    } else {
                        if frame.in_b != 0 {
                            let res_val = act
                                .values
                                .get(frame.in_a as usize)
                                .cloned()
                                .ok_or(ExecError::InvalidOperandState(node_idx))?;
                            let end_slot = act
                                .values
                                .get_mut(rel_pc)
                                .ok_or(ExecError::InvalidOperandState(node_idx))?;
                            *end_slot = res_val;
                        }
                        act.pc += 1;
                        FuncStepAction::Continue
                    }
                }
                Opcode::Br => {
                    let hdr = br_target_header
                        .get(pc)
                        .copied()
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    act.pc = step_func_branch_target(
                        program,
                        hdr,
                        act.func_start,
                        node_idx,
                        &header_to_end,
                        &mut act.loop_fuel,
                    )?;
                    FuncStepAction::Continue
                }
                Opcode::BrIf => {
                    let cond_val = act
                        .values
                        .get(frame.in_a as usize)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let EvalValue::ScalarMask(cond_bool) = *cond_val else {
                        return Err(ExecError::InvalidOperandState(node_idx));
                    };
                    if cond_bool {
                        let hdr = br_target_header
                            .get(pc)
                            .copied()
                            .ok_or(ExecError::InvalidOperandState(node_idx))?;
                        act.pc = step_func_branch_target(
                            program,
                            hdr,
                            act.func_start,
                            node_idx,
                            &header_to_end,
                            &mut act.loop_fuel,
                        )?;
                    } else {
                        act.pc += 1;
                    }
                    FuncStepAction::Continue
                }
                Opcode::BrTable => {
                    let sel_val = act
                        .values
                        .get(frame.in_a as usize)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let (default_hdr, case_hdrs) = br_table_targets
                        .get(pc)
                        .and_then(Option::as_ref)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let hdr = select_br_table_header(sel_val, *default_hdr, case_hdrs, node_idx)?;
                    act.pc = step_func_branch_target(
                        program,
                        hdr,
                        act.func_start,
                        node_idx,
                        &header_to_end,
                        &mut act.loop_fuel,
                    )?;
                    FuncStepAction::Continue
                }
                Opcode::ConstDataLen => {
                    let seg = const_segs
                        .get(frame.in_a as usize)
                        .copied()
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let seg_len = i64::try_from(seg.len)
                        .map_err(|_e| ExecError::InvalidOperandState(node_idx))?;
                    let slot = act
                        .values
                        .get_mut(rel_pc)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    *slot = EvalValue::ScalarI64(seg_len);
                    act.pc += 1;
                    FuncStepAction::Continue
                }
                Opcode::ConstDataLoad => {
                    let idx_val = act
                        .values
                        .get(frame.in_b as usize)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let EvalValue::ScalarI64(idx_i64) = *idx_val else {
                        return Err(ExecError::InvalidOperandState(node_idx));
                    };
                    let loaded =
                        eval_const_data_load_tier0(program, &const_segs, frame, node_idx, idx_i64)?;
                    let slot = act
                        .values
                        .get_mut(rel_pc)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    *slot = loaded;
                    act.pc += 1;
                    FuncStepAction::Continue
                }
                Opcode::CallIndirect => {
                    let (table_id, index_node) = frame.decode_in_a_u16_pair();
                    let table = call_tables
                        .get(table_id as usize)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let idx_val = act
                        .values
                        .get(index_node as usize)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let EvalValue::ScalarI64(idx_i64) = *idx_val else {
                        return Err(ExecError::InvalidOperandState(node_idx));
                    };
                    let entry_count = u32::try_from(table.func_ids.len()).unwrap_or(u32::MAX);
                    let Ok(idx_usize) = usize::try_from(idx_i64) else {
                        return Err(ExecError::IndirectCallOutOfBounds {
                            node_idx,
                            table_id,
                            index: idx_i64,
                            entry_count,
                        });
                    };
                    let target_func_id = table.func_ids.get(idx_usize).copied().ok_or(
                        ExecError::IndirectCallOutOfBounds {
                            node_idx,
                            table_id,
                            index: idx_i64,
                            entry_count,
                        },
                    )?;
                    let callee = funcs
                        .get(target_func_id)
                        .copied()
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let (arg0_raw, arg1_raw) = frame.decode_u16_pair();
                    let sentinel = u32::from(CALL_ARG_SENTINEL);
                    let mut args = [
                        EvalValue::Unused,
                        EvalValue::Unused,
                        EvalValue::Unused,
                        EvalValue::Unused,
                    ];
                    if arg0_raw != sentinel {
                        args[0] = act
                            .values
                            .get(arg0_raw as usize)
                            .cloned()
                            .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    }
                    if arg1_raw != sentinel {
                        args[1] = act
                            .values
                            .get(arg1_raw as usize)
                            .cloned()
                            .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    }
                    if let Some((a2, a3_opt)) = act.pending_call_args.take() {
                        args[2] = a2;
                        if let Some(a3) = a3_opt {
                            args[3] = a3;
                        }
                    }
                    act.pc += 1;
                    let ret_dest_rel = callee.sig.ret_elem.map(|_| rel_pc);
                    FuncStepAction::PushCall {
                        target_func_id,
                        args,
                        arity: usize::from(callee.sig.arity),
                        ret_dest_rel,
                    }
                }
                Opcode::Return => {
                    let ret_val = if frame.in_a != 0 {
                        act.values
                            .get(frame.in_a as usize)
                            .cloned()
                            .ok_or(ExecError::InvalidOperandState(node_idx))?
                    } else {
                        EvalValue::Unused
                    };
                    FuncStepAction::ReturnVal(ret_val)
                }
                Opcode::CallArg => {
                    let arg2 = act
                        .values
                        .get(frame.in_a as usize)
                        .cloned()
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let arg3 = if frame.in_b == u32::from(CALL_ARG_SENTINEL) {
                        None
                    } else {
                        Some(
                            act.values
                                .get(frame.in_b as usize)
                                .cloned()
                                .ok_or(ExecError::InvalidOperandState(node_idx))?,
                        )
                    };
                    act.pending_call_args = Some((arg2, arg3));
                    act.pc += 1;
                    FuncStepAction::Continue
                }
                Opcode::Call => {
                    let target_func_id = frame.in_a as usize;
                    let callee = funcs
                        .get(target_func_id)
                        .copied()
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let (arg0_raw, arg1_raw) = frame.decode_u16_pair();
                    let sentinel = u32::from(CALL_ARG_SENTINEL);
                    let mut args = [
                        EvalValue::Unused,
                        EvalValue::Unused,
                        EvalValue::Unused,
                        EvalValue::Unused,
                    ];
                    if arg0_raw != sentinel {
                        args[0] = act
                            .values
                            .get(arg0_raw as usize)
                            .cloned()
                            .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    }
                    if arg1_raw != sentinel {
                        args[1] = act
                            .values
                            .get(arg1_raw as usize)
                            .cloned()
                            .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    }
                    if let Some((a2, a3_opt)) = act.pending_call_args.take() {
                        args[2] = a2;
                        if let Some(a3) = a3_opt {
                            args[3] = a3;
                        }
                    }
                    act.pc += 1;
                    let ret_dest_rel = callee.sig.ret_elem.map(|_| rel_pc);
                    FuncStepAction::PushCall {
                        target_func_id,
                        args,
                        arity: usize::from(callee.sig.arity),
                        ret_dest_rel,
                    }
                }
                Opcode::RegionNew => {
                    let slot_idx = frame.in_a as usize;
                    let len_val = act
                        .values
                        .get(frame.in_b as usize)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let EvalValue::ScalarI64(len_i64) = *len_val else {
                        return Err(ExecError::InvalidOperandState(node_idx));
                    };
                    let region = regions
                        .get_mut(slot_idx)
                        .and_then(Option::as_mut)
                        .ok_or(ExecError::RegionAllocationFailed { slot: frame.in_a })?;
                    let Ok(new_len) = usize::try_from(len_i64) else {
                        return Err(ExecError::RegionAllocationFailed { slot: frame.in_a });
                    };
                    if new_len < region.min_len || new_len > region.max_len {
                        return Err(ExecError::RegionAllocationFailed { slot: frame.in_a });
                    }
                    region.storage.reset_zeroed(new_len, frame.in_a)?;
                    act.pc += 1;
                    FuncStepAction::Continue
                }
                Opcode::RegionGrow => {
                    let slot_idx = frame.in_a as usize;
                    let add_val = act
                        .values
                        .get(frame.in_b as usize)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let EvalValue::ScalarI64(add_i64) = *add_val else {
                        return Err(ExecError::InvalidOperandState(node_idx));
                    };
                    let region = regions
                        .get_mut(slot_idx)
                        .and_then(Option::as_mut)
                        .ok_or(ExecError::RegionAllocationFailed { slot: frame.in_a })?;
                    let Ok(add_len) = usize::try_from(add_i64) else {
                        return Err(ExecError::RegionAllocationFailed { slot: frame.in_a });
                    };
                    let Some(new_len) = region.storage.len().checked_add(add_len) else {
                        return Err(ExecError::RegionAllocationFailed { slot: frame.in_a });
                    };
                    if new_len < region.min_len || new_len > region.max_len {
                        return Err(ExecError::RegionAllocationFailed { slot: frame.in_a });
                    }
                    region.storage.grow_zeroed(add_len, frame.in_a)?;
                    let new_len_i64 = i64::try_from(new_len)
                        .map_err(|_e| ExecError::RegionAllocationFailed { slot: frame.in_a })?;
                    let slot = act
                        .values
                        .get_mut(rel_pc)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    *slot = EvalValue::ScalarI64(new_len_i64);
                    act.pc += 1;
                    FuncStepAction::Continue
                }
                Opcode::RegionLen => {
                    let slot_idx = frame.in_a as usize;
                    let region = regions
                        .get(slot_idx)
                        .and_then(Option::as_ref)
                        .ok_or(ExecError::RegionAllocationFailed { slot: frame.in_a })?;
                    let cur_len = i64::try_from(region.storage.len())
                        .map_err(|_e| ExecError::RegionAllocationFailed { slot: frame.in_a })?;
                    let slot = act
                        .values
                        .get_mut(rel_pc)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    *slot = EvalValue::ScalarI64(cur_len);
                    act.pc += 1;
                    FuncStepAction::Continue
                }
                Opcode::RegionLoad => {
                    let slot_idx = frame.in_a as usize;
                    let idx_val = act
                        .values
                        .get(frame.in_b as usize)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let EvalValue::ScalarI64(idx_i64) = *idx_val else {
                        return Err(ExecError::InvalidOperandState(node_idx));
                    };
                    let region = regions
                        .get(slot_idx)
                        .and_then(Option::as_ref)
                        .ok_or(ExecError::RegionAllocationFailed { slot: frame.in_a })?;
                    let len = region.storage.len();
                    let Ok(idx_usize) = usize::try_from(idx_i64) else {
                        return Err(ExecError::RegionBoundsOutOfRange {
                            node_idx,
                            slot: frame.in_a,
                            index: idx_i64,
                            len,
                        });
                    };
                    let loaded = match &region.storage {
                        RegionStorage::U8(v) => v.get(idx_usize).copied().map(EvalValue::ScalarU8),
                        RegionStorage::I64(v) => {
                            v.get(idx_usize).copied().map(EvalValue::ScalarI64)
                        }
                        RegionStorage::F64(v) => v.get(idx_usize).copied().map(EvalValue::Scalar),
                    }
                    .ok_or(ExecError::RegionBoundsOutOfRange {
                        node_idx,
                        slot: frame.in_a,
                        index: idx_i64,
                        len,
                    })?;
                    let slot = act
                        .values
                        .get_mut(rel_pc)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    *slot = loaded;
                    act.pc += 1;
                    FuncStepAction::Continue
                }
                Opcode::RegionStore => {
                    let slot_idx = frame.in_a as usize;
                    let (idx_node, val_node) = frame.decode_u16_pair();
                    let idx_val = act
                        .values
                        .get(idx_node as usize)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let EvalValue::ScalarI64(idx_i64) = *idx_val else {
                        return Err(ExecError::InvalidOperandState(node_idx));
                    };
                    let val = act
                        .values
                        .get(val_node as usize)
                        .cloned()
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let region = regions
                        .get_mut(slot_idx)
                        .and_then(Option::as_mut)
                        .ok_or(ExecError::RegionAllocationFailed { slot: frame.in_a })?;
                    let len = region.storage.len();
                    let Ok(idx_usize) = usize::try_from(idx_i64) else {
                        return Err(ExecError::RegionBoundsOutOfRange {
                            node_idx,
                            slot: frame.in_a,
                            index: idx_i64,
                            len,
                        });
                    };
                    match (&mut region.storage, val) {
                        (RegionStorage::U8(v), EvalValue::ScalarU8(b)) => {
                            let dst =
                                v.get_mut(idx_usize)
                                    .ok_or(ExecError::RegionBoundsOutOfRange {
                                        node_idx,
                                        slot: frame.in_a,
                                        index: idx_i64,
                                        len,
                                    })?;
                            *dst = b;
                        }
                        (RegionStorage::I64(v), EvalValue::ScalarI64(x)) => {
                            let dst =
                                v.get_mut(idx_usize)
                                    .ok_or(ExecError::RegionBoundsOutOfRange {
                                        node_idx,
                                        slot: frame.in_a,
                                        index: idx_i64,
                                        len,
                                    })?;
                            *dst = x;
                        }
                        (RegionStorage::F64(v), EvalValue::Scalar(x)) => {
                            let dst =
                                v.get_mut(idx_usize)
                                    .ok_or(ExecError::RegionBoundsOutOfRange {
                                        node_idx,
                                        slot: frame.in_a,
                                        index: idx_i64,
                                        len,
                                    })?;
                            *dst = x;
                        }
                        _ => return Err(ExecError::InvalidOperandState(node_idx)),
                    }
                    act.pc += 1;
                    FuncStepAction::Continue
                }
                Opcode::RegionFree | Opcode::RegionDrop => {
                    let slot_idx = frame.in_a as usize;
                    let region = regions
                        .get_mut(slot_idx)
                        .and_then(Option::as_mut)
                        .ok_or(ExecError::RegionAllocationFailed { slot: frame.in_a })?;
                    region.storage.free_storage();
                    act.pc += 1;
                    FuncStepAction::Continue
                }
                Opcode::RegionLease | Opcode::RegionEndLease => {
                    act.pc += 1;
                    FuncStepAction::Continue
                }
                Opcode::RegionMove => {
                    move_tier0_region_storage(&mut regions, frame.in_a, frame.in_b)?;
                    act.pc += 1;
                    FuncStepAction::Continue
                }
                Opcode::Func => return Err(ExecError::InvalidOperandState(node_idx)),
                _ => {
                    let val = eval_node_tier0(
                        program,
                        frame,
                        node_idx,
                        inputs,
                        &act.values,
                        &mut intermediate_bytes,
                    )?;
                    let slot = act
                        .values
                        .get_mut(rel_pc)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    *slot = val;
                    act.pc += 1;
                    FuncStepAction::Continue
                }
            }
        };

        match action {
            FuncStepAction::Continue => {}
            FuncStepAction::PushCall {
                target_func_id,
                args,
                arity,
                ret_dest_rel,
            } => {
                let callee = funcs
                    .get(target_func_id)
                    .copied()
                    .ok_or(ExecError::InvalidOperandState(0))?;
                push_frame(
                    callee,
                    args,
                    arity,
                    ret_dest_rel,
                    &mut call_stack,
                    &mut live_call_depth,
                    &mut live_stack_units,
                )?;
            }
            FuncStepAction::ReturnVal(ret_val) => {
                let popped = call_stack.pop().ok_or(ExecError::InvalidOperandState(0))?;
                live_call_depth = live_call_depth.saturating_sub(1);
                live_stack_units = live_stack_units.saturating_sub(popped.frame_units);
                if let Some(caller) = call_stack.last_mut() {
                    if let Some(dest_rel) = popped.return_dest_rel {
                        let slot = caller
                            .values
                            .get_mut(dest_rel)
                            .ok_or(ExecError::InvalidOperandState(0))?;
                        *slot = ret_val;
                    }
                } else {
                    entry_result = Some(ret_val);
                    break;
                }
            }
        }
    }

    let mut out_values: Vec<EvalValue<'a>> = vec![EvalValue::Unused; n];
    if let (Some(slot), Some(val)) = (
        out_values.get_mut(program.output_node as usize),
        entry_result,
    ) {
        *slot = val;
    }
    let output_bytes = take_output_region_bytes(&mut regions, output_region_slot);
    Ok((out_values, intermediate_bytes, output_bytes))
}

fn step_func_branch_target(
    program: &RulProgram,
    hdr: usize,
    func_start: usize,
    branch_node: u32,
    header_to_end: &[usize],
    loop_fuel: &mut [u32],
) -> Result<usize, ExecError> {
    let hdr_frame = program
        .nodes
        .get(hdr)
        .ok_or(ExecError::InvalidOperandState(branch_node))?;
    if hdr_frame.opcode == Opcode::Loop {
        if hdr_frame.proof_idx > 0 {
            let proof = program
                .proofs
                .get(usize::from(hdr_frame.proof_idx - 1))
                .ok_or(ExecError::InvalidOperandState(branch_node))?;
            let rel_hdr = hdr
                .checked_sub(func_start)
                .ok_or(ExecError::InvalidOperandState(branch_node))?;
            let fuel_slot = loop_fuel
                .get_mut(rel_hdr)
                .ok_or(ExecError::InvalidOperandState(branch_node))?;
            if *fuel_slot == 0 {
                return Err(ExecError::LoopBoundExceeded {
                    loop_node: u32::try_from(hdr).unwrap_or(u32::MAX),
                    limit: proof.bound_hi,
                });
            }
            *fuel_slot -= 1;
        }
        Ok(hdr + 1)
    } else {
        let end_pc = header_to_end
            .get(hdr)
            .copied()
            .ok_or(ExecError::InvalidOperandState(branch_node))?;
        Ok(end_pc + 1)
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "single-function structured control-flow and declared-region reference interpreter"
)]
fn eval_structured_tier0<'a>(
    program: &RulProgram,
    inputs: &[&'a [f64]],
    input_bytes: &[u8],
) -> Result<StructuredTier0Output<'a>, ExecError> {
    if program.has_functions() {
        return eval_functions_tier0(program, inputs, input_bytes);
    }
    let (mut regions, output_region_slot) = init_tier0_regions(program, input_bytes)?;
    let Tier0ConstTables {
        const_segs,
        br_tables,
        call_tables: _,
    } = scan_tier0_const_tables(program)?;

    let n = program.nodes.len();
    let mut if_to_else: Vec<Option<usize>> = vec![None; n];
    let mut else_to_if: Vec<usize> = vec![0; n];
    let mut header_to_end: Vec<usize> = vec![0; n];
    let mut br_target_header: Vec<usize> = vec![0; n];
    let mut br_table_targets: Vec<Option<(usize, Vec<usize>)>> = vec![None; n];
    let mut stack: Vec<usize> = Vec::new();

    for (idx, frame) in program.nodes.iter().enumerate() {
        let node_idx = u32::try_from(idx).unwrap_or(u32::MAX);
        match frame.opcode {
            Opcode::Block | Opcode::Loop | Opcode::If => {
                stack.push(idx);
            }
            Opcode::Else => {
                let if_hdr = stack
                    .last()
                    .copied()
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                if let Some(slot) = if_to_else.get_mut(if_hdr) {
                    *slot = Some(idx);
                }
                if let Some(slot) = else_to_if.get_mut(idx) {
                    *slot = if_hdr;
                }
            }
            Opcode::End => {
                let hdr = stack
                    .pop()
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                if let Some(slot) = header_to_end.get_mut(hdr) {
                    *slot = idx;
                }
            }
            Opcode::Br => {
                let depth = frame.in_a as usize;
                let target_pos = stack
                    .len()
                    .checked_sub(1 + depth)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                let hdr = stack
                    .get(target_pos)
                    .copied()
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                if let Some(slot) = br_target_header.get_mut(idx) {
                    *slot = hdr;
                }
            }
            Opcode::BrIf => {
                let depth = frame.in_b as usize;
                let target_pos = stack
                    .len()
                    .checked_sub(1 + depth)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                let hdr = stack
                    .get(target_pos)
                    .copied()
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                if let Some(slot) = br_target_header.get_mut(idx) {
                    *slot = hdr;
                }
            }
            Opcode::BrTable => {
                let spec = br_tables
                    .get(frame.in_b as usize)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                let def_pos = stack
                    .len()
                    .checked_sub(1 + spec.default_depth)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                let default_hdr = stack
                    .get(def_pos)
                    .copied()
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                let mut case_hdrs = Vec::with_capacity(spec.case_depths.len());
                for &d in &spec.case_depths {
                    let pos = stack
                        .len()
                        .checked_sub(1 + d)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let hdr = stack
                        .get(pos)
                        .copied()
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    case_hdrs.push(hdr);
                }
                if let Some(slot) = br_table_targets.get_mut(idx) {
                    *slot = Some((default_hdr, case_hdrs));
                }
            }
            _ => {}
        }
    }

    let mut values: Vec<EvalValue<'a>> = vec![EvalValue::Unused; n];
    let mut locals: [EvalValue<'a>; 64] = std::array::from_fn(|_i| EvalValue::Unused);
    let mut arena_state: Vec<Cow<'a, [f64]>> = inputs.iter().map(|&s| Cow::Borrowed(s)).collect();
    let mut loop_fuel: Vec<u32> = vec![0; n];
    let mut intermediate_bytes = 0usize;
    let mut pc = 0usize;

    while pc < n {
        let node_idx = u32::try_from(pc).unwrap_or(u32::MAX);
        let frame = program
            .nodes
            .get(pc)
            .ok_or(ExecError::InvalidOperandState(node_idx))?;

        match frame.opcode {
            Opcode::ArenaInput => {
                let col = arena_state
                    .get(frame.in_a as usize)
                    .cloned()
                    .ok_or(ExecError::MissingArenaInput(frame.in_a))?;
                let slot = values
                    .get_mut(pc)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                *slot = EvalValue::ColumnF64(col);
                pc += 1;
            }
            Opcode::ArenaMutate => {
                let src = values
                    .get(frame.in_b as usize)
                    .cloned()
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                if let EvalValue::ColumnF64(ref col) = src {
                    let arena_slot = arena_state
                        .get_mut(frame.in_a as usize)
                        .ok_or(ExecError::MissingArenaInput(frame.in_a))?;
                    *arena_slot = col.clone();
                }
                let slot = values
                    .get_mut(pc)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                *slot = src;
                pc += 1;
            }
            Opcode::ConstI64 => {
                let slot = values
                    .get_mut(pc)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                *slot = EvalValue::ScalarI64(frame.decode_const_i64());
                pc += 1;
            }
            Opcode::LocalGet => {
                let val = locals
                    .get(frame.in_a as usize)
                    .cloned()
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                if matches!(val, EvalValue::Unused) {
                    return Err(ExecError::InvalidOperandState(node_idx));
                }
                let slot = values
                    .get_mut(pc)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                *slot = val;
                pc += 1;
            }
            Opcode::LocalSet => {
                let val = values
                    .get(frame.in_b as usize)
                    .cloned()
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                let local_slot = locals
                    .get_mut(frame.in_a as usize)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                *local_slot = val.clone();
                let node_slot = values
                    .get_mut(pc)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                *node_slot = val;
                pc += 1;
            }
            Opcode::Block => {
                pc += 1;
            }
            Opcode::Loop => {
                if frame.proof_idx > 0 {
                    let proof = program
                        .proofs
                        .get(usize::from(frame.proof_idx - 1))
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let fuel_slot = loop_fuel
                        .get_mut(pc)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    *fuel_slot = proof.bound_hi;
                }
                pc += 1;
            }
            Opcode::If => {
                let cond_val = values
                    .get(frame.in_a as usize)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                let EvalValue::ScalarMask(cond_bool) = *cond_val else {
                    return Err(ExecError::InvalidOperandState(node_idx));
                };
                if cond_bool {
                    pc += 1;
                } else if let Some(else_pc) = if_to_else.get(pc).copied().flatten() {
                    pc = else_pc + 1;
                } else {
                    let end_pc = header_to_end
                        .get(pc)
                        .copied()
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    pc = end_pc + 1;
                }
            }
            Opcode::Else => {
                let if_hdr = else_to_if
                    .get(pc)
                    .copied()
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                let end_pc = header_to_end
                    .get(if_hdr)
                    .copied()
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                if frame.in_b != 0 {
                    let then_val = values
                        .get(frame.in_a as usize)
                        .cloned()
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let end_slot = values
                        .get_mut(end_pc)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    *end_slot = then_val;
                }
                pc = end_pc + 1;
            }
            Opcode::End => {
                if frame.in_b != 0 {
                    let res_val = values
                        .get(frame.in_a as usize)
                        .cloned()
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    let end_slot = values
                        .get_mut(pc)
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    *end_slot = res_val;
                }
                pc += 1;
            }
            Opcode::Br => {
                let hdr = br_target_header
                    .get(pc)
                    .copied()
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                pc = step_branch_target(program, hdr, node_idx, &header_to_end, &mut loop_fuel)?;
            }
            Opcode::BrIf => {
                let cond_val = values
                    .get(frame.in_a as usize)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                let EvalValue::ScalarMask(cond_bool) = *cond_val else {
                    return Err(ExecError::InvalidOperandState(node_idx));
                };
                if cond_bool {
                    let hdr = br_target_header
                        .get(pc)
                        .copied()
                        .ok_or(ExecError::InvalidOperandState(node_idx))?;
                    pc =
                        step_branch_target(program, hdr, node_idx, &header_to_end, &mut loop_fuel)?;
                } else {
                    pc += 1;
                }
            }
            Opcode::BrTable => {
                let sel_val = values
                    .get(frame.in_a as usize)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                let (default_hdr, case_hdrs) = br_table_targets
                    .get(pc)
                    .and_then(Option::as_ref)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                let hdr = select_br_table_header(sel_val, *default_hdr, case_hdrs, node_idx)?;
                pc = step_branch_target(program, hdr, node_idx, &header_to_end, &mut loop_fuel)?;
            }
            Opcode::ConstDataLen => {
                let seg = const_segs
                    .get(frame.in_a as usize)
                    .copied()
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                let seg_len = i64::try_from(seg.len)
                    .map_err(|_e| ExecError::InvalidOperandState(node_idx))?;
                let slot = values
                    .get_mut(pc)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                *slot = EvalValue::ScalarI64(seg_len);
                pc += 1;
            }
            Opcode::ConstDataLoad => {
                let idx_val = values
                    .get(frame.in_b as usize)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                let EvalValue::ScalarI64(idx_i64) = *idx_val else {
                    return Err(ExecError::InvalidOperandState(node_idx));
                };
                let loaded =
                    eval_const_data_load_tier0(program, &const_segs, frame, node_idx, idx_i64)?;
                let slot = values
                    .get_mut(pc)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                *slot = loaded;
                pc += 1;
            }
            Opcode::RegionNew => {
                let slot_idx = frame.in_a as usize;
                let len_val = values
                    .get(frame.in_b as usize)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                let EvalValue::ScalarI64(len_i64) = *len_val else {
                    return Err(ExecError::InvalidOperandState(node_idx));
                };
                let region = regions
                    .get_mut(slot_idx)
                    .and_then(Option::as_mut)
                    .ok_or(ExecError::RegionAllocationFailed { slot: frame.in_a })?;
                let Ok(new_len) = usize::try_from(len_i64) else {
                    return Err(ExecError::RegionAllocationFailed { slot: frame.in_a });
                };
                if new_len < region.min_len || new_len > region.max_len {
                    return Err(ExecError::RegionAllocationFailed { slot: frame.in_a });
                }
                region.storage.reset_zeroed(new_len, frame.in_a)?;
                pc += 1;
            }
            Opcode::RegionGrow => {
                let slot_idx = frame.in_a as usize;
                let add_val = values
                    .get(frame.in_b as usize)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                let EvalValue::ScalarI64(add_i64) = *add_val else {
                    return Err(ExecError::InvalidOperandState(node_idx));
                };
                let region = regions
                    .get_mut(slot_idx)
                    .and_then(Option::as_mut)
                    .ok_or(ExecError::RegionAllocationFailed { slot: frame.in_a })?;
                let Ok(add_len) = usize::try_from(add_i64) else {
                    return Err(ExecError::RegionAllocationFailed { slot: frame.in_a });
                };
                let Some(new_len) = region.storage.len().checked_add(add_len) else {
                    return Err(ExecError::RegionAllocationFailed { slot: frame.in_a });
                };
                if new_len < region.min_len || new_len > region.max_len {
                    return Err(ExecError::RegionAllocationFailed { slot: frame.in_a });
                }
                region.storage.grow_zeroed(add_len, frame.in_a)?;
                let new_len_i64 = i64::try_from(new_len)
                    .map_err(|_e| ExecError::RegionAllocationFailed { slot: frame.in_a })?;
                let slot = values
                    .get_mut(pc)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                *slot = EvalValue::ScalarI64(new_len_i64);
                pc += 1;
            }
            Opcode::RegionLen => {
                let slot_idx = frame.in_a as usize;
                let region = regions
                    .get(slot_idx)
                    .and_then(Option::as_ref)
                    .ok_or(ExecError::RegionAllocationFailed { slot: frame.in_a })?;
                let cur_len = i64::try_from(region.storage.len())
                    .map_err(|_e| ExecError::RegionAllocationFailed { slot: frame.in_a })?;
                let slot = values
                    .get_mut(pc)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                *slot = EvalValue::ScalarI64(cur_len);
                pc += 1;
            }
            Opcode::RegionLoad => {
                let slot_idx = frame.in_a as usize;
                let idx_val = values
                    .get(frame.in_b as usize)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                let EvalValue::ScalarI64(idx_i64) = *idx_val else {
                    return Err(ExecError::InvalidOperandState(node_idx));
                };
                let region = regions
                    .get(slot_idx)
                    .and_then(Option::as_ref)
                    .ok_or(ExecError::RegionAllocationFailed { slot: frame.in_a })?;
                let len = region.storage.len();
                let Ok(idx_usize) = usize::try_from(idx_i64) else {
                    return Err(ExecError::RegionBoundsOutOfRange {
                        node_idx,
                        slot: frame.in_a,
                        index: idx_i64,
                        len,
                    });
                };
                let loaded = match &region.storage {
                    RegionStorage::U8(v) => v.get(idx_usize).copied().map(EvalValue::ScalarU8),
                    RegionStorage::I64(v) => v.get(idx_usize).copied().map(EvalValue::ScalarI64),
                    RegionStorage::F64(v) => v.get(idx_usize).copied().map(EvalValue::Scalar),
                }
                .ok_or(ExecError::RegionBoundsOutOfRange {
                    node_idx,
                    slot: frame.in_a,
                    index: idx_i64,
                    len,
                })?;
                let slot = values
                    .get_mut(pc)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                *slot = loaded;
                pc += 1;
            }
            Opcode::RegionStore => {
                let slot_idx = frame.in_a as usize;
                let (idx_node, val_node) = frame.decode_u16_pair();
                let idx_val = values
                    .get(idx_node as usize)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                let EvalValue::ScalarI64(idx_i64) = *idx_val else {
                    return Err(ExecError::InvalidOperandState(node_idx));
                };
                let store_val = values
                    .get(val_node as usize)
                    .cloned()
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                let region = regions
                    .get_mut(slot_idx)
                    .and_then(Option::as_mut)
                    .ok_or(ExecError::RegionAllocationFailed { slot: frame.in_a })?;
                let len = region.storage.len();
                let Ok(idx_usize) = usize::try_from(idx_i64) else {
                    return Err(ExecError::RegionBoundsOutOfRange {
                        node_idx,
                        slot: frame.in_a,
                        index: idx_i64,
                        len,
                    });
                };
                match (&mut region.storage, store_val) {
                    (RegionStorage::U8(v), EvalValue::ScalarU8(b)) => {
                        let dst =
                            v.get_mut(idx_usize)
                                .ok_or(ExecError::RegionBoundsOutOfRange {
                                    node_idx,
                                    slot: frame.in_a,
                                    index: idx_i64,
                                    len,
                                })?;
                        *dst = b;
                    }
                    (RegionStorage::I64(v), EvalValue::ScalarI64(x)) => {
                        let dst =
                            v.get_mut(idx_usize)
                                .ok_or(ExecError::RegionBoundsOutOfRange {
                                    node_idx,
                                    slot: frame.in_a,
                                    index: idx_i64,
                                    len,
                                })?;
                        *dst = x;
                    }
                    (RegionStorage::F64(v), EvalValue::Scalar(x)) => {
                        let dst =
                            v.get_mut(idx_usize)
                                .ok_or(ExecError::RegionBoundsOutOfRange {
                                    node_idx,
                                    slot: frame.in_a,
                                    index: idx_i64,
                                    len,
                                })?;
                        *dst = x;
                    }
                    _ => return Err(ExecError::InvalidOperandState(node_idx)),
                }
                pc += 1;
            }
            Opcode::RegionFree | Opcode::RegionDrop => {
                let slot_idx = frame.in_a as usize;
                let region = regions
                    .get_mut(slot_idx)
                    .and_then(Option::as_mut)
                    .ok_or(ExecError::RegionAllocationFailed { slot: frame.in_a })?;
                region.storage.free_storage();
                pc += 1;
            }
            Opcode::RegionLease | Opcode::RegionEndLease => {
                pc += 1;
            }
            Opcode::RegionMove => {
                move_tier0_region_storage(&mut regions, frame.in_a, frame.in_b)?;
                pc += 1;
            }
            _ => {
                let val = eval_node_tier0(
                    program,
                    frame,
                    node_idx,
                    inputs,
                    &values,
                    &mut intermediate_bytes,
                )?;
                let slot = values
                    .get_mut(pc)
                    .ok_or(ExecError::InvalidOperandState(node_idx))?;
                *slot = val;
                pc += 1;
            }
        }
    }

    let output_bytes = take_output_region_bytes(&mut regions, output_region_slot);

    Ok((values, intermediate_bytes, output_bytes))
}

fn step_branch_target(
    program: &RulProgram,
    hdr: usize,
    branch_node: u32,
    header_to_end: &[usize],
    loop_fuel: &mut [u32],
) -> Result<usize, ExecError> {
    let hdr_frame = program
        .nodes
        .get(hdr)
        .ok_or(ExecError::InvalidOperandState(branch_node))?;
    if hdr_frame.opcode == Opcode::Loop {
        if hdr_frame.proof_idx > 0 {
            let proof = program
                .proofs
                .get(usize::from(hdr_frame.proof_idx - 1))
                .ok_or(ExecError::InvalidOperandState(branch_node))?;
            let fuel_slot = loop_fuel
                .get_mut(hdr)
                .ok_or(ExecError::InvalidOperandState(branch_node))?;
            if *fuel_slot == 0 {
                return Err(ExecError::LoopBoundExceeded {
                    loop_node: u32::try_from(hdr).unwrap_or(u32::MAX),
                    limit: proof.bound_hi,
                });
            }
            *fuel_slot -= 1;
        }
        Ok(hdr + 1)
    } else {
        let end_pc = header_to_end
            .get(hdr)
            .copied()
            .ok_or(ExecError::InvalidOperandState(branch_node))?;
        Ok(end_pc + 1)
    }
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
        Opcode::ConstI64 => Ok(EvalValue::ScalarI64(frame.decode_const_i64())),
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
        Opcode::Add
        | Opcode::Sub
        | Opcode::Mul
        | Opcode::Div
        | Opcode::Rem
        | Opcode::Min
        | Opcode::Max => {
            let left = values
                .get(frame.in_a as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            let right = values
                .get(frame.in_b as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            eval_arith_tier0(frame.opcode, left, right, node_idx, intermediate_bytes)
        }
        Opcode::BitAnd
        | Opcode::BitOr
        | Opcode::BitXor
        | Opcode::Shl
        | Opcode::ShrU
        | Opcode::ShrS => {
            let left = values
                .get(frame.in_a as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            let right = values
                .get(frame.in_b as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            eval_bitwise_tier0(frame.opcode, left, right, node_idx)
        }
        Opcode::MaskNot => {
            let src = values
                .get(frame.in_a as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            let EvalValue::ScalarMask(b) = *src else {
                return Err(ExecError::InvalidOperandState(node_idx));
            };
            Ok(EvalValue::ScalarMask(!b))
        }
        Opcode::Cast => {
            let src = values
                .get(frame.in_a as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            eval_cast_tier0(src, frame.type_lane.elem, node_idx)
        }
        Opcode::CmpLtU => {
            let left = values
                .get(frame.in_a as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            let right = values
                .get(frame.in_b as usize)
                .ok_or(ExecError::InvalidOperandState(node_idx))?;
            let (EvalValue::ScalarI64(a), EvalValue::ScalarI64(b)) = (left, right) else {
                return Err(ExecError::InvalidOperandState(node_idx));
            };
            #[expect(
                clippy::cast_sign_loss,
                reason = "CmpLtU compares 64-bit two's-complement bit patterns as unsigned u64"
            )]
            let res = (*a as u64) < (*b as u64);
            Ok(EvalValue::ScalarMask(res))
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
                (EvalValue::Scalar(a), EvalValue::Scalar(b)) => {
                    Ok(EvalValue::ScalarMask(cmp(*a, *b)))
                }
                (EvalValue::ScalarI64(a), EvalValue::ScalarI64(b)) => {
                    let res = match frame.opcode {
                        Opcode::CmpGt => *a > *b,
                        Opcode::CmpLt => *a < *b,
                        _ => *a == *b,
                    };
                    Ok(EvalValue::ScalarMask(res))
                }
                (EvalValue::ScalarU8(a), EvalValue::ScalarU8(b)) => {
                    let res = match frame.opcode {
                        Opcode::CmpGt => *a > *b,
                        Opcode::CmpLt => *a < *b,
                        _ => *a == *b,
                    };
                    Ok(EvalValue::ScalarMask(res))
                }
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
            match (left, right) {
                (EvalValue::ScalarMask(a), EvalValue::ScalarMask(b)) => {
                    let res = match frame.opcode {
                        Opcode::MaskAnd => *a && *b,
                        _ => *a || *b,
                    };
                    Ok(EvalValue::ScalarMask(res))
                }
                (EvalValue::ColumnMask(a), EvalValue::ColumnMask(b)) => {
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
                }
                _ => Err(ExecError::InvalidOperandState(node_idx)),
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
                (EvalValue::ScalarMask(m), EvalValue::Scalar(s)) => {
                    Ok(EvalValue::Scalar(if *m { *s } else { 0.0 }))
                }
                (EvalValue::ScalarMask(m), EvalValue::ScalarI64(v)) => {
                    Ok(EvalValue::ScalarI64(if *m { *v } else { 0 }))
                }
                (EvalValue::ScalarMask(m), EvalValue::ScalarU8(v)) => {
                    Ok(EvalValue::ScalarU8(if *m { *v } else { 0 }))
                }
                (EvalValue::ScalarMask(m), EvalValue::ScalarMask(b)) => {
                    Ok(EvalValue::ScalarMask(*m && *b))
                }
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
        Opcode::LocalGet
        | Opcode::LocalSet
        | Opcode::Block
        | Opcode::Loop
        | Opcode::If
        | Opcode::Else
        | Opcode::End
        | Opcode::Br
        | Opcode::BrIf
        | Opcode::RegionNew
        | Opcode::RegionGrow
        | Opcode::RegionLen
        | Opcode::RegionLoad
        | Opcode::RegionStore
        | Opcode::RegionFree
        | Opcode::RegionLease
        | Opcode::RegionEndLease
        | Opcode::RegionMove
        | Opcode::RegionDrop
        | Opcode::Func
        | Opcode::Return
        | Opcode::Call
        | Opcode::CallArg
        | Opcode::ConstDataLen
        | Opcode::ConstDataLoad
        | Opcode::BrTable
        | Opcode::CallIndirect => Err(ExecError::UnsupportedInColumnarTier),
    }
}

fn eval_bitwise_tier0<'a>(
    opcode: Opcode,
    left: &EvalValue<'a>,
    right: &EvalValue<'a>,
    node_idx: u32,
) -> Result<EvalValue<'a>, ExecError> {
    match (left, right) {
        (EvalValue::ScalarI64(a), EvalValue::ScalarI64(b)) => {
            #[expect(
                clippy::cast_sign_loss,
                reason = "bitwise and logical shift operate on exact 64-bit two's-complement patterns"
            )]
            let (au, shift) = (*a as u64, ((*b as u64) & 63) as u32);
            let res = match opcode {
                Opcode::BitAnd => *a & *b,
                Opcode::BitOr => *a | *b,
                Opcode::BitXor => *a ^ *b,
                Opcode::Shl => a.wrapping_shl(shift),
                Opcode::ShrU => {
                    #[expect(
                        clippy::cast_possible_wrap,
                        reason = "reinterpreting logical right-shifted u64 pattern as i64"
                    )]
                    let signed = (au >> shift) as i64;
                    signed
                }
                Opcode::ShrS => a.wrapping_shr(shift),
                _ => return Err(ExecError::InvalidOperandState(node_idx)),
            };
            Ok(EvalValue::ScalarI64(res))
        }
        (EvalValue::ScalarU8(a), EvalValue::ScalarU8(b)) => {
            let shift = u32::from(*b & 7);
            let res = match opcode {
                Opcode::BitAnd => *a & *b,
                Opcode::BitOr => *a | *b,
                Opcode::BitXor => *a ^ *b,
                Opcode::Shl => a.wrapping_shl(shift),
                Opcode::ShrU => a.wrapping_shr(shift),
                _ => return Err(ExecError::InvalidOperandState(node_idx)),
            };
            Ok(EvalValue::ScalarU8(res))
        }
        _ => Err(ExecError::InvalidOperandState(node_idx)),
    }
}

fn eval_cast_tier0<'a>(
    src: &EvalValue<'a>,
    dst_elem: ElemType,
    node_idx: u32,
) -> Result<EvalValue<'a>, ExecError> {
    match (src, dst_elem) {
        (EvalValue::ScalarI64(v), ElemType::U8) => {
            #[expect(
                clippy::cast_sign_loss,
                reason = "I64 -> U8 cast truncates to exact low 8 bits"
            )]
            let byte = (*v as u64 & 0xFF) as u8;
            Ok(EvalValue::ScalarU8(byte))
        }
        (EvalValue::ScalarU8(v), ElemType::I64) => Ok(EvalValue::ScalarI64(i64::from(*v))),
        (EvalValue::Scalar(x), ElemType::I64) => {
            const MIN_F64: f64 = -9_223_372_036_854_775_808.0; // -2^63 (exact)
            const MAX_EXCL_F64: f64 = 9_223_372_036_854_775_808.0; // +2^63 (exact)
            if *x >= MIN_F64 && *x < MAX_EXCL_F64 {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "bounds-checked strictly within [-2^63, 2^63) before truncating toward zero"
                )]
                let val = x.trunc() as i64;
                Ok(EvalValue::ScalarI64(val))
            } else {
                Err(ExecError::CastOutOfRange { node_idx })
            }
        }
        (EvalValue::Scalar(x), ElemType::U8) => {
            let t = x.trunc();
            if t.is_finite() && (0.0..=255.0).contains(&t) {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "bounds-checked strictly within [0.0, 255.0] after truncating toward zero"
                )]
                let val = t as u8;
                Ok(EvalValue::ScalarU8(val))
            } else {
                Err(ExecError::CastOutOfRange { node_idx })
            }
        }
        (EvalValue::ScalarI64(v), ElemType::F64) => {
            #[expect(
                clippy::cast_precision_loss,
                reason = "I64 -> F64 cast rounds to IEEE-754 nearest-even by specification"
            )]
            let f = *v as f64;
            Ok(EvalValue::Scalar(f))
        }
        (EvalValue::ScalarU8(v), ElemType::F64) => Ok(EvalValue::Scalar(f64::from(*v))),
        (EvalValue::ScalarMask(b), ElemType::I64) => Ok(EvalValue::ScalarI64(i64::from(*b))),
        _ => Err(ExecError::InvalidOperandState(node_idx)),
    }
}

fn eval_arith_tier0<'a>(
    opcode: Opcode,
    left: &EvalValue<'a>,
    right: &EvalValue<'a>,
    node_idx: u32,
    intermediate_bytes: &mut usize,
) -> Result<EvalValue<'a>, ExecError> {
    if let (EvalValue::ScalarI64(a), EvalValue::ScalarI64(b)) = (left, right) {
        return match opcode {
            Opcode::Add => Ok(EvalValue::ScalarI64(a.wrapping_add(*b))),
            Opcode::Sub => Ok(EvalValue::ScalarI64(a.wrapping_sub(*b))),
            Opcode::Mul => Ok(EvalValue::ScalarI64(a.wrapping_mul(*b))),
            Opcode::Div => a
                .checked_div(*b)
                .map(EvalValue::ScalarI64)
                .ok_or(ExecError::IntegerDivisionError { node_idx }),
            Opcode::Rem => a
                .checked_rem(*b)
                .map(EvalValue::ScalarI64)
                .ok_or(ExecError::IntegerDivisionError { node_idx }),
            _ => Err(ExecError::InvalidOperandState(node_idx)),
        };
    }
    if let (EvalValue::ScalarU8(a), EvalValue::ScalarU8(b)) = (left, right) {
        return match opcode {
            Opcode::Add => Ok(EvalValue::ScalarU8(a.wrapping_add(*b))),
            Opcode::Sub => Ok(EvalValue::ScalarU8(a.wrapping_sub(*b))),
            Opcode::Mul => Ok(EvalValue::ScalarU8(a.wrapping_mul(*b))),
            Opcode::Div => a
                .checked_div(*b)
                .map(EvalValue::ScalarU8)
                .ok_or(ExecError::IntegerDivisionError { node_idx }),
            Opcode::Rem => a
                .checked_rem(*b)
                .map(EvalValue::ScalarU8)
                .ok_or(ExecError::IntegerDivisionError { node_idx }),
            _ => Err(ExecError::InvalidOperandState(node_idx)),
        };
    }
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

    if program.uses_structured_control_or_locals() {
        return Err(ExecError::UnsupportedInColumnarTier);
    }

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
        output_bytes: None,
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

    if program.uses_structured_control_or_locals() {
        return Err(ExecError::UnsupportedInColumnarTier);
    }

    let elements_processed = validate_inputs(program, inputs)?;

    let t_exec = Instant::now();
    let live = compute_live_nodes(program);

    // Check if the live DAG is eligible for Zero-Allocation L1-Cache Tile Fusion.
    if let Some(plan) = build_l1_tile_plan(program, &live) {
        let scalar_out = execute_l1_tile_plan(program, inputs, &plan, zonemap_a0)?;
        let exec_ns = u64::try_from(t_exec.elapsed().as_nanos()).unwrap_or(u64::MAX);

        return Ok(ExecutionReport {
            output: RuntimeValue::Scalar(scalar_out),
            output_bytes: None,
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
            Opcode::FilterCompact
            | Opcode::Gather
            | Opcode::ConstI64
            | Opcode::LocalGet
            | Opcode::LocalSet
            | Opcode::MaskNot
            | Opcode::Cast
            | Opcode::BitAnd
            | Opcode::BitOr
            | Opcode::BitXor
            | Opcode::Shl
            | Opcode::ShrU
            | Opcode::Rem
            | Opcode::ShrS
            | Opcode::CmpLtU
            | Opcode::Block
            | Opcode::Loop
            | Opcode::If
            | Opcode::Else
            | Opcode::End
            | Opcode::Br
            | Opcode::BrIf
            | Opcode::RegionNew
            | Opcode::RegionGrow
            | Opcode::RegionLen
            | Opcode::RegionLoad
            | Opcode::RegionStore
            | Opcode::RegionFree
            | Opcode::RegionLease
            | Opcode::RegionEndLease
            | Opcode::RegionMove
            | Opcode::RegionDrop
            | Opcode::Func
            | Opcode::Return
            | Opcode::Call
            | Opcode::CallArg
            | Opcode::ConstDataLen
            | Opcode::ConstDataLoad
            | Opcode::BrTable
            | Opcode::CallIndirect => return None,
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
        (RuntimeValue::TypedScalar(x), RuntimeValue::TypedScalar(y)) => x == y,
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
    let mut entries: Vec<NodeTraceEntry> = Vec::with_capacity(program.nodes.len());
    let (values, intermediate_bytes) = if program.uses_structured_control_or_locals() {
        let (vals, inter_bytes, _out_bytes) = eval_structured_tier0(program, inputs, &[])?;
        for (idx, frame) in program.nodes.iter().enumerate() {
            let node_idx = u32::try_from(idx).unwrap_or(u32::MAX);
            let is_live = live.get(idx).copied().unwrap_or(false);
            let val = vals.get(idx).unwrap_or(&EvalValue::Unused);
            entries.push(summarize_eval_value(node_idx, frame.opcode, is_live, val));
        }
        (vals, inter_bytes)
    } else {
        let mut vals: Vec<EvalValue<'_>> = Vec::with_capacity(program.nodes.len());
        let mut inter_bytes = 0usize;
        for (idx, frame) in program.nodes.iter().enumerate() {
            let node_idx = u32::try_from(idx).unwrap_or(u32::MAX);
            let is_live = live.get(idx).copied().unwrap_or(false);
            let val = eval_node_tier0(program, frame, node_idx, inputs, &vals, &mut inter_bytes)?;
            let entry = summarize_eval_value(node_idx, frame.opcode, is_live, &val);
            entries.push(entry);
            vals.push(val);
        }
        (vals, inter_bytes)
    };

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
        EvalValue::ScalarI64(v) => {
            #[expect(
                clippy::cast_precision_loss,
                reason = "human-readable trace summary converts i64 scalar to approximate f64 display metric"
            )]
            let approx = *v as f64;
            NodeTraceEntry {
                node_idx,
                opcode,
                is_live,
                value_kind: "ScalarI64",
                element_count: 1,
                min_val: approx,
                max_val: approx,
                sum_or_true_count: approx,
                nan_or_inf_count: 0,
                sample_head: vec![approx],
            }
        }
        EvalValue::ScalarU8(v) => {
            let approx = f64::from(*v);
            NodeTraceEntry {
                node_idx,
                opcode,
                is_live,
                value_kind: "ScalarU8",
                element_count: 1,
                min_val: approx,
                max_val: approx,
                sum_or_true_count: approx,
                nan_or_inf_count: 0,
                sample_head: vec![approx],
            }
        }
        EvalValue::ScalarMask(b) => {
            let bit = if *b { 1.0 } else { 0.0 };
            NodeTraceEntry {
                node_idx,
                opcode,
                is_live,
                value_kind: "ScalarMask",
                element_count: 1,
                min_val: bit,
                max_val: bit,
                sum_or_true_count: bit,
                nan_or_inf_count: 0,
                sample_head: vec![bit],
            }
        }
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
