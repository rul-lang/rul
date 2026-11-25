//! Zero-Runtime Standalone Native AOT Compiler (`rul compile`).
//!
//! Compiles a verified `.rl` binary instruction graph directly into a standalone,
//! statically-linked Linux `x86_64` `ELF64` executable in microseconds (`< 1 ms`)
//! using Copy-and-Patch machine code stencils — requiring **zero LLVM passes**,
//! **zero `libc`**, and **zero Rul runtime** at execution time.

use std::fmt;
use std::fs::{self, File};
use std::io::{ErrorKind, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Instant;

use crate::bytecode::{
    func_frame_units, ElemType, FuncSigInfo, Opcode, ProofEntry, ProofRuleKind, RegionRole,
    RulProgram, CALL_ARG_SENTINEL, MAX_CALL_DEPTH, MAX_REGION_SLOTS, MAX_STACK_UNITS,
};
use crate::verifier::{verify_program, ValueShape, VerifiedSummary, VerifyDiagnostic};

/// Base virtual address for the standalone `x86_64` `ELF64` executable.
const ELF_BASE_VADDR: u64 = 0x0040_0000;
/// Size of the 64-bit ELF file header (`Elf64_Ehdr`).
const ELF_EHDR_SIZE: usize = 64;
/// Size of a single 64-bit ELF program header (`Elf64_Phdr`).
const ELF_PHDR_SIZE: usize = 56;
/// Offset where machine code entry `_start` begins (`64 + 56 = 120`).
const ELF_CODE_OFFSET: usize = ELF_EHDR_SIZE + ELF_PHDR_SIZE;

/// Exit status code emitted by the standalone ELF binary when `AssertContract` fails (`NaN`/`Inf`) or a bounded loop exceeds `LoopBound`.
pub const AOT_EXIT_CONTRACT_FAILED: i32 = 70;
/// Exit status code emitted by the standalone ELF binary when `Gather` or `RegionLoad`/`RegionStore` index is out of bounds.
pub const AOT_EXIT_GATHER_OOB: i32 = 71;
/// Exit status code emitted by the standalone ELF binary on checked integer division/remainder by zero or `i64::MIN / -1`.
pub const AOT_EXIT_INT_DIV_ERR: i32 = 72;
/// Exit status code emitted by the standalone ELF binary on `NaN` or out-of-range float-to-integer `Cast`.
pub const AOT_EXIT_CAST_ERR: i32 = 73;
/// Exit status code emitted by the standalone ELF binary on region allocation/capacity/length failure.
pub const AOT_EXIT_REGION_OOM: i32 = 74;
/// Exit status code emitted by the standalone ELF binary when call depth (`MAX_CALL_DEPTH`) or stack budget (`MAX_STACK_UNITS`) is exceeded.
pub const AOT_EXIT_STACK_OVERFLOW: i32 = 75;

/// Report returned after compiling a `.rl` program into a standalone native executable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AotCompileReport {
    /// Total size of the standalone `ELF64` executable in bytes.
    pub elf_bytes: usize,
    /// Number of `x86_64` machine code bytes emitted in the `.text` segment.
    pub machine_code_bytes: usize,
    /// `true` if 256-bit AVX2 `YMM` vector stencils were emitted for the primary loop.
    pub avx2_vectorized: bool,
    /// Time spent in $O(N)$ verification in nanoseconds.
    pub verify_ns: u64,
    /// Time spent patching native machine code stencils in nanoseconds.
    pub codegen_ns: u64,
    /// Total compile latency (`verify_ns + codegen_ns`) in nanoseconds.
    pub total_compile_ns: u64,
}

/// Errors that can occur during AOT compilation or standalone native execution.
#[derive(Debug)]
pub enum AotError {
    /// Static $O(N)$ verification failed prior to codegen.
    VerificationFailed(VerifyDiagnostic),
    /// Program structure is invalid or unsupported for AOT emission.
    UnsupportedGraph(&'static str),
    /// File or process I/O error.
    Io(std::io::Error),
    /// `AssertContract` detected a non-finite (`NaN`/`Inf`) scalar output in the native binary.
    ContractAssertionFailed,
    /// `Gather` or `RegionLoad`/`RegionStore` index out-of-bounds detected by the native binary (`exit 71`).
    GatherOutOfBounds,
    /// Checked integer division or remainder fault detected by the native binary (`exit 72`).
    IntegerDivisionError,
    /// Float-to-integer `Cast` out-of-range or `NaN` fault detected by the native binary (`exit 73`).
    CastOutOfRange,
    /// Region allocation, growth, reset, or input size limit failure detected by the native binary (`exit 74`).
    RegionAllocationFailed,
    /// Call depth (`MAX_CALL_DEPTH`) or stack budget (`MAX_STACK_UNITS`) exceeded by the native binary (`exit 75`).
    StackOverflow,
    /// Native binary exited with a non-zero status code or invalid output.
    ExecutionFailed(String),
}

impl fmt::Display for AotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VerificationFailed(d) => write!(f, "AOT verification failed: {d}"),
            Self::UnsupportedGraph(msg) => write!(f, "unsupported AOT graph: {msg}"),
            Self::Io(e) => write!(f, "AOT I/O error: {e}"),
            Self::ContractAssertionFailed => {
                write!(
                    f,
                    "AOT native binary contract assertion failed (non-finite output)"
                )
            }
            Self::GatherOutOfBounds => {
                write!(
                    f,
                    "AOT native binary detected out-of-bounds Gather or Region index"
                )
            }
            Self::IntegerDivisionError => {
                write!(
                    f,
                    "AOT native binary detected integer division or remainder error"
                )
            }
            Self::CastOutOfRange => {
                write!(
                    f,
                    "AOT native binary detected out-of-range float-to-integer Cast"
                )
            }
            Self::RegionAllocationFailed => {
                write!(
                    f,
                    "AOT native binary detected region allocation or length limit failure"
                )
            }
            Self::StackOverflow => {
                write!(
                    f,
                    "AOT native binary exceeded call depth or stack budget guard (exit 75)"
                )
            }
            Self::ExecutionFailed(msg) => write!(f, "AOT native execution failed: {msg}"),
        }
    }
}

impl std::error::Error for AotError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AotReduction {
    Sum,
    Max,
    Min,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AotNodePhase {
    Dead,
    Const,
    Column,
    Reduce,
    PostReduceScalar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AotReductionSpec {
    reduce_idx: usize,
    reduce_input_idx: usize,
    reduction: AotReduction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AotColMeta {
    origin_arena: Option<usize>,
    slice_offset: u64,
    col_len: u64,
}

/// Compiles a verified [`RulProgram`] into an in-memory Linux `x86_64` `ELF64` executable image.
///
/// The resulting binary has **zero runtime dependencies** (`statically linked`, no `libc`, no VM)
/// and executes directly on the OS kernel.
///
/// # Errors
/// Returns [`AotError`] if $O(N)$ verification fails or the graph structure is unsupported.
#[expect(
    clippy::too_many_lines,
    reason = "single-pass O(N) phase classifier and ELF64 assembler"
)]
pub fn compile_to_elf64_bytes(
    program: &RulProgram,
) -> Result<(Vec<u8>, AotCompileReport), AotError> {
    let t_ver = Instant::now();
    let summary = verify_program(program).map_err(AotError::VerificationFailed)?;
    let verify_ns = u64::try_from(t_ver.elapsed().as_nanos()).unwrap_or(u64::MAX);

    let t_cg = Instant::now();
    if program.uses_structured_control_or_locals() {
        return compile_structured_to_elf64_bytes(program, &summary, verify_ns, t_cg);
    }
    let live = compute_live_nodes(program);

    let out_idx = program.output_node as usize;
    let out_frame = program
        .nodes
        .get(out_idx)
        .ok_or(AotError::UnsupportedGraph("invalid output node index"))?;
    if out_frame.type_lane.elem != ElemType::F64 {
        return Err(AotError::UnsupportedGraph(
            "AOT binary requires F64 scalar or RingStream output",
        ));
    }

    let mut phases = vec![AotNodePhase::Dead; program.nodes.len()];
    let mut col_metas = vec![
        AotColMeta {
            origin_arena: None,
            slice_offset: 0,
            col_len: 0,
        };
        program.nodes.len()
    ];
    let mut reductions = Vec::new();
    let mut post_reduce_nodes = Vec::new();
    let mut filter_mask_node: Option<usize> = None;
    let mut has_gather = false;

    for (idx, frame) in program.nodes.iter().enumerate() {
        if !live.get(idx).copied().unwrap_or(false) {
            continue;
        }
        let phase_a = phases.get(frame.in_a as usize).copied();
        let phase_b = phases.get(frame.in_b as usize).copied();
        let phase = match frame.opcode {
            Opcode::ConstF64 => AotNodePhase::Const,
            Opcode::ArenaInput => {
                let arena_idx = frame.in_a as usize;
                let arena_len = program
                    .arenas
                    .get(arena_idx)
                    .map_or(0, |a| u64::from(a.element_count));
                if let Some(m) = col_metas.get_mut(idx) {
                    *m = AotColMeta {
                        origin_arena: Some(arena_idx),
                        slice_offset: 0,
                        col_len: arena_len,
                    };
                }
                AotNodePhase::Column
            }
            Opcode::ArenaSlice => {
                if phase_a != Some(AotNodePhase::Column) {
                    return Err(AotError::UnsupportedGraph("invalid ArenaSlice input phase"));
                }
                let src_meta = col_metas
                    .get(frame.in_a as usize)
                    .copied()
                    .unwrap_or(AotColMeta {
                        origin_arena: None,
                        slice_offset: 0,
                        col_len: 0,
                    });
                if src_meta.origin_arena.is_none() {
                    return Err(AotError::UnsupportedGraph(
                        "AOT ArenaSlice requires direct ArenaInput or ArenaSlice source",
                    ));
                }
                let proof = program
                    .proofs
                    .get(usize::from(frame.proof_idx.saturating_sub(1)))
                    .ok_or(AotError::UnsupportedGraph("missing slice ProofEntry"))?;
                let slice_len = u64::from(proof.bound_hi.saturating_sub(proof.bound_lo));
                if let Some(m) = col_metas.get_mut(idx) {
                    *m = AotColMeta {
                        origin_arena: src_meta.origin_arena,
                        slice_offset: src_meta
                            .slice_offset
                            .saturating_add(u64::from(proof.bound_lo)),
                        col_len: slice_len,
                    };
                }
                AotNodePhase::Column
            }
            Opcode::ArenaMutate => {
                if !matches!(phase_b, Some(AotNodePhase::Column | AotNodePhase::Const)) {
                    return Err(AotError::UnsupportedGraph(
                        "invalid ArenaMutate input phase",
                    ));
                }
                let arena_len = program
                    .arenas
                    .get(frame.in_a as usize)
                    .map_or(0, |a| u64::from(a.element_count));
                if let Some(m) = col_metas.get_mut(idx) {
                    *m = AotColMeta {
                        origin_arena: None,
                        slice_offset: 0,
                        col_len: arena_len,
                    };
                }
                AotNodePhase::Column
            }
            Opcode::FilterCompact => {
                if phase_a != Some(AotNodePhase::Column) || phase_b != Some(AotNodePhase::Column) {
                    return Err(AotError::UnsupportedGraph(
                        "invalid FilterCompact input phase",
                    ));
                }
                if filter_mask_node.is_some() {
                    return Err(AotError::UnsupportedGraph(
                        "AOT binary does not support multiple FilterCompact nodes",
                    ));
                }
                filter_mask_node = Some(frame.in_a as usize);
                let val_len = col_metas.get(frame.in_b as usize).map_or(0, |m| m.col_len);
                if let Some(m) = col_metas.get_mut(idx) {
                    *m = AotColMeta {
                        origin_arena: None,
                        slice_offset: 0,
                        col_len: val_len,
                    };
                }
                AotNodePhase::Column
            }
            Opcode::Gather => {
                if phase_a != Some(AotNodePhase::Column) || phase_b != Some(AotNodePhase::Column) {
                    return Err(AotError::UnsupportedGraph("invalid Gather input phase"));
                }
                let target_meta =
                    col_metas
                        .get(frame.in_a as usize)
                        .copied()
                        .unwrap_or(AotColMeta {
                            origin_arena: None,
                            slice_offset: 0,
                            col_len: 0,
                        });
                if target_meta.origin_arena.is_none() {
                    return Err(AotError::UnsupportedGraph(
                        "AOT Gather requires direct ArenaInput or ArenaSlice target",
                    ));
                }
                let idx_len = col_metas.get(frame.in_b as usize).map_or(0, |m| m.col_len);
                if let Some(m) = col_metas.get_mut(idx) {
                    *m = AotColMeta {
                        origin_arena: None,
                        slice_offset: 0,
                        col_len: idx_len,
                    };
                }
                has_gather = true;
                AotNodePhase::Column
            }
            Opcode::ReduceSum | Opcode::ReduceMax | Opcode::ReduceMin => {
                if phase_a != Some(AotNodePhase::Column) {
                    return Err(AotError::UnsupportedGraph(
                        "reduction input must be a Column node",
                    ));
                }
                let reduction = match frame.opcode {
                    Opcode::ReduceSum => AotReduction::Sum,
                    Opcode::ReduceMax => AotReduction::Max,
                    Opcode::ReduceMin => AotReduction::Min,
                    _ => AotReduction::Sum,
                };
                reductions.push(AotReductionSpec {
                    reduce_idx: idx,
                    reduce_input_idx: frame.in_a as usize,
                    reduction,
                });
                AotNodePhase::Reduce
            }
            Opcode::Abs
            | Opcode::Sqrt
            | Opcode::Floor
            | Opcode::Ceil
            | Opcode::Ln
            | Opcode::Exp => match phase_a {
                Some(AotNodePhase::Column) => {
                    let len_a = col_metas.get(frame.in_a as usize).map_or(0, |m| m.col_len);
                    if let Some(m) = col_metas.get_mut(idx) {
                        *m = AotColMeta {
                            origin_arena: None,
                            slice_offset: 0,
                            col_len: len_a,
                        };
                    }
                    AotNodePhase::Column
                }
                Some(
                    AotNodePhase::Const | AotNodePhase::Reduce | AotNodePhase::PostReduceScalar,
                ) => {
                    post_reduce_nodes.push(idx);
                    AotNodePhase::PostReduceScalar
                }
                _ => {
                    return Err(AotError::UnsupportedGraph(
                        "invalid input phase for unary math op",
                    ))
                }
            },
            Opcode::Add | Opcode::Sub | Opcode::Mul | Opcode::Div | Opcode::Min | Opcode::Max => {
                if phase_a == Some(AotNodePhase::Column) || phase_b == Some(AotNodePhase::Column) {
                    let a_ok = matches!(phase_a, Some(AotNodePhase::Column | AotNodePhase::Const));
                    let b_ok = matches!(phase_b, Some(AotNodePhase::Column | AotNodePhase::Const));
                    if !a_ok || !b_ok {
                        return Err(AotError::UnsupportedGraph(
                            "column operation depends on reduction output (requires multi-pass execution)",
                        ));
                    }
                    let len_a = col_metas.get(frame.in_a as usize).map_or(0, |m| m.col_len);
                    let len_b = col_metas.get(frame.in_b as usize).map_or(0, |m| m.col_len);
                    let col_len = if len_a > 0 { len_a } else { len_b };
                    if let Some(m) = col_metas.get_mut(idx) {
                        *m = AotColMeta {
                            origin_arena: None,
                            slice_offset: 0,
                            col_len,
                        };
                    }
                    AotNodePhase::Column
                } else {
                    let a_ok = matches!(
                        phase_a,
                        Some(
                            AotNodePhase::Const
                                | AotNodePhase::Reduce
                                | AotNodePhase::PostReduceScalar
                        )
                    );
                    let b_ok = matches!(
                        phase_b,
                        Some(
                            AotNodePhase::Const
                                | AotNodePhase::Reduce
                                | AotNodePhase::PostReduceScalar
                        )
                    );
                    if !a_ok || !b_ok {
                        return Err(AotError::UnsupportedGraph(
                            "invalid scalar binary input phase",
                        ));
                    }
                    post_reduce_nodes.push(idx);
                    AotNodePhase::PostReduceScalar
                }
            }
            Opcode::Fma => {
                let (in_b, in_c) = frame.decode_fma_bc();
                let phase_b = phases.get(in_b as usize).copied();
                let phase_c = phases.get(in_c as usize).copied();
                if phase_a == Some(AotNodePhase::Column)
                    || phase_b == Some(AotNodePhase::Column)
                    || phase_c == Some(AotNodePhase::Column)
                {
                    let a_ok = matches!(phase_a, Some(AotNodePhase::Column | AotNodePhase::Const));
                    let b_ok = matches!(phase_b, Some(AotNodePhase::Column | AotNodePhase::Const));
                    let c_ok = matches!(phase_c, Some(AotNodePhase::Column | AotNodePhase::Const));
                    if !a_ok || !b_ok || !c_ok {
                        return Err(AotError::UnsupportedGraph(
                            "column Fma op depends on reduction output",
                        ));
                    }
                    let len_a = col_metas.get(frame.in_a as usize).map_or(0, |m| m.col_len);
                    let len_b = col_metas.get(in_b as usize).map_or(0, |m| m.col_len);
                    let len_c = col_metas.get(in_c as usize).map_or(0, |m| m.col_len);
                    let col_len = if len_a > 0 {
                        len_a
                    } else if len_b > 0 {
                        len_b
                    } else {
                        len_c
                    };
                    if let Some(m) = col_metas.get_mut(idx) {
                        *m = AotColMeta {
                            origin_arena: None,
                            slice_offset: 0,
                            col_len,
                        };
                    }
                    AotNodePhase::Column
                } else {
                    let a_ok = matches!(
                        phase_a,
                        Some(
                            AotNodePhase::Const
                                | AotNodePhase::Reduce
                                | AotNodePhase::PostReduceScalar
                        )
                    );
                    let b_ok = matches!(
                        phase_b,
                        Some(
                            AotNodePhase::Const
                                | AotNodePhase::Reduce
                                | AotNodePhase::PostReduceScalar
                        )
                    );
                    let c_ok = matches!(
                        phase_c,
                        Some(
                            AotNodePhase::Const
                                | AotNodePhase::Reduce
                                | AotNodePhase::PostReduceScalar
                        )
                    );
                    if !a_ok || !b_ok || !c_ok {
                        return Err(AotError::UnsupportedGraph("invalid scalar Fma input phase"));
                    }
                    post_reduce_nodes.push(idx);
                    AotNodePhase::PostReduceScalar
                }
            }
            Opcode::CmpGt
            | Opcode::CmpLt
            | Opcode::CmpEq
            | Opcode::MaskAnd
            | Opcode::MaskOr
            | Opcode::SelectZero => {
                if phase_a != Some(AotNodePhase::Column) && phase_b != Some(AotNodePhase::Column) {
                    return Err(AotError::UnsupportedGraph(
                        "AOT binary supports comparison/mask/select ops only on Column shape",
                    ));
                }
                let a_ok = matches!(phase_a, Some(AotNodePhase::Column | AotNodePhase::Const));
                let b_ok = matches!(phase_b, Some(AotNodePhase::Column | AotNodePhase::Const));
                if !a_ok || !b_ok {
                    return Err(AotError::UnsupportedGraph(
                        "column comparison/mask/select op depends on reduction output",
                    ));
                }
                let len_a = col_metas.get(frame.in_a as usize).map_or(0, |m| m.col_len);
                let len_b = col_metas.get(frame.in_b as usize).map_or(0, |m| m.col_len);
                let col_len = if len_a > 0 { len_a } else { len_b };
                if let Some(m) = col_metas.get_mut(idx) {
                    *m = AotColMeta {
                        origin_arena: None,
                        slice_offset: 0,
                        col_len,
                    };
                }
                AotNodePhase::Column
            }
            Opcode::RingEnqueue | Opcode::AssertContract => {
                if !matches!(
                    phase_a,
                    Some(
                        AotNodePhase::Const | AotNodePhase::Reduce | AotNodePhase::PostReduceScalar
                    )
                ) {
                    return Err(AotError::UnsupportedGraph(
                        "AssertContract/RingEnqueue requires scalar/reduction input",
                    ));
                }
                post_reduce_nodes.push(idx);
                AotNodePhase::PostReduceScalar
            }
            Opcode::ConstI64
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
            | Opcode::CallArg => {
                return Err(AotError::UnsupportedGraph(
                    "structured control-flow or region opcode reached columnar AOT pipeline",
                ));
            }
        };
        if let Some(slot) = phases.get_mut(idx) {
            *slot = phase;
        }
    }

    if reductions.is_empty() {
        return Err(AotError::UnsupportedGraph(
            "AOT binary requires at least one ReduceSum, ReduceMax, or ReduceMin node",
        ));
    }
    if (filter_mask_node.is_some() || has_gather) && reductions.len() > 1 {
        return Err(AotError::UnsupportedGraph(
            "AOT binary does not support FilterCompact or Gather with multiple reductions",
        ));
    }
    if !matches!(
        phases.get(out_idx).copied(),
        Some(AotNodePhase::Reduce | AotNodePhase::PostReduceScalar)
    ) {
        return Err(AotError::UnsupportedGraph(
            "AOT binary requires output_node to be a reduction or post-reduction scalar expression",
        ));
    }

    // Backward pass to determine which Column nodes are actually evaluated elementwise in the loop
    // (excluding nodes used solely as random-access `Gather` targets or `ArenaSlice` base inputs).
    let mut eval_in_loop = vec![false; program.nodes.len()];
    for spec in &reductions {
        if let Some(slot) = eval_in_loop.get_mut(spec.reduce_input_idx) {
            *slot = true;
        }
    }
    if let Some(mask_idx) = filter_mask_node {
        if let Some(slot) = eval_in_loop.get_mut(mask_idx) {
            *slot = true;
        }
    }
    for idx in (0..program.nodes.len()).rev() {
        if !eval_in_loop.get(idx).copied().unwrap_or(false) {
            continue;
        }
        let Some(frame) = program.nodes.get(idx) else {
            continue;
        };
        match frame.opcode {
            Opcode::ArenaInput | Opcode::ConstF64 | Opcode::ArenaSlice => {}
            Opcode::Gather | Opcode::ArenaMutate => {
                if let Some(b) = eval_in_loop.get_mut(frame.in_b as usize) {
                    *b = true;
                }
            }
            Opcode::Abs
            | Opcode::Sqrt
            | Opcode::Floor
            | Opcode::Ceil
            | Opcode::Ln
            | Opcode::Exp => {
                if let Some(a) = eval_in_loop.get_mut(frame.in_a as usize) {
                    *a = true;
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
            | Opcode::FilterCompact => {
                if let Some(a) = eval_in_loop.get_mut(frame.in_a as usize) {
                    *a = true;
                }
                if let Some(b) = eval_in_loop.get_mut(frame.in_b as usize) {
                    *b = true;
                }
            }
            Opcode::Fma => {
                let (in_b, in_c) = frame.decode_fma_bc();
                if let Some(a) = eval_in_loop.get_mut(frame.in_a as usize) {
                    *a = true;
                }
                if let Some(b) = eval_in_loop.get_mut(in_b as usize) {
                    *b = true;
                }
                if let Some(c) = eval_in_loop.get_mut(in_c as usize) {
                    *c = true;
                }
            }
            Opcode::ReduceSum
            | Opcode::ReduceMax
            | Opcode::ReduceMin
            | Opcode::RingEnqueue
            | Opcode::AssertContract
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
            | Opcode::CallArg => {}
        }
    }

    let mut loop_col_len: Option<u64> = None;
    for (idx, &in_loop) in eval_in_loop.iter().enumerate() {
        if !in_loop || phases.get(idx).copied() != Some(AotNodePhase::Column) {
            continue;
        }
        let c_len = col_metas.get(idx).map_or(0, |m| m.col_len);
        if let Some(prev) = loop_col_len {
            if prev != c_len {
                return Err(AotError::UnsupportedGraph(
                    "AOT binary requires all loop-evaluated columns to have matching length",
                ));
            }
        } else {
            loop_col_len = Some(c_len);
        }
    }

    // Compute byte offsets for each Arena inside the flat arena input file.
    let mut arena_byte_offsets = Vec::with_capacity(program.arenas.len());
    let mut total_arena_bytes: u64 = 0;
    for arena in &program.arenas {
        arena_byte_offsets.push(total_arena_bytes);
        total_arena_bytes =
            total_arena_bytes.saturating_add(u64::from(arena.element_count).saturating_mul(8));
    }
    let domain_start: u64 = 0;
    let final_domain_end: u64 = loop_col_len.unwrap_or(1);
    let avx2_vectorized = !has_gather
        && filter_mask_node.is_none()
        && final_domain_end.saturating_sub(domain_start) >= 4;

    let code = emit_x86_64_machine_code(
        program,
        &live,
        &phases,
        &col_metas,
        &eval_in_loop,
        &reductions,
        &post_reduce_nodes,
        domain_start,
        final_domain_end,
        filter_mask_node,
        avx2_vectorized,
        &arena_byte_offsets,
        total_arena_bytes,
    )?;

    let machine_code_bytes = code.len();
    let total_elf_size = ELF_CODE_OFFSET + machine_code_bytes;
    let mut elf = Vec::with_capacity(total_elf_size);
    write_elf64_headers(&mut elf, total_elf_size as u64);
    elf.extend_from_slice(&code);

    let codegen_ns = u64::try_from(t_cg.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let report = AotCompileReport {
        elf_bytes: elf.len(),
        machine_code_bytes,
        avx2_vectorized,
        verify_ns,
        codegen_ns,
        total_compile_ns: verify_ns.saturating_add(codegen_ns),
    };

    Ok((elf, report))
}

/// Compiles a `.rl` program and writes the standalone executable (`chmod 0755`) to `output_path`.
///
/// # Errors
/// Returns [`AotError`] if verification, codegen, or file writing fails.
pub fn compile_rl_to_executable(
    program: &RulProgram,
    output_path: impl AsRef<Path>,
) -> Result<AotCompileReport, AotError> {
    let (elf_bytes, report) = compile_to_elf64_bytes(program)?;
    let path = output_path.as_ref();

    // Write to a temporary sibling file first and atomically rename to avoid `ETXTBSY` on Linux.
    let tmp_path = path.with_extension("rl_aot_tmp");
    {
        let mut file = File::create(&tmp_path).map_err(AotError::Io)?;
        file.write_all(&elf_bytes).map_err(AotError::Io)?;
        file.sync_all().map_err(AotError::Io)?;
    }
    let perms = fs::Permissions::from_mode(0o755);
    fs::set_permissions(&tmp_path, perms).map_err(AotError::Io)?;
    fs::rename(&tmp_path, path).map_err(AotError::Io)?;

    Ok(report)
}

/// Serializes one or more `f64` arena slices into a flat binary file ready for zero-copy `mmap`
/// by a compiled Rul standalone executable.
///
/// # Errors
/// Returns [`AotError::Io`] if writing the arena file fails.
pub fn write_arena_data_file(
    data_path: impl AsRef<Path>,
    inputs: &[&[f64]],
) -> Result<usize, AotError> {
    let total_elems: usize = inputs.iter().map(|s| s.len()).sum();
    let mut raw = Vec::with_capacity(total_elems * 8);
    for slice in inputs {
        for &val in *slice {
            raw.extend_from_slice(&val.to_le_bytes());
        }
    }
    let mut file = File::create(data_path.as_ref()).map_err(AotError::Io)?;
    file.write_all(&raw).map_err(AotError::Io)?;
    Ok(raw.len())
}

/// Executes a compiled standalone Rul `ELF64` binary as an independent OS process on `data_path`
/// and reads back its 8-byte `f64` scalar output from `stdout`.
///
/// # Errors
/// Returns [`AotError`] if the process fails, contracts are violated, or output is malformed.
pub fn run_compiled_executable(
    exe_path: impl AsRef<Path>,
    data_path: impl AsRef<Path>,
) -> Result<f64, AotError> {
    let exe_ref = exe_path.as_ref();
    let data_ref = data_path.as_ref();
    let mut attempt = 0u32;
    let output = loop {
        match Command::new(exe_ref).arg(data_ref).output() {
            Ok(out) => break out,
            Err(e) if e.raw_os_error() == Some(26) && attempt < 32 => {
                attempt += 1;
                std::thread::sleep(std::time::Duration::from_micros(250));
            }
            Err(e) => return Err(AotError::Io(e)),
        }
    };

    if !output.status.success() {
        match output.status.code() {
            Some(AOT_EXIT_CONTRACT_FAILED) => return Err(AotError::ContractAssertionFailed),
            Some(AOT_EXIT_GATHER_OOB) => return Err(AotError::GatherOutOfBounds),
            other => {
                return Err(AotError::ExecutionFailed(format!(
                    "standalone binary exited with status {other:?}"
                )))
            }
        }
    }

    let bytes: [u8; 8] = output.stdout.as_slice().try_into().map_err(|_e| {
        AotError::ExecutionFailed(format!(
            "expected 8-byte f64 output on stdout, got {} bytes",
            output.stdout.len()
        ))
    })?;
    Ok(f64::from_le_bytes(bytes))
}

/// Executes a compiled standalone structured Rul `ELF64` binary with caller-supplied `stdin_bytes`
/// and returns its raw `stdout` bytes (either the 8-byte scalar bit pattern or the declared output region's bytes).
///
/// Fails closed if the binary exits with a non-zero status or writes any bytes to stdout on a fail-closed exit.
///
/// # Errors
/// Returns [`AotError`] matching exit codes `70..=75`, I/O errors, or unexpected exit status.
pub fn run_compiled_structured_executable(
    exe_path: impl AsRef<Path>,
    stdin_bytes: &[u8],
) -> Result<Vec<u8>, AotError> {
    let exe_ref = exe_path.as_ref();
    let mut attempt = 0u32;
    let mut child = loop {
        match Command::new(exe_ref)
            .stdin(if stdin_bytes.is_empty() {
                Stdio::null()
            } else {
                Stdio::piped()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => break child,
            Err(e) if e.raw_os_error() == Some(26) && attempt < 32 => {
                attempt += 1;
                std::thread::sleep(std::time::Duration::from_micros(250));
            }
            Err(e) => return Err(AotError::Io(e)),
        }
    };

    // Write stdin on a scoped thread while the main thread drains stdout/stderr via
    // `wait_with_output()`, preventing pipe-buffer deadlock when both stdin and stdout exceed 64 KiB
    // (including when an output-only binary never reads stdin).
    let child_stdin = child.stdin.take();
    let (write_res, wait_res) = std::thread::scope(|s| {
        let writer = child_stdin.map(|mut stdin| {
            s.spawn(move || -> Result<(), std::io::Error> {
                if let Err(e) = stdin.write_all(stdin_bytes) {
                    if e.kind() != ErrorKind::BrokenPipe {
                        return Err(e);
                    }
                }
                Ok(())
            })
        });
        let wait_out = child.wait_with_output();
        let write_out = match writer {
            Some(handle) => match handle.join() {
                Ok(res) => res,
                Err(_) => Ok(()),
            },
            None => Ok(()),
        };
        (write_out, wait_out)
    });
    write_res.map_err(AotError::Io)?;
    let output = wait_res.map_err(AotError::Io)?;

    if !output.status.success() {
        if !output.stdout.is_empty() {
            return Err(AotError::ExecutionFailed(format!(
                "standalone binary wrote {} unexpected stdout bytes on fail-closed exit {:?}",
                output.stdout.len(),
                output.status.code()
            )));
        }
        match output.status.code() {
            Some(AOT_EXIT_CONTRACT_FAILED) => return Err(AotError::ContractAssertionFailed),
            Some(AOT_EXIT_GATHER_OOB) => return Err(AotError::GatherOutOfBounds),
            Some(AOT_EXIT_INT_DIV_ERR) => return Err(AotError::IntegerDivisionError),
            Some(AOT_EXIT_CAST_ERR) => return Err(AotError::CastOutOfRange),
            Some(AOT_EXIT_REGION_OOM) => return Err(AotError::RegionAllocationFailed),
            Some(AOT_EXIT_STACK_OVERFLOW) => return Err(AotError::StackOverflow),
            other => {
                return Err(AotError::ExecutionFailed(format!(
                    "standalone binary exited with status {other:?}"
                )))
            }
        }
    }

    Ok(output.stdout)
}

/// Executes a compiled standalone scalar Rul `ELF64` binary (requiring no arena data file)
/// and reads back its 8-byte scalar output bit-pattern as `f64`.
///
/// # Errors
/// Returns [`AotError`] if the process exits with a fail-closed status (`70..=75`) or I/O fails.
pub fn run_compiled_scalar_executable(exe_path: impl AsRef<Path>) -> Result<f64, AotError> {
    let stdout = run_compiled_structured_executable(exe_path, &[])?;
    let bytes: [u8; 8] = stdout.as_slice().try_into().map_err(|_e| {
        AotError::ExecutionFailed(format!(
            "expected 8-byte scalar output on stdout, got {} bytes",
            stdout.len()
        ))
    })?;
    Ok(f64::from_le_bytes(bytes))
}

const SC_SCRATCH_IO_OFF: i32 = 2064;
const SC_R15_SAVE_OFF: i32 = 2080;

const REG_RAX: u8 = 0;
const REG_RCX: u8 = 1;
const REG_RDX: u8 = 2;
const REG_RBX: u8 = 3;
const REG_RBP: u8 = 5;
const REG_RSI: u8 = 6;
const REG_RDI: u8 = 7;
const REG_R8: u8 = 8;
const REG_R9: u8 = 9;
const REG_R10: u8 = 10;
const REG_R11: u8 = 11;
const REG_R12: u8 = 12;
const REG_R13: u8 = 13;
const REG_R14: u8 = 14;

const SC_ARG_GPRS: [u8; 4] = [REG_RDI, REG_RSI, REG_RDX, REG_RCX];
const SC_CALLEE_SAVED_GPRS: [u8; 6] = [REG_RBX, REG_R12, REG_R13, REG_R8, REG_R9, REG_R10];
const SC_ALL_GPRS_ORDER: [u8; 10] = [
    REG_RDI, REG_RSI, REG_RDX, REG_RCX, REG_RBX, REG_R12, REG_R13, REG_R8, REG_R9, REG_R10,
];
const SC_CALLEE_SAVED_MASK: u16 =
    (1 << 3) | (1 << 12) | (1 << 13) | (1 << 8) | (1 << 9) | (1 << 10);
const SC_CALLER_SAVED_MASK: u16 = (1 << 7) | (1 << 6) | (1 << 2) | (1 << 1);
const SC_ALLOC_GPR_MASK: u16 = SC_CALLER_SAVED_MASK | SC_CALLEE_SAVED_MASK;
const SC_ALLOC_XMM_MASK: u16 = 0xFFFC; // xmm2..=xmm15
const SC_R15_SAVE_GPRS: [u8; 7] = [REG_RBX, REG_R12, REG_R13, REG_R8, REG_R9, REG_R10, REG_R14];

const HOME_NONE: u8 = 0;
const HOME_GPR: u8 = 1;
const HOME_ALIAS_GPR: u8 = 2;
const HOME_XMM: u8 = 3;
const HOME_ALIAS_XMM: u8 = 4;
const HOME_SPILL: u8 = 5;
const HOME_CONST_I64: u8 = 6;
const HOME_CONST_U8: u8 = 7;
const HOME_CONST_F64: u8 = 8;

const FLAG_SCOPE_HAS_RESULT: u8 = 0x10;
const FLAG_BRANCH_FUSE_DIRECT: u8 = 0x20;
const FLAG_BRANCH_FUSE_INV: u8 = 0x40;
const FLAG_FUSED_BY_NEXT1: u8 = 0x80;
const FLAG_FUSED_BY_NEXT2: u8 = 0x08;

#[inline]
const fn encode_elem(elem: ElemType) -> u8 {
    match elem {
        ElemType::F64 => 0,
        ElemType::I64 => 1,
        ElemType::BoolMask => 2,
        ElemType::U8 => 3,
    }
}

#[inline]
const fn decode_elem(elem_flags: u8) -> ElemType {
    match elem_flags & 0x07 {
        0 => ElemType::F64,
        1 => ElemType::I64,
        2 => ElemType::BoolMask,
        _ => ElemType::U8,
    }
}

/// Compact 16-byte per-node analysis and allocation record (4 records per 64-byte cache line).
#[derive(Clone, Copy)]
struct ScNodeInfo {
    last_use: u32,
    outer_loop_header: u32,
    /// `scope_end` for `Block`/`Loop`/`If`, or `next_local_set` (and intrusive list link during scan) for `LocalGet`.
    aux: u32,
    /// Holds `def_loop_depth` during `sc_analyze_unit`, and `home_reg` / `spill_slot` during `sc_emit_unit`.
    home_data: u16,
    /// Low 4 bits = `home_tag` (`0..=8`), bits 4..6 = saturated use count (`0`, `1`, `2+`).
    home_tag_uc: u8,
    /// Low 3 bits = `ElemType`, bits 3..7 = scope/fusion flags.
    elem_flags: u8,
}

impl ScNodeInfo {
    const DEFAULT: Self = Self {
        last_use: 0,
        outer_loop_header: u32::MAX,
        aux: u32::MAX,
        home_data: 0,
        home_tag_uc: HOME_NONE,
        elem_flags: 1, // ElemType::I64
    };

    #[inline]
    const fn elem(self) -> ElemType {
        decode_elem(self.elem_flags)
    }

    #[inline]
    fn set_elem(&mut self, elem: ElemType) {
        self.elem_flags = (self.elem_flags & !0x07) | encode_elem(elem);
    }

    #[inline]
    const fn home_tag(self) -> u8 {
        self.home_tag_uc & 0x0F
    }

    #[inline]
    const fn use_count(self) -> u8 {
        self.home_tag_uc >> 4
    }

    #[inline]
    fn inc_use(&mut self) {
        let uc = self.home_tag_uc >> 4;
        if uc < 2 {
            self.home_tag_uc = (self.home_tag_uc & 0x0F) | ((uc + 1) << 4);
        }
    }

    #[inline]
    fn set_home_tag(&mut self, tag: u8) {
        self.home_tag_uc = (self.home_tag_uc & 0xF0) | (tag & 0x0F);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScHome {
    None,
    Gpr(u8),
    AliasGpr(u8),
    Xmm(u8),
    AliasXmm(u8),
    Spill(u32),
    ConstI64(i64),
    ConstU8(u8),
    ConstF64(u64),
}

impl ScHome {
    #[inline]
    const fn gpr_reg(self) -> Option<u8> {
        match self {
            Self::Gpr(r) | Self::AliasGpr(r) => Some(r),
            _ => None,
        }
    }

    #[inline]
    const fn xmm_reg(self) -> Option<u8> {
        match self {
            Self::Xmm(x) | Self::AliasXmm(x) => Some(x),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScLocalHome {
    Unused,
    Gpr(u8),
    Xmm(u8),
    Slot(i32),
}

#[derive(Debug)]
enum AotCtrlKind {
    Block {
        end_node_idx: usize,
    },
    Loop {
        body_start_offset: usize,
        loop_node_idx: usize,
        bounded: bool,
    },
    If {
        end_node_idx: usize,
        false_jump_patch: usize,
        has_else: bool,
    },
}

#[derive(Debug)]
struct AotCtrlFrame {
    kind: AotCtrlKind,
    end_patches: Vec<usize>,
}

struct ScUnitScratch {
    nodes: Vec<ScNodeInfo>,
    call_prefix: Vec<u32>,
    has_any_barrier: bool,
    has_any_loop: bool,
    local_get_head: [u32; 64],
    loop_stack: Vec<u32>,
    ctrl_scan: Vec<(Opcode, u32)>,
    spill_expire_head: Vec<u32>,
    spill_entries: Vec<(u32, u32)>,
    free_spill_slots: Vec<u32>,
    ctrl_stack: Vec<AotCtrlFrame>,
    epilogue_patches: Vec<usize>,
    fn_body: Vec<u8>,
}

impl ScUnitScratch {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            call_prefix: Vec::new(),
            has_any_barrier: false,
            has_any_loop: false,
            local_get_head: [u32::MAX; 64],
            loop_stack: Vec::new(),
            ctrl_scan: Vec::new(),
            spill_expire_head: Vec::new(),
            spill_entries: Vec::new(),
            free_spill_slots: Vec::new(),
            ctrl_stack: Vec::new(),
            epilogue_patches: Vec::new(),
            fn_body: Vec::new(),
        }
    }

    fn reset_for_unit(&mut self, len: usize) {
        self.nodes.clear();
        self.nodes.resize(len, ScNodeInfo::DEFAULT);
        self.call_prefix.clear();
        self.has_any_barrier = false;
        self.has_any_loop = false;
        self.local_get_head = [u32::MAX; 64];
        self.loop_stack.clear();
        self.ctrl_scan.clear();
        self.spill_expire_head.clear();
        self.spill_entries.clear();
        self.free_spill_slots.clear();
        self.ctrl_stack.clear();
        self.epilogue_patches.clear();
    }

    #[inline]
    fn get_home(&self, v: usize, unit_nodes: &[crate::bytecode::NodeFrame]) -> ScHome {
        let Some(info) = self.nodes.get(v) else {
            return ScHome::None;
        };
        match info.home_tag() {
            HOME_GPR => ScHome::Gpr(info.home_data as u8),
            HOME_ALIAS_GPR => ScHome::AliasGpr(info.home_data as u8),
            HOME_XMM => ScHome::Xmm(info.home_data as u8),
            HOME_ALIAS_XMM => ScHome::AliasXmm(info.home_data as u8),
            HOME_SPILL => ScHome::Spill(u32::from(info.home_data)),
            HOME_CONST_I64 => unit_nodes.get(v).map_or(ScHome::ConstI64(0), |f| {
                ScHome::ConstI64(f.decode_const_i64())
            }),
            HOME_CONST_U8 => ScHome::ConstU8(info.home_data as u8),
            HOME_CONST_F64 => unit_nodes.get(v).map_or(ScHome::ConstF64(0), |f| {
                ScHome::ConstF64(f.decode_const_f64().to_bits())
            }),
            _ => ScHome::None,
        }
    }

    #[inline]
    fn set_home(&mut self, v: usize, home: ScHome) -> Result<(), AotError> {
        let Some(info) = self.nodes.get_mut(v) else {
            return Ok(());
        };
        match home {
            ScHome::None => {
                info.set_home_tag(HOME_NONE);
            }
            ScHome::Gpr(r) => {
                info.set_home_tag(HOME_GPR);
                info.home_data = u16::from(r);
            }
            ScHome::AliasGpr(r) => {
                info.set_home_tag(HOME_ALIAS_GPR);
                info.home_data = u16::from(r);
            }
            ScHome::Xmm(x) => {
                info.set_home_tag(HOME_XMM);
                info.home_data = u16::from(x);
            }
            ScHome::AliasXmm(x) => {
                info.set_home_tag(HOME_ALIAS_XMM);
                info.home_data = u16::from(x);
            }
            ScHome::Spill(s) => {
                let spill_idx = u16::try_from(s)
                    .map_err(|_e| AotError::UnsupportedGraph("spill slot index overflow"))?;
                info.set_home_tag(HOME_SPILL);
                info.home_data = spill_idx;
            }
            ScHome::ConstI64(_) => {
                info.set_home_tag(HOME_CONST_I64);
            }
            ScHome::ConstU8(b) => {
                info.set_home_tag(HOME_CONST_U8);
                info.home_data = u16::from(b);
            }
            ScHome::ConstF64(_) => {
                info.set_home_tag(HOME_CONST_F64);
            }
        }
        Ok(())
    }

    #[inline]
    fn node_elem(&self, v: usize) -> ElemType {
        self.nodes.get(v).map_or(ElemType::I64, |n| n.elem())
    }

    #[inline]
    fn eff_last_use(&self, v: usize) -> usize {
        let Some(info) = self.nodes.get(v) else {
            return 0;
        };
        let lu = info.last_use as usize;
        if !self.has_any_loop || info.outer_loop_header == u32::MAX {
            lu
        } else {
            let lend = self
                .nodes
                .get(info.outer_loop_header as usize)
                .map_or(0, |h| h.aux as usize);
            lu.max(lend)
        }
    }

    #[inline]
    fn spans_barrier(&self, start_idx: usize, end_idx: usize) -> bool {
        if !self.has_any_barrier || end_idx <= start_idx {
            return false;
        }
        let c_start = self
            .call_prefix
            .get(start_idx.saturating_add(1))
            .copied()
            .unwrap_or(0);
        let c_end = self.call_prefix.get(end_idx).copied().unwrap_or(0);
        c_end > c_start
    }

    #[inline]
    fn is_branch_fused(&self, branch_idx: usize) -> Option<(usize, bool)> {
        let flags = self.nodes.get(branch_idx)?.elem_flags;
        if (flags & FLAG_BRANCH_FUSE_DIRECT) != 0 {
            let c = branch_idx.checked_sub(1)?;
            if self.nodes.get(c)?.use_count() == 1 {
                return Some((c, false));
            }
        } else if (flags & FLAG_BRANCH_FUSE_INV) != 0 {
            let c = branch_idx.checked_sub(2)?;
            if self.nodes.get(c)?.use_count() == 1
                && self.nodes.get(c.saturating_add(1))?.use_count() == 1
            {
                return Some((c, true));
            }
        }
        None
    }

    #[inline]
    fn is_node_fused_out(&self, idx: usize) -> bool {
        let Some(info) = self.nodes.get(idx) else {
            return false;
        };
        if (info.elem_flags & FLAG_FUSED_BY_NEXT1) != 0 {
            self.is_branch_fused(idx.saturating_add(1)).is_some()
        } else if (info.elem_flags & FLAG_FUSED_BY_NEXT2) != 0 {
            self.is_branch_fused(idx.saturating_add(2)).is_some()
        } else {
            false
        }
    }
}

fn compile_structured_to_elf64_bytes(
    program: &RulProgram,
    summary: &VerifiedSummary,
    verify_ns: u64,
    t_cg: Instant,
) -> Result<(Vec<u8>, AotCompileReport), AotError> {
    if !program.arenas.is_empty() {
        return Err(AotError::UnsupportedGraph(
            "structured control-flow AOT requires scalar-only programs without columnar arenas",
        ));
    }
    if !program.has_functions() && program.nodes.len() > 65_536 {
        return Err(AotError::UnsupportedGraph(
            "structured control-flow AOT exceeds 65,536-node stack frame limit",
        ));
    }
    if program.nodes.get(program.output_node as usize).is_none() {
        return Err(AotError::UnsupportedGraph("invalid output node index"));
    }
    if summary.output_shape != ValueShape::Scalar {
        return Err(AotError::UnsupportedGraph(
            "structured control-flow AOT requires scalar output_node",
        ));
    }

    let mut elf = emit_x86_64_scalar_control_flow(program)?;
    let total_elf_size = elf.len();
    let machine_code_bytes = total_elf_size.saturating_sub(ELF_CODE_OFFSET);
    let mut hdr = Vec::with_capacity(ELF_CODE_OFFSET);
    write_elf64_headers(&mut hdr, total_elf_size as u64);
    if let Some(dst) = elf.get_mut(..ELF_CODE_OFFSET) {
        dst.copy_from_slice(&hdr);
    }

    let codegen_ns = u64::try_from(t_cg.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let report = AotCompileReport {
        elf_bytes: total_elf_size,
        machine_code_bytes,
        avx2_vectorized: false,
        verify_ns,
        codegen_ns,
        total_compile_ns: verify_ns.saturating_add(codegen_ns),
    };
    Ok((elf, report))
}

#[inline]
const fn sc_rex(w: bool, r: u8, x: u8, b: u8) -> u8 {
    0x40 | ((w as u8) << 3) | (((r >> 3) & 1) << 2) | (((x >> 3) & 1) << 1) | ((b >> 3) & 1)
}

fn emit_push_gpr(code: &mut Vec<u8>, reg: u8) {
    if reg >= 8 {
        code.push(0x41);
    }
    code.push(0x50 + (reg & 7));
}

fn emit_pop_gpr(code: &mut Vec<u8>, reg: u8) {
    if reg >= 8 {
        code.push(0x41);
    }
    code.push(0x58 + (reg & 7));
}

fn emit_mov_gpr_gpr(code: &mut Vec<u8>, dst: u8, src: u8) {
    if dst == src {
        return;
    }
    code.extend_from_slice(&[
        sc_rex(true, src, 0, dst),
        0x89,
        0xC0 | ((src & 7) << 3) | (dst & 7),
    ]);
}

/// Emits `mov dst32, src32`, zero-extending the lower 32 bits into `dst64` even when `dst == src`.
fn emit_mov_gpr32_gpr32(code: &mut Vec<u8>, dst: u8, src: u8) {
    if dst >= 8 || src >= 8 {
        code.push(sc_rex(false, src, 0, dst));
    }
    code.extend_from_slice(&[0x89, 0xC0 | ((src & 7) << 3) | (dst & 7)]);
}

fn emit_mov_gpr_imm64(code: &mut Vec<u8>, dst: u8, val: u64) {
    if val == 0 {
        if dst >= 8 {
            code.push(sc_rex(false, dst, 0, dst));
        }
        code.extend_from_slice(&[0x31, 0xC0 | ((dst & 7) << 3) | (dst & 7)]);
    } else if let Ok(v32) = u32::try_from(val) {
        if dst >= 8 {
            code.push(0x41);
        }
        code.push(0xB8 + (dst & 7));
        code.extend_from_slice(&v32.to_le_bytes());
    } else {
        #[expect(
            clippy::cast_possible_wrap,
            reason = "checking if 64-bit bit pattern fits in sign-extended imm32"
        )]
        let signed = val as i64;
        if let Ok(s32) = i32::try_from(signed) {
            code.extend_from_slice(&[sc_rex(true, 0, 0, dst), 0xC7, 0xC0 | (dst & 7)]);
            code.extend_from_slice(&s32.to_le_bytes());
        } else {
            emit_movabs_reg(code, dst, val);
        }
    }
}

fn emit_movabs_reg(code: &mut Vec<u8>, reg: u8, val: u64) {
    let rex = 0x48 | if reg >= 8 { 0x01 } else { 0 };
    code.push(rex);
    code.push(0xB8 + (reg & 7));
    code.extend_from_slice(&val.to_le_bytes());
}

fn emit_rbp_disp_modrm(code: &mut Vec<u8>, reg_field: u8, disp: i32) {
    if let Ok(d8) = i8::try_from(disp) {
        code.push(0x45 | ((reg_field & 7) << 3));
        code.push(d8.to_ne_bytes()[0]);
    } else {
        code.push(0x85 | ((reg_field & 7) << 3));
        code.extend_from_slice(&disp.to_le_bytes());
    }
}

fn emit_load_gpr_rbp(code: &mut Vec<u8>, dst: u8, disp: i32) {
    code.extend_from_slice(&[sc_rex(true, dst, 0, REG_RBP), 0x8B]);
    emit_rbp_disp_modrm(code, dst, disp);
}

fn emit_load_gpr32_rbp(code: &mut Vec<u8>, dst: u8, disp: i32) {
    if dst >= 8 {
        code.push(sc_rex(false, dst, 0, REG_RBP));
    }
    code.push(0x8B);
    emit_rbp_disp_modrm(code, dst, disp);
}

fn emit_store_rbp_gpr(code: &mut Vec<u8>, disp: i32, src: u8) {
    code.extend_from_slice(&[sc_rex(true, src, 0, REG_RBP), 0x89]);
    emit_rbp_disp_modrm(code, src, disp);
}

fn emit_load_xmm_rbp(code: &mut Vec<u8>, xmm: u8, disp: i32) {
    code.push(0xF2);
    if xmm >= 8 {
        code.push(sc_rex(false, xmm, 0, REG_RBP));
    }
    code.extend_from_slice(&[0x0F, 0x10]);
    emit_rbp_disp_modrm(code, xmm, disp);
}

fn emit_store_rbp_xmm(code: &mut Vec<u8>, disp: i32, xmm: u8) {
    code.push(0xF2);
    if xmm >= 8 {
        code.push(sc_rex(false, xmm, 0, REG_RBP));
    }
    code.extend_from_slice(&[0x0F, 0x11]);
    emit_rbp_disp_modrm(code, xmm, disp);
}

fn emit_mov_xmm_xmm(code: &mut Vec<u8>, dst: u8, src: u8) {
    if dst == src {
        return;
    }
    if dst >= 8 || src >= 8 {
        code.push(sc_rex(false, dst, 0, src));
    }
    // movaps dst, src (whole 128-bit reg rename move, no partial-reg dependency)
    code.extend_from_slice(&[0x0F, 0x28, 0xC0 | ((dst & 7) << 3) | (src & 7)]);
}

fn emit_movq_xmm_gpr(code: &mut Vec<u8>, xmm: u8, gpr: u8) {
    code.extend_from_slice(&[
        0x66,
        sc_rex(true, xmm, 0, gpr),
        0x0F,
        0x6E,
        0xC0 | ((xmm & 7) << 3) | (gpr & 7),
    ]);
}

fn emit_movq_gpr_xmm(code: &mut Vec<u8>, gpr: u8, xmm: u8) {
    code.extend_from_slice(&[
        0x66,
        sc_rex(true, xmm, 0, gpr),
        0x0F,
        0x7E,
        0xC0 | ((xmm & 7) << 3) | (gpr & 7),
    ]);
}

fn emit_movzx_gpr32_gpr8(code: &mut Vec<u8>, dst: u8, src: u8) {
    // Always emit REX so `src` 4..7 addresses `spl/bpl/sil/dil` instead of `ah/ch/dh/bh`.
    code.extend_from_slice(&[
        sc_rex(false, dst, 0, src),
        0x0F,
        0xB6,
        0xC0 | ((dst & 7) << 3) | (src & 7),
    ]);
}

fn emit_alu64_rr(code: &mut Vec<u8>, op: u8, dst: u8, src: u8) {
    // `op` in {0x03 add, 0x2B sub, 0x23 and, 0x0B or, 0x33 xor, 0x3B cmp} (reg=dst, rm=src)
    code.extend_from_slice(&[
        sc_rex(true, dst, 0, src),
        op,
        0xC0 | ((dst & 7) << 3) | (src & 7),
    ]);
}

fn emit_alu64_rm_rbp(code: &mut Vec<u8>, op: u8, dst: u8, disp: i32) {
    code.extend_from_slice(&[sc_rex(true, dst, 0, REG_RBP), op]);
    emit_rbp_disp_modrm(code, dst, disp);
}

fn emit_alu64_ri(code: &mut Vec<u8>, ext: u8, dst: u8, imm: i64) {
    // `ext`: 0=add, 1=or, 4=and, 5=sub, 6=xor, 7=cmp
    if let Ok(i8v) = i8::try_from(imm) {
        code.extend_from_slice(&[
            sc_rex(true, 0, 0, dst),
            0x83,
            0xC0 | (ext << 3) | (dst & 7),
            i8v.to_ne_bytes()[0],
        ]);
    } else if let Ok(i32v) = i32::try_from(imm) {
        code.extend_from_slice(&[sc_rex(true, 0, 0, dst), 0x81, 0xC0 | (ext << 3) | (dst & 7)]);
        code.extend_from_slice(&i32v.to_le_bytes());
    } else {
        #[expect(
            clippy::cast_sign_loss,
            reason = "two's-complement 64-bit immediate materialization"
        )]
        emit_mov_gpr_imm64(code, REG_R11, imm as u64);
        let op = match ext {
            0 => 0x03,
            1 => 0x0B,
            4 => 0x23,
            5 => 0x2B,
            6 => 0x33,
            _ => 0x3B,
        };
        emit_alu64_rr(code, op, dst, REG_R11);
    }
}

fn emit_alu32_rr(code: &mut Vec<u8>, op: u8, dst: u8, src: u8) {
    if dst >= 8 || src >= 8 {
        code.push(sc_rex(false, dst, 0, src));
    }
    code.extend_from_slice(&[op, 0xC0 | ((dst & 7) << 3) | (src & 7)]);
}

fn emit_alu32_rm_rbp(code: &mut Vec<u8>, op: u8, dst: u8, disp: i32) {
    if dst >= 8 {
        code.push(sc_rex(false, dst, 0, REG_RBP));
    }
    code.push(op);
    emit_rbp_disp_modrm(code, dst, disp);
}

fn emit_alu32_ri(code: &mut Vec<u8>, ext: u8, dst: u8, imm: i32) {
    if dst >= 8 {
        code.push(sc_rex(false, 0, 0, dst));
    }
    if let Ok(i8v) = i8::try_from(imm) {
        code.extend_from_slice(&[0x83, 0xC0 | (ext << 3) | (dst & 7), i8v.to_ne_bytes()[0]]);
    } else {
        code.extend_from_slice(&[0x81, 0xC0 | (ext << 3) | (dst & 7)]);
        code.extend_from_slice(&imm.to_le_bytes());
    }
}

fn emit_imul64_rr(code: &mut Vec<u8>, dst: u8, src: u8) {
    code.extend_from_slice(&[
        sc_rex(true, dst, 0, src),
        0x0F,
        0xAF,
        0xC0 | ((dst & 7) << 3) | (src & 7),
    ]);
}

fn emit_imul64_rm_rbp(code: &mut Vec<u8>, dst: u8, disp: i32) {
    code.extend_from_slice(&[sc_rex(true, dst, 0, REG_RBP), 0x0F, 0xAF]);
    emit_rbp_disp_modrm(code, dst, disp);
}

fn emit_imul64_rri(code: &mut Vec<u8>, dst: u8, src: u8, imm: i32) {
    if let Ok(i8v) = i8::try_from(imm) {
        code.extend_from_slice(&[
            sc_rex(true, dst, 0, src),
            0x6B,
            0xC0 | ((dst & 7) << 3) | (src & 7),
            i8v.to_ne_bytes()[0],
        ]);
    } else {
        code.extend_from_slice(&[
            sc_rex(true, dst, 0, src),
            0x69,
            0xC0 | ((dst & 7) << 3) | (src & 7),
        ]);
        code.extend_from_slice(&imm.to_le_bytes());
    }
}

fn emit_lea_reg_disp(code: &mut Vec<u8>, dst: u8, base: u8, disp: i32) {
    if disp == 0 {
        emit_mov_gpr_gpr(code, dst, base);
        return;
    }
    code.extend_from_slice(&[sc_rex(true, dst, 0, base), 0x8D]);
    if let Ok(d8) = i8::try_from(disp) {
        code.push(0x40 | ((dst & 7) << 3) | (base & 7));
        if (base & 7) == 4 {
            code.push(0x24);
        }
        code.push(d8.to_ne_bytes()[0]);
    } else {
        code.push(0x80 | ((dst & 7) << 3) | (base & 7));
        if (base & 7) == 4 {
            code.push(0x24);
        }
        code.extend_from_slice(&disp.to_le_bytes());
    }
}

fn emit_lea_reg_plus_reg(code: &mut Vec<u8>, dst: u8, a: u8, b: u8) {
    // Neither `a` nor `b` is `rsp (4)`; if `a` is `r13` (`a & 7 == 5`), swap so base has disp0.
    let (base, index) = if (a & 7) == 5 && (b & 7) != 5 {
        (b, a)
    } else {
        (a, b)
    };
    code.extend_from_slice(&[sc_rex(true, dst, index, base), 0x8D]);
    if (base & 7) == 5 {
        code.push(0x44 | ((dst & 7) << 3));
        code.push(((index & 7) << 3) | (base & 7));
        code.push(0x00);
    } else {
        code.push(0x04 | ((dst & 7) << 3));
        code.push(((index & 7) << 3) | (base & 7));
    }
}

fn emit_shift64_ri(code: &mut Vec<u8>, ext: u8, dst: u8, count: u8) {
    let c = count & 63;
    if c == 0 {
        return;
    }
    code.extend_from_slice(&[
        sc_rex(true, 0, 0, dst),
        0xC1,
        0xC0 | (ext << 3) | (dst & 7),
        c,
    ]);
}

fn emit_shift32_ri(code: &mut Vec<u8>, ext: u8, dst: u8, count: u8) {
    let c = count & 31;
    if c == 0 {
        return;
    }
    if dst >= 8 {
        code.push(sc_rex(false, 0, 0, dst));
    }
    code.extend_from_slice(&[0xC1, 0xC0 | (ext << 3) | (dst & 7), c]);
}

fn emit_sse2_sd_rr(code: &mut Vec<u8>, op: u8, dst: u8, src: u8) {
    code.push(0xF2);
    if dst >= 8 || src >= 8 {
        code.push(sc_rex(false, dst, 0, src));
    }
    code.extend_from_slice(&[0x0F, op, 0xC0 | ((dst & 7) << 3) | (src & 7)]);
}

fn emit_sse2_sd_rm_rbp(code: &mut Vec<u8>, op: u8, dst: u8, disp: i32) {
    code.push(0xF2);
    if dst >= 8 {
        code.push(sc_rex(false, dst, 0, REG_RBP));
    }
    code.extend_from_slice(&[0x0F, op]);
    emit_rbp_disp_modrm(code, dst, disp);
}

fn emit_roundsd_rr(code: &mut Vec<u8>, dst: u8, src: u8, mode: u8) {
    code.push(0x66);
    if dst >= 8 || src >= 8 {
        code.push(sc_rex(false, dst, 0, src));
    }
    code.extend_from_slice(&[0x0F, 0x3A, 0x0B, 0xC0 | ((dst & 7) << 3) | (src & 7), mode]);
}

fn emit_vex_fma213_xmm_rr(code: &mut Vec<u8>, dst: u8, src1: u8, src2: u8) {
    let r_inv = if (dst & 8) == 0 { 0x80 } else { 0x00 };
    let b_inv = if (src2 & 8) == 0 { 0x20 } else { 0x00 };
    let byte1 = r_inv | 0x40 | b_inv | 0x02;
    let vvvv_inv = (!src1) & 0x0F;
    let byte2 = 0x80 | (vvvv_inv << 3) | 0x01;
    let modrm = 0xC0 | ((dst & 7) << 3) | (src2 & 7);
    code.extend_from_slice(&[0xC4, byte1, byte2, 0xA9, modrm]);
}

fn emit_load_reg_r15(code: &mut Vec<u8>, reg: u8, disp: i32) {
    let rex = 0x49 | if reg >= 8 { 0x04 } else { 0 };
    if let Ok(d8) = i8::try_from(disp) {
        let modrm = 0x47 | ((reg & 7) << 3);
        code.extend_from_slice(&[rex, 0x8B, modrm, d8.to_ne_bytes()[0]]);
    } else {
        let modrm = 0x87 | ((reg & 7) << 3);
        code.extend_from_slice(&[rex, 0x8B, modrm]);
        code.extend_from_slice(&disp.to_le_bytes());
    }
}

fn emit_store_r15_reg(code: &mut Vec<u8>, reg: u8, disp: i32) {
    let rex = 0x49 | if reg >= 8 { 0x04 } else { 0 };
    if let Ok(d8) = i8::try_from(disp) {
        let modrm = 0x47 | ((reg & 7) << 3);
        code.extend_from_slice(&[rex, 0x89, modrm, d8.to_ne_bytes()[0]]);
    } else {
        let modrm = 0x87 | ((reg & 7) << 3);
        code.extend_from_slice(&[rex, 0x89, modrm]);
        code.extend_from_slice(&disp.to_le_bytes());
    }
}

fn emit_cmp_reg_r15(code: &mut Vec<u8>, reg: u8, disp: i32) {
    let rex = 0x49 | if reg >= 8 { 0x04 } else { 0 };
    if let Ok(d8) = i8::try_from(disp) {
        let modrm = 0x47 | ((reg & 7) << 3);
        code.extend_from_slice(&[rex, 0x3B, modrm, d8.to_ne_bytes()[0]]);
    } else {
        let modrm = 0x87 | ((reg & 7) << 3);
        code.extend_from_slice(&[rex, 0x3B, modrm]);
        code.extend_from_slice(&disp.to_le_bytes());
    }
}

fn emit_sc_save_callee_to_r15(code: &mut Vec<u8>) {
    for (k, &reg) in SC_R15_SAVE_GPRS.iter().enumerate() {
        let off = SC_R15_SAVE_OFF.saturating_add(i32::try_from(k).unwrap_or(0).saturating_mul(8));
        emit_store_r15_reg(code, reg, off);
    }
}

fn emit_sc_restore_callee_from_r15(code: &mut Vec<u8>) {
    for (k, &reg) in SC_R15_SAVE_GPRS.iter().enumerate() {
        let off = SC_R15_SAVE_OFF.saturating_add(i32::try_from(k).unwrap_or(0).saturating_mul(8));
        emit_load_reg_r15(code, reg, off);
    }
}

/// Resolves up to 4 simultaneous GPR-to-GPR moves `(dst, src)` with distinct `dst`s,
/// using `rax` as cycle-breaking temporary. Neither `dst` nor `src` may be `rax`.
fn emit_parallel_gpr_moves(code: &mut Vec<u8>, pairs: &[(u8, u8)]) {
    let mut pending: [(u8, u8); 4] = [(0, 0); 4];
    let mut len = 0usize;
    for &(d, s) in pairs {
        if d != s {
            if let Some(slot) = pending.get_mut(len) {
                *slot = (d, s);
                len += 1;
            }
        }
    }
    while len > 0 {
        let mut ready = None;
        for i in 0..len {
            let Some(&(d, _)) = pending.get(i) else {
                break;
            };
            let used_as_src = (0..len).any(|j| pending.get(j).is_some_and(|&(_, s)| s == d));
            if !used_as_src {
                ready = Some(i);
                break;
            }
        }
        if let Some(i) = ready {
            if let Some(&(d, s)) = pending.get(i) {
                emit_mov_gpr_gpr(code, d, s);
            }
            len -= 1;
            if i < len {
                if let Some(&last) = pending.get(len) {
                    if let Some(slot) = pending.get_mut(i) {
                        *slot = last;
                    }
                }
            }
        } else if let Some(&(d0, s0)) = pending.first() {
            // Break cycle using `rax`.
            emit_mov_gpr_gpr(code, REG_RAX, d0);
            emit_mov_gpr_gpr(code, d0, s0);
            len -= 1;
            if len > 0 {
                if let Some(&last) = pending.get(len) {
                    if let Some(slot) = pending.first_mut() {
                        *slot = last;
                    }
                }
            }
            for j in 0..len {
                if let Some(slot) = pending.get_mut(j) {
                    if slot.1 == d0 {
                        slot.1 = REG_RAX;
                    }
                }
            }
        } else {
            break;
        }
    }
}

#[inline]
fn sc_spill_disp(spill_base_slot: u32, spill_idx: u32) -> i32 {
    let total = spill_base_slot.saturating_add(spill_idx).saturating_add(1);
    (-8i32).saturating_mul(i32::try_from(total).unwrap_or(1))
}

fn emit_load_home_into_gpr(code: &mut Vec<u8>, dst: u8, home: ScHome, spill_base_slot: u32) {
    match home {
        ScHome::None => {}
        ScHome::Gpr(r) | ScHome::AliasGpr(r) => emit_mov_gpr_gpr(code, dst, r),
        ScHome::Xmm(x) | ScHome::AliasXmm(x) => emit_movq_gpr_xmm(code, dst, x),
        ScHome::Spill(s) => emit_load_gpr_rbp(code, dst, sc_spill_disp(spill_base_slot, s)),
        ScHome::ConstI64(v) => {
            #[expect(
                clippy::cast_sign_loss,
                reason = "two's-complement 64-bit constant bit pattern"
            )]
            emit_mov_gpr_imm64(code, dst, v as u64);
        }
        ScHome::ConstU8(b) => emit_mov_gpr_imm64(code, dst, u64::from(b)),
        ScHome::ConstF64(bits) => emit_mov_gpr_imm64(code, dst, bits),
    }
}

fn emit_store_gpr_into_home(code: &mut Vec<u8>, home: ScHome, src: u8, spill_base_slot: u32) {
    match home {
        ScHome::None | ScHome::ConstI64(_) | ScHome::ConstU8(_) | ScHome::ConstF64(_) => {}
        ScHome::Gpr(r) | ScHome::AliasGpr(r) => emit_mov_gpr_gpr(code, r, src),
        ScHome::Xmm(x) | ScHome::AliasXmm(x) => emit_movq_xmm_gpr(code, x, src),
        ScHome::Spill(s) => emit_store_rbp_gpr(code, sc_spill_disp(spill_base_slot, s), src),
    }
}

fn emit_load_home_into_xmm(code: &mut Vec<u8>, dst_xmm: u8, home: ScHome, spill_base_slot: u32) {
    match home {
        ScHome::None => {}
        ScHome::Xmm(x) | ScHome::AliasXmm(x) => emit_mov_xmm_xmm(code, dst_xmm, x),
        ScHome::Gpr(r) | ScHome::AliasGpr(r) => emit_movq_xmm_gpr(code, dst_xmm, r),
        ScHome::Spill(s) => emit_load_xmm_rbp(code, dst_xmm, sc_spill_disp(spill_base_slot, s)),
        ScHome::ConstF64(bits) => {
            if bits == 0 {
                // xorpd dst_xmm, dst_xmm
                code.push(0x66);
                if dst_xmm >= 8 {
                    code.push(sc_rex(false, dst_xmm, 0, dst_xmm));
                }
                code.extend_from_slice(&[0x0F, 0x57, 0xC0 | ((dst_xmm & 7) << 3) | (dst_xmm & 7)]);
            } else {
                emit_mov_gpr_imm64(code, REG_RAX, bits);
                emit_movq_xmm_gpr(code, dst_xmm, REG_RAX);
            }
        }
        ScHome::ConstI64(v) => {
            #[expect(clippy::cast_sign_loss, reason = "exact 64-bit constant bit pattern")]
            emit_mov_gpr_imm64(code, REG_RAX, v as u64);
            emit_movq_xmm_gpr(code, dst_xmm, REG_RAX);
        }
        ScHome::ConstU8(b) => {
            emit_mov_gpr_imm64(code, REG_RAX, u64::from(b));
            emit_movq_xmm_gpr(code, dst_xmm, REG_RAX);
        }
    }
}

fn emit_store_xmm_into_home(code: &mut Vec<u8>, home: ScHome, src_xmm: u8, spill_base_slot: u32) {
    match home {
        ScHome::None | ScHome::ConstI64(_) | ScHome::ConstU8(_) | ScHome::ConstF64(_) => {}
        ScHome::Xmm(x) | ScHome::AliasXmm(x) => emit_mov_xmm_xmm(code, x, src_xmm),
        ScHome::Gpr(r) | ScHome::AliasGpr(r) => emit_movq_gpr_xmm(code, r, src_xmm),
        ScHome::Spill(s) => emit_store_rbp_xmm(code, sc_spill_disp(spill_base_slot, s), src_xmm),
    }
}

fn emit_move_home_to_home(code: &mut Vec<u8>, dst: ScHome, src: ScHome, spill_base_slot: u32) {
    if dst == src {
        return;
    }
    match dst {
        ScHome::None | ScHome::ConstI64(_) | ScHome::ConstU8(_) | ScHome::ConstF64(_) => {}
        ScHome::Gpr(r) | ScHome::AliasGpr(r) => {
            emit_load_home_into_gpr(code, r, src, spill_base_slot);
        }
        ScHome::Xmm(x) | ScHome::AliasXmm(x) => {
            emit_load_home_into_xmm(code, x, src, spill_base_slot);
        }
        ScHome::Spill(s) => {
            let disp = sc_spill_disp(spill_base_slot, s);
            match src {
                ScHome::None => {}
                ScHome::Gpr(r) | ScHome::AliasGpr(r) => emit_store_rbp_gpr(code, disp, r),
                ScHome::Xmm(x) | ScHome::AliasXmm(x) => emit_store_rbp_xmm(code, disp, x),
                ScHome::Spill(_)
                | ScHome::ConstI64(_)
                | ScHome::ConstU8(_)
                | ScHome::ConstF64(_) => {
                    emit_load_home_into_gpr(code, REG_RAX, src, spill_base_slot);
                    emit_store_rbp_gpr(code, disp, REG_RAX);
                }
            }
        }
    }
}

fn emit_loop_backedge_jump(
    code: &mut Vec<u8>,
    body_start_offset: usize,
    fuel_home: ScHome,
    spill_base_slot: u32,
    bounded: bool,
) -> Result<(), AotError> {
    if bounded {
        match fuel_home {
            ScHome::Gpr(r) | ScHome::AliasGpr(r) => {
                // test r, r; jnz +12; exit(70); dec r
                code.extend_from_slice(&[
                    sc_rex(true, r, 0, r),
                    0x85,
                    0xC0 | ((r & 7) << 3) | (r & 7),
                    0x75,
                    0x0C,
                ]);
                emit_sys_exit(code, AOT_EXIT_CONTRACT_FAILED as u32);
                code.extend_from_slice(&[sc_rex(true, 0, 0, r), 0xFF, 0xC8 | (r & 7)]);
            }
            ScHome::Spill(s) => {
                let disp = sc_spill_disp(spill_base_slot, s);
                emit_load_gpr_rbp(code, REG_RAX, disp);
                code.extend_from_slice(&[0x48, 0x85, 0xC0, 0x75, 0x0C]);
                emit_sys_exit(code, AOT_EXIT_CONTRACT_FAILED as u32);
                code.extend_from_slice(&[0x48, 0xFF, 0xC8]);
                emit_store_rbp_gpr(code, disp, REG_RAX);
            }
            _ => {}
        }
    }
    // jmp rel32 -> body_start_offset
    code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]);
    let patch_pos = code.len() - 4;
    patch_rel32(code, patch_pos, body_start_offset)
}

#[derive(Debug, Clone, Copy)]
struct AotRegionDecl {
    min_len: u64,
    max_len: u64,
    elem: ElemType,
    role: RegionRole,
}

const fn align_up_4096(bytes: u64) -> u64 {
    bytes.div_ceil(4096).saturating_mul(4096)
}

fn sc_region_off(slot: usize, field_off: i32) -> i32 {
    let slot_i32 = i32::try_from(slot).unwrap_or(0);
    slot_i32.saturating_mul(32).saturating_add(field_off)
}

/// Emits `SYS_mmap(NULL, rsi, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)`
/// and fails closed with exit `74` if the kernel returns an error (`rax >= -4095`).
fn emit_sc_sys_mmap_anon(code: &mut Vec<u8>) {
    code.extend_from_slice(&[0x31, 0xFF]); // xor edi, edi
    code.extend_from_slice(&[0xBA, 0x03, 0x00, 0x00, 0x00]); // mov edx, 3
    code.extend_from_slice(&[0x41, 0xBA, 0x22, 0x00, 0x00, 0x00]); // mov r10d, 0x22
    code.extend_from_slice(&[0x49, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF]); // mov r8, -1
    code.extend_from_slice(&[0x45, 0x31, 0xC9]); // xor r9d, r9d
    code.extend_from_slice(&[0xB8, 0x09, 0x00, 0x00, 0x00]); // mov eax, 9
    code.extend_from_slice(&[0x0F, 0x05]); // syscall
    code.extend_from_slice(&[0x48, 0x3D, 0x01, 0xF0, 0xFF, 0xFF, 0x72, 0x0C]);
    emit_sys_exit(code, AOT_EXIT_REGION_OOM as u32);
}

/// Emits `SYS_mremap(rdi, rsi, rdx, MREMAP_MAYMOVE=1, 0)` and fails closed with exit `74` on error.
fn emit_sc_sys_mremap(code: &mut Vec<u8>) {
    code.extend_from_slice(&[0x41, 0xBA, 0x01, 0x00, 0x00, 0x00]); // mov r10d, 1
    code.extend_from_slice(&[0x45, 0x31, 0xC0]); // xor r8d, r8d
    code.extend_from_slice(&[0xB8, 0x19, 0x00, 0x00, 0x00]); // mov eax, 25
    code.extend_from_slice(&[0x0F, 0x05]); // syscall
    code.extend_from_slice(&[0x48, 0x3D, 0x01, 0xF0, 0xFF, 0xFF, 0x72, 0x0C]);
    emit_sys_exit(code, AOT_EXIT_REGION_OOM as u32);
}

/// Emits `SYS_munmap(rdi, rsi)`.
fn emit_sc_sys_munmap(code: &mut Vec<u8>) {
    code.extend_from_slice(&[0xB8, 0x0B, 0x00, 0x00, 0x00]); // mov eax, 11
    code.extend_from_slice(&[0x0F, 0x05]); // syscall
}

/// Allocates the 4096-byte program-global region table in `r15` (unless already mapped),
/// maps all `min_len > 0` regions, and reads stdin to EOF into the declared `RegionRole::Input` region if present.
fn emit_sc_region_startup(
    code: &mut Vec<u8>,
    regions: &[Option<AotRegionDecl>; MAX_REGION_SLOTS],
    r15_already_mapped: bool,
) -> Result<(), AotError> {
    if !r15_already_mapped {
        code.extend_from_slice(&[0xBE, 0x00, 0x10, 0x00, 0x00]);
        emit_sc_sys_mmap_anon(code);
        code.extend_from_slice(&[0x49, 0x89, 0xC7]); // mov r15, rax
    }

    for (slot, opt_decl) in regions.iter().enumerate() {
        let Some(decl) = *opt_decl else {
            continue;
        };
        let off_base = sc_region_off(slot, 0);
        let off_len = sc_region_off(slot, 8);
        let off_cap = sc_region_off(slot, 16);

        if decl.role == RegionRole::Input {
            if decl.max_len == 0 {
                let probe_pos = code.len();
                code.extend_from_slice(&[0x31, 0xFF]);
                code.extend_from_slice(&[0x49, 0x8D, 0xB7]);
                code.extend_from_slice(&SC_SCRATCH_IO_OFF.to_le_bytes());
                code.extend_from_slice(&[0xBA, 0x01, 0x00, 0x00, 0x00]);
                code.extend_from_slice(&[0x31, 0xC0, 0x0F, 0x05]);
                code.extend_from_slice(&[0x48, 0x83, 0xF8, 0xFC]);
                code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
                let eintr_patch = code.len() - 4;
                patch_rel32(code, eintr_patch, probe_pos)?;
                code.extend_from_slice(&[0x48, 0x85, 0xC0, 0x74, 0x0C]);
                emit_sys_exit(code, AOT_EXIT_REGION_OOM as u32);
            } else {
                let max_cap = align_up_4096(decl.max_len);
                code.extend_from_slice(&[0x45, 0x31, 0xE4]); // xor r12d, r12d
                code.extend_from_slice(&[0x45, 0x31, 0xED]); // xor r13d, r13d
                code.extend_from_slice(&[0x45, 0x31, 0xF6]); // xor r14d, r14d

                let read_loop_pos = code.len();
                emit_movabs_reg(code, 0, decl.max_len);
                code.extend_from_slice(&[0x49, 0x39, 0xC4]);
                code.extend_from_slice(&[0x0F, 0x82, 0x00, 0x00, 0x00, 0x00]);
                let not_full_patch = code.len() - 4;

                let probe_eof_pos = code.len();
                code.extend_from_slice(&[0x31, 0xFF]);
                code.extend_from_slice(&[0x49, 0x8D, 0xB7]);
                code.extend_from_slice(&SC_SCRATCH_IO_OFF.to_le_bytes());
                code.extend_from_slice(&[0xBA, 0x01, 0x00, 0x00, 0x00]);
                code.extend_from_slice(&[0x31, 0xC0, 0x0F, 0x05]);
                code.extend_from_slice(&[0x48, 0x83, 0xF8, 0xFC]);
                code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
                let probe_eintr_patch = code.len() - 4;
                patch_rel32(code, probe_eintr_patch, probe_eof_pos)?;
                code.extend_from_slice(&[0x48, 0x85, 0xC0]);
                code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
                let probe_done_patch = code.len() - 4;
                emit_sys_exit(code, AOT_EXIT_REGION_OOM as u32);

                let not_full_pos = code.len();
                patch_rel32(code, not_full_patch, not_full_pos)?;

                code.extend_from_slice(&[0x4D, 0x39, 0xEC]);
                code.extend_from_slice(&[0x0F, 0x82, 0x00, 0x00, 0x00, 0x00]);
                let have_cap_patch_1 = code.len() - 4;

                code.extend_from_slice(&[0x4D, 0x85, 0xED]);
                code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]);
                let grow_mremap_patch = code.len() - 4;

                code.extend_from_slice(&[0xBE, 0x00, 0x10, 0x00, 0x00]);
                emit_sc_sys_mmap_anon(code);
                code.extend_from_slice(&[0x49, 0x89, 0xC6]);
                code.extend_from_slice(&[0x41, 0xBD, 0x00, 0x10, 0x00, 0x00]);
                code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]);
                let have_cap_patch_2 = code.len() - 4;

                let grow_mremap_pos = code.len();
                patch_rel32(code, grow_mremap_patch, grow_mremap_pos)?;
                code.extend_from_slice(&[0x4C, 0x89, 0xEA]);
                code.extend_from_slice(&[0x48, 0xD1, 0xE2]);
                emit_movabs_reg(code, 0, max_cap);
                code.extend_from_slice(&[0x48, 0x39, 0xC2, 0x48, 0x0F, 0x47, 0xD0]);
                code.extend_from_slice(&[0x48, 0x89, 0xD3]);
                code.extend_from_slice(&[0x4C, 0x89, 0xF7]);
                code.extend_from_slice(&[0x4C, 0x89, 0xEE]);
                emit_sc_sys_mremap(code);
                code.extend_from_slice(&[0x49, 0x89, 0xC6]);
                code.extend_from_slice(&[0x49, 0x89, 0xDD]);

                let have_cap_pos = code.len();
                patch_rel32(code, have_cap_patch_1, have_cap_pos)?;
                patch_rel32(code, have_cap_patch_2, have_cap_pos)?;

                emit_movabs_reg(code, 2, decl.max_len);
                code.extend_from_slice(&[0x49, 0x39, 0xD5, 0x49, 0x0F, 0x42, 0xD5]);
                code.extend_from_slice(&[0x4C, 0x29, 0xE2]);
                code.extend_from_slice(&[0x4C, 0x89, 0xF6]);
                code.extend_from_slice(&[0x4C, 0x01, 0xE6]);
                code.extend_from_slice(&[0x31, 0xFF, 0x31, 0xC0, 0x0F, 0x05]);
                code.extend_from_slice(&[0x48, 0x83, 0xF8, 0xFC]);
                code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
                let read_eintr_patch = code.len() - 4;
                patch_rel32(code, read_eintr_patch, read_loop_pos)?;
                code.extend_from_slice(&[0x48, 0x85, 0xC0, 0x79, 0x0C]);
                emit_sys_exit(code, AOT_EXIT_REGION_OOM as u32);
                code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
                let read_done_patch = code.len() - 4;
                code.extend_from_slice(&[0x49, 0x01, 0xC4]);
                code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]);
                let loop_back_patch = code.len() - 4;
                patch_rel32(code, loop_back_patch, read_loop_pos)?;

                let input_done_pos = code.len();
                patch_rel32(code, probe_done_patch, input_done_pos)?;
                patch_rel32(code, read_done_patch, input_done_pos)?;

                emit_store_r15_reg(code, 14, off_base);
                emit_store_r15_reg(code, 12, off_len);
                emit_store_r15_reg(code, 13, off_cap);
            }
        } else if decl.min_len > 0 {
            let elem_bytes = decl.elem.region_elem_bytes() as u64;
            let init_cap = align_up_4096(decl.min_len.saturating_mul(elem_bytes));
            emit_movabs_reg(code, 6, init_cap);
            emit_sc_sys_mmap_anon(code);
            emit_store_r15_reg(code, 0, off_base);
            emit_movabs_reg(code, 0, decl.min_len);
            emit_store_r15_reg(code, 0, off_len);
            emit_movabs_reg(code, 0, init_cap);
            emit_store_r15_reg(code, 0, off_cap);
        }
    }
    Ok(())
}

fn emit_sc_region_new(
    code: &mut Vec<u8>,
    slot: usize,
    decl: AotRegionDecl,
    len_home: ScHome,
    spill_base_slot: u32,
) -> Result<(), AotError> {
    let off_base = sc_region_off(slot, 0);
    let off_len = sc_region_off(slot, 8);
    let off_cap = sc_region_off(slot, 16);
    let elem_bytes = decl.elem.region_elem_bytes();

    emit_load_home_into_gpr(code, REG_RAX, len_home, spill_base_slot);
    emit_sc_save_callee_to_r15(code);
    emit_mov_gpr_gpr(code, REG_R12, REG_RAX);

    if decl.min_len > 0 {
        emit_movabs_reg(code, 0, decl.min_len);
        code.extend_from_slice(&[0x49, 0x39, 0xC4, 0x73, 0x0C]);
        emit_sys_exit(code, AOT_EXIT_REGION_OOM as u32);
    }
    emit_movabs_reg(code, 0, decl.max_len);
    code.extend_from_slice(&[0x49, 0x39, 0xC4, 0x76, 0x0C]);
    emit_sys_exit(code, AOT_EXIT_REGION_OOM as u32);

    code.extend_from_slice(&[0x4C, 0x89, 0xE3]); // mov rbx, r12
    if elem_bytes == 8 {
        code.extend_from_slice(&[0x48, 0xC1, 0xE3, 0x03]);
    }

    emit_load_reg_r15(code, 6, off_cap);
    code.extend_from_slice(&[0x48, 0x39, 0xF3]);
    code.extend_from_slice(&[0x0F, 0x86, 0x00, 0x00, 0x00, 0x00]);
    let reuse_patch = code.len() - 4;

    code.extend_from_slice(&[0x48, 0x85, 0xF6]);
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
    let skip_unmap_patch = code.len() - 4;
    emit_load_reg_r15(code, 7, off_base);
    emit_sc_sys_munmap(code);
    code.extend_from_slice(&[0x31, 0xC0]);
    emit_store_r15_reg(code, 0, off_base);
    emit_store_r15_reg(code, 0, off_len);
    emit_store_r15_reg(code, 0, off_cap);

    let skip_unmap_pos = code.len();
    patch_rel32(code, skip_unmap_patch, skip_unmap_pos)?;

    code.extend_from_slice(&[0x48, 0x81, 0xC3, 0xFF, 0x0F, 0x00, 0x00]);
    code.extend_from_slice(&[0x48, 0x81, 0xE3, 0x00, 0xF0, 0xFF, 0xFF]);
    code.extend_from_slice(&[0x48, 0x89, 0xDE]);
    emit_sc_sys_mmap_anon(code);
    emit_store_r15_reg(code, 0, off_base);
    emit_store_r15_reg(code, 12, off_len);
    emit_store_r15_reg(code, 3, off_cap);
    code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]);
    let done_patch = code.len() - 4;

    let reuse_pos = code.len();
    patch_rel32(code, reuse_patch, reuse_pos)?;
    emit_load_reg_r15(code, 7, off_base);
    code.extend_from_slice(&[0x48, 0x89, 0xD9]);
    code.extend_from_slice(&[0x31, 0xC0, 0xFC, 0xF3, 0xAA]);
    emit_store_r15_reg(code, 12, off_len);

    let done_pos = code.len();
    patch_rel32(code, done_patch, done_pos)?;
    emit_sc_restore_callee_from_r15(code);
    Ok(())
}

fn emit_sc_region_grow(
    code: &mut Vec<u8>,
    slot: usize,
    decl: AotRegionDecl,
    add_home: ScHome,
    dst_home: ScHome,
    spill_base_slot: u32,
) -> Result<(), AotError> {
    let off_base = sc_region_off(slot, 0);
    let off_len = sc_region_off(slot, 8);
    let off_cap = sc_region_off(slot, 16);
    let elem_bytes = decl.elem.region_elem_bytes();
    let max_cap = align_up_4096(decl.max_len.saturating_mul(elem_bytes as u64));

    emit_load_home_into_gpr(code, REG_RAX, add_home, spill_base_slot);
    emit_sc_save_callee_to_r15(code);
    emit_mov_gpr_gpr(code, REG_R12, REG_RAX);

    emit_movabs_reg(code, 0, decl.max_len);
    code.extend_from_slice(&[0x49, 0x39, 0xC4, 0x76, 0x0C]);
    emit_sys_exit(code, AOT_EXIT_REGION_OOM as u32);

    emit_load_reg_r15(code, 13, off_len);
    code.extend_from_slice(&[0x4D, 0x89, 0xEE]); // mov r14, r13
    code.extend_from_slice(&[0x4D, 0x01, 0xE6]); // add r14, r12
    code.extend_from_slice(&[0x49, 0x39, 0xC6, 0x76, 0x0C]);
    emit_sys_exit(code, AOT_EXIT_REGION_OOM as u32);

    code.extend_from_slice(&[0x4D, 0x85, 0xE4]);
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
    let done_patch = code.len() - 4;

    code.extend_from_slice(&[0x4C, 0x89, 0xF3]);
    if elem_bytes == 8 {
        code.extend_from_slice(&[0x48, 0xC1, 0xE3, 0x03]);
    }

    emit_load_reg_r15(code, 6, off_cap);
    code.extend_from_slice(&[0x48, 0x39, 0xF3]);
    code.extend_from_slice(&[0x0F, 0x86, 0x00, 0x00, 0x00, 0x00]);
    let zero_tail_patch_1 = code.len() - 4;

    code.extend_from_slice(&[0x48, 0x89, 0xDA]);
    code.extend_from_slice(&[0x48, 0x81, 0xC2, 0xFF, 0x0F, 0x00, 0x00]);
    code.extend_from_slice(&[0x48, 0x81, 0xE2, 0x00, 0xF0, 0xFF, 0xFF]);

    code.extend_from_slice(&[0x48, 0x85, 0xF6]);
    code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]);
    let mremap_patch = code.len() - 4;

    code.extend_from_slice(&[0x48, 0x89, 0xD3]);
    code.extend_from_slice(&[0x48, 0x89, 0xD6]);
    emit_sc_sys_mmap_anon(code);
    emit_store_r15_reg(code, 0, off_base);
    emit_store_r15_reg(code, 3, off_cap);
    code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]);
    let zero_tail_patch_2 = code.len() - 4;

    let mremap_pos = code.len();
    patch_rel32(code, mremap_patch, mremap_pos)?;
    code.extend_from_slice(&[0x48, 0x89, 0xF0, 0x48, 0xD1, 0xE0]);
    code.extend_from_slice(&[0x48, 0x39, 0xD0, 0x48, 0x0F, 0x47, 0xD0]);
    emit_movabs_reg(code, 0, max_cap);
    code.extend_from_slice(&[0x48, 0x39, 0xC2, 0x48, 0x0F, 0x47, 0xD0]);
    code.extend_from_slice(&[0x48, 0x89, 0xD3]);
    emit_load_reg_r15(code, 7, off_base);
    emit_sc_sys_mremap(code);
    emit_store_r15_reg(code, 0, off_base);
    emit_store_r15_reg(code, 3, off_cap);

    let zero_tail_pos = code.len();
    patch_rel32(code, zero_tail_patch_1, zero_tail_pos)?;
    patch_rel32(code, zero_tail_patch_2, zero_tail_pos)?;

    emit_load_reg_r15(code, 7, off_base);
    code.extend_from_slice(&[0x4C, 0x89, 0xE8]);
    if elem_bytes == 8 {
        code.extend_from_slice(&[0x48, 0xC1, 0xE0, 0x03]);
    }
    code.extend_from_slice(&[0x48, 0x01, 0xC7]);
    code.extend_from_slice(&[0x4C, 0x89, 0xE1]);
    if elem_bytes == 8 {
        code.extend_from_slice(&[0x48, 0xC1, 0xE1, 0x03]);
    }
    code.extend_from_slice(&[0x31, 0xC0, 0xFC, 0xF3, 0xAA]);
    emit_store_r15_reg(code, 14, off_len);

    let done_pos = code.len();
    patch_rel32(code, done_patch, done_pos)?;
    code.extend_from_slice(&[0x4C, 0x89, 0xF0]); // mov rax, r14
    emit_sc_restore_callee_from_r15(code);
    emit_store_gpr_into_home(code, dst_home, REG_RAX, spill_base_slot);
    Ok(())
}

fn emit_sc_region_free(code: &mut Vec<u8>, slot: usize) -> Result<(), AotError> {
    let off_base = sc_region_off(slot, 0);
    let off_len = sc_region_off(slot, 8);
    let off_cap = sc_region_off(slot, 16);

    emit_sc_save_callee_to_r15(code);
    emit_load_reg_r15(code, 6, off_cap);
    code.extend_from_slice(&[0x48, 0x85, 0xF6]);
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
    let skip_patch = code.len() - 4;

    emit_load_reg_r15(code, 7, off_base);
    emit_sc_sys_munmap(code);

    let skip_pos = code.len();
    patch_rel32(code, skip_patch, skip_pos)?;
    code.extend_from_slice(&[0x31, 0xC0]);
    emit_store_r15_reg(code, 0, off_base);
    emit_store_r15_reg(code, 0, off_len);
    emit_store_r15_reg(code, 0, off_cap);
    emit_sc_restore_callee_from_r15(code);
    Ok(())
}

/// Emits $O(1)$ 24-byte descriptor transfer (`base`, `len`, `cap`) from `src_slot` to `dst_slot`
/// using scratch `rax`/`r11` and zeroes `src_slot`'s descriptor (no syscall barrier).
fn emit_sc_region_move(code: &mut Vec<u8>, src_slot: usize, dst_slot: usize) {
    let src_base = sc_region_off(src_slot, 0);
    let src_len = sc_region_off(src_slot, 8);
    let src_cap = sc_region_off(src_slot, 16);
    let dst_base = sc_region_off(dst_slot, 0);
    let dst_len = sc_region_off(dst_slot, 8);
    let dst_cap = sc_region_off(dst_slot, 16);

    emit_load_reg_r15(code, REG_RAX, src_base);
    emit_load_reg_r15(code, REG_R11, src_len);
    emit_store_r15_reg(code, REG_RAX, dst_base);
    emit_store_r15_reg(code, REG_R11, dst_len);
    emit_load_reg_r15(code, REG_RAX, src_cap);
    emit_store_r15_reg(code, REG_RAX, dst_cap);

    code.extend_from_slice(&[0x31, 0xC0]); // xor eax, eax
    emit_store_r15_reg(code, REG_RAX, src_base);
    emit_store_r15_reg(code, REG_RAX, src_len);
    emit_store_r15_reg(code, REG_RAX, src_cap);
}

fn emit_sc_output_region_epilogue(code: &mut Vec<u8>, out_slot: usize) -> Result<(), AotError> {
    let off_base = sc_region_off(out_slot, 0);
    let off_len = sc_region_off(out_slot, 8);

    emit_load_reg_r15(code, 12, off_base);
    emit_load_reg_r15(code, 13, off_len);
    code.extend_from_slice(&[0x45, 0x31, 0xF6]);

    let write_loop_pos = code.len();
    code.extend_from_slice(&[0x4D, 0x39, 0xEE]);
    code.extend_from_slice(&[0x0F, 0x83, 0x00, 0x00, 0x00, 0x00]);
    let done_patch = code.len() - 4;

    code.extend_from_slice(&[0xBF, 0x01, 0x00, 0x00, 0x00]);
    code.extend_from_slice(&[0x4C, 0x89, 0xE6]);
    code.extend_from_slice(&[0x4C, 0x01, 0xF6]);
    code.extend_from_slice(&[0x4C, 0x89, 0xEA]);
    code.extend_from_slice(&[0x4C, 0x29, 0xF2]);
    code.extend_from_slice(&[0xB8, 0x01, 0x00, 0x00, 0x00]);
    code.extend_from_slice(&[0x0F, 0x05]);
    code.extend_from_slice(&[0x48, 0x83, 0xF8, 0xFC]);
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
    let eintr_patch = code.len() - 4;
    patch_rel32(code, eintr_patch, write_loop_pos)?;
    code.extend_from_slice(&[0x48, 0x85, 0xC0, 0x7F, 0x0C]);
    emit_sys_exit(code, AOT_EXIT_REGION_OOM as u32);
    code.extend_from_slice(&[0x49, 0x01, 0xC6]);
    code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]);
    let back_patch = code.len() - 4;
    patch_rel32(code, back_patch, write_loop_pos)?;

    let done_pos = code.len();
    patch_rel32(code, done_patch, done_pos)?;
    emit_sys_exit(code, 0);
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "self-contained checked integer division/remainder emitter"
)]
fn emit_sc_int_div_rem(
    code: &mut Vec<u8>,
    is_u8: bool,
    is_rem: bool,
    home_a: ScHome,
    home_b: ScHome,
    dst_home: ScHome,
    rdx_live: bool,
    spill_base_slot: u32,
) -> Result<(), AotError> {
    emit_load_home_into_gpr(code, REG_R11, home_b, spill_base_slot);
    // test r11, r11; jnz +12; exit(72)
    code.extend_from_slice(&[0x4D, 0x85, 0xDB, 0x75, 0x0C]);
    emit_sys_exit(code, AOT_EXIT_INT_DIV_ERR as u32);

    emit_load_home_into_gpr(code, REG_RAX, home_a, spill_base_slot);
    let save_rdx = rdx_live && dst_home.gpr_reg() != Some(REG_RDX);
    if save_rdx {
        emit_push_gpr(code, REG_RDX);
    }

    if is_u8 {
        // xor edx, edx; div r11
        code.extend_from_slice(&[0x31, 0xD2, 0x49, 0xF7, 0xF3]);
    } else {
        // Guard against i64::MIN / -1 and i64::MIN % -1 -> exit(72).
        emit_movabs_reg(code, REG_RDX, 0x8000_0000_0000_0000);
        code.extend_from_slice(&[0x48, 0x39, 0xD0]); // cmp rax, rdx
        code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]);
        let ok_patch_1 = code.len() - 4;
        code.extend_from_slice(&[0x49, 0x83, 0xFB, 0xFF]); // cmp r11, -1
        code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]);
        let ok_patch_2 = code.len() - 4;
        emit_sys_exit(code, AOT_EXIT_INT_DIV_ERR as u32);

        let ok_pos = code.len();
        patch_rel32(code, ok_patch_1, ok_pos)?;
        patch_rel32(code, ok_patch_2, ok_pos)?;
        // cqo; idiv r11
        code.extend_from_slice(&[0x48, 0x99, 0x49, 0xF7, 0xFB]);
    }

    if is_rem {
        emit_mov_gpr_gpr(code, REG_RAX, REG_RDX);
    }
    if save_rdx {
        emit_pop_gpr(code, REG_RDX);
    }
    emit_store_gpr_into_home(code, dst_home, REG_RAX, spill_base_slot);
    Ok(())
}

fn emit_sc_cast(
    code: &mut Vec<u8>,
    src_elem: ElemType,
    dst_elem: ElemType,
    home_a: ScHome,
    dst_home: ScHome,
    spill_base_slot: u32,
) -> Result<(), AotError> {
    match (src_elem, dst_elem) {
        (ElemType::I64, ElemType::U8) | (ElemType::U8, ElemType::I64) => {
            let dst_r = dst_home.gpr_reg().unwrap_or(REG_RAX);
            if let Some(src_r) = home_a.gpr_reg() {
                emit_movzx_gpr32_gpr8(code, dst_r, src_r);
            } else {
                emit_load_home_into_gpr(code, dst_r, home_a, spill_base_slot);
                emit_movzx_gpr32_gpr8(code, dst_r, dst_r);
            }
            emit_store_gpr_into_home(code, dst_home, dst_r, spill_base_slot);
        }
        (ElemType::BoolMask, ElemType::I64) => {
            let dst_r = dst_home.gpr_reg().unwrap_or(REG_RAX);
            emit_load_home_into_gpr(code, dst_r, home_a, spill_base_slot);
            emit_alu32_ri(code, 4, dst_r, 1);
            emit_store_gpr_into_home(code, dst_home, dst_r, spill_base_slot);
        }
        (ElemType::I64, ElemType::F64) => {
            let dst_x = dst_home.xmm_reg().unwrap_or(0);
            if let Some(src_r) = home_a.gpr_reg() {
                // cvtsi2sd dst_x, src_r
                code.extend_from_slice(&[
                    0xF2,
                    sc_rex(true, dst_x, 0, src_r),
                    0x0F,
                    0x2A,
                    0xC0 | ((dst_x & 7) << 3) | (src_r & 7),
                ]);
            } else if let ScHome::Spill(s) = home_a {
                let disp = sc_spill_disp(spill_base_slot, s);
                code.extend_from_slice(&[0xF2, sc_rex(true, dst_x, 0, REG_RBP), 0x0F, 0x2A]);
                emit_rbp_disp_modrm(code, dst_x, disp);
            } else {
                emit_load_home_into_gpr(code, REG_RAX, home_a, spill_base_slot);
                code.extend_from_slice(&[
                    0xF2,
                    sc_rex(true, dst_x, 0, REG_RAX),
                    0x0F,
                    0x2A,
                    0xC0 | ((dst_x & 7) << 3),
                ]);
            }
            emit_store_xmm_into_home(code, dst_home, dst_x, spill_base_slot);
        }
        (ElemType::U8, ElemType::F64) => {
            let dst_x = dst_home.xmm_reg().unwrap_or(0);
            if let Some(src_r) = home_a.gpr_reg() {
                emit_movzx_gpr32_gpr8(code, REG_RAX, src_r);
            } else {
                emit_load_home_into_gpr(code, REG_RAX, home_a, spill_base_slot);
                emit_movzx_gpr32_gpr8(code, REG_RAX, REG_RAX);
            }
            code.extend_from_slice(&[
                0xF2,
                sc_rex(true, dst_x, 0, REG_RAX),
                0x0F,
                0x2A,
                0xC0 | ((dst_x & 7) << 3),
            ]);
            emit_store_xmm_into_home(code, dst_home, dst_x, spill_base_slot);
        }
        (ElemType::F64, ElemType::I64) => {
            emit_load_home_into_xmm(code, 0, home_a, spill_base_slot);
            emit_movabs_reg(code, 0, (-9_223_372_036_854_775_808.0_f64).to_bits());
            code.extend_from_slice(&[0x66, 0x48, 0x0F, 0x6E, 0xC8]);
            code.extend_from_slice(&[0x66, 0x0F, 0x2E, 0xC1, 0x73, 0x0C]);
            emit_sys_exit(code, AOT_EXIT_CAST_ERR as u32);
            emit_movabs_reg(code, 0, (9_223_372_036_854_775_808.0_f64).to_bits());
            code.extend_from_slice(&[0x66, 0x48, 0x0F, 0x6E, 0xC8]);
            code.extend_from_slice(&[0x66, 0x0F, 0x2E, 0xC8, 0x77, 0x0C]);
            emit_sys_exit(code, AOT_EXIT_CAST_ERR as u32);
            let dst_r = dst_home.gpr_reg().unwrap_or(REG_RAX);
            code.extend_from_slice(&[
                0xF2,
                sc_rex(true, dst_r, 0, 0),
                0x0F,
                0x2C,
                0xC0 | ((dst_r & 7) << 3),
            ]);
            emit_store_gpr_into_home(code, dst_home, dst_r, spill_base_slot);
        }
        (ElemType::F64, ElemType::U8) => {
            emit_load_home_into_xmm(code, 0, home_a, spill_base_slot);
            code.extend_from_slice(&[0x66, 0x0F, 0x3A, 0x0B, 0xC0, 0x0B]);
            code.extend_from_slice(&[0x66, 0x0F, 0x57, 0xC9, 0x66, 0x0F, 0x2E, 0xC1, 0x73, 0x0C]);
            emit_sys_exit(code, AOT_EXIT_CAST_ERR as u32);
            emit_movabs_reg(code, 0, 255.0_f64.to_bits());
            code.extend_from_slice(&[0x66, 0x48, 0x0F, 0x6E, 0xC8]);
            code.extend_from_slice(&[0x66, 0x0F, 0x2E, 0xC8, 0x73, 0x0C]);
            emit_sys_exit(code, AOT_EXIT_CAST_ERR as u32);
            let dst_r = dst_home.gpr_reg().unwrap_or(REG_RAX);
            code.push(0xF2);
            if dst_r >= 8 {
                code.push(sc_rex(false, dst_r, 0, 0));
            }
            code.extend_from_slice(&[0x0F, 0x2C, 0xC0 | ((dst_r & 7) << 3)]);
            emit_store_gpr_into_home(code, dst_home, dst_r, spill_base_slot);
        }
        _ => {
            return Err(AotError::UnsupportedGraph(
                "unsupported Cast element-type pair in structured AOT",
            ));
        }
    }
    Ok(())
}

/// Emits integer comparison (`I64` signed, `U8` unsigned, or `CmpLtU` unsigned) and returns the
/// `jcc rel32` second opcode byte (`0x82..=0x8F`) for the TRUE condition.
fn emit_sc_int_cmp(
    code: &mut Vec<u8>,
    opcode: Opcode,
    elem: ElemType,
    home_a: ScHome,
    home_b: ScHome,
    spill_base_slot: u32,
) -> u8 {
    let mut op = opcode;
    let mut ha = home_a;
    let mut hb = home_b;

    // If `a` is a constant and `b` is a register, swap operands and flip Lt/Gt so we can use `cmp r, imm`.
    let a_is_const = matches!(ha, ScHome::ConstI64(_) | ScHome::ConstU8(_));
    let b_is_gpr = hb.gpr_reg().is_some();
    if a_is_const && b_is_gpr {
        std::mem::swap(&mut ha, &mut hb);
        op = match op {
            Opcode::CmpLt => Opcode::CmpGt,
            Opcode::CmpGt => Opcode::CmpLt,
            Opcode::CmpLtU => Opcode::CmpGt, // handled below via swapped_unsigned flag
            other => other,
        };
        if opcode == Opcode::CmpLtU {
            // Original `const <u reg` == `reg >u const` -> `ja` (`0x87`).
            if let Some(reg_a) = ha.gpr_reg() {
                match hb {
                    ScHome::ConstI64(v) => emit_alu64_ri(code, 7, reg_a, v),
                    ScHome::ConstU8(b) => emit_alu64_ri(code, 7, reg_a, i64::from(b)),
                    _ => {}
                }
                return 0x87;
            }
        }
    }

    let reg_a = if let Some(r) = ha.gpr_reg() {
        r
    } else {
        emit_load_home_into_gpr(code, REG_RAX, ha, spill_base_slot);
        REG_RAX
    };
    match hb {
        ScHome::Gpr(r) | ScHome::AliasGpr(r) => emit_alu64_rr(code, 0x3B, reg_a, r),
        ScHome::Spill(s) => emit_alu64_rm_rbp(code, 0x3B, reg_a, sc_spill_disp(spill_base_slot, s)),
        ScHome::ConstI64(v) => emit_alu64_ri(code, 7, reg_a, v),
        ScHome::ConstU8(b) => emit_alu64_ri(code, 7, reg_a, i64::from(b)),
        _ => {
            emit_load_home_into_gpr(code, REG_R11, hb, spill_base_slot);
            emit_alu64_rr(code, 0x3B, reg_a, REG_R11);
        }
    }

    if opcode == Opcode::CmpLtU {
        0x82 // jb
    } else if elem == ElemType::U8 {
        match op {
            Opcode::CmpGt => 0x87, // ja
            Opcode::CmpLt => 0x82, // jb
            _ => 0x84,             // je
        }
    } else {
        match op {
            Opcode::CmpGt => 0x8F, // jg
            Opcode::CmpLt => 0x8C, // jl
            _ => 0x84,             // je
        }
    }
}

fn emit_sc_int_binary(
    code: &mut Vec<u8>,
    opcode: Opcode,
    elem: ElemType,
    home_a: ScHome,
    home_b: ScHome,
    dst_home: ScHome,
    spill_base_slot: u32,
) {
    // Fast path: `BitAnd` with `0xFFFF_FFFF` lowers to a single 32-bit zero-extending move.
    if opcode == Opcode::BitAnd && elem == ElemType::I64 {
        let masked_src = match (home_a, home_b) {
            (other, ScHome::ConstI64(0xFFFF_FFFF)) | (ScHome::ConstI64(0xFFFF_FFFF), other) => {
                Some(other)
            }
            _ => None,
        };
        if let Some(src) = masked_src {
            let dst_r = dst_home.gpr_reg().unwrap_or(REG_RAX);
            match src {
                ScHome::Gpr(r) | ScHome::AliasGpr(r) => emit_mov_gpr32_gpr32(code, dst_r, r),
                ScHome::Spill(s) => {
                    emit_load_gpr32_rbp(code, dst_r, sc_spill_disp(spill_base_slot, s));
                }
                ScHome::ConstI64(v) => {
                    #[expect(clippy::cast_sign_loss, reason = "masking lower 32 bits")]
                    emit_mov_gpr_imm64(code, dst_r, (v as u64) & 0xFFFF_FFFF);
                }
                ScHome::ConstU8(b) => emit_mov_gpr_imm64(code, dst_r, u64::from(b)),
                _ => {
                    emit_load_home_into_gpr(code, dst_r, src, spill_base_slot);
                    emit_mov_gpr32_gpr32(code, dst_r, dst_r);
                }
            }
            emit_store_gpr_into_home(code, dst_home, dst_r, spill_base_slot);
            return;
        }
    }

    let is_u8 = elem == ElemType::U8;
    let commutative = matches!(
        opcode,
        Opcode::Add
            | Opcode::Mul
            | Opcode::BitAnd
            | Opcode::BitOr
            | Opcode::BitXor
            | Opcode::MaskAnd
            | Opcode::MaskOr
    );
    let dst_r = dst_home.gpr_reg().unwrap_or(REG_RAX);

    // 3-operand `lea` fast paths for `I64` `Add`/`Sub` when `dst_r` differs from source.
    if !is_u8 && dst_home.gpr_reg().is_some() {
        if opcode == Opcode::Add {
            match (home_a, home_b) {
                (ScHome::Gpr(ra) | ScHome::AliasGpr(ra), ScHome::ConstI64(v))
                | (ScHome::ConstI64(v), ScHome::Gpr(ra) | ScHome::AliasGpr(ra))
                    if ra != dst_r =>
                {
                    if let Ok(d32) = i32::try_from(v) {
                        emit_lea_reg_disp(code, dst_r, ra, d32);
                        return;
                    }
                }
                (
                    ScHome::Gpr(ra) | ScHome::AliasGpr(ra),
                    ScHome::Gpr(rb) | ScHome::AliasGpr(rb),
                ) if ra != dst_r && rb != dst_r => {
                    emit_lea_reg_plus_reg(code, dst_r, ra, rb);
                    return;
                }
                _ => {}
            }
        } else if opcode == Opcode::Sub {
            if let (ScHome::Gpr(ra) | ScHome::AliasGpr(ra), ScHome::ConstI64(v)) = (home_a, home_b)
            {
                if ra != dst_r {
                    if let Some(neg) = v.checked_neg().and_then(|n| i32::try_from(n).ok()) {
                        emit_lea_reg_disp(code, dst_r, ra, neg);
                        return;
                    }
                }
            }
        } else if opcode == Opcode::Mul {
            match (home_a, home_b) {
                (ScHome::Gpr(ra) | ScHome::AliasGpr(ra), ScHome::ConstI64(v))
                | (ScHome::ConstI64(v), ScHome::Gpr(ra) | ScHome::AliasGpr(ra)) => {
                    if let Ok(i32v) = i32::try_from(v) {
                        emit_imul64_rri(code, dst_r, ra, i32v);
                        return;
                    }
                }
                _ => {}
            }
        }
    }

    // Order operands so if `dst_r == b` on a commutative op we swap `(a, b)`;
    // on non-commutative `Sub`, if `dst_r == b`, compute in `REG_RAX` first.
    let (mut ha, mut hb) = (home_a, home_b);
    let mut work_r = dst_r;
    if commutative
        && ((hb.gpr_reg() == Some(dst_r) && ha.gpr_reg() != Some(dst_r))
            || (matches!(ha, ScHome::ConstI64(_) | ScHome::ConstU8(_))
                && !matches!(hb, ScHome::ConstI64(_) | ScHome::ConstU8(_))))
    {
        std::mem::swap(&mut ha, &mut hb);
    }
    if !commutative && hb.gpr_reg() == Some(work_r) && ha.gpr_reg() != Some(work_r) {
        work_r = REG_RAX;
    }

    emit_load_home_into_gpr(code, work_r, ha, spill_base_slot);

    if is_u8 {
        match opcode {
            Opcode::Add | Opcode::Sub => {
                let (op, ext) = if opcode == Opcode::Add {
                    (0x03, 0)
                } else {
                    (0x2B, 5)
                };
                match hb {
                    ScHome::Gpr(r) | ScHome::AliasGpr(r) => emit_alu32_rr(code, op, work_r, r),
                    ScHome::Spill(s) => {
                        emit_alu32_rm_rbp(code, op, work_r, sc_spill_disp(spill_base_slot, s));
                    }
                    ScHome::ConstU8(b) => emit_alu32_ri(code, ext, work_r, i32::from(b)),
                    ScHome::ConstI64(v) => {
                        emit_alu32_ri(code, ext, work_r, i32::try_from(v & 0xFF).unwrap_or(0));
                    }
                    _ => {
                        emit_load_home_into_gpr(code, REG_R11, hb, spill_base_slot);
                        emit_alu32_rr(code, op, work_r, REG_R11);
                    }
                }
                // and work_r32, 0xFF
                emit_alu32_ri(code, 4, work_r, 0xFF);
            }
            Opcode::Mul => {
                match hb {
                    ScHome::Gpr(r) | ScHome::AliasGpr(r) => {
                        if work_r >= 8 || r >= 8 {
                            code.push(sc_rex(false, work_r, 0, r));
                        }
                        code.extend_from_slice(&[0x0F, 0xAF, 0xC0 | ((work_r & 7) << 3) | (r & 7)]);
                    }
                    ScHome::Spill(s) => {
                        if work_r >= 8 {
                            code.push(sc_rex(false, work_r, 0, REG_RBP));
                        }
                        code.extend_from_slice(&[0x0F, 0xAF]);
                        emit_rbp_disp_modrm(code, work_r, sc_spill_disp(spill_base_slot, s));
                    }
                    ScHome::ConstU8(b) => {
                        if work_r >= 8 {
                            code.push(sc_rex(false, work_r, 0, work_r));
                        }
                        code.extend_from_slice(&[0x69, 0xC0 | ((work_r & 7) << 3) | (work_r & 7)]);
                        code.extend_from_slice(&i32::from(b).to_le_bytes());
                    }
                    _ => {
                        emit_load_home_into_gpr(code, REG_R11, hb, spill_base_slot);
                        code.extend_from_slice(&[
                            sc_rex(false, work_r, 0, REG_R11),
                            0x0F,
                            0xAF,
                            0xC0 | ((work_r & 7) << 3) | (REG_R11 & 7),
                        ]);
                    }
                }
                emit_alu32_ri(code, 4, work_r, 0xFF);
            }
            Opcode::BitAnd | Opcode::BitOr | Opcode::BitXor => {
                let (op, ext) = match opcode {
                    Opcode::BitAnd => (0x23, 4),
                    Opcode::BitOr => (0x0B, 1),
                    _ => (0x33, 6),
                };
                match hb {
                    ScHome::Gpr(r) | ScHome::AliasGpr(r) => emit_alu64_rr(code, op, work_r, r),
                    ScHome::Spill(s) => {
                        emit_alu64_rm_rbp(code, op, work_r, sc_spill_disp(spill_base_slot, s));
                    }
                    ScHome::ConstU8(b) => emit_alu64_ri(code, ext, work_r, i64::from(b)),
                    _ => {
                        emit_load_home_into_gpr(code, REG_R11, hb, spill_base_slot);
                        emit_alu64_rr(code, op, work_r, REG_R11);
                    }
                }
            }
            _ => {}
        }
    } else if opcode == Opcode::Mul {
        match hb {
            ScHome::Gpr(r) | ScHome::AliasGpr(r) => emit_imul64_rr(code, work_r, r),
            ScHome::Spill(s) => {
                emit_imul64_rm_rbp(code, work_r, sc_spill_disp(spill_base_slot, s));
            }
            ScHome::ConstI64(v) => {
                if let Ok(i32v) = i32::try_from(v) {
                    emit_imul64_rri(code, work_r, work_r, i32v);
                } else {
                    emit_load_home_into_gpr(code, REG_R11, hb, spill_base_slot);
                    emit_imul64_rr(code, work_r, REG_R11);
                }
            }
            _ => {
                emit_load_home_into_gpr(code, REG_R11, hb, spill_base_slot);
                emit_imul64_rr(code, work_r, REG_R11);
            }
        }
    } else {
        let (op, ext) = match opcode {
            Opcode::Add => (0x03, 0),
            Opcode::Sub => (0x2B, 5),
            Opcode::BitAnd | Opcode::MaskAnd => (0x23, 4),
            Opcode::BitOr | Opcode::MaskOr => (0x0B, 1),
            _ => (0x33, 6), // BitXor
        };
        match hb {
            ScHome::Gpr(r) | ScHome::AliasGpr(r) => emit_alu64_rr(code, op, work_r, r),
            ScHome::Spill(s) => {
                emit_alu64_rm_rbp(code, op, work_r, sc_spill_disp(spill_base_slot, s));
            }
            ScHome::ConstI64(v) => emit_alu64_ri(code, ext, work_r, v),
            ScHome::ConstU8(b) => emit_alu64_ri(code, ext, work_r, i64::from(b)),
            _ => {
                emit_load_home_into_gpr(code, REG_R11, hb, spill_base_slot);
                emit_alu64_rr(code, op, work_r, REG_R11);
            }
        }
    }

    emit_store_gpr_into_home(code, dst_home, work_r, spill_base_slot);
}

#[expect(
    clippy::too_many_arguments,
    reason = "self-contained shift emitter handling immediate and dynamic shift counts"
)]
fn emit_sc_shift(
    code: &mut Vec<u8>,
    opcode: Opcode,
    elem: ElemType,
    home_a: ScHome,
    home_b: ScHome,
    dst_home: ScHome,
    rcx_live: bool,
    spill_base_slot: u32,
) {
    let is_u8 = elem == ElemType::U8;
    // Constant shift amount fast path (zero `rcx` traffic).
    let const_cnt = match home_b {
        ScHome::ConstI64(v) => Some(u8::try_from(v & 63).unwrap_or(0)),
        ScHome::ConstU8(b) => Some(b),
        _ => None,
    };
    if let Some(cnt) = const_cnt {
        let dst_r = dst_home.gpr_reg().unwrap_or(REG_RAX);
        emit_load_home_into_gpr(code, dst_r, home_a, spill_base_slot);
        if is_u8 {
            let c = cnt & 7;
            if opcode == Opcode::Shl {
                emit_shift32_ri(code, 4, dst_r, c);
                emit_alu32_ri(code, 4, dst_r, 0xFF);
            } else {
                emit_shift32_ri(code, 5, dst_r, c);
            }
        } else {
            let ext = match opcode {
                Opcode::Shl => 4,
                Opcode::ShrU => 5,
                _ => 7,
            };
            emit_shift64_ri(code, ext, dst_r, cnt);
        }
        emit_store_gpr_into_home(code, dst_home, dst_r, spill_base_slot);
        return;
    }

    // Dynamic shift amount requires `cl` (`REG_RCX`).
    emit_load_home_into_gpr(code, REG_RAX, home_a, spill_base_slot);
    let save_rcx = rcx_live && dst_home.gpr_reg() != Some(REG_RCX);
    if save_rcx {
        emit_push_gpr(code, REG_RCX);
    }
    emit_load_home_into_gpr(code, REG_RCX, home_b, spill_base_slot);
    if is_u8 {
        code.extend_from_slice(&[0x83, 0xE1, 0x07]); // and ecx, 7
        if opcode == Opcode::Shl {
            code.extend_from_slice(&[0xD3, 0xE0, 0x25, 0xFF, 0x00, 0x00, 0x00]);
        } else {
            code.extend_from_slice(&[0xD3, 0xE8]);
        }
    } else {
        let shift_modrm = match opcode {
            Opcode::Shl => 0xE0,
            Opcode::ShrU => 0xE8,
            _ => 0xF8,
        };
        code.extend_from_slice(&[0x48, 0xD3, shift_modrm]);
    }
    if save_rcx {
        emit_pop_gpr(code, REG_RCX);
    }
    emit_store_gpr_into_home(code, dst_home, REG_RAX, spill_base_slot);
}

struct ScUnitMeta {
    local_elem: [ElemType; 64],
    local_used: [bool; 64],
    local_score: [u64; 64],
    has_x87: bool,
}

#[expect(
    clippy::too_many_lines,
    reason = "single-pass O(N) liveness, loop-extension, local-scoring, and branch-fusion pre-scan"
)]
fn sc_analyze_unit(
    unit_nodes: &[crate::bytecode::NodeFrame],
    regions: &[Option<AotRegionDecl>; MAX_REGION_SLOTS],
    func_sigs: &[Option<FuncSigInfo>],
    cur_sig: Option<FuncSigInfo>,
    output_node_opt: Option<usize>,
    scratch: &mut ScUnitScratch,
) -> Result<ScUnitMeta, AotError> {
    let len = unit_nodes.len();
    scratch.reset_for_unit(len);

    let mut local_elem: [ElemType; 64] = [ElemType::I64; 64];
    let mut local_used: [bool; 64] = [false; 64];
    let mut local_score: [u64; 64] = [0; 64];
    let mut has_x87 = false;
    let mut pending_call_args: Option<(usize, Option<usize>)> = None;

    if let Some(sig) = cur_sig {
        for i in 0..usize::from(sig.arity) {
            if let (Some(&pe), Some(le), Some(lu), Some(ls)) = (
                sig.params.get(i),
                local_elem.get_mut(i),
                local_used.get_mut(i),
                local_score.get_mut(i),
            ) {
                *le = pe;
                *lu = true;
                *ls = 1;
            }
        }
    }

    let record_use = |scratch: &mut ScUnitScratch, v: usize, at_idx: usize| {
        let at_u32 = u32::try_from(at_idx).unwrap_or(u32::MAX);
        if let Some(info) = scratch.nodes.get_mut(v) {
            info.inc_use();
            if at_u32 > info.last_use {
                info.last_use = at_u32;
            }
            if !scratch.loop_stack.is_empty() {
                let d = usize::from(info.home_data);
                if let Some(&hdr) = scratch.loop_stack.get(d) {
                    info.outer_loop_header = hdr;
                }
            }
        }
    };

    let mut barriers: u32 = 0;

    for (idx, frame) in unit_nodes.iter().enumerate() {
        if scratch.has_any_barrier {
            if let Some(cp) = scratch.call_prefix.get_mut(idx) {
                *cp = barriers;
            }
        }
        let loop_depth_u16 = u16::try_from(scratch.loop_stack.len()).unwrap_or(u16::MAX);
        if let Some(info) = scratch.nodes.get_mut(idx) {
            info.home_data = loop_depth_u16;
        }
        let idx_u32 = u32::try_from(idx).unwrap_or(0);
        let a_idx = frame.in_a as usize;
        let b_idx = frame.in_b as usize;

        match frame.opcode {
            Opcode::Func => {
                if let Some(info) = scratch.nodes.get_mut(idx) {
                    info.set_elem(ElemType::I64);
                }
                scratch.ctrl_scan.push((Opcode::Func, idx_u32));
            }
            Opcode::ConstF64 => {
                if let Some(info) = scratch.nodes.get_mut(idx) {
                    info.set_elem(ElemType::F64);
                }
            }
            Opcode::ConstI64 => {
                if let Some(info) = scratch.nodes.get_mut(idx) {
                    info.set_elem(ElemType::I64);
                }
            }
            Opcode::LocalGet => {
                let slot = frame.in_a as usize;
                let el_val = local_elem
                    .get(slot)
                    .copied()
                    .unwrap_or(frame.type_lane.elem);
                let prev_head = scratch
                    .local_get_head
                    .get(slot)
                    .copied()
                    .unwrap_or(u32::MAX);
                if let Some(info) = scratch.nodes.get_mut(idx) {
                    info.set_elem(el_val);
                    info.aux = prev_head;
                }
                if let Some(h) = scratch.local_get_head.get_mut(slot) {
                    *h = idx_u32;
                }
                let weight = 1u64 << u32::from(loop_depth_u16.min(20));
                if let Some(lu) = local_used.get_mut(slot) {
                    *lu = true;
                }
                if let Some(ls) = local_score.get_mut(slot) {
                    *ls = ls.saturating_add(weight);
                }
            }
            Opcode::LocalSet => {
                let slot = frame.in_a as usize;
                record_use(scratch, b_idx, idx);
                let el_val = scratch
                    .nodes
                    .get(b_idx)
                    .map_or(frame.type_lane.elem, |n| n.elem());
                if let Some(le) = local_elem.get_mut(slot) {
                    *le = el_val;
                }
                if let Some(info) = scratch.nodes.get_mut(idx) {
                    info.set_elem(el_val);
                }
                let weight = 1u64 << u32::from(loop_depth_u16.min(20));
                if let Some(lu) = local_used.get_mut(slot) {
                    *lu = true;
                }
                if let Some(ls) = local_score.get_mut(slot) {
                    *ls = ls.saturating_add(weight);
                }
                let mut cur = scratch
                    .local_get_head
                    .get_mut(slot)
                    .map_or(u32::MAX, |h| std::mem::replace(h, u32::MAX));
                while cur != u32::MAX {
                    if let Some(info) = scratch.nodes.get_mut(cur as usize) {
                        let next = info.aux;
                        info.aux = idx_u32;
                        cur = next;
                    } else {
                        break;
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
            | Opcode::ShrS => {
                record_use(scratch, a_idx, idx);
                record_use(scratch, b_idx, idx);
                let el_a = scratch
                    .nodes
                    .get(a_idx)
                    .map_or(frame.type_lane.elem, |n| n.elem());
                if let Some(info) = scratch.nodes.get_mut(idx) {
                    info.set_elem(el_a);
                }
            }
            Opcode::Min | Opcode::Max => {
                record_use(scratch, a_idx, idx);
                record_use(scratch, b_idx, idx);
                if let Some(info) = scratch.nodes.get_mut(idx) {
                    info.set_elem(ElemType::F64);
                }
            }
            Opcode::Fma => {
                let (in_b, in_c) = frame.decode_fma_bc();
                record_use(scratch, a_idx, idx);
                record_use(scratch, in_b as usize, idx);
                record_use(scratch, in_c as usize, idx);
                if let Some(info) = scratch.nodes.get_mut(idx) {
                    info.set_elem(ElemType::F64);
                }
            }
            Opcode::Abs | Opcode::Sqrt | Opcode::Floor | Opcode::Ceil => {
                record_use(scratch, a_idx, idx);
                if let Some(info) = scratch.nodes.get_mut(idx) {
                    info.set_elem(ElemType::F64);
                }
            }
            Opcode::Ln | Opcode::Exp => {
                has_x87 = true;
                record_use(scratch, a_idx, idx);
                if let Some(info) = scratch.nodes.get_mut(idx) {
                    info.set_elem(ElemType::F64);
                }
            }
            Opcode::MaskNot => {
                record_use(scratch, a_idx, idx);
                if let Some(info) = scratch.nodes.get_mut(idx) {
                    info.set_elem(ElemType::BoolMask);
                }
            }
            Opcode::Cast => {
                record_use(scratch, a_idx, idx);
                if let Some(info) = scratch.nodes.get_mut(idx) {
                    info.set_elem(frame.type_lane.elem);
                }
            }
            Opcode::CmpGt | Opcode::CmpLt | Opcode::CmpEq | Opcode::CmpLtU => {
                record_use(scratch, a_idx, idx);
                record_use(scratch, b_idx, idx);
                if let Some(info) = scratch.nodes.get_mut(idx) {
                    info.set_elem(ElemType::BoolMask);
                }
            }
            Opcode::MaskAnd | Opcode::MaskOr => {
                record_use(scratch, a_idx, idx);
                record_use(scratch, b_idx, idx);
                if let Some(info) = scratch.nodes.get_mut(idx) {
                    info.set_elem(ElemType::BoolMask);
                }
            }
            Opcode::SelectZero => {
                record_use(scratch, a_idx, idx);
                record_use(scratch, b_idx, idx);
                let el_b = scratch
                    .nodes
                    .get(b_idx)
                    .map_or(frame.type_lane.elem, |n| n.elem());
                if let Some(info) = scratch.nodes.get_mut(idx) {
                    info.set_elem(el_b);
                }
            }
            Opcode::RingEnqueue | Opcode::AssertContract => {
                record_use(scratch, a_idx, idx);
                let el_a = scratch
                    .nodes
                    .get(a_idx)
                    .map_or(frame.type_lane.elem, |n| n.elem());
                if let Some(info) = scratch.nodes.get_mut(idx) {
                    info.set_elem(el_a);
                }
            }
            Opcode::Block => {
                scratch.ctrl_scan.push((Opcode::Block, idx_u32));
            }
            Opcode::Loop => {
                scratch.has_any_loop = true;
                scratch.ctrl_scan.push((Opcode::Loop, idx_u32));
                scratch.loop_stack.push(idx_u32);
            }
            Opcode::If | Opcode::BrIf => {
                record_use(scratch, a_idx, idx);
                // Detect adjacent integer `Cmp*` or `MaskNot(Cmp*)` fusion candidate.
                let mut cand: Option<(usize, bool)> = None;
                if a_idx.saturating_add(1) == idx {
                    if let Some(cf) = unit_nodes.get(a_idx) {
                        let is_int_cmp = matches!(
                            cf.opcode,
                            Opcode::CmpGt | Opcode::CmpLt | Opcode::CmpEq | Opcode::CmpLtU
                        ) && matches!(
                            scratch.nodes.get(cf.in_a as usize).map(|n| n.elem()),
                            Some(ElemType::I64 | ElemType::U8)
                        );
                        if is_int_cmp {
                            cand = Some((a_idx, false));
                        } else if cf.opcode == Opcode::MaskNot && a_idx > 0 {
                            let c_idx = cf.in_a as usize;
                            if c_idx.saturating_add(1) == a_idx {
                                if let Some(cf2) = unit_nodes.get(c_idx) {
                                    let is_int_cmp2 = matches!(
                                        cf2.opcode,
                                        Opcode::CmpGt
                                            | Opcode::CmpLt
                                            | Opcode::CmpEq
                                            | Opcode::CmpLtU
                                    ) && matches!(
                                        scratch.nodes.get(cf2.in_a as usize).map(|n| n.elem()),
                                        Some(ElemType::I64 | ElemType::U8)
                                    );
                                    if is_int_cmp2 {
                                        cand = Some((c_idx, true));
                                    }
                                }
                            }
                        }
                    }
                }
                if let Some((c_idx, inverted)) = cand {
                    if let Some(info) = scratch.nodes.get_mut(idx) {
                        info.elem_flags |= if inverted {
                            FLAG_BRANCH_FUSE_INV
                        } else {
                            FLAG_BRANCH_FUSE_DIRECT
                        };
                    }
                    if let Some(info) = scratch.nodes.get_mut(c_idx) {
                        info.elem_flags |= if inverted {
                            FLAG_FUSED_BY_NEXT2
                        } else {
                            FLAG_FUSED_BY_NEXT1
                        };
                    }
                    if inverted {
                        if let Some(info) = scratch.nodes.get_mut(c_idx.saturating_add(1)) {
                            info.elem_flags |= FLAG_FUSED_BY_NEXT1;
                        }
                    }
                    // Extend operand lifetimes of `c_idx` to `idx` so they remain live at the branch.
                    if let Some(cf) = unit_nodes.get(c_idx) {
                        let ca = cf.in_a as usize;
                        let cb = cf.in_b as usize;
                        if let Some(info) = scratch.nodes.get_mut(ca) {
                            if idx_u32 > info.last_use {
                                info.last_use = idx_u32;
                            }
                        }
                        if let Some(info) = scratch.nodes.get_mut(cb) {
                            if idx_u32 > info.last_use {
                                info.last_use = idx_u32;
                            }
                        }
                    }
                }
                if frame.opcode == Opcode::If {
                    scratch.ctrl_scan.push((Opcode::If, idx_u32));
                }
            }
            Opcode::Else => {
                if frame.in_b != 0 {
                    record_use(scratch, a_idx, idx);
                }
            }
            Opcode::End => {
                if frame.in_b != 0 {
                    record_use(scratch, a_idx, idx);
                    let el_res = scratch
                        .nodes
                        .get(a_idx)
                        .map_or(frame.type_lane.elem, |n| n.elem());
                    if let Some(info) = scratch.nodes.get_mut(idx) {
                        info.set_elem(el_res);
                    }
                }
                let Some((hdr_op, hdr_idx)) = scratch.ctrl_scan.pop() else {
                    return Err(AotError::UnsupportedGraph("End without open block"));
                };
                if let Some(hdr_info) = scratch.nodes.get_mut(hdr_idx as usize) {
                    hdr_info.aux = idx_u32;
                    if matches!(hdr_op, Opcode::If | Opcode::Block) && frame.in_b != 0 {
                        hdr_info.elem_flags |= FLAG_SCOPE_HAS_RESULT;
                    }
                }
                if hdr_op == Opcode::Loop {
                    let _ = scratch.loop_stack.pop();
                }
                // Update `def_loop_depth` (stored in `home_data` during analysis) of `End` to match its header.
                let hdr_depth = scratch
                    .nodes
                    .get(hdr_idx as usize)
                    .map_or(0, |n| n.home_data);
                if let Some(info) = scratch.nodes.get_mut(idx) {
                    info.home_data = hdr_depth;
                }
            }
            Opcode::Br => {}
            Opcode::Return => {
                if frame.in_a != 0 {
                    record_use(scratch, a_idx, idx);
                }
            }
            Opcode::CallArg => {
                let a2 = frame.in_a as usize;
                let a3 = if frame.in_b == u32::from(CALL_ARG_SENTINEL) {
                    None
                } else {
                    Some(frame.in_b as usize)
                };
                pending_call_args = Some((a2, a3));
            }
            Opcode::Call => {
                let target_id = frame.in_a as usize;
                let callee_sig = func_sigs
                    .get(target_id)
                    .copied()
                    .flatten()
                    .ok_or(AotError::UnsupportedGraph("unknown callee FuncSig"))?;
                let (arg0_raw, arg1_raw) = frame.decode_u16_pair();
                let sentinel = u32::from(CALL_ARG_SENTINEL);
                if arg0_raw != sentinel {
                    record_use(scratch, arg0_raw as usize, idx);
                }
                if arg1_raw != sentinel {
                    record_use(scratch, arg1_raw as usize, idx);
                }
                if let Some((a2, a3_opt)) = pending_call_args.take() {
                    record_use(scratch, a2, idx);
                    if let Some(a3) = a3_opt {
                        record_use(scratch, a3, idx);
                    }
                }
                if let Some(ret_elem) = callee_sig.ret_elem {
                    if let Some(info) = scratch.nodes.get_mut(idx) {
                        info.set_elem(ret_elem);
                    }
                }
                if !scratch.has_any_barrier {
                    scratch.has_any_barrier = true;
                    scratch.call_prefix.resize(len.saturating_add(1), 0);
                }
                barriers = barriers.saturating_add(1);
            }
            Opcode::RegionNew => {
                record_use(scratch, b_idx, idx);
                if !scratch.has_any_barrier {
                    scratch.has_any_barrier = true;
                    scratch.call_prefix.resize(len.saturating_add(1), 0);
                }
                barriers = barriers.saturating_add(1);
            }
            Opcode::RegionGrow => {
                record_use(scratch, b_idx, idx);
                if let Some(info) = scratch.nodes.get_mut(idx) {
                    info.set_elem(ElemType::I64);
                }
                if !scratch.has_any_barrier {
                    scratch.has_any_barrier = true;
                    scratch.call_prefix.resize(len.saturating_add(1), 0);
                }
                barriers = barriers.saturating_add(1);
            }
            Opcode::RegionFree | Opcode::RegionDrop => {
                if !scratch.has_any_barrier {
                    scratch.has_any_barrier = true;
                    scratch.call_prefix.resize(len.saturating_add(1), 0);
                }
                barriers = barriers.saturating_add(1);
            }
            Opcode::RegionLease | Opcode::RegionEndLease | Opcode::RegionMove => {}
            Opcode::RegionLen => {
                if let Some(info) = scratch.nodes.get_mut(idx) {
                    info.set_elem(ElemType::I64);
                }
            }
            Opcode::RegionLoad => {
                record_use(scratch, b_idx, idx);
                let slot = frame.in_a as usize;
                let decl = regions
                    .get(slot)
                    .and_then(|d| *d)
                    .ok_or(AotError::UnsupportedGraph("undeclared RegionLoad slot"))?;
                if let Some(info) = scratch.nodes.get_mut(idx) {
                    info.set_elem(decl.elem);
                }
            }
            Opcode::RegionStore => {
                let (idx_node, val_node) = frame.decode_u16_pair();
                record_use(scratch, idx_node as usize, idx);
                record_use(scratch, val_node as usize, idx);
            }
            Opcode::ArenaInput
            | Opcode::ArenaSlice
            | Opcode::ArenaMutate
            | Opcode::FilterCompact
            | Opcode::ReduceSum
            | Opcode::ReduceMax
            | Opcode::ReduceMin
            | Opcode::Gather => {
                return Err(AotError::UnsupportedGraph(
                    "columnar opcode in scalar structured control-flow AOT",
                ));
            }
        }
    }

    if scratch.has_any_barrier {
        if let Some(cp) = scratch.call_prefix.get_mut(len) {
            *cp = barriers;
        }
    }
    if let Some(out_idx) = output_node_opt {
        record_use(scratch, out_idx, len);
    }
    // Any remaining unclosed `LocalGet` nodes have no later `LocalSet` (`aux = u32::MAX`).
    for head in &mut scratch.local_get_head {
        let mut cur = std::mem::replace(head, u32::MAX);
        while cur != u32::MAX {
            if let Some(info) = scratch.nodes.get_mut(cur as usize) {
                let next = info.aux;
                info.aux = u32::MAX;
                cur = next;
            } else {
                break;
            }
        }
    }

    Ok(ScUnitMeta {
        local_elem,
        local_used,
        local_score,
        has_x87,
    })
}

struct ScRegAllocState {
    free_gpr_mask: u16,
    free_xmm_mask: u16,
    gpr_expire: [u32; 16],
    xmm_expire: [u32; 16],
    min_gpr_expire: u32,
    min_xmm_expire: u32,
    min_spill_expire: u32,
    used_callee_saved_mask: u16,
    next_spill_slot: u32,
}

impl ScRegAllocState {
    fn new(free_gpr_mask: u16, free_xmm_mask: u16, used_callee_saved_mask: u16) -> Self {
        Self {
            free_gpr_mask,
            free_xmm_mask,
            gpr_expire: [u32::MAX; 16],
            xmm_expire: [u32::MAX; 16],
            min_gpr_expire: u32::MAX,
            min_xmm_expire: u32::MAX,
            min_spill_expire: u32::MAX,
            used_callee_saved_mask,
            next_spill_slot: 0,
        }
    }

    #[inline]
    fn alloc_spill(&mut self, eff_last_use: usize, scratch: &mut ScUnitScratch) -> u32 {
        let slot = if let Some(slot) = scratch.free_spill_slots.pop() {
            slot
        } else {
            let slot = self.next_spill_slot;
            self.next_spill_slot = self.next_spill_slot.saturating_add(1);
            slot
        };
        let len_plus_1 = scratch.nodes.len().saturating_add(1);
        if scratch.spill_expire_head.is_empty() {
            scratch.spill_expire_head.resize(len_plus_1, u32::MAX);
        }
        let lu_bucket = eff_last_use.min(len_plus_1.saturating_sub(1));
        let lu_u32 = u32::try_from(lu_bucket).unwrap_or(u32::MAX);
        if lu_u32 < self.min_spill_expire {
            self.min_spill_expire = lu_u32;
        }
        if let Some(eh) = scratch.spill_expire_head.get_mut(lu_bucket) {
            let entry_idx = u32::try_from(scratch.spill_entries.len()).unwrap_or(u32::MAX);
            scratch.spill_entries.push((slot, *eh));
            *eh = entry_idx;
        }
        slot
    }

    #[inline]
    fn alloc_home(
        &mut self,
        node_idx: usize,
        elem: ElemType,
        eff_last_use: usize,
        spans_barrier: bool,
        scratch: &mut ScUnitScratch,
    ) -> ScHome {
        let dist = eff_last_use.saturating_sub(node_idx);
        let lu_u32 = u32::try_from(eff_last_use).unwrap_or(u32::MAX);

        if elem == ElemType::F64 {
            if spans_barrier {
                ScHome::Spill(self.alloc_spill(eff_last_use, scratch))
            } else {
                let avail = self.free_xmm_mask;
                if avail != 0 && !(avail.count_ones() <= 2 && dist > 256) {
                    let reg = u8::try_from(avail.trailing_zeros()).unwrap_or(2);
                    self.free_xmm_mask &= !(1u16 << reg);
                    if let Some(ex) = self.xmm_expire.get_mut(usize::from(reg)) {
                        *ex = lu_u32;
                    }
                    if lu_u32 < self.min_xmm_expire {
                        self.min_xmm_expire = lu_u32;
                    }
                    ScHome::Xmm(reg)
                } else {
                    ScHome::Spill(self.alloc_spill(eff_last_use, scratch))
                }
            }
        } else {
            let candidate_mask = if spans_barrier {
                self.free_gpr_mask & SC_CALLEE_SAVED_MASK
            } else {
                self.free_gpr_mask
            };
            let prefer_spill = candidate_mask.count_ones() <= 2 && dist > 256 && !spans_barrier;
            let chosen = if candidate_mask != 0 && !prefer_spill {
                let order: &[u8] = if spans_barrier {
                    &SC_CALLEE_SAVED_GPRS
                } else {
                    &SC_ALL_GPRS_ORDER
                };
                order
                    .iter()
                    .copied()
                    .find(|&r| (candidate_mask & (1u16 << r)) != 0)
            } else {
                None
            };
            if let Some(reg) = chosen {
                self.free_gpr_mask &= !(1u16 << reg);
                if (SC_CALLEE_SAVED_MASK & (1u16 << reg)) != 0 {
                    self.used_callee_saved_mask |= 1u16 << reg;
                }
                if let Some(ex) = self.gpr_expire.get_mut(usize::from(reg)) {
                    *ex = lu_u32;
                }
                if lu_u32 < self.min_gpr_expire {
                    self.min_gpr_expire = lu_u32;
                }
                ScHome::Gpr(reg)
            } else {
                ScHome::Spill(self.alloc_spill(eff_last_use, scratch))
            }
        }
    }

    #[inline]
    fn try_reuse_gpr(
        &mut self,
        src_idx: usize,
        dst_idx: usize,
        dst_eff_lu: usize,
        dst_spans_barrier: bool,
        scratch: &ScUnitScratch,
    ) -> Option<ScHome> {
        let info = scratch.nodes.get(src_idx)?;
        if info.home_tag() != HOME_GPR {
            return None;
        }
        let r = info.home_data as u8;
        if scratch.eff_last_use(src_idx) != dst_idx {
            return None;
        }
        if dst_spans_barrier && (SC_CALLEE_SAVED_MASK & (1u16 << r)) == 0 {
            return None;
        }
        let lu_u32 = u32::try_from(dst_eff_lu).unwrap_or(u32::MAX);
        if let Some(ex) = self.gpr_expire.get_mut(usize::from(r)) {
            *ex = lu_u32;
        }
        if lu_u32 < self.min_gpr_expire {
            self.min_gpr_expire = lu_u32;
        }
        Some(ScHome::Gpr(r))
    }

    #[inline]
    fn try_reuse_xmm(
        &mut self,
        src_idx: usize,
        dst_idx: usize,
        dst_eff_lu: usize,
        dst_spans_barrier: bool,
        scratch: &ScUnitScratch,
    ) -> Option<ScHome> {
        if dst_spans_barrier {
            return None;
        }
        let info = scratch.nodes.get(src_idx)?;
        if info.home_tag() != HOME_XMM {
            return None;
        }
        let x = info.home_data as u8;
        if scratch.eff_last_use(src_idx) != dst_idx {
            return None;
        }
        let lu_u32 = u32::try_from(dst_eff_lu).unwrap_or(u32::MAX);
        if let Some(ex) = self.xmm_expire.get_mut(usize::from(x)) {
            *ex = lu_u32;
        }
        if lu_u32 < self.min_xmm_expire {
            self.min_xmm_expire = lu_u32;
        }
        Some(ScHome::Xmm(x))
    }

    #[inline]
    fn expire_at(&mut self, idx: usize, scratch: &mut ScUnitScratch) {
        let idx_u32 = u32::try_from(idx).unwrap_or(u32::MAX);
        if idx_u32 >= self.min_gpr_expire {
            let mut next_min = u32::MAX;
            for &r in &SC_ALL_GPRS_ORDER {
                if let Some(ex) = self.gpr_expire.get_mut(usize::from(r)) {
                    let v = *ex;
                    if v <= idx_u32 {
                        *ex = u32::MAX;
                        self.free_gpr_mask |= 1u16 << r;
                    } else if v < next_min {
                        next_min = v;
                    }
                }
            }
            self.min_gpr_expire = next_min;
        }
        if idx_u32 >= self.min_xmm_expire {
            let mut next_min = u32::MAX;
            for x in 2u8..16u8 {
                if let Some(ex) = self.xmm_expire.get_mut(usize::from(x)) {
                    let v = *ex;
                    if v <= idx_u32 {
                        *ex = u32::MAX;
                        self.free_xmm_mask |= 1u16 << x;
                    } else if v < next_min {
                        next_min = v;
                    }
                }
            }
            self.min_xmm_expire = next_min;
        }
        if idx_u32 >= self.min_spill_expire {
            let mut cur = scratch
                .spill_expire_head
                .get_mut(idx)
                .map_or(u32::MAX, |h| std::mem::replace(h, u32::MAX));
            while cur != u32::MAX {
                if let Some(&(slot, next)) = scratch.spill_entries.get(cur as usize) {
                    scratch.free_spill_slots.push(slot);
                    cur = next;
                } else {
                    break;
                }
            }
        }
    }
}

/// Detects a pure early-return base-case `If` guard at the entry of a `Func`
/// (`Func`, prelude of `LocalGet(0..arity)`/`ConstI64`/`Cmp*`/`MaskNot`, `If`,
/// body of `LocalGet(0..arity)`/`ConstI64` ending in `Return`, `End`) where no
/// node in `1..=guard_end` is referenced after `guard_end`.
fn detect_early_return_guard(
    unit_nodes: &[crate::bytecode::NodeFrame],
    scratch: &ScUnitScratch,
    sig: FuncSigInfo,
) -> Option<(usize, usize)> {
    let len = unit_nodes.len();
    if len <= 3 {
        return None;
    }
    let arity = usize::from(sig.arity);
    let mut if_idx = None;
    for i in 1..len.min(7) {
        let f = unit_nodes.get(i)?;
        match f.opcode {
            Opcode::LocalGet => {
                if (f.in_a as usize) >= arity || scratch.node_elem(i) == ElemType::F64 {
                    return None;
                }
            }
            Opcode::ConstI64 | Opcode::MaskNot => {}
            Opcode::CmpGt | Opcode::CmpLt | Opcode::CmpEq | Opcode::CmpLtU => {
                if scratch.node_elem(f.in_a as usize) == ElemType::F64 {
                    return None;
                }
            }
            Opcode::If => {
                if_idx = Some(i);
                break;
            }
            _ => return None,
        }
    }
    let if_idx = if_idx?;
    let if_info = scratch.nodes.get(if_idx)?;
    if (if_info.elem_flags & FLAG_SCOPE_HAS_RESULT) != 0 {
        return None;
    }
    let guard_end = if_info.aux as usize;
    if guard_end <= if_idx + 1 || guard_end >= len - 1 || guard_end > 14 {
        return None;
    }
    let (cmp_idx, _inv) = scratch.is_branch_fused(if_idx)?;
    let cmp_f = unit_nodes.get(cmp_idx)?;
    for op_idx in [cmp_f.in_a as usize, cmp_f.in_b as usize] {
        let op_f = unit_nodes.get(op_idx)?;
        match op_f.opcode {
            Opcode::ConstI64 => {}
            Opcode::LocalGet if (op_f.in_a as usize) < arity => {}
            _ => return None,
        }
    }
    let ret_idx = guard_end - 1;
    let ret_f = unit_nodes.get(ret_idx)?;
    if ret_f.opcode != Opcode::Return {
        return None;
    }
    for k in (if_idx + 1)..ret_idx {
        let kf = unit_nodes.get(k)?;
        match kf.opcode {
            Opcode::ConstI64 => {}
            Opcode::LocalGet
                if (kf.in_a as usize) < arity && scratch.node_elem(k) != ElemType::F64 => {}
            _ => return None,
        }
    }
    if ret_f.in_a != 0 {
        let rv = ret_f.in_a as usize;
        if rv == 0 || rv >= ret_idx {
            return None;
        }
        let rvf = unit_nodes.get(rv)?;
        match rvf.opcode {
            Opcode::ConstI64 => {}
            Opcode::LocalGet if (rvf.in_a as usize) < arity => {}
            _ => return None,
        }
    }
    for k in 1..=guard_end {
        if scratch.eff_last_use(k) > guard_end {
            return None;
        }
    }
    Some((if_idx, guard_end))
}

fn early_guard_arg_home(unit_nodes: &[crate::bytecode::NodeFrame], idx: usize) -> Option<ScHome> {
    let f = unit_nodes.get(idx)?;
    match f.opcode {
        Opcode::ConstI64 => Some(ScHome::ConstI64(f.decode_const_i64())),
        Opcode::LocalGet => SC_ARG_GPRS
            .get(f.in_a as usize)
            .copied()
            .map(ScHome::AliasGpr),
        _ => None,
    }
}

/// Emits pure-register macro-fused `sub reg, imm32(delta); js target; test reg32, 0x0020_0000; jnz target`.
/// When `short_patches` is `Some`, emits 2-byte `js rel8` / `jnz rel8` and pushes the patch offsets.
fn emit_sc_budget_sub_check(
    code: &mut Vec<u8>,
    reg: u8,
    delta: u32,
    stack_overflow_stub: usize,
    mut short_patches: Option<&mut Vec<usize>>,
) -> Result<(), AotError> {
    if reg == REG_RAX {
        // sub rax, imm32 (6B)
        code.extend_from_slice(&[0x48, 0x2D]);
        code.extend_from_slice(&delta.to_le_bytes());
    } else {
        // sub reg, imm32 (7B)
        code.extend_from_slice(&[sc_rex(true, 0, 0, reg), 0x81, 0xE8 | (reg & 7)]);
        code.extend_from_slice(&delta.to_le_bytes());
    }
    if let Some(patches) = short_patches.as_deref_mut() {
        code.extend_from_slice(&[0x78, 0x00]); // js rel8
        patches.push(code.len() - 1);
    } else {
        // js rel32 (macro-fused with `sub`)
        code.extend_from_slice(&[0x0F, 0x88, 0x00, 0x00, 0x00, 0x00]);
        let p1 = code.len() - 4;
        patch_rel32(code, p1, stack_overflow_stub)?;
    }
    if reg == REG_RAX {
        // test eax, 0x0020_0000 (5B)
        code.extend_from_slice(&[0xA9, 0x00, 0x00, 0x20, 0x00]);
    } else {
        // test reg32, 0x0020_0000 (7B)
        code.extend_from_slice(&[
            sc_rex(false, 0, 0, reg),
            0xF7,
            0xC0 | (reg & 7),
            0x00,
            0x00,
            0x20,
            0x00,
        ]);
    }
    if let Some(patches) = short_patches {
        code.extend_from_slice(&[0x75, 0x00]); // jnz rel8
        patches.push(code.len() - 1);
        Ok(())
    } else {
        // jnz rel32 (macro-fused with `test`)
        code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]);
        let p2 = code.len() - 4;
        patch_rel32(code, p2, stack_overflow_stub)
    }
}

/// Returns true when an integer-class node that would otherwise spill to the stack
/// can stay in scratch `REG_RAX` because its only use is the immediately following
/// commutative integer binary op whose other operand is already in a non-`rax` GPR.
#[inline]
fn can_use_transient_rax(
    unit_nodes: &[crate::bytecode::NodeFrame],
    scratch: &ScUnitScratch,
    idx: usize,
    elem: ElemType,
    uc: u8,
    eff_lu: usize,
    spans_bar: bool,
) -> bool {
    if elem == ElemType::F64 || spans_bar || uc != 1 || eff_lu != idx.saturating_add(1) {
        return false;
    }
    let next_idx = idx.saturating_add(1);
    let Some(next_f) = unit_nodes.get(next_idx) else {
        return false;
    };
    if !matches!(
        next_f.opcode,
        Opcode::Add
            | Opcode::Mul
            | Opcode::BitAnd
            | Opcode::BitOr
            | Opcode::BitXor
            | Opcode::MaskAnd
            | Opcode::MaskOr
    ) || scratch.node_elem(next_idx) == ElemType::F64
    {
        return false;
    }
    let other_idx = if next_f.in_a as usize == idx {
        next_f.in_b as usize
    } else if next_f.in_b as usize == idx {
        next_f.in_a as usize
    } else {
        return false;
    };
    if other_idx == idx {
        return false;
    }
    scratch
        .get_home(other_idx, unit_nodes)
        .gpr_reg()
        .is_some_and(|r| r != REG_RAX)
}

#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "single-pass O(N) structured x86_64 register-allocating unit emitter"
)]
fn sc_emit_unit(
    code: &mut Vec<u8>,
    program: &RulProgram,
    unit_nodes: &[crate::bytecode::NodeFrame],
    regions: &[Option<AotRegionDecl>; MAX_REGION_SLOTS],
    func_sigs: &[Option<FuncSigInfo>],
    cur_sig: Option<FuncSigInfo>,
    output_region_slot: Option<usize>,
    stack_overflow_stub: Option<usize>,
    call_patches: &mut Vec<(usize, usize)>,
    scratch: &mut ScUnitScratch,
) -> Result<(), AotError> {
    let func_mode = cur_sig.is_some();
    let len = unit_nodes.len();
    let output_node_opt = if func_mode {
        None
    } else {
        Some(program.output_node as usize)
    };

    let ScUnitMeta {
        local_elem,
        local_used,
        local_score,
        has_x87,
    } = sc_analyze_unit(
        unit_nodes,
        regions,
        func_sigs,
        cur_sig,
        output_node_opt,
        scratch,
    )?;

    let has_any_barrier = scratch.has_any_barrier;

    // Assign homes to hot locals (pinned registers first, ties broken by slot number).
    let mut gpr_candidates: Vec<(u64, usize)> = Vec::new();
    let mut xmm_candidates: Vec<(u64, usize)> = Vec::new();
    for slot in 0..64usize {
        if !local_used.get(slot).copied().unwrap_or(false) {
            continue;
        }
        let score = local_score.get(slot).copied().unwrap_or(0);
        if local_elem.get(slot).copied() == Some(ElemType::F64) {
            xmm_candidates.push((score, slot));
        } else {
            gpr_candidates.push((score, slot));
        }
    }
    gpr_candidates.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    xmm_candidates.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

    let mut local_home: [ScLocalHome; 64] = [ScLocalHome::Unused; 64];
    let mut free_gpr_mask = SC_ALLOC_GPR_MASK;
    let mut free_xmm_mask = SC_ALLOC_XMM_MASK;
    let mut used_callee_saved_mask: u16 = 0;

    let max_pinned_gpr = if has_any_barrier { 5usize } else { 7usize };
    let winner_count = gpr_candidates.len().min(max_pinned_gpr);
    let gpr_winners = gpr_candidates.get(..winner_count).unwrap_or(&[]);

    // In leaf functions, if an incoming param `slot < arity` earned a top-scored pinned spot,
    // assign it to its incoming arg register `SC_ARG_GPRS[slot]` to eliminate a prologue move.
    if !has_any_barrier {
        if let Some(sig) = cur_sig {
            let arity = usize::from(sig.arity);
            for &(_score, slot) in gpr_winners {
                if slot < arity {
                    if let Some(&arg_reg) = SC_ARG_GPRS.get(slot) {
                        if (free_gpr_mask & (1u16 << arg_reg)) != 0 {
                            free_gpr_mask &= !(1u16 << arg_reg);
                            if let Some(lh) = local_home.get_mut(slot) {
                                *lh = ScLocalHome::Gpr(arg_reg);
                            }
                        }
                    }
                }
            }
        }
    }

    let gpr_pool: &[u8] = if has_any_barrier {
        &SC_CALLEE_SAVED_GPRS
    } else {
        &SC_ALL_GPRS_ORDER
    };
    for &(_score, slot) in gpr_winners {
        if local_home.get(slot).copied() != Some(ScLocalHome::Unused) {
            continue;
        }
        if let Some(reg) = gpr_pool
            .iter()
            .copied()
            .find(|&r| (free_gpr_mask & (1u16 << r)) != 0)
        {
            free_gpr_mask &= !(1u16 << reg);
            if (SC_CALLEE_SAVED_MASK & (1u16 << reg)) != 0 {
                used_callee_saved_mask |= 1u16 << reg;
            }
            if let Some(lh) = local_home.get_mut(slot) {
                *lh = ScLocalHome::Gpr(reg);
            }
        }
    }

    if !has_any_barrier {
        for (pinned_xmm_count, &(_score, slot)) in xmm_candidates.iter().enumerate() {
            if pinned_xmm_count >= 10 || free_xmm_mask == 0 {
                break;
            }
            let xreg = u8::try_from(free_xmm_mask.trailing_zeros()).unwrap_or(2);
            free_xmm_mask &= !(1u16 << xreg);
            if let Some(lh) = local_home.get_mut(slot) {
                *lh = ScLocalHome::Xmm(xreg);
            }
        }
    }

    // Assign frame slots to all remaining used locals, optional x87 scratch, and non-func stdout staging.
    let mut fixed_slots: u32 = 0;
    for slot in 0..64usize {
        if local_used.get(slot).copied().unwrap_or(false)
            && local_home.get(slot).copied() == Some(ScLocalHome::Unused)
        {
            let disp =
                (-8i32).saturating_mul(i32::try_from(fixed_slots.saturating_add(1)).unwrap_or(1));
            if let Some(lh) = local_home.get_mut(slot) {
                *lh = ScLocalHome::Slot(disp);
            }
            fixed_slots = fixed_slots.saturating_add(1);
        }
    }
    let x87_scratch_disp = if has_x87 {
        let disp =
            (-8i32).saturating_mul(i32::try_from(fixed_slots.saturating_add(1)).unwrap_or(1));
        fixed_slots = fixed_slots.saturating_add(1);
        disp
    } else {
        0
    };
    let stdout_disp = if !func_mode && output_region_slot.is_none() {
        let disp =
            (-8i32).saturating_mul(i32::try_from(fixed_slots.saturating_add(1)).unwrap_or(1));
        fixed_slots = fixed_slots.saturating_add(1);
        disp
    } else {
        0
    };
    let spill_base_slot = fixed_slots;

    let mut alloc = ScRegAllocState::new(free_gpr_mask, free_xmm_mask, used_callee_saved_mask);

    let mut top_frame_sub_patch: Option<usize> = None;
    let fn_frame_units = func_frame_units(len);
    let budget_delta_u32: u32 = (1u32 << 22) | fn_frame_units;
    let early_guard = cur_sig.and_then(|sig| detect_early_return_guard(unit_nodes, scratch, sig));
    let start_idx: usize = early_guard.map_or(0, |(_if_idx, guard_end)| guard_end + 1);

    if !func_mode {
        // Top-level structured program prologue.
        code.extend_from_slice(&[0x48, 0x83, 0xE4, 0xE0]); // and rsp, -32
        code.extend_from_slice(&[0x48, 0x89, 0xE5]); // mov rbp, rsp
        code.extend_from_slice(&[0x48, 0x81, 0xEC, 0x00, 0x00, 0x00, 0x00]); // sub rsp, imm32
        top_frame_sub_patch = Some(code.len() - 4);
        if regions.iter().any(Option::is_some) {
            emit_sc_region_startup(code, regions, false)?;
        }
    }

    let mut fn_body_buf = std::mem::take(&mut scratch.fn_body);
    fn_body_buf.clear();
    let body_call_patch_start = call_patches.len();
    let mut pending_call_args: Option<(usize, Option<usize>)> = None;

    {
        let out: &mut Vec<u8> = if func_mode { &mut fn_body_buf } else { code };

        // Helper: check if `cur_idx` can write directly into the pinned register of an immediately following `LocalSet`.
        let direct_local_dst = |unit_nodes: &[crate::bytecode::NodeFrame],
                                scratch: &ScUnitScratch,
                                local_home: &[ScLocalHome; 64],
                                cur_idx: usize|
         -> Option<ScHome> {
            if scratch.nodes.get(cur_idx).map_or(0, |n| n.use_count()) != 1 {
                return None;
            }
            let next_idx = cur_idx.saturating_add(1);
            let next_f = unit_nodes.get(next_idx)?;
            if next_f.opcode != Opcode::LocalSet || next_f.in_b as usize != cur_idx {
                return None;
            }
            if scratch.nodes.get(next_idx).map_or(0, |n| n.use_count()) != 0 {
                return None;
            }
            match local_home.get(next_f.in_a as usize).copied()? {
                ScLocalHome::Gpr(r) => Some(ScHome::AliasGpr(r)),
                ScLocalHome::Xmm(x) => Some(ScHome::AliasXmm(x)),
                ScLocalHome::Slot(_) | ScLocalHome::Unused => None,
            }
        };

        // Helper: check if an integer node `cur_idx` is immediately returned by `Return` or `Func End`.
        let is_immediate_func_return = |cur_idx: usize| -> bool {
            if !func_mode {
                return false;
            }
            let next_idx = cur_idx.saturating_add(1);
            let Some(next_f) = unit_nodes.get(next_idx) else {
                return false;
            };
            (next_f.opcode == Opcode::Return && next_f.in_a as usize == cur_idx)
                || (next_idx.saturating_add(1) == len
                    && next_f.opcode == Opcode::End
                    && next_f.in_b != 0
                    && next_f.in_a as usize == cur_idx)
        };

        for (idx, frame) in unit_nodes.iter().enumerate().skip(start_idx) {
            let a_idx = frame.in_a as usize;
            let b_idx = frame.in_b as usize;
            let home_a = scratch.get_home(a_idx, unit_nodes);
            let home_b = scratch.get_home(b_idx, unit_nodes);
            let uc = scratch.nodes.get(idx).map_or(0, |n| n.use_count());
            let eff_lu = scratch.eff_last_use(idx);
            let spans_bar = scratch.spans_barrier(idx, eff_lu);

            match frame.opcode {
                Opcode::Func => {}
                Opcode::ConstF64 => {
                    scratch.set_home(idx, ScHome::ConstF64(0))?;
                }
                Opcode::ConstI64 => {
                    scratch.set_home(idx, ScHome::ConstI64(0))?;
                }
                Opcode::LocalGet => {
                    if uc > 0 {
                        let slot = frame.in_a as usize;
                        let nls = scratch.nodes.get(idx).map_or(u32::MAX, |n| n.aux) as usize;
                        let can_alias = eff_lu < nls;
                        let lh = local_home.get(slot).copied().unwrap_or(ScLocalHome::Unused);
                        let dst = match lh {
                            ScLocalHome::Gpr(r) if can_alias => ScHome::AliasGpr(r),
                            ScLocalHome::Xmm(x) if can_alias => ScHome::AliasXmm(x),
                            ScLocalHome::Gpr(r) => {
                                let h = alloc.alloc_home(
                                    idx,
                                    scratch.node_elem(idx),
                                    eff_lu,
                                    spans_bar,
                                    scratch,
                                );
                                emit_store_gpr_into_home(out, h, r, spill_base_slot);
                                h
                            }
                            ScLocalHome::Xmm(x) => {
                                let h = alloc.alloc_home(
                                    idx,
                                    ElemType::F64,
                                    eff_lu,
                                    spans_bar,
                                    scratch,
                                );
                                emit_store_xmm_into_home(out, h, x, spill_base_slot);
                                h
                            }
                            ScLocalHome::Slot(disp) => {
                                let el = scratch.node_elem(idx);
                                let h = if alloc.free_gpr_mask == 0
                                    && can_use_transient_rax(
                                        unit_nodes, scratch, idx, el, uc, eff_lu, spans_bar,
                                    ) {
                                    ScHome::AliasGpr(REG_RAX)
                                } else {
                                    alloc.alloc_home(idx, el, eff_lu, spans_bar, scratch)
                                };
                                match h {
                                    ScHome::Gpr(r) | ScHome::AliasGpr(r) => {
                                        emit_load_gpr_rbp(out, r, disp);
                                    }
                                    ScHome::Xmm(x) | ScHome::AliasXmm(x) => {
                                        emit_load_xmm_rbp(out, x, disp);
                                    }
                                    ScHome::Spill(s) => {
                                        emit_load_gpr_rbp(out, REG_RAX, disp);
                                        emit_store_rbp_gpr(
                                            out,
                                            sc_spill_disp(spill_base_slot, s),
                                            REG_RAX,
                                        );
                                    }
                                    _ => {}
                                }
                                h
                            }
                            ScLocalHome::Unused => ScHome::None,
                        };
                        scratch.set_home(idx, dst)?;
                    }
                }
                Opcode::LocalSet => {
                    let slot = frame.in_a as usize;
                    let lh = local_home.get(slot).copied().unwrap_or(ScLocalHome::Unused);
                    match lh {
                        ScLocalHome::Gpr(r) => {
                            if home_b.gpr_reg() != Some(r) {
                                emit_load_home_into_gpr(out, r, home_b, spill_base_slot);
                            }
                        }
                        ScLocalHome::Xmm(x) => {
                            if home_b.xmm_reg() != Some(x) {
                                emit_load_home_into_xmm(out, x, home_b, spill_base_slot);
                            }
                        }
                        ScLocalHome::Slot(disp) => match home_b {
                            ScHome::Gpr(r) | ScHome::AliasGpr(r) => {
                                emit_store_rbp_gpr(out, disp, r);
                            }
                            ScHome::Xmm(x) | ScHome::AliasXmm(x) => {
                                emit_store_rbp_xmm(out, disp, x);
                            }
                            _ => {
                                emit_load_home_into_gpr(out, REG_RAX, home_b, spill_base_slot);
                                emit_store_rbp_gpr(out, disp, REG_RAX);
                            }
                        },
                        ScLocalHome::Unused => {}
                    }
                    if uc > 0 {
                        let el = scratch.node_elem(idx);
                        let dst = alloc.alloc_home(idx, el, eff_lu, spans_bar, scratch);
                        emit_move_home_to_home(out, dst, home_b, spill_base_slot);
                        scratch.set_home(idx, dst)?;
                    }
                }
                Opcode::Add | Opcode::Sub | Opcode::Mul => {
                    if uc > 0 {
                        let elem_a = scratch.node_elem(a_idx);
                        if elem_a == ElemType::F64 {
                            let dst = direct_local_dst(unit_nodes, scratch, &local_home, idx)
                                .or_else(|| {
                                    alloc.try_reuse_xmm(a_idx, idx, eff_lu, spans_bar, scratch)
                                })
                                .or_else(|| {
                                    if frame.opcode != Opcode::Sub {
                                        alloc.try_reuse_xmm(b_idx, idx, eff_lu, spans_bar, scratch)
                                    } else {
                                        None
                                    }
                                })
                                .unwrap_or_else(|| {
                                    alloc.alloc_home(idx, ElemType::F64, eff_lu, spans_bar, scratch)
                                });
                            let op_byte = match frame.opcode {
                                Opcode::Add => 0x58,
                                Opcode::Sub => 0x5C,
                                _ => 0x59,
                            };
                            let (mut ha, mut hb) = (home_a, home_b);
                            let mut work_x = dst.xmm_reg().unwrap_or(0);
                            if frame.opcode != Opcode::Sub
                                && hb.xmm_reg() == Some(work_x)
                                && ha.xmm_reg() != Some(work_x)
                            {
                                std::mem::swap(&mut ha, &mut hb);
                            }
                            if frame.opcode == Opcode::Sub
                                && hb.xmm_reg() == Some(work_x)
                                && ha.xmm_reg() != Some(work_x)
                            {
                                work_x = 0;
                            }
                            emit_load_home_into_xmm(out, work_x, ha, spill_base_slot);
                            match hb {
                                ScHome::Xmm(x) | ScHome::AliasXmm(x) => {
                                    emit_sse2_sd_rr(out, op_byte, work_x, x);
                                }
                                ScHome::Spill(s) => {
                                    emit_sse2_sd_rm_rbp(
                                        out,
                                        op_byte,
                                        work_x,
                                        sc_spill_disp(spill_base_slot, s),
                                    );
                                }
                                _ => {
                                    emit_load_home_into_xmm(out, 1, hb, spill_base_slot);
                                    emit_sse2_sd_rr(out, op_byte, work_x, 1);
                                }
                            }
                            emit_store_xmm_into_home(out, dst, work_x, spill_base_slot);
                            scratch.set_home(idx, dst)?;
                        } else {
                            let direct_ret = if uc == 1
                                && is_immediate_func_return(idx)
                                && !(frame.opcode == Opcode::Sub
                                    && home_b.gpr_reg() == Some(REG_RAX))
                            {
                                Some(ScHome::AliasGpr(REG_RAX))
                            } else {
                                None
                            };
                            let dst = direct_ret
                                .or_else(|| direct_local_dst(unit_nodes, scratch, &local_home, idx))
                                .or_else(|| {
                                    alloc.try_reuse_gpr(a_idx, idx, eff_lu, spans_bar, scratch)
                                })
                                .or_else(|| {
                                    if frame.opcode != Opcode::Sub {
                                        alloc.try_reuse_gpr(b_idx, idx, eff_lu, spans_bar, scratch)
                                    } else {
                                        None
                                    }
                                })
                                .unwrap_or_else(|| {
                                    if alloc.free_gpr_mask == 0
                                        && can_use_transient_rax(
                                            unit_nodes, scratch, idx, elem_a, uc, eff_lu, spans_bar,
                                        )
                                    {
                                        ScHome::AliasGpr(REG_RAX)
                                    } else {
                                        alloc.alloc_home(idx, elem_a, eff_lu, spans_bar, scratch)
                                    }
                                });
                            emit_sc_int_binary(
                                out,
                                frame.opcode,
                                elem_a,
                                home_a,
                                home_b,
                                dst,
                                spill_base_slot,
                            );
                            scratch.set_home(idx, dst)?;
                        }
                    }
                }
                Opcode::Div => {
                    let elem_a = scratch.node_elem(a_idx);
                    if matches!(elem_a, ElemType::I64 | ElemType::U8) {
                        // Integer division can trap (`exit(72)`), so always emit even if `uc == 0`.
                        let dst = if uc > 0 {
                            direct_local_dst(unit_nodes, scratch, &local_home, idx).unwrap_or_else(
                                || alloc.alloc_home(idx, elem_a, eff_lu, spans_bar, scratch),
                            )
                        } else {
                            ScHome::None
                        };
                        let rdx_live = (alloc.free_gpr_mask & (1u16 << REG_RDX)) == 0;
                        emit_sc_int_div_rem(
                            out,
                            elem_a == ElemType::U8,
                            false,
                            home_a,
                            home_b,
                            dst,
                            rdx_live,
                            spill_base_slot,
                        )?;
                        scratch.set_home(idx, dst)?;
                    } else if uc > 0 {
                        let dst = direct_local_dst(unit_nodes, scratch, &local_home, idx)
                            .or_else(|| alloc.try_reuse_xmm(a_idx, idx, eff_lu, spans_bar, scratch))
                            .unwrap_or_else(|| {
                                alloc.alloc_home(idx, ElemType::F64, eff_lu, spans_bar, scratch)
                            });
                        let mut work_x = dst.xmm_reg().unwrap_or(0);
                        if home_b.xmm_reg() == Some(work_x) && home_a.xmm_reg() != Some(work_x) {
                            work_x = 0;
                        }
                        emit_load_home_into_xmm(out, work_x, home_a, spill_base_slot);
                        match home_b {
                            ScHome::Xmm(x) | ScHome::AliasXmm(x) => {
                                emit_sse2_sd_rr(out, 0x5E, work_x, x);
                            }
                            ScHome::Spill(s) => {
                                emit_sse2_sd_rm_rbp(
                                    out,
                                    0x5E,
                                    work_x,
                                    sc_spill_disp(spill_base_slot, s),
                                );
                            }
                            _ => {
                                emit_load_home_into_xmm(out, 1, home_b, spill_base_slot);
                                emit_sse2_sd_rr(out, 0x5E, work_x, 1);
                            }
                        }
                        emit_store_xmm_into_home(out, dst, work_x, spill_base_slot);
                        scratch.set_home(idx, dst)?;
                    }
                }
                Opcode::Rem => {
                    let elem_a = scratch.node_elem(a_idx);
                    let dst = if uc > 0 {
                        direct_local_dst(unit_nodes, scratch, &local_home, idx).unwrap_or_else(
                            || alloc.alloc_home(idx, elem_a, eff_lu, spans_bar, scratch),
                        )
                    } else {
                        ScHome::None
                    };
                    let rdx_live = (alloc.free_gpr_mask & (1u16 << REG_RDX)) == 0;
                    emit_sc_int_div_rem(
                        out,
                        elem_a == ElemType::U8,
                        true,
                        home_a,
                        home_b,
                        dst,
                        rdx_live,
                        spill_base_slot,
                    )?;
                    scratch.set_home(idx, dst)?;
                }
                Opcode::BitAnd
                | Opcode::BitOr
                | Opcode::BitXor
                | Opcode::MaskAnd
                | Opcode::MaskOr => {
                    if uc > 0 {
                        let elem = scratch.node_elem(idx);
                        let direct_ret = if uc == 1 && is_immediate_func_return(idx) {
                            Some(ScHome::AliasGpr(REG_RAX))
                        } else {
                            None
                        };
                        let dst = direct_ret
                            .or_else(|| direct_local_dst(unit_nodes, scratch, &local_home, idx))
                            .or_else(|| alloc.try_reuse_gpr(a_idx, idx, eff_lu, spans_bar, scratch))
                            .or_else(|| alloc.try_reuse_gpr(b_idx, idx, eff_lu, spans_bar, scratch))
                            .unwrap_or_else(|| {
                                if alloc.free_gpr_mask == 0
                                    && can_use_transient_rax(
                                        unit_nodes, scratch, idx, elem, uc, eff_lu, spans_bar,
                                    )
                                {
                                    ScHome::AliasGpr(REG_RAX)
                                } else {
                                    alloc.alloc_home(idx, elem, eff_lu, spans_bar, scratch)
                                }
                            });
                        emit_sc_int_binary(
                            out,
                            frame.opcode,
                            elem,
                            home_a,
                            home_b,
                            dst,
                            spill_base_slot,
                        );
                        scratch.set_home(idx, dst)?;
                    }
                }
                Opcode::Shl | Opcode::ShrU | Opcode::ShrS => {
                    if uc > 0 {
                        let elem_a = scratch.node_elem(a_idx);
                        let dst = direct_local_dst(unit_nodes, scratch, &local_home, idx)
                            .or_else(|| alloc.try_reuse_gpr(a_idx, idx, eff_lu, spans_bar, scratch))
                            .unwrap_or_else(|| {
                                if alloc.free_gpr_mask == 0
                                    && can_use_transient_rax(
                                        unit_nodes, scratch, idx, elem_a, uc, eff_lu, spans_bar,
                                    )
                                {
                                    ScHome::AliasGpr(REG_RAX)
                                } else {
                                    alloc.alloc_home(idx, elem_a, eff_lu, spans_bar, scratch)
                                }
                            });
                        let rcx_live = (alloc.free_gpr_mask & (1u16 << REG_RCX)) == 0;
                        emit_sc_shift(
                            out,
                            frame.opcode,
                            elem_a,
                            home_a,
                            home_b,
                            dst,
                            rcx_live,
                            spill_base_slot,
                        );
                        scratch.set_home(idx, dst)?;
                    }
                }
                Opcode::MaskNot => {
                    if uc > 0 && !scratch.is_node_fused_out(idx) {
                        let dst = direct_local_dst(unit_nodes, scratch, &local_home, idx)
                            .or_else(|| alloc.try_reuse_gpr(a_idx, idx, eff_lu, spans_bar, scratch))
                            .unwrap_or_else(|| {
                                alloc.alloc_home(
                                    idx,
                                    ElemType::BoolMask,
                                    eff_lu,
                                    spans_bar,
                                    scratch,
                                )
                            });
                        let dst_r = dst.gpr_reg().unwrap_or(REG_RAX);
                        emit_load_home_into_gpr(out, dst_r, home_a, spill_base_slot);
                        emit_alu32_ri(out, 6, dst_r, 1);
                        emit_store_gpr_into_home(out, dst, dst_r, spill_base_slot);
                        scratch.set_home(idx, dst)?;
                    }
                }
                Opcode::Cast => {
                    let src_elem = scratch.node_elem(a_idx);
                    let dst_elem = frame.type_lane.elem;
                    // Fold `Cast(ConstI64(v), U8)` directly into `ScHome::ConstU8` (zero instructions).
                    if src_elem == ElemType::I64 && dst_elem == ElemType::U8 {
                        if let ScHome::ConstI64(v) = home_a {
                            scratch.set_home(
                                idx,
                                ScHome::ConstU8(u8::try_from(v & 0xFF).unwrap_or(0)),
                            )?;
                            alloc.expire_at(idx, scratch);
                            continue;
                        }
                    }
                    let can_trap = src_elem == ElemType::F64
                        && matches!(dst_elem, ElemType::I64 | ElemType::U8);
                    if uc > 0 || can_trap {
                        let dst = if uc > 0 {
                            direct_local_dst(unit_nodes, scratch, &local_home, idx)
                                .or_else(|| {
                                    if dst_elem != ElemType::F64 && src_elem != ElemType::F64 {
                                        alloc.try_reuse_gpr(a_idx, idx, eff_lu, spans_bar, scratch)
                                    } else {
                                        None
                                    }
                                })
                                .unwrap_or_else(|| {
                                    alloc.alloc_home(idx, dst_elem, eff_lu, spans_bar, scratch)
                                })
                        } else {
                            ScHome::None
                        };
                        emit_sc_cast(out, src_elem, dst_elem, home_a, dst, spill_base_slot)?;
                        scratch.set_home(idx, dst)?;
                    }
                }
                Opcode::Min | Opcode::Max => {
                    if uc > 0 {
                        let dst = direct_local_dst(unit_nodes, scratch, &local_home, idx)
                            .or_else(|| alloc.try_reuse_xmm(a_idx, idx, eff_lu, spans_bar, scratch))
                            .unwrap_or_else(|| {
                                alloc.alloc_home(idx, ElemType::F64, eff_lu, spans_bar, scratch)
                            });
                        let op_byte = if frame.opcode == Opcode::Min {
                            0x5D
                        } else {
                            0x5F
                        };
                        let mut work_x = dst.xmm_reg().unwrap_or(0);
                        if home_b.xmm_reg() == Some(work_x) && home_a.xmm_reg() != Some(work_x) {
                            work_x = 0;
                        }
                        emit_load_home_into_xmm(out, work_x, home_a, spill_base_slot);
                        match home_b {
                            ScHome::Xmm(x) | ScHome::AliasXmm(x) => {
                                emit_sse2_sd_rr(out, op_byte, work_x, x);
                            }
                            ScHome::Spill(s) => {
                                emit_sse2_sd_rm_rbp(
                                    out,
                                    op_byte,
                                    work_x,
                                    sc_spill_disp(spill_base_slot, s),
                                );
                            }
                            _ => {
                                emit_load_home_into_xmm(out, 1, home_b, spill_base_slot);
                                emit_sse2_sd_rr(out, op_byte, work_x, 1);
                            }
                        }
                        emit_store_xmm_into_home(out, dst, work_x, spill_base_slot);
                        scratch.set_home(idx, dst)?;
                    }
                }
                Opcode::Fma => {
                    if uc > 0 {
                        let (in_b, in_c) = frame.decode_fma_bc();
                        let hb_fma = scratch.get_home(in_b as usize, unit_nodes);
                        let hc_fma = scratch.get_home(in_c as usize, unit_nodes);
                        let dst = direct_local_dst(unit_nodes, scratch, &local_home, idx)
                            .or_else(|| alloc.try_reuse_xmm(a_idx, idx, eff_lu, spans_bar, scratch))
                            .unwrap_or_else(|| {
                                alloc.alloc_home(idx, ElemType::F64, eff_lu, spans_bar, scratch)
                            });
                        let mut work_x = dst.xmm_reg().unwrap_or(0);
                        if (hb_fma.xmm_reg() == Some(work_x) || hc_fma.xmm_reg() == Some(work_x))
                            && home_a.xmm_reg() != Some(work_x)
                        {
                            work_x = 0;
                        }
                        emit_load_home_into_xmm(out, work_x, home_a, spill_base_slot);
                        let reg_b = if let Some(xb) = hb_fma.xmm_reg() {
                            xb
                        } else {
                            emit_load_home_into_xmm(out, 1, hb_fma, spill_base_slot);
                            1
                        };
                        match hc_fma {
                            ScHome::Xmm(xc) | ScHome::AliasXmm(xc) => {
                                emit_vex_fma213_xmm_rr(out, work_x, reg_b, xc);
                            }
                            ScHome::Spill(s) => {
                                emit_vex_fma213_xmm_rm_rbp(
                                    out,
                                    work_x,
                                    reg_b,
                                    sc_spill_disp(spill_base_slot, s),
                                );
                            }
                            _ => {
                                if reg_b == 1 {
                                    emit_load_home_into_xmm(out, 0, home_a, spill_base_slot);
                                    emit_load_home_into_xmm(out, 1, hb_fma, spill_base_slot);
                                    if let ScHome::ConstF64(bits) = hc_fma {
                                        emit_mov_gpr_imm64(out, REG_RAX, bits);
                                    } else {
                                        emit_load_home_into_gpr(
                                            out,
                                            REG_RAX,
                                            hc_fma,
                                            spill_base_slot,
                                        );
                                    }
                                    emit_push_gpr(out, REG_RAX);
                                    // vfmadd213sd xmm0, xmm1, [rsp]
                                    out.extend_from_slice(&[0xC4, 0xE2, 0xF1, 0xA9, 0x04, 0x24]);
                                    out.extend_from_slice(&[0x48, 0x83, 0xC4, 0x08]); // add rsp, 8
                                    work_x = 0;
                                } else {
                                    emit_load_home_into_xmm(out, 1, hc_fma, spill_base_slot);
                                    emit_vex_fma213_xmm_rr(out, work_x, reg_b, 1);
                                }
                            }
                        }
                        emit_store_xmm_into_home(out, dst, work_x, spill_base_slot);
                        scratch.set_home(idx, dst)?;
                    }
                }
                Opcode::Abs => {
                    if uc > 0 {
                        let dst = direct_local_dst(unit_nodes, scratch, &local_home, idx)
                            .or_else(|| alloc.try_reuse_xmm(a_idx, idx, eff_lu, spans_bar, scratch))
                            .unwrap_or_else(|| {
                                alloc.alloc_home(idx, ElemType::F64, eff_lu, spans_bar, scratch)
                            });
                        emit_load_home_into_gpr(out, REG_RAX, home_a, spill_base_slot);
                        out.extend_from_slice(&[0x48, 0x0F, 0xBA, 0xF0, 0x3F]); // btr rax, 63
                        emit_store_gpr_into_home(out, dst, REG_RAX, spill_base_slot);
                        scratch.set_home(idx, dst)?;
                    }
                }
                Opcode::Sqrt => {
                    if uc > 0 {
                        let dst = direct_local_dst(unit_nodes, scratch, &local_home, idx)
                            .or_else(|| alloc.try_reuse_xmm(a_idx, idx, eff_lu, spans_bar, scratch))
                            .unwrap_or_else(|| {
                                alloc.alloc_home(idx, ElemType::F64, eff_lu, spans_bar, scratch)
                            });
                        let dst_x = dst.xmm_reg().unwrap_or(0);
                        match home_a {
                            ScHome::Xmm(xa) | ScHome::AliasXmm(xa) => {
                                emit_sse2_sd_rr(out, 0x51, dst_x, xa);
                            }
                            ScHome::Spill(s) => {
                                emit_sse2_sd_rm_rbp(
                                    out,
                                    0x51,
                                    dst_x,
                                    sc_spill_disp(spill_base_slot, s),
                                );
                            }
                            _ => {
                                emit_load_home_into_xmm(out, dst_x, home_a, spill_base_slot);
                                emit_sse2_sd_rr(out, 0x51, dst_x, dst_x);
                            }
                        }
                        emit_store_xmm_into_home(out, dst, dst_x, spill_base_slot);
                        scratch.set_home(idx, dst)?;
                    }
                }
                Opcode::Floor | Opcode::Ceil => {
                    if uc > 0 {
                        let dst = direct_local_dst(unit_nodes, scratch, &local_home, idx)
                            .or_else(|| alloc.try_reuse_xmm(a_idx, idx, eff_lu, spans_bar, scratch))
                            .unwrap_or_else(|| {
                                alloc.alloc_home(idx, ElemType::F64, eff_lu, spans_bar, scratch)
                            });
                        let mode = if frame.opcode == Opcode::Floor {
                            0x09
                        } else {
                            0x0A
                        };
                        let dst_x = dst.xmm_reg().unwrap_or(0);
                        if let Some(xa) = home_a.xmm_reg() {
                            emit_roundsd_rr(out, dst_x, xa, mode);
                        } else {
                            emit_load_home_into_xmm(out, dst_x, home_a, spill_base_slot);
                            emit_roundsd_rr(out, dst_x, dst_x, mode);
                        }
                        emit_store_xmm_into_home(out, dst, dst_x, spill_base_slot);
                        scratch.set_home(idx, dst)?;
                    }
                }
                Opcode::Ln | Opcode::Exp => {
                    if uc > 0 {
                        let dst = direct_local_dst(unit_nodes, scratch, &local_home, idx)
                            .or_else(|| alloc.try_reuse_xmm(a_idx, idx, eff_lu, spans_bar, scratch))
                            .unwrap_or_else(|| {
                                alloc.alloc_home(idx, ElemType::F64, eff_lu, spans_bar, scratch)
                            });
                        let src_disp = if let ScHome::Spill(s) = home_a {
                            sc_spill_disp(spill_base_slot, s)
                        } else {
                            emit_load_home_into_xmm(out, 0, home_a, spill_base_slot);
                            emit_store_rbp_xmm(out, x87_scratch_disp, 0);
                            x87_scratch_disp
                        };
                        let dst_disp = if let ScHome::Spill(s) = dst {
                            sc_spill_disp(spill_base_slot, s)
                        } else {
                            x87_scratch_disp
                        };
                        if frame.opcode == Opcode::Ln {
                            emit_x87_ln_slot(out, src_disp, dst_disp);
                        } else {
                            emit_x87_exp_slot(out, src_disp, dst_disp);
                        }
                        if let Some(dx) = dst.xmm_reg() {
                            emit_load_xmm_rbp(out, dx, x87_scratch_disp);
                        }
                        scratch.set_home(idx, dst)?;
                    }
                }
                Opcode::CmpGt | Opcode::CmpLt | Opcode::CmpEq | Opcode::CmpLtU => {
                    if uc > 0 && !scratch.is_node_fused_out(idx) {
                        let elem_a = scratch.node_elem(a_idx);
                        let dst = direct_local_dst(unit_nodes, scratch, &local_home, idx)
                            .unwrap_or_else(|| {
                                alloc.alloc_home(
                                    idx,
                                    ElemType::BoolMask,
                                    eff_lu,
                                    spans_bar,
                                    scratch,
                                )
                            });
                        let dst_r = dst.gpr_reg().unwrap_or(REG_RAX);
                        if elem_a == ElemType::I64 || elem_a == ElemType::U8 {
                            let jcc = emit_sc_int_cmp(
                                out,
                                frame.opcode,
                                elem_a,
                                home_a,
                                home_b,
                                spill_base_slot,
                            );
                            let setcc = jcc.saturating_add(0x10); // 0x8x -> 0x9x (`setcc`)
                            out.extend_from_slice(&[
                                sc_rex(false, 0, 0, dst_r),
                                0x0F,
                                setcc,
                                0xC0 | (dst_r & 7),
                            ]);
                            emit_movzx_gpr32_gpr8(out, dst_r, dst_r);
                        } else {
                            match frame.opcode {
                                Opcode::CmpGt => {
                                    emit_load_home_into_xmm(out, 0, home_b, spill_base_slot);
                                    emit_load_home_into_xmm(out, 1, home_a, spill_base_slot);
                                    out.extend_from_slice(&[0xF2, 0x0F, 0xC2, 0xC1, 0x01]);
                                }
                                Opcode::CmpLt => {
                                    emit_load_home_into_xmm(out, 0, home_a, spill_base_slot);
                                    emit_load_home_into_xmm(out, 1, home_b, spill_base_slot);
                                    out.extend_from_slice(&[0xF2, 0x0F, 0xC2, 0xC1, 0x01]);
                                }
                                _ => {
                                    emit_load_home_into_xmm(out, 0, home_a, spill_base_slot);
                                    emit_load_home_into_xmm(out, 1, home_b, spill_base_slot);
                                    out.extend_from_slice(&[0xF2, 0x0F, 0xC2, 0xC1, 0x00]);
                                }
                            }
                            emit_movq_gpr_xmm(out, dst_r, 0);
                            emit_alu32_ri(out, 4, dst_r, 1);
                        }
                        emit_store_gpr_into_home(out, dst, dst_r, spill_base_slot);
                        scratch.set_home(idx, dst)?;
                    }
                }
                Opcode::SelectZero => {
                    if uc > 0 {
                        let el_b = scratch.node_elem(b_idx);
                        let dst = direct_local_dst(unit_nodes, scratch, &local_home, idx)
                            .unwrap_or_else(|| {
                                alloc.alloc_home(idx, el_b, eff_lu, spans_bar, scratch)
                            });
                        emit_load_home_into_gpr(out, REG_RAX, home_a, spill_base_slot);
                        out.extend_from_slice(&[0x48, 0xF7, 0xD8]); // neg rax
                        match home_b {
                            ScHome::Gpr(r) | ScHome::AliasGpr(r) => {
                                emit_alu64_rr(out, 0x23, REG_RAX, r);
                            }
                            ScHome::Spill(s) => {
                                emit_alu64_rm_rbp(
                                    out,
                                    0x23,
                                    REG_RAX,
                                    sc_spill_disp(spill_base_slot, s),
                                );
                            }
                            _ => {
                                emit_load_home_into_gpr(out, REG_R11, home_b, spill_base_slot);
                                emit_alu64_rr(out, 0x23, REG_RAX, REG_R11);
                            }
                        }
                        emit_store_gpr_into_home(out, dst, REG_RAX, spill_base_slot);
                        scratch.set_home(idx, dst)?;
                    }
                }
                Opcode::RingEnqueue => {
                    if uc > 0 {
                        let el_a = scratch.node_elem(a_idx);
                        let dst = alloc.alloc_home(idx, el_a, eff_lu, spans_bar, scratch);
                        emit_move_home_to_home(out, dst, home_a, spill_base_slot);
                        scratch.set_home(idx, dst)?;
                    }
                }
                Opcode::AssertContract => {
                    let el_a = scratch.node_elem(a_idx);
                    let dst = if uc > 0 {
                        alloc.alloc_home(idx, el_a, eff_lu, spans_bar, scratch)
                    } else {
                        ScHome::None
                    };
                    if el_a == ElemType::F64 {
                        emit_load_home_into_gpr(out, REG_RAX, home_a, spill_base_slot);
                        emit_store_gpr_into_home(out, dst, REG_RAX, spill_base_slot);
                        emit_movabs_reg(out, REG_R11, 0x7FF0_0000_0000_0000);
                        out.extend_from_slice(&[0x4C, 0x21, 0xD8]); // and rax, r11
                        out.extend_from_slice(&[0x4C, 0x39, 0xD8]); // cmp rax, r11
                        out.extend_from_slice(&[0x75, 0x0C]); // jne +12
                        emit_sys_exit(out, AOT_EXIT_CONTRACT_FAILED as u32);
                    } else {
                        emit_move_home_to_home(out, dst, home_a, spill_base_slot);
                    }
                    scratch.set_home(idx, dst)?;
                }
                Opcode::Block => {
                    let hdr_info = scratch.nodes.get(idx).copied();
                    let end_idx = hdr_info.map_or(0, |n| n.aux as usize);
                    if hdr_info.is_some_and(|n| (n.elem_flags & FLAG_SCOPE_HAS_RESULT) != 0) {
                        let end_uc = scratch.nodes.get(end_idx).map_or(0, |n| n.use_count());
                        if end_uc > 0 {
                            let end_el = scratch.node_elem(end_idx);
                            let end_lu = scratch.eff_last_use(end_idx);
                            let end_spans = scratch.spans_barrier(end_idx, end_lu);
                            let res_home =
                                alloc.alloc_home(end_idx, end_el, end_lu, end_spans, scratch);
                            scratch.set_home(end_idx, res_home)?;
                        }
                    }
                    scratch.ctrl_stack.push(AotCtrlFrame {
                        kind: AotCtrlKind::Block {
                            end_node_idx: end_idx,
                        },
                        end_patches: Vec::new(),
                    });
                }
                Opcode::Loop => {
                    let end_idx = scratch.nodes.get(idx).map_or(0, |n| n.aux as usize);
                    let bounded = if frame.proof_idx > 0 {
                        let proof = program
                            .proofs
                            .get(usize::from(frame.proof_idx - 1))
                            .ok_or(AotError::UnsupportedGraph("missing LoopBound proof"))?;
                        let fuel_spans = scratch.spans_barrier(idx, end_idx);
                        let fuel_home =
                            alloc.alloc_home(idx, ElemType::I64, end_idx, fuel_spans, scratch);
                        match fuel_home {
                            ScHome::Gpr(r) | ScHome::AliasGpr(r) => {
                                emit_mov_gpr_imm64(out, r, u64::from(proof.bound_hi));
                            }
                            ScHome::Spill(s) => {
                                emit_mov_gpr_imm64(out, REG_RAX, u64::from(proof.bound_hi));
                                emit_store_rbp_gpr(out, sc_spill_disp(spill_base_slot, s), REG_RAX);
                            }
                            _ => {}
                        }
                        scratch.set_home(idx, fuel_home)?;
                        true
                    } else {
                        false
                    };
                    let body_start_offset = out.len();
                    scratch.ctrl_stack.push(AotCtrlFrame {
                        kind: AotCtrlKind::Loop {
                            body_start_offset,
                            loop_node_idx: idx,
                            bounded,
                        },
                        end_patches: Vec::new(),
                    });
                }
                Opcode::If => {
                    let hdr_info = scratch.nodes.get(idx).copied();
                    let end_idx = hdr_info.map_or(0, |n| n.aux as usize);
                    if hdr_info.is_some_and(|n| (n.elem_flags & FLAG_SCOPE_HAS_RESULT) != 0) {
                        let end_uc = scratch.nodes.get(end_idx).map_or(0, |n| n.use_count());
                        if end_uc > 0 {
                            let end_el = scratch.node_elem(end_idx);
                            let end_lu = scratch.eff_last_use(end_idx);
                            let end_spans = scratch.spans_barrier(end_idx, end_lu);
                            let res_home =
                                alloc.alloc_home(end_idx, end_el, end_lu, end_spans, scratch);
                            scratch.set_home(end_idx, res_home)?;
                        }
                    }
                    let jcc_false = if let Some((cmp_idx, inverted)) = scratch.is_branch_fused(idx)
                    {
                        let cf = unit_nodes
                            .get(cmp_idx)
                            .ok_or(AotError::UnsupportedGraph("missing fused cmp node"))?;
                        let ca_home = scratch.get_home(cf.in_a as usize, unit_nodes);
                        let cb_home = scratch.get_home(cf.in_b as usize, unit_nodes);
                        let cel = scratch.node_elem(cf.in_a as usize);
                        let true_cc =
                            emit_sc_int_cmp(out, cf.opcode, cel, ca_home, cb_home, spill_base_slot);
                        if inverted {
                            true_cc
                        } else {
                            true_cc ^ 1
                        }
                    } else {
                        if let Some(r) = home_a.gpr_reg() {
                            out.extend_from_slice(&[
                                sc_rex(true, r, 0, r),
                                0x85,
                                0xC0 | ((r & 7) << 3) | (r & 7),
                            ]);
                        } else {
                            emit_load_home_into_gpr(out, REG_RAX, home_a, spill_base_slot);
                            out.extend_from_slice(&[0x48, 0x85, 0xC0]);
                        }
                        0x84 // jz
                    };
                    out.extend_from_slice(&[0x0F, jcc_false, 0x00, 0x00, 0x00, 0x00]);
                    let false_jump_patch = out.len() - 4;
                    scratch.ctrl_stack.push(AotCtrlFrame {
                        kind: AotCtrlKind::If {
                            end_node_idx: end_idx,
                            false_jump_patch,
                            has_else: false,
                        },
                        end_patches: Vec::new(),
                    });
                }
                Opcode::Else => {
                    let (end_node_idx, false_jump_patch) = {
                        let top = scratch
                            .ctrl_stack
                            .last()
                            .ok_or(AotError::UnsupportedGraph("Else without open If"))?;
                        let AotCtrlKind::If {
                            end_node_idx,
                            false_jump_patch,
                            ..
                        } = top.kind
                        else {
                            return Err(AotError::UnsupportedGraph("Else without matching If"));
                        };
                        (end_node_idx, false_jump_patch)
                    };
                    if frame.in_b != 0 {
                        let res_home = scratch.get_home(end_node_idx, unit_nodes);
                        emit_move_home_to_home(out, res_home, home_a, spill_base_slot);
                    }
                    out.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]);
                    let end_patch = out.len() - 4;
                    let else_body_pos = out.len();
                    patch_rel32(out, false_jump_patch, else_body_pos)?;
                    if let Some(top) = scratch.ctrl_stack.last_mut() {
                        top.end_patches.push(end_patch);
                        if let AotCtrlKind::If {
                            ref mut has_else, ..
                        } = top.kind
                        {
                            *has_else = true;
                        }
                    }
                }
                Opcode::End => {
                    if let Some(popped) = scratch.ctrl_stack.pop() {
                        match popped.kind {
                            AotCtrlKind::Block { end_node_idx } => {
                                if frame.in_b != 0 {
                                    let res_home = scratch.get_home(end_node_idx, unit_nodes);
                                    emit_move_home_to_home(out, res_home, home_a, spill_base_slot);
                                }
                                let end_pos = out.len();
                                for p in popped.end_patches {
                                    patch_rel32(out, p, end_pos)?;
                                }
                            }
                            AotCtrlKind::Loop { .. } => {
                                let end_pos = out.len();
                                for p in popped.end_patches {
                                    patch_rel32(out, p, end_pos)?;
                                }
                            }
                            AotCtrlKind::If {
                                end_node_idx,
                                false_jump_patch,
                                has_else,
                            } => {
                                if has_else {
                                    if frame.in_b != 0 {
                                        let res_home = scratch.get_home(end_node_idx, unit_nodes);
                                        emit_move_home_to_home(
                                            out,
                                            res_home,
                                            home_a,
                                            spill_base_slot,
                                        );
                                    }
                                    let join_pos = out.len();
                                    for p in popped.end_patches {
                                        patch_rel32(out, p, join_pos)?;
                                    }
                                } else {
                                    let end_pos = out.len();
                                    patch_rel32(out, false_jump_patch, end_pos)?;
                                    for p in popped.end_patches {
                                        patch_rel32(out, p, end_pos)?;
                                    }
                                }
                            }
                        }
                    } else if func_mode {
                        // End of `Func`: emit return value move and epilogue into `out`.
                        if frame.in_b != 0 {
                            emit_load_home_into_gpr(out, REG_RAX, home_a, spill_base_slot);
                        }
                        let epilogue_pos = out.len();
                        for p in scratch.epilogue_patches.drain(..) {
                            patch_rel32(out, p, epilogue_pos)?;
                        }
                        let total_slots = spill_base_slot.saturating_add(alloc.next_spill_slot);
                        let frame_bytes = total_slots.saturating_mul(8);
                        let saved_mask = alloc.used_callee_saved_mask;
                        if frame_bytes > 0 {
                            out.extend_from_slice(&[0x48, 0x89, 0xEC, 0x5D]); // mov rsp, rbp; pop rbp
                        }
                        for &reg in SC_CALLEE_SAVED_GPRS.iter().rev() {
                            if (saved_mask & (1u16 << reg)) != 0 {
                                emit_pop_gpr(out, reg);
                            }
                        }
                        // add r14, imm32(budget_delta_u32); ret
                        out.extend_from_slice(&[0x49, 0x81, 0xC6]);
                        out.extend_from_slice(&budget_delta_u32.to_le_bytes());
                        out.push(0xC3);
                    } else {
                        return Err(AotError::UnsupportedGraph("End without open block"));
                    }
                }
                Opcode::Br => {
                    let depth = frame.in_a as usize;
                    let pos = scratch
                        .ctrl_stack
                        .len()
                        .checked_sub(1 + depth)
                        .ok_or(AotError::UnsupportedGraph("Br depth overflow"))?;
                    let target = scratch
                        .ctrl_stack
                        .get_mut(pos)
                        .ok_or(AotError::UnsupportedGraph("Br target missing"))?;
                    match target.kind {
                        AotCtrlKind::Loop {
                            body_start_offset,
                            loop_node_idx,
                            bounded,
                        } => {
                            let fuel_home = scratch.get_home(loop_node_idx, unit_nodes);
                            emit_loop_backedge_jump(
                                out,
                                body_start_offset,
                                fuel_home,
                                spill_base_slot,
                                bounded,
                            )?;
                        }
                        AotCtrlKind::Block { .. } | AotCtrlKind::If { .. } => {
                            out.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]);
                            target.end_patches.push(out.len() - 4);
                        }
                    }
                }
                Opcode::BrIf => {
                    let depth = frame.in_b as usize;
                    let pos = scratch
                        .ctrl_stack
                        .len()
                        .checked_sub(1 + depth)
                        .ok_or(AotError::UnsupportedGraph("BrIf depth overflow"))?;
                    let jcc_true = if let Some((cmp_idx, inverted)) = scratch.is_branch_fused(idx) {
                        let cf = unit_nodes
                            .get(cmp_idx)
                            .ok_or(AotError::UnsupportedGraph("missing fused cmp node"))?;
                        let ca_home = scratch.get_home(cf.in_a as usize, unit_nodes);
                        let cb_home = scratch.get_home(cf.in_b as usize, unit_nodes);
                        let cel = scratch.node_elem(cf.in_a as usize);
                        let cc =
                            emit_sc_int_cmp(out, cf.opcode, cel, ca_home, cb_home, spill_base_slot);
                        if inverted {
                            cc ^ 1
                        } else {
                            cc
                        }
                    } else {
                        if let Some(r) = home_a.gpr_reg() {
                            out.extend_from_slice(&[
                                sc_rex(true, r, 0, r),
                                0x85,
                                0xC0 | ((r & 7) << 3) | (r & 7),
                            ]);
                        } else {
                            emit_load_home_into_gpr(out, REG_RAX, home_a, spill_base_slot);
                            out.extend_from_slice(&[0x48, 0x85, 0xC0]);
                        }
                        0x85 // jnz
                    };

                    let target_kind = match scratch.ctrl_stack.get(pos).map(|f| &f.kind) {
                        Some(AotCtrlKind::Loop {
                            body_start_offset,
                            loop_node_idx,
                            bounded,
                        }) => Some((*body_start_offset, *loop_node_idx, *bounded)),
                        Some(AotCtrlKind::Block { .. } | AotCtrlKind::If { .. }) => None,
                        None => return Err(AotError::UnsupportedGraph("BrIf target missing")),
                    };
                    if let Some((body_start_offset, loop_node_idx, bounded)) = target_kind {
                        if bounded {
                            out.extend_from_slice(&[0x0F, jcc_true ^ 1, 0x00, 0x00, 0x00, 0x00]);
                            let skip_patch = out.len() - 4;
                            let fuel_home = scratch.get_home(loop_node_idx, unit_nodes);
                            emit_loop_backedge_jump(
                                out,
                                body_start_offset,
                                fuel_home,
                                spill_base_slot,
                                true,
                            )?;
                            let after_pos = out.len();
                            patch_rel32(out, skip_patch, after_pos)?;
                        } else {
                            out.extend_from_slice(&[0x0F, jcc_true, 0x00, 0x00, 0x00, 0x00]);
                            let patch_pos = out.len() - 4;
                            patch_rel32(out, patch_pos, body_start_offset)?;
                        }
                    } else if let Some(target) = scratch.ctrl_stack.get_mut(pos) {
                        out.extend_from_slice(&[0x0F, jcc_true, 0x00, 0x00, 0x00, 0x00]);
                        target.end_patches.push(out.len() - 4);
                    }
                }
                Opcode::RegionNew => {
                    let slot = frame.in_a as usize;
                    let decl = regions
                        .get(slot)
                        .and_then(|d| *d)
                        .ok_or(AotError::UnsupportedGraph("undeclared RegionNew slot"))?;
                    emit_sc_region_new(out, slot, decl, home_b, spill_base_slot)?;
                }
                Opcode::RegionGrow => {
                    let slot = frame.in_a as usize;
                    let decl = regions
                        .get(slot)
                        .and_then(|d| *d)
                        .ok_or(AotError::UnsupportedGraph("undeclared RegionGrow slot"))?;
                    let dst = if uc > 0 {
                        alloc.alloc_home(idx, ElemType::I64, eff_lu, spans_bar, scratch)
                    } else {
                        ScHome::None
                    };
                    emit_sc_region_grow(out, slot, decl, home_b, dst, spill_base_slot)?;
                    scratch.set_home(idx, dst)?;
                }
                Opcode::RegionLen => {
                    if uc > 0 {
                        let slot = frame.in_a as usize;
                        let off_len = sc_region_off(slot, 8);
                        let dst = direct_local_dst(unit_nodes, scratch, &local_home, idx)
                            .unwrap_or_else(|| {
                                alloc.alloc_home(idx, ElemType::I64, eff_lu, spans_bar, scratch)
                            });
                        let dst_r = dst.gpr_reg().unwrap_or(REG_RAX);
                        emit_load_reg_r15(out, dst_r, off_len);
                        emit_store_gpr_into_home(out, dst, dst_r, spill_base_slot);
                        scratch.set_home(idx, dst)?;
                    }
                }
                Opcode::RegionLoad => {
                    let slot = frame.in_a as usize;
                    let decl = regions
                        .get(slot)
                        .and_then(|d| *d)
                        .ok_or(AotError::UnsupportedGraph("undeclared RegionLoad slot"))?;
                    let off_base = sc_region_off(slot, 0);
                    let off_len = sc_region_off(slot, 8);
                    let dst = if uc > 0 {
                        direct_local_dst(unit_nodes, scratch, &local_home, idx).unwrap_or_else(
                            || {
                                if alloc.free_gpr_mask == 0
                                    && can_use_transient_rax(
                                        unit_nodes, scratch, idx, decl.elem, uc, eff_lu, spans_bar,
                                    )
                                {
                                    ScHome::AliasGpr(REG_RAX)
                                } else {
                                    alloc.alloc_home(idx, decl.elem, eff_lu, spans_bar, scratch)
                                }
                            },
                        )
                    } else {
                        ScHome::None
                    };

                    // Fast path: proven-in-bounds constant index.
                    let const_proven_disp = if frame.proof_idx != 0 {
                        if let ScHome::ConstI64(c) = home_b {
                            let scale = i64::try_from(decl.elem.region_elem_bytes()).unwrap_or(8);
                            if c >= 0 {
                                c.checked_mul(scale).and_then(|off| i32::try_from(off).ok())
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    if let Some(byte_disp) = const_proven_disp {
                        if uc > 0 {
                            emit_load_reg_r15(out, REG_R11, off_base);
                            if decl.elem == ElemType::F64 {
                                let dx = dst.xmm_reg().unwrap_or(0);
                                out.extend_from_slice(&[
                                    0xF2,
                                    sc_rex(false, dx, 0, REG_R11),
                                    0x0F,
                                    0x10,
                                    0x83 | ((dx & 7) << 3),
                                ]);
                                out.extend_from_slice(&byte_disp.to_le_bytes());
                                emit_store_xmm_into_home(out, dst, dx, spill_base_slot);
                            } else {
                                let dr = dst.gpr_reg().unwrap_or(REG_RAX);
                                if decl.elem == ElemType::U8 {
                                    out.extend_from_slice(&[
                                        sc_rex(false, dr, 0, REG_R11),
                                        0x0F,
                                        0xB6,
                                        0x83 | ((dr & 7) << 3),
                                    ]);
                                } else {
                                    out.extend_from_slice(&[
                                        sc_rex(true, dr, 0, REG_R11),
                                        0x8B,
                                        0x83 | ((dr & 7) << 3),
                                    ]);
                                }
                                out.extend_from_slice(&byte_disp.to_le_bytes());
                                emit_store_gpr_into_home(out, dst, dr, spill_base_slot);
                            }
                        }
                    } else {
                        let r_idx = if let Some(r) = home_b.gpr_reg() {
                            r
                        } else {
                            emit_load_home_into_gpr(out, REG_RAX, home_b, spill_base_slot);
                            REG_RAX
                        };
                        if frame.proof_idx == 0 {
                            emit_cmp_reg_r15(out, r_idx, off_len);
                            out.extend_from_slice(&[0x72, 0x0C]);
                            emit_sys_exit(out, AOT_EXIT_GATHER_OOB as u32);
                        }
                        if uc > 0 {
                            emit_load_reg_r15(out, REG_R11, off_base);
                            if decl.elem == ElemType::F64 {
                                let dx = dst.xmm_reg().unwrap_or(0);
                                out.extend_from_slice(&[
                                    0xF2,
                                    sc_rex(false, dx, r_idx, REG_R11),
                                    0x0F,
                                    0x10,
                                    ((dx & 7) << 3) | 0x04,
                                    0xC0 | ((r_idx & 7) << 3) | 0x03,
                                ]);
                                emit_store_xmm_into_home(out, dst, dx, spill_base_slot);
                            } else {
                                let dr = dst.gpr_reg().unwrap_or(REG_RAX);
                                if decl.elem == ElemType::U8 {
                                    out.extend_from_slice(&[
                                        sc_rex(false, dr, r_idx, REG_R11),
                                        0x0F,
                                        0xB6,
                                        ((dr & 7) << 3) | 0x04,
                                        ((r_idx & 7) << 3) | 0x03,
                                    ]);
                                } else {
                                    out.extend_from_slice(&[
                                        sc_rex(true, dr, r_idx, REG_R11),
                                        0x8B,
                                        ((dr & 7) << 3) | 0x04,
                                        0xC0 | ((r_idx & 7) << 3) | 0x03,
                                    ]);
                                }
                                emit_store_gpr_into_home(out, dst, dr, spill_base_slot);
                            }
                        }
                    }
                    scratch.set_home(idx, dst)?;
                }
                Opcode::RegionStore => {
                    let slot = frame.in_a as usize;
                    let decl = regions
                        .get(slot)
                        .and_then(|d| *d)
                        .ok_or(AotError::UnsupportedGraph("undeclared RegionStore slot"))?;
                    let (idx_node, val_node) = frame.decode_u16_pair();
                    let idx_home = scratch.get_home(idx_node as usize, unit_nodes);
                    let val_home = scratch.get_home(val_node as usize, unit_nodes);
                    let off_base = sc_region_off(slot, 0);
                    let off_len = sc_region_off(slot, 8);

                    let r_idx = if let Some(r) = idx_home.gpr_reg() {
                        r
                    } else {
                        emit_load_home_into_gpr(out, REG_RAX, idx_home, spill_base_slot);
                        REG_RAX
                    };
                    if frame.proof_idx == 0 {
                        emit_cmp_reg_r15(out, r_idx, off_len);
                        out.extend_from_slice(&[0x72, 0x0C]);
                        emit_sys_exit(out, AOT_EXIT_GATHER_OOB as u32);
                    }
                    emit_load_reg_r15(out, REG_R11, off_base);

                    match (decl.elem, val_home) {
                        (ElemType::U8, ScHome::ConstU8(b)) => {
                            out.extend_from_slice(&[
                                sc_rex(false, 0, r_idx, REG_R11),
                                0xC6,
                                0x04,
                                ((r_idx & 7) << 3) | 0x03,
                                b,
                            ]);
                        }
                        (ElemType::U8, ScHome::Gpr(rv) | ScHome::AliasGpr(rv)) => {
                            out.extend_from_slice(&[
                                sc_rex(false, rv, r_idx, REG_R11),
                                0x88,
                                ((rv & 7) << 3) | 0x04,
                                ((r_idx & 7) << 3) | 0x03,
                            ]);
                        }
                        (ElemType::I64, ScHome::Gpr(rv) | ScHome::AliasGpr(rv)) => {
                            out.extend_from_slice(&[
                                sc_rex(true, rv, r_idx, REG_R11),
                                0x89,
                                ((rv & 7) << 3) | 0x04,
                                0xC0 | ((r_idx & 7) << 3) | 0x03,
                            ]);
                        }
                        (ElemType::I64, ScHome::ConstI64(v)) if i32::try_from(v).is_ok() => {
                            let i32v = i32::try_from(v).unwrap_or(0);
                            out.extend_from_slice(&[
                                sc_rex(true, 0, r_idx, REG_R11),
                                0xC7,
                                0x04,
                                0xC0 | ((r_idx & 7) << 3) | 0x03,
                            ]);
                            out.extend_from_slice(&i32v.to_le_bytes());
                        }
                        (ElemType::F64, ScHome::Xmm(xv) | ScHome::AliasXmm(xv)) => {
                            out.extend_from_slice(&[
                                0xF2,
                                sc_rex(false, xv, r_idx, REG_R11),
                                0x0F,
                                0x11,
                                ((xv & 7) << 3) | 0x04,
                                0xC0 | ((r_idx & 7) << 3) | 0x03,
                            ]);
                        }
                        _ => {
                            if decl.elem == ElemType::U8 {
                                emit_alu64_rr(out, 0x03, REG_R11, r_idx);
                                emit_load_home_into_gpr(out, REG_RAX, val_home, spill_base_slot);
                                out.extend_from_slice(&[0x41, 0x88, 0x03]);
                            } else {
                                out.extend_from_slice(&[
                                    sc_rex(true, REG_R11, r_idx, REG_R11),
                                    0x8D,
                                    0x1C,
                                    0xC0 | ((r_idx & 7) << 3) | 0x03,
                                ]);
                                emit_load_home_into_gpr(out, REG_RAX, val_home, spill_base_slot);
                                out.extend_from_slice(&[0x49, 0x89, 0x03]);
                            }
                        }
                    }
                }
                Opcode::RegionFree | Opcode::RegionDrop => {
                    let slot = frame.in_a as usize;
                    emit_sc_region_free(out, slot)?;
                }
                Opcode::RegionLease | Opcode::RegionEndLease => {}
                Opcode::RegionMove => {
                    emit_sc_region_move(out, frame.in_a as usize, frame.in_b as usize);
                }
                Opcode::Return => {
                    if frame.in_a != 0 {
                        emit_load_home_into_gpr(out, REG_RAX, home_a, spill_base_slot);
                    }
                    out.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]);
                    scratch.epilogue_patches.push(out.len() - 4);
                }
                Opcode::CallArg => {
                    let a2 = frame.in_a as usize;
                    let a3 = if frame.in_b == u32::from(CALL_ARG_SENTINEL) {
                        None
                    } else {
                        Some(frame.in_b as usize)
                    };
                    pending_call_args = Some((a2, a3));
                }
                Opcode::Call => {
                    let target_id = frame.in_a as usize;
                    let callee_sig = func_sigs
                        .get(target_id)
                        .copied()
                        .flatten()
                        .ok_or(AotError::UnsupportedGraph("unknown callee FuncSig"))?;
                    let (arg0_raw, arg1_raw) = frame.decode_u16_pair();
                    let sentinel = u32::from(CALL_ARG_SENTINEL);
                    let mut arg_homes: [Option<ScHome>; 4] = [None; 4];
                    if arg0_raw != sentinel {
                        arg_homes[0] = Some(scratch.get_home(arg0_raw as usize, unit_nodes));
                    }
                    if arg1_raw != sentinel {
                        arg_homes[1] = Some(scratch.get_home(arg1_raw as usize, unit_nodes));
                    }
                    if let Some((a2, a3_opt)) = pending_call_args.take() {
                        arg_homes[2] = Some(scratch.get_home(a2, unit_nodes));
                        if let Some(a3) = a3_opt {
                            arg_homes[3] = Some(scratch.get_home(a3, unit_nodes));
                        }
                    }
                    // 1. Parallel-move all GPR arguments into their target arg registers first.
                    let mut gpr_moves: [(u8, u8); 4] = [(0, 0); 4];
                    let mut gm_len = 0usize;
                    for (i, opt_h) in arg_homes.iter().enumerate() {
                        if let (Some(&dst_arg), Some(h)) = (SC_ARG_GPRS.get(i), *opt_h) {
                            if let Some(src_r) = h.gpr_reg() {
                                if let Some(slot) = gpr_moves.get_mut(gm_len) {
                                    *slot = (dst_arg, src_r);
                                    gm_len += 1;
                                }
                            }
                        }
                    }
                    if let Some(slice) = gpr_moves.get(..gm_len) {
                        emit_parallel_gpr_moves(out, slice);
                    }
                    // 2. Materialize non-GPR arguments (constants, spills, XMMs) into their target arg registers.
                    for (i, opt_h) in arg_homes.iter().enumerate() {
                        if let (Some(&dst_arg), Some(h)) = (SC_ARG_GPRS.get(i), *opt_h) {
                            if h.gpr_reg().is_none() {
                                emit_load_home_into_gpr(out, dst_arg, h, spill_base_slot);
                            }
                        }
                    }

                    // Expire arguments that die at this Call BEFORE allocating the return value's home.
                    alloc.expire_at(idx, scratch);

                    out.extend_from_slice(&[0xE8, 0x00, 0x00, 0x00, 0x00]); // call rel32
                    call_patches.push((out.len() - 4, target_id));

                    if let Some(ret_elem) = callee_sig.ret_elem {
                        if uc > 0 {
                            let keep_in_rax_local_set = ret_elem != ElemType::F64
                                && uc == 1
                                && !spans_bar
                                && eff_lu > idx
                                && eff_lu <= idx.saturating_add(4)
                                && unit_nodes.get(eff_lu).is_some_and(|ls_f| {
                                    ls_f.opcode == Opcode::LocalSet
                                        && ls_f.in_b as usize == idx
                                        && scratch.nodes.get(eff_lu).map_or(0, |n| n.use_count())
                                            == 0
                                        && matches!(
                                            local_home.get(ls_f.in_a as usize).copied(),
                                            Some(ScLocalHome::Gpr(_))
                                        )
                                })
                                && {
                                    // Each live Add/Sub in `(idx + 1)..eff_lu` allocates at most
                                    // one fresh GPR while `ConstI64` and aliased `LocalGet` take
                                    // none. Require enough free GPRs up front so no window node
                                    // can spill or fall back to scratch `rax` before `eff_lu`.
                                    let mut window_gprs_needed = 0usize;
                                    let mut window_ok = true;
                                    for k in (idx + 1)..eff_lu {
                                        let Some(kf) = unit_nodes.get(k) else {
                                            window_ok = false;
                                            break;
                                        };
                                        match kf.opcode {
                                            Opcode::ConstI64 => {}
                                            Opcode::LocalGet => {
                                                let nls = scratch
                                                    .nodes
                                                    .get(k)
                                                    .map_or(u32::MAX, |n| n.aux)
                                                    as usize;
                                                if !matches!(
                                                    local_home.get(kf.in_a as usize).copied(),
                                                    Some(ScLocalHome::Gpr(_))
                                                ) || scratch.eff_last_use(k) >= nls
                                                {
                                                    window_ok = false;
                                                    break;
                                                }
                                            }
                                            Opcode::Add | Opcode::Sub => {
                                                let ka = kf.in_a as usize;
                                                let kb = kf.in_b as usize;
                                                let k_lu = scratch.eff_last_use(k);
                                                let k_ok = ka != idx
                                                    && kb != idx
                                                    && scratch.node_elem(k) == ElemType::I64
                                                    && !scratch.spans_barrier(k, k_lu)
                                                    && k_lu.saturating_sub(k) <= 256
                                                    && unit_nodes.get(ka).is_some_and(|af| {
                                                        af.opcode == Opcode::LocalGet
                                                            && matches!(
                                                                local_home
                                                                    .get(af.in_a as usize)
                                                                    .copied(),
                                                                Some(ScLocalHome::Gpr(_))
                                                            )
                                                    })
                                                    && unit_nodes.get(kb).is_some_and(|bf| {
                                                        if bf.opcode != Opcode::ConstI64 {
                                                            return false;
                                                        }
                                                        let v = bf.decode_const_i64();
                                                        if kf.opcode == Opcode::Sub {
                                                            v.checked_neg()
                                                                .and_then(|n| i32::try_from(n).ok())
                                                                .is_some()
                                                        } else {
                                                            i32::try_from(v).is_ok()
                                                        }
                                                    });
                                                if !k_ok {
                                                    window_ok = false;
                                                    break;
                                                }
                                                if scratch.nodes.get(k).map_or(0, |n| n.use_count())
                                                    > 0
                                                {
                                                    window_gprs_needed += 1;
                                                }
                                            }
                                            _ => {
                                                window_ok = false;
                                                break;
                                            }
                                        }
                                    }
                                    window_ok
                                        && (alloc.free_gpr_mask.count_ones() as usize)
                                            >= window_gprs_needed
                                };
                            let keep_in_rax = keep_in_rax_local_set
                                || (ret_elem != ElemType::F64
                                    && uc == 1
                                    && (is_immediate_func_return(idx)
                                        || unit_nodes.get(idx.saturating_add(1)).is_some_and(
                                            |nf| {
                                                let next_idx = idx.saturating_add(1);
                                                matches!(
                                                    nf.opcode,
                                                    Opcode::Add
                                                        | Opcode::Sub
                                                        | Opcode::Mul
                                                        | Opcode::BitAnd
                                                        | Opcode::BitOr
                                                        | Opcode::BitXor
                                                ) && scratch.node_elem(next_idx) != ElemType::F64
                                                    && (nf.in_a as usize == idx
                                                        || (nf.opcode != Opcode::Sub
                                                            && nf.in_b as usize == idx))
                                                    && scratch
                                                        .nodes
                                                        .get(next_idx)
                                                        .map_or(0, |n| n.use_count())
                                                        == 1
                                                    && is_immediate_func_return(next_idx)
                                            },
                                        )));
                            let dst = if keep_in_rax {
                                ScHome::AliasGpr(REG_RAX)
                            } else {
                                direct_local_dst(unit_nodes, scratch, &local_home, idx)
                                    .unwrap_or_else(|| {
                                        alloc.alloc_home(idx, ret_elem, eff_lu, spans_bar, scratch)
                                    })
                            };
                            emit_store_gpr_into_home(out, dst, REG_RAX, spill_base_slot);
                            scratch.set_home(idx, dst)?;
                        }
                    }
                    continue;
                }
                Opcode::ArenaInput
                | Opcode::ArenaSlice
                | Opcode::ArenaMutate
                | Opcode::FilterCompact
                | Opcode::ReduceSum
                | Opcode::ReduceMax
                | Opcode::ReduceMin
                | Opcode::Gather => {
                    return Err(AotError::UnsupportedGraph(
                        "columnar opcode in scalar structured control-flow AOT",
                    ));
                }
            }

            alloc.expire_at(idx, scratch);
        }
    }

    if let Some(sig) = cur_sig {
        let stub_pos =
            stack_overflow_stub.ok_or(AotError::UnsupportedGraph("missing stack overflow stub"))?;
        let total_slots = spill_base_slot.saturating_add(alloc.next_spill_slot);
        let frame_bytes = total_slots.saturating_mul(8);
        let saved_mask = alloc.used_callee_saved_mask;
        // Use compact 2-byte `js rel8` / `jnz rel8` into a trailing 5-byte `jmp stack_overflow_stub`
        // when the entire function (including early guard and prologue) is guaranteed <= 127 bytes.
        let use_short_budget_jumps = fn_body_buf.len() <= 48 && frame_bytes == 0 && sig.arity <= 2;
        let mut short_oom_patches: Vec<usize> = Vec::new();

        // 1. Optional early-return base-case `If` guard (non-mutating `r14` budget check).
        if let Some((if_idx, guard_end)) = early_guard {
            if let Some((cmp_idx, inverted)) = scratch.is_branch_fused(if_idx) {
                if let (Some(cf), Some(ret_f)) =
                    (unit_nodes.get(cmp_idx), unit_nodes.get(guard_end - 1))
                {
                    if let (Some(ca_home), Some(cb_home)) = (
                        early_guard_arg_home(unit_nodes, cf.in_a as usize),
                        early_guard_arg_home(unit_nodes, cf.in_b as usize),
                    ) {
                        let cel = scratch.node_elem(cf.in_a as usize);
                        let true_cc = emit_sc_int_cmp(code, cf.opcode, cel, ca_home, cb_home, 0);
                        let jcc_false = if inverted { true_cc } else { true_cc ^ 1 };
                        code.extend_from_slice(&[0x70 | (jcc_false & 0x0F), 0x00]);
                        let skip_patch = code.len() - 1;
                        emit_mov_gpr_gpr(code, REG_RAX, REG_R14);
                        emit_sc_budget_sub_check(
                            code,
                            REG_RAX,
                            budget_delta_u32,
                            stub_pos,
                            if use_short_budget_jumps {
                                Some(&mut short_oom_patches)
                            } else {
                                None
                            },
                        )?;
                        if ret_f.in_a != 0 {
                            if let Some(ret_home) =
                                early_guard_arg_home(unit_nodes, ret_f.in_a as usize)
                            {
                                emit_load_home_into_gpr(code, REG_RAX, ret_home, 0);
                            }
                        }
                        code.push(0xC3); // ret
                        let rel8 = code.len().saturating_sub(skip_patch + 1);
                        if let (Ok(r8), Some(slot)) = (u8::try_from(rel8), code.get_mut(skip_patch))
                        {
                            *slot = r8;
                        } else {
                            return Err(AotError::UnsupportedGraph(
                                "early guard short branch overflow",
                            ));
                        }
                    }
                }
            }
        }

        // 2. Main-path packed depth + stack-unit budget check using `r14` (zero L1D loads, macro-fused).
        emit_sc_budget_sub_check(
            code,
            REG_R14,
            budget_delta_u32,
            stub_pos,
            if use_short_budget_jumps {
                Some(&mut short_oom_patches)
            } else {
                None
            },
        )?;

        // 3. Emit exact minimal prologue (zero NOPs) + param moves, then append `fn_body_buf`.
        for &reg in &SC_CALLEE_SAVED_GPRS {
            if (saved_mask & (1u16 << reg)) != 0 {
                emit_push_gpr(code, reg);
            }
        }
        if frame_bytes > 0 {
            code.extend_from_slice(&[0x55, 0x48, 0x89, 0xE5]); // push rbp; mov rbp, rsp
            if let Ok(fb8) = i8::try_from(frame_bytes) {
                code.extend_from_slice(&[0x48, 0x83, 0xEC, fb8 as u8]); // sub rsp, imm8
            } else {
                code.extend_from_slice(&[0x48, 0x81, 0xEC]); // sub rsp, imm32
                code.extend_from_slice(&frame_bytes.to_le_bytes());
            }
        }
        let arity = usize::from(sig.arity);
        for i in 0..arity {
            let Some(&arg_reg) = SC_ARG_GPRS.get(i) else {
                continue;
            };
            match local_home.get(i).copied().unwrap_or(ScLocalHome::Unused) {
                ScLocalHome::Slot(disp) => emit_store_rbp_gpr(code, disp, arg_reg),
                ScLocalHome::Xmm(x) => emit_movq_xmm_gpr(code, x, arg_reg),
                ScLocalHome::Gpr(_) | ScLocalHome::Unused => {}
            }
        }
        let mut param_moves: [(u8, u8); 4] = [(0, 0); 4];
        let mut pm_len = 0usize;
        for i in 0..arity {
            if let (Some(&arg_reg), Some(ScLocalHome::Gpr(dst_r))) =
                (SC_ARG_GPRS.get(i), local_home.get(i).copied())
            {
                if let Some(slot) = param_moves.get_mut(pm_len) {
                    *slot = (dst_r, arg_reg);
                    pm_len += 1;
                }
            }
        }
        if let Some(slice) = param_moves.get(..pm_len) {
            emit_parallel_gpr_moves(code, slice);
        }

        let body_base = code.len();
        code.extend_from_slice(&fn_body_buf);
        if !short_oom_patches.is_empty() {
            let local_oom_pos = code.len();
            for p in short_oom_patches {
                let rel = local_oom_pos.saturating_sub(p + 1);
                if let (Ok(r8), Some(slot)) = (i8::try_from(rel), code.get_mut(p)) {
                    *slot = r8 as u8;
                } else {
                    return Err(AotError::UnsupportedGraph("short budget branch overflow"));
                }
            }
            code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]); // jmp stack_overflow_stub
            let jp = code.len() - 4;
            patch_rel32(code, jp, stub_pos)?;
        }
        if let Some(patches) = call_patches.get_mut(body_call_patch_start..) {
            for (patch_pos, _) in patches {
                *patch_pos = patch_pos.saturating_add(body_base);
            }
        }
        scratch.fn_body = fn_body_buf;
    } else {
        scratch.fn_body = fn_body_buf;
        let total_slots = spill_base_slot.saturating_add(alloc.next_spill_slot);
        let raw_bytes = usize::try_from(total_slots).unwrap_or(0).saturating_mul(8);
        let frame_bytes = u32::try_from(raw_bytes.div_ceil(32) * 32)
            .map_err(|_e| AotError::UnsupportedGraph("stack frame overflow"))?;
        if let Some(patch_pos) = top_frame_sub_patch {
            if let Some(dst) = code.get_mut(patch_pos..patch_pos + 4) {
                dst.copy_from_slice(&frame_bytes.to_le_bytes());
            }
        }
        if let Some(out_slot) = output_region_slot {
            emit_sc_output_region_epilogue(code, out_slot)?;
        } else {
            let out_home = scratch.get_home(program.output_node as usize, unit_nodes);
            emit_load_home_into_gpr(code, REG_RAX, out_home, spill_base_slot);
            emit_store_rbp_gpr(code, stdout_disp, REG_RAX);
            code.extend_from_slice(&[0xB8, 0x01, 0x00, 0x00, 0x00]); // mov eax, 1 (SYS_write)
            code.extend_from_slice(&[0xBF, 0x01, 0x00, 0x00, 0x00]); // mov edi, 1
            code.extend_from_slice(&[0x48, 0x8D, 0xB5]); // lea rsi, [rbp + stdout_disp]
            code.extend_from_slice(&stdout_disp.to_le_bytes());
            code.extend_from_slice(&[0xBA, 0x08, 0x00, 0x00, 0x00]); // mov edx, 8
            code.extend_from_slice(&[0x0F, 0x05]); // syscall
            emit_sys_exit(code, 0);
        }
    }

    Ok(())
}

fn emit_x86_64_scalar_control_flow(program: &RulProgram) -> Result<Vec<u8>, AotError> {
    let node_count = program.nodes.len();
    let func_mode = program.has_functions();
    let mut code: Vec<u8> =
        Vec::with_capacity(ELF_CODE_OFFSET + 256 + node_count.saturating_mul(16));
    code.resize(ELF_CODE_OFFSET, 0);

    let mut regions: [Option<AotRegionDecl>; MAX_REGION_SLOTS] = [None; MAX_REGION_SLOTS];
    let mut has_any_region = false;
    let mut output_region_slot: Option<usize> = None;
    let mut func_sigs_raw: Vec<FuncSigInfo> = Vec::new();
    for proof in &program.proofs {
        match proof.rule_kind {
            ProofRuleKind::RegionDecl => {
                let Some((elem, role)) = ProofEntry::decode_region_decl_witness(proof.witness_node)
                else {
                    return Err(AotError::UnsupportedGraph("invalid RegionDecl witness"));
                };
                let slot = usize::from(proof.target_arena);
                if let Some(entry) = regions.get_mut(slot) {
                    *entry = Some(AotRegionDecl {
                        min_len: u64::from(proof.bound_lo),
                        max_len: u64::from(proof.bound_hi),
                        elem,
                        role,
                    });
                    has_any_region = true;
                    if role == RegionRole::Output {
                        output_region_slot = Some(slot);
                    }
                }
            }
            ProofRuleKind::FuncSig => {
                let sig = proof
                    .decode_func_sig()
                    .ok_or(AotError::UnsupportedGraph("invalid FuncSig proof"))?;
                func_sigs_raw.push(sig);
            }
            _ => {}
        }
    }

    let func_count = func_sigs_raw.len();
    let mut func_sigs: Vec<Option<FuncSigInfo>> = vec![None; func_count];
    for sig in func_sigs_raw {
        if let Some(slot) = func_sigs.get_mut(usize::from(sig.func_id)) {
            *slot = Some(sig);
        }
    }
    let mut func_offsets: Vec<Option<usize>> = vec![None; func_count];
    let mut call_patches: Vec<(usize, usize)> = Vec::new();
    let mut scratch = ScUnitScratch::new();

    if func_mode {
        // 1. Entry stub: map 4096B global page in `r15`.
        code.extend_from_slice(&[0xBE, 0x00, 0x10, 0x00, 0x00]);
        emit_sc_sys_mmap_anon(&mut code);
        code.extend_from_slice(&[0x49, 0x89, 0xC7]); // mov r15, rax

        // 2. Map private stack (`8 * MAX_STACK_UNITS + 4096`) with bottom `PROT_NONE` guard page.
        let stack_total_bytes = u64::from(MAX_STACK_UNITS)
            .saturating_mul(8)
            .saturating_add(4096);
        emit_movabs_reg(&mut code, 6, stack_total_bytes);
        emit_sc_sys_mmap_anon(&mut code);
        code.extend_from_slice(&[0x49, 0x89, 0xC4]); // mov r12, rax
        code.extend_from_slice(&[0x48, 0x89, 0xC7]); // mov rdi, rax
        code.extend_from_slice(&[0xBE, 0x00, 0x10, 0x00, 0x00]); // mov esi, 4096
        code.extend_from_slice(&[0x31, 0xD2]); // xor edx, edx (PROT_NONE)
        code.extend_from_slice(&[0xB8, 0x0A, 0x00, 0x00, 0x00]); // mov eax, 10 (SYS_mprotect)
        code.extend_from_slice(&[0x0F, 0x05]);
        code.extend_from_slice(&[0x48, 0x3D, 0x01, 0xF0, 0xFF, 0xFF, 0x72, 0x0C]);
        emit_sys_exit(&mut code, AOT_EXIT_REGION_OOM as u32);

        emit_movabs_reg(&mut code, 0, stack_total_bytes);
        code.extend_from_slice(&[0x49, 0x01, 0xC4]); // add r12, rax
        code.extend_from_slice(&[0x4C, 0x89, 0xE4]); // mov rsp, r12
        code.extend_from_slice(&[0x48, 0x89, 0xE5]); // mov rbp, rsp

        if has_any_region {
            emit_sc_region_startup(&mut code, &regions, true)?;
        }

        // 3. Initialize packed remaining depth + stack-unit budget register `r14` AFTER region startup.
        // Layout: bits 22..63 = remaining call depth, bit 21 = unit underflow guard, bits 0..20 = remaining stack units.
        let init_budget = (u64::from(MAX_CALL_DEPTH) << 22) | u64::from(MAX_STACK_UNITS);
        emit_movabs_reg(&mut code, REG_R14, init_budget);

        // 4. Call entry function (`func_count - 1`).
        let entry_func_id = func_count
            .checked_sub(1)
            .ok_or(AotError::UnsupportedGraph("empty function table"))?;
        code.extend_from_slice(&[0xE8, 0x00, 0x00, 0x00, 0x00]);
        call_patches.push((code.len() - 4, entry_func_id));

        if let Some(out_slot) = output_region_slot {
            emit_sc_output_region_epilogue(&mut code, out_slot)?;
        } else {
            emit_store_r15_reg(&mut code, 0, SC_SCRATCH_IO_OFF);
            code.extend_from_slice(&[0x45, 0x31, 0xE4]); // xor r12d, r12d
            let write_loop_pos = code.len();
            code.extend_from_slice(&[0x49, 0x83, 0xFC, 0x08]);
            code.extend_from_slice(&[0x0F, 0x83, 0x00, 0x00, 0x00, 0x00]);
            let done_patch = code.len() - 4;
            code.extend_from_slice(&[0xBF, 0x01, 0x00, 0x00, 0x00]);
            code.extend_from_slice(&[0x49, 0x8D, 0xB7]);
            code.extend_from_slice(&SC_SCRATCH_IO_OFF.to_le_bytes());
            code.extend_from_slice(&[0x4C, 0x01, 0xE6]);
            code.extend_from_slice(&[0xBA, 0x08, 0x00, 0x00, 0x00]);
            code.extend_from_slice(&[0x4C, 0x29, 0xE2]);
            code.extend_from_slice(&[0xB8, 0x01, 0x00, 0x00, 0x00]);
            code.extend_from_slice(&[0x0F, 0x05]);
            code.extend_from_slice(&[0x48, 0x83, 0xF8, 0xFC]);
            code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
            let eintr_patch = code.len() - 4;
            patch_rel32(&mut code, eintr_patch, write_loop_pos)?;
            code.extend_from_slice(&[0x48, 0x85, 0xC0, 0x7F, 0x0C]);
            emit_sys_exit(&mut code, AOT_EXIT_REGION_OOM as u32);
            code.extend_from_slice(&[0x49, 0x01, 0xC4]);
            code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]);
            let back_patch = code.len() - 4;
            patch_rel32(&mut code, back_patch, write_loop_pos)?;
            let done_pos = code.len();
            patch_rel32(&mut code, done_patch, done_pos)?;
            emit_sys_exit(&mut code, 0);
        }

        // Shared out-of-line fail-closed stack/call-depth overflow stub (`exit(75)`).
        let stack_overflow_stub = code.len();
        emit_sys_exit(&mut code, AOT_EXIT_STACK_OVERFLOW as u32);

        // 5. Emit each `Func..End` unit independently with its own O(N) pre-scan and linear-scan allocation.
        let mut cursor = 0usize;
        while cursor < node_count {
            let Some(hdr) = program.nodes.get(cursor) else {
                break;
            };
            if hdr.opcode != Opcode::Func {
                return Err(AotError::UnsupportedGraph(
                    "expected Func header at top level of function program",
                ));
            }
            let func_id = hdr.in_a as usize;
            let sig = func_sigs
                .get(func_id)
                .copied()
                .flatten()
                .ok_or(AotError::UnsupportedGraph("missing FuncSig in AOT"))?;
            // Find matching End of this Func scope in one scan.
            let mut depth = 0usize;
            let mut fn_end = cursor;
            for (rel, nf) in program
                .nodes
                .get(cursor..)
                .unwrap_or(&[])
                .iter()
                .enumerate()
            {
                match nf.opcode {
                    Opcode::Func | Opcode::Block | Opcode::Loop | Opcode::If => {
                        depth += 1;
                    }
                    Opcode::End => {
                        depth = depth.saturating_sub(1);
                        if depth == 0 {
                            fn_end = cursor + rel;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if depth != 0 {
                return Err(AotError::UnsupportedGraph("unclosed Func scope in AOT"));
            }
            // Align each function entry to a 16-byte boundary (never executed padding between functions).
            while (code.len() & 15) != 0 {
                code.push(0x90);
            }
            if let Some(slot) = func_offsets.get_mut(func_id) {
                *slot = Some(code.len());
            }
            let unit_slice = program
                .nodes
                .get(cursor..=fn_end)
                .ok_or(AotError::UnsupportedGraph("Func slice out of bounds"))?;
            sc_emit_unit(
                &mut code,
                program,
                unit_slice,
                &regions,
                &func_sigs,
                Some(sig),
                output_region_slot,
                Some(stack_overflow_stub),
                &mut call_patches,
                &mut scratch,
            )?;
            cursor = fn_end + 1;
        }

        for (patch_pos, target_id) in call_patches {
            let target_offset = func_offsets
                .get(target_id)
                .copied()
                .flatten()
                .ok_or(AotError::UnsupportedGraph("unresolved Call target"))?;
            patch_rel32(&mut code, patch_pos, target_offset)?;
        }
        return Ok(code);
    }

    sc_emit_unit(
        &mut code,
        program,
        &program.nodes,
        &regions,
        &func_sigs,
        None,
        output_region_slot,
        None,
        &mut call_patches,
        &mut scratch,
    )?;
    Ok(code)
}

// =========================================================================
// Rul AOT x86_64 Structured Scalar Register Contract (`v0-5-tc-core-4`)
// =========================================================================
// - Fixed Non-Allocatable Registers:
//   * `rsp`: native stack pointer (private `8 * MAX_STACK_UNITS + 4096` mmap stack with bottom
//     `PROT_NONE` guard page in function programs).
//   * `rbp`: activation frame base pointer for unpinned locals, x87 scratch, and spill slots
//     (omitted in zero-spill functions whose active locals are all pinned to registers).
//   * `r15`: 4096-byte program-global descriptor page (`[0..2048)` = 64 x 32B region descriptors,
//     `[2064..2072)` = scalar stdout / EOF probe scratch, `[2080..2136)` = callee-save spill area
//     used only around `RegionNew`/`RegionGrow`/`RegionFree` syscalls).
//   * `r14`: in function programs, packed remaining stack budget
//     `((MAX_CALL_DEPTH - live_call_depth) << 22) | (MAX_STACK_UNITS - live_stack_units)`.
//     Each `Func` prologue subtracts the 32-bit immediate `(1 << 22) | func_frame_units(node_count)`
//     and fails closed with exit `75` if bit 63 (call-depth underflow) or bit 21 (stack-unit
//     underflow) is set; each `Func` epilogue adds the same 32-bit delta back.
//   * Functions starting with a pure early-return base-case `If` guard evaluate that guard before
//     pushing callee-saved registers or allocating a frame, checking the packed `r14` budget via a
//     non-mutating copy in `rax` into a shared out-of-line `exit(75)` stub; the remaining body is
//     emitted into a reusable scratch buffer so the prologue has zero NOPs.
// - Scratch Registers (clobberable within any single opcode emission):
//   * GPR: `rax` (primary ALU scratch, function return value, `div`/`idiv` dividend/quotient, and
//     transient home for immediately-consumed single-use commutative integer spills),
//     `r11` (secondary scratch, region base pointer, syscall clobber).
//   * SIMD: `xmm0`, `xmm1` (FP scratch and FMA/cast/comparison temporaries).
// - Allocatable GPR Pool (10 registers for `I64`, `U8`, `BoolMask`, and loop fuel):
//   * Caller-saved (4): `rdi` (arg0), `rsi` (arg1), `rdx` (arg2), `rcx` (arg3).
//     Preferred first for non-call-spanning intervals so leaf functions avoid prologue pushes.
//     `rdx` (`Div`/`Rem`) and `rcx` (dynamic shift) are preserved on the stack only if live.
//   * Callee-saved (6): `rbx`, `r12`, `r13`, `r8`, `r9`, `r10`.
//     Intervals spanning any `Call` / `RegionNew` / `RegionGrow` / `RegionFree` barrier take a
//     callee-saved GPR or a frame spill slot. Each `Func` saves only the subset it assigns.
// - Allocatable XMM Pool (14 registers for `F64`):
//   * `xmm2 ..= xmm15` (all caller-saved across `Call` barriers; `F64` values spanning a barrier
//     take an 8-byte frame spill slot).
// - Frame Layout & Stack Budget Guarantee:
//   * Below `rbp`: `[unpinned used locals] [optional 8B x87 scratch] [optional 8B stdout slot]
//     [reused 8B spill slots]`.
//   * Native frame size per activation (`ret_addr + saved_callee_gprs + saved_rbp + frame_bytes`)
//     is strictly `<= 8 * func_frame_units(node_count) = 8 * (72 + node_count)` bytes.
//   * Every SSA scalar and local has exactly one home for its lifetime (no interval splitting or
//     join reconciliation). `ConstI64`/`ConstF64`/`Cast(ConstI64, U8)` materialize on demand;
//     adjacent integer `Cmp*` (and `MaskNot(Cmp*)`) fuse into `cmp` + `jcc`; `BitAnd 0xFFFF_FFFF`
//     lowers to a 32-bit zero-extending move; dying operands reuse their home in two-operand ops.
//
// =========================================================================
// Rul AOT x86_64 Columnar AVX2 Register Contract (`RegisterContract`)
// =========================================================================
// - Fixed Frame / Loop Registers:
//   * `rbp`: 32-byte aligned stack frame base pointer
//   * `r12`: read-only `mmap` arena file base pointer
//   * `rcx`: loop element index (`0 .. final_domain_end`)
//   * `rbx`: `final_domain_end` (`loop_col_len`)
//   * `r11`: active element/vector step counter for 8-lane deterministic reductions
// - Scratch Registers (clobberable within any single opcode stencil):
//   * GPR: `rax`, `rdx`, `r8` (plus `rdi`, `rsi`, `r9`, `r10`, `r13`, `r14`, `r15` in syscalls)
//   * SIMD: `ymm0`/`xmm0`, `ymm1`/`xmm1`
// - Primary Reduction Accumulators (in `5a` AVX2 loop when `reductions.len() == 1`):
//   * `ymm14`: lanes `0..4` (`acc[0..4]`)
//   * `ymm15`: lanes `4..8` (`acc[4..8]`)
// - SSA Column Register Pool (in `5a` AVX2 loop):
//   * `ymm2 .. ymm13` (12 x 256-bit `4 x f64` registers, `free_ymm_mask = 0x3FFC`)
//   * Allocated in a single linear pass via `last_use[node_idx]`; spilled to `[rbp + disp]`
//     only if > 12 column values are simultaneously live.
//   * LICM `ConstF64` nodes remain pre-broadcast in `[rbp + disp]` as memory operands.
#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "self-contained linear x86_64 AVX2 + scalar stencil emitter"
)]
fn emit_x86_64_machine_code(
    program: &RulProgram,
    live: &[bool],
    phases: &[AotNodePhase],
    col_metas: &[AotColMeta],
    eval_in_loop: &[bool],
    reductions: &[AotReductionSpec],
    post_reduce_nodes: &[usize],
    domain_start: u64,
    domain_end: u64,
    filter_mask_node: Option<usize>,
    avx2_vectorized: bool,
    arena_byte_offsets: &[u64],
    total_arena_bytes: u64,
) -> Result<Vec<u8>, AotError> {
    let mut code: Vec<u8> = Vec::with_capacity(1024);
    let node_count = program.nodes.len();

    // 1. Entry prologue: capture `argc` (`r14`) and `argv[1]` (`r13`), align stack to 32B, allocate frame.
    // mov r14, [rsp]
    code.extend_from_slice(&[0x4C, 0x8B, 0x34, 0x24]);
    // mov r13, [rsp + 16]
    code.extend_from_slice(&[0x4C, 0x8B, 0x6C, 0x24, 0x10]);
    // and rsp, -32 (32-byte alignment for 256-bit AVX2 YMM stack slots)
    code.extend_from_slice(&[0x48, 0x83, 0xE4, 0xE0]);
    // mov rbp, rsp
    code.extend_from_slice(&[0x48, 0x89, 0xE5]);

    // Stack frame layout below `rbp` (all 32-byte aligned):
    // [rbp - 64 .. rbp - 32]                               : 32-byte (4 x f64) IEEE-754 sign-clear mask (`0x7FFF_FFFF_FFFF_FFFF`) for `Opcode::Abs`
    // [rbp - 72]                                           : 8-byte stdout staging slot
    // [rbp - 128 - idx * 32]                               : 32-byte (256-bit YMM / 64-bit XMM) SSA slot for each node `idx`
    // [rbp - 128 - node_count * 32 - (red_k + 1) * 64 ..]  : 64-byte (8 x f64) accumulator lanes per reduction `red_k`
    // [below reduction accumulators]                       : 32-byte AVX2 transcendental math constant & scratch slots (allocated only when `needs_vec_math`)
    let has_live_ln = program.nodes.iter().enumerate().any(|(idx, frame)| {
        eval_in_loop.get(idx).copied().unwrap_or(false)
            && phases.get(idx).copied() == Some(AotNodePhase::Column)
            && frame.opcode == Opcode::Ln
    });
    let has_live_exp = program.nodes.iter().enumerate().any(|(idx, frame)| {
        eval_in_loop.get(idx).copied().unwrap_or(false)
            && phases.get(idx).copied() == Some(AotNodePhase::Column)
            && frame.opcode == Opcode::Exp
    });
    let needs_vec_math = avx2_vectorized && (has_live_ln || has_live_exp);
    let math_frame_bytes = if needs_vec_math {
        MATH_TOTAL_SLOTS.saturating_mul(32)
    } else {
        0
    };
    let raw_frame = 128usize
        .saturating_add(node_count.saturating_mul(32))
        .saturating_add(reductions.len().saturating_mul(64))
        .saturating_add(math_frame_bytes);
    let frame_bytes = u32::try_from(raw_frame.div_ceil(32) * 32).unwrap_or(16384);
    // sub rsp, frame_bytes
    code.extend_from_slice(&[0x48, 0x81, 0xEC]);
    code.extend_from_slice(&frame_bytes.to_le_bytes());

    // 2. Zero-copy `mmap` of the arena data file (`argv[1]`) if `total_arena_bytes > 0`.
    if total_arena_bytes > 0 {
        // cmp r14, 2
        code.extend_from_slice(&[0x49, 0x83, 0xFE, 0x02]);
        // jae +12 (skip exit(64))
        code.extend_from_slice(&[0x73, 0x0C]);
        emit_sys_exit(&mut code, 64);

        // SYS_open(argv[1], O_RDONLY=0, mode=0) -> rax
        // mov rdi, r13
        code.extend_from_slice(&[0x4C, 0x89, 0xEF]);
        // xor esi, esi
        code.extend_from_slice(&[0x31, 0xF6]);
        // xor edx, edx
        code.extend_from_slice(&[0x31, 0xD2]);
        // mov eax, 2
        code.extend_from_slice(&[0xB8, 0x02, 0x00, 0x00, 0x00]);
        // syscall
        code.extend_from_slice(&[0x0F, 0x05]);
        // test rax, rax
        code.extend_from_slice(&[0x48, 0x85, 0xC0]);
        // jns +12 (skip exit(66))
        code.extend_from_slice(&[0x79, 0x0C]);
        emit_sys_exit(&mut code, 66);

        // Save `fd` in `r15`: mov r15, rax
        code.extend_from_slice(&[0x49, 0x89, 0xC7]);

        // Verify file size >= `total_arena_bytes` via `SYS_lseek(fd=r15, offset=0, whence=SEEK_END=2)`
        // before `SYS_mmap` so a truncated/short file exits cleanly with code 66 instead of `SIGBUS`.
        // Note: A residual TOCTOU window remains only if an external process concurrently truncates
        // the backing file after `SYS_lseek` while the `MAP_PRIVATE` mapping is actively faulted in.
        // mov rdi, r15
        code.extend_from_slice(&[0x4C, 0x89, 0xFF]);
        // xor esi, esi
        code.extend_from_slice(&[0x31, 0xF6]);
        // mov edx, 2 (SEEK_END = 2)
        code.extend_from_slice(&[0xBA, 0x02, 0x00, 0x00, 0x00]);
        // mov eax, 8 (SYS_lseek = 8)
        code.extend_from_slice(&[0xB8, 0x08, 0x00, 0x00, 0x00]);
        // syscall
        code.extend_from_slice(&[0x0F, 0x05]);
        // test rax, rax
        code.extend_from_slice(&[0x48, 0x85, 0xC0]);
        // js +15 (jump to exit(66) on negative error return)
        code.extend_from_slice(&[0x78, 0x0F]);
        // movabs rcx, total_arena_bytes
        code.extend_from_slice(&[0x48, 0xB9]);
        code.extend_from_slice(&total_arena_bytes.to_le_bytes());
        // cmp rax, rcx
        code.extend_from_slice(&[0x48, 0x39, 0xC8]);
        // jae +12 (skip exit(66))
        code.extend_from_slice(&[0x73, 0x0C]);
        emit_sys_exit(&mut code, 66);

        // SYS_mmap(NULL, total_arena_bytes, PROT_READ=1, MAP_PRIVATE=2, fd=r15, offset=0)
        // xor edi, edi
        code.extend_from_slice(&[0x31, 0xFF]);
        // movabs rsi, total_arena_bytes
        code.extend_from_slice(&[0x48, 0xBE]);
        code.extend_from_slice(&total_arena_bytes.to_le_bytes());
        // mov edx, 1
        code.extend_from_slice(&[0xBA, 0x01, 0x00, 0x00, 0x00]);
        // mov r10d, 2
        code.extend_from_slice(&[0x41, 0xBA, 0x02, 0x00, 0x00, 0x00]);
        // mov r8, r15
        code.extend_from_slice(&[0x4D, 0x89, 0xF8]);
        // xor r9d, r9d
        code.extend_from_slice(&[0x45, 0x31, 0xC9]);
        // mov eax, 9
        code.extend_from_slice(&[0xB8, 0x09, 0x00, 0x00, 0x00]);
        // syscall
        code.extend_from_slice(&[0x0F, 0x05]);
        // cmp rax, -4096
        code.extend_from_slice(&[0x48, 0x3D, 0x00, 0xF0, 0xFF, 0xFF]);
        // jb +12 (skip exit(66))
        code.extend_from_slice(&[0x72, 0x0C]);
        emit_sys_exit(&mut code, 66);

        // Save `mmap` base pointer in `r12`: mov r12, rax
        code.extend_from_slice(&[0x49, 0x89, 0xC4]);

        // Close `fd` (SYS_close = 3): mov rdi, r15; mov eax, 3; syscall
        code.extend_from_slice(&[0x4C, 0x89, 0xFF]);
        code.extend_from_slice(&[0xB8, 0x03, 0x00, 0x00, 0x00]);
        code.extend_from_slice(&[0x0F, 0x05]);
    }

    // 3. Loop-Invariant Code Motion (LICM): pre-broadcast all `ConstF64` nodes into all 4 lanes
    //    of their 32-byte YMM slot `[rbp + disp + 0..32]` before the loop!
    let has_live_abs =
        program.nodes.iter().enumerate().any(|(idx, frame)| {
            live.get(idx).copied().unwrap_or(false) && frame.opcode == Opcode::Abs
        });
    if has_live_abs {
        // Pre-broadcast IEEE-754 sign-clear mask `0x7FFF_FFFF_FFFF_FFFF` into `[rbp - 64 .. rbp - 32]`.
        let abs_mask_bits = 0x7FFF_FFFF_FFFF_FFFF_u64;
        code.extend_from_slice(&[0x48, 0xB8]);
        code.extend_from_slice(&abs_mask_bits.to_le_bytes());
        for lane_off in [0i32, 8, 16, 24] {
            let d = (-64i32).saturating_add(lane_off);
            code.extend_from_slice(&[0x48, 0x89, 0x85]);
            code.extend_from_slice(&d.to_le_bytes());
        }
    }
    for (idx, frame) in program.nodes.iter().enumerate() {
        if !live.get(idx).copied().unwrap_or(false) {
            continue;
        }
        if frame.opcode == Opcode::ConstF64 {
            let bits = frame.decode_const_f64().to_bits();
            let disp = node_disp(idx);
            // movabs rax, bits
            code.extend_from_slice(&[0x48, 0xB8]);
            code.extend_from_slice(&bits.to_le_bytes());
            for lane_off in [0i32, 8, 16, 24] {
                let d = disp.saturating_add(lane_off);
                // mov [rbp + d], rax
                code.extend_from_slice(&[0x48, 0x89, 0x85]);
                code.extend_from_slice(&d.to_le_bytes());
            }
        }
    }
    if needs_vec_math {
        for &(slot_idx, bits) in MATH_CONST_TABLE {
            let disp = math_slot_disp(node_count, reductions.len(), slot_idx);
            if bits == 0 {
                // xor eax, eax
                code.extend_from_slice(&[0x31, 0xC0]);
            } else {
                // movabs rax, bits
                code.extend_from_slice(&[0x48, 0xB8]);
                code.extend_from_slice(&bits.to_le_bytes());
            }
            for lane_off in [0i32, 8, 16, 24] {
                let d = disp.saturating_add(lane_off);
                // mov [rbp + d], rax
                code.extend_from_slice(&[0x48, 0x89, 0x85]);
                code.extend_from_slice(&d.to_le_bytes());
            }
        }
    }

    // 4. Initialize reduction accumulators across all `reductions`.
    // xor r11d, r11d (active element/vector counter for Sum reductions)
    code.extend_from_slice(&[0x45, 0x31, 0xDB]);
    for (red_k, spec) in reductions.iter().enumerate() {
        let init_bits = match spec.reduction {
            AotReduction::Sum => 0.0_f64.to_bits(),
            AotReduction::Max => f64::NEG_INFINITY.to_bits(),
            AotReduction::Min => f64::INFINITY.to_bits(),
        };
        if init_bits == 0 {
            // xor eax, eax
            code.extend_from_slice(&[0x31, 0xC0]);
        } else {
            // movabs rax, init_bits
            code.extend_from_slice(&[0x48, 0xB8]);
            code.extend_from_slice(&init_bits.to_le_bytes());
        }
        for lane in 0..8i32 {
            let acc_disp = red_acc_disp(node_count, red_k, lane);
            // mov [rbp + acc_disp], rax
            code.extend_from_slice(&[0x48, 0x89, 0x85]);
            code.extend_from_slice(&acc_disp.to_le_bytes());
        }
    }

    // 5. Initialize domain bounds: `rcx` = `domain_start`, `rbx` = `domain_end`.
    // movabs rcx, domain_start
    code.extend_from_slice(&[0x48, 0xB9]);
    code.extend_from_slice(&domain_start.to_le_bytes());
    // movabs rbx, domain_end
    code.extend_from_slice(&[0x48, 0xBB]);
    code.extend_from_slice(&domain_end.to_le_bytes());

    // 5a. Primary 256-Bit AVX2 `YMM` Vector Loop (`4 x f64` = `32 bytes` per step,
    //     with single-pass O(N) `last_use` register allocation across `ymm2..ymm13`
    //     and primary reduction accumulators pinned in `ymm14` & `ymm15`).
    if avx2_vectorized {
        let mut last_use = vec![0usize; node_count];
        for (idx, frame) in program.nodes.iter().enumerate() {
            if !eval_in_loop.get(idx).copied().unwrap_or(false)
                || phases.get(idx).copied() != Some(AotNodePhase::Column)
            {
                continue;
            }
            match frame.opcode {
                Opcode::ArenaInput | Opcode::ConstF64 | Opcode::ArenaSlice => {}
                Opcode::ArenaMutate => {
                    if let Some(lu) = last_use.get_mut(frame.in_b as usize) {
                        *lu = idx;
                    }
                }
                Opcode::Abs
                | Opcode::Sqrt
                | Opcode::Floor
                | Opcode::Ceil
                | Opcode::Ln
                | Opcode::Exp => {
                    if let Some(lu) = last_use.get_mut(frame.in_a as usize) {
                        *lu = idx;
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
                | Opcode::FilterCompact => {
                    if let Some(lu) = last_use.get_mut(frame.in_a as usize) {
                        *lu = idx;
                    }
                    if let Some(lu) = last_use.get_mut(frame.in_b as usize) {
                        *lu = idx;
                    }
                }
                Opcode::Fma => {
                    let (in_b, in_c) = frame.decode_fma_bc();
                    if let Some(lu) = last_use.get_mut(frame.in_a as usize) {
                        *lu = idx;
                    }
                    if let Some(lu) = last_use.get_mut(in_b as usize) {
                        *lu = idx;
                    }
                    if let Some(lu) = last_use.get_mut(in_c as usize) {
                        *lu = idx;
                    }
                }
                Opcode::Gather
                | Opcode::ReduceSum
                | Opcode::ReduceMax
                | Opcode::ReduceMin
                | Opcode::RingEnqueue
                | Opcode::AssertContract
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
                | Opcode::CallArg => {}
            }
        }
        for spec in reductions {
            if let Some(lu) = last_use.get_mut(spec.reduce_input_idx) {
                *lu = node_count;
            }
        }

        // Pin primary reduction accumulator in `ymm14` (`acc[0..4]`) and `ymm15` (`acc[4..8]`)
        // before entering the AVX2 vector loop when `reductions.len() == 1`.
        if reductions.len() == 1 {
            emit_load_ymm_rbp(&mut code, 14, red_acc_disp(node_count, 0, 0));
            emit_load_ymm_rbp(&mut code, 15, red_acc_disp(node_count, 0, 4));
        }

        let mut free_ymm_mask: u16 = 0x3FFC; // bits 2..=13 (`ymm2..ymm13`)
        let mut node_ymm: Vec<Option<u8>> = vec![None; node_count];

        let avx2_loop_top = code.len();
        // lea rax, [rcx + 4]
        code.extend_from_slice(&[0x48, 0x8D, 0x41, 0x04]);
        // cmp rax, rbx
        code.extend_from_slice(&[0x48, 0x39, 0xD8]);
        // ja avx2_loop_exit (0F 87 <rel32>)
        code.extend_from_slice(&[0x0F, 0x87, 0x00, 0x00, 0x00, 0x00]);
        let ja_avx2_exit_patch_pos = code.len() - 4;

        for (idx, frame) in program.nodes.iter().enumerate() {
            if !eval_in_loop.get(idx).copied().unwrap_or(false)
                || phases.get(idx).copied() != Some(AotNodePhase::Column)
            {
                continue;
            }
            let a = frame.in_a as usize;
            let b = frame.in_b as usize;
            let disp = node_disp(idx);
            let disp_a = node_disp(a);
            let disp_b = node_disp(b);

            match frame.opcode {
                Opcode::ConstF64 => {}
                Opcode::ArenaInput | Opcode::ArenaSlice => {
                    let meta = col_metas.get(idx).copied().unwrap_or(AotColMeta {
                        origin_arena: None,
                        slice_offset: 0,
                        col_len: 0,
                    });
                    let orig_arena = meta.origin_arena.unwrap_or(a);
                    let arena_off = arena_byte_offsets
                        .get(orig_arena)
                        .copied()
                        .unwrap_or(0)
                        .saturating_add(meta.slice_offset.saturating_mul(8));

                    let dst_reg = alloc_ymm_slot(idx, &mut free_ymm_mask, &mut node_ymm);
                    // mov rax, r12
                    code.extend_from_slice(&[0x4C, 0x89, 0xE0]);
                    if arena_off > 0 {
                        // movabs rdx, arena_off; add rax, rdx
                        code.extend_from_slice(&[0x48, 0xBA]);
                        code.extend_from_slice(&arena_off.to_le_bytes());
                        code.extend_from_slice(&[0x48, 0x01, 0xD0]);
                    }
                    emit_load_ymm_rax_rcx8(&mut code, dst_reg);
                    if dst_reg == 0 {
                        emit_store_ymm_rbp(&mut code, 0, disp);
                    }
                }
                Opcode::ArenaMutate | Opcode::FilterCompact => {
                    let reg_b = if let Some(rb) = node_ymm.get(b).copied().flatten() {
                        rb
                    } else {
                        emit_load_ymm_rbp(&mut code, 0, disp_b);
                        0
                    };
                    free_if_dying(b, idx, &last_use, &mut free_ymm_mask, &mut node_ymm);
                    let dst_reg = alloc_ymm_slot(idx, &mut free_ymm_mask, &mut node_ymm);
                    if dst_reg == 0 {
                        emit_store_ymm_rbp(&mut code, reg_b, disp);
                    } else if dst_reg != reg_b {
                        // vmovapd ymm_dst, ymm_reg_b
                        emit_vex_ymm_rr(&mut code, 0x28, dst_reg, 0, reg_b);
                    }
                }
                Opcode::Add
                | Opcode::Sub
                | Opcode::Mul
                | Opcode::Div
                | Opcode::Min
                | Opcode::Max => {
                    let vex_op = match frame.opcode {
                        Opcode::Add => 0x58,
                        Opcode::Sub => 0x5C,
                        Opcode::Mul => 0x59,
                        Opcode::Div => 0x5E,
                        Opcode::Min => 0x5D,
                        _ => 0x5F,
                    };
                    let reg_a = if let Some(ra) = node_ymm.get(a).copied().flatten() {
                        ra
                    } else {
                        emit_load_ymm_rbp(&mut code, 0, disp_a);
                        0
                    };
                    let opt_rb = node_ymm.get(b).copied().flatten();
                    free_if_dying(a, idx, &last_use, &mut free_ymm_mask, &mut node_ymm);
                    if b != a {
                        free_if_dying(b, idx, &last_use, &mut free_ymm_mask, &mut node_ymm);
                    }
                    let dst_reg = alloc_ymm_slot(idx, &mut free_ymm_mask, &mut node_ymm);
                    if let Some(rb) = opt_rb {
                        emit_vex_ymm_rr(&mut code, vex_op, dst_reg, reg_a, rb);
                    } else {
                        emit_vex_ymm_rm_rbp(&mut code, vex_op, dst_reg, reg_a, disp_b);
                    }
                    if dst_reg == 0 {
                        emit_store_ymm_rbp(&mut code, 0, disp);
                    }
                }
                Opcode::Fma => {
                    let (in_b, in_c) = frame.decode_fma_bc();
                    let b_idx = in_b as usize;
                    let c_idx = in_c as usize;
                    let disp_b_fma = node_disp(b_idx);
                    let disp_c_fma = node_disp(c_idx);

                    let reg_a = if let Some(ra) = node_ymm.get(a).copied().flatten() {
                        ra
                    } else {
                        emit_load_ymm_rbp(&mut code, 0, disp_a);
                        0
                    };
                    let reg_b = if let Some(rb) = node_ymm.get(b_idx).copied().flatten() {
                        rb
                    } else if b_idx == a {
                        reg_a
                    } else {
                        emit_load_ymm_rbp(&mut code, 1, disp_b_fma);
                        1
                    };
                    let opt_rc = if c_idx == a {
                        Some(reg_a)
                    } else if c_idx == b_idx {
                        Some(reg_b)
                    } else {
                        node_ymm.get(c_idx).copied().flatten()
                    };

                    free_if_dying(a, idx, &last_use, &mut free_ymm_mask, &mut node_ymm);
                    if b_idx != a {
                        free_if_dying(b_idx, idx, &last_use, &mut free_ymm_mask, &mut node_ymm);
                    }
                    if c_idx != a && c_idx != b_idx {
                        free_if_dying(c_idx, idx, &last_use, &mut free_ymm_mask, &mut node_ymm);
                    }
                    let dst_reg = alloc_ymm_slot(idx, &mut free_ymm_mask, &mut node_ymm);

                    if dst_reg == reg_a {
                        if let Some(rc) = opt_rc {
                            emit_vex_fma_ymm_rr(&mut code, 0xA8, dst_reg, reg_b, rc);
                        } else {
                            emit_vex_fma_ymm_rm_rbp(&mut code, 0xA8, dst_reg, reg_b, disp_c_fma);
                        }
                    } else if dst_reg == reg_b {
                        if let Some(rc) = opt_rc {
                            emit_vex_fma_ymm_rr(&mut code, 0xA8, dst_reg, reg_a, rc);
                        } else {
                            emit_vex_fma_ymm_rm_rbp(&mut code, 0xA8, dst_reg, reg_a, disp_c_fma);
                        }
                    } else if Some(dst_reg) == opt_rc {
                        emit_vex_fma_ymm_rr(&mut code, 0xB8, dst_reg, reg_a, reg_b);
                    } else {
                        emit_vex_ymm_rr(&mut code, 0x28, dst_reg, 0, reg_a);
                        if let Some(rc) = opt_rc {
                            emit_vex_fma_ymm_rr(&mut code, 0xA8, dst_reg, reg_b, rc);
                        } else {
                            emit_vex_fma_ymm_rm_rbp(&mut code, 0xA8, dst_reg, reg_b, disp_c_fma);
                        }
                    }
                    if dst_reg == 0 {
                        emit_store_ymm_rbp(&mut code, 0, disp);
                    }
                }
                Opcode::CmpGt | Opcode::CmpLt | Opcode::CmpEq => {
                    let (lhs, disp_lhs, rhs, disp_rhs, imm8) = match frame.opcode {
                        Opcode::CmpGt => (b, disp_b, a, disp_a, 0x01u8),
                        Opcode::CmpLt => (a, disp_a, b, disp_b, 0x01u8),
                        _ => (a, disp_a, b, disp_b, 0x00u8),
                    };
                    let reg_lhs = if let Some(rl) = node_ymm.get(lhs).copied().flatten() {
                        rl
                    } else {
                        emit_load_ymm_rbp(&mut code, 0, disp_lhs);
                        0
                    };
                    let reg_rhs = if let Some(rr) = node_ymm.get(rhs).copied().flatten() {
                        rr
                    } else {
                        emit_load_ymm_rbp(&mut code, 1, disp_rhs);
                        1
                    };
                    free_if_dying(a, idx, &last_use, &mut free_ymm_mask, &mut node_ymm);
                    if b != a {
                        free_if_dying(b, idx, &last_use, &mut free_ymm_mask, &mut node_ymm);
                    }
                    let dst_reg = alloc_ymm_slot(idx, &mut free_ymm_mask, &mut node_ymm);
                    emit_vex_ymm_rr(&mut code, 0xC2, dst_reg, reg_lhs, reg_rhs);
                    code.push(imm8);
                    if dst_reg == 0 {
                        emit_store_ymm_rbp(&mut code, 0, disp);
                    }
                }
                Opcode::SelectZero | Opcode::MaskAnd | Opcode::MaskOr => {
                    let vex_op = if frame.opcode == Opcode::MaskOr {
                        0x56
                    } else {
                        0x54
                    };
                    let reg_a = if let Some(ra) = node_ymm.get(a).copied().flatten() {
                        ra
                    } else {
                        emit_load_ymm_rbp(&mut code, 0, disp_a);
                        0
                    };
                    let opt_rb = node_ymm.get(b).copied().flatten();
                    free_if_dying(a, idx, &last_use, &mut free_ymm_mask, &mut node_ymm);
                    if b != a {
                        free_if_dying(b, idx, &last_use, &mut free_ymm_mask, &mut node_ymm);
                    }
                    let dst_reg = alloc_ymm_slot(idx, &mut free_ymm_mask, &mut node_ymm);
                    if let Some(rb) = opt_rb {
                        emit_vex_ymm_rr(&mut code, vex_op, dst_reg, reg_a, rb);
                    } else {
                        emit_vex_ymm_rm_rbp(&mut code, vex_op, dst_reg, reg_a, disp_b);
                    }
                    if dst_reg == 0 {
                        emit_store_ymm_rbp(&mut code, 0, disp);
                    }
                }
                Opcode::Sqrt => {
                    let opt_ra = node_ymm.get(a).copied().flatten();
                    free_if_dying(a, idx, &last_use, &mut free_ymm_mask, &mut node_ymm);
                    let dst_reg = alloc_ymm_slot(idx, &mut free_ymm_mask, &mut node_ymm);
                    if let Some(ra) = opt_ra {
                        emit_vex_ymm_rr(&mut code, 0x51, dst_reg, 0, ra);
                    } else {
                        emit_vex_ymm_rm_rbp(&mut code, 0x51, dst_reg, 0, disp_a);
                    }
                    if dst_reg == 0 {
                        emit_store_ymm_rbp(&mut code, 0, disp);
                    }
                }
                Opcode::Floor | Opcode::Ceil => {
                    let imm8 = if frame.opcode == Opcode::Floor {
                        0x09u8
                    } else {
                        0x0Au8
                    };
                    let opt_ra = node_ymm.get(a).copied().flatten();
                    free_if_dying(a, idx, &last_use, &mut free_ymm_mask, &mut node_ymm);
                    let dst_reg = alloc_ymm_slot(idx, &mut free_ymm_mask, &mut node_ymm);
                    if let Some(ra) = opt_ra {
                        emit_vex_roundpd_rr(&mut code, dst_reg, ra, imm8);
                    } else {
                        emit_vex_roundpd_rm_rbp(&mut code, dst_reg, disp_a, imm8);
                    }
                    if dst_reg == 0 {
                        emit_store_ymm_rbp(&mut code, 0, disp);
                    }
                }
                Opcode::Abs => {
                    let reg_a = if let Some(ra) = node_ymm.get(a).copied().flatten() {
                        ra
                    } else {
                        emit_load_ymm_rbp(&mut code, 0, disp_a);
                        0
                    };
                    free_if_dying(a, idx, &last_use, &mut free_ymm_mask, &mut node_ymm);
                    let dst_reg = alloc_ymm_slot(idx, &mut free_ymm_mask, &mut node_ymm);
                    // vandpd ymm_dst, ymm_reg_a, ymmword ptr [rbp - 64]
                    emit_vex_ymm_rm_rbp(&mut code, 0x54, dst_reg, reg_a, -64);
                    if dst_reg == 0 {
                        emit_store_ymm_rbp(&mut code, 0, disp);
                    }
                }
                Opcode::Ln => {
                    if let Some(ra) = node_ymm.get(a).copied().flatten() {
                        emit_vex_ymm_rr(&mut code, 0x28, 0, 0, ra);
                    } else {
                        emit_load_ymm_rbp(&mut code, 0, disp_a);
                    }
                    free_if_dying(a, idx, &last_use, &mut free_ymm_mask, &mut node_ymm);
                    emit_avx2_ln_ymm(&mut code, node_count, reductions.len(), disp);
                    let dst_reg = alloc_ymm_slot(idx, &mut free_ymm_mask, &mut node_ymm);
                    if dst_reg == 0 {
                        emit_store_ymm_rbp(&mut code, 0, disp);
                    } else {
                        emit_vex_ymm_rr(&mut code, 0x28, dst_reg, 0, 0);
                    }
                }
                Opcode::Exp => {
                    if let Some(ra) = node_ymm.get(a).copied().flatten() {
                        emit_vex_ymm_rr(&mut code, 0x28, 0, 0, ra);
                    } else {
                        emit_load_ymm_rbp(&mut code, 0, disp_a);
                    }
                    free_if_dying(a, idx, &last_use, &mut free_ymm_mask, &mut node_ymm);
                    emit_avx2_exp_ymm(&mut code, node_count, reductions.len(), disp);
                    let dst_reg = alloc_ymm_slot(idx, &mut free_ymm_mask, &mut node_ymm);
                    if dst_reg == 0 {
                        emit_store_ymm_rbp(&mut code, 0, disp);
                    } else {
                        emit_vex_ymm_rr(&mut code, 0x28, dst_reg, 0, 0);
                    }
                }
                Opcode::Gather
                | Opcode::ReduceSum
                | Opcode::ReduceMax
                | Opcode::ReduceMin
                | Opcode::RingEnqueue
                | Opcode::AssertContract
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
                | Opcode::CallArg => {}
            }
        }

        if let [spec] = reductions {
            let reg_in = if let Some(r) = node_ymm.get(spec.reduce_input_idx).copied().flatten() {
                r
            } else {
                emit_load_ymm_rbp(&mut code, 0, node_disp(spec.reduce_input_idx));
                0
            };
            match spec.reduction {
                AotReduction::Sum => {
                    // test r11b, 1
                    code.extend_from_slice(&[0x41, 0xF6, 0xC3, 0x01]);
                    // jnz +7 (skip 5-byte vaddpd ymm14 + 2-byte jmp)
                    code.extend_from_slice(&[0x75, 0x07]);
                    emit_vex_ymm_rr(&mut code, 0x58, 14, 14, reg_in);
                    // jmp +5 (skip 5-byte vaddpd ymm15)
                    code.extend_from_slice(&[0xEB, 0x05]);
                    emit_vex_ymm_rr(&mut code, 0x58, 15, 15, reg_in);
                    // inc r11
                    code.extend_from_slice(&[0x49, 0xFF, 0xC3]);
                }
                AotReduction::Max => {
                    // vmaxpd ymm14, ymm_reg_in, ymm14 (preserves non-NaN accumulator in src2)
                    emit_vex_ymm_rr(&mut code, 0x5F, 14, reg_in, 14);
                    // inc r11
                    code.extend_from_slice(&[0x49, 0xFF, 0xC3]);
                }
                AotReduction::Min => {
                    // vminpd ymm14, ymm_reg_in, ymm14 (preserves non-NaN accumulator in src2)
                    emit_vex_ymm_rr(&mut code, 0x5D, 14, reg_in, 14);
                    // inc r11
                    code.extend_from_slice(&[0x49, 0xFF, 0xC3]);
                }
            }
        } else {
            // Multi-reduction fallback: flush any register-allocated reduction inputs to stack slots
            // and accumulate into `[rbp + acc0 + rdx]`.
            for spec in reductions {
                if let Some(r) = node_ymm.get(spec.reduce_input_idx).copied().flatten() {
                    emit_store_ymm_rbp(&mut code, r, node_disp(spec.reduce_input_idx));
                }
            }
            // mov rdx, r11; and edx, 1; shl rdx, 5; inc r11
            code.extend_from_slice(&[0x4C, 0x89, 0xDA]);
            code.extend_from_slice(&[0x83, 0xE2, 0x01]);
            code.extend_from_slice(&[0x48, 0xC1, 0xE2, 0x05]);
            code.extend_from_slice(&[0x49, 0xFF, 0xC3]);

            for (red_k, spec) in reductions.iter().enumerate() {
                let disp_red_in = node_disp(spec.reduce_input_idx);
                let acc0 = red_acc_disp(node_count, red_k, 0);
                match spec.reduction {
                    AotReduction::Sum => {
                        code.extend_from_slice(&[0xC5, 0xFD, 0x10, 0x84, 0x15]);
                        code.extend_from_slice(&acc0.to_le_bytes());
                        code.extend_from_slice(&[0xC5, 0xFD, 0x58, 0x85]);
                        code.extend_from_slice(&disp_red_in.to_le_bytes());
                        code.extend_from_slice(&[0xC5, 0xFD, 0x11, 0x84, 0x15]);
                        code.extend_from_slice(&acc0.to_le_bytes());
                    }
                    AotReduction::Max => {
                        emit_load_ymm0_rbp(&mut code, disp_red_in);
                        code.extend_from_slice(&[0xC5, 0xFD, 0x5F, 0x85]);
                        code.extend_from_slice(&acc0.to_le_bytes());
                        emit_store_ymm0_rbp(&mut code, acc0);
                    }
                    AotReduction::Min => {
                        emit_load_ymm0_rbp(&mut code, disp_red_in);
                        code.extend_from_slice(&[0xC5, 0xFD, 0x5D, 0x85]);
                        code.extend_from_slice(&acc0.to_le_bytes());
                        emit_store_ymm0_rbp(&mut code, acc0);
                    }
                }
            }
        }

        // add rcx, 4
        code.extend_from_slice(&[0x48, 0x83, 0xC1, 0x04]);
        // jmp avx2_loop_top (E9 <rel32>)
        code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]);
        let jmp_avx2_back = code.len() - 4;
        patch_rel32(&mut code, jmp_avx2_back, avx2_loop_top)?;

        let avx2_exit_pos = code.len();
        patch_rel32(&mut code, ja_avx2_exit_patch_pos, avx2_exit_pos)?;

        // Flush register-pinned primary reduction accumulators (`ymm14`, `ymm15`) back to stack
        // slots before `vzeroupper` and scalar cleanup.
        if reductions.len() == 1 {
            emit_store_ymm_rbp(&mut code, 14, red_acc_disp(node_count, 0, 0));
            if reductions.first().map(|s| s.reduction) == Some(AotReduction::Sum) {
                emit_store_ymm_rbp(&mut code, 15, red_acc_disp(node_count, 0, 4));
            }
        }

        // Clear upper 128 bits of all YMM registers (`vzeroupper`) before scalar tail/reduction.
        code.extend_from_slice(&[0xC5, 0xF8, 0x77]);

        // Convert `r11` from 4-element vector count to scalar element index (`shl r11, 2`)
        // so any scalar tail elements land in the exact matching `(r11 & 7)` lane.
        // shl r11, 2
        code.extend_from_slice(&[0x49, 0xC1, 0xE3, 0x02]);
    }

    // 5b. Scalar Cleanup / Indirect-Gather / FilterCompact Loop (`rcx` .. `rbx`).
    let loop_top = code.len();
    // cmp rcx, rbx
    code.extend_from_slice(&[0x48, 0x39, 0xD9]);
    // jae loop_exit (0F 83 <rel32>)
    code.extend_from_slice(&[0x0F, 0x83, 0x00, 0x00, 0x00, 0x00]);
    let jae_exit_patch_pos = code.len() - 4;

    // Emit native stencils for each loop-evaluated Column node.
    for (idx, frame) in program.nodes.iter().enumerate() {
        if !eval_in_loop.get(idx).copied().unwrap_or(false)
            || phases.get(idx).copied() != Some(AotNodePhase::Column)
        {
            continue;
        }
        let disp = node_disp(idx);
        let disp_a = node_disp(frame.in_a as usize);
        let disp_b = node_disp(frame.in_b as usize);

        match frame.opcode {
            Opcode::ConstF64 => {}
            Opcode::ArenaInput | Opcode::ArenaSlice => {
                let meta = col_metas.get(idx).copied().unwrap_or(AotColMeta {
                    origin_arena: None,
                    slice_offset: 0,
                    col_len: 0,
                });
                let orig_arena = meta.origin_arena.unwrap_or(frame.in_a as usize);
                let arena_off = arena_byte_offsets
                    .get(orig_arena)
                    .copied()
                    .unwrap_or(0)
                    .saturating_add(meta.slice_offset.saturating_mul(8));
                // mov rax, r12
                code.extend_from_slice(&[0x4C, 0x89, 0xE0]);
                if arena_off > 0 {
                    // movabs rdx, arena_off; add rax, rdx
                    code.extend_from_slice(&[0x48, 0xBA]);
                    code.extend_from_slice(&arena_off.to_le_bytes());
                    code.extend_from_slice(&[0x48, 0x01, 0xD0]);
                }
                // movsd xmm0, [rax + rcx * 8]
                code.extend_from_slice(&[0xF2, 0x0F, 0x10, 0x04, 0xC8]);
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::ArenaMutate | Opcode::FilterCompact => {
                emit_load_xmm0_rbp(&mut code, disp_b);
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::Add => {
                emit_load_xmm0_rbp(&mut code, disp_a);
                // addsd xmm0, [rbp + disp_b]
                code.extend_from_slice(&[0xF2, 0x0F, 0x58, 0x85]);
                code.extend_from_slice(&disp_b.to_le_bytes());
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::Sub => {
                emit_load_xmm0_rbp(&mut code, disp_a);
                // subsd xmm0, [rbp + disp_b]
                code.extend_from_slice(&[0xF2, 0x0F, 0x5C, 0x85]);
                code.extend_from_slice(&disp_b.to_le_bytes());
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::Mul => {
                emit_load_xmm0_rbp(&mut code, disp_a);
                // mulsd xmm0, [rbp + disp_b]
                code.extend_from_slice(&[0xF2, 0x0F, 0x59, 0x85]);
                code.extend_from_slice(&disp_b.to_le_bytes());
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::Div => {
                emit_load_xmm0_rbp(&mut code, disp_a);
                // divsd xmm0, [rbp + disp_b]
                code.extend_from_slice(&[0xF2, 0x0F, 0x5E, 0x85]);
                code.extend_from_slice(&disp_b.to_le_bytes());
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::Min => {
                emit_load_xmm0_rbp(&mut code, disp_a);
                // minsd xmm0, [rbp + disp_b]
                code.extend_from_slice(&[0xF2, 0x0F, 0x5D, 0x85]);
                code.extend_from_slice(&disp_b.to_le_bytes());
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::Max => {
                emit_load_xmm0_rbp(&mut code, disp_a);
                // maxsd xmm0, [rbp + disp_b]
                code.extend_from_slice(&[0xF2, 0x0F, 0x5F, 0x85]);
                code.extend_from_slice(&disp_b.to_le_bytes());
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::Fma => {
                let (in_b, in_c) = frame.decode_fma_bc();
                let disp_b_fma = node_disp(in_b as usize);
                let disp_c_fma = node_disp(in_c as usize);
                emit_load_xmm0_rbp(&mut code, disp_a);
                // movsd xmm1, [rbp + disp_b_fma]
                code.extend_from_slice(&[0xF2, 0x0F, 0x10, 0x8D]);
                code.extend_from_slice(&disp_b_fma.to_le_bytes());
                emit_vex_fma213_xmm_rm_rbp(&mut code, 0, 1, disp_c_fma);
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::CmpGt => {
                // Load `b` into xmm0 and `a` into xmm1; `cmpltsd xmm0, xmm1` computes `b < a` (`a > b`).
                emit_load_xmm0_rbp(&mut code, disp_b);
                // movsd xmm1, [rbp + disp_a]
                code.extend_from_slice(&[0xF2, 0x0F, 0x10, 0x8D]);
                code.extend_from_slice(&disp_a.to_le_bytes());
                // cmpltsd xmm0, xmm1 (F2 0F C2 C1 01)
                code.extend_from_slice(&[0xF2, 0x0F, 0xC2, 0xC1, 0x01]);
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::CmpLt => {
                emit_load_xmm0_rbp(&mut code, disp_a);
                // movsd xmm1, [rbp + disp_b]
                code.extend_from_slice(&[0xF2, 0x0F, 0x10, 0x8D]);
                code.extend_from_slice(&disp_b.to_le_bytes());
                // cmpltsd xmm0, xmm1 (F2 0F C2 C1 01)
                code.extend_from_slice(&[0xF2, 0x0F, 0xC2, 0xC1, 0x01]);
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::CmpEq => {
                emit_load_xmm0_rbp(&mut code, disp_a);
                // movsd xmm1, [rbp + disp_b]
                code.extend_from_slice(&[0xF2, 0x0F, 0x10, 0x8D]);
                code.extend_from_slice(&disp_b.to_le_bytes());
                // cmpeqsd xmm0, xmm1 (F2 0F C2 C1 00)
                code.extend_from_slice(&[0xF2, 0x0F, 0xC2, 0xC1, 0x00]);
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::SelectZero | Opcode::MaskAnd => {
                emit_load_xmm0_rbp(&mut code, disp_a);
                // movsd xmm1, [rbp + disp_b]
                code.extend_from_slice(&[0xF2, 0x0F, 0x10, 0x8D]);
                code.extend_from_slice(&disp_b.to_le_bytes());
                // andpd xmm0, xmm1 (branchless bitmask select / conjunction)
                code.extend_from_slice(&[0x66, 0x0F, 0x54, 0xC1]);
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::MaskOr => {
                emit_load_xmm0_rbp(&mut code, disp_a);
                // movsd xmm1, [rbp + disp_b]
                code.extend_from_slice(&[0xF2, 0x0F, 0x10, 0x8D]);
                code.extend_from_slice(&disp_b.to_le_bytes());
                // orpd xmm0, xmm1 (branchless bitmask disjunction)
                code.extend_from_slice(&[0x66, 0x0F, 0x56, 0xC1]);
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::Abs => {
                // mov rax, [rbp + disp_a]; btr rax, 63; mov [rbp + disp], rax
                code.extend_from_slice(&[0x48, 0x8B, 0x85]);
                code.extend_from_slice(&disp_a.to_le_bytes());
                code.extend_from_slice(&[0x48, 0x0F, 0xBA, 0xF0, 0x3F]);
                code.extend_from_slice(&[0x48, 0x89, 0x85]);
                code.extend_from_slice(&disp.to_le_bytes());
            }
            Opcode::Sqrt => {
                // sqrtsd xmm0, [rbp + disp_a]
                code.extend_from_slice(&[0xF2, 0x0F, 0x51, 0x85]);
                code.extend_from_slice(&disp_a.to_le_bytes());
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::Floor => {
                emit_load_xmm0_rbp(&mut code, disp_a);
                // roundsd xmm0, xmm0, 0x09
                code.extend_from_slice(&[0x66, 0x0F, 0x3A, 0x0B, 0xC0, 0x09]);
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::Ceil => {
                emit_load_xmm0_rbp(&mut code, disp_a);
                // roundsd xmm0, xmm0, 0x0A
                code.extend_from_slice(&[0x66, 0x0F, 0x3A, 0x0B, 0xC0, 0x0A]);
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::Ln => {
                emit_x87_ln_slot(&mut code, disp_a, disp);
            }
            Opcode::Exp => {
                emit_x87_exp_slot(&mut code, disp_a, disp);
            }
            Opcode::Gather => {
                let proof = program
                    .proofs
                    .get(usize::from(frame.proof_idx.saturating_sub(1)))
                    .ok_or(AotError::UnsupportedGraph("missing gather ProofEntry"))?;
                let target_meta =
                    col_metas
                        .get(frame.in_a as usize)
                        .copied()
                        .unwrap_or(AotColMeta {
                            origin_arena: None,
                            slice_offset: 0,
                            col_len: 0,
                        });
                let target_arena_idx = target_meta
                    .origin_arena
                    .unwrap_or(usize::from(proof.target_arena));
                let target_off = arena_byte_offsets
                    .get(target_arena_idx)
                    .copied()
                    .unwrap_or(0)
                    .saturating_add(target_meta.slice_offset.saturating_mul(8));
                let bound_hi = u64::from(proof.bound_hi).min(target_meta.col_len);

                // Load index f64 into xmm0: movsd xmm0, [rbp + disp_b]
                emit_load_xmm0_rbp(&mut code, disp_b);
                // Convert f64 to signed 64-bit integer in rdx: cvttsd2si rdx, xmm0
                code.extend_from_slice(&[0xF2, 0x48, 0x0F, 0x2C, 0xD0]);
                // Verify exact integer round-trip (`xmm0 == (f64)rdx` and not NaN) to reject
                // fractional indices (`1.5`), negative fractions (`-0.5`), and NaN/Inf:
                // xorpd xmm1, xmm1
                code.extend_from_slice(&[0x66, 0x0F, 0x57, 0xC9]);
                // cvtsi2sd xmm1, rdx
                code.extend_from_slice(&[0xF2, 0x48, 0x0F, 0x2A, 0xCA]);
                // ucomisd xmm0, xmm1
                code.extend_from_slice(&[0x66, 0x0F, 0x2E, 0xC1]);
                // jp +17 (NaN parity -> jump to exit(71))
                code.extend_from_slice(&[0x7A, 0x11]);
                // jne +15 (fractional/non-integer -> jump to exit(71))
                code.extend_from_slice(&[0x75, 0x0F]);
                // Bounds check: cmp rdx, bound_hi (unsigned comparison catches < 0 and >= bound_hi)
                code.extend_from_slice(&[0x48, 0xB8]);
                code.extend_from_slice(&bound_hi.to_le_bytes());
                // cmp rdx, rax
                code.extend_from_slice(&[0x48, 0x39, 0xC2]);
                // jb +12 (skip exit(71))
                code.extend_from_slice(&[0x72, 0x0C]);
                emit_sys_exit(&mut code, AOT_EXIT_GATHER_OOB as u32);

                // mov rax, r12
                code.extend_from_slice(&[0x4C, 0x89, 0xE0]);
                if target_off > 0 {
                    // movabs r8, target_off; add rax, r8
                    code.extend_from_slice(&[0x49, 0xB8]);
                    code.extend_from_slice(&target_off.to_le_bytes());
                    code.extend_from_slice(&[0x4C, 0x01, 0xC0]);
                }
                // movsd xmm0, [rax + rdx * 8]
                code.extend_from_slice(&[0xF2, 0x0F, 0x10, 0x04, 0xD0]);
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::ReduceSum
            | Opcode::ReduceMax
            | Opcode::ReduceMin
            | Opcode::RingEnqueue
            | Opcode::AssertContract
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
            | Opcode::CallArg => {}
        }
    }

    // 6. Accumulate each reduction's input into its scalar accumulator lane.
    let mut jz_skip_accum_patch: Option<usize> = None;
    if let Some(mask_idx) = filter_mask_node {
        let disp_mask = node_disp(mask_idx);
        // mov rax, [rbp + disp_mask]
        code.extend_from_slice(&[0x48, 0x8B, 0x85]);
        code.extend_from_slice(&disp_mask.to_le_bytes());
        // test rax, rax
        code.extend_from_slice(&[0x48, 0x85, 0xC0]);
        // jz skip_accum (0F 84 <rel32>)
        code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
        jz_skip_accum_patch = Some(code.len() - 4);
    }

    // rdx = r11 & 7; r11 += 1
    // mov rdx, r11
    code.extend_from_slice(&[0x4C, 0x89, 0xDA]);
    // and edx, 7
    code.extend_from_slice(&[0x83, 0xE2, 0x07]);
    // inc r11
    code.extend_from_slice(&[0x49, 0xFF, 0xC3]);

    for (red_k, spec) in reductions.iter().enumerate() {
        let disp_red_in = node_disp(spec.reduce_input_idx);
        let acc0 = red_acc_disp(node_count, red_k, 0);
        let acc4 = red_acc_disp(node_count, red_k, 4);
        match spec.reduction {
            AotReduction::Sum => {
                // movsd xmm0, [rbp + rdx * 8 + acc0]
                code.extend_from_slice(&[0xF2, 0x0F, 0x10, 0x84, 0xD5]);
                code.extend_from_slice(&acc0.to_le_bytes());
                // addsd xmm0, [rbp + disp_red_in]
                code.extend_from_slice(&[0xF2, 0x0F, 0x58, 0x85]);
                code.extend_from_slice(&disp_red_in.to_le_bytes());
                // movsd [rbp + rdx * 8 + acc0], xmm0
                code.extend_from_slice(&[0xF2, 0x0F, 0x11, 0x84, 0xD5]);
                code.extend_from_slice(&acc0.to_le_bytes());
            }
            AotReduction::Max => {
                // Load input into xmm0 (`src1`) and accumulator `[rbp + acc4]` as `src2`:
                // x86 `maxsd` returns `src2` on NaN, preserving the non-NaN accumulator (matching `f64::max`).
                emit_load_xmm0_rbp(&mut code, disp_red_in);
                // maxsd xmm0, [rbp + acc4]
                code.extend_from_slice(&[0xF2, 0x0F, 0x5F, 0x85]);
                code.extend_from_slice(&acc4.to_le_bytes());
                emit_store_xmm0_rbp(&mut code, acc4);
            }
            AotReduction::Min => {
                // Load input into xmm0 (`src1`) and accumulator `[rbp + acc4]` as `src2`:
                // x86 `minsd` returns `src2` on NaN, preserving the non-NaN accumulator (matching `f64::min`).
                emit_load_xmm0_rbp(&mut code, disp_red_in);
                // minsd xmm0, [rbp + acc4]
                code.extend_from_slice(&[0xF2, 0x0F, 0x5D, 0x85]);
                code.extend_from_slice(&acc4.to_le_bytes());
                emit_store_xmm0_rbp(&mut code, acc4);
            }
        }
    }

    if let Some(patch_pos) = jz_skip_accum_patch {
        let target = code.len();
        patch_rel32(&mut code, patch_pos, target)?;
    }

    // Loop increment & jump back to `loop_top`:
    // inc rcx
    code.extend_from_slice(&[0x48, 0xFF, 0xC1]);
    // jmp loop_top (E9 <rel32>)
    code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]);
    let jmp_back_pos = code.len() - 4;
    patch_rel32(&mut code, jmp_back_pos, loop_top)?;

    // Patch `loop_exit` target:
    let loop_exit_pos = code.len();
    patch_rel32(&mut code, jae_exit_patch_pos, loop_exit_pos)?;

    // 7. Post-loop reduction across accumulator lanes for each `AotReductionSpec`,
    //    storing the scalar reduction result into `[rbp + node_disp(spec.reduce_idx)]`.
    for (red_k, spec) in reductions.iter().enumerate() {
        match spec.reduction {
            AotReduction::Sum => {
                // ((acc[0] + acc[1]) + (acc[2] + acc[3])) + (acc[4] + acc[5]) + (acc[6] + acc[7]) -> xmm0
                let d0 = red_acc_disp(node_count, red_k, 0);
                let d1 = red_acc_disp(node_count, red_k, 1);
                let d2 = red_acc_disp(node_count, red_k, 2);
                let d3 = red_acc_disp(node_count, red_k, 3);
                let d4 = red_acc_disp(node_count, red_k, 4);
                let d5 = red_acc_disp(node_count, red_k, 5);
                let d6 = red_acc_disp(node_count, red_k, 6);
                let d7 = red_acc_disp(node_count, red_k, 7);

                emit_load_xmm0_rbp(&mut code, d0);
                // addsd xmm0, [rbp + d1]
                code.extend_from_slice(&[0xF2, 0x0F, 0x58, 0x85]);
                code.extend_from_slice(&d1.to_le_bytes());
                // movsd xmm1, [rbp + d2]
                code.extend_from_slice(&[0xF2, 0x0F, 0x10, 0x8D]);
                code.extend_from_slice(&d2.to_le_bytes());
                // addsd xmm1, [rbp + d3]
                code.extend_from_slice(&[0xF2, 0x0F, 0x58, 0x8D]);
                code.extend_from_slice(&d3.to_le_bytes());
                // addsd xmm0, xmm1
                code.extend_from_slice(&[0xF2, 0x0F, 0x58, 0xC1]);
                // movsd xmm1, [rbp + d4]
                code.extend_from_slice(&[0xF2, 0x0F, 0x10, 0x8D]);
                code.extend_from_slice(&d4.to_le_bytes());
                // addsd xmm1, [rbp + d5]
                code.extend_from_slice(&[0xF2, 0x0F, 0x58, 0x8D]);
                code.extend_from_slice(&d5.to_le_bytes());
                // addsd xmm0, xmm1
                code.extend_from_slice(&[0xF2, 0x0F, 0x58, 0xC1]);
                // movsd xmm1, [rbp + d6]
                code.extend_from_slice(&[0xF2, 0x0F, 0x10, 0x8D]);
                code.extend_from_slice(&d6.to_le_bytes());
                // addsd xmm1, [rbp + d7]
                code.extend_from_slice(&[0xF2, 0x0F, 0x58, 0x8D]);
                code.extend_from_slice(&d7.to_le_bytes());
                // addsd xmm0, xmm1
                code.extend_from_slice(&[0xF2, 0x0F, 0x58, 0xC1]);
            }
            AotReduction::Max => {
                emit_load_xmm0_rbp(&mut code, red_acc_disp(node_count, red_k, 4));
                for lane in 0..4i32 {
                    let dl = red_acc_disp(node_count, red_k, lane);
                    // maxsd xmm0, [rbp + dl]
                    code.extend_from_slice(&[0xF2, 0x0F, 0x5F, 0x85]);
                    code.extend_from_slice(&dl.to_le_bytes());
                }
            }
            AotReduction::Min => {
                emit_load_xmm0_rbp(&mut code, red_acc_disp(node_count, red_k, 4));
                for lane in 0..4i32 {
                    let dl = red_acc_disp(node_count, red_k, lane);
                    // minsd xmm0, [rbp + dl]
                    code.extend_from_slice(&[0xF2, 0x0F, 0x5D, 0x85]);
                    code.extend_from_slice(&dl.to_le_bytes());
                }
            }
        }
        emit_store_xmm0_rbp(&mut code, node_disp(spec.reduce_idx));
    }

    // 8. Post-loop scalar epilogue: evaluate all `PostReduceScalar` nodes in topological SSA order.
    for &idx in post_reduce_nodes {
        let Some(frame) = program.nodes.get(idx) else {
            continue;
        };
        let disp = node_disp(idx);
        let disp_a = node_disp(frame.in_a as usize);
        let disp_b = node_disp(frame.in_b as usize);

        match frame.opcode {
            Opcode::Add => {
                emit_load_xmm0_rbp(&mut code, disp_a);
                // addsd xmm0, [rbp + disp_b]
                code.extend_from_slice(&[0xF2, 0x0F, 0x58, 0x85]);
                code.extend_from_slice(&disp_b.to_le_bytes());
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::Sub => {
                emit_load_xmm0_rbp(&mut code, disp_a);
                // subsd xmm0, [rbp + disp_b]
                code.extend_from_slice(&[0xF2, 0x0F, 0x5C, 0x85]);
                code.extend_from_slice(&disp_b.to_le_bytes());
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::Mul => {
                emit_load_xmm0_rbp(&mut code, disp_a);
                // mulsd xmm0, [rbp + disp_b]
                code.extend_from_slice(&[0xF2, 0x0F, 0x59, 0x85]);
                code.extend_from_slice(&disp_b.to_le_bytes());
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::Div => {
                emit_load_xmm0_rbp(&mut code, disp_a);
                // divsd xmm0, [rbp + disp_b]
                code.extend_from_slice(&[0xF2, 0x0F, 0x5E, 0x85]);
                code.extend_from_slice(&disp_b.to_le_bytes());
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::Min => {
                emit_load_xmm0_rbp(&mut code, disp_a);
                // minsd xmm0, [rbp + disp_b]
                code.extend_from_slice(&[0xF2, 0x0F, 0x5D, 0x85]);
                code.extend_from_slice(&disp_b.to_le_bytes());
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::Max => {
                emit_load_xmm0_rbp(&mut code, disp_a);
                // maxsd xmm0, [rbp + disp_b]
                code.extend_from_slice(&[0xF2, 0x0F, 0x5F, 0x85]);
                code.extend_from_slice(&disp_b.to_le_bytes());
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::Fma => {
                let (in_b, in_c) = frame.decode_fma_bc();
                let disp_b_fma = node_disp(in_b as usize);
                let disp_c_fma = node_disp(in_c as usize);
                emit_load_xmm0_rbp(&mut code, disp_a);
                // movsd xmm1, [rbp + disp_b_fma]
                code.extend_from_slice(&[0xF2, 0x0F, 0x10, 0x8D]);
                code.extend_from_slice(&disp_b_fma.to_le_bytes());
                emit_vex_fma213_xmm_rm_rbp(&mut code, 0, 1, disp_c_fma);
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::Abs => {
                // mov rax, [rbp + disp_a]; btr rax, 63; mov [rbp + disp], rax
                code.extend_from_slice(&[0x48, 0x8B, 0x85]);
                code.extend_from_slice(&disp_a.to_le_bytes());
                code.extend_from_slice(&[0x48, 0x0F, 0xBA, 0xF0, 0x3F]);
                code.extend_from_slice(&[0x48, 0x89, 0x85]);
                code.extend_from_slice(&disp.to_le_bytes());
            }
            Opcode::Sqrt => {
                // sqrtsd xmm0, [rbp + disp_a]
                code.extend_from_slice(&[0xF2, 0x0F, 0x51, 0x85]);
                code.extend_from_slice(&disp_a.to_le_bytes());
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::Floor => {
                emit_load_xmm0_rbp(&mut code, disp_a);
                // roundsd xmm0, xmm0, 0x09
                code.extend_from_slice(&[0x66, 0x0F, 0x3A, 0x0B, 0xC0, 0x09]);
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::Ceil => {
                emit_load_xmm0_rbp(&mut code, disp_a);
                // roundsd xmm0, xmm0, 0x0A
                code.extend_from_slice(&[0x66, 0x0F, 0x3A, 0x0B, 0xC0, 0x0A]);
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::Ln => {
                emit_x87_ln_slot(&mut code, disp_a, disp);
            }
            Opcode::Exp => {
                emit_x87_exp_slot(&mut code, disp_a, disp);
            }
            Opcode::RingEnqueue => {
                emit_load_xmm0_rbp(&mut code, disp_a);
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::AssertContract => {
                emit_load_xmm0_rbp(&mut code, disp_a);
                emit_store_xmm0_rbp(&mut code, disp);
                // mov rax, [rbp + disp_a]
                code.extend_from_slice(&[0x48, 0x8B, 0x85]);
                code.extend_from_slice(&disp_a.to_le_bytes());
                // movabs rdx, 0x7FF0_0000_0000_0000
                code.extend_from_slice(&[0x48, 0xBA]);
                code.extend_from_slice(&0x7FF0_0000_0000_0000_u64.to_le_bytes());
                // and rax, rdx
                code.extend_from_slice(&[0x48, 0x21, 0xD0]);
                // cmp rax, rdx
                code.extend_from_slice(&[0x48, 0x39, 0xD0]);
                // jne +12 (skip exit(70))
                code.extend_from_slice(&[0x75, 0x0C]);
                emit_sys_exit(&mut code, AOT_EXIT_CONTRACT_FAILED as u32);
            }
            _ => {
                return Err(AotError::UnsupportedGraph(
                    "unsupported opcode in post-reduction scalar phase",
                ))
            }
        }
    }

    // 9. Store final scalar `output_node` to `[rbp - 72]` (`disp8 = 0xB8`),
    //    write 8-byte little-endian `f64` to `stdout` (`SYS_write = 1`, `fd = 1`), and `SYS_exit(0)`.
    emit_load_xmm0_rbp(&mut code, node_disp(program.output_node as usize));
    // movsd [rbp - 72], xmm0
    code.extend_from_slice(&[0xF2, 0x0F, 0x11, 0x45, 0xB8]);

    // mov eax, 1
    code.extend_from_slice(&[0xB8, 0x01, 0x00, 0x00, 0x00]);
    // mov edi, 1
    code.extend_from_slice(&[0xBF, 0x01, 0x00, 0x00, 0x00]);
    // lea rsi, [rbp - 72]
    code.extend_from_slice(&[0x48, 0x8D, 0x75, 0xB8]);
    // mov edx, 8
    code.extend_from_slice(&[0xBA, 0x08, 0x00, 0x00, 0x00]);
    // syscall
    code.extend_from_slice(&[0x0F, 0x05]);

    emit_sys_exit(&mut code, 0);
    Ok(code)
}

fn free_if_dying(
    src_node: usize,
    cur_idx: usize,
    last_use: &[usize],
    free_ymm_mask: &mut u16,
    node_ymm: &mut [Option<u8>],
) {
    if last_use.get(src_node).copied() == Some(cur_idx) {
        if let Some(slot) = node_ymm.get_mut(src_node) {
            if let Some(reg) = *slot {
                *free_ymm_mask |= 1u16 << reg;
                *slot = None;
            }
        }
    }
}

fn alloc_ymm_slot(node_idx: usize, free_ymm_mask: &mut u16, node_ymm: &mut [Option<u8>]) -> u8 {
    if *free_ymm_mask != 0 {
        let reg = u8::try_from(free_ymm_mask.trailing_zeros()).unwrap_or(0);
        *free_ymm_mask &= !(1u16 << reg);
        if let Some(slot) = node_ymm.get_mut(node_idx) {
            *slot = Some(reg);
        }
        reg
    } else {
        if let Some(slot) = node_ymm.get_mut(node_idx) {
            *slot = None;
        }
        0
    }
}

fn node_disp(node_idx: usize) -> i32 {
    let idx_i32 = i32::try_from(node_idx).unwrap_or(0);
    (-128i32).saturating_sub(idx_i32.saturating_mul(32))
}

fn red_acc_disp(node_count: usize, red_k: usize, lane: i32) -> i32 {
    let base_nodes =
        (-128i32).saturating_sub(i32::try_from(node_count).unwrap_or(0).saturating_mul(32));
    let red_base = base_nodes.saturating_sub(
        i32::try_from(red_k.saturating_add(1))
            .unwrap_or(1)
            .saturating_mul(64),
    );
    red_base.saturating_add(lane.saturating_mul(8))
}

fn emit_load_ymm_rax_rcx8(code: &mut Vec<u8>, dst: u8) {
    // vmovupd ymm_dst, ymmword ptr [rax + rcx * 8]
    let r_inv = if (dst & 8) == 0 { 0x80 } else { 0x00 };
    let byte1 = r_inv | 0x61; // ~X = 1, ~B = 1, mmmmm = 0x01 (0F)
    let byte2 = 0x7D; // W = 0, ~vvvv = 1111, L = 1 (256b), pp = 01 (0x66)
    let modrm = 0x04 | ((dst & 7) << 3);
    code.extend_from_slice(&[0xC4, byte1, byte2, 0x10, modrm, 0xC8]);
}

fn emit_load_ymm_rbp(code: &mut Vec<u8>, dst: u8, disp: i32) {
    // vmovupd ymm_dst, ymmword ptr [rbp + disp32]
    let r_inv = if (dst & 8) == 0 { 0x80 } else { 0x00 };
    let byte1 = r_inv | 0x61;
    let byte2 = 0x7D;
    let modrm = 0x85 | ((dst & 7) << 3);
    code.extend_from_slice(&[0xC4, byte1, byte2, 0x10, modrm]);
    code.extend_from_slice(&disp.to_le_bytes());
}

fn emit_store_ymm_rbp(code: &mut Vec<u8>, src: u8, disp: i32) {
    // vmovupd ymmword ptr [rbp + disp32], ymm_src
    let r_inv = if (src & 8) == 0 { 0x80 } else { 0x00 };
    let byte1 = r_inv | 0x61;
    let byte2 = 0x7D;
    let modrm = 0x85 | ((src & 7) << 3);
    code.extend_from_slice(&[0xC4, byte1, byte2, 0x11, modrm]);
    code.extend_from_slice(&disp.to_le_bytes());
}

fn emit_vex_ymm_rr(code: &mut Vec<u8>, opcode: u8, dst: u8, src1: u8, src2: u8) {
    let r_inv = if (dst & 8) == 0 { 0x80 } else { 0x00 };
    let b_inv = if (src2 & 8) == 0 { 0x20 } else { 0x00 };
    let byte1 = r_inv | 0x40 | b_inv | 0x01; // mmmmm = 0x01 (0F)
    let vvvv_inv = (!src1) & 0x0F;
    let byte2 = (vvvv_inv << 3) | 0x05; // L = 1 (256b), pp = 01 (0x66)
    let modrm = 0xC0 | ((dst & 7) << 3) | (src2 & 7);
    code.extend_from_slice(&[0xC4, byte1, byte2, opcode, modrm]);
}

fn emit_vex_ymm_rm_rbp(code: &mut Vec<u8>, opcode: u8, dst: u8, src1: u8, disp: i32) {
    let r_inv = if (dst & 8) == 0 { 0x80 } else { 0x00 };
    let byte1 = r_inv | 0x61; // ~X = 1, ~B = 1 (rbp), mmmmm = 0x01 (0F)
    let vvvv_inv = (!src1) & 0x0F;
    let byte2 = (vvvv_inv << 3) | 0x05; // L = 1 (256b), pp = 01 (0x66)
    let modrm = 0x85 | ((dst & 7) << 3);
    code.extend_from_slice(&[0xC4, byte1, byte2, opcode, modrm]);
    code.extend_from_slice(&disp.to_le_bytes());
}

fn emit_vex_roundpd_rr(code: &mut Vec<u8>, dst: u8, src: u8, imm8: u8) {
    let r_inv = if (dst & 8) == 0 { 0x80 } else { 0x00 };
    let b_inv = if (src & 8) == 0 { 0x20 } else { 0x00 };
    let byte1 = r_inv | 0x40 | b_inv | 0x03; // mmmmm = 0x03 (0F 3A)
    let byte2 = 0x7D; // ~vvvv = 1111, L = 1, pp = 01 (0x66)
    let modrm = 0xC0 | ((dst & 7) << 3) | (src & 7);
    code.extend_from_slice(&[0xC4, byte1, byte2, 0x09, modrm, imm8]);
}

fn emit_vex_roundpd_rm_rbp(code: &mut Vec<u8>, dst: u8, disp: i32, imm8: u8) {
    let r_inv = if (dst & 8) == 0 { 0x80 } else { 0x00 };
    let byte1 = r_inv | 0x63; // ~X = 1, ~B = 1, mmmmm = 0x03 (0F 3A)
    let byte2 = 0x7D;
    let modrm = 0x85 | ((dst & 7) << 3);
    code.extend_from_slice(&[0xC4, byte1, byte2, 0x09, modrm]);
    code.extend_from_slice(&disp.to_le_bytes());
    code.push(imm8);
}

fn emit_vex_fma_ymm_rr(code: &mut Vec<u8>, opcode: u8, dst: u8, src1: u8, src2: u8) {
    let r_inv = if (dst & 8) == 0 { 0x80 } else { 0x00 };
    let b_inv = if (src2 & 8) == 0 { 0x20 } else { 0x00 };
    let byte1 = r_inv | 0x40 | b_inv | 0x02; // mmmmm = 0x02 (0F 38)
    let vvvv_inv = (!src1) & 0x0F;
    let byte2 = 0x80 | (vvvv_inv << 3) | 0x05; // W = 1 (f64), L = 1 (256b), pp = 01 (0x66)
    let modrm = 0xC0 | ((dst & 7) << 3) | (src2 & 7);
    code.extend_from_slice(&[0xC4, byte1, byte2, opcode, modrm]);
}

fn emit_vex_fma_ymm_rm_rbp(code: &mut Vec<u8>, opcode: u8, dst: u8, src1: u8, disp: i32) {
    let r_inv = if (dst & 8) == 0 { 0x80 } else { 0x00 };
    let byte1 = r_inv | 0x62; // ~X = 1, ~B = 1 (rbp), mmmmm = 0x02 (0F 38)
    let vvvv_inv = (!src1) & 0x0F;
    let byte2 = 0x80 | (vvvv_inv << 3) | 0x05; // W = 1 (f64), L = 1 (256b), pp = 01 (0x66)
    let modrm = 0x85 | ((dst & 7) << 3);
    code.extend_from_slice(&[0xC4, byte1, byte2, opcode, modrm]);
    code.extend_from_slice(&disp.to_le_bytes());
}

fn emit_vex_fma213_xmm_rm_rbp(code: &mut Vec<u8>, dst: u8, src1: u8, disp: i32) {
    let r_inv = if (dst & 8) == 0 { 0x80 } else { 0x00 };
    let byte1 = r_inv | 0x62; // ~X = 1, ~B = 1 (rbp), mmmmm = 0x02 (0F 38)
    let vvvv_inv = (!src1) & 0x0F;
    let byte2 = 0x80 | (vvvv_inv << 3) | 0x01; // W = 1 (f64), L = 0 (scalar), pp = 01 (0x66)
    let modrm = 0x85 | ((dst & 7) << 3);
    code.extend_from_slice(&[0xC4, byte1, byte2, 0xA9, modrm]);
    code.extend_from_slice(&disp.to_le_bytes());
}

fn emit_vex_vpsrlq_ymm(code: &mut Vec<u8>, dst: u8, src: u8, imm8: u8) {
    let b_inv = if (src & 8) == 0 { 0x20 } else { 0x00 };
    let byte1 = 0xC0 | b_inv | 0x01; // ~R = 1, ~X = 1, ~B, mmmmm = 0x01 (0F)
    let vvvv_inv = (!dst) & 0x0F;
    let byte2 = (vvvv_inv << 3) | 0x05; // W = 0, L = 1 (256b), pp = 01 (0x66)
    let modrm = 0xD0 | (src & 7); // /2
    code.extend_from_slice(&[0xC4, byte1, byte2, 0x73, modrm, imm8]);
}

fn emit_vex_vpsllq_ymm(code: &mut Vec<u8>, dst: u8, src: u8, imm8: u8) {
    let b_inv = if (src & 8) == 0 { 0x20 } else { 0x00 };
    let byte1 = 0xC0 | b_inv | 0x01; // ~R = 1, ~X = 1, ~B, mmmmm = 0x01 (0F)
    let vvvv_inv = (!dst) & 0x0F;
    let byte2 = (vvvv_inv << 3) | 0x05; // W = 0, L = 1 (256b), pp = 01 (0x66)
    let modrm = 0xF0 | (src & 7); // /6
    code.extend_from_slice(&[0xC4, byte1, byte2, 0x73, modrm, imm8]);
}

const MATH_SCRATCH_YMM2: usize = 0;
const MATH_SCRATCH_YMM3: usize = 1;
const MATH_ZERO: usize = 2;
const MATH_ONE: usize = 3;
const MATH_EXP_MASK: usize = 4;
const MATH_MANT_MASK: usize = 5;
const MATH_HALF_EXP: usize = 6;
const MATH_SQRT_HALF: usize = 7;
const MATH_TWO52_BITS: usize = 8;
const MATH_TWO52_PLUS_1022: usize = 9;
const MATH_TWO52_PLUS_1023: usize = 10;
const MATH_LN2_HI: usize = 11;
const MATH_LN2_LO: usize = 12;
const MATH_LOG2_E: usize = 13;
const MATH_NEG_INF: usize = 14;
const MATH_QNAN: usize = 15;
const MATH_LG1: usize = 16;
const MATH_LG2: usize = 17;
const MATH_LG3: usize = 18;
const MATH_LG4: usize = 19;
const MATH_LG5: usize = 20;
const MATH_LG6: usize = 21;
const MATH_LG7: usize = 22;
const MATH_EXP_CLAMP_LO: usize = 23;
const MATH_EXP_CLAMP_HI: usize = 24;
const MATH_EXP_C3: usize = 25;
const MATH_EXP_C4: usize = 26;
const MATH_EXP_C5: usize = 27;
const MATH_EXP_C6: usize = 28;
const MATH_EXP_C7: usize = 29;
const MATH_EXP_C8: usize = 30;
const MATH_EXP_C9: usize = 31;
const MATH_EXP_C10: usize = 32;
const MATH_EXP_C11: usize = 33;
const MATH_EXP_C12: usize = 34;
const MATH_DBL_MIN: usize = 35;
const MATH_TWO54: usize = 36;
const MATH_FIFTY_FOUR: usize = 37;
const MATH_TOTAL_SLOTS: usize = 38;

const MATH_CONST_TABLE: &[(usize, u64)] = &[
    (MATH_ZERO, 0x0000_0000_0000_0000),
    (MATH_ONE, 0x3FF0_0000_0000_0000),
    (MATH_EXP_MASK, 0x7FF0_0000_0000_0000),
    (MATH_MANT_MASK, 0x000F_FFFF_FFFF_FFFF),
    (MATH_HALF_EXP, 0x3FE0_0000_0000_0000),
    (MATH_SQRT_HALF, 0x3FE6_A09E_667F_3BCD),
    (MATH_TWO52_BITS, 0x4330_0000_0000_0000),
    (MATH_TWO52_PLUS_1022, 0x4330_0000_0000_03FE),
    (MATH_TWO52_PLUS_1023, 0x4330_0000_0000_03FF),
    (MATH_LN2_HI, 0x3FE6_2E42_FEE0_0000),
    (MATH_LN2_LO, 0x3DEA_39EF_3579_3C76),
    (MATH_LOG2_E, 0x3FF7_1547_652B_82FE),
    (MATH_NEG_INF, 0xFFF0_0000_0000_0000),
    (MATH_QNAN, 0x7FF8_0000_0000_0000),
    (MATH_LG1, 0x3FE5_5555_5555_5593),
    (MATH_LG2, 0x3FD9_9999_9997_FA04),
    (MATH_LG3, 0x3FD2_4924_9422_9359),
    (MATH_LG4, 0x3FCC_71C5_1D8E_78AF),
    (MATH_LG5, 0x3FC7_4664_96CB_03DE),
    (MATH_LG6, 0x3FC3_9A09_D078_C69F),
    (MATH_LG7, 0x3FC2_F112_DF3E_5244),
    (MATH_EXP_CLAMP_LO, 0xC087_4910_D52D_3052),
    (MATH_EXP_CLAMP_HI, 0x4086_2E42_FEFA_39EF),
    (MATH_EXP_C3, (1.0_f64 / 6.0_f64).to_bits()),
    (MATH_EXP_C4, (1.0_f64 / 24.0_f64).to_bits()),
    (MATH_EXP_C5, (1.0_f64 / 120.0_f64).to_bits()),
    (MATH_EXP_C6, (1.0_f64 / 720.0_f64).to_bits()),
    (MATH_EXP_C7, (1.0_f64 / 5_040.0_f64).to_bits()),
    (MATH_EXP_C8, (1.0_f64 / 40_320.0_f64).to_bits()),
    (MATH_EXP_C9, (1.0_f64 / 362_880.0_f64).to_bits()),
    (MATH_EXP_C10, (1.0_f64 / 3_628_800.0_f64).to_bits()),
    (MATH_EXP_C11, (1.0_f64 / 39_916_800.0_f64).to_bits()),
    (MATH_EXP_C12, (1.0_f64 / 479_001_600.0_f64).to_bits()),
    (MATH_DBL_MIN, 0x0010_0000_0000_0000),
    (MATH_TWO54, 0x4350_0000_0000_0000),
    (MATH_FIFTY_FOUR, 0x404B_0000_0000_0000),
];

fn math_slot_disp(node_count: usize, red_count: usize, slot_idx: usize) -> i32 {
    let base_nodes =
        (-128i32).saturating_sub(i32::try_from(node_count).unwrap_or(0).saturating_mul(32));
    let base_reds =
        base_nodes.saturating_sub(i32::try_from(red_count).unwrap_or(0).saturating_mul(64));
    base_reds.saturating_sub(
        i32::try_from(slot_idx.saturating_add(1))
            .unwrap_or(1)
            .saturating_mul(32),
    )
}

/// Emits a 4-way vectorized 256-bit AVX2 + FMA3 natural logarithm (`ln(x)`) evaluating `ymm0 -> ymm0`
/// using fdlibm's rational transformation `s = (m - 1) / (m + 1)` and degree-7 minimax series `Lg1..Lg7`
/// with full IEEE-754 special-case masking (`x == 0 -> -Inf`, `x < 0 -> NaN`, `+Inf -> +Inf`, `NaN -> NaN`).
fn emit_avx2_ln_ymm(code: &mut Vec<u8>, node_count: usize, red_count: usize, scratch_disp: i32) {
    let mslot = |s: usize| math_slot_disp(node_count, red_count, s);

    // Save `ymm2` and `ymm3` to scratch slots and save original input `x` (`ymm0`) to `[rbp + scratch_disp]`.
    emit_store_ymm_rbp(code, 2, mslot(MATH_SCRATCH_YMM2));
    emit_store_ymm_rbp(code, 3, mslot(MATH_SCRATCH_YMM3));
    emit_store_ymm_rbp(code, 0, scratch_disp);

    // 0. Normalize subnormal inputs (`0 < x < 2^-1022`) by scaling `x * 2^54` and subtracting `54.0` from `e0`:
    // vcmppd ymm3, ymm0, [MATH_DBL_MIN], 0x01 (_CMP_LT_OS)
    emit_vex_ymm_rm_rbp(code, 0xC2, 3, 0, mslot(MATH_DBL_MIN));
    code.push(0x01);
    // vmulpd ymm2, ymm0, [MATH_TWO54]
    emit_vex_ymm_rm_rbp(code, 0x59, 2, 0, mslot(MATH_TWO54));
    // vandpd ymm2, ymm3, ymm2
    emit_vex_ymm_rr(code, 0x54, 2, 3, 2);
    // vandnpd ymm0, ymm3, ymm0
    emit_vex_ymm_rr(code, 0x55, 0, 3, 0);
    // vorpd ymm0, ymm0, ymm2
    emit_vex_ymm_rr(code, 0x56, 0, 0, 2);

    // 1. Extract unbiased exponent `e0 = ((bits >> 52) & 0x7FF) - 1022` as f64 in `ymm1`:
    // vandpd ymm1, ymm0, [MATH_EXP_MASK]
    emit_vex_ymm_rm_rbp(code, 0x54, 1, 0, mslot(MATH_EXP_MASK));
    // vpsrlq ymm1, ymm1, 52
    emit_vex_vpsrlq_ymm(code, 1, 1, 52);
    // vorpd ymm1, ymm1, [MATH_TWO52_BITS]
    emit_vex_ymm_rm_rbp(code, 0x56, 1, 1, mslot(MATH_TWO52_BITS));
    // vsubpd ymm1, ymm1, [MATH_TWO52_PLUS_1022]
    emit_vex_ymm_rm_rbp(code, 0x5C, 1, 1, mslot(MATH_TWO52_PLUS_1022));
    // vandpd ymm3, ymm3, [MATH_FIFTY_FOUR]
    emit_vex_ymm_rm_rbp(code, 0x54, 3, 3, mslot(MATH_FIFTY_FOUR));
    // vsubpd ymm1, ymm1, ymm3 (`e0 -= 54.0` for subnormal lanes)
    emit_vex_ymm_rr(code, 0x5C, 1, 1, 3);

    // 2. Extract mantissa `m in [0.5, 1.0)` in `ymm2`:
    // vandpd ymm2, ymm0, [MATH_MANT_MASK]
    emit_vex_ymm_rm_rbp(code, 0x54, 2, 0, mslot(MATH_MANT_MASK));
    // vorpd ymm2, ymm2, [MATH_HALF_EXP]
    emit_vex_ymm_rm_rbp(code, 0x56, 2, 2, mslot(MATH_HALF_EXP));

    // 3. Center `m` into `[sqrt(0.5), sqrt(2))`: if `m < sqrt(0.5)`, `m *= 2` and `e -= 1.0`:
    // vcmppd ymm3, ymm2, [MATH_SQRT_HALF], 0x01 (_CMP_LT_OS)
    emit_vex_ymm_rm_rbp(code, 0xC2, 3, 2, mslot(MATH_SQRT_HALF));
    code.push(0x01);
    // vandpd ymm0, ymm2, ymm3
    emit_vex_ymm_rr(code, 0x54, 0, 2, 3);
    // vaddpd ymm2, ymm2, ymm0
    emit_vex_ymm_rr(code, 0x58, 2, 2, 0);
    // vandpd ymm3, ymm3, [MATH_ONE]
    emit_vex_ymm_rm_rbp(code, 0x54, 3, 3, mslot(MATH_ONE));
    // vsubpd ymm1, ymm1, ymm3 (`ymm1 = e`)
    emit_vex_ymm_rr(code, 0x5C, 1, 1, 3);

    // 4. Compute `s = (m - 1.0) / (m + 1.0)` in `ymm2` and `z = s * s` in `ymm3`:
    // vsubpd ymm0, ymm2, [MATH_ONE]
    emit_vex_ymm_rm_rbp(code, 0x5C, 0, 2, mslot(MATH_ONE));
    // vaddpd ymm2, ymm2, [MATH_ONE]
    emit_vex_ymm_rm_rbp(code, 0x58, 2, 2, mslot(MATH_ONE));
    // vdivpd ymm2, ymm0, ymm2 (`ymm2 = s`)
    emit_vex_ymm_rr(code, 0x5E, 2, 0, 2);
    // vmulpd ymm3, ymm2, ymm2 (`ymm3 = z = s * s`)
    emit_vex_ymm_rr(code, 0x59, 3, 2, 2);

    // 5. Horner evaluation of `R(z) = z * (Lg1 + z * (Lg2 + ... + z * Lg7))` in `ymm0`:
    emit_load_ymm_rbp(code, 0, mslot(MATH_LG7));
    for &lg_slot in &[MATH_LG6, MATH_LG5, MATH_LG4, MATH_LG3, MATH_LG2, MATH_LG1] {
        emit_vex_fma_ymm_rm_rbp(code, 0xA8, 0, 3, mslot(lg_slot));
    }
    // vmulpd ymm0, ymm0, ymm3 (`ymm0 = R(z)`)
    emit_vex_ymm_rr(code, 0x59, 0, 0, 3);
    // vaddpd ymm3, ymm2, ymm2 (`ymm3 = 2.0 * s`)
    emit_vex_ymm_rr(code, 0x58, 3, 2, 2);
    // vfmadd213pd ymm0, ymm2, ymm3 (`ymm0 = R(z) * s + 2.0 * s = ln(m)`)
    emit_vex_fma_ymm_rr(code, 0xA8, 0, 2, 3);

    // 6. Combine with `e * ln(2)` using Cody-Waite `LN2_HI + LN2_LO`:
    emit_load_ymm_rbp(code, 2, mslot(MATH_LN2_LO));
    // vfmadd213pd ymm2, ymm1, ymm0 (`ymm2 = e * LN2_LO + ln(m)`)
    emit_vex_fma_ymm_rr(code, 0xA8, 2, 1, 0);
    emit_load_ymm_rbp(code, 0, mslot(MATH_LN2_HI));
    // vfmadd213pd ymm0, ymm1, ymm2 (`ymm0 = e * LN2_HI + (e * LN2_LO + ln(m))`)
    emit_vex_fma_ymm_rr(code, 0xA8, 0, 1, 2);

    // 7. IEEE-754 special cases using saved input `x` in `[rbp + scratch_disp]`:
    emit_load_ymm_rbp(code, 1, scratch_disp);
    // 7a. `x == 0.0` (including `-0.0`) -> `-Inf`:
    emit_vex_ymm_rm_rbp(code, 0xC2, 2, 1, mslot(MATH_ZERO));
    code.push(0x00); // _CMP_EQ_OQ
    emit_vex_ymm_rr(code, 0x55, 0, 2, 0); // vandnpd ymm0, ymm2, ymm0
    emit_vex_ymm_rm_rbp(code, 0x54, 2, 2, mslot(MATH_NEG_INF));
    emit_vex_ymm_rr(code, 0x56, 0, 0, 2);
    // 7b. `x == +Inf` -> `+Inf`:
    emit_vex_ymm_rm_rbp(code, 0xC2, 2, 1, mslot(MATH_EXP_MASK));
    code.push(0x00); // _CMP_EQ_OQ
    emit_vex_ymm_rr(code, 0x55, 0, 2, 0);
    emit_vex_ymm_rm_rbp(code, 0x54, 2, 2, mslot(MATH_EXP_MASK));
    emit_vex_ymm_rr(code, 0x56, 0, 0, 2);
    // 7c. `x < 0.0 || x.is_nan()` -> `QNaN`:
    emit_vex_ymm_rm_rbp(code, 0xC2, 2, 1, mslot(MATH_ZERO));
    code.push(0x09); // _CMP_NGE_UQ (`!(x >= 0.0)`)
    emit_vex_ymm_rr(code, 0x55, 0, 2, 0);
    emit_vex_ymm_rm_rbp(code, 0x54, 2, 2, mslot(MATH_QNAN));
    emit_vex_ymm_rr(code, 0x56, 0, 0, 2);

    // Restore `ymm2` and `ymm3`.
    emit_load_ymm_rbp(code, 2, mslot(MATH_SCRATCH_YMM2));
    emit_load_ymm_rbp(code, 3, mslot(MATH_SCRATCH_YMM3));
}

/// Emits a 4-way vectorized 256-bit AVX2 + FMA3 exponential (`exp(x)`) evaluating `ymm0 -> ymm0`
/// using Cody-Waite range reduction `x = k * ln(2) + r` (`|r| <= ln(2)/2`) and degree-12 Taylor/minimax
/// evaluation with IEEE-754 overflow (`+Inf`), underflow (`0.0`), and `NaN` preservation.
fn emit_avx2_exp_ymm(code: &mut Vec<u8>, node_count: usize, red_count: usize, scratch_disp: i32) {
    let mslot = |s: usize| math_slot_disp(node_count, red_count, s);

    // Save `ymm2` and `ymm3` to scratch slots and save original input `x` (`ymm0`) to `[rbp + scratch_disp]`.
    emit_store_ymm_rbp(code, 2, mslot(MATH_SCRATCH_YMM2));
    emit_store_ymm_rbp(code, 3, mslot(MATH_SCRATCH_YMM3));
    emit_store_ymm_rbp(code, 0, scratch_disp);

    // 1. Clamp `x` into `[-745.1332191019412, 709.78271289338397]` so `k` stays in `[-1075, 1024]`:
    emit_vex_ymm_rm_rbp(code, 0x5F, 0, 0, mslot(MATH_EXP_CLAMP_LO)); // vmaxpd
    emit_vex_ymm_rm_rbp(code, 0x5D, 0, 0, mslot(MATH_EXP_CLAMP_HI)); // vminpd

    // 2. Range reduction `k = round_nearest_even(x * LOG2_E)` in `ymm1` and `r = (x - k*LN2_HI) - k*LN2_LO` in `ymm0`:
    emit_vex_ymm_rm_rbp(code, 0x59, 1, 0, mslot(MATH_LOG2_E));
    emit_vex_roundpd_rr(code, 1, 1, 0x08);
    emit_vex_ymm_rm_rbp(code, 0x59, 2, 1, mslot(MATH_LN2_HI));
    emit_vex_ymm_rr(code, 0x5C, 0, 0, 2);
    emit_vex_ymm_rm_rbp(code, 0x59, 2, 1, mslot(MATH_LN2_LO));
    emit_vex_ymm_rr(code, 0x5C, 0, 0, 2);

    // 3. Split `k = k1 + k2` (`k1 = floor(k * 0.5)` in `ymm3`, `k2 = k - k1` in `ymm1`) so both
    //    `2^k1` and `2^k2` stay normal in `[-538, 512]`, avoiding `2^1024` overflow and supporting gradual underflow:
    emit_vex_ymm_rm_rbp(code, 0x59, 3, 1, mslot(MATH_HALF_EXP));
    emit_vex_roundpd_rr(code, 3, 3, 0x09); // floor(k * 0.5)
    emit_vex_ymm_rr(code, 0x5C, 1, 1, 3); // k2 = k - k1
    emit_vex_ymm_rm_rbp(code, 0x58, 3, 3, mslot(MATH_TWO52_PLUS_1023));
    emit_vex_vpsllq_ymm(code, 3, 3, 52); // ymm3 = 2^k1
    emit_vex_ymm_rm_rbp(code, 0x58, 1, 1, mslot(MATH_TWO52_PLUS_1023));
    emit_vex_vpsllq_ymm(code, 1, 1, 52); // ymm1 = 2^k2

    // 4. Horner evaluation of `exp(r)` (`|r| <= ln(2)/2`) in `ymm2` using `r` in `ymm0`:
    emit_load_ymm_rbp(code, 2, mslot(MATH_EXP_C12));
    for &c_slot in &[
        MATH_EXP_C11,
        MATH_EXP_C10,
        MATH_EXP_C9,
        MATH_EXP_C8,
        MATH_EXP_C7,
        MATH_EXP_C6,
        MATH_EXP_C5,
        MATH_EXP_C4,
        MATH_EXP_C3,
        MATH_HALF_EXP,
        MATH_ONE,
        MATH_ONE,
    ] {
        emit_vex_fma_ymm_rm_rbp(code, 0xA8, 2, 0, mslot(c_slot));
    }
    // vmulpd ymm0, ymm2, ymm3 (`ymm0 = exp(r) * 2^k1`)
    emit_vex_ymm_rr(code, 0x59, 0, 2, 3);
    // vmulpd ymm0, ymm0, ymm1 (`ymm0 = (exp(r) * 2^k1) * 2^k2`)
    emit_vex_ymm_rr(code, 0x59, 0, 0, 1);

    // 5. IEEE-754 special cases using saved input `x` in `[rbp + scratch_disp]`:
    emit_load_ymm_rbp(code, 1, scratch_disp);
    // 5a. `x < -745.1332191019412` (including `-Inf`) -> `+0.0`:
    emit_vex_ymm_rm_rbp(code, 0xC2, 2, 1, mslot(MATH_EXP_CLAMP_LO));
    code.push(0x01); // _CMP_LT_OS
    emit_vex_ymm_rr(code, 0x55, 0, 2, 0); // vandnpd ymm0, ymm2, ymm0
                                          // 5b. `x > 709.782712893384` (including `+Inf`) -> `+Inf`:
    emit_load_ymm_rbp(code, 2, mslot(MATH_EXP_CLAMP_HI));
    emit_vex_ymm_rr(code, 0xC2, 2, 2, 1);
    code.push(0x01); // _CMP_LT_OS (`EXP_CLAMP_HI < x`)
    emit_vex_ymm_rr(code, 0x55, 0, 2, 0);
    emit_vex_ymm_rm_rbp(code, 0x54, 2, 2, mslot(MATH_EXP_MASK));
    emit_vex_ymm_rr(code, 0x56, 0, 0, 2);
    // 5c. `x.is_nan()` -> `QNaN`:
    emit_vex_ymm_rr(code, 0xC2, 2, 1, 1);
    code.push(0x03); // _CMP_UNORD_Q
    emit_vex_ymm_rr(code, 0x55, 0, 2, 0);
    emit_vex_ymm_rm_rbp(code, 0x54, 2, 2, mslot(MATH_QNAN));
    emit_vex_ymm_rr(code, 0x56, 0, 0, 2);

    // Restore `ymm2` and `ymm3`.
    emit_load_ymm_rbp(code, 2, mslot(MATH_SCRATCH_YMM2));
    emit_load_ymm_rbp(code, 3, mslot(MATH_SCRATCH_YMM3));
}

fn emit_load_ymm0_rbp(code: &mut Vec<u8>, disp: i32) {
    emit_load_ymm_rbp(code, 0, disp);
}

fn emit_store_ymm0_rbp(code: &mut Vec<u8>, disp: i32) {
    emit_store_ymm_rbp(code, 0, disp);
}

fn emit_load_xmm0_rbp(code: &mut Vec<u8>, disp: i32) {
    // movsd xmm0, [rbp + disp32]
    code.extend_from_slice(&[0xF2, 0x0F, 0x10, 0x85]);
    code.extend_from_slice(&disp.to_le_bytes());
}

fn emit_store_xmm0_rbp(code: &mut Vec<u8>, disp: i32) {
    // movsd [rbp + disp32], xmm0
    code.extend_from_slice(&[0xF2, 0x0F, 0x11, 0x85]);
    code.extend_from_slice(&disp.to_le_bytes());
}

fn emit_x87_ln_slot(code: &mut Vec<u8>, src_disp: i32, dst_disp: i32) {
    // fldln2
    code.extend_from_slice(&[0xD9, 0xED]);
    // fld qword ptr [rbp + src_disp]
    code.extend_from_slice(&[0xDD, 0x85]);
    code.extend_from_slice(&src_disp.to_le_bytes());
    // fyl2x (computes ln(2) * log2(x) = ln(x) and pops ST)
    code.extend_from_slice(&[0xD9, 0xF1]);
    // fstp qword ptr [rbp + dst_disp]
    code.extend_from_slice(&[0xDD, 0x9D]);
    code.extend_from_slice(&dst_disp.to_le_bytes());
}

fn emit_x87_exp_slot(code: &mut Vec<u8>, src_disp: i32, dst_disp: i32) {
    // fld qword ptr [rbp + src_disp]
    code.extend_from_slice(&[0xDD, 0x85]);
    code.extend_from_slice(&src_disp.to_le_bytes());
    // fldl2e; fmulp; fld st(0); frndint; fsub st(1), st(0); fxch st(1);
    // f2xm1; fld1; faddp; fscale; fstp st(1)
    code.extend_from_slice(&[
        0xD9, 0xEA, 0xDE, 0xC9, 0xD9, 0xC0, 0xD9, 0xFC, 0xDC, 0xE9, 0xD9, 0xC9, 0xD9, 0xF0, 0xD9,
        0xE8, 0xDE, 0xC1, 0xD9, 0xFD, 0xDD, 0xD9,
    ]);
    // fstp qword ptr [rbp + dst_disp]
    code.extend_from_slice(&[0xDD, 0x9D]);
    code.extend_from_slice(&dst_disp.to_le_bytes());
}

fn emit_sys_exit(code: &mut Vec<u8>, status: u32) {
    // mov edi, status (5 bytes)
    code.push(0xBF);
    code.extend_from_slice(&status.to_le_bytes());
    // mov eax, 60 (5 bytes)
    code.extend_from_slice(&[0xB8, 0x3C, 0x00, 0x00, 0x00]);
    // syscall (2 bytes) -> 12 bytes total
    code.extend_from_slice(&[0x0F, 0x05]);
}

fn patch_rel32(code: &mut [u8], patch_pos: usize, target_pos: usize) -> Result<(), AotError> {
    let next_ip = i64::try_from(patch_pos + 4).unwrap_or(0);
    let target = i64::try_from(target_pos).unwrap_or(0);
    let rel = i32::try_from(target - next_ip)
        .map_err(|_e| AotError::UnsupportedGraph("jump displacement exceeds rel32"))?;
    let bytes = rel.to_le_bytes();
    let dst = code
        .get_mut(patch_pos..patch_pos + 4)
        .ok_or(AotError::UnsupportedGraph("invalid rel32 patch offset"))?;
    dst.copy_from_slice(&bytes);
    Ok(())
}

fn write_elf64_headers(buf: &mut Vec<u8>, total_file_size: u64) {
    let entry_vaddr = ELF_BASE_VADDR + (ELF_CODE_OFFSET as u64);

    // 1. Elf64_Ehdr (64 bytes)
    buf.extend_from_slice(&[
        0x7F, b'E', b'L', b'F', // Magic
        2,    // EI_CLASS = ELFCLASS64
        1,    // EI_DATA = ELFDATA2LSB
        1,    // EI_VERSION = EV_CURRENT
        0,    // EI_OSABI = ELFOSABI_NONE
        0, 0, 0, 0, 0, 0, 0, 0, // EI_PAD
    ]);
    buf.extend_from_slice(&2u16.to_le_bytes()); // e_type = ET_EXEC (2)
    buf.extend_from_slice(&0x3Eu16.to_le_bytes()); // e_machine = EM_X86_64 (62)
    buf.extend_from_slice(&1u32.to_le_bytes()); // e_version = EV_CURRENT (1)
    buf.extend_from_slice(&entry_vaddr.to_le_bytes()); // e_entry
    buf.extend_from_slice(&(ELF_EHDR_SIZE as u64).to_le_bytes()); // e_phoff = 64
    buf.extend_from_slice(&0u64.to_le_bytes()); // e_shoff = 0
    buf.extend_from_slice(&0u32.to_le_bytes()); // e_flags = 0
    buf.extend_from_slice(&(ELF_EHDR_SIZE as u16).to_le_bytes()); // e_ehsize = 64
    buf.extend_from_slice(&(ELF_PHDR_SIZE as u16).to_le_bytes()); // e_phentsize = 56
    buf.extend_from_slice(&1u16.to_le_bytes()); // e_phnum = 1
    buf.extend_from_slice(&0u16.to_le_bytes()); // e_shentsize = 0
    buf.extend_from_slice(&0u16.to_le_bytes()); // e_shnum = 0
    buf.extend_from_slice(&0u16.to_le_bytes()); // e_shstrndx = 0

    // 2. Elf64_Phdr (56 bytes) — Single PT_LOAD segment (PF_R | PF_X = 5)
    buf.extend_from_slice(&1u32.to_le_bytes()); // p_type = PT_LOAD (1)
    buf.extend_from_slice(&5u32.to_le_bytes()); // p_flags = PF_R | PF_X (5)
    buf.extend_from_slice(&0u64.to_le_bytes()); // p_offset = 0
    buf.extend_from_slice(&ELF_BASE_VADDR.to_le_bytes()); // p_vaddr
    buf.extend_from_slice(&ELF_BASE_VADDR.to_le_bytes()); // p_paddr
    buf.extend_from_slice(&total_file_size.to_le_bytes()); // p_filesz
    buf.extend_from_slice(&total_file_size.to_le_bytes()); // p_memsz
    buf.extend_from_slice(&0x1000u64.to_le_bytes()); // p_align = 4096
}

fn compute_live_nodes(program: &RulProgram) -> Vec<bool> {
    let n = program.nodes.len();
    let mut live = vec![false; n];
    let out_idx = program.output_node as usize;
    if let Some(slot) = live.get_mut(out_idx) {
        *slot = true;
    }

    for idx in (0..n).rev() {
        if !live.get(idx).copied().unwrap_or(false) {
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
            | Opcode::CallArg => {}
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
            | Opcode::BrIf => {
                if let Some(a) = live.get_mut(frame.in_a as usize) {
                    *a = true;
                }
            }
            Opcode::ArenaMutate
            | Opcode::LocalSet
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
