//! Zero-Runtime Standalone Native AOT Compiler (`rul compile`).
//!
//! Compiles a verified `.rl` binary instruction graph directly into a standalone,
//! statically-linked Linux `x86_64` `ELF64` executable in microseconds (`< 1 ms`)
//! using Copy-and-Patch machine code stencils — requiring **zero LLVM passes**,
//! **zero `libc`**, and **zero Rul runtime** at execution time.

use std::fmt;
use std::fs::{self, File};
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use crate::bytecode::{ElemType, Opcode, RulProgram};
use crate::verifier::{verify_program, VerifyDiagnostic};

/// Base virtual address for the standalone `x86_64` `ELF64` executable.
const ELF_BASE_VADDR: u64 = 0x0040_0000;
/// Size of the 64-bit ELF file header (`Elf64_Ehdr`).
const ELF_EHDR_SIZE: usize = 64;
/// Size of a single 64-bit ELF program header (`Elf64_Phdr`).
const ELF_PHDR_SIZE: usize = 56;
/// Offset where machine code entry `_start` begins (`64 + 56 = 120`).
const ELF_CODE_OFFSET: usize = ELF_EHDR_SIZE + ELF_PHDR_SIZE;

/// Exit status code emitted by the standalone ELF binary when `AssertContract` fails (`NaN`/`Inf`).
pub const AOT_EXIT_CONTRACT_FAILED: i32 = 70;
/// Exit status code emitted by the standalone ELF binary when `Gather` index is out of bounds.
pub const AOT_EXIT_GATHER_OOB: i32 = 71;

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
    /// `Gather` index out-of-bounds detected by the native binary.
    GatherOutOfBounds,
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
                write!(f, "AOT native binary detected out-of-bounds Gather index")
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
    verify_program(program).map_err(AotError::VerificationFailed)?;
    let verify_ns = u64::try_from(t_ver.elapsed().as_nanos()).unwrap_or(u64::MAX);

    let t_cg = Instant::now();
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
            Opcode::ReduceSum
            | Opcode::ReduceMax
            | Opcode::ReduceMin
            | Opcode::RingEnqueue
            | Opcode::AssertContract => {}
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

// =========================================================================
// Rul AOT x86_64 Register Contract (`RegisterContract`)
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
    let raw_frame = 128usize
        .saturating_add(node_count.saturating_mul(32))
        .saturating_add(reductions.len().saturating_mul(64));
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
                Opcode::Gather
                | Opcode::ReduceSum
                | Opcode::ReduceMax
                | Opcode::ReduceMin
                | Opcode::RingEnqueue
                | Opcode::AssertContract => {}
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
                        emit_store_ymm_rbp(&mut code, ra, disp_a);
                    }
                    free_if_dying(a, idx, &last_use, &mut free_ymm_mask, &mut node_ymm);
                    for lane_off in [0i32, 8, 16, 24] {
                        emit_x87_ln_slot(
                            &mut code,
                            disp_a.saturating_add(lane_off),
                            disp.saturating_add(lane_off),
                        );
                    }
                    let dst_reg = alloc_ymm_slot(idx, &mut free_ymm_mask, &mut node_ymm);
                    if dst_reg != 0 {
                        emit_load_ymm_rbp(&mut code, dst_reg, disp);
                    }
                }
                Opcode::Exp => {
                    if let Some(ra) = node_ymm.get(a).copied().flatten() {
                        emit_store_ymm_rbp(&mut code, ra, disp_a);
                    }
                    free_if_dying(a, idx, &last_use, &mut free_ymm_mask, &mut node_ymm);
                    for lane_off in [0i32, 8, 16, 24] {
                        emit_x87_exp_slot(
                            &mut code,
                            disp_a.saturating_add(lane_off),
                            disp.saturating_add(lane_off),
                        );
                    }
                    let dst_reg = alloc_ymm_slot(idx, &mut free_ymm_mask, &mut node_ymm);
                    if dst_reg != 0 {
                        emit_load_ymm_rbp(&mut code, dst_reg, disp);
                    }
                }
                Opcode::Gather
                | Opcode::ReduceSum
                | Opcode::ReduceMax
                | Opcode::ReduceMin
                | Opcode::RingEnqueue
                | Opcode::AssertContract => {}
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
            | Opcode::AssertContract => {}
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
        }
    }
    live
}
