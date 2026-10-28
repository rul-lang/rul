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

use crate::bytecode::{Opcode, RulProgram};
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
}

/// Compiles a verified [`RulProgram`] into an in-memory Linux `x86_64` `ELF64` executable image.
///
/// The resulting binary has **zero runtime dependencies** (`statically linked`, no `libc`, no VM)
/// and executes directly on the OS kernel.
///
/// # Errors
/// Returns [`AotError`] if $O(N)$ verification fails or the graph has no reduction output.
pub fn compile_to_elf64_bytes(
    program: &RulProgram,
) -> Result<(Vec<u8>, AotCompileReport), AotError> {
    let t_ver = Instant::now();
    verify_program(program).map_err(AotError::VerificationFailed)?;
    let verify_ns = u64::try_from(t_ver.elapsed().as_nanos()).unwrap_or(u64::MAX);

    let t_cg = Instant::now();
    let live = compute_live_nodes(program);

    // Locate the terminal reduction node (`ReduceSum` or `ReduceMax`) and optional `AssertContract`.
    let mut reduce_node: Option<(usize, usize, AotReduction)> = None;
    let mut has_assert_contract = false;
    let mut domain_start: u64 = 0;
    let mut domain_end: Option<u64> = None;
    let mut filter_mask_node: Option<usize> = None;
    let mut has_gather = false;

    for (idx, frame) in program.nodes.iter().enumerate() {
        if !live.get(idx).copied().unwrap_or(false) {
            continue;
        }
        match frame.opcode {
            Opcode::ReduceSum => {
                reduce_node = Some((idx, frame.in_a as usize, AotReduction::Sum));
            }
            Opcode::ReduceMax => {
                reduce_node = Some((idx, frame.in_a as usize, AotReduction::Max));
            }
            Opcode::AssertContract => {
                has_assert_contract = true;
            }
            Opcode::ArenaSlice => {
                let proof = program
                    .proofs
                    .get(usize::from(frame.proof_idx.saturating_sub(1)))
                    .ok_or(AotError::UnsupportedGraph("missing slice ProofEntry"))?;
                domain_start = u64::from(proof.bound_lo);
                domain_end = Some(u64::from(proof.bound_hi));
            }
            Opcode::FilterCompact => {
                filter_mask_node = Some(frame.in_a as usize);
            }
            Opcode::Gather => {
                has_gather = true;
            }
            _ => {}
        }
    }

    let (reduce_idx, reduce_input_idx, reduction) = reduce_node.ok_or(
        AotError::UnsupportedGraph("AOT binary requires a ReduceSum or ReduceMax node"),
    )?;

    // Compute byte offsets for each Arena inside the flat arena input file.
    let mut arena_byte_offsets = Vec::with_capacity(program.arenas.len());
    let mut total_arena_bytes: u64 = 0;
    for arena in &program.arenas {
        arena_byte_offsets.push(total_arena_bytes);
        total_arena_bytes =
            total_arena_bytes.saturating_add(u64::from(arena.element_count).saturating_mul(8));
        if domain_end.is_none() {
            domain_end = Some(u64::from(arena.element_count));
        }
    }
    let final_domain_end = domain_end.unwrap_or(1);
    let avx2_vectorized = !has_gather
        && filter_mask_node.is_none()
        && final_domain_end.saturating_sub(domain_start) >= 4;

    let code = emit_x86_64_machine_code(
        program,
        &live,
        reduce_idx,
        reduce_input_idx,
        reduction,
        has_assert_contract,
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
    let output = Command::new(exe_path.as_ref())
        .arg(data_path.as_ref())
        .output()
        .map_err(AotError::Io)?;

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

#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "self-contained linear x86_64 AVX2 + scalar stencil emitter"
)]
fn emit_x86_64_machine_code(
    program: &RulProgram,
    live: &[bool],
    reduce_idx: usize,
    reduce_input_idx: usize,
    reduction: AotReduction,
    has_assert_contract: bool,
    domain_start: u64,
    domain_end: u64,
    filter_mask_node: Option<usize>,
    avx2_vectorized: bool,
    arena_byte_offsets: &[u64],
    total_arena_bytes: u64,
) -> Result<Vec<u8>, AotError> {
    let mut code: Vec<u8> = Vec::with_capacity(768);

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
    // [rbp - 64 .. rbp - 8]   : 8 f64 accumulator lanes (`acc[0..8]` = 2 x 256-bit YMM vectors)
    // [rbp - 72]              : 8-byte stdout staging slot
    // [rbp - 128 - idx * 32]  : 32-byte (256-bit YMM / 64-bit XMM) SSA slot for each node `idx`
    let raw_frame = 128usize.saturating_add(program.nodes.len().saturating_mul(32));
    let frame_bytes = u32::try_from(raw_frame.div_ceil(32) * 32).unwrap_or(8192);
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

    // 4. Initialize reduction accumulators.
    match reduction {
        AotReduction::Sum => {
            // xor eax, eax
            code.extend_from_slice(&[0x31, 0xC0]);
            for lane in 0..8i32 {
                let acc_disp = -64 + lane * 8;
                // mov [rbp + acc_disp], rax
                code.extend_from_slice(&[0x48, 0x89, 0x85]);
                code.extend_from_slice(&acc_disp.to_le_bytes());
            }
            // xorpd xmm7, xmm7
            code.extend_from_slice(&[0x66, 0x0F, 0x57, 0xFF]);
            // xor r11d, r11d (active element/vector counter)
            code.extend_from_slice(&[0x45, 0x31, 0xDB]);
        }
        AotReduction::Max => {
            let neg_inf = f64::NEG_INFINITY.to_bits();
            // movabs rax, neg_inf
            code.extend_from_slice(&[0x48, 0xB8]);
            code.extend_from_slice(&neg_inf.to_le_bytes());
            for lane in 0..4i32 {
                let acc_disp = -64 + lane * 8;
                // mov [rbp + acc_disp], rax
                code.extend_from_slice(&[0x48, 0x89, 0x85]);
                code.extend_from_slice(&acc_disp.to_le_bytes());
            }
            // movq xmm7, rax
            code.extend_from_slice(&[0x66, 0x48, 0x0F, 0x6E, 0xF8]);
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
    //     alternating between the two 256-bit accumulators `[rbp - 64]` and `[rbp - 32]`).
    if avx2_vectorized {
        let avx2_loop_top = code.len();
        // lea rax, [rcx + 4]
        code.extend_from_slice(&[0x48, 0x8D, 0x41, 0x04]);
        // cmp rax, rbx
        code.extend_from_slice(&[0x48, 0x39, 0xD8]);
        // ja avx2_loop_exit (0F 87 <rel32>)
        code.extend_from_slice(&[0x0F, 0x87, 0x00, 0x00, 0x00, 0x00]);
        let ja_avx2_exit_patch_pos = code.len() - 4;

        for (idx, frame) in program.nodes.iter().enumerate() {
            if idx >= reduce_idx || !live.get(idx).copied().unwrap_or(false) {
                continue;
            }
            let disp = node_disp(idx);
            let disp_a = node_disp(frame.in_a as usize);
            let disp_b = node_disp(frame.in_b as usize);

            match frame.opcode {
                Opcode::ConstF64 => {
                    // Already pre-broadcast across all 4 lanes via LICM.
                }
                Opcode::ArenaInput => {
                    let arena_off = arena_byte_offsets
                        .get(frame.in_a as usize)
                        .copied()
                        .unwrap_or(0);
                    // mov rax, r12
                    code.extend_from_slice(&[0x4C, 0x89, 0xE0]);
                    if arena_off > 0 {
                        // movabs rdx, arena_off; add rax, rdx
                        code.extend_from_slice(&[0x48, 0xBA]);
                        code.extend_from_slice(&arena_off.to_le_bytes());
                        code.extend_from_slice(&[0x48, 0x01, 0xD0]);
                    }
                    // vmovupd ymm0, ymmword ptr [rax + rcx * 8]
                    code.extend_from_slice(&[0xC5, 0xFD, 0x10, 0x04, 0xC8]);
                    emit_store_ymm0_rbp(&mut code, disp);
                }
                Opcode::ArenaSlice => {
                    emit_load_ymm0_rbp(&mut code, disp_a);
                    emit_store_ymm0_rbp(&mut code, disp);
                }
                Opcode::ArenaMutate | Opcode::FilterCompact => {
                    emit_load_ymm0_rbp(&mut code, disp_b);
                    emit_store_ymm0_rbp(&mut code, disp);
                }
                Opcode::Add => {
                    emit_load_ymm0_rbp(&mut code, disp_a);
                    // vaddpd ymm0, ymm0, ymmword ptr [rbp + disp_b]
                    code.extend_from_slice(&[0xC5, 0xFD, 0x58, 0x85]);
                    code.extend_from_slice(&disp_b.to_le_bytes());
                    emit_store_ymm0_rbp(&mut code, disp);
                }
                Opcode::Sub => {
                    emit_load_ymm0_rbp(&mut code, disp_a);
                    // vsubpd ymm0, ymm0, ymmword ptr [rbp + disp_b]
                    code.extend_from_slice(&[0xC5, 0xFD, 0x5C, 0x85]);
                    code.extend_from_slice(&disp_b.to_le_bytes());
                    emit_store_ymm0_rbp(&mut code, disp);
                }
                Opcode::Mul => {
                    emit_load_ymm0_rbp(&mut code, disp_a);
                    // vmulpd ymm0, ymm0, ymmword ptr [rbp + disp_b]
                    code.extend_from_slice(&[0xC5, 0xFD, 0x59, 0x85]);
                    code.extend_from_slice(&disp_b.to_le_bytes());
                    emit_store_ymm0_rbp(&mut code, disp);
                }
                Opcode::CmpGt => {
                    // Load `b` into ymm0 and `a` into ymm1; `vcmpltpd ymm0, ymm0, ymm1` computes `b < a` (`a > b`).
                    emit_load_ymm0_rbp(&mut code, disp_b);
                    // vmovupd ymm1, ymmword ptr [rbp + disp_a]
                    code.extend_from_slice(&[0xC5, 0xFD, 0x10, 0x8D]);
                    code.extend_from_slice(&disp_a.to_le_bytes());
                    // vcmpltpd ymm0, ymm0, ymm1 (C5 FD C2 C1 01)
                    code.extend_from_slice(&[0xC5, 0xFD, 0xC2, 0xC1, 0x01]);
                    emit_store_ymm0_rbp(&mut code, disp);
                }
                Opcode::SelectZero => {
                    emit_load_ymm0_rbp(&mut code, disp_a);
                    // vmovupd ymm1, ymmword ptr [rbp + disp_b]
                    code.extend_from_slice(&[0xC5, 0xFD, 0x10, 0x8D]);
                    code.extend_from_slice(&disp_b.to_le_bytes());
                    // vandpd ymm0, ymm0, ymm1 (C5 FD 54 C1)
                    code.extend_from_slice(&[0xC5, 0xFD, 0x54, 0xC1]);
                    emit_store_ymm0_rbp(&mut code, disp);
                }
                Opcode::Gather
                | Opcode::ReduceSum
                | Opcode::ReduceMax
                | Opcode::RingEnqueue
                | Opcode::AssertContract => {}
            }
        }

        // Accumulate `reduce_input_idx` (`ymm0`) into the 256-bit reduction state.
        let disp_red_in = node_disp(reduce_input_idx);
        emit_load_ymm0_rbp(&mut code, disp_red_in);

        match reduction {
            AotReduction::Sum => {
                // Alternate between `[rbp - 64]` (even vector step: lanes 0..4)
                // and `[rbp - 32]` (odd vector step: lanes 4..8) to preserve exact 8-lane order.
                // mov rdx, r11
                code.extend_from_slice(&[0x4C, 0x89, 0xDA]);
                // and edx, 1
                code.extend_from_slice(&[0x83, 0xE2, 0x01]);
                // shl rdx, 5 (rdx = 0 or 32)
                code.extend_from_slice(&[0x48, 0xC1, 0xE2, 0x05]);
                // inc r11
                code.extend_from_slice(&[0x49, 0xFF, 0xC3]);
                // vaddpd ymm0, ymm0, ymmword ptr [rbp + rdx * 1 - 64]
                code.extend_from_slice(&[0xC5, 0xFD, 0x58, 0x44, 0x15, 0xC0]);
                // vmovupd ymmword ptr [rbp + rdx * 1 - 64], ymm0
                code.extend_from_slice(&[0xC5, 0xFD, 0x11, 0x44, 0x15, 0xC0]);
            }
            AotReduction::Max => {
                // vmaxpd ymm0, ymm0, ymmword ptr [rbp - 64]
                code.extend_from_slice(&[0xC5, 0xFD, 0x5F, 0x45, 0xC0]);
                // vmovupd ymmword ptr [rbp - 64], ymm0
                code.extend_from_slice(&[0xC5, 0xFD, 0x11, 0x45, 0xC0]);
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

        // Clear upper 128 bits of all YMM registers (`vzeroupper`) before scalar tail/reduction.
        code.extend_from_slice(&[0xC5, 0xF8, 0x77]);

        // Convert `r11` from 4-element vector count to scalar element index (`shl r11, 2`)
        // so any scalar tail elements land in the exact matching `(r11 & 7)` lane.
        if reduction == AotReduction::Sum {
            // shl r11, 2
            code.extend_from_slice(&[0x49, 0xC1, 0xE3, 0x02]);
        }
    }

    // 5b. Scalar Cleanup / Indirect-Gather / FilterCompact Loop (`rcx` .. `rbx`).
    let loop_top = code.len();
    // cmp rcx, rbx
    code.extend_from_slice(&[0x48, 0x39, 0xD9]);
    // jae loop_exit (0F 83 <rel32>)
    code.extend_from_slice(&[0x0F, 0x83, 0x00, 0x00, 0x00, 0x00]);
    let jae_exit_patch_pos = code.len() - 4;

    // Emit native stencils for each live node up to `reduce_idx`.
    for (idx, frame) in program.nodes.iter().enumerate() {
        if idx >= reduce_idx || !live.get(idx).copied().unwrap_or(false) {
            continue;
        }
        let disp = node_disp(idx);
        let disp_a = node_disp(frame.in_a as usize);
        let disp_b = node_disp(frame.in_b as usize);

        match frame.opcode {
            Opcode::ConstF64 => {
                // Hoisted above the loop via LICM (0 instructions in loop).
            }
            Opcode::ArenaInput => {
                let arena_off = arena_byte_offsets
                    .get(frame.in_a as usize)
                    .copied()
                    .unwrap_or(0);
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
            Opcode::ArenaSlice => {
                emit_load_xmm0_rbp(&mut code, disp_a);
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
            Opcode::SelectZero => {
                emit_load_xmm0_rbp(&mut code, disp_a);
                // movsd xmm1, [rbp + disp_b]
                code.extend_from_slice(&[0xF2, 0x0F, 0x10, 0x8D]);
                code.extend_from_slice(&disp_b.to_le_bytes());
                // andpd xmm0, xmm1 (branchless bitmask select)
                code.extend_from_slice(&[0x66, 0x0F, 0x54, 0xC1]);
                emit_store_xmm0_rbp(&mut code, disp);
            }
            Opcode::Gather => {
                let proof = program
                    .proofs
                    .get(usize::from(frame.proof_idx.saturating_sub(1)))
                    .ok_or(AotError::UnsupportedGraph("missing gather ProofEntry"))?;
                let target_off = arena_byte_offsets
                    .get(usize::from(proof.target_arena))
                    .copied()
                    .unwrap_or(0);
                let bound_hi = u64::from(proof.bound_hi);

                // Load index f64 into xmm0: movsd xmm0, [rbp + disp_b]
                emit_load_xmm0_rbp(&mut code, disp_b);
                // Convert f64 to signed 64-bit integer in rdx: cvttsd2si rdx, xmm0
                code.extend_from_slice(&[0xF2, 0x48, 0x0F, 0x2C, 0xD0]);
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
            | Opcode::RingEnqueue
            | Opcode::AssertContract => {}
        }
    }

    // 6. Accumulate `reduce_input_idx` into the reduction state.
    let disp_red_in = node_disp(reduce_input_idx);
    emit_load_xmm0_rbp(&mut code, disp_red_in);

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

    match reduction {
        AotReduction::Sum => {
            // rdx = r11 & 7; r11 += 1
            // mov rdx, r11
            code.extend_from_slice(&[0x4C, 0x89, 0xDA]);
            // and edx, 7
            code.extend_from_slice(&[0x83, 0xE2, 0x07]);
            // inc r11
            code.extend_from_slice(&[0x49, 0xFF, 0xC3]);
            // addsd xmm0, [rbp + rdx * 8 - 64]
            code.extend_from_slice(&[0xF2, 0x0F, 0x58, 0x44, 0xD5, 0xC0]);
            // movsd [rbp + rdx * 8 - 64], xmm0
            code.extend_from_slice(&[0xF2, 0x0F, 0x11, 0x44, 0xD5, 0xC0]);
        }
        AotReduction::Max => {
            // maxsd xmm7, xmm0
            code.extend_from_slice(&[0xF2, 0x0F, 0x5F, 0xF8]);
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

    // 7. Post-loop reduction across accumulator lanes:
    match reduction {
        AotReduction::Sum => {
            // ((acc[0] + acc[1]) + (acc[2] + acc[3])) + (acc[4] + acc[5]) + (acc[6] + acc[7]) -> xmm7
            // movsd xmm0, [rbp - 64]
            code.extend_from_slice(&[0xF2, 0x0F, 0x10, 0x45, 0xC0]);
            // addsd xmm0, [rbp - 56]
            code.extend_from_slice(&[0xF2, 0x0F, 0x58, 0x45, 0xC8]);
            // movsd xmm1, [rbp - 48]
            code.extend_from_slice(&[0xF2, 0x0F, 0x10, 0x4D, 0xD0]);
            // addsd xmm1, [rbp - 40]
            code.extend_from_slice(&[0xF2, 0x0F, 0x58, 0x4D, 0xD8]);
            // addsd xmm0, xmm1
            code.extend_from_slice(&[0xF2, 0x0F, 0x58, 0xC1]);
            // movsd xmm1, [rbp - 32]
            code.extend_from_slice(&[0xF2, 0x0F, 0x10, 0x4D, 0xE0]);
            // addsd xmm1, [rbp - 24]
            code.extend_from_slice(&[0xF2, 0x0F, 0x58, 0x4D, 0xE8]);
            // addsd xmm0, xmm1
            code.extend_from_slice(&[0xF2, 0x0F, 0x58, 0xC1]);
            // movsd xmm1, [rbp - 16]
            code.extend_from_slice(&[0xF2, 0x0F, 0x10, 0x4D, 0xF0]);
            // addsd xmm1, [rbp - 8]
            code.extend_from_slice(&[0xF2, 0x0F, 0x58, 0x4D, 0xF8]);
            // addsd xmm0, xmm1
            code.extend_from_slice(&[0xF2, 0x0F, 0x58, 0xC1]);
            // movapd xmm7, xmm0
            code.extend_from_slice(&[0x66, 0x0F, 0x28, 0xF8]);
        }
        AotReduction::Max => {
            // Merge the 4 AVX2 `vmaxpd` lanes `[rbp - 64 .. rbp - 40]` into `xmm7`
            for disp8 in [0xC0u8, 0xC8, 0xD0, 0xD8] {
                // maxsd xmm7, [rbp + disp8]
                code.extend_from_slice(&[0xF2, 0x0F, 0x5F, 0x7D, disp8]);
            }
        }
    }

    // Store final scalar `xmm7` to `[rbp - 72]` (`disp8 = 0xB8`):
    // movsd [rbp - 72], xmm7
    code.extend_from_slice(&[0xF2, 0x0F, 0x11, 0x7D, 0xB8]);

    // 8. Optional `AssertContract` check: verify that the IEEE-754 f64 result is finite.
    if has_assert_contract {
        // mov rax, [rbp - 72]
        code.extend_from_slice(&[0x48, 0x8B, 0x45, 0xB8]);
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

    // 9. Write the 8-byte little-endian `f64` result to `stdout` (`SYS_write = 1`, `fd = 1`) and `SYS_exit(0)`.
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

fn node_disp(node_idx: usize) -> i32 {
    let idx_i32 = i32::try_from(node_idx).unwrap_or(0);
    (-128i32).saturating_sub(idx_i32.saturating_mul(32))
}

fn emit_load_ymm0_rbp(code: &mut Vec<u8>, disp: i32) {
    // vmovupd ymm0, ymmword ptr [rbp + disp32]
    code.extend_from_slice(&[0xC5, 0xFD, 0x10, 0x85]);
    code.extend_from_slice(&disp.to_le_bytes());
}

fn emit_store_ymm0_rbp(code: &mut Vec<u8>, disp: i32) {
    // vmovupd ymmword ptr [rbp + disp32], ymm0
    code.extend_from_slice(&[0xC5, 0xFD, 0x11, 0x85]);
    code.extend_from_slice(&disp.to_le_bytes());
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
            | Opcode::ReduceSum
            | Opcode::ReduceMax
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
            | Opcode::CmpGt
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
