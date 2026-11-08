# Changelog

All notable changes to the **Rul** (`.rl`) binary representation, verifier, 3-Tier execution engine, and `ELF64` AOT compiler are documented in this file.

## [Unreleased] — Phase P0 (Safety & Correctness), Phase P1 (AOT `YMM` Register Allocation) & Phase P1b (`Opcode::Fma` & AVX2 Vectorized `Ln`/`Exp`)

### Added (Phase P1b — 30-Opcode ISA: `Opcode::Fma` & Vectorized 256-Bit AVX2 `Ln`/`Exp`)
- **Explicit `Opcode::Fma = 0x27` (`src/bytecode.rs`, `src/verifier.rs`, `src/transpiler.rs`, `src/engine.rs`, `src/aot.rs`)**:
  - Added single-rounded IEEE-754 fused multiply-add (`out[i] = fma(a[i], b[i], c[i]) = a[i] * b[i] + c[i]`) as the 30th opcode in the 128-bit `NodeFrame` ISA, packing ternary operands `(b: u16, c: u16)` into `in_b: u32` (`NodeFrame::encode_fma_bc` / `NodeFrame::decode_fma_bc`) while preserving the fixed 16-byte frame layout.
  - Added `.rls` $\leftrightarrow$ `.rl` assembly/decompilation/disassembly support (`node $dst = Fma f64x8 a=$a b=$b c=$c`, `SymbolicGraphBuilder::emit_fma_op`).
  - Lowered `Opcode::Fma` in `src/aot.rs` to 256-bit AVX2 FMA3 (`vfmadd213pd` / `vfmadd231pd`) in the `5a` vector loop and 64-bit FMA3 (`vfmadd213sd`) in the `5b` scalar tail and `8` post-reduction scalar epilogue.
  - **Measured Speedup (`BM-HORNER-50`, 50-step Horner polynomial, 10,000,000 `f64` elements = 80 MB SoA)**:
    - Explicit `Opcode::Fma` cuts the graph from 104 SSA nodes (`Mul+Add`) to 54 SSA nodes (`Fma`), shrinking the standalone `ELF64` binary from **3,872 B** to **2,877 B** (`2,757 B` machine code, compiled in **`10.07 us`**).
    - Execution time on 10,000,000 elements drops to **40.34 ms median (`34.74 ms` min, `1.98 GB/s`)** — **`3.33x` faster** than `[A] Hand-Written C/Rust 8-Acc Horner-50` (**134.24 ms**), **`2.70x` faster** than `Mul+Add` AOT (**108.75 ms**), and **`9.81x` faster** than the pre-P1 stack-slot AOT baseline (**396 ms**).
- **Vectorized 256-Bit AVX2 `Opcode::Ln` and `Opcode::Exp` (`src/aot.rs`)**:
  - Replaced the 4-way scalar `x87` (`fyl2x` / `f2xm1`) calls inside the 256-bit AVX2 loop (`5a`) with branchless 4-lane AVX2 + FMA3 polynomial evaluation (`emit_avx2_ln_ymm` using `fdlibm` rational reduction `s = (m - 1) / (m + 1)` + `Lg1..Lg7` and Cody-Waite `LN2_HI`/`LN2_LO`; `emit_avx2_exp_ymm` using Cody-Waite range reduction + degree-12 FMA3 Horner evaluation) with full IEEE-754 special-case masking (`Ln`: `0 -> -Inf`, `< 0 -> NaN`, `+Inf -> +Inf`, `NaN -> NaN`; `Exp`: `< -708.3964 -> 0.0`, `> 709.7827 -> +Inf`, `NaN -> NaN`).
  - Math constant slots (`35 x 32 B = 1,120 B`) are allocated and pre-broadcast only when `needs_vec_math` is true (`has_live_ln || has_live_exp`), incurring zero frame or code size overhead for non-transcendental programs.
  - **Measured Speedup (`BM-TRANSCENDENTAL-LN-EXP`, `sum(exp(ln(x) * 0.5))` on 1,000,000 `f64` elements = 8 MB SoA)**:
    - Standalone `ELF64` binary (`2,760 B`, compiled in **`13.00 us`**) executes in **4.44 ms median (`4.23 ms` min, `1.80 GB/s`)** — **`2.26x` faster** than `Rust/glibc` scalar `x.ln() + exp()` (**10.02 ms**) while matching `libm` output to `< 1e-13` relative error (`2.202161542e6`).

### Breaking / Tightened Verifier Semantics
- **`src/verifier.rs` (`ArenaSlice` & `Gather` Effective Target Bound Check)**:
  - Tightened `ProofEntry` validation for `Opcode::ArenaSlice` (`proof.rule_kind == 1`) and `Opcode::Gather` (`proof.rule_kind == 2`).
  - Previously, `proof.bound_hi` was checked only against `target_arena.element_count`. When the target operand `in_a` was an `ArenaSlice` of length `n < target_arena.element_count`, a proof with `n < bound_hi <= target_arena.element_count` was accepted by the verifier.
  - Now, `proof.bound_hi` is checked against `effective_target_len = n.min(target_arena.element_count)`, rejecting any `ArenaSlice` or `Gather` whose `bound_hi` exceeds the sliced target column length with `VerifyErrorKind::InvalidRefinementProof`.

### Fixed (Phase P0 — AOT & CLI Safety)
- **`src/aot.rs` (Per-Column `slice_offset` & `effective_len` Propagation)**:
  - Replaced global `domain_start` overwrite on `ArenaSlice` with per-column `AotColMeta { arena_id, slice_offset, effective_len }` tracking across `ArenaInput`, `ArenaSlice` (including nested `ArenaSlice(ArenaSlice)`), and `Gather(ArenaSlice)`.
  - `Gather` now validates runtime index `rax < target_meta.effective_len` (`cmp rax, r9; jae exit(67)`) and addresses `target_base + (target_meta.slice_offset + rax) * 8`, eliminating cross-arena out-of-bounds reads on sliced arenas.
- **`src/aot.rs` (Zero-SIGBUS Arena File Size Verification)**:
  - Added `SYS_lseek(fd, 0, SEEK_END)` file-size verification before `SYS_mmap` in standalone `ELF64` binaries so truncated or short arena data files exit cleanly with code `66` instead of crashing with `SIGBUS`.
- **`src/aot.rs` (Exact IEEE-754 Integer Validation in `Gather`)**:
  - Added `roundsd xmm1, xmm0, 3` (`round toward zero`) + `ucomisd xmm0, xmm1` (`jne` / `jp` to `exit(67)`) before `cvttsd2si rax, xmm0` in both the AVX2 4-lane loop and SSE2 scalar tail, rejecting fractional (`1.5`), negative (`-0.5`), out-of-range (`1e20`), and `NaN` indices identically to Tier 0/1/2 while accepting `-0.0` as index `0`.
- **`src/bin/rul.rs` (1 GiB Arena Data File Limit)**:
  - Separated `MAX_ARENA_DATA_BYTES = 1 GiB` (`1,073,741,824` bytes) for `arena_*.bin` columnar inputs from `MAX_RL_FILE_BYTES = 16 MiB` for `.rl` bytecode files.

### Performance (Phase P1 — Linear-Scan `YMM` Register Allocator & Vectorized `Abs`)
- **`src/aot.rs` (Linear-Scan `YMM` Register Allocation)**:
  - Computed exact per-node `last_use` lifetimes inside the elementwise loop and allocated 12 caller-saved `YMM` registers (`ymm2..ymm13`) across `ConstF64` (up to 4 hot constants pre-broadcast into `YMM` before the loop), `ArenaInput`, `ArenaSlice`, `Add`, `Sub`, `Mul`, `Div`, `Min`, `Max`, `Sqrt`, `Floor`, `Ceil`, `Abs`, `CmpGt`, `CmpLt`, `CmpEq`, `MaskAnd`, `MaskOr`, and `SelectZero`, with automatic stack-frame spill/reload when more than 12 columns are simultaneously live.
  - Vectorized `Opcode::Abs` in the AVX2 loop into a single 256-bit `vandpd` instruction against a pre-broadcast `0x7FFF_FFFF_FFFF_FFFF` sign-clear mask at `[rbp - 64]`.
  - **Measured Speedup (`BM-HORNER-50`, 50-step Horner polynomial, 104 SSA nodes, 10,000,000 `f64` elements = 80 MB SoA)**:
    - AOT loop instructions per 4-lane vector iteration dropped from **303 (`vmovupd` stack round-trips)** to **103 (`vmulpd`/`vaddpd` register-to-register)**.
    - Standalone `ELF64` binary size for `BM-HORNER-50` shrank from **5,312 B (`main`)** to **3,872 B** (compile latency `~20.4 us`).
    - Execution time on 10,000,000 elements dropped from **396 ms (`main`)** to **106.58 ms (`3.72x`..`3.96x` speedup)**, matching `[A] Hand-Written C/Rust 8-Acc Horner-50` (**107.96 ms**, **`1.01x`**) and `gcc -O3 -mavx2 -mno-fma` (**94 ms**).

---

## [0.4.0] — 2026-11-03

### Added
- **29-Opcode ISA (`src/bytecode.rs`, `src/verifier.rs`, `src/transpiler.rs`, `src/engine.rs`, `src/aot.rs`)**:
  - Added unary math opcodes: `Abs` (`0x1A`), `Sqrt` (`0x1B`), `Floor` (`0x1C`), `Ceil` (`0x1D`), `Ln` (`0x1E`), and `Exp` (`0x1F`), including hardware x87 transcendental stencils (`fyl2x`, `f2xm1`, `fscale`) in `src/aot.rs`.
  - Added binary boolean mask logic opcodes: `MaskAnd` (`0x25`) and `MaskOr` (`0x26`).
  - Added horizontal minimum reduction opcode: `ReduceMin` (`0x24`).
- **Multi-Reduction and Post-Reduction Scalar DAGs (`src/engine.rs`, `src/aot.rs`)**:
  - Extended Tier 2 fused execution and standalone `ELF64` AOT compilation to evaluate multiple reductions in a single pass and compute arbitrary scalar DAG expressions after reduction.
- **Community Governance Documents**:
  - Added `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `SECURITY.md`.

---

## [0.3.1] — 2026-11-02

### Fixed
- Eliminated redundant arena file writes during `rul run` AOT execution.
- Added explicit disclosure when `rul-mcp` tools (`rul_trace`, `rul_test_differential`, `rul_autotune_schedule`) evaluate synthetic fallback inputs.
- Synchronized crate version and `rul-mcp` protocol server metadata.

---

## [0.3.0] — 2026-11-01

### Added
- **Expanded 20-Opcode ISA**: Added `Div` (`0x13`), `Min` (`0x16`), `Max` (`0x17`), `CmpLt` (`0x18`), and `CmpEq` (`0x19`) across all execution tiers and `x86_64` AVX2 `ELF64` AOT codegen.
- **White-Box Execution Tracer (`rul trace` & `rul_trace` MCP tool)**:
  - Implemented `trace_program` and `format_trace_table` in `src/engine.rs` to inspect per-node `min`, `max`, `sum`, `nan_or_inf_count`, and sample head values across all SSA nodes.

### Fixed
- Replaced `wrapping_add(1)` on `AffineConsume` version transitions with `checked_add(1)` returning `VerifyErrorKind::TokenVersionOverflow` on `u16::MAX`.
- Hardened `Opcode::Gather` to reject fractional floating-point indices (`raw_idx.fract() != 0.0`).
- Updated `VerifiedSummary.zero_runtime_bounds_checks` to reflect `!has_gather`.

---

## [0.2.0] — 2026-10-29

### Added
- Standalone `rul` CLI toolchain (`new`, `check`, `build`, `run`, `autotune`, `inspect`, `decompile`) and `install.sh` installer.
- Lossless `.rls` <-> `.rl` symbolic assembler and decompiler (`src/transpiler.rs`).
- In-memory JSON-RPC 2.0 MCP server (`src/mcp.rs`, `src/bin/rul_mcp.rs`).
- Content-Addressed Store (`RulCasStore`) with `I64`/`U8` verified intrinsics (`src/cas.rs`) and zero-panic `#[repr(C)]` embedding API (`src/ffi.rs`).

---

## [0.1.0] — 2026-10-28

### Added
- Initial `.rl` binary specification (`RulHeader`, `ArenaDescriptor`, `ProofEntry`, and 16-byte `NodeFrame`).
- Single-pass `O(N)` linear static verifier (`src/verifier.rs`).
- 3-tier execution engine (`Tier 0` interpreter, `Tier 1` vector stencils, `Tier 2` L1-cache tile fuser with `ArenaZoneMap` skipping) and direct Linux `x86_64` `ELF64` AOT compiler (`src/aot.rs`).
- 10-pipeline differential verification suite (`src/suite10.rs`) and Zenodo archival registration (`DOI: 10.5281/zenodo.23722094`).
