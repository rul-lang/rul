# Changelog

All notable changes to the **Rul** (`.rl`) binary representation, verifier, 3-Tier execution engine, and `ELF64` AOT compiler are documented in this file.

## [Unreleased] — Phase P0 (Safety & Correctness) & Phase P1 (AOT `YMM` Register Allocation)

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
    - Standalone `ELF64` binary size for `BM-HORNER-50` shrank from **5,312 B (`main`)** to **3,872 B** (compile latency `~20.4 µs`).
    - Execution time on 10,000,000 elements dropped from **396 ms (`main`)** to **106.58 ms (`3.72x`..`3.96x` speedup)**, matching `[A] Hand-Written C/Rust 8-Acc Horner-50` (**107.96 ms**, **`1.01x`**) and `gcc -O3 -mavx2 -mno-fma` (**94 ms**), while beating `[B] Rul Tier 2 Fused` (**644.96 ms**) by **`6.05x`**.
