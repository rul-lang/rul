# Contributing to Rul (`.rl`)

Thank you for your interest in contributing to **Rul (`.rl` — Rule-Carrying Unified Linear Graph)**!

Rul is an AI-native 128-bit binary programming language, single-pass $O(N)$ static verifier, 3-tier vector execution engine, and zero-runtime `x86_64` AVX2 `ELF64` AOT compiler. Because Rul provides mathematical safety and deterministic execution guarantees, all contributions must uphold strict verification and engineering standards.

---

## 1. Where to Start

- **Questions & Usage Help**: Open a thread in [GitHub Discussions (Q&A)](https://github.com/rul-lang/rul/discussions/categories/q-a).
- **New Opcodes, Schedule Hints, or Architectural RFCs**: Propose and discuss design changes first in [GitHub Discussions (Ideas)](https://github.com/rul-lang/rul/discussions/categories/ideas) before submitting a large Pull Request.
- **Bug Reports & Compiler Crashes**: Use the structured [Bug Report](https://github.com/rul-lang/rul/issues/new?template=bug_report.yml) or [Crash / Verifier Escape](https://github.com/rul-lang/rul/issues/new?template=crash_ice.yml) issue forms and include a minimal `.rls` reproduction.
- **Security Vulnerabilities**: Do **not** open a public issue for verifier bypasses or out-of-bounds memory escapes. Please follow [`SECURITY.md`](./SECURITY.md) to report privately via [GitHub Security Advisories](https://github.com/rul-lang/rul/security/advisories/new).

---

## 2. Core Engineering Invariants (Non-Negotiable)

Every Pull Request to `rul-lang/rul` must preserve the following invariants across all library and binary targets:

1. **100% Safe, Zero-Panic Rust**:
   - The entire codebase enforces `#![forbid(unsafe_code)]`, `#![deny(clippy::unwrap_used)]`, `#![deny(clippy::expect_used)]`, `#![deny(clippy::panic)]`, and `#![deny(clippy::indexing_slicing)]`.
   - Never introduce `unsafe` blocks, `.unwrap()`, `.expect()`, `panic!()`, or unchecked slice indexing (`slice[i]`). Always use `.get(i)`, `.get_mut(i)`, checked/saturating arithmetic, and typed `Result` diagnostics.
2. **Single-Pass $O(N)$ Linear Verification (`src/verifier.rs`)**:
   - Verification must remain strictly single-pass $O(N)$ over `0..node_count` with $O(1)$ per-node checks (topological SSA ordering `in_a < idx`, `AffineConsume`/`ReadLease` ownership, `ProofEntry` origin-arena bounds, and `CapabilityMask` sandboxing).
   - Never introduce iterative constraint solvers, unbounded recursion, or multi-pass fixpoint loops into the verifier.
3. **4-Way Differential Equivalence (`Tier 0 == Tier 1 == Tier 2 == ELF64 AOT`)**:
   - Any change to an opcode or execution path must produce matching results across:
     - **Tier 0 (`execute_tier0_interpreter`)**: Reference Golden Oracle.
     - **Tier 1 (`execute_tier1_stencil`)**: 8-lane SIMD vector stencils.
     - **Tier 2 (`execute_tier2_fused`)**: L1-cache tile fuser (`1,024` elements/tile) & `MultiCore` scheduler.
     - **Standalone `ELF64` AOT (`compile_to_elf64_bytes`)**: Direct `x86_64` AVX2 + SSE2 native machine code emitter.
   - If a graph shape is not supported by AOT compilation, `compile_to_elf64_bytes` **must fail closed** with `Err(AotError::UnsupportedGraph(...))`—never silently emit a binary that computes an incorrect result.
4. **Zero External Runtime Dependencies**:
   - Do not add heavy third-party crates to `[dependencies]` in `Cargo.toml` without prior architectural approval.
5. **English-Only Repository Artifacts**:
   - All source code, comments, docstrings, CLI diagnostics, commit messages, and documentation must be written strictly in English.

---

## 3. Repository Architecture Map

| Module | Responsibility |
| :--- | :--- |
| [`src/bytecode.rs`](./src/bytecode.rs) | 64-byte `RulHeader`, 16-byte `ArenaDescriptor`, 16-byte `ProofEntry`, and 16-byte (128-bit) `NodeFrame` across the 29-opcode ISA. |
| [`src/verifier.rs`](./src/verifier.rs) | Single-pass $O(N)$ static verifier enforcing SSA dominance, `AffineConsume`/`ReadLease`, `ProofEntry` bounds, and `CapabilityMask`. |
| [`src/engine.rs`](./src/engine.rs) | Tier 0 Oracle, Tier 1 Vector Stencil, Tier 2 L1-Tile Fuser + MultiCore + `ArenaZoneMap`, `autotune_schedule`, and `trace_program`. |
| [`src/aot.rs`](./src/aot.rs) | Zero-dependency Copy-and-Patch `x86_64` AVX2 (`YMM`) + SSE2 (`XMM`) standalone Linux `ELF64` AOT compiler. |
| [`src/transpiler.rs`](./src/transpiler.rs) | Lossless `.rls` $\leftrightarrow$ `.rl` symbolic assembler and decompiler (`assemble_rls`, `program_to_rls`). |
| [`src/mcp.rs`](./src/mcp.rs) & [`src/bin/rul_mcp.rs`](./src/bin/rul_mcp.rs) | In-memory `stdio` JSON-RPC 2.0 Model Context Protocol (`MCP`) server for AI Agents. |
| [`src/cas.rs`](./src/cas.rs) & [`src/ffi.rs`](./src/ffi.rs) | Content-Addressed Store (`RulCasStore`), `I64`/`U8` verified intrinsics, and zero-panic `#[repr(C)]` embedding API. |
| [`src/bin/rul.rs`](./src/bin/rul.rs) & [`src/bin/rul_bench.rs`](./src/bin/rul_bench.rs) | Standalone `rul` CLI toolchain (`new`, `check`, `build`, `run`, `autotune`, `trace`, `inspect`, `decompile`) and `rul-bench` suite. |

---

## 4. Development & Quality Gate Workflow

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/rul.git
   cd rul
   ```
2. **Create a focused feature or bugfix branch**:
   ```bash
   git checkout -b feat/your-feature-name
   ```
3. **Run all mandatory quality gates before opening a Pull Request**:
   ```bash
   # 1. Verify formatting
   cargo fmt --all -- --check

   # 2. Verify zero compiler or Clippy warnings across all targets
   cargo clippy --all-targets -- -D warnings

   # 3. Run all unit, verifier, differential, and AOT tests
   cargo test --all-targets

   # 4. Run the release 10-pipeline differential & AOT benchmark suite
   cargo run --release --bin rul-bench
   ```
4. **Test Hygiene**:
   - Tests must return `Result<(), String>` (using `?`) or assert explicitly on expected results. Never write silent early returns (`let Ok(..) = .. else { return; }`) that allow a broken test to pass silently.

---

## 5. Dual Licensing (`MIT OR Apache-2.0`)

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual-licensed under both the [`MIT License`](./LICENSE-MIT) and the [`Apache License, Version 2.0`](./LICENSE-APACHE), without any additional terms or conditions.
