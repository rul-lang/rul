# Rul (.rl) Technical Roadmap

This document outlines the engineering roadmap for **Rul (`.rl`)**, tracking what has already shipped (`v0.1.0` through `v0.4.0`) and the four milestones taking Rul from a verified compute engine to a general-purpose systems programming language (`v1.0`).

---

## Why We Started with Data Pipelines (`v0.1` to `v0.4`)

Traditional languages like C, Go, and Rust were designed for humans typing text at a keyboard. Because humans cannot manually write memory-lifetime proofs and CPU scheduling hints on every instruction, compilers like `rustc` and `LLVM` spend seconds or minutes parsing text, solving borrow-checker constraints, and searching for optimizations.

Rul is built for AI agents. Instead of text files, an AI emits a flat list of 16-byte binary instructions (`.rl`). Each 16-byte instruction carries three things up front:
1. **The operation** (what to compute).
2. **The hardware schedule hint** (single-core SIMD or multi-core).
3. **The safety witness** (memory version tag and bounds index).

To prove that this 16-byte binary format works on real hardware before adding general-purpose language features, we restricted Phases `v0.1` through `v0.4` to array and vector processing pipelines:
- **`v0.1.0` (Initial Binary Core)**: 64-byte header, 16-byte `NodeFrame` format, single-pass `O(N)` verifier, 3-tier execution engine, and direct Linux `x86_64` `ELF64` compiler.
- **`v0.2.0` (Standalone Toolchain & MCP)**: Standalone `rul` CLI (`new`, `check`, `build`, `run`, `autotune`, `inspect`, `decompile`), lossless `.rls` symbolic assembler, `rul-mcp` server, content-addressed intrinsics (`rul-cas`), and C-compatible embedding pods (`rul-ffi`).
- **`v0.3.0` (White-Box Tracer & Verifier Hardening)**: Expanded to 20 opcodes, added `rul trace` for node-by-node inspection (`min`, `max`, `sum`, `NaN/Inf` counts), and added IEEE-754 edge-case verification.
- **`v0.4.0` (29-Opcode ISA, Multi-Reduce & Linear-Scan `YMM` Allocator)**: Added transcendental/math opcodes (`Abs`, `Sqrt`, `Floor`, `Ceil`, `Ln`, `Exp`, `MaskAnd`, `MaskOr`, `ReduceMin`), multi-reduction and post-reduction scalar DAG support, nested slice bounds verification, and a linear-scan `YMM` register allocator in the native `ELF64` compiler (`~5.8 ns/node` verification, `~6.4 us` native compilation, `51.53 GB/s` effective throughput).

---

## Six-Month Roadmap to `v1.0` (General-Purpose Systems Language)

Over the next four milestones, we are expanding Rul from a columnar compute engine into a general-purpose, proof-carrying systems language capable of replacing Rust and Go for AI-generated software while keeping single-pass `O(N)` verification.

### Milestone 1 (`v0.5` — Months 1-2): Control Flow, Functions, and Data Types

Expand the 16-byte `NodeFrame` instruction set beyond numeric arrays to support general-purpose data structures, structured control flow, and modular functions:
- **General Scalar and Composite Types**: First-class `i64`, `i32`, `u8`, `f32`, fixed-layout structs, algebraic enums (tagged unions), validated UTF-8 strings, and arena-backed hash maps.
- **Structured Control Flow (`SelectBranch` and `RegionLoop`)**:
  - `SelectBranch` evaluates conditional branches while requiring both branches to end at the same memory version state, allowing the verifier to check branch merges in `O(1)`.
  - `RegionLoop` supports general iteration while requiring an explicit termination witness in the `ProofEntry` table (either a strictly decreasing integer variant or an explicit step/fuel bound). This provides practical Turing-completeness for systems software while preserving guaranteed termination and single-pass `O(N)` verification.
- **Modular Function Table (`Call`)**: Function headers declare explicit input and output memory version signatures so each function is verified independently in a single linear scan without whole-program inlining.

### Milestone 2 (`v0.6` to `v0.7` — Months 2-3): Rust-Grade Ownership and Lifetimes in `O(N)` Time

Generalize Rul's 32-bit memory version token (`TokenOp`) and 16-byte `ProofEntry` sidecar table from flat arrays to general systems memory:
- **Region-Based Heap Allocations**: Deterministic allocation and deallocation tied to linear ownership tokens, with zero garbage collection and zero per-object `malloc`/`free` fragmentation.
- **Struct Field Borrows and Slice Lifetimes**: Sidecar `ProofEntry` witnesses for struct field offsets, disjoint field mutations, and zero-copy sub-slice lifetimes checked in `O(1)` per instruction.
- **Result**: Rust-level protection against use-after-free, double-free, and data aliasing without the multi-second constraint solver overhead of `rustc`.

### Milestone 3 (`v0.8` to `v0.9` — Months 4-5): Go-Style Concurrency and Capability-Gated Systems I/O

Introduce lightweight concurrency and operating system I/O governed by compile-time ownership and header capability bits:
- **Deterministic Structured Concurrency**: Lightweight tasks and typed channels. Sending a buffer over a channel consumes the sender's memory version token (`AffineConsume`) and transfers ownership to the receiver, preventing data races at load time without a garbage collector.
- **Fine-Grained `CapabilityMask` for OS Syscalls**: Extend the 64-bit header `CapabilityMask` to gate filesystem reads/writes, network sockets, timers, and subprocess execution. An AI agent running a `.rl` binary with `CAP_NONE (0x0)` is mathematically blocked from making network or disk calls before execution begins.

### Milestone 4 (`v1.0` — Month 6): Multi-Architecture Native Backends and Formal Soundness Proofs

Harden the compiler and specification for production `v1.0` stability across major hardware targets:
- **Multi-Target Zero-LLVM Native Compilation**: Extend `src/aot.rs` beyond Linux `x86_64` to emit standalone executables for `x86_64` and `AArch64` (`ARM64`) across Linux, macOS, and Windows, plus deterministic WebAssembly (`WASI`) modules.
- **Formal Soundness Verification and Continuous Fuzzing**: Bounded model-checking proofs (Kani) and continuous differential fuzzing (`cargo-fuzz` / property testing) across the binary decoder, `O(N)` verifier, interpreter, and native code generator.
- **`v1.0` Binary Format Stability Guarantee**: Freeze the `.rl` `v1.0` binary specification and publish a 50-program conformance test suite.

---

## Core Engineering Rules (Non-Goals)

To keep verification at nanoseconds per instruction and compilation at microseconds, Rul intentionally avoids the following:
1. **No Backtracking Constraint Solvers in the Verifier**: Every safety property (types, memory ownership, bounds, loop termination, and I/O permissions) must be verifiable in a single forward pass `O(N)` using witnesses attached to the `.rl` binary.
2. **No Garbage Collector or Hidden Runtime Threads**: Memory lifetime is tracked explicitly through versioned arenas and regions.
3. **No Silent Fallbacks in Native Code Generation**: Every execution tier (`Tier 0`, `Tier 1`, `Tier 2`, and `AOT`) must produce matching results or fail closed with a clear error.
