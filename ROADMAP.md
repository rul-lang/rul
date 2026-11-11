# Rul (.rl) Technical Roadmap

This document defines the engineering roadmap for **Rul (`.rl`)**, tracking what has already shipped (`v0.1.0` through `v0.4.0`) and the six program-driven milestones (`v0.5` through `v1.0`) taking Rul from a verified columnar compute engine to a Turing-complete systems programming language designed to replace Rust and Go for AI-generated services and CLIs.

---

## 1. Foundation Shipped (`v0.1` to `v0.4` — Columnar & Vector Compute Core)

Traditional languages like C, Go, and Rust were designed around human text files (`fn main() { ... }`), requiring multi-second lexing, parsing, constraint-solving borrow checks, and 100+ heuristic LLVM passes.

Rul replaces text source files with a flat array of 16-byte binary instruction frames (`.rl`), where each instruction carries its **operation**, **hardware schedule hint**, and **safety witness** up front:
- **`v0.1.0` (Initial Binary Core)**: 64-byte `RulHeader`, 16-byte `NodeFrame`, single-pass `O(N)` verifier, 3-tier execution engine, and direct Linux `x86_64` `ELF64` native compiler.
- **`v0.2.0` (Standalone Toolchain & MCP)**: Standalone `rul` CLI (`new`, `check`, `build`, `run`, `autotune`, `inspect`, `decompile`), lossless `.rls` symbolic assembler, `rul-mcp` server, content-addressed intrinsics (`rul-cas`), and C-compatible embedding pods (`rul-ffi`).
- **`v0.3.0` (White-Box Tracer & Verifier Hardening)**: Expanded ISA, node-by-node white-box execution tracer (`rul trace`), and IEEE-754 edge-case verification.
- **`v0.4.0` + `P0/P1/P1b` (30-Opcode ISA, `YMM` Register Allocator & AVX2 Vector Math)**: Linear-scan `ymm2..ymm13` register allocator + `ymm14/ymm15` reduction accumulators, explicit single-rounded `Opcode::Fma (0x27)`, 256-bit AVX2 vectorized `Ln`/`Exp`, multi-reduction scalar DAGs, and hardened slice/gather bounds (`~5.8 ns/node` verification, `~6.4–10.1 us` native compilation, `51.53 GB/s` effective columnar throughput, `3.33x` faster than hand-written C/Rust 8-accumulator Horner-50).

---

## 2. Architectural Principles for `v0.5`–`v1.0` (Turing-Complete Systems Language)

To break the traditional compiler triangle (**Memory Safe · C-Level Execution Speed · Microsecond Compilation**) while achieving full Turing-completeness without bloating the core compiler:

1. **Termination as a Capability Policy (`CAP_DIVERGE = 0x08`), Not a Hardwired Cage**:
   - **Without `CAP_DIVERGE` (`0x00` Sandbox Mode)**: Every loop must carry a verified `LoopBound` witness and recursive function calls are rejected at load time. AI plugins, UDFs, and sandboxed filters are mathematically guaranteed to halt.
   - **With `CAP_DIVERGE` (`0x08` Systems/Daemon Mode)**: Unbounded structured loops and general recursion are permitted (with deterministic fail-closed call-depth guards), enabling Turing-complete interpreters, parsers, and long-lived network servers.
2. **WASM-Style Structured Control Flow + Typed Locals + Witness Signatures in `O(N)`**:
   - Instead of inventing ad-hoc graph-sealing rules for branches, loops, and functions separately, Rul uses a single push/pop control stack (`Block`, `Loop`, `If`, `Else`, `End`, `Br`, `BrIf`, `Func`, `Return`) paired with definitely-initialized typed scalar locals (`LocalGet`/`LocalSet`) and sidecar witness signatures (`BlockSig` / `FuncSig`) carrying return types and `AffineToken` ownership states. Validation remains a single forward `O(N)` pass with unlimited nesting depth.
3. **Runtime Bounds Checks by Default, Static Proofs to Elide Checks**:
   - Dynamic indexed memory accesses (`RegionLoad` / `RegionStore`) emit a fast unsigned bounds check (`cmp + jae`, ~1–3% overhead, identical to Rust) by default when `proof_idx == 0`, and elide the runtime check whenever a `ProofEntry` witness certifies the index bound.
4. **Dual-Track Execution Equivalence (`Tier 0 == AOT` for General Code, `Tier 1/2` for Columnar SIMD)**:
   - General-purpose control flow, growable heap regions, syscalls, and concurrency target **`Tier 0` (Reference Oracle) == `ELF64` AOT**.
   - `Tier 1` (vector stencils) and `Tier 2` (L1-cache tile fuser + `ArenaZoneMap`) remain dedicated high-throughput accelerators for columnar subgraphs and fail closed with an explicit error on non-columnar constructs.
5. **Library Over ISA Bloat**:
   - Data structures and text routines (`Vec`, `String`, `utf8_validate`, `HashMap`, JSON parsing) are implemented as standard library modules in Rul rather than dedicated compiler opcodes.

---

## 3. Program-Driven Roadmap to `v1.0` (7–9 Months)

Every milestone is defined by **concrete runnable programs passing `Tier 0 == ELF64 AOT`**, not merely a checklist of opcodes:

### Milestone 1 (`v0.5` — Weeks 1–7): Turing-Complete Core (`TC Core`)
- **Scope**:
  - **`v0-5-tc-core-1` (Structured Control Flow & `CAP_DIVERGE`)**: `ConstI64`, definitely-initialized scalar locals (`LocalGet`, `LocalSet`), structured control stack (`Block`, `Loop`, `If`, `Else`, `End`, `Br`, `BrIf`), `BlockSig`/`LoopBound` proofs, and `CAP_DIVERGE` (`0x08`) termination policy across `.rl`, `.rls`, `O(N)` Verifier, `Tier 0`, and `x86_64` `ELF64` AOT.
  - **`v0-5-tc-core-2` (`I64`/`U8` Arithmetic & Growable Regions)**: Scalar `I64`/`U8` operations (`Cast`, wrapping `Add`/`Sub`/`Mul`, checked `Div`/`Rem`, bitwise `BitAnd`/`BitOr`/`BitXor`/`Shl`/`ShrU`), linear growable regions (`RegionNew`, `RegionGrow`, `RegionLen`, `RegionLoad`, `RegionStore` backed by `SYS_mmap`/`SYS_mremap`), and runtime bounds checks (elided when `proof_idx > 0`).
  - **`v0-5-tc-core-3` (Function Table & Recursion)**: `Func`..`End` scopes (`FuncSig`), `Call`/`CallArg`/`Return`, `CAP_DIVERGE`-gated recursion, and deterministic call-stack depth guards (`MAX_CALL_DEPTH` fail-closed in `Tier 0` and `ELF64` AOT).
- **Exit Programs (`Tier 0 == ELF64 AOT`)**:
  - **A complete Brainf\*ck interpreter written in Rul** (executable proof of Turing-completeness), recursive `fib`, in-place `quicksort`, prime `sieve`, and `sha256`.

### Milestone 2 (`v0.6` — Weeks 8–13): Ownership Across Joins & Rul Standard Library
- **Scope**: Full `AffineToken` consumption/transfer and scoped `ReadLease` tracking across `BlockSig`/`FuncSig` control-flow joins, region-backed struct/enum layouts, and core Rul standard library modules (`Vec`, `String` + `utf8_validate`, `HashMap`).
- **Exit Programs (`Tier 0 == ELF64 AOT`)**: Zero-GC JSON parser, in-memory key-value store, and buffer-level `wc`/`grep`.

### Milestone 3 (`v0.7` — Weeks 14–17): Capability-Gated Linux Systems I/O
- **Scope**: Direct Linux syscalls (`openat`, `read`, `write`, `close`, `stat`, `clock_gettime`, `socket`, `epoll`) governed at load time by fine-grained `CapabilityMask` bits (`CAP_FS_READ`, `CAP_FS_WRITE`, `CAP_NET`, `CAP_CLOCK`).
- **Exit Programs (`Tier 0 == ELF64 AOT`)**: Standalone native `cat`, `grep`, and `wc` CLI binaries, plus a CSV file aggregator invoking Rul's columnar SIMD engine.

### Milestone 4 (`v0.8` — Weeks 18–25): Thread-Per-Core Concurrency & Network Servers
- **Scope**: Share-nothing thread-per-core worker tasks (preserving zero hidden runtime threads and zero GC), typed channels that linearly transfer `AffineToken` region ownership across cores (preventing data races at load time), and `epoll`/`io_uring` event loops under `CAP_DIVERGE`.
- **Exit Programs (`Tier 0 == ELF64 AOT`)**: Standalone **HTTP/1.1 Key-Value Server** benchmarked head-to-head against Go (`net/http`) and Rust (`axum`) on RPS, p99 latency, binary size, and microsecond compile time.

### Milestone 5 (`v0.9` — Weeks 26–29): C ABI Ecosystem FFI & AI Synthesis (`pass@1`)
- **Scope**: Explicit `CAP_FFI`-gated C ABI foreign function calls (unlocking `libc`, `SQLite`, and `OpenSSL`/crypto libraries with clear trust boundaries), structured JSON verifier diagnostics with repair hints, and the **Rul-50 LLM Synthesis Benchmark** measuring first-shot AI code generation (`pass@1`).
- **Exit Programs**: Rul service querying `SQLite` over `CAP_FFI` and measured LLM `pass@1` benchmark report.

### Milestone 6 (`v1.0` — Weeks 30–35): Linux `AArch64` Backend, Formal Proofs & Binary Freeze
- **Scope**: Native zero-LLVM `AArch64` (`ARM64`) Linux `ELF64` backend alongside `x86_64`, continuous differential fuzzing (`Tier 0` vs `AOT`), Kani bounded model-checking proofs for the `O(N)` verifier, and `.rl` `v1.0` format freeze.
- **Exit Deliverable**: 50-program cross-architecture (`x86_64` + `AArch64`) conformance suite.

---

## 4. Core Engineering Invariants (Non-Goals)

1. **No Backtracking Constraint Solvers in the Verifier**: Every safety property (types, definite local initialization, `AffineToken` ownership, `ProofEntry` bounds, and `CapabilityMask` permissions) must be verifiable in a single forward `O(N)` pass.
2. **No Garbage Collector or Hidden M:N Runtime Scheduler**: Memory lifetime is governed explicitly by versioned arenas/regions, and concurrency uses explicit share-nothing thread-per-core execution.
3. **No Silent Fallbacks**: Every supported execution tier must match the `Tier 0` reference oracle (`Tier 0 == AOT` for general programs; `Tier 0 == Tier 1 == Tier 2 == AOT` for columnar pipelines) or fail closed with an explicit error.
