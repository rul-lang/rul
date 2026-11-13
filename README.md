<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/logo/rul-banner-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./assets/logo/rul-banner-light.svg">
    <img alt="Rul (.rl) — AI-Native 128-Bit Binary Programming Language" src="./assets/logo/rul-banner-dark.svg" width="100%">
  </picture>
</p>

# Rul (`.rl`) — A 128-Bit Binary Programming Language for AI Agents

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23722094.svg)](https://doi.org/10.5281/zenodo.23722094)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](./LICENSE-APACHE)
[![Spec: Rul v0.4](https://img.shields.io/badge/Spec-.rl%20v0.4%20(128--bit%20Binary%20IR)-success.svg)](./docs/rul_spec_v0.md)
[![Roadmap: v1.0](https://img.shields.io/badge/Roadmap-v0.4%20to%20v1.0-informational.svg)](./ROADMAP.md)
[![Whitepaper: v0.1](https://img.shields.io/badge/Whitepaper-Zenodo%20Open%20Archive-orange.svg)](./docs/WHITEPAPER_RUL_V0_1.md)

## What is Rul in Plain English?

Languages like C, Go, and Rust were built for humans typing text at a keyboard. When an AI agent writes Rust or C today, it outputs thousands of characters of text, and the compiler (`rustc` or `LLVM`) spends seconds parsing that text, checking memory rules, and searching for loop optimizations.

**Rul (`.rl`)** removes the human text layer. Instead of writing text source files, an AI agent emits a compact list of fixed **16-byte (128-bit) binary instructions**. Every 16-byte instruction carries three pieces of information up front:

1. **The operation** (such as add, compare, filter, or reduce).
2. **How to run it on the CPU** (a 1-byte schedule hint selecting single-core SIMD or multi-core execution).
3. **A memory safety tag** (a version number and bounds index proving the memory access is valid).

Because the AI provides the CPU schedule and safety tags inside each instruction, the Rul compiler does not need to guess or run slow optimization passes:
- **Fast safety checks**: Rul verifies memory ownership, array bounds, the termination policy, and I/O permissions in a single forward pass (**~5.8 nanoseconds per instruction**, or under 60 microseconds for 10,000 instructions).
- **Microsecond native compilation**: Rul compiles verified `.rl` programs directly into standalone Linux `x86_64` executables (**628 to 1,037 bytes**) in **~6.4 microseconds**, without LLVM or an external linker.
- **Easy for humans to inspect**: Humans never need to write `.rl` by hand, but can inspect any program at any time using `rul decompile` (readable text view) or `rul trace` (which prints the exact `min`, `max`, `sum`, and `NaN` count at every step).

---

## 30-Second Quickstart

Install the standalone `rul` binary and create, verify, compile, run, and trace a `.rl` program in 30 seconds (no `Cargo.toml` or `rustc` required to run `.rl` projects):

```bash
# 1. Install the standalone rul toolchain
curl -fsSL https://raw.githubusercontent.com/rul-lang/rul/main/install.sh | sh

# 2. Create a sample Rul project (rul.toml, src/main.rls, data/arena_0.bin)
rul new my-ai-pipeline && cd my-ai-pipeline

# 3. Verify in microseconds, compile a standalone ~880-byte Linux ELF64 binary, and run
rul check
rul run

# 4. Inspect the exact min/max/sum/NaN output at every step (#0..#6)
rul trace
```

**Readable Assembly View (`src/main.rls` — 7 instructions = `208 bytes` in binary `.rl`)**:
```text
cap 0x0
arena 0 len=4096 lane=f64x8 mut=false align=64 ver=1
node $n0 = ArenaInput f64x8 fg=1 pol=simd proof=0 tok=read(0,1) a=0 b=0
node $n1 = ConstF64   f64x1 fg=1 pol=simd proof=0 tok=pure const=100.0
node $n2 = CmpGt      maskx8 fg=1 pol=simd proof=0 tok=pure a=$n0 b=$n1
node $n3 = Mul        f64x8 fg=1 pol=simd proof=0 tok=pure a=$n0 b=$n0
node $n4 = SelectZero f64x8 fg=1 pol=simd proof=0 tok=pure a=$n2 b=$n3
node $n5 = ReduceSum  f64x1 fg=1 pol=simd proof=0 tok=pure a=$n4 b=0
node $n6 = AssertContract f64x1 fg=1 pol=simd proof=0 tok=pure a=$n5 b=0
output $n6
```

**Output of `rul trace` (Step-by-Step Inspector)**:
```text
=== RUL (.rl) WHITE-BOX EXECUTION TRACE (7 SSA Nodes) ===
NODE   OPCODE          LIVE  KIND             LEN          MIN          MAX      SUM/TRUES NAN/INF  SAMPLE[0..4]
#0     ArenaInput      yes   ColumnF64       4096       0.5000     249.5000    504608.0000       0  [0.500, 1.500, 2.500, 3.500]
#1     ConstF64        yes   Scalar             1     100.0000     100.0000       100.0000       0  [100.000]
#2     CmpGt           yes   ColumnMask      4096       0.0000       1.0000      2400.0000       0  [0.000, 0.000, 0.000, 0.000]
#3     Mul             yes   ColumnF64       4096       0.2500   62250.2500  83627904.0000       0  [0.250, 2.250, 6.250, 12.250]
#4     SelectZero      yes   ColumnF64       4096       0.0000   62250.2500  77999800.0000       0  [0.000, 0.000, 0.000, 0.000]
#5     ReduceSum       yes   Scalar             1 77999800.0000 77999800.0000  77999800.0000       0  [77999800.000]
#6     AssertContract  yes   Scalar             1 77999800.0000 77999800.0000  77999800.0000       0  [77999800.000]
```

---

## How Rul Achieves Safety, Runtime Speed, and Microsecond Compilation Together

Traditional languages force a trade-off between memory safety, runtime speed, and compile time because the compiler has to infer what the human programmer meant. Rul avoids that trade-off through four simple mechanisms:

1. **No Text Parsing (`0%` Syntax Errors)**:
   - AI agents emit 16-byte binary instructions directly (or assemble flat `.rls` listings in ~8.8 microseconds).
   - A 7-step pipeline over 10 million numbers is **`192 bytes`** in `.rl` and compiles into a standalone **`817-byte`** Linux `ELF64` binary.
2. **Versioned Memory Buffers Instead of a Slow Borrow Checker**:
   - Every memory buffer (`Arena`) starts at version `1`.
   - Multiple instructions can read version `1` at the same time (`ReadLease`).
   - If an instruction modifies a mutable buffer in place (`AffineConsume`), it consumes version `1` and advances the buffer to version `2`. Any later instruction that still tries to read or write version `1` is rejected immediately.
   - This gives AI-generated code Rust-grade protection against use-after-free and aliasing bugs in a single linear scan (`5.78 ns/instruction`).
3. **Compile-Time Bounds Proofs and Hardware Capability Bits**:
   - Operations on matching array lengths need zero runtime bounds checks. Sub-slices (`ArenaSlice`) and indirect lookups (`Gather`) reference a 16-byte `ProofEntry` table checked at load time.
   - The 64-byte file header includes a `CapabilityMask` bitfield. When set to `0x0` (`CAP_NONE`), instructions that perform external I/O are rejected before the program can run.
4. **Cache-Friendly Columnar Memory and Single-Byte Schedule Tuning**:
   - Data is stored in 64-byte-aligned columnar arrays (`Structure-of-Arrays`) so CPU vector registers only load the fields they actually use.
   - Changing a single byte (`ScheduleHint`) switches execution from single-core SIMD (`pol=simd`) to all CPU cores (`pol=multicore`) without rewriting the algorithm.

---

## Benchmark (`10,000,000` `f64` Elements = `80 MB` Array)

Workload: Filter elements `> 1000.0`, square matching values, and sum to a scalar.

| Implementation | Execution Mode | Median Latency | Effective Logical Bandwidth | Speedup vs. C 8-Acc (1-Core) | Speedup vs. Legacy AoS |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `[A]` Imperative AoS + `Vec::collect()` | 1-Core AoS + Heap | `70.66 ms` | `1.13 GB/s` | `0.10x` | `1.00x` |
| `[B]` Idiomatic Rust SoA (`filter().map().sum()`) | 1-Core SoA Iterator | `19.12 ms` | `4.18 GB/s` | `0.38x` | `3.70x` |
| `[C]` Hand-Written C/Rust 8-Accumulator SoA | **1-Core 8-Lane Unrolled** | `7.32 ms` | `10.92 GB/s` | **`1.00x` (Baseline)** | `9.65x` |
| **`[D]` Rul (`.rl`) Tier 2 Fused SIMD** | **1-Core L1-Tile SIMD (`SimdStream`)** | **`5.91 ms`** | **`13.53 GB/s`** | **`1.24x` (1-Core vs. 1-Core)** | **`11.95x`** |
| **`[E]` Rul (`.rl`) Tier 2 MultiCore** | **All Cores (`MultiCore` 1-Byte Hint)** | **`3.19 ms`** | **`25.04 GB/s`** | **`2.29x` (Schedule Mutation)** | **`22.12x`** |
| **`[F]` Rul (`.rl`) Tier 2 MultiCore + ZoneMap** | **All Cores + `O(1)` Tile Skipping** | **`1.55 ms`** | **`51.53 GB/s`** | **`4.72x` (With Block Min/Max)** | **`45.51x`** |

Notes on the numbers above:
- **`[C]` vs. `[D]` (`1.24x`)** is the strict single-core comparison on the exact same `80 MB` array.
- **`[E]` (`2.29x`)** changes only the 1-byte `ScheduleHint` from `pol=simd` (`0x11`) to `pol=multicore` (`0x12`), parallelizing 1,024-element L1 tiles across cores.
- **`[F]` (`4.72x` / `51.53 GB/s` logical bandwidth)** adds block min/max metadata (`ArenaZoneMap`) on partitioned data so cold 1,024-element tiles that cannot match the filter are skipped in `O(1)` without reading them from RAM.

On deep arithmetic pipelines (`BM-HORNER-50`, a 50-step Horner polynomial with 104 SSA nodes over 10,000,000 `f64` elements), the linear-scan `YMM` register allocator in `src/aot.rs` compiles a `3,872-byte` `ELF64` binary in `~20.4 us` and executes in **`106.58 ms`**, matching hand-written 8-accumulator C/Rust (`107.96 ms`).

---

## Why `.rl` Does Not Become an Unmaintainable Black Box

A common question is: *"If humans do not write `.rl` code, how do teams debug or maintain a system over time?"*

1. **Humans Define the Boundaries; AI Agents Regenerate the Binary**:
   - Human engineers define the input/output schemas (`ArenaDescriptor`), hardware permissions (`CapabilityMask`), and output assertions (`AssertContract`).
   - Because verifying and compiling a `.rl` module takes microseconds, you never patch old AI-generated code. When requirements change, the AI regenerates the `.rl` module and verifies it from scratch.
2. **Scheduling is Separate from Logic**:
   - In C or Rust, adding SIMD intrinsics, cache tiling, and multi-threading tangles hardware tuning with business logic.
   - In Rul, the operation (`opcode`, `in_a`, `in_b`), the CPU schedule (`schedule_hint`), and the safety tags (`proof_idx`, `token_op`) sit in separate fields of each 16-byte instruction. Running `rul autotune` updates only the schedule byte.
3. **Built-In Step-by-Step Inspection (`rul trace`, `rul decompile`, `rul inspect`)**:
   - At any time, engineers can run `rul decompile` to view the readable `.rls` listing or `rul trace` to inspect `min`, `max`, `sum`, `NaN/Inf` counts, and sample values at every instruction node `#0..#N-1`.

---

## Where Rul is Today and Roadmap to `v1.0`

We intentionally restricted `v0.1` through `v0.4.0` to **array and vector dataflow pipelines** (`29 opcodes` at the `v0.4.0` release) to prove the 16-byte binary instruction format, single-pass `O(N)` verifier, and zero-LLVM native `ELF64` compiler on real hardware first.

The first `v0.5` slice is implemented on top of that base. Structured control flow (`Block`, `Loop`, `If`, `Else`, `End`, `Br`, `BrIf`), 64-bit integer constants, and typed scalar locals bring the instruction set to 40 opcodes. Termination is now a capability policy: without `CAP_DIVERGE` every loop needs a verified iteration bound, and with `CAP_DIVERGE` a program may loop without bound. Structured programs run in `Tier 0` and as native `ELF64` binaries with identical results.

We are executing a six-milestone roadmap (see **[`ROADMAP.md`](./ROADMAP.md)**) to expand Rul from a columnar compute engine into a **general-purpose systems language capable of replacing Rust and Go for AI-generated software**. Each milestone is defined by programs that must run identically in `Tier 0` and as native binaries:

1. **Milestone 1 (`v0.5`): Turing-Complete Core**
   - Integer and byte arithmetic, declared growable memory regions with runtime bounds checks, byte input and output, a function table with recursion guarded by host-independent call-depth and stack limits, and single-pass scalar register allocation. Exit programs: a Brainf\*ck interpreter written in Rul, recursive `fib`, `quicksort`, a prime `sieve`, and `sha256`.
2. **Milestone 2 (`v0.6`): Ownership Across Joins, Linking & Standard Library**
   - Region ownership and leases checked across branches, loops, and calls in `O(N)`, a constant data section, linking of separately verified modules, and `Vec`, `String`, and `HashMap` written in Rul.
3. **Milestone 3 (`v0.7`): Capability-Gated Linux I/O & Columnar Bridge**
   - File, clock, and network syscalls gated by fine-grained `CapabilityMask` bits, and a region-as-arena view so general programs call the columnar SIMD engine.
4. **Milestone 4 (`v0.8`): Thread-Per-Core Concurrency & Network Servers**
   - Share-nothing worker tasks and typed channels that transfer region ownership across cores, with an HTTP key-value server benchmarked against Go and Rust.
5. **Milestone 5 (`v0.9`): C ABI FFI & AI Synthesis Benchmark**
   - `CAP_FFI`-gated calls into C libraries, verifier diagnostics with repair hints, and a measured first-shot code generation benchmark.
6. **Milestone 6 (`v1.0`): `AArch64` Backend, Formal Proofs & Format Freeze**
   - A second zero-LLVM native backend, continuous differential fuzzing, Kani proofs for the verifier, and the frozen `.rl` `v1.0` format.

---

## Comparison: C vs. Go vs. Rust vs. Rul (`.rl`)

| Dimension | C (`clang -O3`) | Go (`gc`) | Rust (`rustc + LLVM`) | **Rul (`.rl`)** |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Author** | Human programmer | Human engineering teams | Human systems engineers | **AI Agents (with human-auditable tools)** |
| **Artifact Format** | Text (`.c`, `.h`) | Text (`.go`) | Text (`.rs`) | **Flat Binary `.rl` (16-Byte Instructions)** |
| **Parse Overhead** | High | Medium | Very High (Macros, AST) | **0 ms (Direct binary load)** |
| **Memory Safety** | None (Manual pointers) | Garbage Collector | Borrow Checker (Constraint solver) | **Versioned `TokenOp` (`O(1)` check per node)** |
| **Bounds Checking** | Unchecked | Runtime branch per access | Runtime branch (unless elided) | **Load-time `Shape` + `ProofEntry` table** |
| **Verify + Compile Time** | Seconds | Sub-second to seconds | Seconds to minutes | **`~5.8 ns/node` Verify + `~6.4 us` `ELF64` AOT** |
| **Sandbox Security** | Full OS process | Full OS process | Full OS process (`unsafe` crates) | **64-Bit `CapabilityMask` in File Header** |

---

## Repository Architecture

1. **16-Byte Binary Format ([`src/bytecode.rs`](./src/bytecode.rs)) & Symbolic Assembler ([`src/transpiler.rs`](./src/transpiler.rs))**:
   - 64-byte `RulHeader`, 16-byte `ArenaDescriptor` table, 16-byte `ProofEntry` table, and 16-byte `NodeFrame` instructions across 40 opcodes (30 columnar dataflow opcodes plus 10 structured control-flow and scalar-local opcodes).
   - Lossless `.rls` <-> `.rl` assembler and decompiler (`assemble_rls`, `program_to_rls`).
2. **Single-Pass `O(N)` Static Verifier ([`src/verifier.rs`](./src/verifier.rs))**:
   - Checks acyclic instruction ordering (`in_a < idx`), memory version transitions (`AffineConsume` / `ReadLease`), slice/gather bounds (`ProofEntry`), and `CapabilityMask` permissions in one pass.
3. **3-Tier Execution Engine & Autotuner ([`src/engine.rs`](./src/engine.rs))**:
   - **Tier 0 (`execute_tier0_interpreter`)**: Reference interpreter and golden oracle.
   - **Tier 1 (`execute_tier1_stencil`)**: 8-lane vectorized column kernels.
   - **Tier 2 (`execute_tier2_fused` & `execute_tier2_with_zonemap`)**: L1-cache tile fusion (`1,024` elements/tile), multi-core execution, and `ArenaZoneMap` tile skipping.
   - **White-Box Tracer (`trace_program`)**: Computes per-node `min`, `max`, `sum`, `NaN/Inf` counts, and sample values.
4. **Direct Linux `x86_64` AVX2 `ELF64` Compiler ([`src/aot.rs`](./src/aot.rs))**:
   - Emits standalone Linux `x86_64` `ELF64` binaries in microseconds using 256-bit AVX2 `YMM` instructions (with linear-scan register allocation across `ymm2..ymm13`), x87 transcendental stencils, and zero-copy `mmap`, with zero `LLVM` or `libc` dependencies.
5. **Model Context Protocol Server ([`src/mcp.rs`](./src/mcp.rs) & [`src/bin/rul_mcp.rs`](./src/bin/rul_mcp.rs))**:
   - In-memory `stdio` JSON-RPC 2.0 MCP server (`rul-mcp`) providing 6 tools for AI agents: `rul_assemble_and_verify`, `rul_test_differential`, `rul_autotune_schedule`, `rul_compile_standalone`, `rul_trace`, and `rul_cas_intrinsics`.
6. **Content-Addressed Intrinsics ([`src/cas.rs`](./src/cas.rs)) & Embedding API ([`src/ffi.rs`](./src/ffi.rs))**:
   - 64-bit FNV-1a content-addressed store (`RulCasStore`) with `I64`/`U8` kernels (`sort`, `hash_join`, `group_by`, `simd_scan_byte`, `fnv1a_hash`) and zero-panic `#[repr(C)]` embedding functions under `#![forbid(unsafe_code)]`.

---

## Toolchain Commands

```bash
# Install from prebuilt release tarball or source
curl -fsSL https://raw.githubusercontent.com/rul-lang/rul/main/install.sh | sh

# Or build and install with Cargo
cargo install --git https://github.com/rul-lang/rul --bins --force

# CLI workflow
rul new my-ai-pipeline
cd my-ai-pipeline
rul check      # Run single-pass O(N) static verifier
rul build      # Assemble .rls -> build/main.rl and emit standalone ELF64 binary
rul run        # Execute Tier 2 + ELF64 AOT with differential verification
rul trace      # Print per-node min/max/sum/NaN inspection table
rul autotune   # Sweep ScheduleHint policies and pick fastest schedule

# Run the full 10-pipeline differential and AOT benchmark suite
rul-bench

# Run the MCP server for AI agents
rul-mcp

# Run all unit, verifier, differential, and AOT tests
cargo test
```

---

## Documentation

- **[`ROADMAP.md`](./ROADMAP.md)**: Six-Milestone Technical Roadmap from the `v0.4` Columnar Engine to the `v1.0` General-Purpose Systems Language.
- **[`docs/rul_spec_v0.md`](./docs/rul_spec_v0.md)**: Official `.rl` Binary Layout, 128-Bit Instruction Frame & `O(N)` Verifier Specification (`v0.4` baseline plus the implemented `v0.5` structured control-flow subset).
- **[`docs/about_rul.md`](./docs/about_rul.md)**: Plain-English Overview of How Rul Works.
- **[`docs/new_language_for_AI.md`](./docs/new_language_for_AI.md)**: Design Notes & Architectural Breakdown.
- **[`docs/WHITEPAPER_RUL_V0_1.md`](./docs/WHITEPAPER_RUL_V0_1.md)**: Technical Whitepaper & Empirical Evaluation (Zenodo Archive).
- **[`docs/prior_art/`](./docs/prior_art/)**: Cryptographic SHA-256 Prior Art Manifest & OpenTimestamps Proofs.

---

## License, Citation & Trademark

- **License**: Dual-licensed under either the **[MIT License](./LICENSE-MIT)** or the **[Apache License, Version 2.0](./LICENSE-APACHE)** at your option.
- **Attribution**: Redistributions must preserve the attribution notice in [`NOTICE`](./NOTICE).
- **Trademark Policy**: The **"Rul"** name, logo, and `RL_MAGIC` conformance mark are governed by [`TRADEMARK.md`](./TRADEMARK.md).
- **Citation**: See [`CITATION.cff`](./CITATION.cff).

**Creator & Lead Architect**: Tran Trong Tri (Tymon Tran) — `<tymon3568@gmail.com>`
