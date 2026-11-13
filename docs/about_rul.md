# About Rul (`.rl`) — A Plain-English Guide

This guide explains what **Rul (`.rl`)** is, why it uses a 16-byte binary instruction format instead of human text syntax, and how it is evolving from a high-speed data engine (`v0.4`) into a general-purpose systems language (`v1.0`).

---

## 1. The Problem with Human Text Languages for AI Agents

Languages like C, Python, Go, and Rust were designed for human eyes and ten fingers:
- Humans type slowly (~40 bits per second) and can only keep a few variables in short-term memory at once.
- Because humans cannot manually write memory-lifetime proofs, array bounds proofs, and CPU cache schedules on every single instruction, compilers like `rustc` and `LLVM` have to infer them.
- When an AI agent writes Rust code today, it outputs thousands of characters of text (`pub fn`, `{`, `}`, `;`, variable names). Then `rustc` and `LLVM` spend seconds or minutes parsing that text back into a graph, solving borrow-checker equations, and running over a hundred optimization passes.

An AI agent does not have fingers or eyes. Once an AI knows what it wants to compute, serializing that logic into human text and waiting seconds for a compiler is pure overhead.

## 2. How Rul Works Instead

Rul removes the text step in the core engine. A `.rl` program is a flat binary file made of fixed **16-byte (128-bit) instruction frames**.

Every 16-byte instruction carries three pieces of information side by side:

```text
  .rl Instruction Frame (16 bytes) =  Operation   +     CPU Schedule     +   Safety Witness
                                     (What to do)    (SIMD / Multi-Core)   (Memory Version & Bounds)
```

1. **Operation (`opcode`, `type_lane`, `in_a`, `in_b`)**: What calculation to run and which previous instruction outputs to use as inputs.
2. **CPU Schedule (`schedule_hint`)**: A 1-byte hint telling the engine how to run the instruction on hardware (for example, single-core SIMD registers or multi-core L1 cache tiles).
3. **Safety Witness (`token_op`, `proof_idx`)**: Explicit tags proving that the memory buffer is still valid and the array indices are in bounds.

> **Why is the Rul toolchain itself written in Rust?**
> Every new language needs an initial host language for its first compiler (just as the first C compiler was written in assembly and the first Rust compiler was written in OCaml). We built the Rul verifier, execution engine, and native `ELF64` compiler in dependency-free Rust (`#![forbid(unsafe_code)]`). Once the `rul` binary is installed, users and AI agents build and run `.rl` programs directly using `rul check`, `rul build`, `rul run`, and `rul trace` without needing `rustc` or `Cargo`.

---

## 3. Four Core Mechanisms Inside Rul

### Mechanism 1: The 16-Byte Binary Layout (`src/bytecode.rs`)

Instead of text keywords and nested braces, a `.rl` file has four fixed binary sections:
- **`RulHeader` (64 bytes)**: File signature (`b"RUL\x00\x01\x00\x00\x00"`), integrity hash, a 64-bit `CapabilityMask` controlling hardware I/O permissions, and table counts.
- **`ArenaDescriptor` Table (16 bytes per entry)**: Declares 64-byte-aligned memory buffers (`Arenas`), their element counts, types, and whether they are read-only or mutable.
- **`ProofEntry` Table (16 bytes per entry)**: Stores compile-time bounds certificates for sub-slices (`ArenaSlice`) and indirect array lookups (`Gather`).
- **`NodeFrame` Array (16 bytes per instruction)**:
  `[ opcode (8b) | type_lane (8b) | schedule_hint (8b) | proof_idx (8b) | token_op (32b) | in_a (32b) | in_b (32b) ]`

### Mechanism 2: Single-Pass `O(N)` Safety Verification (`src/verifier.rs`)

Because every instruction carries its own safety tags, Rul verifies memory safety in a single forward loop from instruction `0` to `N-1` (**~5.8 nanoseconds per instruction**):

1. **No Infinite Loops or Cycles**: Every instruction at index `i` can only read outputs from earlier instructions (`in_a < i` and `in_b < i`).
2. **Versioned Memory Ownership (No Use-After-Free or Aliasing)**:
   - Each memory buffer starts at version `1`.
   - Multiple instructions can read version `1` concurrently using a shared read tag (`ReadLease`).
   - Modifying a buffer in place requires an exclusive consume tag (`AffineConsume`), which checks that the buffer is mutable and bumps its live version from `1` to `2`.
   - If any later instruction tries to use version `1` after it was bumped to `2`, the verifier rejects the program immediately.
3. **Load-Time Bounds Checks**: Elementwise operations check that input array lengths match at load time, avoiding per-element bounds checks inside inner CPU loops.
4. **Hardware Permission Sandbox**: Instructions that talk to external queues or OS resources are rejected at load time unless the corresponding bit in `RulHeader.capability_mask` is enabled.

### Mechanism 3: 3-Tier Execution and Direct `ELF64` Native Compilation (`src/engine.rs`, `src/aot.rs`)

1. **Tier 0 (Reference Interpreter)**: Runs instructions step by step to serve as a correctness oracle and powers `rul trace`.
2. **Tier 1 (Vector Stencil Engine)**: Runs 8-lane SIMD loops over live instructions.
3. **Tier 2 (L1-Cache Tile Fuser)**: Processes arrays in `1,024`-element (`8 KB`) L1-cache tiles with zero intermediate array allocations, optional multi-core parallelism, and `O(1)` tile skipping via `ArenaZoneMap`.
4. **Direct Linux `x86_64` AVX2 `ELF64` Compiler**: Emits standalone **628 to 1,037 byte** native Linux binaries in **~6.4 microseconds** using 256-bit AVX2 `YMM` registers without `LLVM` or `libc`.

### Mechanism 4: Automated Hardware Tuning (`rul autotune` & `rul-mcp`)

After running a program, Rul returns a `HardwareTelemetry` struct reporting verification time, execution time, memory bandwidth (`GB/s`), and intermediate bytes allocated. An AI agent (or the `rul autotune` command) can test different 1-byte `ScheduleHint` settings and keep the fastest schedule while verifying that the output matches Tier 0.

---

## 4. From Data Engine (`v0.4`) to General-Purpose Systems Language (`v1.0`)

In versions `v0.1` through `v0.4`, we intentionally focused on array and vector calculations (`29 opcodes` at the `v0.4.0` release) to prove that 16-byte binary instructions, single-pass `O(N)` verification, and microsecond native compilation work on real hardware.

As detailed in **[`ROADMAP.md`](../ROADMAP.md)**, we are expanding Rul over six milestones into a general-purpose systems language designed to replace Rust and Go for AI-written software. A milestone is finished when a set of concrete programs gives the same result in the reference interpreter (`Tier 0`) and as a native binary:
- **Milestone 1 (`v0.5`) — Turing-Complete Core**: Loops, branches, and typed local variables are already implemented (40 opcodes in total). Next come integer and byte arithmetic, growable memory regions with bounds checks, byte input and output, and functions with recursion. A program that must be guaranteed to stop simply leaves out the `CAP_DIVERGE` permission: the verifier then requires a proven limit on every loop and rejects recursion. Exit programs: a Brainf\*ck interpreter written in Rul, recursive `fib`, `quicksort`, a prime `sieve`, and `sha256`.
- **Milestone 2 (`v0.6`) — Ownership, Linking & Standard Library**: Ownership of memory regions is tracked across branches, loops, and function calls in a single pass. Programs gain constant data, can be linked from separately verified modules, and get `Vec`, `String`, and `HashMap` written in Rul itself.
- **Milestone 3 (`v0.7`) — Systems I/O**: File, clock, and network system calls, each allowed only by its own permission bit, plus a bridge that lets general programs call the SIMD column engine.
- **Milestone 4 (`v0.8`) — Concurrency**: One worker per CPU core with no shared memory, and typed channels that hand a memory region to another core so data races cannot happen without a garbage collector. Exit program: an HTTP key-value server benchmarked against Go and Rust.
- **Milestone 5 (`v0.9`) — C Libraries & AI Benchmark**: Calls into C libraries behind the `CAP_FFI` permission, verifier error messages with repair hints, and a measured benchmark of first-shot AI code generation.
- **Milestone 6 (`v1.0`) — Second CPU Architecture & Freeze**: A native `AArch64` (`ARM64`) Linux backend next to `x86_64`, continuous differential fuzzing, formal model checking (Kani) of the verifier, and the frozen `.rl` `v1.0` format.
