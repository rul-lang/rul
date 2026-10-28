# Rul (`.rl`): A 128-Bit Schedule- and Proof-Carrying Binary SSA-DAG Programming Language and Execution Engine for Autonomous AI Agents

- **Author & Inventor**: Tran Trong Tri (Tymon Tran) `<tymon3568@gmail.com>`
- **Version**: `v0.1.0` (2026-10-28)
- **Canonical Repository**: `https://github.com/rul-lang/rul`
- **License**: Dual-licensed under `MIT OR Apache-2.0`

---

## Abstract

Existing systems programming languages—including C, Go, and Rust—were fundamentally co-designed around human biological constraints: a ~40 bit/second keyboard output bandwidth, visual inspection of UTF-8 text files, and a working memory limit of $7 \pm 2$ variables. Because humans cannot manually annotate explicit memory lifetimes, refinement bounds proofs, and loop-fusion schedules on every instruction, traditional compilers must expend seconds to minutes performing lexical analysis, AST parsing, monomorphization, constraint-solving borrow checking, and over 100 heuristic LLVM optimization passes. For autonomous AI Agents operating in high-frequency synthesis and verification loops, this "human serialization and compiler search tax" is a primary bottleneck.

We introduce **Rul (`.rl` — Rule-Carrying Unified Linear Graph)**, an AI-native binary programming language and execution engine. Rul eliminates textual source code from the core execution pipeline, replacing it with a memory-mapped, zero-copy topological Static Single Assignment Directed Acyclic Graph (SSA-DAG) in which every instruction is a fixed 128-bit (16-byte) frame (`NodeFrame`) carrying three explicit components:

$$\text{Artifact (}\mathtt{.rl}\text{)} = \text{Algorithm} + \text{Schedule} + \text{Proof}$$

Because the AI Agent emits explicit schedule hints (`ScheduleHint`) and memory/bounds witnesses (`TokenOp`, `ProofIdx`) alongside each operation, the Rul compiler performs **zero inference and zero search**:
1. A **Single-Pass $O(N)$ Linear Verifier** validates topological termination, linear memory ownership (`AffineConsume` and shared `ReadLease`), zero-overhead bounds safety (`Bounds-Safe by Construction` via columnar `Arena<Shape>` and sidecar `ProofEntry` witnesses), and hardware sandboxing (`CapabilityMask`) at ~46 nanoseconds per node ($< 0.5\text{ ms}$ for a 10,000-node program).
2. A **3-Tier Execution Engine** (Tier 0 Golden Oracle Interpreter, Tier 1 Vector Stencils, Tier 2 Whole-Graph Kernel Fuser with L1-Cache `ArenaZoneMap` skipping) and a **Direct `x86_64` `ELF64` Ahead-of-Time (AOT) Compiler** (`src/aot.rs`) execute verified `.rl` graphs with zero intermediate heap allocations, achieving $20\times$–$36\times$ speedups over legacy Array-of-Structs (AoS) pipelines and sub-millisecond standalone native binary compilation without LLVM.

---

## 1. First-Principles Motivation: Why AI Agents Need a Non-Human Language

### 1.1 The Serialization and Search Tax of Textual Languages
When a Large Language Model (LLM) or autonomous agent synthesizes a data processing pipeline in Rust or C:
1. It emits thousands of ASCII tokens (`pub fn`, `{`, `}`, `;`, variable identifiers) that represent syntax sugar for human eyes.
2. The compiler parses those characters back into an Abstract Syntax Tree (AST), solves trait and lifetime equations, lowers high-level semantics into scalar-oriented LLVM IR, and runs heuristic auto-vectorization passes to guess whether loops can be fused or vectorized.

In Rul, the contract between the AI Agent and the compiler is inverted:
$$\text{Rul Engine} = \text{Linear Verifier } O(N) + \text{Vector Stencil / Kernel Fuser / ELF64 Emitter } O(N)$$

### 1.2 Overcoming the Von Neumann Bottleneck via Columnar SoA & Kernel Fusion
On modern hardware, sequential imperative loops over Array-of-Structs (`Vec<Struct>`) waste up to 90% of memory bandwidth fetching unused struct fields into L1/L2 caches and materializing intermediate vectors in RAM between pipeline stages. Rul mandates **64-byte cache-line-aligned Structure-of-Arrays (SoA) Columnar Arenas** (`ArenaDescriptor`) as its sole primary memory primitive, enabling whole-graph register fusion and $O(1)$ L1-tile skipping via `ArenaZoneMap`.

---

## 2. The `.rl` Binary Specification (v0.1)

A `.rl` binary image consists of four contiguous, aligned sections:

```text
+-----------------------------------------------------------------------+
| 1. RulHeader (64 bytes, magic = b"RUL\x00\x01\x00\x00\x00")           |
+-----------------------------------------------------------------------+
| 2. ArenaDescriptor Table (arena_count * 16 bytes)                     |
+-----------------------------------------------------------------------+
| 3. ProofEntry Sidecar Table (proof_count * 16 bytes)                  |
+-----------------------------------------------------------------------+
| 4. NodeFrame Table (node_count * 16 bytes — 128-bit Instruction DAG)  |
+-----------------------------------------------------------------------+
```

### 2.1 The 128-Bit `NodeFrame` (16 Bytes)

```text
Byte:  0          1            2            3            4..7 (32-bit)
     +----------+------------+------------+------------+------------------------+
     | opcode   | type_lane  | schedule   | proof_idx  | token_op (32 bits)     |
     | (u8)     | (u8)       | _hint (u8) | (u8)       | [mode:2b|id:14b|ver:16]|
     +----------+------------+------------+------------+------------------------+
     | in_a (u32)                         | in_b (u32)                          |
     +------------------------------------+-------------------------------------+
Byte:  8..11 (32-bit)                       12..15 (32-bit)
```

- **`opcode (u8)`**: Columnar and scalar algebraic primitives (`ArenaInput`, `ConstF64`, `ArenaSlice`, `ArenaMutate`, `Add`, `Sub`, `Mul`, `CmpGt`, `SelectZero`, `FilterCompact`, `ReduceSum`, `ReduceMax`, `Gather`, `RingEnqueue`, `AssertContract`).
- **`type_lane (u8)`**: High nibble encodes element type (`F64`, `I64`, `BoolMask`); low nibble encodes SIMD width (`Scalar`, `Lane4`, `Lane8`), eliminating generic monomorphization.
- **`schedule_hint (u8)`**: High nibble specifies `fusion_group` (`0` = unfused, `1..15` = kernel fusion group ID); low nibble specifies execution policy (`AutoL1Tile`, `SimdStream`, `MultiCore`, `GpuDma`).
- **`proof_idx (u8)`**: `0` indicates `TRIVIAL_BY_SHAPE` (verified in $O(1)$ by matching column lengths); `1..255` indexes into the `ProofEntry` table (`SliceBounds` or `IndexInRange`).
- **`token_op (u32)`**: Encodes `PureValue (0b00)`, shared `ReadLease (0b01)`, or linear destructive `AffineConsume (0b10)` over `(arena_id, version)`.

---

## 3. Single-Pass $O(N)$ Linear Verification Rules

The static verifier (`src/verifier.rs`) scans `nodes[0..node_count]` in a single forward loop without backtracking:

1. **Topological SSA Invariant**: Every node $i$ referencing prior node indices (`in_a`, `in_b`) must satisfy $\text{in\_a} < i$ and $\text{in\_b} < i$, guaranteeing acyclic dataflow and deterministic termination.
2. **Affine Ownership & Read-Lease Invariant**: Each `Arena` maintains a monotonically increasing `current_version`. `ReadLease(a, v)` requires `current_version[a] == v`. `AffineConsume(a, v)` requires `current_version[a] == v` and `Arena[a].mutable == true`, immediately incrementing `current_version[a] = v + 1` and invalidating all stale tokens in $O(1)$.
3. **Bounds-Safety by Construction**: Elementwise binary operations require identical column lengths (or scalar broadcast). Indirect operations (`ArenaSlice`, `Gather`) require a valid `ProofEntry` bound against `target_arena.element_count`, eliminating runtime bounds checks in inner loops.
4. **Zero-Trust Capability Sandbox**: Hardware I/O opcodes (`RingEnqueue`) are rejected unless the corresponding bit in `RulHeader.capability_mask` is explicitly granted.

---

## 4. Empirical Evaluation (`rul-bench` & `suite10`)

Evaluated on the canonical 10,000,000-element financial risk pipeline (`Filter(amount > 1000.0) -> Square -> ReduceSum`) and the 10-Pipeline Differential Verification Suite (`src/suite10.rs`):

- **Binary Compactness**: The complete 7-node verified `.rl` binary image is **192 bytes** (64B header + 16B arena descriptor + 112B instruction frames).
- **Verification Throughput**: Checking a 10,000-node `.rl` graph completes in **~0.46 ms** (~46 ns/node, >340 MB/s verification bandwidth).
- **Direct AOT Native Compilation (`src/aot.rs`)**: Compiles a verified `.rl` graph directly into a standalone **~4.3 KB Linux `x86_64` `ELF64` executable** in microseconds with **zero external linker or LLVM dependency**.
- **Runtime Execution Throughput**:
  - **Zero Intermediate Heap Allocations (`0 bytes`)** in Tier 2 fused execution.
  - Up to **23x–36x faster** than legacy AoS and unfused multi-pass pipelines, and **order-of-magnitude faster** when combined with `ArenaZoneMap` L1-cache tile skipping on clustered analytical columns.
  - **100% Differential Agreement** across Tier 0 (Oracle), Tier 1 (Vector Stencils), Tier 2 (Whole-Graph Fuser), and standalone AOT `ELF64` binaries across all 10 benchmark pipelines (`P01`–`P10`).

---

## 5. Citation

```bibtex
@software{tran2026rul,
  author       = {Tran, Trong Tri (Tymon Tran)},
  title        = {{Rul (.rl): A 128-Bit Schedule- and Proof-Carrying Binary SSA-DAG Programming Language and Execution Engine for Autonomous AI Agents}},
  year         = {2026},
  month        = oct,
  version      = {0.1.0},
  url          = {https://github.com/rul-lang/rul},
  license      = {MIT OR Apache-2.0}
}
```
