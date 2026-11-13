# Rul Language & `.rl` Binary Execution Specification (v0.4 + v0.5 TC Core Slice 1)

- **Status**: APPROVED (`v0.4.0` Production Baseline — Columnar Dataflow, Math & Vector Compute Domain), extended with the implemented `v0.5` TC Core slice 1 (structured control flow, typed scalar locals, and the `CAP_DIVERGE` termination policy). Later `v0.5` slices listed in [`ROADMAP.md`](../ROADMAP.md) are not part of this specification until they are implemented.
- **File Extensions**: `.rl` (128-bit binary graph), `.rls` (lossless flat symbolic IR)
- **Core Equation**:
  $$\text{Artifact (}\mathtt{.rl}\text{)} = \text{Algorithm} + \text{Schedule} + \text{Proof}$$
  $$\text{Engine} = \text{Linear Verifier } O(N) + \text{Tiered Stencil/Fuser/ELF64 AOT Engine } O(N)$$

---

## 1. Scope & Architectural Invariants

1. **Binary-First Single Source of Truth**:
   - Every executable Rul program is a flat binary `.rl` image aligned on 16-byte and 64-byte boundaries for direct zero-copy memory mapping (`mmap`).
   - The core execution engine contains zero textual lexers or AST parsers. Any conversion between an AI Agent's symbolic representation (`.rls`) and `.rl` is performed at the boundary by the $O(N)$ symbolic assembler/decompiler ([`src/transpiler.rs`](../src/transpiler.rs)).
2. **Forward-Ordered SSA Graph with Termination as a Capability Policy**:
   - Every instruction node at index $i$ (`0 <= i < node_count`) may only reference input nodes with indices strictly `< i`, so the entire program is verified in **a single forward linear scan $O(N)$**.
   - A columnar program (no structured opcodes) is a **Topological Single-Static-Assignment Directed Acyclic Graph (SSA-DAG)** and always terminates.
   - A structured program adds back edges only through `Loop` with `Br`/`BrIf`. Without `CAP_DIVERGE` (`0x08`) in `capability_mask`, every `Loop` must carry a `LoopBound` proof, so the program is guaranteed to halt. With `CAP_DIVERGE`, unbounded loops are admitted and the program may run forever by design.
3. **Orthogonal Separation of `Safety Verification` and `Intent Verification`**:
   - **Layer 1 — Safety Verification ($O(N)$ Static Check)**: Mathematically proves 100% absence of Use-After-Consume, Read/Write aliasing, buffer overflows, token version wraparound, and unauthorized hardware I/O (`CapabilityMask`).
   - **Layer 2 — Intent & White-Box Verification (`AssertContract` + Tier 0 Golden Oracle + `rul trace`)**: Verifies output domain invariants (`!is_nan() && is_finite()`) and provides deterministic node-by-node telemetry (`min`, `max`, `sum`, `nan_or_inf_count`, and sample head values) across every SSA instruction.

---

## 2. Binary Layout of `.rl`

A `.rl` file or memory buffer consists of 4 contiguous sections:

```text
+-----------------------------------------------------------------------+
| 1. RulHeader (64 bytes, 64-byte aligned)                              |
+-----------------------------------------------------------------------+
| 2. ArenaDescriptor Table (arena_count * 16 bytes)                     |
+-----------------------------------------------------------------------+
| 3. ProofEntry Sidecar Table (proof_count * 16 bytes)                  |
+-----------------------------------------------------------------------+
| 4. NodeFrame Table (node_count * 16 bytes — 128-bit Instruction DAG)  |
+-----------------------------------------------------------------------+
```

### 2.1. `RulHeader` (64 bytes)

| Byte Offset | Field | Type | Description |
| :--- | :--- | :--- | :--- |
| `0..8` | `magic` | `[u8; 8]` | Canonical magic bytes: `b"RUL\x00\x01\x00\x00\x00"` |
| `8..16` | `graph_hash` | `u64` | 64-bit FNV-1a integrity hash over the Arena, Proof, and Node sections |
| `16..24` | `capability_mask` | `u64` | Hardware capability bitmask (`0x00` = Pure Compute, `0x01` = NetRing, `0x02` = StorageDMA, `0x04` = GpuDispatch, `0x08` = Diverge: unbounded `Loop` allowed) |
| `24..28` | `arena_count` | `u32` | Number of declared columnar `ArenaDescriptor` entries |
| `28..32` | `proof_count` | `u32` | Number of refinement certificates in the `ProofEntry` sidecar table |
| `32..36` | `node_count` | `u32` | Number of 128-bit instruction frames in the `NodeFrame` table |
| `36..40` | `output_node` | `u32` | 0-based index of the primary output node |
| `40..64` | `reserved` | `[u8; 24]` | Reserved padding (must be all zeros) |

### 2.2. `ArenaDescriptor` (16 bytes)

Every input or mutable memory region is a homogeneous **Structure-of-Arrays (SoA) Column**:

| Byte Offset | Field | Type | Description |
| :--- | :--- | :--- | :--- |
| `0..4` | `arena_id` | `u32` | Arena identifier (`0 .. arena_count - 1`) |
| `4..8` | `element_count` | `u32` | Number of elements in the column (`Shape Length`) |
| `8` | `type_lane` | `u8` | Element type and SIMD lane width (see Section 2.4) |
| `9` | `mutability` | `u8` | `0 = Immutable Input`, `1 = Linear Mutable Region` |
| `10..12` | `alignment` | `u16` | Byte alignment requirement (`64` bytes for AVX-512 / CPU cache line) |
| `12..16` | `initial_version` | `u32` | Initial `AffineToken` version (`u16` lower bits) for this `Arena` |

### 2.3. `ProofEntry` (16 bytes — Sidecar Refinement Table)

In columnar data pipelines, over 90% of nodes use `proof_idx = 0` (`TRIVIAL_BY_SHAPE` — bounds-safe by construction because operand columns share an identical `element_count`). The `ProofEntry` table is consulted only for sub-slicing (`ArenaSlice`), indirect indexing (`Gather`), and structured control-flow witnesses (`BlockSig`, `LoopBound`):

| Byte Offset | Field | Type | Description |
| :--- | :--- | :--- | :--- |
| `0..2` | `rule_kind` | `u16` | `1 = SliceBounds (lo <= hi <= len)`, `2 = GatherIndexInBounds (max_idx <= target_len)`, `3 = BlockSig`, `4 = LoopBound` |
| `2..4` | `target_arena` | `u16` | Target `Arena` ID certified by this proof (strictly bound to operand `origin_arena`) |
| `4..8` | `bound_lo` | `u32` | Non-negative lower bound |
| `8..12` | `bound_hi` | `u32` | Upper bound (must be `<= effective_target_len`, i.e., `n.min(target_arena.element_count)` where `n` is the sliced column length of `in_a`) |
| `12..16` | `witness_node` | `u32` | Witness index node (`in_b` for `Gather`) |

The structured rule kinds reuse the same four fields:

| `rule_kind` | `target_arena` | `bound_lo` | `bound_hi` | `witness_node` |
| :--- | :--- | :--- | :--- | :--- |
| `3 = BlockSig` | Tracked arena ID, or `0` when the block mutates no arena | Arena version on entry | Arena version on exit (`bound_lo <= bound_hi`) | Result `type_lane` byte, or `0` for a block without a result |
| `4 = LoopBound` | `0` | `0` | Maximum back-edge iterations (`1 <= bound_hi <= 1_000_000`) | `0` |

### 2.4. 128-Bit `NodeFrame` Instruction Layout (16 bytes)

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

1. **`opcode (u8)` — 40-Opcode Instruction Set (30 columnar opcodes from `v0.4` plus 10 structured control-flow and scalar-local opcodes from `v0.5` TC Core slice 1)**:
   - `0x01` `ArenaInput`: Load columnar slice from `ArenaDescriptor[in_a]`.
   - `0x02` `ConstF64`: 64-bit IEEE-754 floating-point constant (`bits = in_a | (in_b << 32)`).
   - `0x03` `ArenaSlice`: Zero-copy sub-slice of `in_a` certified by `proof_idx`.
   - `0x04` `ArenaMutate`: Linear in-place mutation of mutable `Arena[in_a]` with values from `in_b`, consuming the current `AffineToken` and advancing its version by `+1`.
   - `0x05` `ConstI64`: 64-bit integer constant (`bits = in_a | (in_b << 32)`).
   - `0x06` `LocalGet`: Read scalar local slot `in_a` (`0 <= in_a < 64`).
   - `0x07` `LocalSet`: Write scalar node `in_b` to local slot `in_a` (`0 <= in_a < 64`).
   - `0x10` `Add`: Elementwise addition (`out[i] = a[i] + b[i]`).
   - `0x11` `Sub`: Elementwise subtraction (`out[i] = a[i] - b[i]`).
   - `0x12` `Mul`: Elementwise multiplication (`out[i] = a[i] * b[i]`).
   - `0x13` `Div`: Elementwise IEEE-754 division (`out[i] = a[i] / b[i]`).
   - `0x14` `CmpGt`: Branchless greater-than predicate (`mask[i] = a[i] > b[i]`).
   - `0x15` `SelectZero`: Branchless predicated selection (`out[i] = mask[i] ? val[i] : 0.0`).
   - `0x16` `Min`: Branchless elementwise minimum (`out[i] = a[i] < b[i] ? a[i] : b[i]`).
   - `0x17` `Max`: Branchless elementwise maximum (`out[i] = a[i] > b[i] ? a[i] : b[i]`).
   - `0x18` `CmpLt`: Branchless less-than predicate (`mask[i] = a[i] < b[i]`).
   - `0x19` `CmpEq`: Branchless equality predicate (`mask[i] = a[i] == b[i]`).
   - `0x1A` `Abs`: Branchless IEEE-754 absolute value (`out[i] = |a[i]|`).
   - `0x1B` `Sqrt`: Branchless IEEE-754 square root (`out[i] = sqrt(a[i])`).
   - `0x1C` `Floor`: Branchless floor rounding toward $-\infty$ (`out[i] = floor(a[i])`).
   - `0x1D` `Ceil`: Branchless ceiling rounding toward $+\infty$ (`out[i] = ceil(a[i])`).
   - `0x1E` `Ln`: Elementwise natural logarithm (`out[i] = ln(a[i])`).
   - `0x1F` `Exp`: Elementwise natural exponential (`out[i] = exp(a[i])`).
   - `0x20` `FilterCompact`: Stream compaction filtering elements of `in_b` where mask `in_a` is true.
   - `0x21` `ReduceSum`: Horizontal sum reduction of column `in_a` to a scalar `f64`.
   - `0x22` `ReduceMax`: Horizontal maximum reduction of column `in_a` to a scalar `f64`.
   - `0x23` `Gather`: Indirect indexed gather `target[indices[i]]` (requires `proof_idx > 0` and exact integer float indices `idx.fract() == 0.0`).
   - `0x24` `ReduceMin`: Horizontal minimum reduction of column `in_a` to a scalar `f64`.
   - `0x25` `MaskAnd`: Branchless boolean mask conjunction (`mask[i] = a[i] && b[i]`).
   - `0x26` `MaskOr`: Branchless boolean mask disjunction (`mask[i] = a[i] || b[i]`).
   - `0x27` `Fma`: Single-rounded IEEE-754 fused multiply-add (`out[i] = fma(a[i], b[i], c[i]) = a[i] * b[i] + c[i]`, with `in_b` packing `(b_idx: u16) | ((c_idx: u16) << 16)`).
   - `0x30` `RingEnqueue`: Enqueue output to a hardware ring buffer (requires `CAP_NET_RING` in `RulHeader.capability_mask`).
   - `0x40` `AssertContract`: Runtime contract assertion verifying finite, non-NaN scalar output.
   - `0x50` `Block`: Structured block header; `Br`/`BrIf` targeting it jump forward to its `End` (optional `proof_idx -> BlockSig`).
   - `0x51` `Loop`: Structured loop header; `Br`/`BrIf` targeting it jump backward to the loop start (`proof_idx -> LoopBound`, or `0` with `CAP_DIVERGE`).
   - `0x52` `If`: Structured conditional header (`in_a = cond_node`, a scalar `BoolMask`; optional `proof_idx -> BlockSig`).
   - `0x53` `Else`: Alternate branch delimiter of the enclosing `If` (`in_a = optional_then_result`, `in_b = has_result`).
   - `0x54` `End`: Terminator of the innermost `Block`, `Loop`, or `If` (`in_a = optional_result`, `in_b = has_result`).
   - `0x55` `Br`: Unconditional branch to the enclosing scope at `in_a = relative_depth` (`0` = innermost).
   - `0x56` `BrIf`: Conditional branch (`in_a = cond_node`, `in_b = relative_depth`).
   - On scalar `I64` operands, `Add`, `Sub`, and `Mul` wrap on overflow (two's complement).
2. **`type_lane (u8)`**:
   - High 4 bits (`ElemType`): `0x1 = F64`, `0x2 = I64`, `0x3 = BoolMask`.
   - Low 4 bits (`SimdWidth`): `0x1 = Scalar`, `0x4 = Lane4 (256-bit AVX2)`, `0x8 = Lane8 (512-bit AVX-512 / 8-way unrolled)`.
3. **`schedule_hint (u8)`**:
   - High 4 bits (`fusion_group`): `0` = Unfused; `1..15` = Kernel Fusion Group ID.
   - Low 4 bits (`exec_policy`): `0 = AutoL1Tile`, `1 = SimdStream`, `2 = MultiCore`, `3 = GpuDma`.
4. **`token_op (u32)`**:
   - High 2 bits (`TokenMode`):
     - `0b00` (`PureValue`): Pure value operation; does not mutate `Arena` ownership state.
     - `0b01` (`ReadLease`): Shared immutable borrow of `Arena[id]` at version `ver`. Multiple nodes may hold concurrent `ReadLease` tokens as long as `ver` matches the live arena version.
     - `0b10` (`AffineConsume`): Exclusive linear consumption of mutable `Arena[id]` at version `ver`, advancing the arena's live version to `checked_add(ver, 1)`. Any subsequent reuse of stale version `ver` is immediately rejected by the verifier (`StaleOrConsumedToken`), and any attempt to increment past `u16::MAX` is rejected with `TokenVersionOverflow`.

---

## 3. Single-Pass $O(N)$ Linear Verification Rules

The verifier ([`verify_program`](../src/verifier.rs)) scans `nodes[0..node_count]` exactly once, maintaining $O(N)$ metadata:

1. **Topological SSA Rule**:
   - For every node at index $i$, all node-index operands (`in_a`, `in_b`) must satisfy `in_a < i` and `in_b < i`.
2. **Linear Ownership Conservation (`AffineToken` & `ReadLease` Rule)**:
   - Each `Arena[a]` initializes with `live_version = initial_version`.
   - `ReadLease(a, ver)` requires `live_version[a] == ver`.
   - `AffineConsume(a, ver)` requires `live_version[a] == ver` AND `Arena[a].mutable == true`, then updates `live_version[a] = live_version[a].checked_add(1)` (failing with `TokenVersionOverflow` on `u16::MAX`).
3. **Shape Unification & Origin-Bound Refinement Proof Rule**:
   - Elementwise binary ops (`Add`, `Sub`, `Mul`, `Div`, `Min`, `Max`, `CmpGt`, `CmpLt`, `CmpEq`, `SelectZero`) require matching column shapes (`Column(N) == Column(N)`) or scalar broadcasting (`Scalar` with `Column(N)`).
   - `ArenaSlice` and `Gather` require a valid 1-based `proof_idx` whose `ProofEntry.target_arena` matches the input operand's `origin_arena` and satisfies `bound_lo <= bound_hi <= effective_target_len`, where `effective_target_len = n.min(target_arena.element_count)` (`n` is the column length of operand `in_a`). When `in_a` is an `ArenaSlice`, `bound_hi` is bounded by the sliced column length rather than only the base arena's `element_count`.
   - `VerifiedSummary.zero_runtime_bounds_checks` is `true` iff the program contains no dynamic `Gather` instructions (`!has_gather`).
4. **Zero-Trust Hardware Capability Rule**:
   - `RingEnqueue` requires `(header.capability_mask & CAP_NET_RING) != 0`.
5. **Structured Control Stack Rule**:
   - `Block`, `Loop`, and `If` push a frame on an explicit control stack; `Else` must match an open `If`; `End` pops the innermost frame. Unmatched `Else`/`End` and frames left open at the last node are rejected.
   - `Br` and `BrIf` must name a `relative_depth` smaller than the control stack depth, and `If`/`BrIf` conditions must be scalar `BoolMask` nodes.
   - SSA nodes defined inside a scope are sealed at its `Else`/`End` and cannot be referenced afterwards. State crosses a scope boundary only through typed locals or the scope's declared result, and `output_node` must be a non-void node that is visible at the end of the program.
   - A `BlockSig` proof declares the arena version transition and result type of a scope, so both arms of an `If` leave every `AffineToken` at the same version.
6. **Typed Local Rule**:
   - There are 64 scalar local slots. `LocalSet` fixes the slot's `ElemType` on first write and every later write must use the same type (`LocalTypeMismatch` otherwise).
   - `LocalGet` requires the slot to be definitely initialized on every path that reaches it (`UninitializedLocalRead` otherwise).
7. **Termination Capability Rule**:
   - When `(header.capability_mask & CAP_DIVERGE) == 0`, a `Loop` with `proof_idx == 0` is rejected with `MissingCapability`, and a `Loop` with `proof_idx > 0` must reference a valid `LoopBound` proof.
   - When `CAP_DIVERGE` is set, both unbounded (`proof_idx == 0`) and bounded (`proof_idx > 0`) loops are admitted.

---

## 4. 3-Tier Execution Engine, Direct `ELF64` AOT & White-Box Tracer

1. **Tier 0 — Zero-Copy Reference Columnar Interpreter (`execute_tier0_interpreter`)**:
   - Evaluates each node sequentially with zero-copy `Cow<[f64]>` borrows on `ArenaInput`/`ArenaSlice`, serving as the mathematical Golden Oracle.
2. **Tier 1 — Dead-Node-Eliminated Vector Stencil Engine (`execute_tier1_stencil`)**:
   - Computes exact backward liveness from `output_node` and applies 8-lane unrolled vector stencils (`CHUNK = 8` `f64`).
3. **Tier 2 — Exact-DAG L1-Cache Tile Fuser (`execute_tier2_fused` & `execute_tier2_with_zonemap`)**:
   - Evaluates arbitrary elementwise-to-reduction and post-reduction scalar DAGs in `1,024`-element (`8 KB`) L1-cache tiles with **0 bytes of intermediate column allocation** ($O(1)$ L1-tile scratchpad independent of column length $N$), Auto-MultiCore thread dispatch (`ExecPolicy::MultiCore`), and $O(1)$ cold-tile skipping via `ArenaZoneMap`.
4. **Direct Linux `x86_64` AVX2 `ELF64` AOT Compiler (`compile_rl_to_executable`)**:
   - Emits standalone 628–1037 byte statically-linked Linux `ELF64` native binaries in microseconds using 256-bit AVX2 `YMM` vector instructions and 64-bit SSE2 scalar tail cleanup.
5. **White-Box Execution Tracer (`trace_program` & `rul trace`)**:
   - Records deterministic per-node statistics (`min`, `max`, `sum/trues`, `nan_or_inf_count`, and sample head values) for every SSA node `#0..#N-1`.
6. **Structured Scalar Path (`Tier 0 == ELF64 AOT`)**:
   - Programs that use structured control-flow or scalar-local opcodes run in `Tier 0` and compile through a dedicated single-pass scalar `x86_64` code generator (one 8-byte frame slot per SSA node, 64 local slots, forward `rel32` patch lists resolved at `Else`/`End`). Results must be bit-identical between `Tier 0` and the native binary; a scalar `I64` output is carried as its 64-bit pattern.
   - `Tier 1` and `Tier 2` are columnar accelerators and fail closed on these programs with `ExecError::UnsupportedInColumnarTier`.
   - A bounded `Loop` that takes more than `bound_hi` back edges fails closed: `ExecError::LoopBoundExceeded` in `Tier 0` and process exit code `70` in the native binary.
   - Current limits of this path: no arena operands, and at most `65,536` nodes per program.
