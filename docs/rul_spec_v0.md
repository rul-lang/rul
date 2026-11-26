# Rul Language & `.rl` Binary Execution Specification (`v0.6.0` — 68-Opcode Core, Ownership, Linking & Standard Library)

- **Status**: APPROVED (`v0.6.0` Release — Columnar Dataflow, AVX2 Vector Math, Complete `v0.5` Turing-Complete Core, and Complete `v0.6` Milestone 2 across `v0-6-ownership-1`, `v0-6-data-dispatch-2`, `v0-6-linker-3`, and `v0-6-stdlib-exit-4`: `ArenaVersionTrie` `O(N + A)` multi-arena `AffineToken` joins, statically named typestate-tracked regions with frame-rule `FuncRegionEffect`, `CAP_NONE` `StaticCostSummary` & `rul check --json`, Section 5a `const_data` segments, dense `BrTable`, typed `CallIndirect`, Section 5b `link_meta` & `rul link`, region-backed `stdlib/` modules, and three linked `v0.6` exit programs). Later milestones (`v0.7`–`v1.0`) listed in [`ROADMAP.md`](../ROADMAP.md) are not part of this specification until they are implemented.
- **File Extensions**: `.rl` (128-bit binary graph), `.rls` (lossless flat symbolic IR)
- **Core Equation**:
  $$\text{Artifact (}\mathtt{.rl}\text{)} = \text{Algorithm} + \text{Schedule} + \text{Proof}$$
  $$\text{Engine} = \text{Linear Verifier } O(N) + \text{Tiered Stencil/Fuser/ELF64 AOT Engine } O(N)$$

---

## 1. Scope & Architectural Invariants

1. **Binary-First Single Source of Truth**:
   - Every executable Rul program and linkable library module is a flat binary `.rl` image aligned on 16-byte and 64-byte boundaries for direct zero-copy memory mapping (`mmap`).
   - The core execution engine contains zero textual lexers or AST parsers. Any conversion between an AI Agent's symbolic representation (`.rls`) and `.rl` is performed at the boundary by the $O(N)$ symbolic assembler/decompiler ([`src/transpiler.rs`](../src/transpiler.rs)).
2. **Forward-Ordered SSA Graph with Termination as a Capability Policy**:
   - Every instruction node at index $i$ (`0 <= i < node_count`, or function-relative index inside a `Func`..`End` scope) may only reference input nodes with indices strictly `< i`, so the entire program is verified in **a single forward linear scan $O(N)$**.
   - A columnar program (no structured opcodes) is a **Topological Single-Static-Assignment Directed Acyclic Graph (SSA-DAG)** and always terminates.
   - A structured program adds back edges only through `Loop` with `Br`/`BrIf` and recursion through `Call`/`CallIndirect`. Without `CAP_DIVERGE` (`0x08`) in `capability_mask`, every `Loop` must carry a `LoopBound` proof and the call graph must be acyclic, so the program is guaranteed to halt and the verifier computes a deterministic `StaticCostSummary`. With `CAP_DIVERGE`, unbounded loops and recursion are admitted subject to host-independent ISA stack/depth bounds (`MAX_CALL_DEPTH = 4096`, `MAX_STACK_UNITS = 1 << 20`).
3. **Orthogonal Separation of `Safety Verification` and `Intent Verification`**:
   - **Layer 1 — Safety Verification ($O(N)$ Static Check)**: Mathematically proves 100% absence of Use-After-Consume, Read/Write aliasing, mutation under live read leases, use of moved/dropped regions, buffer overflows, token version wraparound, capability escalation across linked modules, and unauthorized hardware I/O (`CapabilityMask`).
   - **Layer 2 — Intent & White-Box Verification (`AssertContract` + Tier 0 Golden Oracle + `rul trace`)**: Verifies output domain invariants (`!is_nan() && is_finite()`) and provides deterministic node-by-node telemetry (`min`, `max`, `sum`, `nan_or_inf_count`, and sample head values) across every SSA instruction.

---

## 2. Binary Layout of `.rl`

A `.rl` file or memory buffer consists of 4 mandatory sections followed by optional `v0.6` Section 5a (`const_data`) and Section 5b (`link_meta`):

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
| 5a. Constant Data Section (const_data_bytes bytes, max 8 MiB) [v0.6]  |
+-----------------------------------------------------------------------+
| 5b. Module Link Metadata (link_meta_bytes bytes, max 2 MiB) [v0.6]    |
+-----------------------------------------------------------------------+
```

When `const_data_bytes == 0` and `link_meta_bytes == 0`, the binary image and its 64-bit FNV-1a `graph_hash` are byte-for-byte identical to `v0.5.0`.

### 2.1. `RulHeader` (64 bytes)

| Byte Offset | Field | Type | Description |
| :--- | :--- | :--- | :--- |
| `0..8` | `magic` | `[u8; 8]` | Canonical magic bytes: `b"RUL\x00\x01\x00\x00\x00"` |
| `8..16` | `graph_hash` | `u64` | 64-bit FNV-1a integrity hash over Sections 2–4; when Section 5a or 5b is non-empty, folds domain-separated `const_data_bytes`, `const_data`, `link_meta_bytes`, and `link_meta` |
| `16..24` | `capability_mask` | `u64` | Hardware capability bitmask (`0x00` = Pure Compute, `0x01` = NetRing, `0x02` = StorageDMA, `0x04` = GpuDispatch, `0x08` = Diverge: unbounded `Loop` & recursion allowed) |
| `24..28` | `arena_count` | `u32` | Number of declared columnar `ArenaDescriptor` entries |
| `28..32` | `proof_count` | `u32` | Number of refinement certificates in the `ProofEntry` sidecar table |
| `32..36` | `node_count` | `u32` | Number of 128-bit instruction frames in the `NodeFrame` table |
| `36..40` | `output_node` | `u32` | 0-based index of the primary output node (entry `Func` node in function programs) |
| `40..44` | `const_data_bytes` | `u32` | Byte length of Section 5a (`const_data`, `<= MAX_CONST_DATA_BYTES = 8 MiB`) |
| `44..48` | `link_meta_bytes` | `u32` | Byte length of Section 5b (`link_meta`, `<= MAX_LINK_META_BYTES = 2 MiB`) |
| `48..64` | `reserved` | `[u8; 16]` | Reserved trailing header bytes (must be all zeros; non-zero fails closed with `InvalidReservedHeader`) |

### 2.2. `ArenaDescriptor` (16 bytes)

Every columnar input or mutable memory arena is a homogeneous **Structure-of-Arrays (SoA) Column**:

| Byte Offset | Field | Type | Description |
| :--- | :--- | :--- | :--- |
| `0..4` | `arena_id` | `u32` | Arena identifier (`0 .. arena_count - 1`) |
| `4..8` | `element_count` | `u32` | Number of elements in the column (`Shape Length`) |
| `8` | `type_lane` | `u8` | Element type and SIMD lane width (see Section 2.4) |
| `9` | `mutability` | `u8` | `0 = Immutable Input`, `1 = Linear Mutable Region` |
| `10..12` | `alignment` | `u16` | Byte alignment requirement (`64` bytes for AVX-512 / CPU cache line) |
| `12..16` | `initial_version` | `u32` | Initial `AffineToken` version (`u16` lower bits) for this `Arena` |

### 2.3. `ProofEntry` (16 bytes — Sidecar Refinement & Metadata Table)

In columnar data pipelines, over 90% of nodes use `proof_idx = 0` (`TRIVIAL_BY_SHAPE` — bounds-safe by construction because operand columns share an identical `element_count`). The `ProofEntry` sidecar table carries 12 `ProofRuleKind`s (`1..=12`), split between node-referenced proofs (`proof_idx` in `1..=255`) and pre-scanned declarative tables (`FuncSig`, `RegionDecl`, `FuncRegionEffect`, `ConstDataSeg`, `BrTableSpec`, `CallTable`):

| `rule_kind` | `target_arena` (`u16`) | `bound_lo` (`u32`) | `bound_hi` (`u32`) | `witness_node` (`u32`) |
| :--- | :--- | :--- | :--- | :--- |
| `1 = SliceBounds` | Target `Arena` ID | Non-negative `lo` | `hi <= effective_target_len` | `0` |
| `2 = GatherIndexInBounds` | Target `Arena` ID | `0` | `hi <= effective_target_len` | Witness index node (`in_b`) |
| `3 = BlockSig` | Tracked arena ID (`0` if none) | Arena version on entry | Arena version on exit (`lo <= hi`) | Result `type_lane` byte, or `0` for void |
| `4 = LoopBound` | `0` | `0` | Max back-edge iterations (`1..=1_000_000`) | `0` |
| `5 = FuncSig` | Dense `func_id` (`0..N`) | `arity` (`0..=4`) | Packed 4-bit param `ElemType`s | Return `ElemType`, or `0` for `void` |
| `6 = RegionDecl` | Region `slot` (`0..63`) | `min_len` | `max_len` (`max_len * elem_bytes <= 256 MiB`) | `elem \| (role << 8)` (`0=Plain`, `1=Input`, `2=Output`) |
| `7 = RegionIndexInBounds` | Region `slot` (`0..63`) | `0` | `hi <= min_len` (proves `ConstI64` or `BitAnd(_, C)`) | `0` |
| `8 = FuncRegionEffect` | `func_id` | Region `slot` (`0..63`), or `0xFFFF` (`REGION_FOOTPRINT_NONE`) | `pre` state (`0=Owned`, `1=Leased`, `2=Moved`) | `post` state (`0=Owned`, `1=Leased`, `2=Moved`) |
| `9 = ConstDataSeg` | Dense `seg_id` (`0..S`) | Byte offset in Section 5a | Byte length in Section 5a | `0` |
| `10 = ConstDataInBounds` | `seg_id` | `0` | `hi <= seg_len` (elides `ConstDataLoad` check) | `0` |
| `11 = BrTableSpec` | `default_depth` | Byte offset of `u32` target depths in Section 5a | `case_count` (`1..=4096`) | Dense `br_table_id` (`0..B`) |
| `12 = CallTable` | Dense `table_id` (`0..T`) | Byte offset of `u32` `func_id`s in Section 5a | `entry_count` (`1..=4096`) | `0` |

**`FuncRegionEffect` (`rule_kind = 8`) & `pre=leased` Semantics**:
- Each function `f` declares its touched region footprint `dom(f)` via one or more `FuncRegionEffect` entries (or a single empty-footprint sentinel `slot=none` / `0xFFFF`). Omitting all `FuncRegionEffect` entries for `f` defaults `dom(f)` to all declared regions as `Owned -> Owned` (backward-compatible with `v0.5`).
- At every `Call` and `CallIndirect` site, the verifier enforces the **frame rule**: `dom(callee) ⊆ dom(caller)`, and for every slot in `dom(callee)`, the caller's live typestate bitmasks must **match the callee's precondition exactly** (`(state.region_owned & dom) == callee.pre_owned && (state.region_leased & dom) == callee.pre_leased`).
- Specifically, declaring **`pre=leased, post=leased`** on slot `S` means:
  1. The caller **must** hold an active `RegionLease` on slot `S` at the call site (calling with `S` in `Owned` or `Moved` state is rejected with `RegionTypestateMismatch`).
  2. Inside the callee, slot `S` begins in `Leased` state, allowing read-only operations (`RegionLen`, `RegionLoad`) while statically forbidding mutation (`RegionStore`, `RegionGrow`, `RegionNew`, `RegionFree`, `RegionDrop`, `RegionMove`) or nested `RegionLease`.
  3. On return, slot `S` remains `Leased` in the caller until the caller executes `RegionEndLease`.
  4. Slots outside `dom(callee)` retain their caller typestate unchanged across the call (frame rule).
  5. Entry functions must require `Owned` (`pre_leased == 0`) on every slot in `dom(entry)`.

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

1. **`opcode (u8)` — 68-Opcode Instruction Set (30 columnar opcodes from `v0.4`, 30 `v0.5` TC Core opcodes across slices `1..3`, and 8 `v0.6` ownership, constant-data, and dispatch opcodes)**:
   - **Data, Constants & Locals (`0x01..=0x09`)**:
     - `0x01` `ArenaInput`: Load columnar slice from `ArenaDescriptor[in_a]`.
     - `0x02` `ConstF64`: 64-bit IEEE-754 floating-point constant (`bits = in_a | (in_b << 32)`).
     - `0x03` `ArenaSlice`: Zero-copy sub-slice of `in_a` certified by `proof_idx`.
     - `0x04` `ArenaMutate`: Linear in-place mutation of mutable `Arena[in_a]` with values from `in_b`, consuming the current `AffineToken` and advancing its version by `+1`.
     - `0x05` `ConstI64`: 64-bit integer constant (`bits = in_a | (in_b << 32)`).
     - `0x06` `LocalGet`: Read scalar local slot `in_a` (`0 <= in_a < 64`).
     - `0x07` `LocalSet`: Write scalar node `in_b` to local slot `in_a` (`0 <= in_a < 64`).
     - `0x08` `ConstDataLen`: Query byte length of declared `ConstDataSeg[in_a]` (`I64`, `in_b == 0`, `proof_idx == 0`).
     - `0x09` `ConstDataLoad`: Load scalar `U8` or little-endian `I64` at byte offset `in_b` from `ConstDataSeg[in_a]` (runtime bounds-checked unless `proof_idx -> ConstDataInBounds`).
   - **Arithmetic, Comparisons, Math, Reductions, Masks & Bitwise (`0x10..=0x2F`, `0x38..=0x39`)**:
     - `0x10` `Add`: Elementwise/scalar addition (`I64`/`U8` wrapping).
     - `0x11` `Sub`: Elementwise/scalar subtraction (`I64`/`U8` wrapping).
     - `0x12` `Mul`: Elementwise/scalar multiplication (`I64`/`U8` wrapping).
     - `0x13` `Div`: Elementwise IEEE-754 `F64` division, or checked scalar `I64` (signed) / `U8` (unsigned) division (traps with exit `72` on zero or `i64::MIN / -1`).
     - `0x14` `CmpGt`: Branchless greater-than predicate (`a > b`).
     - `0x15` `SelectZero`: Branchless predicated selection (`mask ? val : 0`).
     - `0x16` `Min`: Branchless elementwise minimum.
     - `0x17` `Max`: Branchless elementwise maximum.
     - `0x18` `CmpLt`: Branchless less-than predicate (`a < b`).
     - `0x19` `CmpEq`: Branchless equality predicate (`a == b`).
     - `0x1A` `Abs`: Branchless IEEE-754 absolute value.
     - `0x1B` `Sqrt`: Branchless IEEE-754 square root.
     - `0x1C` `Floor`: Branchless floor rounding toward $-\infty$.
     - `0x1D` `Ceil`: Branchless ceiling rounding toward $+\infty$.
     - `0x1E` `Ln`: Elementwise natural logarithm.
     - `0x1F` `Exp`: Elementwise natural exponential.
     - `0x20` `FilterCompact`: Stream compaction filtering elements of `in_b` where mask `in_a` is true.
     - `0x21` `ReduceSum`: Horizontal sum reduction of column `in_a` to scalar `f64`.
     - `0x22` `ReduceMax`: Horizontal maximum reduction of column `in_a` to scalar `f64`.
     - `0x23` `Gather`: Indirect indexed gather `target[indices[i]]` (requires `proof_idx > 0` and exact integer float indices `idx.fract() == 0.0`).
     - `0x24` `ReduceMin`: Horizontal minimum reduction of column `in_a` to scalar `f64`.
     - `0x25` `MaskAnd`: Branchless boolean mask conjunction.
     - `0x26` `MaskOr`: Branchless boolean mask disjunction.
     - `0x27` `Fma`: Single-rounded IEEE-754 fused multiply-add (`in_b` packs `(b: u16) | (c << 16)`).
     - `0x28` `MaskNot`: Scalar boolean mask negation (`!in_a`).
     - `0x29` `Cast`: Explicit scalar cast (`F64`/`I64`/`U8`/`BoolMask`; float-to-int out-of-range/NaN traps with exit `73`).
     - `0x2A` `BitAnd`: Scalar `I64`/`U8` bitwise AND.
     - `0x2B` `BitOr`: Scalar `I64`/`U8` bitwise OR.
     - `0x2C` `BitXor`: Scalar `I64`/`U8` bitwise XOR.
     - `0x2D` `Shl`: Scalar `I64`/`U8` shift left (`& 63` / `& 7`).
     - `0x2E` `ShrU`: Scalar `I64`/`U8` logical shift right (`& 63` / `& 7`).
     - `0x2F` `Rem`: Checked scalar `I64` (signed) / `U8` (unsigned) remainder (traps with exit `72`).
     - `0x38` `ShrS`: Scalar `I64` arithmetic shift right (`& 63`).
     - `0x39` `CmpLtU`: Scalar `I64` unsigned less-than comparison (`(a as u64) < (b as u64)`).
   - **Hardware I/O & Contract (`0x30`, `0x40`)**:
     - `0x30` `RingEnqueue`: Enqueue output to hardware ring buffer (requires `CAP_NET_RING`).
     - `0x40` `AssertContract`: Runtime contract assertion verifying finite, non-NaN scalar output.
   - **Structured Control Flow, Functions & Dispatch (`0x50..=0x5C`)**:
     - `0x50` `Block`: Structured forward-exit block header (optional `proof_idx -> BlockSig`).
     - `0x51` `Loop`: Structured backward-branch loop header (`proof_idx -> LoopBound`, or `0` with `CAP_DIVERGE`).
     - `0x52` `If`: Structured conditional header (`in_a = cond_node`; optional `proof_idx -> BlockSig`).
     - `0x53` `Else`: Alternate branch delimiter of enclosing `If` (`in_a = optional_then_result`, `in_b = has_result`).
     - `0x54` `End`: Terminator of innermost `Func`, `Block`, `Loop`, or `If` (`in_a = optional_result`, `in_b = has_result`).
     - `0x55` `Br`: Unconditional relative structured branch (`in_a = relative_depth`).
     - `0x56` `BrIf`: Conditional relative structured branch (`in_a = cond_node`, `in_b = relative_depth`).
     - `0x57` `Func`: Function scope header (`proof_idx == 0`, `in_a = func_id`, `in_b = 0`; function-relative operands, `<= 65,535` nodes per function).
     - `0x58` `Return`: Return from enclosing function (`in_a = rel_val_node` or `0` for `void`).
     - `0x59` `Call`: Direct function call (`in_a = target_func_id`, `in_b = (arg0: u16) | (arg1 << 16)` with `0xFFFF` sentinel).
     - `0x5A` `CallArg`: Extra call arguments `arg2`/`arg3` immediately preceding `Call` or `CallIndirect`.
     - `0x5B` `BrTable`: Dense multi-way structured switch (`in_a = selector_node` of `I64` or `U8`, `in_b = br_table_id`, `proof_idx == 0`; each `BrTableSpec` is single-use).
     - `0x5C` `CallIndirect`: Typed indirect function call (`in_a = (table_id: u16) | (idx_node << 16)`, `in_b = (arg0: u16) | (arg1 << 16)`, optional preceding `CallArg`; out-of-range table index traps with exit `76`).
   - **Statically Named Growable Regions & Typestate Ownership (`0x60..=0x69`)**:
     - `0x60` `RegionNew`: Reset declared region `in_a` to `in_b` zeroed elements (`Moved` or `Owned` -> `Owned`).
     - `0x61` `RegionGrow`: Append `in_b` zeroed elements to `Owned` region `in_a`; yields new length (`I64`).
     - `0x62` `RegionLen`: Query current element length of `Owned` or `Leased` region `in_a` (`I64`).
     - `0x63` `RegionLoad`: Load element at index `in_b` from `Owned` or `Leased` region `in_a` (runtime bounds-checked unless `proof_idx -> RegionIndexInBounds`).
     - `0x64` `RegionStore`: Store element into `Owned` region `in_a` (`in_b = (idx: u16) | (val << 16)`).
     - `0x65` `RegionFree`: Unmap `Owned` region `in_a` (`min_len == 0`) and set length to `0` (remains `Owned`).
     - `0x66` `RegionLease`: Begin zero-cost compile-time read lease on region `in_a` (`Owned -> Leased`).
     - `0x67` `RegionEndLease`: End zero-cost compile-time read lease on region `in_a` (`Leased -> Owned`).
     - `0x68` `RegionMove`: Move descriptor ownership from `Owned` `in_a` to `Moved` `in_b` (`src: Owned -> Moved`, `dst: Moved -> Owned`; requires matching `elem`, `Plain` role, `dst.min_len == 0`, `src.max_len <= dst.max_len`).
     - `0x69` `RegionDrop`: Unmap and mark `Owned` `Plain` (`min_len == 0`) region `in_a` as `Moved` (`Owned -> Moved`).
2. **`type_lane (u8)`**:
   - High 4 bits (`ElemType`): `0x1 = F64`, `0x2 = I64`, `0x3 = BoolMask`, `0x5 = U8`.
   - Low 4 bits (`SimdWidth`): `0x1 = Scalar`, `0x4 = Lane4 (256-bit AVX2)`, `0x8 = Lane8 (512-bit AVX-512 / 8-way unrolled)`.
3. **`schedule_hint (u8)`**:
   - High 4 bits (`fusion_group`): `0` = Unfused; `1..15` = Kernel Fusion Group ID.
   - Low 4 bits (`exec_policy`): `0 = AutoL1Tile`, `1 = SimdStream`, `2 = MultiCore`, `3 = GpuDma`.
4. **`token_op (u32)`**:
   - High 2 bits (`TokenMode`): `0b00` (`PureValue`), `0b01` (`ReadLease`), `0b10` (`AffineConsume`, advancing live version via `checked_add(1)` with `TokenVersionOverflow` protection on `u16::MAX`).

### 2.5. Section 5a (`const_data`) Layout & Validation Rules

When non-empty, Section 5a carries up to `MAX_CONST_DATA_BYTES = 8 MiB` (`8,388,608` bytes) of immutable read-only payload bytes referenced by `ProofRuleKind::ConstDataSeg`, `BrTableSpec`, and `CallTable`:

- **In-bounds, ascending, non-overlapping table rule**: Scanning `program.proofs` in proof-table order, every `ConstDataSeg`, `BrTableSpec`, and `CallTable` payload byte range `[lo, hi)` must satisfy `lo >= prev_end && hi <= const_data_bytes` (monotonically ascending and strictly non-overlapping across all three table kinds). Any violation fails verification with `VerifyErrorKind::InvalidConstDataTable` (`out of bounds, unordered, or overlaps a previous table`).
- **Payload encodings and dense IDs**:
  - `ConstDataSeg` (`rule_kind = 9`): `[bound_lo .. bound_lo + bound_hi)` stores raw immutable segment bytes (`witness_node == 0`; `seg_id` must form a dense `0..S` range with no duplicates or gaps).
  - `BrTableSpec` (`rule_kind = 11`): `[bound_lo .. bound_lo + 4 * case_count)` stores `case_count` (`1..=4096`) little-endian `u32` relative branch target depths (`br_table_id` in `witness_node` must form a dense `0..B` range; each table may be referenced by at most one `BrTable`).
  - `CallTable` (`rule_kind = 12`): `[bound_lo .. bound_lo + 4 * entry_count)` stores `entry_count` (`1..=4096`) little-endian `u32` `func_id`s (`witness_node == 0`; `table_id` must form a dense `0..T` range; every entry must name a declared function and all entries in a table must share an identical `FuncSig` and `FuncRegionContract`).

### 2.6. Section 5b (`link_meta`) Canonical Symbol Table (`RLMK` v1)

When non-empty, Section 5b encodes module exports and imports for separate verification (`verify_module`) and linking (`link_modules` / `rul link`):

| Offset | Field | Type | Description |
| :--- | :--- | :--- | :--- |
| `0..4` | `magic` | `[u8; 4]` | `b"RLMK"` |
| `4..6` | `version` | `u16` | `1` |
| `6..8` | `entry_count` | `u16` | Number of `LinkSymbol` records (`<= 4096`) |
| `8..` | `entries` | `[LinkSymbol]` | Packed `(direction: u8 [0=Export, 1=Import], kind: u8 [0=Func, 1=Region], local_id: u16, name_len: u16 [1..=128], name_ascii: [u8; name_len])` with charset `[A-Za-z0-9_.:/-]+` |

In `.rls`, these correspond losslessly to `export func <name> <fid>`, `import func <name> <fid>`, `export region <name> <slot>`, and `import region <name> <slot>`.

---

## 3. Single-Pass $O(N)$ Linear Verification, Ownership, Cost & Linking Rules

The verifier ([`verify_program`](../src/verifier.rs) / [`verify_module`](../src/verifier.rs)) scans `nodes[0..node_count]` exactly once in linear time:

1. **Topological SSA Rule**:
   - For every node at index $i$, all node-index operands (`in_a`, `in_b`, and packed `u16` operands) must satisfy `operand < i` (function-relative inside `Func`..`End` scopes).
2. **Linear Ownership Conservation (`AffineToken`, `ReadLease` & `ArenaVersionTrie` `O(N + A)`)**:
   - Each `Arena[a]` initializes with `live_version = initial_version`.
   - `ReadLease(a, ver)` requires `live_version[a] == ver` (multiple nodes may hold concurrent `ReadLease` tokens at the same live version).
   - `AffineConsume(a, ver)` requires `live_version[a] == ver` AND `Arena[a].mutable == true`, then updates `live_version[a] = live_version[a].checked_add(1)`. Any reuse of a stale or already-consumed version fails with `StaleOrConsumedToken`, and any increment past `u16::MAX` fails with `TokenVersionOverflow`.
   - Multi-arena `AffineToken` version vectors across `A` arenas (addressable via `u16`, so trie height $H = \lceil \log_2(\min(A, 65536)) \rceil \le 16$) are tracked in **`ArenaVersionTrie`** — a persistent, hash-consed canonical binary trie paired with a straight-line write-combining buffer (`pending`). Identical version vectors produce the exact same canonical `root: u32` ID regardless of mutation order or control nesting, so control-flow joins (`If`/`Else`/`End`), `Loop` invariant checks, and branch target checks compare `root: u32` in **`O(1)`** on the happy path (and pinpoint the lowest differing arena in $O(H) \le 16$ steps on mismatch), avoiding $O(N \cdot A)$ blowup and allocating zero heap memory when no `ArenaMutate` occurs.
3. **Statically Named, Typestate-Tracked Regions & Frame-Rule Composition**:
   - Each of the 64 declared region slots tracks its compile-time typestate (`Owned`, `Leased`, or `Moved`) in two 64-bit bitmasks (`region_owned: u64`, `region_leased: u64`, with `Moved` represented by `0, 0`).
   - Control-flow joins (`Else`/`End`, `Br`/`BrIf`/`BrTable`) join typestate masks by exact equality in `O(1)`; a `Loop` body must exit with the exact typestate it entered with.
   - `RegionLease`/`RegionEndLease` transition `Owned <-> Leased` at zero runtime cost; mutations and nested leases under a live lease fail closed with `MutationUnderLiveLease`; accessing a moved/dropped slot fails closed with `UseOfMovedRegion`.
   - Every `Call`/`CallIndirect` checks `FuncRegionEffect` frame-rule inclusion (`dom(callee) ⊆ dom(caller)`) and exact precondition matching (`RegionTypestateMismatch` on mismatch, including `pre=leased` semantics).
4. **Shape Unification, Origin-Bound Refinement Proofs & `zero_runtime_bounds_checks`**:
   - Elementwise binary ops (`Add`, `Sub`, `Mul`, `Div`, `Min`, `Max`, `CmpGt`, `CmpLt`, `CmpEq`, `SelectZero`, `MaskAnd`, `MaskOr`) require matching column shapes (`Column(N) == Column(N)`) or scalar broadcasting (`Scalar` with `Column(N)`), failing with `ShapeMismatch` otherwise.
   - `ArenaSlice` and `Gather` require a valid 1-based `proof_idx` whose `ProofEntry.target_arena` matches the input operand's `origin_arena` and satisfies `bound_lo <= bound_hi <= effective_target_len`, where `effective_target_len = n.min(target_arena.element_count)` (`n` is the column length of operand `in_a`). When `in_a` is an `ArenaSlice`, `bound_hi` is bounded by the sliced column length rather than only the base arena's `element_count`.
   - `RegionIndexInBounds` and `ConstDataInBounds` elide runtime bounds checks in `O(1)` for `ConstI64` and `BitAnd(_, ConstI64)` indices proven strictly below `bound_hi` (`bound_hi <= min_len` or `bound_hi <= seg_len`).
   - `VerifiedSummary.zero_runtime_bounds_checks` is `true` iff the program contains no dynamic `Gather`, no unproven `RegionLoad`/`RegionStore`, and no unproven `ConstDataLoad`.
5. **Structured Control Stack Rule**:
   - `Func`, `Block`, `Loop`, and `If` push a frame on an explicit control stack; `Else` must match an open `If`; `End` pops the innermost frame. Unmatched `Else`/`End` and frames left open at the last node are rejected.
   - `Br`, `BrIf`, and every target depth in `BrTable` (including `default_depth`) must name a `relative_depth` strictly smaller than the open control stack depth (`BrTable` cannot target a `Func` frame, a value-returning block, or a bounded/non-`CAP_DIVERGE` `Loop`, and each `BrTableSpec` is single-use). `If` and `BrIf` conditions must be scalar `BoolMask` nodes.
   - SSA nodes defined inside a scope are sealed at its `Else`/`End` and cannot be referenced afterwards (`ScopedValueEscaped`). State crosses a scope boundary only through typed locals or the scope's declared result, and `output_node` must be a non-void node visible at the end of the program (or the entry `Func` node in function programs).
   - A `BlockSig` proof declares the arena version transition and result type of a `Block`/`If` scope so both arms of an `If` join every `AffineToken` version and region typestate identically.
6. **Typed Local Rule**:
   - There are 64 scalar local slots (`0..63`; out-of-range indices fail with `LocalSlotOutOfRange`). Inside a `Func`, slots `0..arity` are pre-initialized with the function's declared parameter types.
   - `LocalSet` fixes a slot's `ElemType` on first write and every later write must use the same type (`LocalTypeMismatch` otherwise).
   - `LocalGet` requires the slot to be definitely initialized on every path that reaches it (`UninitializedLocalRead` otherwise).
7. **Zero-Trust Capability Policy, Termination & Sandbox `StaticCostSummary` (`rul check --json`)**:
   - `RingEnqueue` requires `(header.capability_mask & CAP_NET_RING) != 0` (`MissingCapability` otherwise).
   - When `(header.capability_mask & CAP_DIVERGE) == 0`, a `Loop` with `proof_idx == 0` is rejected with `MissingCapability`, every `Loop` must reference a valid `LoopBound` proof (`1..=1_000_000`), and `Call`/`CallIndirect` targets must have strictly lower `func_id`s (acyclic call graph; self/mutual recursion or forward calls are rejected with `MissingCapability`). In this sandbox mode, `verify_program` computes a deterministic `StaticCostSummary { worst_case_steps, max_call_depth, max_stack_units, max_stack_bytes, max_region_bytes }`.
   - When `CAP_DIVERGE` (`0x08`) is set, both unbounded (`proof_idx == 0`) and bounded (`proof_idx > 0`) loops and general recursion are admitted, and `static_cost` is `None`.
   - `rul check [--json] [file]` reports verification status, telemetry, and `static_cost` (or `null` under `CAP_DIVERGE`).
8. **Separate Verification (`verify_module`) & Verified Linking (`link_modules` / `rul link`)**:
   - `verify_module` validates partial library modules with imported functions/regions and returns `static_cost = None` when unresolved imports exist.
   - `verify_program`, `Tier 0`, and `ELF64` AOT fail closed with `UnresolvedImport` if invoked on an unlinked module.
   - `link_modules` / `rul link <mods...> -o <out.rl> [--entry <name>] [--cap <hex>]` verifies every input module, enforces **zero silent capability escalation** (`(module.capability_mask & !final_cap) == 0`), checks exact `FuncSig`, remapped `FuncRegionContract`, and `RegionDecl` agreement across exports/imports, relocates functions, regions, Section 5a tables (`ConstDataSeg`, `BrTableSpec`, `CallTable`), and deduplicated node-referenced proofs (`<= 255`), orders functions deterministically via min-heap Kahn topological sort (rejecting cycles without `CAP_DIVERGE` and placing the entry function last), strips Section 5b, and re-verifies the linked `.rl` executable via `verify_program`.

---

## 4. 3-Tier Execution Engine, Direct `ELF64` AOT, Stdlib & Exit Programs

1. **Tier 0 — Zero-Copy Reference Columnar & Structured Interpreter (`execute_tier0_interpreter`)**:
   - Evaluates columnar pipelines and general structured/function/region programs with explicit heap-stack frames (zero host stack recursion), serving as the Golden Oracle (`Tier 0 == ELF64 AOT`).
2. **Tier 1 & Tier 2 — Columnar Accelerators (`execute_tier1_stencil`, `execute_tier2_fused`, `execute_tier2_with_zonemap`)**:
   - Dedicated SIMD/L1-tile/MultiCore/`ArenaZoneMap` accelerators for columnar subgraphs; fail closed with `ExecError::UnsupportedInColumnarTier` on structured programs.
3. **Direct Linux `x86_64` `ELF64` AOT Compiler (`compile_rl_to_executable`, `compile_to_elf64_bytes`)**:
   - **Columnar path**: emits standalone statically-linked `x86_64` AVX2 `YMM` + SSE2 `ELF64` binaries in microseconds.
   - **Structured scalar/function path (`compile_structured_to_elf64_bytes`) & Register Contract**:
     - Per function, compilation runs one `O(N)` liveness pre-scan (loop-extended last uses, barrier prefix counts, loop-weighted local scores, scope result slots, and adjacent compare-branch fusion flags) followed by one forward linear-scan allocation and emission pass with forward `rel32` patching (single home per SSA value/local, no interval splitting, no IR, no inlining, and no interprocedural passes).
     - **Fixed non-allocatable registers**: `rsp` (private `8 * MAX_STACK_UNITS + 4096` `mmap` stack with bottom `PROT_NONE` guard page in function programs), `rbp` (frame pointer for unpinned locals, optional `x87` scratch, and spill slots; omitted in zero-spill functions), `r15` (4096-byte program-global descriptor page: `[0..2048)` = 64 x 32-byte region descriptors, `[2064..2072)` = scalar stdout/EOF scratch, `[2080..2136)` = callee-save spill area used only around `RegionNew`/`RegionGrow`/`RegionFree` syscalls), and `r14` (packed remaining stack budget `((MAX_CALL_DEPTH - live_call_depth) << 22) | (MAX_STACK_UNITS - live_stack_units)` updated via 32-bit immediate `(1 << 22) | func_frame_units(node_count)`, checking bit 63 for call-depth underflow and bit 21 for stack-unit underflow against a shared out-of-line `exit(75)` stub).
     - **Scratch registers**: `rax`, `r11` (GPR) and `xmm0`, `xmm1` (`F64`), clobberable within any single opcode emission.
     - **Allocatable pools**: 10 GPRs (`rdi`/`rsi`/`rdx`/`rcx` caller-saved matching incoming arguments `0..4`, `rbx`/`r12`/`r13`/`r8`/`r9`/`r10` callee-saved; intervals crossing `Call`/`CallIndirect`/`RegionNew`/`RegionGrow`/`RegionFree` take a callee-saved GPR or reused 8-byte spill slot; each `Func` saves only the callee-saved subset it assigns) and 14 caller-saved `XMM` registers (`xmm2..=xmm15` for `F64`; barrier-crossing `F64` values spill to an 8-byte frame slot).
     - **Frame layout & stack budget**: below `rbp`, the frame holds `[unpinned used locals] [optional 8B x87 scratch] [optional 8B stdout staging] [reused 8B spill slots]`. The native activation frame (including return address, saved callee registers, and `rbp`) is strictly `<= 8 * func_frame_units(node_count) = 8 * (72 + node_count)` bytes.
     - **Single-pass instruction selection & `v0.6` dispatch**: on-demand `ConstI64`/`ConstF64`/`Cast(ConstI64, U8)` materialization, fused integer `Cmp*` / `MaskNot(Cmp*)` branches, 32-bit zero-extending move for `BitAnd 0xFFFF_FFFF`, 3-operand `lea` for `I64` `Add`/`Sub`, dying-operand register reuse, in-place GPR-immediate ALU fast path, non-mutating early-return base-case `If` guards, a separate read-only `PF_R` `PT_LOAD` ELF segment for Section 5a (`const_data`), inline PC-relative `i32` displacement jump tables for `BrTable`, and PC-relative `CallTable` dispatch with bounds checking for `CallIndirect`.
     - **Current path limits**: no columnar arena operands, and at most `65,535` nodes per `Func` scope (`65,536` nodes for single-body structured programs without `Func`).
   - **Fail-closed exit codes (identical in `Tier 0` and `ELF64` AOT, emitting zero stdout on trap)**:
     - `70`: `AssertContract` failure or bounded-loop `LoopBound` overrun
     - `71`: Region or `ConstDataLoad` index out of bounds
     - `72`: Integer division/remainder by zero or `i64::MIN / -1`
     - `73`: Float-to-integer `Cast` out of range or `NaN`
     - `74`: Region allocation / growth failure (`> max_len` or `mmap`/`mremap` error)
     - `75`: Call-depth (`MAX_CALL_DEPTH = 4096`) or stack-budget (`MAX_STACK_UNITS = 1 << 20`) overflow (host-independent private `mmap` stack with bottom `PROT_NONE` guard page)
     - `76`: `CallIndirect` table index out of bounds (`< 0` or `>= entry_count`, `AOT_EXIT_INDIRECT_CALL_OOB`)
4. **Rul Standard Library (`stdlib/`, `cap 0x0`) & `v0.6` Linked Exit Programs (`perf/v0_6_exit/`)**:
   - Implemented strictly in Rul `.rls` with **zero new opcodes** (Principle 5: "Library Over ISA Bloat") over declared region bounds (`std.io.in <= 128 MiB`, `std.io.out <= 256 MiB` = `MAX_REGION_BYTES`):
     - [`stdlib/vec.rls`](../stdlib/vec.rls): region-backed multi-instance growable `I64` & `U8` vector module (descriptor table in `vec.meta` up to 1,023 instances, `vec.i64_data` up to `4,194,304` `i64`s / `32 MiB`, `vec.u8_data` up to `16,777,216` bytes / `16 MiB`, tagged-enum `BrTable` dispatch in `vec::clear`, bulk `vec::append_i64` / `vec::append_u8`).
     - [`stdlib/string.rls`](../stdlib/string.rls): bulk/span RFC 3629 `str::utf8_validate` (`const_data` byte-class table + `BrTable`), `str::eq_span`, 4x-unrolled `str::find_substr`, `str::parse_i64`, `str::fmt_i64`, and `str::append_input_span`, with `std.io.in` declared `Leased -> Leased` and chunked `256 x 999,999` static `LoopBound` loops covering the full `128 MiB` input region (including long digit runs and `> 1,000,000`-byte patterns).
     - [`stdlib/hashmap.rls`](../stdlib/hashmap.rls): power-of-two open-addressing byte-keyed `I64` hash map using 64-bit FNV-1a (`map::init`, `map::set_span`, `map::get_span`, `map::del_span`, `map::count`, `map::last_found`) over SoA columns (`map.kmeta` packing `[state:2 | koff:30 | klen:30]`, `map.hash`, `map.val`, `map.keys` + zero-copy `RegionDrop`/`RegionMove` scratch regions), `BrTable` state dispatch, key-arena compaction during rehash/churn, chunked hash/copy/compare/rehash/probe loops, up to `33,554,432` slots (`23.48M` live keys at `0.7` load, exceeding the theoretical `~14.92M` distinct-key ceiling of a `128 MiB` KV stream) and `128 MiB` compacted live key bytes, with `std.io.in` declared `Leased -> Leased`.
   - Three separately verified, `rul link`-assembled exit programs under [`perf/v0_6_exit/`](../perf/v0_6_exit/) (`json_parser` with shared `MAX_JSON_DEPTH = 128`, `kv_store`, `wc_grep`) match their safe-Rust `rustc -O` references byte-for-byte across `Tier 0 == ELF64 AOT == rustc -O` within these declared region bounds (exceeding `std.io.in = 128 MiB` or `std.io.out = 256 MiB` fails closed via ISA exit codes `71`/`74`), and pass the `<= 2.0x` release performance gate and `O(N)` verify/link/codegen scaling gate (see `CHANGELOG.md` for reference-host measurements).
5. **White-Box Execution Tracer (`trace_program` & `rul trace`)**:
   - Records deterministic per-node statistics (`min`, `max`, `sum/trues`, `nan_or_inf_count`, and sample head values) for every SSA node `#0..#N-1`.
