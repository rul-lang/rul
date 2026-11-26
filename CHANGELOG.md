# Changelog

All notable changes to the **Rul** (`.rl`) binary representation, verifier, 3-Tier execution engine, and `ELF64` AOT compiler are documented in this file.

## [0.6.0] — 2026-11-26

Milestone 2 (**Ownership Across Joins, Linking & Rul Standard Library**: `v0-6-ownership-1` through `v0-6-stdlib-exit-4`) — completing the 68-opcode ISA with zero new opcodes added for the standard library or exit programs.

### Added (`v0-6-stdlib-exit-4` — Region-Backed Stdlib, Three Linked Exit Programs & Performance Gate)
- **Region-backed Rul standard library modules (`stdlib/`, `cap 0x0`, zero new opcodes)**:
  - Declared I/O region bounds across stdlib and exit programs: `std.io.in <= 128 MiB` (`134,217,728` bytes), `std.io.out <= 256 MiB` (`268,435,456` bytes, `MAX_REGION_BYTES`); exceeding declared region bounds fails closed via ISA exit codes `71`/`74`.
  - `stdlib/vec.rls`: multi-instance growable `I64` & `U8` vector module using a descriptor table (`vec.meta`: up to 1,023 descriptors `[kind, offset, len, cap]`) and backing storage regions (`vec.i64_data` up to `4,194,304` `i64`s / `32 MiB`, `vec.u8_data` up to `16,777,216` bytes / `16 MiB`) supporting `>= 2` concurrent logical `Vec` instances, `BrTable` tagged-enum dispatch in `vec::clear`, exponential growth with frontier/relocation handling, and bulk span `vec::append_i64` / `vec::append_u8`.
  - `stdlib/string.rls`: bulk/span RFC 3629 `str::utf8_validate` (256-byte `const_data` byte-class table + `BrTable` validating 1..4-byte UTF-8 and rejecting overlong encodings, UTF-16 surrogates `U+D800..U+DFFF`, `> U+10FFFF`, and truncated sequences), `str::eq_span`, 4x-unrolled `str::find_substr`, `str::parse_i64`, `str::fmt_i64`, and `str::append_input_span`, declaring `FuncRegionEffect` `Leased -> Leased` on `std.io.in`, with chunked `256 x 999,999` static `LoopBound` loops covering the full `128 MiB` input region (including long digit strings and `> 1,000,000`-byte patterns).
  - `stdlib/hashmap.rls`: open-addressing power-of-two byte-keyed `I64` hash map using 64-bit FNV-1a (`map::init`, `map::set_span`, `map::get_span`, `map::del_span`, `map::count`, `map::last_found`) over SoA columns (`map.kmeta` packing `[state:2 | koff:30 | klen:30]`, `map.hash`, `map.val`, `map.keys` + zero-copy `RegionDrop`/`RegionMove` scratch regions), supporting up to `33,554,432` slots (`23.48M` live keys at `0.7` load, exceeding the `~14.92M` theoretical distinct-key ceiling of a `128 MiB` KV stream) and `128 MiB` compacted live key bytes, tombstone deletion dispatched via `BrTable`, live-key compaction during rehash and under dead-key churn, chunked hash/copy/compare/rehash/probe loops, and `Leased -> Leased` on `std.io.in`.
- **Three separately verified, `rul link`-assembled exit programs & safe-Rust references (`perf/v0_6_exit/`, `src/v06_stdlib_tests.rs`)**:
  - Within the declared `128 MiB` input / `256 MiB` output bounds (and shared `MAX_JSON_DEPTH = 128` for `json_parser`), all three linked exit programs match their safe-Rust `rustc -O` references byte-for-byte across `Tier 0 == ELF64 AOT == rustc -O`:
  - `json_parser.{rls,rs}` (`stdlib/string.rls` + `stdlib/vec.rls`): zero-GC RFC 8259 JSON parser with `BrTable` token dispatch, `str::utf8_validate` on strings under live `RegionLease`, and shared deterministic `MAX_JSON_DEPTH = 128`.
  - `kv_store.{rls,rs}` (`stdlib/hashmap.rls` + `stdlib/string.rls`): line-delimited batch KV store (`SET`/`GET`/`DEL`/`COUNT`) using 64-bit FNV-1a hashing in both Rul and safe `std`-only Rust, backed by a `1 MiB` flush-on-full scratch buffer into `std.io.out` (`<= 256 MiB`).
  - `wc_grep.{rls,rs}` (`stdlib/string.rls`): buffer-level `wc` (`lines`/`words`/`bytes`) + `grep` (`matches` + 63-bit FNV-1a matched-line digest) using `str::find_substr` under live `RegionLease`.
  - Added single-pass `ScRegAllocState::live_gpr_mask` tracking, in-place GPR-immediate ALU fast path, single-lookup node analysis/emission, and `fn_body_buf` pre-reservation in `src/aot.rs` (preserving `Tier 0 == ELF64 AOT` equivalence while keeping `stress_60k` codegen well under `50 ns/node` on an unthrottled host).
  - **Representative `v0.6` Performance Gate (`x86_64`, release build, min of 5 alternating single-threaded runs, `Tier 0 == ELF64 AOT == rustc -O`, `>= 100 ms` reference; exact timings vary with host CPU frequency/load)**:
    - `json_parser` (`100.80 MiB`, 900k records): `631.42 ms` vs `rustc -O` `373.15 ms` (**`1.69x`** `<= 2.0x`, `25,506 B` ELF64)
    - `kv_store` (`32.80 MiB`, 2.0M ops, FNV-1a): `631.99 ms` vs `rustc -O` `406.86 ms` (**`1.55x`** `<= 2.0x`, `25,110 B` ELF64)
    - `wc_grep` (`86.64 MiB`, 1.4M lines): `885.27 ms` vs `rustc -O` `530.86 ms` (**`1.67x`** `<= 2.0x`, `12,880 B` ELF64)
  - **Measured `v0.6` Linearity Gate (`x86_64`, release build, 2-module 6,000-node vs 60,000-node stress inputs & exit programs; load average ~9.4)**:
    - `stress_6k` (6,000 nodes): verify `230.13 us`, link `532.33 us`, AOT codegen `179.24 us` (`29.87 ns/node`)
    - `stress_60k` (60,000 nodes): verify `2.30 ms` (**`10.01x`** `<= 15.0x`), link `5.58 ms` (**`10.48x`** `<= 15.0x`), AOT codegen `1.87 ms` (**`31.17 ns/node`** `<= 50 ns/node`, **`10.43x`** `<= 15.0x`)
    - Linked exit programs: `json_parser` (3,477 nodes: `161.07 us` verify / `389.86 us` link / `219.08 us` codegen), `kv_store` (2,666 nodes: `128.49 us` / `313.55 us` / `181.46 us`), `wc_grep` (1,369 nodes: `63.78 us` / `156.52 us` / `79.64 us`).
- **Specification & Roadmap Finalization (`docs/rul_spec_v0.md`, `ROADMAP.md`, `CHANGELOG.md`)**:
  - Updated `docs/rul_spec_v0.md` to the full 68-opcode `v0.6` catalog, Section 5a/5b binary layout, `ProofRuleKind` `1..=12`, `ArenaVersionTrie` `O(N + A)` multi-arena joins, `FuncRegionEffect` `pre=leased` exact-match semantics, exit code `76`, `rul check --json`, `rul link`, and `stdlib/`.
  - Aligned `ROADMAP.md` terminology to "statically named, typestate-tracked regions" and recorded all Milestone 2 deliverables and measured benchmark ratios.

### Added (`v0-6-linker-3` — Section 5b `link_meta`, Separate Verification & Verified `rul link`)
- Added Section 5b (`link_meta`, canonical `RLMK` v1 symbol table, `RulHeader[44..48]`, max `2 MiB`), lossless `.rls` `export`/`import` directives (`func` and `region`), `verify_module` vs fail-closed `UnresolvedImport` in `verify_program`, and `link_modules` / `rul link` enforcing zero silent capability escalation, exact `FuncSig`/`FuncRegionContract`/`RegionDecl` checking, Section 5a + proof relocation with `<= 255` deduplicated node-proof ceiling, and deterministic min-heap Kahn topological ordering (`src/linker.rs`, `src/bin/rul.rs`, `src/v06_linker_tests.rs`, `tests/cli_link_v06.rs`).

### Added (`v0-6-data-dispatch-2` — 68-Opcode ISA: Section 5a `const_data`, Dense `BrTable` & Typed `CallIndirect`)
- Added Section 5a (`const_data`, `RulHeader[40..44]`, max `8 MiB`, `PF_R` ELF64 segment), domain-separated 64-bit FNV-1a hashing (preserving `v0.5.0` binary/hash compatibility when empty), and `RulCasStore` hash-collision rejection.
- Added `ConstDataLen = 0x08`, `ConstDataLoad = 0x09`, `BrTable = 0x5B` (inline PC-relative jump table), `CallIndirect = 0x5C` (fail-closed exit code `76` / `AOT_EXIT_INDIRECT_CALL_OOB` on out-of-range table index), and `ProofRuleKind` `9..=12` (`ConstDataSeg`, `ConstDataInBounds`, `BrTableSpec`, `CallTable`) (`src/v06_data_dispatch_tests.rs`).

### Added (`v0-6-ownership-1` — 64-Opcode ISA: `ArenaVersionTrie`, Region Typestate, Frame Rule & `StaticCostSummary`)
- Added `ArenaVersionTrie` (`O(N + A)` multi-arena `AffineToken` tracking with `O(1)` canonical root-ID joins across control-flow boundaries).
- Added statically named, typestate-tracked region ownership (`Owned`/`Leased`/`Moved` bitmasks), `RegionLease = 0x66`, `RegionEndLease = 0x67`, `RegionMove = 0x68`, `RegionDrop = 0x69`, and `ProofRuleKind::FuncRegionEffect = 8` (frame-rule composition with exact `pre=leased` semantics).
- Added `CAP_NONE` `StaticCostSummary` and `rul check --json` (`src/v06_ownership_tests.rs`, `tests/cli_check_v06.rs`).

---

## [0.5.0] — 2026-11-20

Milestone 1 (**Turing-Complete Core**: `v0-5-tc-core-1` through `v0-5-tc-core-4`) plus Phase P0 (Safety & Correctness), Phase P1 (AOT `YMM` Register Allocation), and Phase P1b (`Opcode::Fma` & AVX2 Vectorized `Ln`/`Exp`).

### Added (`v0-5-tc-core-4` — Single-Pass Structured Scalar Register Allocator & `v0.5` Performance Gate)
- **Single-pass linear-scan scalar register allocator (`src/aot.rs`, `docs/rul_spec_v0.md` §4)**:
  - Replaced the one-stack-slot-per-node structured `x86_64` `ELF64` emitter with one `O(N)` liveness pre-scan (16-byte compact `ScNodeInfo` records with loop-extended `last_use`, lazy barrier prefix counts, loop-depth-weighted local scores, scope-result reservation, and adjacent compare-branch fusion) plus one forward linear-scan register-allocating emission pass with forward `rel32` patching.
  - Allocated 10 GPRs (4 caller-saved `rdi`/`rsi`/`rdx`/`rcx` + 6 callee-saved `rbx`/`r12`/`r13`/`r8`/`r9`/`r10`) for `I64`/`U8`/`BoolMask`/loop fuel and 14 caller-saved `XMM` registers (`xmm2..=xmm15`) for `F64`, with reused 8-byte frame spill slots strictly bounded by `<= 8 * func_frame_units(node_count) = 8 * (72 + node_count)` bytes.
  - Added single-pass instruction selection: on-demand `ConstI64`/`ConstF64`/`Cast(ConstI64, U8)` materialization, fused integer `Cmp*` / `MaskNot(Cmp*)` branches, 32-bit `mov` lowering for `BitAnd 0xFFFF_FFFF`, 3-operand `lea` for `I64` `Add`/`Sub`, dying-operand two-operand register reuse, transient `rax` spill avoidance for immediately-consumed commutative ops, packed 32-bit-delta `r14` call-depth + stack-unit budget checks with a shared out-of-line `exit(75)` stub, zero-NOP function prologues, and non-mutating early-return base-case guards.
  - Kept the `.rl` binary format, opcode set, proof rules, `src/verifier.rs`, `src/engine.rs` (`Tier 0`), and the columnar AVX2 AOT path unchanged.
- **`v0.5` Exit Benchmarks, Rust References & Regression Gates (`perf/v0_5_exit/`, `src/tc_core4_tests.rs`)**:
  - Added standalone `.rls` programs and safe-Rust `std`-only references under `perf/v0_5_exit/` for `fib`, `quicksort`, `sieve`, `sha256`, and `brainfuck`.
  - Added `test_tc_core4_regalloc_differential` (520 deterministic fixed-seed structured programs including 120 high-pressure >16 live GPR + >16 live `F64` cases across loops, `If`/`Else`, and `Call`, verifying `Tier 0 == ELF64 AOT` and byte-identical deterministic `ELF64` output), `test_tc_core4_spilled_frames_stay_within_stack_budget` (>40 live `I64` values across recursive `Call` verifying exact `MAX_STACK_UNITS` boundary and exit `75` under `ulimit -s 64`), `test_tc_core4_performance_gate` (`#[ignore]`), and `test_tc_core4_codegen_time_is_linear` (`#[ignore]`).
  - **Measured `v0.5` Performance Gate (`x86_64`, release build, 5 alternating single-threaded runs, `ELF64 AOT == rustc -O`; `Tier 0 == ELF64 AOT` verified at `tc-core-3` sizes)**:
    - `fib(40)`: `378.47 ms` vs `rustc -O` `190.77 ms` (**`1.98x`** <= `2.0x`, `1,262 B` ELF64, `4.28 us` min codegen)
    - `quicksort` (2,000,000 `I64`s): `179.36 ms` vs `rustc -O` `122.80 ms` (**`1.46x`** <= `2.0x`, `2,084 B` ELF64, `8.54 us` min codegen)
    - `sieve` (`N = 50,000,000`): `408.08 ms` vs `rustc -O` `362.25 ms` (**`1.13x`** <= `2.0x`, `1,752 B` ELF64, `6.10 us` min codegen)
    - `sha256` (64 MiB): `396.71 ms` vs `rustc -O` `239.52 ms` (**`1.66x`** <= `2.0x`, `5,368 B` ELF64, `30.12 us` min codegen)
    - `brainfuck` (3x nested loop): `510.44 ms` vs `rustc -O` `326.14 ms` (**`1.57x`** <= `2.0x`, `2,964 B` ELF64, `10.02 us` min codegen)
  - **Measured Codegen Linearity (`AotCompileReport.codegen_ns`, min of 5 release compilations)**:
    - `stress_6k` (6,000 nodes, >=64 live): `189.64 us` (`31.61 ns/node` <= `50 ns/node`)
    - `stress_60k` (60,000 nodes, >=64 live): `1,932.44 us` (`32.21 ns/node` <= `50 ns/node`, **`10.19x`** scaling <= `15.0x`)

### Added (`v0-5-tc-core-3` — 60-Opcode ISA: Modular Function Table, `CAP_DIVERGE`-Gated Recursion, Host-Independent Stack Budget & Five `v0.5` Exit Programs)
- **Modular function scopes & `FuncSig` (`src/bytecode.rs`, `src/verifier.rs`)**:
  - Added `Func = 0x57`, `Return = 0x58`, `Call = 0x59`, `CallArg = 0x5A`, `ProofRuleKind::FuncSig = 5`, `FuncSigInfo`, `MAX_CALL_DEPTH = 4096`, `MAX_STACK_UNITS = 1 << 20`, `MAX_FUNC_NODES = 65_535`, `CALL_ARG_SENTINEL = 0xFFFF`, and `func_frame_units(node_count) = 72 + node_count`.
  - Indexed `FuncSig` entries by `func_id` in one linear proof pre-scan (supporting >255 functions and >65,536 total program nodes), enforced dense ascending declaration order `0..N`, `Func.proof_idx == 0`, function-relative node operands (`<= 65,535` nodes per function scope), per-activation 64 locals pre-initialized with `0..arity` parameters (`0..=4`), pure `token_op` enforcement in function programs, and entry point validation (`last Func`, `arity == 0`, non-`void` return, `header.output_node == entry Func`).
  - Gated recursive and forward `Call` edges (`target_func_id >= caller_func_id`) behind `CAP_DIVERGE = 0x08` (`VerifyErrorKind::MissingCapability`), admitting self and mutual recursion under `CAP_DIVERGE`.
- **Host-independent fail-closed execution (`src/engine.rs`, `src/aot.rs`)**:
  - Implemented explicit heap-stack `Tier 0` function interpreter (`eval_functions_tier0`) with zero host stack recursion, enforcing `MAX_CALL_DEPTH` and `MAX_STACK_UNITS` (`ExecError::CallDepthExceeded`).
  - Extended single-pass forward-patch `ELF64` AOT emitter (`emit_x86_64_scalar_control_flow`) to map a private stack (`8 * MAX_STACK_UNITS + 4096` bytes with bottom `PROT_NONE` guard page), enforce identical `live_call_depth` and `live_stack_units` counters via `[r15 + 2048/2056]` (`AOT_EXIT_STACK_OVERFLOW = 75` / `AotError::StackOverflow`, never `SIGSEGV`, invariant under `ulimit -s 64`), and patch forward/recursive `call rel32` targets in one pass.
- **`.rls` assembler/decompiler (`src/transpiler.rs`)**: Lossless `.rls` $\leftrightarrow$ `.rl` round-trip for `Func`, `Return`, `Call`, `CallArg`, `func_sig` proof lines, and per-function symbolic-to-relative operand translation.
- **`v0.5` Exit Programs Suite (`src/tc_core3_tests.rs`)**: Verified `Tier 0 == ELF64 AOT` bit-for-bit and lossless `.rls` round-trip for all five `v0.5` exit programs: **Brainf\*ck interpreter** (Turing-completeness proof with nested `[...]`, `<>+-.,`, `!` input separator, 30,000-byte `U8` tape), recursive `fib(20)`, in-place `quicksort`, prime `sieve(N = 10_000)`, and FIPS 180-4 `sha256`.

### Added (`v0-5-tc-core-2` — 56-Opcode ISA: `U8`, `I64`/`U8` Bitwise & Checked Div/Rem, Declared Growable Regions, Byte I/O Regions & Proof-Based Bounds Elision)
- **Unsigned byte type & scalar integer/cast/mask ops (`src/bytecode.rs`, `src/verifier.rs`, `src/engine.rs`, `src/aot.rs`)**:
  - Added `ElemType::U8 = 0x5` (`u8x1` in `.rls`), `MaskNot = 0x28`, `Cast = 0x29`, `BitAnd = 0x2A`, `BitOr = 0x2B`, `BitXor = 0x2C`, `Shl = 0x2D`, `ShrU = 0x2E`, `Rem = 0x2F`, `ShrS = 0x38`, and `CmpLtU = 0x39`.
  - Extended `Add`/`Sub`/`Mul`/`Div`/`CmpGt`/`CmpLt`/`CmpEq`/`SelectZero` to scalar `U8` (wrapping modulo 256, unsigned compare/div/rem, shifts masked `& 7`) and `Div`/`Rem` to checked signed `I64` (shifts masked `& 63`).
  - Fail-closed equivalence across `Tier 0` and `ELF64` AOT on integer division/remainder by zero or `i64::MIN / -1` (`ExecError::IntegerDivisionError` / `AOT_EXIT_INT_DIV_ERR = 72`) and on `NaN` or out-of-range float-to-integer `Cast` (`ExecError::CastOutOfRange` / `AOT_EXIT_CAST_ERR = 73`).
  - Added `TypedScalar` / `RuntimeValue::TypedScalar` in `Tier 0` so non-`F64` scalar outputs report `(ElemType, bits)` directly instead of smuggling through `f64::from_bits`.
- **Statically declared growable regions & byte I/O (`ProofRuleKind::RegionDecl = 6`, `RegionIndexInBounds = 7`)**:
  - Added `RegionNew = 0x60`, `RegionGrow = 0x61`, `RegionLen = 0x62`, `RegionLoad = 0x63`, `RegionStore = 0x64`, and `RegionFree = 0x65` over `U8`, `I64`, and `F64` across up to 64 program-global region slots (`MAX_REGION_BYTES = 256 MiB`).
  - Verified declarations in one linear pre-scan of the proof table and checked every region opcode in `O(1)` (`tok=pure`), enforcing `min_len <= len <= max_len` (`ExecError::RegionAllocationFailed` / `AOT_EXIT_REGION_OOM = 74`), zero-initialization on startup/`RegionNew`/`RegionGrow`, and unsigned runtime bounds checks (`ExecError::RegionBoundsOutOfRange` / `AOT_EXIT_GATHER_OOB = 71`).
  - Added `RegionIndexInBounds` proof-based bounds-check elision for `ConstI64` and `BitAnd(_, ConstI64)` indices strictly below `bound_hi <= min_len`, preserving `VerifiedSummary.zero_runtime_bounds_checks = true` when all accesses are proven.
  - Added declarative byte input (`u8:input`, pre-filled from stdin / caller slice) and output (`u8:output`, flushed to stdout / returned bytes with zero stdout on fail-closed exit) regions, plus `SYS_mmap`/`SYS_mremap`/`SYS_munmap`/`SYS_read`/`SYS_write` lowering in `src/aot.rs` and stdin/stdout forwarding in `rul run`.
- **`.rls` assembler/decompiler (`src/transpiler.rs`)**: Lossless round-trip for `u8x1`, all 16 new opcodes, `region_decl` (`<elem>:<role>`), and `region_index_in_bounds`.

### Added (`v0-5-tc-core-1` — 40-Opcode ISA: Structured Control Flow, Typed Scalar Locals & `CAP_DIVERGE`)
- **Termination capability `CAP_DIVERGE = 0x08` (`src/bytecode.rs`, `src/verifier.rs`)**:
  - Termination is now a capability policy. Without `CAP_DIVERGE`, every `Loop` must carry a `ProofRuleKind::LoopBound = 4` witness (`1 <= bound_hi <= 1_000_000` back-edge iterations) or the verifier rejects the program with `VerifyErrorKind::MissingCapability`. With `CAP_DIVERGE`, unbounded loops (`proof_idx == 0`) are admitted.
- **Structured control flow (`Block = 0x50`, `Loop = 0x51`, `If = 0x52`, `Else = 0x53`, `End = 0x54`, `Br = 0x55`, `BrIf = 0x56`)**:
  - Verified in the same single forward `O(N)` pass with an explicit control stack: block matching, relative branch depths, scalar `BoolMask` conditions, sealing of inner SSA temporaries at `Else`/`End`, and `AffineToken` conservation across joins through `ProofRuleKind::BlockSig = 3`.
- **64-bit integer constants and typed scalar locals (`ConstI64 = 0x05`, `LocalGet = 0x06`, `LocalSet = 0x07`)**:
  - 64 local slots with definite-initialization and fixed-type checks (`VerifyErrorKind::UninitializedLocalRead`, `VerifyErrorKind::LocalTypeMismatch`); scalar `I64` `Add`/`Sub`/`Mul` wrap.
- **Execution (`src/engine.rs`, `src/aot.rs`)**:
  - Structured programs run in `Tier 0` and compile to standalone `x86_64` `ELF64` binaries through a dedicated single-pass scalar code generator (`compile_structured_to_elf64_bytes`) with `Tier 0 == AOT` results. A bounded loop that exceeds its `LoopBound` fails closed (`ExecError::LoopBoundExceeded` in `Tier 0`, exit code `70` in AOT).
  - `Tier 1` and `Tier 2` fail closed on structured programs with `ExecError::UnsupportedInColumnarTier`.
- **`.rls` assembler/decompiler (`src/transpiler.rs`)**: Lossless round-trip for all ten new opcodes and the `block_sig` / `loop_bound` proof rules.

### Changed (Roadmap & Documentation)
- `ROADMAP.md` now states host-independent recursion limits, function signatures resolved by function id with function-relative operands, statically declared regions, static resource bounds in sandbox mode, a scalar register-allocation slice with a performance gate, the `v0.6` ownership/linking/data-section work, the `v0.7` columnar bridge, and a binary format change schedule.
- `README.md`, `docs/about_rul.md`, `docs/new_language_for_AI.md`, and `docs/rul_spec_v0.md` are aligned with the six-milestone roadmap and the 40-opcode ISA.

### Added (Phase P1b — 30-Opcode ISA: `Opcode::Fma` & Vectorized 256-Bit AVX2 `Ln`/`Exp`)
- **Explicit `Opcode::Fma = 0x27` (`src/bytecode.rs`, `src/verifier.rs`, `src/transpiler.rs`, `src/engine.rs`, `src/aot.rs`)**:
  - Added single-rounded IEEE-754 fused multiply-add (`out[i] = fma(a[i], b[i], c[i]) = a[i] * b[i] + c[i]`) as the 30th opcode in the 128-bit `NodeFrame` ISA, packing ternary operands `(b: u16, c: u16)` into `in_b: u32` (`NodeFrame::encode_fma_bc` / `NodeFrame::decode_fma_bc`) while preserving the fixed 16-byte frame layout.
  - Added `.rls` $\leftrightarrow$ `.rl` assembly/decompilation/disassembly support (`node $dst = Fma f64x8 a=$a b=$b c=$c`, `SymbolicGraphBuilder::emit_fma_op`).
  - Lowered `Opcode::Fma` in `src/aot.rs` to 256-bit AVX2 FMA3 (`vfmadd213pd` / `vfmadd231pd`) in the `5a` vector loop and 64-bit FMA3 (`vfmadd213sd`) in the `5b` scalar tail and `8` post-reduction scalar epilogue.
  - **Measured Speedup (`BM-HORNER-50`, 50-step Horner polynomial, 10,000,000 `f64` elements = 80 MB SoA)**:
    - Explicit `Opcode::Fma` cuts the graph from 104 SSA nodes (`Mul+Add`) to 54 SSA nodes (`Fma`), shrinking the standalone `ELF64` binary from **3,872 B** to **2,877 B** (`2,757 B` machine code, compiled in **`10.07 us`**).
    - Execution time on 10,000,000 elements drops to **40.34 ms median (`34.74 ms` min, `1.98 GB/s`)** — **`3.33x` faster** than `[A] Hand-Written C/Rust 8-Acc Horner-50` (**134.24 ms**), **`2.70x` faster** than `Mul+Add` AOT (**108.75 ms**), and **`9.81x` faster** than the pre-P1 stack-slot AOT baseline (**396 ms**).
- **Vectorized 256-Bit AVX2 `Opcode::Ln` and `Opcode::Exp` (`src/aot.rs`)**:
  - Replaced the 4-way scalar `x87` (`fyl2x` / `f2xm1`) calls inside the 256-bit AVX2 loop (`5a`) with branchless 4-lane AVX2 + FMA3 polynomial evaluation (`emit_avx2_ln_ymm` using `2^54` subnormal normalization, `fdlibm` rational reduction `s = (m - 1) / (m + 1)` + `Lg1..Lg7`, and Cody-Waite `LN2_HI`/`LN2_LO`; `emit_avx2_exp_ymm` using Cody-Waite range reduction, split `2^k1 * 2^k2` scaling for `k = 1024` and gradual subnormal underflow, and degree-12 FMA3 Horner evaluation) with full IEEE-754 special-case masking (`Ln`: `0 -> -Inf`, `< 0 -> NaN`, `+Inf -> +Inf`, `NaN -> NaN`; `Exp`: `< -745.1332 -> 0.0`, `> 709.7827 -> +Inf`, `NaN -> NaN`).
  - Math constant slots (`38 x 32 B = 1,216 B`) are allocated and pre-broadcast only when `needs_vec_math` is true (`has_live_ln || has_live_exp`), incurring zero frame or code size overhead for non-transcendental programs.
  - **Measured Speedup (`BM-TRANSCENDENTAL-LN-EXP`, `sum(exp(ln(x) * 0.5))` on 1,000,000 `f64` elements = 8 MB SoA)**:
    - Standalone `ELF64` binary (`2,962 B`, `2,842 B` machine code) executes in **4.44..8.17 ms median across runs (`1.22x..2.26x` faster than `Rust/glibc` scalar `x.ln() + exp()` at `6.95..12.87 ms`, including standalone `ELF64` subprocess spawn and `mmap` overhead)** while matching `libm` output to `< 1e-13` relative error (`2.202161542e6`) across normal and subnormal domains.

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
