//! Single-pass $O(N)$ linear verifier for Rul (`.rl`) binary graphs.
//!
//! Instead of running an expensive type/borrow inference solver, [`verify_program`]
//! scans the flat [`NodeFrame`] array in a single forward loop to verify:
//! 1. Topological SSA-DAG ordering (`in_idx < current_idx`).
//! 2. `AffineToken` and `ReadLease` ownership conservation (0% Use-After-Consume or aliasing).
//! 3. `Bounds-Safe by Construction` shape propagation and sidecar [`ProofEntry`] certificates
//!    with strict `origin_arena` binding (preventing cross-arena proof spoofing).
//! 4. Zero-trust hardware [`CAP_NET_RING`] capability sandboxing.

use std::collections::HashMap;
use std::fmt;

use crate::bytecode::{
    func_frame_units, ElemType, FuncSigInfo, NodeFrame, Opcode, ProofEntry, ProofRuleKind,
    RegionRole, RegionState, RulProgram, SimdWidth, TokenMode, TypeLane, CALL_ARG_SENTINEL,
    CAP_DIVERGE, CAP_NET_RING, MAX_CONST_DATA_BYTES, MAX_FUNC_NODES, MAX_LINK_META_BYTES,
    MAX_REGION_BYTES, MAX_REGION_SLOTS, MAX_SECTION_ITEMS, MAX_TABLE_ENTRIES,
};

/// Shape and static length inferred in $O(1)$ per node during the single-pass scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueShape {
    /// A single scalar value (`ConstF64`, `ReduceSum`, `ReduceMax`).
    Scalar,
    /// A fixed-length columnar vector (`ArenaInput`, `ArenaSlice`, `Add`, `Mul`, etc.).
    Column(u32),
    /// A dynamically compacted stream produced by `FilterCompact` (`len <= max_len`).
    Compacted {
        /// Upper bound on element count.
        max_len: u32,
        /// Mask node index that produced this compaction.
        mask_node: u32,
    },
}

/// Sound static worst-case resource bounds computed in a single $O(N)$ verification pass
/// for sandbox programs without [`CAP_DIVERGE`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaticCostSummary {
    /// Sound saturating upper bound on dynamic execution steps (scalar ops = `1`, columnar ops = `len`).
    pub worst_case_steps: u64,
    /// Maximum live call depth (`1` for top-level/leaf programs, `1 + max(callee.max_call_depth)` across calls).
    pub max_call_depth: u32,
    /// Maximum sum of ISA frame units (`func_frame_units(f)`) along any call chain from entry (`0` for non-function programs).
    pub max_stack_units: u32,
    /// Maximum stack allocation in bytes (`max_stack_units * 8 + 4096` guard page when functions exist, else `0`).
    pub max_stack_bytes: u64,
    /// Maximum declared region bytes (page-rounded + 4 KiB descriptor page if any region) plus columnar arena bytes.
    pub max_region_bytes: u64,
}

/// Summary returned when a `.rl` program passes $O(N)$ verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedSummary {
    /// Number of 128-bit instruction frames verified.
    pub node_count: usize,
    /// Number of distinct kernel fusion groups (`ScheduleHint.fusion_group > 0`).
    pub fusion_groups: usize,
    /// Shape of the program's primary output node.
    pub output_shape: ValueShape,
    /// Element type (`F64`, `I64`, or `BoolMask`) of the program's primary output node.
    pub output_elem: ElemType,
    /// True if all array accesses are proven bounds-safe at compile time (0 runtime checks).
    pub zero_runtime_bounds_checks: bool,
    /// Static worst-case resource summary for sandbox programs without [`CAP_DIVERGE`], or `None` when [`CAP_DIVERGE`] is set.
    pub static_cost: Option<StaticCostSummary>,
}

/// Machine-readable diagnostic returned to an AI Agent on verification failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyDiagnostic {
    /// Index of the offending 128-bit [`NodeFrame`].
    pub node_idx: u32,
    /// Specific verification failure kind.
    pub kind: VerifyErrorKind,
}

/// Specific violation detected by the $O(N)$ linear verifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyErrorKind {
    /// Program contains zero instruction nodes or `output_node` is out of range.
    InvalidOutputNode(u32),
    /// Input operand refers to a node index `>= current_node_idx` (cycle or forward ref).
    NonTopologicalEdge {
        /// Referenced input node index.
        input_node: u32,
    },
    /// Referenced arena ID does not exist in `ArenaDescriptor` table.
    UnknownArena(u32),
    /// Token version does not match current arena version (Use-After-Consume or stale lease).
    StaleOrConsumedToken {
        /// Target arena ID.
        arena_id: u16,
        /// Token version presented by the instruction.
        provided_version: u16,
        /// Current live token version required by the verifier.
        expected_version: u16,
    },
    /// Attempted `AffineConsume` or `ArenaMutate` on an immutable input arena.
    ImmutableArenaMutation(u16),
    /// `AffineConsume` token version reached `u16::MAX` (65,535) and cannot be incremented without overflow.
    TokenVersionOverflow(u16),
    /// Instruction expected a `ReadLease` or `AffineConsume` token for the target arena.
    MissingArenaToken(u16),
    /// Incompatible `ValueShape` between binary operands (`in_a` vs `in_b`).
    ShapeMismatch {
        /// Shape of `in_a`.
        left: ValueShape,
        /// Shape of `in_b`.
        right: ValueShape,
    },
    /// Incompatible element type (`ElemType`) for the operation.
    ElementTypeMismatch {
        /// Expected element type.
        expected: ElemType,
        /// Actual element type found.
        actual: ElemType,
    },
    /// Indirect access (`Gather` or `ArenaSlice`) missing a valid `proof_idx`.
    MissingRefinementProof,
    /// Sidecar `ProofEntry` failed bounds verification against target arena length.
    InvalidRefinementProof {
        /// 1-based proof index.
        proof_idx: u8,
        /// Upper bound claimed by the proof.
        bound_hi: u32,
        /// Actual length of the target arena.
        arena_len: u32,
    },
    /// `ProofEntry.target_arena` does not match the actual origin arena of the operand.
    ProofTargetArenaMismatch {
        /// Arena ID claimed inside `ProofEntry`.
        proof_target: u16,
        /// Actual origin arena ID of the input node.
        actual_arena: Option<u16>,
    },
    /// `ProofEntry.witness_node` for `Gather` does not match the index operand node (`in_b`).
    ProofWitnessMismatch {
        /// Witness node recorded in `ProofEntry`.
        proof_witness: u32,
        /// Actual index node passed in `in_b`.
        index_node: u32,
    },
    /// Instruction requires hardware capability bits not granted in `RulHeader.capability_mask`.
    CapabilityViolation {
        /// Required capability bitmask.
        required: u64,
        /// Granted capability bitmask in header.
        granted: u64,
    },
    /// Structured `Loop` without `LoopBound` proof requires `CAP_DIVERGE` in `capability_mask`.
    MissingCapability {
        /// Required capability bitmask (`CAP_DIVERGE`).
        required: u64,
        /// Actual capability bitmask declared in header.
        actual: u64,
    },
    /// Typed local slot index exceeds the 64-slot register file (`0..64`).
    LocalSlotOutOfRange {
        /// Offending local slot index.
        slot: u32,
    },
    /// `LocalGet` attempted to read a slot that is not definitely initialized on all incoming paths.
    UninitializedLocalRead {
        /// Offending local slot index.
        slot: u32,
    },
    /// `LocalSet` attempted to write a different `ElemType` into an already-typed local slot.
    LocalTypeMismatch {
        /// Target local slot index.
        slot: u32,
        /// Previously established `ElemType` for this slot.
        expected: ElemType,
        /// Conflicting `ElemType` provided by this `LocalSet`.
        actual: ElemType,
    },
    /// Operand references an SSA node sealed inside a closed inner structured block or sibling branch.
    ScopedValueEscaped {
        /// Sealed inner SSA node index.
        input_node: u32,
    },
    /// Malformed structured control-flow construct (`Block`/`Loop`/`If`/`Else`/`End`/`Br`/`BrIf`).
    InvalidControlFlow(&'static str),
    /// Invalid [`ProofRuleKind::RegionDecl`] entry in the proof table.
    InvalidRegionDecl {
        /// Declared region slot (`0..64`).
        slot: u16,
        /// Reason the declaration was rejected.
        reason: &'static str,
    },
    /// Region opcode referenced an undeclared or out-of-range region slot.
    UnknownRegion(u32),
    /// Invalid [`ProofRuleKind::FuncSig`] entry or function table declaration.
    InvalidFuncSig {
        /// Offending function identifier.
        func_id: u32,
        /// Reason the declaration was rejected.
        reason: &'static str,
    },
    /// `Call` or `Func` referenced an undeclared function identifier.
    UnknownFunction(u32),
    /// Function scope (`Func` through matching `End`) exceeds [`MAX_FUNC_NODES`] (65,535).
    FunctionTooLarge {
        /// Offending function identifier.
        func_id: u16,
        /// Observed node count inside the function scope.
        node_count: usize,
    },
    /// Attempted a region mutation or nested lease while a compile-time read lease is live.
    MutationUnderLiveLease {
        /// Offending declared region slot (`0..64`).
        slot: u16,
    },
    /// Attempted to access, lease, drop, or move from a region slot currently in the `Moved` state.
    UseOfMovedRegion {
        /// Offending declared region slot (`0..64`).
        slot: u16,
    },
    /// Function accessed a region slot outside its declared footprint `dom(f)`.
    RegionOutsideFootprint {
        /// Offending declared region slot (`0..64`).
        slot: u16,
    },
    /// Callee function footprint `dom(g)` is not a subset of caller footprint `dom(f)`.
    CalleeFootprintExceedsCaller {
        /// Target callee function identifier.
        callee_func_id: u16,
        /// Bitmask of slots in `dom(g) \ dom(f)`.
        extra_mask: u64,
    },
    /// Region `(owned_mask, leased_mask)` typestate mismatched at a join, call precondition, function return, or `RegionEndLease`/`RegionMove`.
    RegionTypestateMismatch {
        /// Expected `owned_mask`.
        expected_owned: u64,
        /// Expected `leased_mask`.
        expected_leased: u64,
        /// Observed `owned_mask`.
        actual_owned: u64,
        /// Observed `leased_mask`.
        actual_leased: u64,
    },
    /// Loop body fall-through or back-edge violated the loop entry `(owned_mask, leased_mask)` typestate invariant.
    LoopTypestateInvariantViolation {
        /// Loop entry `owned_mask`.
        expected_owned: u64,
        /// Loop entry `leased_mask`.
        expected_leased: u64,
        /// Observed `owned_mask`.
        actual_owned: u64,
        /// Observed `leased_mask`.
        actual_leased: u64,
    },
    /// Invalid [`ProofRuleKind::FuncRegionEffect`] sidecar proof entry or function footprint contract.
    InvalidRegionEffect {
        /// Offending function identifier.
        func_id: u16,
        /// Reason the effect declaration was rejected.
        reason: &'static str,
    },
    /// Invalid [`ProofRuleKind::ConstDataSeg`], [`ProofRuleKind::BrTableSpec`], or [`ProofRuleKind::CallTable`] metadata or `const_data` payload layout.
    InvalidConstDataTable {
        /// Reason the constant-data segment or dispatch table was rejected.
        reason: &'static str,
    },
    /// `ConstDataLen` or `ConstDataLoad` referenced an undeclared constant-data segment identifier.
    UnknownConstDataSeg(u32),
    /// `BrTable` referenced an undeclared `br_table_id`.
    UnknownBrTable(u32),
    /// A `BrTableSpec` table identifier was referenced by more than one `BrTable` instruction.
    DuplicateBrTableUse {
        /// Reused `br_table_id`.
        br_table_id: u32,
    },
    /// `CallIndirect` referenced an undeclared `CallTable` identifier.
    UnknownCallTable(u32),
}

impl fmt::Display for VerifyDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "verify error at node #{}: {:?}",
            self.node_idx, self.kind
        )
    }
}

impl std::error::Error for VerifyDiagnostic {}

/// Per-node static metadata tracked during the $O(N)$ verification scan.
#[derive(Debug, Clone, Copy)]
struct NodeMeta {
    shape: ValueShape,
    elem: ElemType,
    origin_arena: Option<u16>,
    is_void: bool,
    const_i64: Option<i64>,
    nonneg_upper_bound: Option<u64>,
}

impl NodeMeta {
    const VOID: Self = Self {
        shape: ValueShape::Scalar,
        elem: ElemType::I64,
        origin_arena: None,
        is_void: true,
        const_i64: None,
        nonneg_upper_bound: None,
    };

    const fn value(shape: ValueShape, elem: ElemType, origin_arena: Option<u16>) -> Self {
        Self {
            shape,
            elem,
            origin_arena,
            is_void: false,
            const_i64: None,
            nonneg_upper_bound: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RegionDeclInfo {
    slot: u16,
    min_len: u32,
    max_len: u32,
    elem: ElemType,
    role: RegionRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ControlKind {
    Block,
    Loop,
    If { has_else: bool },
    Func { sig: FuncSigInfo, start_node: u32 },
}

#[derive(Debug, Clone, Copy)]
struct ActiveFunc {
    sig: FuncSigInfo,
    start_node: u32,
}

#[derive(Debug, Clone, Copy)]
struct PendingCallArg {
    node_idx: u32,
    arg2_elem: ElemType,
    arg3_elem: Option<ElemType>,
}

#[derive(Debug, Clone, Copy)]
struct CompletedFunc {
    sig: FuncSigInfo,
    start_node: u32,
    worst_case_steps: u64,
    max_call_depth: u32,
    max_stack_units: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FuncRegionContract {
    dom_mask: u64,
    pre_owned: u64,
    pre_leased: u64,
    post_owned: u64,
    post_leased: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ConstDataSegInfo {
    seg_id: u16,
    offset: u32,
    len: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BrTableInfo {
    br_table_id: u32,
    default_depth: u16,
    offset: u32,
    case_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CallTableSummary {
    table_id: u16,
    rep_func_id: u16,
    sig: FuncSigInfo,
    contract: FuncRegionContract,
    max_func_id: u32,
    offset: u32,
    entry_count: u32,
    cost_summary: Option<(u64, u32, u32)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ArenaDelta {
    arena_id: u16,
    entry_version: u16,
    exit_version: u16,
}

#[derive(Debug, Clone, Copy)]
struct ParsedBlockSig {
    arena_delta: Option<ArenaDelta>,
    result_type: Option<TypeLane>,
}

#[derive(Debug)]
struct ControlFrame {
    kind: ControlKind,
    header_node: u32,
    sig: Option<ParsedBlockSig>,
    entry_init: u64,
    entry_arena_root: u32,
    entry_reachable: bool,
    entry_region_owned: u64,
    entry_region_leased: u64,
    has_branch_or_join: bool,
    has_exit_path: bool,
    exit_init: u64,
    exit_region_owned: u64,
    exit_region_leased: u64,
    exit_arena_root: u32,
    parent_steps: u64,
    then_steps: u64,
    loop_bound: u32,
    scope_nodes: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
struct ExitSnapshot {
    local_init: u64,
    region_owned: u64,
    region_leased: u64,
}

/// Hash-consed persistent binary trie for exact, order-independent `AffineToken` version states.
///
/// Guarantees across any program with `N` nodes and `A` arenas (where addressable `arena_id: u16`,
/// so trie height `H = ceil(log2(min(A, 65536))) <= 16`):
/// - Zero heap allocation when no `ArenaMutate` occurs.
/// - Canonical state IDs: two arena version vectors are identical across all `A` arenas if and
///   only if their `root: u32` IDs are equal, regardless of mutation order or control nesting.
/// - Straight-line write-combining (`pending`) + batch trie updates bound total trie work and
///   allocated nodes across the entire verification pass to `O(N * H) = O(N)` time and `O(N + A)`
///   memory (at most `16` internal-node steps per distinct mutated arena per control boundary).
/// - Control-flow joins (`join_control_exit`), `Else`/`End` rollbacks/restores, and `Loop`
///   invariant checks each run in `O(1)` (`u32` comparison/assignment) on the happy path and
///   `O(H) <= 16` steps to pinpoint the lowest-indexed differing arena on mismatch.
#[derive(Debug, Default)]
struct ArenaVersionTrie {
    height: u32,
    /// Internal trie nodes indexed by `id - 1`; ID `0` is the canonical all-initial subtree.
    nodes: Vec<(u32, u32)>,
    /// Canonical interning table mapping `(left_id, right_id)` to a unique non-zero node ID.
    intern: HashMap<(u32, u32), u32>,
    /// Straight-line write-combining buffer `(arena_id, live_version)` since the last control boundary.
    pending: Vec<(u16, u16)>,
    /// Lazily allocated 1-based index into `pending` per addressable arena.
    pending_slot: Vec<u32>,
    /// Current committed canonical trie root.
    root: u32,
    #[cfg(test)]
    join_work_steps: u64,
}

impl ArenaVersionTrie {
    fn new(num_arenas: usize) -> Self {
        let effective = num_arenas.min(usize::from(u16::MAX) + 1);
        let height = if effective <= 1 {
            0
        } else {
            (effective - 1).ilog2() + 1
        };
        Self {
            height,
            nodes: Vec::new(),
            intern: HashMap::new(),
            pending: Vec::new(),
            pending_slot: Vec::new(),
            root: 0,
            #[cfg(test)]
            join_work_steps: 0,
        }
    }

    #[cfg(test)]
    const fn join_work_steps(&self) -> u64 {
        self.join_work_steps
    }

    #[cfg(not(test))]
    const fn join_work_steps(&self) -> u64 {
        0
    }

    fn note_join_work(&mut self) {
        #[cfg(test)]
        {
            self.join_work_steps = self.join_work_steps.saturating_add(1);
        }
    }

    fn encode_leaf(ver: u16, initial_ver: u16) -> u32 {
        if ver == initial_ver {
            0
        } else {
            u32::from(ver) + 1
        }
    }

    fn decode_leaf(leaf_id: u32, initial_ver: u16) -> u16 {
        if leaf_id == 0 {
            initial_ver
        } else {
            u16::try_from(leaf_id - 1).unwrap_or(initial_ver)
        }
    }

    fn intern_pair(&mut self, left: u32, right: u32) -> u32 {
        if left == 0 && right == 0 {
            return 0;
        }
        if let Some(&existing) = self.intern.get(&(left, right)) {
            return existing;
        }
        let new_id = u32::try_from(self.nodes.len() + 1).unwrap_or(u32::MAX);
        self.nodes.push((left, right));
        self.intern.insert((left, right), new_id);
        new_id
    }

    fn get_version_from_root(&self, root: u32, arena_id: u16, initial_ver: u16) -> u16 {
        let mut cur = root;
        let mut level = self.height;
        while level > 0 {
            if cur == 0 {
                return initial_ver;
            }
            level -= 1;
            let bit = (u32::from(arena_id) >> level) & 1;
            let (left, right) = self
                .nodes
                .get((cur - 1) as usize)
                .copied()
                .unwrap_or((0, 0));
            cur = if bit == 0 { left } else { right };
        }
        Self::decode_leaf(cur, initial_ver)
    }

    fn get_version(&self, arena_id: u16, initial_ver: u16) -> u16 {
        if !self.pending.is_empty() {
            let slot = self
                .pending_slot
                .get(usize::from(arena_id))
                .copied()
                .unwrap_or(0);
            if slot != 0 {
                if let Some(&(_, ver)) = self.pending.get((slot - 1) as usize) {
                    return ver;
                }
            }
        }
        self.get_version_from_root(self.root, arena_id, initial_ver)
    }

    fn set_live_version(&mut self, arena_id: u16, new_ver: u16, num_arenas: usize) {
        let a_idx = usize::from(arena_id);
        let effective_len = num_arenas.min(usize::from(u16::MAX) + 1);
        if self.pending_slot.len() < effective_len {
            self.pending_slot.resize(effective_len, 0);
        }
        let slot = self.pending_slot.get(a_idx).copied().unwrap_or(0);
        if slot != 0 {
            if let Some(entry) = self.pending.get_mut((slot - 1) as usize) {
                entry.1 = new_ver;
            }
        } else {
            let new_slot = u32::try_from(self.pending.len() + 1).unwrap_or(u32::MAX);
            self.pending.push((arena_id, new_ver));
            if let Some(s) = self.pending_slot.get_mut(a_idx) {
                *s = new_slot;
            }
        }
    }

    fn set_version_in_root(
        &mut self,
        root: u32,
        arena_id: u16,
        new_ver: u16,
        initial_ver: u16,
    ) -> u32 {
        let leaf = Self::encode_leaf(new_ver, initial_ver);
        if self.height == 0 {
            return leaf;
        }
        let mut path = [(0u32, 0u8); 16];
        let mut cur = root;
        let mut level = self.height;
        while level > 0 {
            level -= 1;
            let bit = ((u32::from(arena_id) >> level) & 1) as u8;
            let (left, right) = if cur == 0 {
                (0, 0)
            } else {
                self.nodes
                    .get((cur - 1) as usize)
                    .copied()
                    .unwrap_or((0, 0))
            };
            let (next, sibling) = if bit == 0 {
                (left, right)
            } else {
                (right, left)
            };
            if let Some(slot) = path.get_mut(level as usize) {
                *slot = (sibling, bit);
            }
            cur = next;
        }
        let mut child = leaf;
        for level in 0..self.height {
            self.note_join_work();
            let (sibling, bit) = path.get(level as usize).copied().unwrap_or((0, 0));
            let (left, right) = if bit == 0 {
                (child, sibling)
            } else {
                (sibling, child)
            };
            child = self.intern_pair(left, right);
        }
        child
    }

    fn batch_set_range(
        &mut self,
        cur: u32,
        level: u32,
        base: u32,
        entries: &[(u16, u16)],
        program: &RulProgram,
    ) -> u32 {
        if entries.is_empty() {
            return cur;
        }
        if level == 0 {
            let Some(&(arena_id, new_ver)) = entries.first() else {
                return cur;
            };
            let init_ver = program
                .arenas
                .get(usize::from(arena_id))
                .map_or(0, |desc| desc.initial_version);
            return Self::encode_leaf(new_ver, init_ver);
        }
        self.note_join_work();
        let child_level = level - 1;
        let mid = base + (1u32 << child_level);
        let split = entries.partition_point(|&(id, _)| u32::from(id) < mid);
        let left_entries = entries.get(..split).unwrap_or(&[]);
        let right_entries = entries.get(split..).unwrap_or(&[]);
        let (old_left, old_right) = if cur == 0 {
            (0, 0)
        } else {
            self.nodes
                .get((cur - 1) as usize)
                .copied()
                .unwrap_or((0, 0))
        };
        let new_left = self.batch_set_range(old_left, child_level, base, left_entries, program);
        let new_right = self.batch_set_range(old_right, child_level, mid, right_entries, program);
        self.intern_pair(new_left, new_right)
    }

    fn commit_root(&mut self, program: &RulProgram) -> u32 {
        if self.pending.is_empty() {
            return self.root;
        }
        if self.pending.len() == 1 {
            if let Some((arena_id, new_ver)) = self.pending.pop() {
                let a_idx = usize::from(arena_id);
                if let Some(s) = self.pending_slot.get_mut(a_idx) {
                    *s = 0;
                }
                let init_ver = program
                    .arenas
                    .get(a_idx)
                    .map_or(0, |desc| desc.initial_version);
                self.root = self.set_version_in_root(self.root, arena_id, new_ver, init_ver);
            }
            return self.root;
        }
        let mut pending = std::mem::take(&mut self.pending);
        for &(arena_id, _) in &pending {
            if let Some(s) = self.pending_slot.get_mut(usize::from(arena_id)) {
                *s = 0;
            }
        }
        pending.sort_unstable_by_key(|&(id, _)| id);
        self.root = self.batch_set_range(self.root, self.height, 0, &pending, program);
        pending.clear();
        self.pending = pending;
        self.root
    }

    fn restore_root(&mut self, target_root: u32) {
        if !self.pending.is_empty() {
            for &(arena_id, _) in &self.pending {
                if let Some(s) = self.pending_slot.get_mut(usize::from(arena_id)) {
                    *s = 0;
                }
            }
            self.pending.clear();
        }
        self.root = target_root;
    }

    fn find_first_diff(
        &self,
        root_a: u32,
        root_b: u32,
        program: &RulProgram,
    ) -> Option<(u16, u16, u16)> {
        if root_a == root_b {
            return None;
        }
        let mut a = root_a;
        let mut b = root_b;
        let mut arena_bits: u32 = 0;
        let mut level = self.height;
        while level > 0 {
            level -= 1;
            let (al, ar) = if a == 0 {
                (0, 0)
            } else {
                self.nodes.get((a - 1) as usize).copied().unwrap_or((0, 0))
            };
            let (bl, br) = if b == 0 {
                (0, 0)
            } else {
                self.nodes.get((b - 1) as usize).copied().unwrap_or((0, 0))
            };
            if al != bl {
                a = al;
                b = bl;
            } else {
                arena_bits |= 1u32 << level;
                a = ar;
                b = br;
            }
        }
        let arena_id = u16::try_from(arena_bits).unwrap_or(0);
        let init_ver = program
            .arenas
            .get(usize::from(arena_id))
            .map_or(0, |desc| desc.initial_version);
        Some((
            arena_id,
            Self::decode_leaf(a, init_ver),
            Self::decode_leaf(b, init_ver),
        ))
    }
}

struct VerifierState {
    local_types: [Option<ElemType>; 64],
    local_init: u64,
    reachable: bool,
    arena_state: ArenaVersionTrie,
    node_visible: Vec<bool>,
    control_stack: Vec<ControlFrame>,
    regions: [Option<RegionDeclInfo>; MAX_REGION_SLOTS],
    region_owned: u64,
    region_leased: u64,
    has_unproven_region_access: bool,
    const_data_segs: Vec<ConstDataSegInfo>,
    br_tables: Vec<BrTableInfo>,
    br_table_used: Vec<bool>,
    call_tables: Vec<CallTableSummary>,
    func_sigs: Vec<FuncSigInfo>,
    func_contracts: Vec<FuncRegionContract>,
    current_func: Option<ActiveFunc>,
    pending_call_arg: Option<PendingCallArg>,
    next_func_id: u32,
    completed_funcs: Vec<CompletedFunc>,
    has_func_mode: bool,
    seen_non_func_top_level: bool,
    cur_steps: u64,
    cur_max_callee_depth: u32,
    cur_max_callee_stack_units: u32,
}

fn scan_func_sigs(program: &RulProgram) -> Result<Vec<FuncSigInfo>, VerifyDiagnostic> {
    let mut raw: Vec<FuncSigInfo> = Vec::new();
    for proof in &program.proofs {
        if proof.rule_kind != ProofRuleKind::FuncSig {
            continue;
        }
        let sig = proof.decode_func_sig().ok_or(VerifyDiagnostic {
            node_idx: 0,
            kind: VerifyErrorKind::InvalidFuncSig {
                func_id: u32::from(proof.target_arena),
                reason: "malformed FuncSig proof entry",
            },
        })?;
        raw.push(sig);
    }

    let count = raw.len();
    if count > usize::from(u16::MAX) + 1 {
        return Err(VerifyDiagnostic {
            node_idx: 0,
            kind: VerifyErrorKind::InvalidFuncSig {
                func_id: u32::try_from(count).unwrap_or(u32::MAX),
                reason: "FuncSig count exceeds u16 func_id space (65,536)",
            },
        });
    }
    let mut table: Vec<Option<FuncSigInfo>> = vec![None; count];
    for sig in raw {
        let idx = usize::from(sig.func_id);
        let Some(slot) = table.get_mut(idx) else {
            return Err(VerifyDiagnostic {
                node_idx: 0,
                kind: VerifyErrorKind::InvalidFuncSig {
                    func_id: u32::from(sig.func_id),
                    reason: "FuncSig func_id is out of dense 0..N range",
                },
            });
        };
        if slot.is_some() {
            return Err(VerifyDiagnostic {
                node_idx: 0,
                kind: VerifyErrorKind::InvalidFuncSig {
                    func_id: u32::from(sig.func_id),
                    reason: "duplicate FuncSig for func_id",
                },
            });
        }
        *slot = Some(sig);
    }

    let mut dense = Vec::with_capacity(count);
    for (idx, entry) in table.into_iter().enumerate() {
        let sig = entry.ok_or(VerifyDiagnostic {
            node_idx: 0,
            kind: VerifyErrorKind::InvalidFuncSig {
                func_id: u32::try_from(idx).unwrap_or(u32::MAX),
                reason: "missing FuncSig in dense 0..N range",
            },
        })?;
        dense.push(sig);
    }
    Ok(dense)
}

fn scan_region_decls(
    program: &RulProgram,
) -> Result<[Option<RegionDeclInfo>; MAX_REGION_SLOTS], VerifyDiagnostic> {
    let mut regions: [Option<RegionDeclInfo>; MAX_REGION_SLOTS] = [None; MAX_REGION_SLOTS];
    let mut has_input = false;
    let mut has_output = false;

    for proof in &program.proofs {
        if proof.rule_kind != ProofRuleKind::RegionDecl {
            continue;
        }
        let slot = proof.target_arena;
        let slot_idx = usize::from(slot);
        let Some(slot_entry) = regions.get_mut(slot_idx) else {
            return Err(VerifyDiagnostic {
                node_idx: 0,
                kind: VerifyErrorKind::InvalidRegionDecl {
                    slot,
                    reason: "region slot must be < 64",
                },
            });
        };
        if slot_entry.is_some() {
            return Err(VerifyDiagnostic {
                node_idx: 0,
                kind: VerifyErrorKind::InvalidRegionDecl {
                    slot,
                    reason: "duplicate RegionDecl for slot",
                },
            });
        }
        if proof.bound_lo > proof.bound_hi {
            return Err(VerifyDiagnostic {
                node_idx: 0,
                kind: VerifyErrorKind::InvalidRegionDecl {
                    slot,
                    reason: "region min_len exceeds max_len",
                },
            });
        }
        let Some((elem, role)) = ProofEntry::decode_region_decl_witness(proof.witness_node) else {
            return Err(VerifyDiagnostic {
                node_idx: 0,
                kind: VerifyErrorKind::InvalidRegionDecl {
                    slot,
                    reason: "unsupported region element type or role",
                },
            });
        };
        let max_bytes = u64::from(proof.bound_hi).saturating_mul(elem.region_elem_bytes() as u64);
        if max_bytes > (MAX_REGION_BYTES as u64) {
            return Err(VerifyDiagnostic {
                node_idx: 0,
                kind: VerifyErrorKind::InvalidRegionDecl {
                    slot,
                    reason: "region max_len bytes exceeds MAX_REGION_BYTES (256 MiB)",
                },
            });
        }
        match role {
            RegionRole::Plain => {}
            RegionRole::Input => {
                if elem != ElemType::U8 {
                    return Err(VerifyDiagnostic {
                        node_idx: 0,
                        kind: VerifyErrorKind::InvalidRegionDecl {
                            slot,
                            reason: "input region must have element type U8",
                        },
                    });
                }
                if proof.bound_lo != 0 {
                    return Err(VerifyDiagnostic {
                        node_idx: 0,
                        kind: VerifyErrorKind::InvalidRegionDecl {
                            slot,
                            reason: "input region must have min_len == 0",
                        },
                    });
                }
                if has_input {
                    return Err(VerifyDiagnostic {
                        node_idx: 0,
                        kind: VerifyErrorKind::InvalidRegionDecl {
                            slot,
                            reason: "at most one input region may be declared",
                        },
                    });
                }
                has_input = true;
            }
            RegionRole::Output => {
                if elem != ElemType::U8 {
                    return Err(VerifyDiagnostic {
                        node_idx: 0,
                        kind: VerifyErrorKind::InvalidRegionDecl {
                            slot,
                            reason: "output region must have element type U8",
                        },
                    });
                }
                if has_output {
                    return Err(VerifyDiagnostic {
                        node_idx: 0,
                        kind: VerifyErrorKind::InvalidRegionDecl {
                            slot,
                            reason: "at most one output region may be declared",
                        },
                    });
                }
                has_output = true;
            }
        }
        *slot_entry = Some(RegionDeclInfo {
            slot,
            min_len: proof.bound_lo,
            max_len: proof.bound_hi,
            elem,
            role,
        });
    }

    Ok(regions)
}

const fn align_up_4096(bytes: u64) -> u64 {
    bytes.div_ceil(4096).saturating_mul(4096)
}

const fn shape_steps(shape: ValueShape) -> u64 {
    match shape {
        ValueShape::Scalar => 1,
        ValueShape::Column(len) | ValueShape::Compacted { max_len: len, .. } => {
            if len == 0 {
                1
            } else {
                len as u64
            }
        }
    }
}

fn declared_region_mask(regions: &[Option<RegionDeclInfo>; MAX_REGION_SLOTS]) -> u64 {
    let mut mask = 0u64;
    for (idx, opt) in regions.iter().enumerate() {
        if opt.is_some() {
            mask |= 1u64 << idx;
        }
    }
    mask
}

fn compute_max_region_bytes(
    program: &RulProgram,
    regions: &[Option<RegionDeclInfo>; MAX_REGION_SLOTS],
) -> u64 {
    let mut total: u64 = 0;
    let mut has_any_region = false;
    for decl in regions.iter().flatten() {
        has_any_region = true;
        let raw_bytes =
            u64::from(decl.max_len).saturating_mul(decl.elem.region_elem_bytes() as u64);
        total = total.saturating_add(align_up_4096(raw_bytes));
    }
    if has_any_region {
        total = total.saturating_add(4096);
    }
    for arena in &program.arenas {
        let arena_bytes = u64::from(arena.element_count)
            .saturating_mul(arena.type_lane.elem.region_elem_bytes() as u64);
        total = total.saturating_add(arena_bytes);
    }
    total
}

fn scan_func_region_contracts(
    program: &RulProgram,
    regions: &[Option<RegionDeclInfo>; MAX_REGION_SLOTS],
    declared_mask: u64,
    func_sigs: &[FuncSigInfo],
) -> Result<Vec<FuncRegionContract>, VerifyDiagnostic> {
    #[derive(Clone, Copy, Default)]
    struct RawContract {
        has_any: bool,
        has_none: bool,
        dom_mask: u64,
        pre_owned: u64,
        pre_leased: u64,
        post_owned: u64,
        post_leased: u64,
    }

    let func_count = func_sigs.len();
    let mut raw: Vec<RawContract> = vec![RawContract::default(); func_count];

    for proof in &program.proofs {
        if proof.rule_kind != ProofRuleKind::FuncRegionEffect {
            continue;
        }
        if func_count == 0 {
            return Err(VerifyDiagnostic {
                node_idx: 0,
                kind: VerifyErrorKind::InvalidRegionEffect {
                    func_id: proof.target_arena,
                    reason: "FuncRegionEffect requires FuncSig declarations",
                },
            });
        }
        let eff = proof.decode_func_region_effect().ok_or(VerifyDiagnostic {
            node_idx: 0,
            kind: VerifyErrorKind::InvalidRegionEffect {
                func_id: proof.target_arena,
                reason: "malformed FuncRegionEffect proof entry",
            },
        })?;
        let Some(entry) = raw.get_mut(usize::from(eff.func_id)) else {
            return Err(VerifyDiagnostic {
                node_idx: 0,
                kind: VerifyErrorKind::UnknownFunction(u32::from(eff.func_id)),
            });
        };
        if let Some(slot) = eff.slot {
            if entry.has_none {
                return Err(VerifyDiagnostic {
                    node_idx: 0,
                    kind: VerifyErrorKind::InvalidRegionEffect {
                        func_id: eff.func_id,
                        reason: "cannot combine empty-footprint sentinel with region slot effects",
                    },
                });
            }
            if regions.get(usize::from(slot)).and_then(|r| *r).is_none() {
                return Err(VerifyDiagnostic {
                    node_idx: 0,
                    kind: VerifyErrorKind::UnknownRegion(u32::from(slot)),
                });
            }
            let bit = 1u64 << slot;
            if (entry.dom_mask & bit) != 0 {
                return Err(VerifyDiagnostic {
                    node_idx: 0,
                    kind: VerifyErrorKind::InvalidRegionEffect {
                        func_id: eff.func_id,
                        reason: "duplicate FuncRegionEffect for same function and region slot",
                    },
                });
            }
            entry.has_any = true;
            entry.dom_mask |= bit;
            match eff.pre {
                RegionState::Owned => entry.pre_owned |= bit,
                RegionState::Leased => entry.pre_leased |= bit,
                RegionState::Moved => {}
            }
            match eff.post {
                RegionState::Owned => entry.post_owned |= bit,
                RegionState::Leased => entry.post_leased |= bit,
                RegionState::Moved => {}
            }
        } else {
            if entry.has_none {
                return Err(VerifyDiagnostic {
                    node_idx: 0,
                    kind: VerifyErrorKind::InvalidRegionEffect {
                        func_id: eff.func_id,
                        reason: "duplicate empty-footprint FuncRegionEffect sentinel",
                    },
                });
            }
            if entry.dom_mask != 0 {
                return Err(VerifyDiagnostic {
                    node_idx: 0,
                    kind: VerifyErrorKind::InvalidRegionEffect {
                        func_id: eff.func_id,
                        reason: "cannot combine empty-footprint sentinel with region slot effects",
                    },
                });
            }
            entry.has_any = true;
            entry.has_none = true;
        }
    }

    let mut contracts = Vec::with_capacity(func_count);
    for r in raw {
        let c = if !r.has_any {
            FuncRegionContract {
                dom_mask: declared_mask,
                pre_owned: declared_mask,
                pre_leased: 0,
                post_owned: declared_mask,
                post_leased: 0,
            }
        } else if r.has_none {
            FuncRegionContract {
                dom_mask: 0,
                pre_owned: 0,
                pre_leased: 0,
                post_owned: 0,
                post_leased: 0,
            }
        } else {
            FuncRegionContract {
                dom_mask: r.dom_mask,
                pre_owned: r.pre_owned,
                pre_leased: r.pre_leased,
                post_owned: r.post_owned,
                post_leased: r.post_leased,
            }
        };
        contracts.push(c);
    }

    if let Some(entry_contract) = contracts.last().copied() {
        let entry_id = u16::try_from(func_count.saturating_sub(1)).unwrap_or(0);
        if (entry_contract.pre_owned & entry_contract.dom_mask) != entry_contract.dom_mask
            || entry_contract.pre_leased != 0
        {
            return Err(VerifyDiagnostic {
                node_idx: 0,
                kind: VerifyErrorKind::InvalidRegionEffect {
                    func_id: entry_id,
                    reason: "entry function precondition must be Owned for every region in its footprint",
                },
            });
        }
    }

    Ok(contracts)
}

fn func_sig_matches(a: FuncSigInfo, b: FuncSigInfo) -> bool {
    a.arity == b.arity
        && a.ret_elem == b.ret_elem
        && (0..usize::from(a.arity)).all(|i| a.params.get(i) == b.params.get(i))
}

struct ScannedConstTables {
    const_data_segs: Vec<ConstDataSegInfo>,
    br_tables: Vec<BrTableInfo>,
    call_tables: Vec<CallTableSummary>,
}

#[expect(
    clippy::too_many_lines,
    reason = "single-pass O(proofs + const_data) validation of non-overlapping constant segments, BrTableSpecs, and CallTables"
)]
fn scan_const_data_tables(
    program: &RulProgram,
    func_sigs: &[FuncSigInfo],
    func_contracts: &[FuncRegionContract],
) -> Result<ScannedConstTables, VerifyDiagnostic> {
    if program.const_data.len() > (MAX_CONST_DATA_BYTES as usize) {
        return Err(VerifyDiagnostic {
            node_idx: 0,
            kind: VerifyErrorKind::InvalidConstDataTable {
                reason: "const_data section exceeds MAX_CONST_DATA_BYTES (8 MiB)",
            },
        });
    }
    if program.link_meta.len() > (MAX_LINK_META_BYTES as usize) {
        return Err(VerifyDiagnostic {
            node_idx: 0,
            kind: VerifyErrorKind::InvalidConstDataTable {
                reason: "link_meta section exceeds MAX_LINK_META_BYTES (2 MiB)",
            },
        });
    }

    let const_len = program.const_data.len() as u64;
    let mut prev_end: u64 = 0;
    let mut raw_segs: Vec<ConstDataSegInfo> = Vec::new();
    let mut raw_br_tables: Vec<BrTableInfo> = Vec::new();
    let mut raw_call_tables: Vec<CallTableSummary> = Vec::new();

    for proof in &program.proofs {
        match proof.rule_kind {
            ProofRuleKind::ConstDataSeg => {
                if proof.witness_node != 0 {
                    return Err(VerifyDiagnostic {
                        node_idx: 0,
                        kind: VerifyErrorKind::InvalidConstDataTable {
                            reason: "ConstDataSeg requires witness_node == 0",
                        },
                    });
                }
                let lo = u64::from(proof.bound_lo);
                let hi = lo.saturating_add(u64::from(proof.bound_hi));
                if lo < prev_end || hi > const_len {
                    return Err(VerifyDiagnostic {
                        node_idx: 0,
                        kind: VerifyErrorKind::InvalidConstDataTable {
                            reason: "ConstDataSeg range is out of bounds, unordered, or overlaps a previous table",
                        },
                    });
                }
                prev_end = hi;
                raw_segs.push(ConstDataSegInfo {
                    seg_id: proof.target_arena,
                    offset: proof.bound_lo,
                    len: proof.bound_hi,
                });
            }
            ProofRuleKind::BrTableSpec => {
                let case_count = proof.bound_hi;
                if case_count == 0 || case_count > MAX_TABLE_ENTRIES {
                    return Err(VerifyDiagnostic {
                        node_idx: 0,
                        kind: VerifyErrorKind::InvalidConstDataTable {
                            reason: "BrTableSpec case_count must be in 1..=4096",
                        },
                    });
                }
                let lo = u64::from(proof.bound_lo);
                let byte_len = u64::from(case_count).saturating_mul(4);
                let hi = lo.saturating_add(byte_len);
                if lo < prev_end || hi > const_len {
                    return Err(VerifyDiagnostic {
                        node_idx: 0,
                        kind: VerifyErrorKind::InvalidConstDataTable {
                            reason: "BrTableSpec payload is out of bounds, unordered, or overlaps a previous table",
                        },
                    });
                }
                prev_end = hi;
                raw_br_tables.push(BrTableInfo {
                    br_table_id: proof.witness_node,
                    default_depth: proof.target_arena,
                    offset: proof.bound_lo,
                    case_count,
                });
            }
            ProofRuleKind::CallTable => {
                if proof.witness_node != 0 {
                    return Err(VerifyDiagnostic {
                        node_idx: 0,
                        kind: VerifyErrorKind::InvalidConstDataTable {
                            reason: "CallTable requires witness_node == 0",
                        },
                    });
                }
                let entry_count = proof.bound_hi;
                if entry_count == 0 || entry_count > MAX_TABLE_ENTRIES {
                    return Err(VerifyDiagnostic {
                        node_idx: 0,
                        kind: VerifyErrorKind::InvalidConstDataTable {
                            reason: "CallTable entry_count must be in 1..=4096",
                        },
                    });
                }
                if func_sigs.is_empty() {
                    return Err(VerifyDiagnostic {
                        node_idx: 0,
                        kind: VerifyErrorKind::InvalidConstDataTable {
                            reason: "CallTable requires FuncSig declarations",
                        },
                    });
                }
                let lo = u64::from(proof.bound_lo);
                let byte_len = u64::from(entry_count).saturating_mul(4);
                let hi = lo.saturating_add(byte_len);
                if lo < prev_end || hi > const_len {
                    return Err(VerifyDiagnostic {
                        node_idx: 0,
                        kind: VerifyErrorKind::InvalidConstDataTable {
                            reason: "CallTable payload is out of bounds, unordered, or overlaps a previous table",
                        },
                    });
                }
                prev_end = hi;

                let slice = program.const_data.get((lo as usize)..(hi as usize)).ok_or(
                    VerifyDiagnostic {
                        node_idx: 0,
                        kind: VerifyErrorKind::InvalidConstDataTable {
                            reason: "CallTable payload slice out of bounds",
                        },
                    },
                )?;
                let mut expected_sig: Option<FuncSigInfo> = None;
                let mut expected_contract: Option<FuncRegionContract> = None;
                let mut rep_func_id: u16 = 0;
                let mut max_func_id: u32 = 0;
                for chunk in slice.as_chunks::<4>().0 {
                    let fid = u32::from_le_bytes(*chunk);
                    let sig = func_sigs
                        .get(fid as usize)
                        .copied()
                        .ok_or(VerifyDiagnostic {
                            node_idx: 0,
                            kind: VerifyErrorKind::UnknownFunction(fid),
                        })?;
                    let contract =
                        func_contracts
                            .get(fid as usize)
                            .copied()
                            .ok_or(VerifyDiagnostic {
                                node_idx: 0,
                                kind: VerifyErrorKind::UnknownFunction(fid),
                            })?;
                    if let Some(ref_sig) = expected_sig {
                        if !func_sig_matches(ref_sig, sig) {
                            return Err(VerifyDiagnostic {
                                node_idx: 0,
                                kind: VerifyErrorKind::InvalidConstDataTable {
                                    reason: "CallTable entries disagree on FuncSig",
                                },
                            });
                        }
                    } else {
                        expected_sig = Some(sig);
                        rep_func_id = sig.func_id;
                    }
                    if let Some(ref_contract) = expected_contract {
                        if ref_contract != contract {
                            return Err(VerifyDiagnostic {
                                node_idx: 0,
                                kind: VerifyErrorKind::InvalidConstDataTable {
                                    reason: "CallTable entries disagree on FuncRegionContract",
                                },
                            });
                        }
                    } else {
                        expected_contract = Some(contract);
                    }
                    max_func_id = max_func_id.max(fid);
                }
                let (Some(sig), Some(contract)) = (expected_sig, expected_contract) else {
                    return Err(VerifyDiagnostic {
                        node_idx: 0,
                        kind: VerifyErrorKind::InvalidConstDataTable {
                            reason: "CallTable cannot be empty",
                        },
                    });
                };
                raw_call_tables.push(CallTableSummary {
                    table_id: proof.target_arena,
                    rep_func_id,
                    sig,
                    contract,
                    max_func_id,
                    offset: proof.bound_lo,
                    entry_count,
                    cost_summary: None,
                });
            }
            _ => {}
        }
    }

    // Build dense 0..N tables and reject duplicates or gaps.
    let mut dense_segs: Vec<Option<ConstDataSegInfo>> = vec![None; raw_segs.len()];
    for seg in raw_segs {
        let Some(slot) = dense_segs.get_mut(usize::from(seg.seg_id)) else {
            return Err(VerifyDiagnostic {
                node_idx: 0,
                kind: VerifyErrorKind::InvalidConstDataTable {
                    reason: "ConstDataSeg seg_id is out of dense 0..N range",
                },
            });
        };
        if slot.is_some() {
            return Err(VerifyDiagnostic {
                node_idx: 0,
                kind: VerifyErrorKind::InvalidConstDataTable {
                    reason: "duplicate ConstDataSeg seg_id",
                },
            });
        }
        *slot = Some(seg);
    }
    let mut segs = Vec::with_capacity(dense_segs.len());
    for entry in dense_segs {
        segs.push(entry.ok_or(VerifyDiagnostic {
            node_idx: 0,
            kind: VerifyErrorKind::InvalidConstDataTable {
                reason: "missing ConstDataSeg seg_id in dense 0..N range",
            },
        })?);
    }

    let mut dense_br: Vec<Option<BrTableInfo>> = vec![None; raw_br_tables.len()];
    for spec in raw_br_tables {
        let Some(slot) = dense_br.get_mut(spec.br_table_id as usize) else {
            return Err(VerifyDiagnostic {
                node_idx: 0,
                kind: VerifyErrorKind::InvalidConstDataTable {
                    reason: "BrTableSpec br_table_id is out of dense 0..N range",
                },
            });
        };
        if slot.is_some() {
            return Err(VerifyDiagnostic {
                node_idx: 0,
                kind: VerifyErrorKind::InvalidConstDataTable {
                    reason: "duplicate BrTableSpec br_table_id",
                },
            });
        }
        *slot = Some(spec);
    }
    let mut br_tables = Vec::with_capacity(dense_br.len());
    for entry in dense_br {
        br_tables.push(entry.ok_or(VerifyDiagnostic {
            node_idx: 0,
            kind: VerifyErrorKind::InvalidConstDataTable {
                reason: "missing BrTableSpec br_table_id in dense 0..N range",
            },
        })?);
    }

    let mut dense_call: Vec<Option<CallTableSummary>> = vec![None; raw_call_tables.len()];
    for ct in raw_call_tables {
        let Some(slot) = dense_call.get_mut(usize::from(ct.table_id)) else {
            return Err(VerifyDiagnostic {
                node_idx: 0,
                kind: VerifyErrorKind::InvalidConstDataTable {
                    reason: "CallTable table_id is out of dense 0..N range",
                },
            });
        };
        if slot.is_some() {
            return Err(VerifyDiagnostic {
                node_idx: 0,
                kind: VerifyErrorKind::InvalidConstDataTable {
                    reason: "duplicate CallTable table_id",
                },
            });
        }
        *slot = Some(ct);
    }
    let mut call_tables = Vec::with_capacity(dense_call.len());
    for entry in dense_call {
        call_tables.push(entry.ok_or(VerifyDiagnostic {
            node_idx: 0,
            kind: VerifyErrorKind::InvalidConstDataTable {
                reason: "missing CallTable table_id in dense 0..N range",
            },
        })?);
    }

    Ok(ScannedConstTables {
        const_data_segs: segs,
        br_tables,
        call_tables,
    })
}

/// Verifies an [`RulProgram`] in a single forward pass ($O(N)$ time and $O(N)$ state).
///
/// # Errors
/// Returns a structured [`VerifyDiagnostic`] pinpointing the exact node index and rule
/// violation if any safety, ownership, bounds, or capability invariant fails.
pub fn verify_program(program: &RulProgram) -> Result<VerifiedSummary, VerifyDiagnostic> {
    verify_program_core(program).map(|(summary, _)| summary)
}

#[cfg(test)]
pub(crate) fn verify_program_with_join_work(
    program: &RulProgram,
) -> Result<(VerifiedSummary, u64), VerifyDiagnostic> {
    verify_program_core(program)
}

fn verify_program_core(program: &RulProgram) -> Result<(VerifiedSummary, u64), VerifyDiagnostic> {
    let node_count = program.nodes.len();
    if node_count == 0 || (program.output_node as usize) >= node_count {
        return Err(VerifyDiagnostic {
            node_idx: program.output_node,
            kind: VerifyErrorKind::InvalidOutputNode(program.output_node),
        });
    }

    // One linear pre-scan of the proof table to index declared regions, function signatures, region contracts, and constant-data tables.
    let regions = scan_region_decls(program)?;
    let declared_mask = declared_region_mask(&regions);
    let func_sigs = scan_func_sigs(program)?;
    let func_contracts = scan_func_region_contracts(program, &regions, declared_mask, &func_sigs)?;
    let ScannedConstTables {
        const_data_segs,
        br_tables,
        call_tables,
    } = scan_const_data_tables(program, &func_sigs, &func_contracts)?;
    let br_table_used = vec![false; br_tables.len()];
    let has_func_mode = !func_sigs.is_empty();
    let sandbox_mode = (program.capability_mask & CAP_DIVERGE) == 0;
    let max_region_bytes = compute_max_region_bytes(program, &regions);

    // Track inferred ValueShape, ElemType, and origin Arena per SSA node.
    let mut metas: Vec<NodeMeta> = Vec::with_capacity(node_count);
    let mut state = VerifierState {
        local_types: [None; 64],
        local_init: 0,
        reachable: true,
        arena_state: ArenaVersionTrie::new(program.arenas.len()),
        node_visible: Vec::with_capacity(node_count),
        control_stack: Vec::new(),
        regions,
        region_owned: if has_func_mode { 0 } else { declared_mask },
        region_leased: 0,
        has_unproven_region_access: false,
        const_data_segs,
        br_tables,
        br_table_used,
        call_tables,
        func_sigs,
        func_contracts,
        current_func: None,
        pending_call_arg: None,
        next_func_id: 0,
        completed_funcs: Vec::new(),
        has_func_mode,
        seen_non_func_top_level: false,
        cur_steps: 0,
        cur_max_callee_depth: 0,
        cur_max_callee_stack_units: 0,
    };
    let mut max_fusion_group: u8 = 0;
    let mut has_gather = false;

    for (idx, frame) in program.nodes.iter().enumerate() {
        let node_idx = u32::try_from(idx).unwrap_or(u32::MAX);

        if let Some(pending) = state.pending_call_arg {
            if !matches!(frame.opcode, Opcode::Call | Opcode::CallIndirect) {
                return Err(VerifyDiagnostic {
                    node_idx: pending.node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "CallArg must immediately precede Call or CallIndirect",
                    ),
                });
            }
        }

        if frame.opcode == Opcode::Func {
            if state.seen_non_func_top_level {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "node outside Func scope in function program",
                    ),
                });
            }
            state.has_func_mode = true;
        } else if state.current_func.is_none() {
            if state.has_func_mode {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "node outside Func scope in function program",
                    ),
                });
            }
            state.seen_non_func_top_level = true;
        }

        if state.has_func_mode && frame.token_op.mode != TokenMode::PureValue {
            return Err(VerifyDiagnostic {
                node_idx,
                kind: VerifyErrorKind::InvalidControlFlow(
                    "non-pure token_op is forbidden in programs containing Func",
                ),
            });
        }

        if let Some(active) = state.current_func {
            let scope_nodes_so_far = (node_idx - active.start_node) as usize + 1;
            if scope_nodes_so_far > MAX_FUNC_NODES {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::FunctionTooLarge {
                        func_id: active.sig.func_id,
                        node_count: scope_nodes_so_far,
                    },
                });
            }
        }

        if frame.schedule.fusion_group > max_fusion_group {
            max_fusion_group = frame.schedule.fusion_group;
        }
        if frame.opcode == Opcode::Gather {
            has_gather = true;
        }

        let was_reachable = state.reachable;

        // 1. Verify AffineToken / ReadLease transition if present.
        verify_token_transition(program, frame, node_idx, &mut state)?;

        // 2. Verify opcode-specific topological edges, types, shapes, control stack, and proofs.
        let meta = verify_node_semantics(program, frame, node_idx, &metas, &mut state)?;

        if was_reachable
            && !matches!(
                frame.opcode,
                Opcode::Block
                    | Opcode::Loop
                    | Opcode::If
                    | Opcode::Else
                    | Opcode::End
                    | Opcode::Func
                    | Opcode::Call
                    | Opcode::CallIndirect
            )
        {
            let step_cost = match frame.opcode {
                Opcode::ReduceSum | Opcode::ReduceMax | Opcode::ReduceMin => {
                    let abs_src = state
                        .current_func
                        .map_or(frame.in_a, |a| a.start_node.saturating_add(frame.in_a));
                    metas
                        .get(abs_src as usize)
                        .map_or(1, |m| shape_steps(m.shape))
                }
                _ => shape_steps(meta.shape),
            };
            state.cur_steps = state.cur_steps.saturating_add(step_cost);
        }

        metas.push(meta);
        state.node_visible.push(!meta.is_void);
        if !meta.is_void {
            if let Some(active_scope) = state.control_stack.last_mut() {
                active_scope.scope_nodes.push(idx);
            }
        }
    }

    if let Some(pending) = state.pending_call_arg {
        return Err(VerifyDiagnostic {
            node_idx: pending.node_idx,
            kind: VerifyErrorKind::InvalidControlFlow("dangling CallArg at end of program"),
        });
    }

    if let Some(unclosed) = state.control_stack.last() {
        return Err(VerifyDiagnostic {
            node_idx: unclosed.header_node,
            kind: VerifyErrorKind::InvalidControlFlow(
                "unclosed structured control block at end of program",
            ),
        });
    }

    if state.has_func_mode {
        if state.completed_funcs.len() != state.func_sigs.len() {
            return Err(VerifyDiagnostic {
                node_idx: 0,
                kind: VerifyErrorKind::InvalidFuncSig {
                    func_id: u32::try_from(state.completed_funcs.len()).unwrap_or(u32::MAX),
                    reason: "Func count does not match FuncSig proof count",
                },
            });
        }
        let entry = state
            .completed_funcs
            .last()
            .copied()
            .ok_or(VerifyDiagnostic {
                node_idx: 0,
                kind: VerifyErrorKind::InvalidFuncSig {
                    func_id: 0,
                    reason: "function program must declare at least one Func",
                },
            })?;
        if entry.sig.arity != 0 {
            return Err(VerifyDiagnostic {
                node_idx: entry.start_node,
                kind: VerifyErrorKind::InvalidFuncSig {
                    func_id: u32::from(entry.sig.func_id),
                    reason: "entry function must have arity 0",
                },
            });
        }
        let Some(entry_ret_elem) = entry.sig.ret_elem else {
            return Err(VerifyDiagnostic {
                node_idx: entry.start_node,
                kind: VerifyErrorKind::InvalidFuncSig {
                    func_id: u32::from(entry.sig.func_id),
                    reason: "entry function must have a non-void return type",
                },
            });
        };
        if program.output_node != entry.start_node {
            return Err(VerifyDiagnostic {
                node_idx: program.output_node,
                kind: VerifyErrorKind::InvalidOutputNode(program.output_node),
            });
        }
        let static_cost = if sandbox_mode {
            let max_stack_bytes = u64::from(entry.max_stack_units)
                .saturating_mul(8)
                .saturating_add(4096);
            Some(StaticCostSummary {
                worst_case_steps: entry.worst_case_steps,
                max_call_depth: entry.max_call_depth,
                max_stack_units: entry.max_stack_units,
                max_stack_bytes,
                max_region_bytes,
            })
        } else {
            None
        };
        return Ok((
            VerifiedSummary {
                node_count,
                fusion_groups: usize::from(max_fusion_group),
                output_shape: ValueShape::Scalar,
                output_elem: entry_ret_elem,
                zero_runtime_bounds_checks: !has_gather && !state.has_unproven_region_access,
                static_cost,
            },
            state.arena_state.join_work_steps(),
        ));
    }

    let output_idx = program.output_node as usize;
    let output_meta = metas.get(output_idx).copied().ok_or(VerifyDiagnostic {
        node_idx: program.output_node,
        kind: VerifyErrorKind::InvalidOutputNode(program.output_node),
    })?;
    if output_meta.is_void {
        return Err(VerifyDiagnostic {
            node_idx: program.output_node,
            kind: VerifyErrorKind::InvalidOutputNode(program.output_node),
        });
    }
    if !state.node_visible.get(output_idx).copied().unwrap_or(false) {
        return Err(VerifyDiagnostic {
            node_idx: program.output_node,
            kind: VerifyErrorKind::ScopedValueEscaped {
                input_node: program.output_node,
            },
        });
    }

    let static_cost = if sandbox_mode {
        Some(StaticCostSummary {
            worst_case_steps: state.cur_steps,
            max_call_depth: 1,
            max_stack_units: 0,
            max_stack_bytes: 0,
            max_region_bytes,
        })
    } else {
        None
    };

    Ok((
        VerifiedSummary {
            node_count,
            fusion_groups: usize::from(max_fusion_group),
            output_shape: output_meta.shape,
            output_elem: output_meta.elem,
            zero_runtime_bounds_checks: !has_gather && !state.has_unproven_region_access,
            static_cost,
        },
        state.arena_state.join_work_steps(),
    ))
}

fn verify_token_transition(
    program: &RulProgram,
    frame: &NodeFrame,
    node_idx: u32,
    state: &mut VerifierState,
) -> Result<(), VerifyDiagnostic> {
    match frame.token_op.mode {
        TokenMode::PureValue => Ok(()),
        TokenMode::ReadLease => {
            let arena_idx = usize::from(frame.token_op.arena_id);
            let arena_desc = program.arenas.get(arena_idx).ok_or(VerifyDiagnostic {
                node_idx,
                kind: VerifyErrorKind::UnknownArena(u32::from(frame.token_op.arena_id)),
            })?;
            if state.reachable {
                let live_ver = state
                    .arena_state
                    .get_version(frame.token_op.arena_id, arena_desc.initial_version);
                if frame.token_op.version != live_ver {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::StaleOrConsumedToken {
                            arena_id: frame.token_op.arena_id,
                            provided_version: frame.token_op.version,
                            expected_version: live_ver,
                        },
                    });
                }
            }
            Ok(())
        }
        TokenMode::AffineConsume => {
            let arena_idx = usize::from(frame.token_op.arena_id);
            let arena_desc = program.arenas.get(arena_idx).ok_or(VerifyDiagnostic {
                node_idx,
                kind: VerifyErrorKind::UnknownArena(u32::from(frame.token_op.arena_id)),
            })?;
            if !arena_desc.mutable {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ImmutableArenaMutation(frame.token_op.arena_id),
                });
            }
            if !state.reachable {
                let _ = frame
                    .token_op
                    .version
                    .checked_add(1)
                    .ok_or(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::TokenVersionOverflow(frame.token_op.arena_id),
                    })?;
                return Ok(());
            }
            let live_ver = state
                .arena_state
                .get_version(frame.token_op.arena_id, arena_desc.initial_version);
            if frame.token_op.version != live_ver {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::StaleOrConsumedToken {
                        arena_id: frame.token_op.arena_id,
                        provided_version: frame.token_op.version,
                        expected_version: live_ver,
                    },
                });
            }
            // Linear consumption: advance version so old token can never be reused,
            // and reject u16::MAX overflow so version can never wrap around to 0!
            let next_ver = live_ver.checked_add(1).ok_or(VerifyDiagnostic {
                node_idx,
                kind: VerifyErrorKind::TokenVersionOverflow(frame.token_op.arena_id),
            })?;
            state.arena_state.set_live_version(
                frame.token_op.arena_id,
                next_ver,
                program.arenas.len(),
            );
            Ok(())
        }
    }
}

fn get_prior_meta(
    node_idx: u32,
    input_idx: u32,
    metas: &[NodeMeta],
    state: &VerifierState,
) -> Result<NodeMeta, VerifyDiagnostic> {
    let abs_idx = if let Some(active) = state.current_func {
        if input_idx == 0 {
            return Err(VerifyDiagnostic {
                node_idx,
                kind: VerifyErrorKind::InvalidControlFlow(
                    "relative operand 0 names the enclosing Func node and is not a value",
                ),
            });
        }
        active
            .start_node
            .checked_add(input_idx)
            .ok_or(VerifyDiagnostic {
                node_idx,
                kind: VerifyErrorKind::NonTopologicalEdge {
                    input_node: input_idx,
                },
            })?
    } else {
        input_idx
    };
    if abs_idx >= node_idx {
        return Err(VerifyDiagnostic {
            node_idx,
            kind: VerifyErrorKind::NonTopologicalEdge {
                input_node: input_idx,
            },
        });
    }
    let idx = abs_idx as usize;
    let meta = metas.get(idx).copied().ok_or(VerifyDiagnostic {
        node_idx,
        kind: VerifyErrorKind::NonTopologicalEdge {
            input_node: input_idx,
        },
    })?;
    if meta.is_void {
        return Err(VerifyDiagnostic {
            node_idx,
            kind: VerifyErrorKind::InvalidControlFlow(
                "void control-flow instruction cannot be used as an SSA value operand",
            ),
        });
    }
    if !state.node_visible.get(idx).copied().unwrap_or(false) {
        return Err(VerifyDiagnostic {
            node_idx,
            kind: VerifyErrorKind::ScopedValueEscaped {
                input_node: input_idx,
            },
        });
    }
    Ok(meta)
}

fn require_pure_token(node_idx: u32, frame: &NodeFrame) -> Result<(), VerifyDiagnostic> {
    if frame.token_op.mode != TokenMode::PureValue {
        return Err(VerifyDiagnostic {
            node_idx,
            kind: VerifyErrorKind::InvalidControlFlow(
                "structured control-flow and local instructions must carry pure token mode",
            ),
        });
    }
    Ok(())
}

fn parse_block_sig(
    program: &RulProgram,
    frame: &NodeFrame,
    node_idx: u32,
    arena_state: &ArenaVersionTrie,
    reachable: bool,
) -> Result<Option<ParsedBlockSig>, VerifyDiagnostic> {
    if frame.proof_idx == 0 {
        return Ok(None);
    }
    let proof = program
        .proofs
        .get(usize::from(frame.proof_idx - 1))
        .ok_or(VerifyDiagnostic {
            node_idx,
            kind: VerifyErrorKind::MissingRefinementProof,
        })?;
    if proof.rule_kind != ProofRuleKind::BlockSig {
        return Err(VerifyDiagnostic {
            node_idx,
            kind: VerifyErrorKind::InvalidRefinementProof {
                proof_idx: frame.proof_idx,
                bound_hi: proof.bound_hi,
                arena_len: 0,
            },
        });
    }

    let result_type = if proof.witness_node == 0 {
        None
    } else {
        let tl_byte = u8::try_from(proof.witness_node).map_err(|_err| VerifyDiagnostic {
            node_idx,
            kind: VerifyErrorKind::InvalidRefinementProof {
                proof_idx: frame.proof_idx,
                bound_hi: proof.witness_node,
                arena_len: 0,
            },
        })?;
        let tl = TypeLane::from_u8(tl_byte).map_err(|_err| VerifyDiagnostic {
            node_idx,
            kind: VerifyErrorKind::InvalidRefinementProof {
                proof_idx: frame.proof_idx,
                bound_hi: proof.witness_node,
                arena_len: 0,
            },
        })?;
        if tl.width != SimdWidth::Scalar {
            return Err(VerifyDiagnostic {
                node_idx,
                kind: VerifyErrorKind::InvalidRefinementProof {
                    proof_idx: frame.proof_idx,
                    bound_hi: proof.witness_node,
                    arena_len: 1,
                },
            });
        }
        if frame.type_lane != tl {
            return Err(VerifyDiagnostic {
                node_idx,
                kind: VerifyErrorKind::ElementTypeMismatch {
                    expected: tl.elem,
                    actual: frame.type_lane.elem,
                },
            });
        }
        Some(tl)
    };

    let tracks_arena = proof.target_arena != 0 || proof.bound_lo != 0 || proof.bound_hi != 0;
    if !tracks_arena {
        return Ok(Some(ParsedBlockSig {
            arena_delta: None,
            result_type,
        }));
    }

    let arena_idx = usize::from(proof.target_arena);
    let arena_desc = program.arenas.get(arena_idx).ok_or(VerifyDiagnostic {
        node_idx,
        kind: VerifyErrorKind::UnknownArena(u32::from(proof.target_arena)),
    })?;
    if proof.bound_lo > proof.bound_hi {
        return Err(VerifyDiagnostic {
            node_idx,
            kind: VerifyErrorKind::InvalidRefinementProof {
                proof_idx: frame.proof_idx,
                bound_hi: proof.bound_hi,
                arena_len: proof.bound_lo,
            },
        });
    }
    let entry_version = u16::try_from(proof.bound_lo).map_err(|_err| VerifyDiagnostic {
        node_idx,
        kind: VerifyErrorKind::InvalidRefinementProof {
            proof_idx: frame.proof_idx,
            bound_hi: proof.bound_lo,
            arena_len: u32::from(u16::MAX),
        },
    })?;
    let exit_version = u16::try_from(proof.bound_hi).map_err(|_err| VerifyDiagnostic {
        node_idx,
        kind: VerifyErrorKind::InvalidRefinementProof {
            proof_idx: frame.proof_idx,
            bound_hi: proof.bound_hi,
            arena_len: u32::from(u16::MAX),
        },
    })?;
    if entry_version < exit_version && !arena_desc.mutable {
        return Err(VerifyDiagnostic {
            node_idx,
            kind: VerifyErrorKind::ImmutableArenaMutation(proof.target_arena),
        });
    }
    if reachable {
        let live_ver = arena_state.get_version(proof.target_arena, arena_desc.initial_version);
        if live_ver != entry_version {
            return Err(VerifyDiagnostic {
                node_idx,
                kind: VerifyErrorKind::StaleOrConsumedToken {
                    arena_id: proof.target_arena,
                    provided_version: entry_version,
                    expected_version: live_ver,
                },
            });
        }
    }

    Ok(Some(ParsedBlockSig {
        arena_delta: Some(ArenaDelta {
            arena_id: proof.target_arena,
            entry_version,
            exit_version,
        }),
        result_type,
    }))
}

fn check_loop_arena_invariant(
    program: &RulProgram,
    node_idx: u32,
    loop_frame: &ControlFrame,
    arena_state: &mut ArenaVersionTrie,
) -> Result<(), VerifyDiagnostic> {
    let cur_root = arena_state.commit_root(program);
    if cur_root == loop_frame.entry_arena_root {
        return Ok(());
    }
    let (arena_id, actual_ver, entry_ver) = arena_state
        .find_first_diff(cur_root, loop_frame.entry_arena_root, program)
        .unwrap_or((0, 0, 0));
    Err(VerifyDiagnostic {
        node_idx,
        kind: VerifyErrorKind::StaleOrConsumedToken {
            arena_id,
            provided_version: actual_ver,
            expected_version: entry_ver,
        },
    })
}

fn check_loop_region_invariant(
    node_idx: u32,
    loop_frame: &ControlFrame,
    cur_region_owned: u64,
    cur_region_leased: u64,
) -> Result<(), VerifyDiagnostic> {
    if cur_region_owned == loop_frame.entry_region_owned
        && cur_region_leased == loop_frame.entry_region_leased
    {
        return Ok(());
    }
    Err(VerifyDiagnostic {
        node_idx,
        kind: VerifyErrorKind::LoopTypestateInvariantViolation {
            expected_owned: loop_frame.entry_region_owned,
            expected_leased: loop_frame.entry_region_leased,
            actual_owned: cur_region_owned,
            actual_leased: cur_region_leased,
        },
    })
}

fn check_region_in_footprint(
    node_idx: u32,
    slot: u16,
    state: &VerifierState,
) -> Result<(), VerifyDiagnostic> {
    if let Some(active) = state.current_func {
        let dom = state
            .func_contracts
            .get(usize::from(active.sig.func_id))
            .map_or(0, |c| c.dom_mask);
        if (dom & (1u64 << slot)) == 0 {
            return Err(VerifyDiagnostic {
                node_idx,
                kind: VerifyErrorKind::RegionOutsideFootprint { slot },
            });
        }
    }
    Ok(())
}

fn check_region_read(
    node_idx: u32,
    slot: u16,
    state: &VerifierState,
) -> Result<(), VerifyDiagnostic> {
    if !state.reachable {
        return Ok(());
    }
    let bit = 1u64 << slot;
    if ((state.region_owned | state.region_leased) & bit) == 0 {
        return Err(VerifyDiagnostic {
            node_idx,
            kind: VerifyErrorKind::UseOfMovedRegion { slot },
        });
    }
    Ok(())
}

fn check_region_mutate(
    node_idx: u32,
    slot: u16,
    state: &VerifierState,
) -> Result<(), VerifyDiagnostic> {
    if !state.reachable {
        return Ok(());
    }
    let bit = 1u64 << slot;
    if (state.region_leased & bit) != 0 {
        return Err(VerifyDiagnostic {
            node_idx,
            kind: VerifyErrorKind::MutationUnderLiveLease { slot },
        });
    }
    if (state.region_owned & bit) == 0 {
        return Err(VerifyDiagnostic {
            node_idx,
            kind: VerifyErrorKind::UseOfMovedRegion { slot },
        });
    }
    Ok(())
}

fn join_control_exit(
    program: &RulProgram,
    ctrl: &mut ControlFrame,
    node_idx: u32,
    snap: ExitSnapshot,
    arena_state: &mut ArenaVersionTrie,
) -> Result<(), VerifyDiagnostic> {
    let cur_root = arena_state.commit_root(program);
    arena_state.note_join_work();
    if let Some(decl) = ctrl.sig.and_then(|s| s.arena_delta) {
        let init_ver = program
            .arenas
            .get(usize::from(decl.arena_id))
            .map_or(0, |a| a.initial_version);
        let expected_root = if ctrl.has_exit_path {
            ctrl.exit_arena_root
        } else {
            arena_state.set_version_in_root(
                ctrl.entry_arena_root,
                decl.arena_id,
                decl.exit_version,
                init_ver,
            )
        };
        if cur_root != expected_root {
            let act_target = arena_state.get_version_from_root(cur_root, decl.arena_id, init_ver);
            if act_target != decl.exit_version {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::StaleOrConsumedToken {
                        arena_id: decl.arena_id,
                        provided_version: act_target,
                        expected_version: decl.exit_version,
                    },
                });
            }
            if let Some((arena_id, provided_version, expected_version)) =
                arena_state.find_first_diff(cur_root, expected_root, program)
            {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::StaleOrConsumedToken {
                        arena_id,
                        provided_version,
                        expected_version,
                    },
                });
            }
        }
        if !ctrl.has_exit_path {
            ctrl.has_exit_path = true;
            ctrl.exit_init = snap.local_init;
            ctrl.exit_region_owned = snap.region_owned;
            ctrl.exit_region_leased = snap.region_leased;
            ctrl.exit_arena_root = expected_root;
            return Ok(());
        }
        ctrl.exit_init &= snap.local_init;
        if ctrl.exit_region_owned != snap.region_owned
            || ctrl.exit_region_leased != snap.region_leased
        {
            return Err(VerifyDiagnostic {
                node_idx,
                kind: VerifyErrorKind::RegionTypestateMismatch {
                    expected_owned: ctrl.exit_region_owned,
                    expected_leased: ctrl.exit_region_leased,
                    actual_owned: snap.region_owned,
                    actual_leased: snap.region_leased,
                },
            });
        }
        return Ok(());
    }

    if !ctrl.has_exit_path {
        ctrl.has_exit_path = true;
        ctrl.exit_init = snap.local_init;
        ctrl.exit_region_owned = snap.region_owned;
        ctrl.exit_region_leased = snap.region_leased;
        ctrl.exit_arena_root = cur_root;
        return Ok(());
    }

    ctrl.exit_init &= snap.local_init;
    if ctrl.exit_region_owned != snap.region_owned || ctrl.exit_region_leased != snap.region_leased
    {
        return Err(VerifyDiagnostic {
            node_idx,
            kind: VerifyErrorKind::RegionTypestateMismatch {
                expected_owned: ctrl.exit_region_owned,
                expected_leased: ctrl.exit_region_leased,
                actual_owned: snap.region_owned,
                actual_leased: snap.region_leased,
            },
        });
    }

    if cur_root != ctrl.exit_arena_root {
        if let Some((arena_id, provided_version, expected_version)) =
            arena_state.find_first_diff(cur_root, ctrl.exit_arena_root, program)
        {
            return Err(VerifyDiagnostic {
                node_idx,
                kind: VerifyErrorKind::StaleOrConsumedToken {
                    arena_id,
                    provided_version,
                    expected_version,
                },
            });
        }
    }
    Ok(())
}

fn seal_scope_nodes(scope_nodes: &mut Vec<usize>, node_visible: &mut [bool]) {
    for inner_idx in scope_nodes.drain(..) {
        if let Some(vis) = node_visible.get_mut(inner_idx) {
            *vis = false;
        }
    }
}

fn unify_elementwise_shapes(
    node_idx: u32,
    left: ValueShape,
    right: ValueShape,
) -> Result<ValueShape, VerifyDiagnostic> {
    match (left, right) {
        (ValueShape::Scalar, other) | (other, ValueShape::Scalar) => Ok(other),
        (ValueShape::Column(a), ValueShape::Column(b)) if a == b => Ok(ValueShape::Column(a)),
        (
            ValueShape::Compacted {
                max_len: a,
                mask_node: ma,
            },
            ValueShape::Compacted {
                max_len: b,
                mask_node: mb,
            },
        ) if a == b && ma == mb => Ok(left),
        _ => Err(VerifyDiagnostic {
            node_idx,
            kind: VerifyErrorKind::ShapeMismatch { left, right },
        }),
    }
}

fn lookup_region_decl(
    node_idx: u32,
    slot: u32,
    regions: &[Option<RegionDeclInfo>; MAX_REGION_SLOTS],
) -> Result<RegionDeclInfo, VerifyDiagnostic> {
    let slot_idx = usize::try_from(slot).unwrap_or(usize::MAX);
    regions
        .get(slot_idx)
        .and_then(|opt| *opt)
        .ok_or(VerifyDiagnostic {
            node_idx,
            kind: VerifyErrorKind::UnknownRegion(slot),
        })
}

fn verify_region_index_proof(
    program: &RulProgram,
    node_idx: u32,
    proof_idx: u8,
    decl: RegionDeclInfo,
    idx_meta: NodeMeta,
) -> Result<(), VerifyDiagnostic> {
    let err = |bound_hi: u32| VerifyDiagnostic {
        node_idx,
        kind: VerifyErrorKind::InvalidRefinementProof {
            proof_idx,
            bound_hi,
            arena_len: decl.min_len,
        },
    };
    let proof = program
        .proofs
        .get(usize::from(proof_idx.saturating_sub(1)))
        .ok_or_else(|| err(0))?;
    if proof.rule_kind != ProofRuleKind::RegionIndexInBounds
        || proof.target_arena != decl.slot
        || proof.bound_lo != 0
        || proof.witness_node != 0
        || proof.bound_hi == 0
        || proof.bound_hi > decl.min_len
    {
        return Err(err(proof.bound_hi));
    }
    let Some(upper) = idx_meta.nonneg_upper_bound else {
        return Err(err(proof.bound_hi));
    };
    if upper >= u64::from(proof.bound_hi) {
        return Err(err(proof.bound_hi));
    }
    Ok(())
}

fn verify_const_data_index_proof(
    program: &RulProgram,
    node_idx: u32,
    proof_idx: u8,
    seg: ConstDataSegInfo,
    elem_bytes: u64,
    idx_meta: NodeMeta,
) -> Result<(), VerifyDiagnostic> {
    let err = |bound_hi: u32| VerifyDiagnostic {
        node_idx,
        kind: VerifyErrorKind::InvalidRefinementProof {
            proof_idx,
            bound_hi,
            arena_len: seg.len,
        },
    };
    let proof = program
        .proofs
        .get(usize::from(proof_idx.saturating_sub(1)))
        .ok_or_else(|| err(0))?;
    if proof.rule_kind != ProofRuleKind::ConstDataInBounds
        || proof.target_arena != seg.seg_id
        || proof.bound_lo != 0
        || proof.witness_node != 0
        || proof.bound_hi == 0
        || proof.bound_hi > seg.len
    {
        return Err(err(proof.bound_hi));
    }
    let Some(upper) = idx_meta.nonneg_upper_bound else {
        return Err(err(proof.bound_hi));
    };
    let Some(end_byte) = upper.checked_add(elem_bytes) else {
        return Err(err(proof.bound_hi));
    };
    if upper >= u64::from(proof.bound_hi) || end_byte > u64::from(proof.bound_hi) {
        return Err(err(proof.bound_hi));
    }
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "single-pass O(N) opcode semantic verifier"
)]
fn verify_node_semantics(
    program: &RulProgram,
    frame: &NodeFrame,
    node_idx: u32,
    metas: &[NodeMeta],
    state: &mut VerifierState,
) -> Result<NodeMeta, VerifyDiagnostic> {
    match frame.opcode {
        Opcode::ArenaInput => {
            let arena = program
                .arenas
                .get(frame.in_a as usize)
                .ok_or(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::UnknownArena(frame.in_a),
                })?;
            let expected_id = u16::try_from(frame.in_a).unwrap_or(u16::MAX);
            if frame.token_op.mode == TokenMode::PureValue || frame.token_op.arena_id != expected_id
            {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::MissingArenaToken(expected_id),
                });
            }
            if arena.type_lane.elem == ElemType::U8 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::F64,
                        actual: ElemType::U8,
                    },
                });
            }
            Ok(NodeMeta::value(
                ValueShape::Column(arena.element_count),
                arena.type_lane.elem,
                Some(expected_id),
            ))
        }
        Opcode::ConstF64 => Ok(NodeMeta::value(ValueShape::Scalar, ElemType::F64, None)),
        Opcode::ConstI64 => {
            require_pure_token(node_idx, frame)?;
            if frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow("ConstI64 requires proof_idx == 0"),
                });
            }
            if frame.type_lane.elem != ElemType::I64 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::I64,
                        actual: frame.type_lane.elem,
                    },
                });
            }
            if frame.type_lane.width != SimdWidth::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: ValueShape::Column(0),
                    },
                });
            }
            let c = frame.decode_const_i64();
            Ok(NodeMeta {
                shape: ValueShape::Scalar,
                elem: ElemType::I64,
                origin_arena: None,
                is_void: false,
                const_i64: Some(c),
                nonneg_upper_bound: u64::try_from(c).ok(),
            })
        }
        Opcode::LocalGet => {
            require_pure_token(node_idx, frame)?;
            if frame.in_b != 0 || frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "LocalGet requires in_b == 0 and proof_idx == 0",
                    ),
                });
            }
            let slot = frame.in_a;
            let slot_idx = slot as usize;
            let slot_type_opt =
                state
                    .local_types
                    .get(slot_idx)
                    .copied()
                    .ok_or(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::LocalSlotOutOfRange { slot },
                    })?;
            if (state.local_init & (1_u64 << slot)) == 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::UninitializedLocalRead { slot },
                });
            }
            let elem = slot_type_opt.ok_or(VerifyDiagnostic {
                node_idx,
                kind: VerifyErrorKind::UninitializedLocalRead { slot },
            })?;
            if frame.type_lane.elem != elem {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: elem,
                        actual: frame.type_lane.elem,
                    },
                });
            }
            if frame.type_lane.width != SimdWidth::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: ValueShape::Column(0),
                    },
                });
            }
            Ok(NodeMeta::value(ValueShape::Scalar, elem, None))
        }
        Opcode::LocalSet => {
            require_pure_token(node_idx, frame)?;
            if frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow("LocalSet requires proof_idx == 0"),
                });
            }
            let slot = frame.in_a;
            let val_meta = get_prior_meta(node_idx, frame.in_b, metas, state)?;
            if val_meta.shape != ValueShape::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: val_meta.shape,
                    },
                });
            }
            if frame.type_lane.elem != val_meta.elem {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: val_meta.elem,
                        actual: frame.type_lane.elem,
                    },
                });
            }
            if frame.type_lane.width != SimdWidth::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: ValueShape::Column(0),
                    },
                });
            }
            let slot_entry = state
                .local_types
                .get_mut(slot as usize)
                .ok_or(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::LocalSlotOutOfRange { slot },
                })?;
            if let Some(existing_elem) = *slot_entry {
                if existing_elem != val_meta.elem {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::LocalTypeMismatch {
                            slot,
                            expected: existing_elem,
                            actual: val_meta.elem,
                        },
                    });
                }
            } else {
                *slot_entry = Some(val_meta.elem);
            }
            state.local_init |= 1_u64 << slot;
            Ok(NodeMeta::value(ValueShape::Scalar, val_meta.elem, None))
        }
        Opcode::ArenaSlice => {
            let src_meta = get_prior_meta(node_idx, frame.in_a, metas, state)?;
            if frame.proof_idx == 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::MissingRefinementProof,
                });
            }
            let proof =
                program
                    .proofs
                    .get(usize::from(frame.proof_idx - 1))
                    .ok_or(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::MissingRefinementProof,
                    })?;
            if proof.rule_kind != ProofRuleKind::SliceBounds || proof.bound_lo > proof.bound_hi {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::MissingRefinementProof,
                });
            }
            // Enforce strict binding between ProofEntry.target_arena and the operand's origin Arena!
            if src_meta.origin_arena != Some(proof.target_arena) {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ProofTargetArenaMismatch {
                        proof_target: proof.target_arena,
                        actual_arena: src_meta.origin_arena,
                    },
                });
            }
            let target_arena =
                program
                    .arenas
                    .get(usize::from(proof.target_arena))
                    .ok_or(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::UnknownArena(u32::from(proof.target_arena)),
                    })?;
            let effective_src_len = match src_meta.shape {
                ValueShape::Column(n) => n.min(target_arena.element_count),
                _ => target_arena.element_count,
            };
            if proof.bound_hi > effective_src_len {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidRefinementProof {
                        proof_idx: frame.proof_idx,
                        bound_hi: proof.bound_hi,
                        arena_len: effective_src_len,
                    },
                });
            }
            let slice_len = proof.bound_hi - proof.bound_lo;
            Ok(NodeMeta::value(
                ValueShape::Column(slice_len),
                src_meta.elem,
                src_meta.origin_arena,
            ))
        }
        Opcode::ArenaMutate => {
            let arena = program
                .arenas
                .get(frame.in_a as usize)
                .ok_or(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::UnknownArena(frame.in_a),
                })?;
            let expected_id = u16::try_from(frame.in_a).unwrap_or(u16::MAX);
            if frame.token_op.mode != TokenMode::AffineConsume
                || frame.token_op.arena_id != expected_id
            {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::MissingArenaToken(expected_id),
                });
            }
            let val_meta = get_prior_meta(node_idx, frame.in_b, metas, state)?;
            if val_meta.elem != arena.type_lane.elem {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: arena.type_lane.elem,
                        actual: val_meta.elem,
                    },
                });
            }
            let unified = unify_elementwise_shapes(
                node_idx,
                ValueShape::Column(arena.element_count),
                val_meta.shape,
            )?;
            Ok(NodeMeta::value(unified, val_meta.elem, Some(expected_id)))
        }
        Opcode::Add | Opcode::Sub | Opcode::Mul | Opcode::Div => {
            let meta_a = get_prior_meta(node_idx, frame.in_a, metas, state)?;
            let meta_b = get_prior_meta(node_idx, frame.in_b, metas, state)?;
            if matches!(meta_a.elem, ElemType::I64 | ElemType::U8) && meta_a.elem == meta_b.elem {
                let int_elem = meta_a.elem;
                if meta_a.shape != ValueShape::Scalar || meta_b.shape != ValueShape::Scalar {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::ShapeMismatch {
                            left: meta_a.shape,
                            right: if meta_a.shape != ValueShape::Scalar {
                                ValueShape::Scalar
                            } else {
                                meta_b.shape
                            },
                        },
                    });
                }
                if frame.type_lane.elem != int_elem {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::ElementTypeMismatch {
                            expected: int_elem,
                            actual: frame.type_lane.elem,
                        },
                    });
                }
                if frame.type_lane.width != SimdWidth::Scalar {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::ShapeMismatch {
                            left: ValueShape::Scalar,
                            right: ValueShape::Column(0),
                        },
                    });
                }
                return Ok(NodeMeta::value(ValueShape::Scalar, int_elem, None));
            }
            if meta_a.elem != ElemType::F64 || meta_b.elem != ElemType::F64 {
                let (expected, actual) = if matches!(meta_a.elem, ElemType::I64 | ElemType::U8) {
                    (meta_a.elem, meta_b.elem)
                } else {
                    (
                        ElemType::F64,
                        if meta_a.elem != ElemType::F64 {
                            meta_a.elem
                        } else {
                            meta_b.elem
                        },
                    )
                };
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch { expected, actual },
                });
            }
            let shape = unify_elementwise_shapes(node_idx, meta_a.shape, meta_b.shape)?;
            Ok(NodeMeta::value(shape, ElemType::F64, None))
        }
        Opcode::Min | Opcode::Max => {
            let meta_a = get_prior_meta(node_idx, frame.in_a, metas, state)?;
            let meta_b = get_prior_meta(node_idx, frame.in_b, metas, state)?;
            if meta_a.elem != ElemType::F64 || meta_b.elem != ElemType::F64 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::F64,
                        actual: if meta_a.elem != ElemType::F64 {
                            meta_a.elem
                        } else {
                            meta_b.elem
                        },
                    },
                });
            }
            let shape = unify_elementwise_shapes(node_idx, meta_a.shape, meta_b.shape)?;
            Ok(NodeMeta::value(shape, ElemType::F64, None))
        }
        Opcode::Fma => {
            let (in_b, in_c) = frame.decode_fma_bc();
            let meta_a = get_prior_meta(node_idx, frame.in_a, metas, state)?;
            let meta_b = get_prior_meta(node_idx, in_b, metas, state)?;
            let meta_c = get_prior_meta(node_idx, in_c, metas, state)?;
            for m in [meta_a, meta_b, meta_c] {
                if m.elem != ElemType::F64 {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::ElementTypeMismatch {
                            expected: ElemType::F64,
                            actual: m.elem,
                        },
                    });
                }
            }
            let shape_ab = unify_elementwise_shapes(node_idx, meta_a.shape, meta_b.shape)?;
            let shape = unify_elementwise_shapes(node_idx, shape_ab, meta_c.shape)?;
            Ok(NodeMeta::value(shape, ElemType::F64, None))
        }
        Opcode::Abs | Opcode::Sqrt | Opcode::Floor | Opcode::Ceil | Opcode::Ln | Opcode::Exp => {
            let meta_a = get_prior_meta(node_idx, frame.in_a, metas, state)?;
            if meta_a.elem != ElemType::F64 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::F64,
                        actual: meta_a.elem,
                    },
                });
            }
            Ok(NodeMeta::value(meta_a.shape, ElemType::F64, None))
        }
        Opcode::CmpGt | Opcode::CmpLt | Opcode::CmpEq => {
            let meta_a = get_prior_meta(node_idx, frame.in_a, metas, state)?;
            let meta_b = get_prior_meta(node_idx, frame.in_b, metas, state)?;
            if matches!(meta_a.elem, ElemType::I64 | ElemType::U8) && meta_a.elem == meta_b.elem {
                if meta_a.shape != ValueShape::Scalar || meta_b.shape != ValueShape::Scalar {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::ShapeMismatch {
                            left: meta_a.shape,
                            right: if meta_a.shape != ValueShape::Scalar {
                                ValueShape::Scalar
                            } else {
                                meta_b.shape
                            },
                        },
                    });
                }
                if frame.type_lane.elem != ElemType::BoolMask {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::ElementTypeMismatch {
                            expected: ElemType::BoolMask,
                            actual: frame.type_lane.elem,
                        },
                    });
                }
                if frame.type_lane.width != SimdWidth::Scalar {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::ShapeMismatch {
                            left: ValueShape::Scalar,
                            right: ValueShape::Column(0),
                        },
                    });
                }
                return Ok(NodeMeta::value(
                    ValueShape::Scalar,
                    ElemType::BoolMask,
                    None,
                ));
            }
            if meta_a.elem != ElemType::F64 || meta_b.elem != ElemType::F64 {
                let (expected, actual) = if matches!(meta_a.elem, ElemType::I64 | ElemType::U8) {
                    (meta_a.elem, meta_b.elem)
                } else {
                    (
                        ElemType::F64,
                        if meta_a.elem != ElemType::F64 {
                            meta_a.elem
                        } else {
                            meta_b.elem
                        },
                    )
                };
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch { expected, actual },
                });
            }
            let shape = unify_elementwise_shapes(node_idx, meta_a.shape, meta_b.shape)?;
            Ok(NodeMeta::value(shape, ElemType::BoolMask, None))
        }
        Opcode::CmpLtU => {
            require_pure_token(node_idx, frame)?;
            if frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow("CmpLtU requires proof_idx == 0"),
                });
            }
            let meta_a = get_prior_meta(node_idx, frame.in_a, metas, state)?;
            let meta_b = get_prior_meta(node_idx, frame.in_b, metas, state)?;
            if meta_a.elem != ElemType::I64 || meta_b.elem != ElemType::I64 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::I64,
                        actual: if meta_a.elem != ElemType::I64 {
                            meta_a.elem
                        } else {
                            meta_b.elem
                        },
                    },
                });
            }
            if meta_a.shape != ValueShape::Scalar || meta_b.shape != ValueShape::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: meta_a.shape,
                        right: if meta_a.shape != ValueShape::Scalar {
                            ValueShape::Scalar
                        } else {
                            meta_b.shape
                        },
                    },
                });
            }
            if frame.type_lane.elem != ElemType::BoolMask {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::BoolMask,
                        actual: frame.type_lane.elem,
                    },
                });
            }
            if frame.type_lane.width != SimdWidth::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: ValueShape::Column(0),
                    },
                });
            }
            Ok(NodeMeta::value(
                ValueShape::Scalar,
                ElemType::BoolMask,
                None,
            ))
        }
        Opcode::Rem
        | Opcode::BitAnd
        | Opcode::BitOr
        | Opcode::BitXor
        | Opcode::Shl
        | Opcode::ShrU => {
            require_pure_token(node_idx, frame)?;
            if frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "integer bitwise/shift/rem opcode requires proof_idx == 0",
                    ),
                });
            }
            let meta_a = get_prior_meta(node_idx, frame.in_a, metas, state)?;
            let meta_b = get_prior_meta(node_idx, frame.in_b, metas, state)?;
            if !matches!(meta_a.elem, ElemType::I64 | ElemType::U8) || meta_a.elem != meta_b.elem {
                let (expected, actual) = if matches!(meta_a.elem, ElemType::I64 | ElemType::U8) {
                    (meta_a.elem, meta_b.elem)
                } else {
                    (ElemType::I64, meta_a.elem)
                };
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch { expected, actual },
                });
            }
            let int_elem = meta_a.elem;
            if meta_a.shape != ValueShape::Scalar || meta_b.shape != ValueShape::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: meta_a.shape,
                        right: if meta_a.shape != ValueShape::Scalar {
                            ValueShape::Scalar
                        } else {
                            meta_b.shape
                        },
                    },
                });
            }
            if frame.type_lane.elem != int_elem {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: int_elem,
                        actual: frame.type_lane.elem,
                    },
                });
            }
            if frame.type_lane.width != SimdWidth::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: ValueShape::Column(0),
                    },
                });
            }
            let nonneg_upper_bound = if frame.opcode == Opcode::BitAnd && int_elem == ElemType::I64
            {
                match (
                    meta_a.const_i64.and_then(|v| u64::try_from(v).ok()),
                    meta_b.const_i64.and_then(|v| u64::try_from(v).ok()),
                ) {
                    (Some(x), Some(y)) => Some(x.min(y)),
                    (Some(x), None) => Some(x),
                    (None, Some(y)) => Some(y),
                    (None, None) => None,
                }
            } else {
                None
            };
            Ok(NodeMeta {
                shape: ValueShape::Scalar,
                elem: int_elem,
                origin_arena: None,
                is_void: false,
                const_i64: None,
                nonneg_upper_bound,
            })
        }
        Opcode::ShrS => {
            require_pure_token(node_idx, frame)?;
            if frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow("ShrS requires proof_idx == 0"),
                });
            }
            let meta_a = get_prior_meta(node_idx, frame.in_a, metas, state)?;
            let meta_b = get_prior_meta(node_idx, frame.in_b, metas, state)?;
            if meta_a.elem != ElemType::I64 || meta_b.elem != ElemType::I64 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::I64,
                        actual: if meta_a.elem != ElemType::I64 {
                            meta_a.elem
                        } else {
                            meta_b.elem
                        },
                    },
                });
            }
            if meta_a.shape != ValueShape::Scalar || meta_b.shape != ValueShape::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: meta_a.shape,
                        right: if meta_a.shape != ValueShape::Scalar {
                            ValueShape::Scalar
                        } else {
                            meta_b.shape
                        },
                    },
                });
            }
            if frame.type_lane.elem != ElemType::I64 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::I64,
                        actual: frame.type_lane.elem,
                    },
                });
            }
            if frame.type_lane.width != SimdWidth::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: ValueShape::Column(0),
                    },
                });
            }
            Ok(NodeMeta::value(ValueShape::Scalar, ElemType::I64, None))
        }
        Opcode::MaskNot => {
            require_pure_token(node_idx, frame)?;
            if frame.in_b != 0 || frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "MaskNot requires in_b == 0 and proof_idx == 0",
                    ),
                });
            }
            let meta_a = get_prior_meta(node_idx, frame.in_a, metas, state)?;
            if meta_a.elem != ElemType::BoolMask {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::BoolMask,
                        actual: meta_a.elem,
                    },
                });
            }
            if meta_a.shape != ValueShape::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: meta_a.shape,
                    },
                });
            }
            if frame.type_lane.elem != ElemType::BoolMask {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::BoolMask,
                        actual: frame.type_lane.elem,
                    },
                });
            }
            if frame.type_lane.width != SimdWidth::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: ValueShape::Column(0),
                    },
                });
            }
            Ok(NodeMeta::value(
                ValueShape::Scalar,
                ElemType::BoolMask,
                None,
            ))
        }
        Opcode::Cast => {
            require_pure_token(node_idx, frame)?;
            if frame.in_b != 0 || frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "Cast requires in_b == 0 and proof_idx == 0",
                    ),
                });
            }
            let src_meta = get_prior_meta(node_idx, frame.in_a, metas, state)?;
            if src_meta.shape != ValueShape::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: src_meta.shape,
                    },
                });
            }
            if frame.type_lane.width != SimdWidth::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: ValueShape::Column(0),
                    },
                });
            }
            let dst_elem = frame.type_lane.elem;
            let valid_pair = matches!(
                (src_meta.elem, dst_elem),
                (ElemType::I64, ElemType::U8)
                    | (ElemType::U8, ElemType::I64)
                    | (ElemType::F64, ElemType::I64)
                    | (ElemType::F64, ElemType::U8)
                    | (ElemType::I64, ElemType::F64)
                    | (ElemType::U8, ElemType::F64)
                    | (ElemType::BoolMask, ElemType::I64)
            );
            if !valid_pair {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: dst_elem,
                        actual: src_meta.elem,
                    },
                });
            }
            Ok(NodeMeta::value(ValueShape::Scalar, dst_elem, None))
        }
        Opcode::MaskAnd | Opcode::MaskOr => {
            let meta_a = get_prior_meta(node_idx, frame.in_a, metas, state)?;
            let meta_b = get_prior_meta(node_idx, frame.in_b, metas, state)?;
            if meta_a.elem != ElemType::BoolMask || meta_b.elem != ElemType::BoolMask {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::BoolMask,
                        actual: if meta_a.elem != ElemType::BoolMask {
                            meta_a.elem
                        } else {
                            meta_b.elem
                        },
                    },
                });
            }
            let shape = unify_elementwise_shapes(node_idx, meta_a.shape, meta_b.shape)?;
            Ok(NodeMeta::value(shape, ElemType::BoolMask, None))
        }
        Opcode::SelectZero => {
            let mask_meta = get_prior_meta(node_idx, frame.in_a, metas, state)?;
            let val_meta = get_prior_meta(node_idx, frame.in_b, metas, state)?;
            if mask_meta.elem != ElemType::BoolMask {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::BoolMask,
                        actual: mask_meta.elem,
                    },
                });
            }
            let shape = unify_elementwise_shapes(node_idx, mask_meta.shape, val_meta.shape)?;
            Ok(NodeMeta::value(shape, val_meta.elem, None))
        }
        Opcode::FilterCompact => {
            let mask_meta = get_prior_meta(node_idx, frame.in_a, metas, state)?;
            let val_meta = get_prior_meta(node_idx, frame.in_b, metas, state)?;
            if mask_meta.elem != ElemType::BoolMask {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::BoolMask,
                        actual: mask_meta.elem,
                    },
                });
            }
            let unified = unify_elementwise_shapes(node_idx, mask_meta.shape, val_meta.shape)?;
            let max_len = match unified {
                ValueShape::Column(n) | ValueShape::Compacted { max_len: n, .. } => n,
                ValueShape::Scalar => 1,
            };
            Ok(NodeMeta::value(
                ValueShape::Compacted {
                    max_len,
                    mask_node: frame.in_a,
                },
                val_meta.elem,
                None,
            ))
        }
        Opcode::ReduceSum | Opcode::ReduceMax | Opcode::ReduceMin => {
            let src_meta = get_prior_meta(node_idx, frame.in_a, metas, state)?;
            if src_meta.elem != ElemType::F64 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::F64,
                        actual: src_meta.elem,
                    },
                });
            }
            Ok(NodeMeta::value(ValueShape::Scalar, ElemType::F64, None))
        }
        Opcode::Gather => {
            let target_meta = get_prior_meta(node_idx, frame.in_a, metas, state)?;
            let idx_meta = get_prior_meta(node_idx, frame.in_b, metas, state)?;
            if frame.proof_idx == 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::MissingRefinementProof,
                });
            }
            let proof =
                program
                    .proofs
                    .get(usize::from(frame.proof_idx - 1))
                    .ok_or(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::MissingRefinementProof,
                    })?;
            if proof.rule_kind != ProofRuleKind::GatherIndexInBounds || proof.bound_hi == 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::MissingRefinementProof,
                });
            }
            // 1. Enforce that ProofEntry.target_arena matches the actual origin arena of `in_a`!
            if target_meta.origin_arena != Some(proof.target_arena) {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ProofTargetArenaMismatch {
                        proof_target: proof.target_arena,
                        actual_arena: target_meta.origin_arena,
                    },
                });
            }
            // 2. Enforce that ProofEntry.witness_node is bound to the exact index node (`in_b`)!
            if proof.witness_node != frame.in_b {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ProofWitnessMismatch {
                        proof_witness: proof.witness_node,
                        index_node: frame.in_b,
                    },
                });
            }
            let target_arena =
                program
                    .arenas
                    .get(usize::from(proof.target_arena))
                    .ok_or(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::UnknownArena(u32::from(proof.target_arena)),
                    })?;
            let effective_target_len = match target_meta.shape {
                ValueShape::Column(n) => n.min(target_arena.element_count),
                _ => target_arena.element_count,
            };
            if proof.bound_hi > effective_target_len {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidRefinementProof {
                        proof_idx: frame.proof_idx,
                        bound_hi: proof.bound_hi,
                        arena_len: effective_target_len,
                    },
                });
            }
            Ok(NodeMeta::value(idx_meta.shape, target_meta.elem, None))
        }
        Opcode::RingEnqueue => {
            if (program.capability_mask & CAP_NET_RING) == 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::CapabilityViolation {
                        required: CAP_NET_RING,
                        granted: program.capability_mask,
                    },
                });
            }
            let src_meta = get_prior_meta(node_idx, frame.in_a, metas, state)?;
            Ok(NodeMeta::value(
                src_meta.shape,
                src_meta.elem,
                src_meta.origin_arena,
            ))
        }
        Opcode::AssertContract => {
            let src_meta = get_prior_meta(node_idx, frame.in_a, metas, state)?;
            Ok(NodeMeta::value(
                src_meta.shape,
                src_meta.elem,
                src_meta.origin_arena,
            ))
        }
        Opcode::Block => {
            require_pure_token(node_idx, frame)?;
            if frame.in_a != 0 || frame.in_b != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "Block header requires in_a == 0 and in_b == 0",
                    ),
                });
            }
            let sig = parse_block_sig(
                program,
                frame,
                node_idx,
                &state.arena_state,
                state.reachable,
            )?;
            let entry_arena_root = state.arena_state.commit_root(program);
            if state.reachable {
                state.cur_steps = state.cur_steps.saturating_add(1);
            }
            let parent_steps = state.cur_steps;
            state.cur_steps = 0;
            state.control_stack.push(ControlFrame {
                kind: ControlKind::Block,
                header_node: node_idx,
                sig,
                entry_init: state.local_init,
                entry_arena_root,
                entry_reachable: state.reachable,
                entry_region_owned: state.region_owned,
                entry_region_leased: state.region_leased,
                has_branch_or_join: false,
                has_exit_path: false,
                exit_init: state.local_init,
                exit_region_owned: state.region_owned,
                exit_region_leased: state.region_leased,
                exit_arena_root: entry_arena_root,
                parent_steps,
                then_steps: 0,
                loop_bound: 0,
                scope_nodes: Vec::new(),
            });
            Ok(NodeMeta::VOID)
        }
        Opcode::Loop => {
            require_pure_token(node_idx, frame)?;
            if frame.in_a != 0 || frame.in_b != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "Loop header requires in_a == 0 and in_b == 0",
                    ),
                });
            }
            let loop_bound = if frame.proof_idx == 0 {
                if (program.capability_mask & CAP_DIVERGE) == 0 {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::MissingCapability {
                            required: CAP_DIVERGE,
                            actual: program.capability_mask,
                        },
                    });
                }
                0
            } else {
                let proof = program.proofs.get(usize::from(frame.proof_idx - 1)).ok_or(
                    VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::InvalidRefinementProof {
                            proof_idx: frame.proof_idx,
                            bound_hi: 0,
                            arena_len: MAX_SECTION_ITEMS,
                        },
                    },
                )?;
                if proof.rule_kind != ProofRuleKind::LoopBound
                    || proof.target_arena != 0
                    || proof.bound_lo != 0
                    || proof.bound_hi == 0
                    || proof.bound_hi > MAX_SECTION_ITEMS
                    || proof.witness_node != 0
                {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::InvalidRefinementProof {
                            proof_idx: frame.proof_idx,
                            bound_hi: proof.bound_hi,
                            arena_len: MAX_SECTION_ITEMS,
                        },
                    });
                }
                proof.bound_hi
            };
            let entry_arena_root = state.arena_state.commit_root(program);
            if state.reachable {
                state.cur_steps = state.cur_steps.saturating_add(1);
            }
            let parent_steps = state.cur_steps;
            state.cur_steps = 0;
            state.control_stack.push(ControlFrame {
                kind: ControlKind::Loop,
                header_node: node_idx,
                sig: None,
                entry_init: state.local_init,
                entry_arena_root,
                entry_reachable: state.reachable,
                entry_region_owned: state.region_owned,
                entry_region_leased: state.region_leased,
                has_branch_or_join: false,
                has_exit_path: false,
                exit_init: state.local_init,
                exit_region_owned: state.region_owned,
                exit_region_leased: state.region_leased,
                exit_arena_root: entry_arena_root,
                parent_steps,
                then_steps: 0,
                loop_bound,
                scope_nodes: Vec::new(),
            });
            Ok(NodeMeta::VOID)
        }
        Opcode::If => {
            require_pure_token(node_idx, frame)?;
            if frame.in_b != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow("If header requires in_b == 0"),
                });
            }
            let cond_meta = get_prior_meta(node_idx, frame.in_a, metas, state)?;
            if cond_meta.elem != ElemType::BoolMask {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::BoolMask,
                        actual: cond_meta.elem,
                    },
                });
            }
            if cond_meta.shape != ValueShape::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: cond_meta.shape,
                    },
                });
            }
            let sig = parse_block_sig(
                program,
                frame,
                node_idx,
                &state.arena_state,
                state.reachable,
            )?;
            let entry_arena_root = state.arena_state.commit_root(program);
            if state.reachable {
                state.cur_steps = state.cur_steps.saturating_add(1);
            }
            let parent_steps = state.cur_steps;
            state.cur_steps = 0;
            state.control_stack.push(ControlFrame {
                kind: ControlKind::If { has_else: false },
                header_node: node_idx,
                sig,
                entry_init: state.local_init,
                entry_arena_root,
                entry_reachable: state.reachable,
                entry_region_owned: state.region_owned,
                entry_region_leased: state.region_leased,
                has_branch_or_join: true,
                has_exit_path: false,
                exit_init: state.local_init,
                exit_region_owned: state.region_owned,
                exit_region_leased: state.region_leased,
                exit_arena_root: entry_arena_root,
                parent_steps,
                then_steps: 0,
                loop_bound: 0,
                scope_nodes: Vec::new(),
            });
            Ok(NodeMeta::VOID)
        }
        Opcode::Else => {
            require_pure_token(node_idx, frame)?;
            if frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow("Else requires proof_idx == 0"),
                });
            }
            let result_tl_opt = {
                let top = state.control_stack.last().ok_or(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow("Else without matching If"),
                })?;
                match top.kind {
                    ControlKind::If { has_else: false } => top.sig.and_then(|s| s.result_type),
                    ControlKind::If { has_else: true } => {
                        return Err(VerifyDiagnostic {
                            node_idx,
                            kind: VerifyErrorKind::InvalidControlFlow("duplicate Else in If block"),
                        });
                    }
                    ControlKind::Block | ControlKind::Loop | ControlKind::Func { .. } => {
                        return Err(VerifyDiagnostic {
                            node_idx,
                            kind: VerifyErrorKind::InvalidControlFlow(
                                "Else is only valid inside an If block",
                            ),
                        });
                    }
                }
            };

            if let Some(res_tl) = result_tl_opt {
                if frame.in_b != 1 {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::InvalidControlFlow(
                            "Else for value-returning If requires in_b == 1 and in_a == then_result",
                        ),
                    });
                }
                let then_meta = get_prior_meta(node_idx, frame.in_a, metas, state)?;
                if then_meta.elem != res_tl.elem {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::ElementTypeMismatch {
                            expected: res_tl.elem,
                            actual: then_meta.elem,
                        },
                    });
                }
                if then_meta.shape != ValueShape::Scalar {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::ShapeMismatch {
                            left: ValueShape::Scalar,
                            right: then_meta.shape,
                        },
                    });
                }
            } else if frame.in_a != 0 || frame.in_b != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "void Else requires in_a == 0 and in_b == 0",
                    ),
                });
            }

            let (
                entry_init,
                entry_arena_root,
                entry_reachable,
                entry_region_owned,
                entry_region_leased,
            ) = {
                let VerifierState {
                    ref mut control_stack,
                    ref mut node_visible,
                    ref mut arena_state,
                    local_init,
                    region_owned,
                    region_leased,
                    reachable,
                    cur_steps,
                    ..
                } = *state;
                let top = control_stack.last_mut().ok_or(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow("Else without matching If"),
                })?;
                seal_scope_nodes(&mut top.scope_nodes, node_visible);
                if reachable {
                    join_control_exit(
                        program,
                        top,
                        node_idx,
                        ExitSnapshot {
                            local_init,
                            region_owned,
                            region_leased,
                        },
                        arena_state,
                    )?;
                }
                top.then_steps = cur_steps;
                top.kind = ControlKind::If { has_else: true };
                (
                    top.entry_init,
                    top.entry_arena_root,
                    top.entry_reachable,
                    top.entry_region_owned,
                    top.entry_region_leased,
                )
            };
            state.local_init = entry_init;
            state.region_owned = entry_region_owned;
            state.region_leased = entry_region_leased;
            state.cur_steps = 0;
            state.arena_state.restore_root(entry_arena_root);
            state.reachable = entry_reachable;
            Ok(NodeMeta::VOID)
        }
        Opcode::End => {
            require_pure_token(node_idx, frame)?;
            if frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow("End requires proof_idx == 0"),
                });
            }
            let (kind, result_tl_opt) = {
                let top = state.control_stack.last().ok_or(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow("End without matching open block"),
                })?;
                (top.kind, top.sig.and_then(|s| s.result_type))
            };

            let out_meta = if let ControlKind::Func { sig, .. } = kind {
                if let Some(ret_elem) = sig.ret_elem {
                    if frame.in_b == 1 {
                        if frame.type_lane.elem != ret_elem {
                            return Err(VerifyDiagnostic {
                                node_idx,
                                kind: VerifyErrorKind::ElementTypeMismatch {
                                    expected: ret_elem,
                                    actual: frame.type_lane.elem,
                                },
                            });
                        }
                        if frame.type_lane.width != SimdWidth::Scalar {
                            return Err(VerifyDiagnostic {
                                node_idx,
                                kind: VerifyErrorKind::ShapeMismatch {
                                    left: ValueShape::Scalar,
                                    right: ValueShape::Column(0),
                                },
                            });
                        }
                        let res_meta = get_prior_meta(node_idx, frame.in_a, metas, state)?;
                        if res_meta.elem != ret_elem {
                            return Err(VerifyDiagnostic {
                                node_idx,
                                kind: VerifyErrorKind::ElementTypeMismatch {
                                    expected: ret_elem,
                                    actual: res_meta.elem,
                                },
                            });
                        }
                        if res_meta.shape != ValueShape::Scalar {
                            return Err(VerifyDiagnostic {
                                node_idx,
                                kind: VerifyErrorKind::ShapeMismatch {
                                    left: ValueShape::Scalar,
                                    right: res_meta.shape,
                                },
                            });
                        }
                    } else if frame.in_a == 0 && frame.in_b == 0 {
                        if state.reachable {
                            return Err(VerifyDiagnostic {
                                node_idx,
                                kind: VerifyErrorKind::InvalidControlFlow(
                                    "non-void Func may fall through without returning a value",
                                ),
                            });
                        }
                    } else {
                        return Err(VerifyDiagnostic {
                            node_idx,
                            kind: VerifyErrorKind::InvalidControlFlow(
                                "End for non-void Func requires in_b == 1 and in_a == result, or unreachable void End",
                            ),
                        });
                    }
                } else if frame.in_a != 0 || frame.in_b != 0 {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::InvalidControlFlow(
                            "End for void Func requires in_a == 0 and in_b == 0",
                        ),
                    });
                }
                NodeMeta::VOID
            } else if let Some(res_tl) = result_tl_opt {
                if matches!(kind, ControlKind::If { has_else: false }) {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::InvalidControlFlow(
                            "value-returning If requires an Else branch",
                        ),
                    });
                }
                if frame.in_b != 1 {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::InvalidControlFlow(
                            "End for value-returning block requires in_b == 1 and in_a == result",
                        ),
                    });
                }
                if frame.type_lane != res_tl {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::ElementTypeMismatch {
                            expected: res_tl.elem,
                            actual: frame.type_lane.elem,
                        },
                    });
                }
                let res_meta = get_prior_meta(node_idx, frame.in_a, metas, state)?;
                if res_meta.elem != res_tl.elem {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::ElementTypeMismatch {
                            expected: res_tl.elem,
                            actual: res_meta.elem,
                        },
                    });
                }
                if res_meta.shape != ValueShape::Scalar {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::ShapeMismatch {
                            left: ValueShape::Scalar,
                            right: res_meta.shape,
                        },
                    });
                }
                NodeMeta::value(ValueShape::Scalar, res_tl.elem, None)
            } else {
                if frame.in_a != 0 || frame.in_b != 0 {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::InvalidControlFlow(
                            "void End requires in_a == 0 and in_b == 0",
                        ),
                    });
                }
                NodeMeta::VOID
            };

            let mut popped = state.control_stack.pop().ok_or(VerifyDiagnostic {
                node_idx,
                kind: VerifyErrorKind::InvalidControlFlow("End without matching open block"),
            })?;
            seal_scope_nodes(&mut popped.scope_nodes, &mut state.node_visible);

            match popped.kind {
                ControlKind::Block => {
                    let body_steps = state.cur_steps;
                    state.cur_steps = popped.parent_steps.saturating_add(body_steps);
                    let needs_join = popped.has_branch_or_join
                        || popped.sig.and_then(|s| s.arena_delta).is_some();
                    if !needs_join {
                        if !state.reachable {
                            state.arena_state.restore_root(popped.entry_arena_root);
                            state.local_init = popped.entry_init;
                            state.region_owned = popped.entry_region_owned;
                            state.region_leased = popped.entry_region_leased;
                        }
                    } else {
                        if state.reachable {
                            join_control_exit(
                                program,
                                &mut popped,
                                node_idx,
                                ExitSnapshot {
                                    local_init: state.local_init,
                                    region_owned: state.region_owned,
                                    region_leased: state.region_leased,
                                },
                                &mut state.arena_state,
                            )?;
                        }
                        if popped.has_exit_path {
                            state.reachable = popped.entry_reachable;
                            state.local_init = popped.exit_init;
                            state.region_owned = popped.exit_region_owned;
                            state.region_leased = popped.exit_region_leased;
                            if popped.entry_reachable {
                                state.arena_state.restore_root(popped.exit_arena_root);
                            } else {
                                state.arena_state.restore_root(popped.entry_arena_root);
                            }
                        } else {
                            state.reachable = false;
                            state.local_init = popped.entry_init;
                            state.region_owned = popped.entry_region_owned;
                            state.region_leased = popped.entry_region_leased;
                            state.arena_state.restore_root(popped.entry_arena_root);
                        }
                    }
                }
                ControlKind::If { has_else } => {
                    let branch_steps = if has_else {
                        popped.then_steps.max(state.cur_steps)
                    } else {
                        state.cur_steps
                    };
                    state.cur_steps = popped.parent_steps.saturating_add(branch_steps);
                    if state.reachable {
                        join_control_exit(
                            program,
                            &mut popped,
                            node_idx,
                            ExitSnapshot {
                                local_init: state.local_init,
                                region_owned: state.region_owned,
                                region_leased: state.region_leased,
                            },
                            &mut state.arena_state,
                        )?;
                    }
                    state.arena_state.restore_root(popped.entry_arena_root);
                    if !has_else && popped.entry_reachable {
                        let entry_snap = ExitSnapshot {
                            local_init: popped.entry_init,
                            region_owned: popped.entry_region_owned,
                            region_leased: popped.entry_region_leased,
                        };
                        join_control_exit(
                            program,
                            &mut popped,
                            node_idx,
                            entry_snap,
                            &mut state.arena_state,
                        )?;
                    }
                    if popped.has_exit_path {
                        state.reachable = popped.entry_reachable;
                        state.local_init = popped.exit_init;
                        state.region_owned = popped.exit_region_owned;
                        state.region_leased = popped.exit_region_leased;
                        if popped.entry_reachable {
                            state.arena_state.restore_root(popped.exit_arena_root);
                        }
                    } else {
                        state.reachable = false;
                        state.local_init = popped.entry_init;
                        state.region_owned = popped.entry_region_owned;
                        state.region_leased = popped.entry_region_leased;
                    }
                }
                ControlKind::Loop => {
                    let iters = u64::from(popped.loop_bound).saturating_add(1);
                    let loop_steps = state.cur_steps.saturating_mul(iters);
                    state.cur_steps = popped.parent_steps.saturating_add(loop_steps);
                    if state.reachable {
                        check_loop_arena_invariant(
                            program,
                            node_idx,
                            &popped,
                            &mut state.arena_state,
                        )?;
                        check_loop_region_invariant(
                            node_idx,
                            &popped,
                            state.region_owned,
                            state.region_leased,
                        )?;
                    } else {
                        state.arena_state.restore_root(popped.entry_arena_root);
                        state.local_init = popped.entry_init;
                        state.region_owned = popped.entry_region_owned;
                        state.region_leased = popped.entry_region_leased;
                    }
                }
                ControlKind::Func { sig, start_node } => {
                    if state.reachable {
                        if let Some(contract) =
                            state.func_contracts.get(usize::from(sig.func_id)).copied()
                        {
                            if state.region_owned != contract.post_owned
                                || state.region_leased != contract.post_leased
                            {
                                return Err(VerifyDiagnostic {
                                    node_idx,
                                    kind: VerifyErrorKind::RegionTypestateMismatch {
                                        expected_owned: contract.post_owned,
                                        expected_leased: contract.post_leased,
                                        actual_owned: state.region_owned,
                                        actual_leased: state.region_leased,
                                    },
                                });
                            }
                        }
                    }
                    let func_node_count = (node_idx - start_node) as usize + 1;
                    let frame_units = func_frame_units(func_node_count);
                    let max_call_depth = 1u32.saturating_add(state.cur_max_callee_depth);
                    let max_stack_units =
                        frame_units.saturating_add(state.cur_max_callee_stack_units);
                    let worst_case_steps = state.cur_steps.max(1);
                    state.current_func = None;
                    state.local_types = [None; 64];
                    state.local_init = 0;
                    state.region_owned = 0;
                    state.region_leased = 0;
                    state.reachable = true;
                    state.cur_steps = 0;
                    state.cur_max_callee_depth = 0;
                    state.cur_max_callee_stack_units = 0;
                    state.completed_funcs.push(CompletedFunc {
                        sig,
                        start_node,
                        worst_case_steps,
                        max_call_depth,
                        max_stack_units,
                    });
                }
            }

            Ok(out_meta)
        }
        Opcode::Br => {
            require_pure_token(node_idx, frame)?;
            if frame.in_b != 0 || frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "Br requires in_b == 0 and proof_idx == 0",
                    ),
                });
            }
            let depth = frame.in_a as usize;
            let VerifierState {
                ref mut control_stack,
                ref mut arena_state,
                local_init,
                region_owned,
                region_leased,
                ref mut reachable,
                ..
            } = *state;
            let stack_len = control_stack.len();
            if depth >= stack_len {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "Br relative depth exceeds open control stack",
                    ),
                });
            }
            let target_idx = stack_len - 1 - depth;
            let target = control_stack.get_mut(target_idx).ok_or(VerifyDiagnostic {
                node_idx,
                kind: VerifyErrorKind::InvalidControlFlow(
                    "Br relative depth exceeds open control stack",
                ),
            })?;
            if matches!(target.kind, ControlKind::Func { .. }) {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "Br cannot target a Func scope; use Return",
                    ),
                });
            }
            if target.sig.and_then(|s| s.result_type).is_some() {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "Br cannot target value-returning block; use typed locals for early exits",
                    ),
                });
            }
            if *reachable {
                match target.kind {
                    ControlKind::Loop => {
                        check_loop_arena_invariant(program, node_idx, target, arena_state)?;
                        check_loop_region_invariant(node_idx, target, region_owned, region_leased)?;
                    }
                    ControlKind::Block | ControlKind::If { .. } => {
                        target.has_branch_or_join = true;
                        join_control_exit(
                            program,
                            target,
                            node_idx,
                            ExitSnapshot {
                                local_init,
                                region_owned,
                                region_leased,
                            },
                            arena_state,
                        )?;
                    }
                    ControlKind::Func { .. } => {}
                }
            }
            *reachable = false;
            Ok(NodeMeta::VOID)
        }
        Opcode::BrIf => {
            require_pure_token(node_idx, frame)?;
            if frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow("BrIf requires proof_idx == 0"),
                });
            }
            let cond_meta = get_prior_meta(node_idx, frame.in_a, metas, state)?;
            if cond_meta.elem != ElemType::BoolMask {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::BoolMask,
                        actual: cond_meta.elem,
                    },
                });
            }
            if cond_meta.shape != ValueShape::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: cond_meta.shape,
                    },
                });
            }
            let depth = frame.in_b as usize;
            let VerifierState {
                ref mut control_stack,
                ref mut arena_state,
                local_init,
                region_owned,
                region_leased,
                reachable,
                ..
            } = *state;
            let stack_len = control_stack.len();
            if depth >= stack_len {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "BrIf relative depth exceeds open control stack",
                    ),
                });
            }
            let target_idx = stack_len - 1 - depth;
            let target = control_stack.get_mut(target_idx).ok_or(VerifyDiagnostic {
                node_idx,
                kind: VerifyErrorKind::InvalidControlFlow(
                    "BrIf relative depth exceeds open control stack",
                ),
            })?;
            if matches!(target.kind, ControlKind::Func { .. }) {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "BrIf cannot target a Func scope; use Return",
                    ),
                });
            }
            if target.sig.and_then(|s| s.result_type).is_some() {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "BrIf cannot target value-returning block; use typed locals for early exits",
                    ),
                });
            }
            if reachable {
                match target.kind {
                    ControlKind::Loop => {
                        check_loop_arena_invariant(program, node_idx, target, arena_state)?;
                        check_loop_region_invariant(node_idx, target, region_owned, region_leased)?;
                    }
                    ControlKind::Block | ControlKind::If { .. } => {
                        target.has_branch_or_join = true;
                        join_control_exit(
                            program,
                            target,
                            node_idx,
                            ExitSnapshot {
                                local_init,
                                region_owned,
                                region_leased,
                            },
                            arena_state,
                        )?;
                    }
                    ControlKind::Func { .. } => {}
                }
            }
            Ok(NodeMeta::VOID)
        }
        Opcode::RegionNew => {
            require_pure_token(node_idx, frame)?;
            if frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow("RegionNew requires proof_idx == 0"),
                });
            }
            let decl = lookup_region_decl(node_idx, frame.in_a, &state.regions)?;
            check_region_in_footprint(node_idx, decl.slot, state)?;
            let len_meta = get_prior_meta(node_idx, frame.in_b, metas, state)?;
            if len_meta.elem != ElemType::I64 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::I64,
                        actual: len_meta.elem,
                    },
                });
            }
            if len_meta.shape != ValueShape::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: len_meta.shape,
                    },
                });
            }
            if frame.type_lane.elem != decl.elem {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: decl.elem,
                        actual: frame.type_lane.elem,
                    },
                });
            }
            if frame.type_lane.width != SimdWidth::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: ValueShape::Column(0),
                    },
                });
            }
            if state.reachable {
                let bit = 1u64 << decl.slot;
                if (state.region_leased & bit) != 0 {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::MutationUnderLiveLease { slot: decl.slot },
                    });
                }
                if (state.region_owned & bit) == 0 {
                    if decl.role != RegionRole::Plain {
                        return Err(VerifyDiagnostic {
                            node_idx,
                            kind: VerifyErrorKind::UseOfMovedRegion { slot: decl.slot },
                        });
                    }
                    state.region_owned |= bit;
                }
            }
            Ok(NodeMeta::VOID)
        }
        Opcode::RegionGrow => {
            require_pure_token(node_idx, frame)?;
            if frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow("RegionGrow requires proof_idx == 0"),
                });
            }
            let decl = lookup_region_decl(node_idx, frame.in_a, &state.regions)?;
            check_region_in_footprint(node_idx, decl.slot, state)?;
            let add_meta = get_prior_meta(node_idx, frame.in_b, metas, state)?;
            if add_meta.elem != ElemType::I64 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::I64,
                        actual: add_meta.elem,
                    },
                });
            }
            if add_meta.shape != ValueShape::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: add_meta.shape,
                    },
                });
            }
            if frame.type_lane.elem != ElemType::I64 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::I64,
                        actual: frame.type_lane.elem,
                    },
                });
            }
            if frame.type_lane.width != SimdWidth::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: ValueShape::Column(0),
                    },
                });
            }
            check_region_mutate(node_idx, decl.slot, state)?;
            Ok(NodeMeta::value(ValueShape::Scalar, ElemType::I64, None))
        }
        Opcode::RegionLen => {
            require_pure_token(node_idx, frame)?;
            if frame.in_b != 0 || frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "RegionLen requires in_b == 0 and proof_idx == 0",
                    ),
                });
            }
            let decl = lookup_region_decl(node_idx, frame.in_a, &state.regions)?;
            check_region_in_footprint(node_idx, decl.slot, state)?;
            if frame.type_lane.elem != ElemType::I64 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::I64,
                        actual: frame.type_lane.elem,
                    },
                });
            }
            if frame.type_lane.width != SimdWidth::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: ValueShape::Column(0),
                    },
                });
            }
            check_region_read(node_idx, decl.slot, state)?;
            Ok(NodeMeta::value(ValueShape::Scalar, ElemType::I64, None))
        }
        Opcode::RegionLoad => {
            require_pure_token(node_idx, frame)?;
            let decl = lookup_region_decl(node_idx, frame.in_a, &state.regions)?;
            check_region_in_footprint(node_idx, decl.slot, state)?;
            let idx_meta = get_prior_meta(node_idx, frame.in_b, metas, state)?;
            if idx_meta.elem != ElemType::I64 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::I64,
                        actual: idx_meta.elem,
                    },
                });
            }
            if idx_meta.shape != ValueShape::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: idx_meta.shape,
                    },
                });
            }
            if frame.type_lane.elem != decl.elem {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: decl.elem,
                        actual: frame.type_lane.elem,
                    },
                });
            }
            if frame.type_lane.width != SimdWidth::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: ValueShape::Column(0),
                    },
                });
            }
            check_region_read(node_idx, decl.slot, state)?;
            if frame.proof_idx == 0 {
                state.has_unproven_region_access = true;
            } else {
                verify_region_index_proof(program, node_idx, frame.proof_idx, decl, idx_meta)?;
            }
            Ok(NodeMeta::value(ValueShape::Scalar, decl.elem, None))
        }
        Opcode::RegionStore => {
            require_pure_token(node_idx, frame)?;
            let decl = lookup_region_decl(node_idx, frame.in_a, &state.regions)?;
            check_region_in_footprint(node_idx, decl.slot, state)?;
            let (idx_node, val_node) = frame.decode_u16_pair();
            let idx_meta = get_prior_meta(node_idx, idx_node, metas, state)?;
            if idx_meta.elem != ElemType::I64 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::I64,
                        actual: idx_meta.elem,
                    },
                });
            }
            if idx_meta.shape != ValueShape::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: idx_meta.shape,
                    },
                });
            }
            let val_meta = get_prior_meta(node_idx, val_node, metas, state)?;
            if val_meta.elem != decl.elem {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: decl.elem,
                        actual: val_meta.elem,
                    },
                });
            }
            if val_meta.shape != ValueShape::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: val_meta.shape,
                    },
                });
            }
            if frame.type_lane.elem != decl.elem {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: decl.elem,
                        actual: frame.type_lane.elem,
                    },
                });
            }
            if frame.type_lane.width != SimdWidth::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: ValueShape::Column(0),
                    },
                });
            }
            check_region_mutate(node_idx, decl.slot, state)?;
            if frame.proof_idx == 0 {
                state.has_unproven_region_access = true;
            } else {
                verify_region_index_proof(program, node_idx, frame.proof_idx, decl, idx_meta)?;
            }
            Ok(NodeMeta::VOID)
        }
        Opcode::RegionFree => {
            require_pure_token(node_idx, frame)?;
            if frame.in_b != 0 || frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "RegionFree requires in_b == 0 and proof_idx == 0",
                    ),
                });
            }
            let decl = lookup_region_decl(node_idx, frame.in_a, &state.regions)?;
            check_region_in_footprint(node_idx, decl.slot, state)?;
            if decl.min_len != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidRegionDecl {
                        slot: decl.slot,
                        reason: "RegionFree is only permitted when min_len == 0",
                    },
                });
            }
            if frame.type_lane.elem != decl.elem {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: decl.elem,
                        actual: frame.type_lane.elem,
                    },
                });
            }
            if frame.type_lane.width != SimdWidth::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: ValueShape::Column(0),
                    },
                });
            }
            check_region_mutate(node_idx, decl.slot, state)?;
            Ok(NodeMeta::VOID)
        }
        Opcode::RegionLease => {
            require_pure_token(node_idx, frame)?;
            if frame.in_b != 0 || frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "RegionLease requires in_b == 0 and proof_idx == 0",
                    ),
                });
            }
            let decl = lookup_region_decl(node_idx, frame.in_a, &state.regions)?;
            check_region_in_footprint(node_idx, decl.slot, state)?;
            if frame.type_lane.elem != decl.elem {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: decl.elem,
                        actual: frame.type_lane.elem,
                    },
                });
            }
            if frame.type_lane.width != SimdWidth::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: ValueShape::Column(0),
                    },
                });
            }
            if state.reachable {
                let bit = 1u64 << decl.slot;
                if (state.region_leased & bit) != 0 {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::MutationUnderLiveLease { slot: decl.slot },
                    });
                }
                if (state.region_owned & bit) == 0 {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::UseOfMovedRegion { slot: decl.slot },
                    });
                }
                state.region_owned &= !bit;
                state.region_leased |= bit;
            }
            Ok(NodeMeta::VOID)
        }
        Opcode::RegionEndLease => {
            require_pure_token(node_idx, frame)?;
            if frame.in_b != 0 || frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "RegionEndLease requires in_b == 0 and proof_idx == 0",
                    ),
                });
            }
            let decl = lookup_region_decl(node_idx, frame.in_a, &state.regions)?;
            check_region_in_footprint(node_idx, decl.slot, state)?;
            if frame.type_lane.elem != decl.elem {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: decl.elem,
                        actual: frame.type_lane.elem,
                    },
                });
            }
            if frame.type_lane.width != SimdWidth::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: ValueShape::Column(0),
                    },
                });
            }
            if state.reachable {
                let bit = 1u64 << decl.slot;
                if ((state.region_owned | state.region_leased) & bit) == 0 {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::UseOfMovedRegion { slot: decl.slot },
                    });
                }
                if (state.region_leased & bit) == 0 {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::RegionTypestateMismatch {
                            expected_owned: state.region_owned & !bit,
                            expected_leased: state.region_leased | bit,
                            actual_owned: state.region_owned,
                            actual_leased: state.region_leased,
                        },
                    });
                }
                state.region_leased &= !bit;
                state.region_owned |= bit;
            }
            Ok(NodeMeta::VOID)
        }
        Opcode::RegionMove => {
            require_pure_token(node_idx, frame)?;
            if frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow("RegionMove requires proof_idx == 0"),
                });
            }
            let src_decl = lookup_region_decl(node_idx, frame.in_a, &state.regions)?;
            let dst_decl = lookup_region_decl(node_idx, frame.in_b, &state.regions)?;
            check_region_in_footprint(node_idx, src_decl.slot, state)?;
            check_region_in_footprint(node_idx, dst_decl.slot, state)?;
            if src_decl.slot == dst_decl.slot {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "RegionMove requires distinct source and destination region slots",
                    ),
                });
            }
            if src_decl.role != RegionRole::Plain {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidRegionDecl {
                        slot: src_decl.slot,
                        reason: "RegionMove requires Plain role on source region",
                    },
                });
            }
            if dst_decl.role != RegionRole::Plain {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidRegionDecl {
                        slot: dst_decl.slot,
                        reason: "RegionMove requires Plain role on destination region",
                    },
                });
            }
            if src_decl.elem != dst_decl.elem {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: src_decl.elem,
                        actual: dst_decl.elem,
                    },
                });
            }
            if src_decl.max_len > dst_decl.max_len {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidRegionDecl {
                        slot: dst_decl.slot,
                        reason: "RegionMove source max_len exceeds destination max_len",
                    },
                });
            }
            if dst_decl.min_len != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidRegionDecl {
                        slot: dst_decl.slot,
                        reason: "RegionMove requires destination min_len == 0",
                    },
                });
            }
            if frame.type_lane.elem != src_decl.elem {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: src_decl.elem,
                        actual: frame.type_lane.elem,
                    },
                });
            }
            if frame.type_lane.width != SimdWidth::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: ValueShape::Column(0),
                    },
                });
            }
            if state.reachable {
                let src_bit = 1u64 << src_decl.slot;
                let dst_bit = 1u64 << dst_decl.slot;
                if (state.region_leased & src_bit) != 0 {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::MutationUnderLiveLease {
                            slot: src_decl.slot,
                        },
                    });
                }
                if (state.region_owned & src_bit) == 0 {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::UseOfMovedRegion {
                            slot: src_decl.slot,
                        },
                    });
                }
                if (state.region_leased & dst_bit) != 0 {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::MutationUnderLiveLease {
                            slot: dst_decl.slot,
                        },
                    });
                }
                if (state.region_owned & dst_bit) != 0 {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::RegionTypestateMismatch {
                            expected_owned: state.region_owned & !dst_bit,
                            expected_leased: state.region_leased,
                            actual_owned: state.region_owned,
                            actual_leased: state.region_leased,
                        },
                    });
                }
                state.region_owned &= !src_bit;
                state.region_owned |= dst_bit;
            }
            Ok(NodeMeta::VOID)
        }
        Opcode::RegionDrop => {
            require_pure_token(node_idx, frame)?;
            if frame.in_b != 0 || frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "RegionDrop requires in_b == 0 and proof_idx == 0",
                    ),
                });
            }
            let decl = lookup_region_decl(node_idx, frame.in_a, &state.regions)?;
            check_region_in_footprint(node_idx, decl.slot, state)?;
            if decl.role != RegionRole::Plain {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidRegionDecl {
                        slot: decl.slot,
                        reason: "RegionDrop requires Plain role",
                    },
                });
            }
            if decl.min_len != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidRegionDecl {
                        slot: decl.slot,
                        reason: "RegionDrop is only permitted when min_len == 0",
                    },
                });
            }
            if frame.type_lane.elem != decl.elem {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: decl.elem,
                        actual: frame.type_lane.elem,
                    },
                });
            }
            if frame.type_lane.width != SimdWidth::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: ValueShape::Column(0),
                    },
                });
            }
            check_region_mutate(node_idx, decl.slot, state)?;
            if state.reachable {
                state.region_owned &= !(1u64 << decl.slot);
            }
            Ok(NodeMeta::VOID)
        }
        Opcode::Func => {
            require_pure_token(node_idx, frame)?;
            if frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidFuncSig {
                        func_id: frame.in_a,
                        reason: "Func node must have proof_idx == 0",
                    },
                });
            }
            if frame.in_b != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow("Func requires in_b == 0"),
                });
            }
            if !state.control_stack.is_empty() || state.current_func.is_some() {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "Func cannot be nested inside an open control scope",
                    ),
                });
            }
            let expected_id = state.next_func_id;
            if frame.in_a != expected_id {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidFuncSig {
                        func_id: frame.in_a,
                        reason: "Func ids must be dense and declared in ascending order 0..N",
                    },
                });
            }
            let sig =
                state
                    .func_sigs
                    .get(expected_id as usize)
                    .copied()
                    .ok_or(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::UnknownFunction(frame.in_a),
                    })?;
            if let Some(ret_elem) = sig.ret_elem {
                if frame.type_lane.elem != ret_elem {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::ElementTypeMismatch {
                            expected: ret_elem,
                            actual: frame.type_lane.elem,
                        },
                    });
                }
                if frame.type_lane.width != SimdWidth::Scalar {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::ShapeMismatch {
                            left: ValueShape::Scalar,
                            right: ValueShape::Column(0),
                        },
                    });
                }
            }
            state.local_types = [None; 64];
            state.local_init = 0;
            for i in 0..usize::from(sig.arity) {
                if let (Some(slot), Some(&param_elem)) =
                    (state.local_types.get_mut(i), sig.params.get(i))
                {
                    *slot = Some(param_elem);
                    state.local_init |= 1u64 << i;
                }
            }
            let contract = state
                .func_contracts
                .get(expected_id as usize)
                .copied()
                .unwrap_or(FuncRegionContract {
                    dom_mask: 0,
                    pre_owned: 0,
                    pre_leased: 0,
                    post_owned: 0,
                    post_leased: 0,
                });
            state.region_owned = contract.pre_owned;
            state.region_leased = contract.pre_leased;
            state.reachable = true;
            state.cur_steps = 0;
            state.cur_max_callee_depth = 0;
            state.cur_max_callee_stack_units = 0;
            state.current_func = Some(ActiveFunc {
                sig,
                start_node: node_idx,
            });
            state.next_func_id = expected_id.saturating_add(1);
            let entry_arena_root = state.arena_state.commit_root(program);
            state.control_stack.push(ControlFrame {
                kind: ControlKind::Func {
                    sig,
                    start_node: node_idx,
                },
                header_node: node_idx,
                sig: None,
                entry_init: state.local_init,
                entry_arena_root,
                entry_reachable: true,
                entry_region_owned: state.region_owned,
                entry_region_leased: state.region_leased,
                has_branch_or_join: false,
                has_exit_path: false,
                exit_init: state.local_init,
                exit_region_owned: state.region_owned,
                exit_region_leased: state.region_leased,
                exit_arena_root: entry_arena_root,
                parent_steps: 0,
                then_steps: 0,
                loop_bound: 0,
                scope_nodes: Vec::new(),
            });
            Ok(NodeMeta::VOID)
        }
        Opcode::Return => {
            require_pure_token(node_idx, frame)?;
            if frame.in_b != 0 || frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "Return requires in_b == 0 and proof_idx == 0",
                    ),
                });
            }
            let active = state.current_func.ok_or(VerifyDiagnostic {
                node_idx,
                kind: VerifyErrorKind::InvalidControlFlow("Return is only valid inside a Func"),
            })?;
            if let Some(ret_elem) = active.sig.ret_elem {
                let val_meta = get_prior_meta(node_idx, frame.in_a, metas, state)?;
                if val_meta.elem != ret_elem {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::ElementTypeMismatch {
                            expected: ret_elem,
                            actual: val_meta.elem,
                        },
                    });
                }
                if val_meta.shape != ValueShape::Scalar {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::ShapeMismatch {
                            left: ValueShape::Scalar,
                            right: val_meta.shape,
                        },
                    });
                }
            } else if frame.in_a != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow("void Return requires in_a == 0"),
                });
            }
            if state.reachable {
                if let Some(contract) = state
                    .func_contracts
                    .get(usize::from(active.sig.func_id))
                    .copied()
                {
                    if state.region_owned != contract.post_owned
                        || state.region_leased != contract.post_leased
                    {
                        return Err(VerifyDiagnostic {
                            node_idx,
                            kind: VerifyErrorKind::RegionTypestateMismatch {
                                expected_owned: contract.post_owned,
                                expected_leased: contract.post_leased,
                                actual_owned: state.region_owned,
                                actual_leased: state.region_leased,
                            },
                        });
                    }
                }
            }
            state.reachable = false;
            Ok(NodeMeta::VOID)
        }
        Opcode::CallArg => {
            require_pure_token(node_idx, frame)?;
            if frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow("CallArg requires proof_idx == 0"),
                });
            }
            if state.current_func.is_none() {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "CallArg is only valid inside a Func",
                    ),
                });
            }
            let sentinel = u32::from(CALL_ARG_SENTINEL);
            if frame.in_a == 0 || frame.in_a >= sentinel {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "CallArg requires a valid relative operand in in_a (arg2)",
                    ),
                });
            }
            let arg2_meta = get_prior_meta(node_idx, frame.in_a, metas, state)?;
            if arg2_meta.shape != ValueShape::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: arg2_meta.shape,
                    },
                });
            }
            let arg3_elem = if frame.in_b == sentinel {
                None
            } else {
                if frame.in_b == 0 || frame.in_b > sentinel {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::InvalidControlFlow(
                            "CallArg in_b (arg3) must be a valid relative operand or 0xFFFF sentinel",
                        ),
                    });
                }
                let arg3_meta = get_prior_meta(node_idx, frame.in_b, metas, state)?;
                if arg3_meta.shape != ValueShape::Scalar {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::ShapeMismatch {
                            left: ValueShape::Scalar,
                            right: arg3_meta.shape,
                        },
                    });
                }
                Some(arg3_meta.elem)
            };
            state.pending_call_arg = Some(PendingCallArg {
                node_idx,
                arg2_elem: arg2_meta.elem,
                arg3_elem,
            });
            Ok(NodeMeta::VOID)
        }
        Opcode::Call => {
            require_pure_token(node_idx, frame)?;
            if frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow("Call requires proof_idx == 0"),
                });
            }
            let active = state.current_func.ok_or(VerifyDiagnostic {
                node_idx,
                kind: VerifyErrorKind::InvalidControlFlow("Call is only valid inside a Func"),
            })?;
            let target_id = frame.in_a;
            let callee_sig =
                state
                    .func_sigs
                    .get(target_id as usize)
                    .copied()
                    .ok_or(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::UnknownFunction(target_id),
                    })?;
            if target_id >= u32::from(active.sig.func_id)
                && (program.capability_mask & CAP_DIVERGE) == 0
            {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::MissingCapability {
                        required: CAP_DIVERGE,
                        actual: program.capability_mask,
                    },
                });
            }
            let pending = state.pending_call_arg.take();
            let (arg0_raw, arg1_raw) = frame.decode_u16_pair();
            let sentinel = u32::from(CALL_ARG_SENTINEL);
            let mut arg_elems: [Option<ElemType>; 4] = [None; 4];

            if arg0_raw != sentinel {
                let meta0 = get_prior_meta(node_idx, arg0_raw, metas, state)?;
                if meta0.shape != ValueShape::Scalar {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::ShapeMismatch {
                            left: ValueShape::Scalar,
                            right: meta0.shape,
                        },
                    });
                }
                arg_elems[0] = Some(meta0.elem);
            }
            if arg1_raw != sentinel {
                if arg_elems[0].is_none() {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::InvalidControlFlow(
                            "Call arg1 provided without arg0",
                        ),
                    });
                }
                let meta1 = get_prior_meta(node_idx, arg1_raw, metas, state)?;
                if meta1.shape != ValueShape::Scalar {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::ShapeMismatch {
                            left: ValueShape::Scalar,
                            right: meta1.shape,
                        },
                    });
                }
                arg_elems[1] = Some(meta1.elem);
            }
            if let Some(p) = pending {
                if arg_elems[0].is_none() || arg_elems[1].is_none() {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::InvalidControlFlow(
                            "CallArg provided when Call has fewer than 2 arguments",
                        ),
                    });
                }
                arg_elems[2] = Some(p.arg2_elem);
                arg_elems[3] = p.arg3_elem;
            }

            let provided_arity = arg_elems.iter().take_while(|e| e.is_some()).count();
            if provided_arity != usize::from(callee_sig.arity) {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "Call argument count does not match callee FuncSig arity",
                    ),
                });
            }
            for i in 0..usize::from(callee_sig.arity) {
                if let (Some(&expected), Some(Some(actual))) =
                    (callee_sig.params.get(i), arg_elems.get(i))
                {
                    if *actual != expected {
                        return Err(VerifyDiagnostic {
                            node_idx,
                            kind: VerifyErrorKind::ElementTypeMismatch {
                                expected,
                                actual: *actual,
                            },
                        });
                    }
                }
            }

            // Enforce the composable frame rule: dom(callee) ⊆ dom(caller), check callee pre on dom(callee),
            // and apply callee post on dom(callee) while preserving slots outside dom(callee).
            let caller_contract = state
                .func_contracts
                .get(usize::from(active.sig.func_id))
                .copied()
                .unwrap_or(FuncRegionContract {
                    dom_mask: 0,
                    pre_owned: 0,
                    pre_leased: 0,
                    post_owned: 0,
                    post_leased: 0,
                });
            let callee_contract = state
                .func_contracts
                .get(target_id as usize)
                .copied()
                .unwrap_or(FuncRegionContract {
                    dom_mask: 0,
                    pre_owned: 0,
                    pre_leased: 0,
                    post_owned: 0,
                    post_leased: 0,
                });
            let extra_mask = callee_contract.dom_mask & !caller_contract.dom_mask;
            if extra_mask != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::CalleeFootprintExceedsCaller {
                        callee_func_id: callee_sig.func_id,
                        extra_mask,
                    },
                });
            }
            if state.reachable {
                let dom = callee_contract.dom_mask;
                let observed_owned = state.region_owned & dom;
                let observed_leased = state.region_leased & dom;
                if observed_owned != callee_contract.pre_owned
                    || observed_leased != callee_contract.pre_leased
                {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::RegionTypestateMismatch {
                            expected_owned: callee_contract.pre_owned,
                            expected_leased: callee_contract.pre_leased,
                            actual_owned: observed_owned,
                            actual_leased: observed_leased,
                        },
                    });
                }
                state.region_owned = (state.region_owned & !dom) | callee_contract.post_owned;
                state.region_leased = (state.region_leased & !dom) | callee_contract.post_leased;

                if (program.capability_mask & CAP_DIVERGE) == 0 {
                    if let Some(callee) = state.completed_funcs.get(target_id as usize).copied() {
                        state.cur_steps = state
                            .cur_steps
                            .saturating_add(1)
                            .saturating_add(callee.worst_case_steps);
                        state.cur_max_callee_depth =
                            state.cur_max_callee_depth.max(callee.max_call_depth);
                        state.cur_max_callee_stack_units =
                            state.cur_max_callee_stack_units.max(callee.max_stack_units);
                    } else {
                        state.cur_steps = state.cur_steps.saturating_add(1);
                    }
                }
            }

            if let Some(ret_elem) = callee_sig.ret_elem {
                if frame.type_lane.elem != ret_elem {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::ElementTypeMismatch {
                            expected: ret_elem,
                            actual: frame.type_lane.elem,
                        },
                    });
                }
                if frame.type_lane.width != SimdWidth::Scalar {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::ShapeMismatch {
                            left: ValueShape::Scalar,
                            right: ValueShape::Column(0),
                        },
                    });
                }
                Ok(NodeMeta::value(ValueShape::Scalar, ret_elem, None))
            } else {
                Ok(NodeMeta::VOID)
            }
        }
        Opcode::ConstDataLen => {
            require_pure_token(node_idx, frame)?;
            if frame.in_b != 0 || frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "ConstDataLen requires in_b == 0 and proof_idx == 0",
                    ),
                });
            }
            let seg = state
                .const_data_segs
                .get(frame.in_a as usize)
                .copied()
                .ok_or(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::UnknownConstDataSeg(frame.in_a),
                })?;
            if frame.type_lane.elem != ElemType::I64 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::I64,
                        actual: frame.type_lane.elem,
                    },
                });
            }
            if frame.type_lane.width != SimdWidth::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: ValueShape::Column(0),
                    },
                });
            }
            Ok(NodeMeta {
                shape: ValueShape::Scalar,
                elem: ElemType::I64,
                origin_arena: None,
                is_void: false,
                const_i64: Some(i64::from(seg.len)),
                nonneg_upper_bound: Some(u64::from(seg.len)),
            })
        }
        Opcode::ConstDataLoad => {
            require_pure_token(node_idx, frame)?;
            let seg = state
                .const_data_segs
                .get(frame.in_a as usize)
                .copied()
                .ok_or(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::UnknownConstDataSeg(frame.in_a),
                })?;
            let idx_meta = get_prior_meta(node_idx, frame.in_b, metas, state)?;
            if idx_meta.elem != ElemType::I64 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::I64,
                        actual: idx_meta.elem,
                    },
                });
            }
            if idx_meta.shape != ValueShape::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: idx_meta.shape,
                    },
                });
            }
            if !matches!(frame.type_lane.elem, ElemType::U8 | ElemType::I64) {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::U8,
                        actual: frame.type_lane.elem,
                    },
                });
            }
            if frame.type_lane.width != SimdWidth::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: ValueShape::Column(0),
                    },
                });
            }
            let elem_bytes: u64 = if frame.type_lane.elem == ElemType::I64 {
                8
            } else {
                1
            };
            if frame.proof_idx == 0 {
                state.has_unproven_region_access = true;
            } else {
                verify_const_data_index_proof(
                    program,
                    node_idx,
                    frame.proof_idx,
                    seg,
                    elem_bytes,
                    idx_meta,
                )?;
            }
            Ok(NodeMeta::value(
                ValueShape::Scalar,
                frame.type_lane.elem,
                None,
            ))
        }
        Opcode::BrTable => {
            require_pure_token(node_idx, frame)?;
            if frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow("BrTable requires proof_idx == 0"),
                });
            }
            let sel_meta = get_prior_meta(node_idx, frame.in_a, metas, state)?;
            if !matches!(sel_meta.elem, ElemType::I64 | ElemType::U8) {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::I64,
                        actual: sel_meta.elem,
                    },
                });
            }
            if sel_meta.shape != ValueShape::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: sel_meta.shape,
                    },
                });
            }
            let br_table_id = frame.in_b;
            let spec =
                state
                    .br_tables
                    .get(br_table_id as usize)
                    .copied()
                    .ok_or(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::UnknownBrTable(br_table_id),
                    })?;
            let used_slot =
                state
                    .br_table_used
                    .get_mut(br_table_id as usize)
                    .ok_or(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::UnknownBrTable(br_table_id),
                    })?;
            if *used_slot {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::DuplicateBrTableUse { br_table_id },
                });
            }
            *used_slot = true;

            let lo = spec.offset as usize;
            let byte_len = (spec.case_count as usize).saturating_mul(4);
            let payload = program
                .const_data
                .get(lo..lo.saturating_add(byte_len))
                .ok_or(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidConstDataTable {
                        reason: "BrTableSpec payload out of bounds",
                    },
                })?;

            let VerifierState {
                ref mut control_stack,
                ref mut arena_state,
                local_init,
                region_owned,
                region_leased,
                ref mut reachable,
                ..
            } = *state;

            let mut check_depth = |depth_u32: u32| -> Result<(), VerifyDiagnostic> {
                let depth = depth_u32 as usize;
                let stack_len = control_stack.len();
                if depth >= stack_len {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::InvalidControlFlow(
                            "BrTable relative depth exceeds open control stack",
                        ),
                    });
                }
                let target_idx = stack_len - 1 - depth;
                let target = control_stack.get_mut(target_idx).ok_or(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "BrTable relative depth exceeds open control stack",
                    ),
                })?;
                if matches!(target.kind, ControlKind::Func { .. }) {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::InvalidControlFlow(
                            "BrTable cannot target a Func scope; use Return",
                        ),
                    });
                }
                if target.sig.and_then(|s| s.result_type).is_some() {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::InvalidControlFlow(
                            "BrTable cannot target value-returning block; use typed locals for early exits",
                        ),
                    });
                }
                if matches!(target.kind, ControlKind::Loop)
                    && (target.loop_bound > 0 || (program.capability_mask & CAP_DIVERGE) == 0)
                {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::InvalidControlFlow(
                            "BrTable cannot target a bounded Loop or any Loop without CAP_DIVERGE",
                        ),
                    });
                }
                if *reachable {
                    match target.kind {
                        ControlKind::Loop => {
                            check_loop_arena_invariant(program, node_idx, target, arena_state)?;
                            check_loop_region_invariant(
                                node_idx,
                                target,
                                region_owned,
                                region_leased,
                            )?;
                        }
                        ControlKind::Block | ControlKind::If { .. } => {
                            target.has_branch_or_join = true;
                            join_control_exit(
                                program,
                                target,
                                node_idx,
                                ExitSnapshot {
                                    local_init,
                                    region_owned,
                                    region_leased,
                                },
                                arena_state,
                            )?;
                        }
                        ControlKind::Func { .. } => {}
                    }
                }
                Ok(())
            };

            for chunk in payload.as_chunks::<4>().0 {
                let d = u32::from_le_bytes(*chunk);
                check_depth(d)?;
            }
            check_depth(u32::from(spec.default_depth))?;

            *reachable = false;
            Ok(NodeMeta::VOID)
        }
        Opcode::CallIndirect => {
            require_pure_token(node_idx, frame)?;
            if frame.proof_idx != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "CallIndirect requires proof_idx == 0",
                    ),
                });
            }
            let active = state.current_func.ok_or(VerifyDiagnostic {
                node_idx,
                kind: VerifyErrorKind::InvalidControlFlow(
                    "CallIndirect is only valid inside a Func",
                ),
            })?;
            let (table_id, index_node) = frame.decode_in_a_u16_pair();
            let table =
                state
                    .call_tables
                    .get(table_id as usize)
                    .copied()
                    .ok_or(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::UnknownCallTable(table_id),
                    })?;
            let idx_meta = get_prior_meta(node_idx, index_node, metas, state)?;
            if idx_meta.elem != ElemType::I64 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::I64,
                        actual: idx_meta.elem,
                    },
                });
            }
            if idx_meta.shape != ValueShape::Scalar {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ShapeMismatch {
                        left: ValueShape::Scalar,
                        right: idx_meta.shape,
                    },
                });
            }
            if (program.capability_mask & CAP_DIVERGE) == 0
                && table.max_func_id >= u32::from(active.sig.func_id)
            {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::MissingCapability {
                        required: CAP_DIVERGE,
                        actual: program.capability_mask,
                    },
                });
            }

            let pending = state.pending_call_arg.take();
            let (arg0_raw, arg1_raw) = frame.decode_u16_pair();
            let sentinel = u32::from(CALL_ARG_SENTINEL);
            let mut arg_elems: [Option<ElemType>; 4] = [None; 4];

            if arg0_raw != sentinel {
                let meta0 = get_prior_meta(node_idx, arg0_raw, metas, state)?;
                if meta0.shape != ValueShape::Scalar {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::ShapeMismatch {
                            left: ValueShape::Scalar,
                            right: meta0.shape,
                        },
                    });
                }
                arg_elems[0] = Some(meta0.elem);
            }
            if arg1_raw != sentinel {
                if arg_elems[0].is_none() {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::InvalidControlFlow(
                            "CallIndirect arg1 provided without arg0",
                        ),
                    });
                }
                let meta1 = get_prior_meta(node_idx, arg1_raw, metas, state)?;
                if meta1.shape != ValueShape::Scalar {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::ShapeMismatch {
                            left: ValueShape::Scalar,
                            right: meta1.shape,
                        },
                    });
                }
                arg_elems[1] = Some(meta1.elem);
            }
            if let Some(p) = pending {
                if arg_elems[0].is_none() || arg_elems[1].is_none() {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::InvalidControlFlow(
                            "CallArg provided when CallIndirect has fewer than 2 arguments",
                        ),
                    });
                }
                arg_elems[2] = Some(p.arg2_elem);
                arg_elems[3] = p.arg3_elem;
            }

            let provided_arity = arg_elems.iter().take_while(|e| e.is_some()).count();
            if provided_arity != usize::from(table.sig.arity) {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "CallIndirect argument count does not match CallTable FuncSig arity",
                    ),
                });
            }
            for i in 0..usize::from(table.sig.arity) {
                if let (Some(&expected), Some(Some(actual))) =
                    (table.sig.params.get(i), arg_elems.get(i))
                {
                    if *actual != expected {
                        return Err(VerifyDiagnostic {
                            node_idx,
                            kind: VerifyErrorKind::ElementTypeMismatch {
                                expected,
                                actual: *actual,
                            },
                        });
                    }
                }
            }

            let caller_contract = state
                .func_contracts
                .get(usize::from(active.sig.func_id))
                .copied()
                .unwrap_or(FuncRegionContract {
                    dom_mask: 0,
                    pre_owned: 0,
                    pre_leased: 0,
                    post_owned: 0,
                    post_leased: 0,
                });
            let callee_contract = table.contract;
            let extra_mask = callee_contract.dom_mask & !caller_contract.dom_mask;
            if extra_mask != 0 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::CalleeFootprintExceedsCaller {
                        callee_func_id: table.rep_func_id,
                        extra_mask,
                    },
                });
            }
            if state.reachable {
                let dom = callee_contract.dom_mask;
                let observed_owned = state.region_owned & dom;
                let observed_leased = state.region_leased & dom;
                if observed_owned != callee_contract.pre_owned
                    || observed_leased != callee_contract.pre_leased
                {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::RegionTypestateMismatch {
                            expected_owned: callee_contract.pre_owned,
                            expected_leased: callee_contract.pre_leased,
                            actual_owned: observed_owned,
                            actual_leased: observed_leased,
                        },
                    });
                }
                state.region_owned = (state.region_owned & !dom) | callee_contract.post_owned;
                state.region_leased = (state.region_leased & !dom) | callee_contract.post_leased;

                if (program.capability_mask & CAP_DIVERGE) == 0 {
                    let (max_steps, max_depth, max_stack) = if let Some(cached) = table.cost_summary
                    {
                        cached
                    } else {
                        let lo = table.offset as usize;
                        let byte_len = (table.entry_count as usize).saturating_mul(4);
                        let mut ms: u64 = 0;
                        let mut md: u32 = 0;
                        let mut msu: u32 = 0;
                        if let Some(slice) = program.const_data.get(lo..lo.saturating_add(byte_len))
                        {
                            for chunk in slice.as_chunks::<4>().0 {
                                let fid = u32::from_le_bytes(*chunk);
                                if let Some(cf) = state.completed_funcs.get(fid as usize).copied() {
                                    ms = ms.max(cf.worst_case_steps);
                                    md = md.max(cf.max_call_depth);
                                    msu = msu.max(cf.max_stack_units);
                                }
                            }
                        }
                        if let Some(slot) = state.call_tables.get_mut(table_id as usize) {
                            slot.cost_summary = Some((ms, md, msu));
                        }
                        (ms, md, msu)
                    };
                    state.cur_steps = state.cur_steps.saturating_add(1).saturating_add(max_steps);
                    state.cur_max_callee_depth = state.cur_max_callee_depth.max(max_depth);
                    state.cur_max_callee_stack_units =
                        state.cur_max_callee_stack_units.max(max_stack);
                }
            }

            if let Some(ret_elem) = table.sig.ret_elem {
                if frame.type_lane.elem != ret_elem {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::ElementTypeMismatch {
                            expected: ret_elem,
                            actual: frame.type_lane.elem,
                        },
                    });
                }
                if frame.type_lane.width != SimdWidth::Scalar {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::ShapeMismatch {
                            left: ValueShape::Scalar,
                            right: ValueShape::Column(0),
                        },
                    });
                }
                Ok(NodeMeta::value(ValueShape::Scalar, ret_elem, None))
            } else {
                Ok(NodeMeta::VOID)
            }
        }
    }
}
