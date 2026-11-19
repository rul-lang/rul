//! Single-pass $O(N)$ linear verifier for Rul (`.rl`) binary graphs.
//!
//! Instead of running an expensive type/borrow inference solver, [`verify_program`]
//! scans the flat [`NodeFrame`] array in a single forward loop to verify:
//! 1. Topological SSA-DAG ordering (`in_idx < current_idx`).
//! 2. `AffineToken` and `ReadLease` ownership conservation (0% Use-After-Consume or aliasing).
//! 3. `Bounds-Safe by Construction` shape propagation and sidecar [`ProofEntry`] certificates
//!    with strict `origin_arena` binding (preventing cross-arena proof spoofing).
//! 4. Zero-trust hardware [`CAP_NET_RING`] capability sandboxing.

use std::fmt;

use crate::bytecode::{
    ElemType, FuncSigInfo, NodeFrame, Opcode, ProofEntry, ProofRuleKind, RegionRole, RulProgram,
    SimdWidth, TokenMode, TypeLane, CALL_ARG_SENTINEL, CAP_DIVERGE, CAP_NET_RING, MAX_FUNC_NODES,
    MAX_REGION_BYTES, MAX_REGION_SLOTS, MAX_SECTION_ITEMS,
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
    entry_undo_len: usize,
    entry_bumps: u64,
    entry_reachable: bool,
    has_branch_or_join: bool,
    has_exit_path: bool,
    exit_init: u64,
    exit_delta: Option<ArenaDelta>,
    scope_nodes: Vec<usize>,
}

struct VerifierState {
    local_types: [Option<ElemType>; 64],
    local_init: u64,
    reachable: bool,
    undo_log: Vec<(u16, u16)>,
    version_bumps: u64,
    node_visible: Vec<bool>,
    control_stack: Vec<ControlFrame>,
    regions: [Option<RegionDeclInfo>; MAX_REGION_SLOTS],
    has_unproven_region_access: bool,
    func_sigs: Vec<FuncSigInfo>,
    current_func: Option<ActiveFunc>,
    pending_call_arg: Option<PendingCallArg>,
    next_func_id: u32,
    completed_funcs: Vec<CompletedFunc>,
    has_func_mode: bool,
    seen_non_func_top_level: bool,
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

/// Verifies an [`RulProgram`] in a single forward pass ($O(N)$ time and $O(N)$ state).
///
/// # Errors
/// Returns a structured [`VerifyDiagnostic`] pinpointing the exact node index and rule
/// violation if any safety, ownership, bounds, or capability invariant fails.
pub fn verify_program(program: &RulProgram) -> Result<VerifiedSummary, VerifyDiagnostic> {
    let node_count = program.nodes.len();
    if node_count == 0 || (program.output_node as usize) >= node_count {
        return Err(VerifyDiagnostic {
            node_idx: program.output_node,
            kind: VerifyErrorKind::InvalidOutputNode(program.output_node),
        });
    }

    // One linear pre-scan of the proof table to index statically declared regions and function signatures.
    let regions = scan_region_decls(program)?;
    let func_sigs = scan_func_sigs(program)?;
    let has_func_mode = !func_sigs.is_empty();

    // Track live AffineToken version per Arena.
    let mut arena_versions: Vec<u16> = program.arenas.iter().map(|a| a.initial_version).collect();

    // Track inferred ValueShape, ElemType, and origin Arena per SSA node.
    let mut metas: Vec<NodeMeta> = Vec::with_capacity(node_count);
    let mut state = VerifierState {
        local_types: [None; 64],
        local_init: 0,
        reachable: true,
        undo_log: Vec::new(),
        version_bumps: 0,
        node_visible: Vec::with_capacity(node_count),
        control_stack: Vec::new(),
        regions,
        has_unproven_region_access: false,
        func_sigs,
        current_func: None,
        pending_call_arg: None,
        next_func_id: 0,
        completed_funcs: Vec::new(),
        has_func_mode,
        seen_non_func_top_level: false,
    };
    let mut max_fusion_group: u8 = 0;
    let mut has_gather = false;

    for (idx, frame) in program.nodes.iter().enumerate() {
        let node_idx = u32::try_from(idx).unwrap_or(u32::MAX);

        if let Some(pending) = state.pending_call_arg {
            if frame.opcode != Opcode::Call {
                return Err(VerifyDiagnostic {
                    node_idx: pending.node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow(
                        "CallArg must immediately precede Call",
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

        // 1. Verify AffineToken / ReadLease transition if present.
        verify_token_transition(program, frame, node_idx, &mut arena_versions, &mut state)?;

        // 2. Verify opcode-specific topological edges, types, shapes, control stack, and proofs.
        let meta = verify_node_semantics(
            program,
            frame,
            node_idx,
            &metas,
            &mut arena_versions,
            &mut state,
        )?;
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
        return Ok(VerifiedSummary {
            node_count,
            fusion_groups: usize::from(max_fusion_group),
            output_shape: ValueShape::Scalar,
            output_elem: entry_ret_elem,
            zero_runtime_bounds_checks: !has_gather && !state.has_unproven_region_access,
        });
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

    Ok(VerifiedSummary {
        node_count,
        fusion_groups: usize::from(max_fusion_group),
        output_shape: output_meta.shape,
        output_elem: output_meta.elem,
        zero_runtime_bounds_checks: !has_gather && !state.has_unproven_region_access,
    })
}

fn verify_token_transition(
    program: &RulProgram,
    frame: &NodeFrame,
    node_idx: u32,
    arena_versions: &mut [u16],
    state: &mut VerifierState,
) -> Result<(), VerifyDiagnostic> {
    match frame.token_op.mode {
        TokenMode::PureValue => Ok(()),
        TokenMode::ReadLease => {
            let arena_idx = usize::from(frame.token_op.arena_id);
            let live_ver = arena_versions
                .get(arena_idx)
                .copied()
                .ok_or(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::UnknownArena(u32::from(frame.token_op.arena_id)),
                })?;
            if state.reachable && frame.token_op.version != live_ver {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::StaleOrConsumedToken {
                        arena_id: frame.token_op.arena_id,
                        provided_version: frame.token_op.version,
                        expected_version: live_ver,
                    },
                });
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
            let live_ver_slot = arena_versions.get_mut(arena_idx).ok_or(VerifyDiagnostic {
                node_idx,
                kind: VerifyErrorKind::UnknownArena(u32::from(frame.token_op.arena_id)),
            })?;
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
            if frame.token_op.version != *live_ver_slot {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::StaleOrConsumedToken {
                        arena_id: frame.token_op.arena_id,
                        provided_version: frame.token_op.version,
                        expected_version: *live_ver_slot,
                    },
                });
            }
            // Linear consumption: advance version so old token can never be reused,
            // and reject u16::MAX overflow so version can never wrap around to 0!
            let prev_ver = *live_ver_slot;
            let next_ver = prev_ver.checked_add(1).ok_or(VerifyDiagnostic {
                node_idx,
                kind: VerifyErrorKind::TokenVersionOverflow(frame.token_op.arena_id),
            })?;
            *live_ver_slot = next_ver;
            state.undo_log.push((frame.token_op.arena_id, prev_ver));
            state.version_bumps += 1;
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
    arena_versions: &[u16],
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
        let live_ver = arena_versions
            .get(arena_idx)
            .copied()
            .ok_or(VerifyDiagnostic {
                node_idx,
                kind: VerifyErrorKind::UnknownArena(u32::from(proof.target_arena)),
            })?;
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

fn rollback_arena_mutations(
    arena_versions: &mut [u16],
    undo_log: &mut Vec<(u16, u16)>,
    version_bumps: &mut u64,
    target_undo_len: usize,
    target_bumps: u64,
) {
    while undo_log.len() > target_undo_len {
        if let Some((arena_id, prev_ver)) = undo_log.pop() {
            if let Some(slot) = arena_versions.get_mut(usize::from(arena_id)) {
                *slot = prev_ver;
            }
        }
    }
    *version_bumps = target_bumps;
}

fn check_loop_arena_invariant(
    node_idx: u32,
    loop_frame: &ControlFrame,
    arena_versions: &[u16],
    undo_log: &[(u16, u16)],
    version_bumps: u64,
) -> Result<(), VerifyDiagnostic> {
    if version_bumps == loop_frame.entry_bumps {
        return Ok(());
    }
    let (arena_id, entry_ver) = undo_log
        .get(loop_frame.entry_undo_len)
        .copied()
        .unwrap_or((0, 0));
    let actual_ver = arena_versions
        .get(usize::from(arena_id))
        .copied()
        .unwrap_or(0);
    Err(VerifyDiagnostic {
        node_idx,
        kind: VerifyErrorKind::StaleOrConsumedToken {
            arena_id,
            provided_version: actual_ver,
            expected_version: entry_ver,
        },
    })
}

fn merge_exit_delta(
    ctrl: &mut ControlFrame,
    node_idx: u32,
    cur_init: u64,
    path_delta: Option<ArenaDelta>,
) -> Result<(), VerifyDiagnostic> {
    if !ctrl.has_exit_path {
        ctrl.has_exit_path = true;
        ctrl.exit_init = cur_init;
        ctrl.exit_delta = path_delta;
        return Ok(());
    }

    ctrl.exit_init &= cur_init;
    if ctrl.exit_delta == path_delta {
        return Ok(());
    }

    let (arena_id, provided_version, expected_version) = match (ctrl.exit_delta, path_delta) {
        (Some(prev), None) => (prev.arena_id, prev.entry_version, prev.exit_version),
        (None, Some(cur)) => (cur.arena_id, cur.exit_version, cur.entry_version),
        (Some(prev), Some(cur)) if prev.arena_id == cur.arena_id => {
            (cur.arena_id, cur.exit_version, prev.exit_version)
        }
        (Some(prev), Some(_cur)) => (prev.arena_id, prev.entry_version, prev.exit_version),
        (None, None) => (0, 0, 0),
    };
    Err(VerifyDiagnostic {
        node_idx,
        kind: VerifyErrorKind::StaleOrConsumedToken {
            arena_id,
            provided_version,
            expected_version,
        },
    })
}

fn join_control_exit(
    ctrl: &mut ControlFrame,
    node_idx: u32,
    cur_init: u64,
    cur_arenas: &[u16],
    undo_log: &[(u16, u16)],
    cur_bumps: u64,
) -> Result<(), VerifyDiagnostic> {
    let bumps = cur_bumps.saturating_sub(ctrl.entry_bumps);
    let path_delta = if let Some(decl) = ctrl.sig.and_then(|s| s.arena_delta) {
        let act_target = cur_arenas
            .get(usize::from(decl.arena_id))
            .copied()
            .unwrap_or(0);
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
        let expected_bumps = u64::from(decl.exit_version.saturating_sub(decl.entry_version));
        if bumps != expected_bumps {
            let tail = undo_log.get(ctrl.entry_undo_len..).unwrap_or(&[]);
            for &(id, prev_v) in tail {
                if id != decl.arena_id {
                    let act_other = cur_arenas.get(usize::from(id)).copied().unwrap_or(0);
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::StaleOrConsumedToken {
                            arena_id: id,
                            provided_version: act_other,
                            expected_version: prev_v,
                        },
                    });
                }
            }
        }
        if decl.exit_version > decl.entry_version {
            Some(decl)
        } else {
            None
        }
    } else if bumps == 0 {
        None
    } else {
        let (first_id, first_entry_v) =
            undo_log.get(ctrl.entry_undo_len).copied().unwrap_or((0, 0));
        let first_cur_v = cur_arenas.get(usize::from(first_id)).copied().unwrap_or(0);
        let first_bumps = u64::from(first_cur_v.saturating_sub(first_entry_v));
        if first_bumps != bumps {
            return Err(VerifyDiagnostic {
                node_idx,
                kind: VerifyErrorKind::InvalidControlFlow(
                    "mutating multiple distinct arenas across a branch join requires nested BlockSig scopes",
                ),
            });
        }
        Some(ArenaDelta {
            arena_id: first_id,
            entry_version: first_entry_v,
            exit_version: first_cur_v,
        })
    };

    merge_exit_delta(ctrl, node_idx, cur_init, path_delta)
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

#[expect(
    clippy::too_many_lines,
    reason = "single-pass O(N) opcode semantic verifier"
)]
fn verify_node_semantics(
    program: &RulProgram,
    frame: &NodeFrame,
    node_idx: u32,
    metas: &[NodeMeta],
    arena_versions: &mut [u16],
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
            let sig = parse_block_sig(program, frame, node_idx, arena_versions, state.reachable)?;
            state.control_stack.push(ControlFrame {
                kind: ControlKind::Block,
                header_node: node_idx,
                sig,
                entry_init: state.local_init,
                entry_undo_len: state.undo_log.len(),
                entry_bumps: state.version_bumps,
                entry_reachable: state.reachable,
                has_branch_or_join: false,
                has_exit_path: false,
                exit_init: state.local_init,
                exit_delta: None,
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
            if frame.proof_idx == 0 {
                if (program.capability_mask & CAP_DIVERGE) == 0 {
                    return Err(VerifyDiagnostic {
                        node_idx,
                        kind: VerifyErrorKind::MissingCapability {
                            required: CAP_DIVERGE,
                            actual: program.capability_mask,
                        },
                    });
                }
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
            }
            state.control_stack.push(ControlFrame {
                kind: ControlKind::Loop,
                header_node: node_idx,
                sig: None,
                entry_init: state.local_init,
                entry_undo_len: state.undo_log.len(),
                entry_bumps: state.version_bumps,
                entry_reachable: state.reachable,
                has_branch_or_join: false,
                has_exit_path: false,
                exit_init: state.local_init,
                exit_delta: None,
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
            let sig = parse_block_sig(program, frame, node_idx, arena_versions, state.reachable)?;
            state.control_stack.push(ControlFrame {
                kind: ControlKind::If { has_else: false },
                header_node: node_idx,
                sig,
                entry_init: state.local_init,
                entry_undo_len: state.undo_log.len(),
                entry_bumps: state.version_bumps,
                entry_reachable: state.reachable,
                has_branch_or_join: true,
                has_exit_path: false,
                exit_init: state.local_init,
                exit_delta: None,
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

            let (entry_init, entry_undo_len, entry_bumps, entry_reachable) = {
                let VerifierState {
                    ref mut control_stack,
                    ref mut node_visible,
                    ref undo_log,
                    version_bumps,
                    local_init,
                    reachable,
                    ..
                } = *state;
                let top = control_stack.last_mut().ok_or(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidControlFlow("Else without matching If"),
                })?;
                seal_scope_nodes(&mut top.scope_nodes, node_visible);
                if reachable {
                    join_control_exit(
                        top,
                        node_idx,
                        local_init,
                        arena_versions,
                        undo_log,
                        version_bumps,
                    )?;
                }
                top.kind = ControlKind::If { has_else: true };
                (
                    top.entry_init,
                    top.entry_undo_len,
                    top.entry_bumps,
                    top.entry_reachable,
                )
            };
            state.local_init = entry_init;
            rollback_arena_mutations(
                arena_versions,
                &mut state.undo_log,
                &mut state.version_bumps,
                entry_undo_len,
                entry_bumps,
            );
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
                    let needs_join = popped.has_branch_or_join
                        || popped.sig.and_then(|s| s.arena_delta).is_some();
                    if !needs_join {
                        if !state.reachable {
                            rollback_arena_mutations(
                                arena_versions,
                                &mut state.undo_log,
                                &mut state.version_bumps,
                                popped.entry_undo_len,
                                popped.entry_bumps,
                            );
                            state.local_init = popped.entry_init;
                        }
                    } else {
                        if state.reachable {
                            join_control_exit(
                                &mut popped,
                                node_idx,
                                state.local_init,
                                arena_versions,
                                &state.undo_log,
                                state.version_bumps,
                            )?;
                        }
                        rollback_arena_mutations(
                            arena_versions,
                            &mut state.undo_log,
                            &mut state.version_bumps,
                            popped.entry_undo_len,
                            popped.entry_bumps,
                        );
                        if popped.has_exit_path {
                            state.reachable = popped.entry_reachable;
                            state.local_init = popped.exit_init;
                            if popped.entry_reachable {
                                if let Some(d) = popped.exit_delta {
                                    if let Some(slot) =
                                        arena_versions.get_mut(usize::from(d.arena_id))
                                    {
                                        *slot = d.exit_version;
                                    }
                                    state.undo_log.push((d.arena_id, d.entry_version));
                                    state.version_bumps +=
                                        u64::from(d.exit_version.saturating_sub(d.entry_version));
                                }
                            }
                        } else {
                            state.reachable = false;
                            state.local_init = popped.entry_init;
                        }
                    }
                }
                ControlKind::If { has_else } => {
                    if state.reachable {
                        join_control_exit(
                            &mut popped,
                            node_idx,
                            state.local_init,
                            arena_versions,
                            &state.undo_log,
                            state.version_bumps,
                        )?;
                    }
                    rollback_arena_mutations(
                        arena_versions,
                        &mut state.undo_log,
                        &mut state.version_bumps,
                        popped.entry_undo_len,
                        popped.entry_bumps,
                    );
                    if !has_else && popped.entry_reachable {
                        let entry_init = popped.entry_init;
                        join_control_exit(
                            &mut popped,
                            node_idx,
                            entry_init,
                            arena_versions,
                            &state.undo_log,
                            state.version_bumps,
                        )?;
                    }
                    if popped.has_exit_path {
                        state.reachable = popped.entry_reachable;
                        state.local_init = popped.exit_init;
                        if popped.entry_reachable {
                            if let Some(d) = popped.exit_delta {
                                if let Some(slot) = arena_versions.get_mut(usize::from(d.arena_id))
                                {
                                    *slot = d.exit_version;
                                }
                                state.undo_log.push((d.arena_id, d.entry_version));
                                state.version_bumps +=
                                    u64::from(d.exit_version.saturating_sub(d.entry_version));
                            }
                        }
                    } else {
                        state.reachable = false;
                        state.local_init = popped.entry_init;
                    }
                }
                ControlKind::Loop => {
                    if state.reachable {
                        check_loop_arena_invariant(
                            node_idx,
                            &popped,
                            arena_versions,
                            &state.undo_log,
                            state.version_bumps,
                        )?;
                    } else {
                        rollback_arena_mutations(
                            arena_versions,
                            &mut state.undo_log,
                            &mut state.version_bumps,
                            popped.entry_undo_len,
                            popped.entry_bumps,
                        );
                        state.local_init = popped.entry_init;
                    }
                }
                ControlKind::Func { sig, start_node } => {
                    state.current_func = None;
                    state.local_types = [None; 64];
                    state.local_init = 0;
                    state.reachable = true;
                    state
                        .completed_funcs
                        .push(CompletedFunc { sig, start_node });
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
                ref undo_log,
                version_bumps,
                local_init,
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
                        check_loop_arena_invariant(
                            node_idx,
                            target,
                            arena_versions,
                            undo_log,
                            version_bumps,
                        )?;
                    }
                    ControlKind::Block | ControlKind::If { .. } => {
                        target.has_branch_or_join = true;
                        join_control_exit(
                            target,
                            node_idx,
                            local_init,
                            arena_versions,
                            undo_log,
                            version_bumps,
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
                ref undo_log,
                version_bumps,
                local_init,
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
                        check_loop_arena_invariant(
                            node_idx,
                            target,
                            arena_versions,
                            undo_log,
                            version_bumps,
                        )?;
                    }
                    ControlKind::Block | ControlKind::If { .. } => {
                        target.has_branch_or_join = true;
                        join_control_exit(
                            target,
                            node_idx,
                            local_init,
                            arena_versions,
                            undo_log,
                            version_bumps,
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
            let _decl = lookup_region_decl(node_idx, frame.in_a, &state.regions)?;
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
            let _decl = lookup_region_decl(node_idx, frame.in_a, &state.regions)?;
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
        Opcode::RegionLoad => {
            require_pure_token(node_idx, frame)?;
            let decl = lookup_region_decl(node_idx, frame.in_a, &state.regions)?;
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
            state.reachable = true;
            state.current_func = Some(ActiveFunc {
                sig,
                start_node: node_idx,
            });
            state.next_func_id = expected_id.saturating_add(1);
            state.control_stack.push(ControlFrame {
                kind: ControlKind::Func {
                    sig,
                    start_node: node_idx,
                },
                header_node: node_idx,
                sig: None,
                entry_init: state.local_init,
                entry_undo_len: state.undo_log.len(),
                entry_bumps: state.version_bumps,
                entry_reachable: true,
                has_branch_or_join: false,
                has_exit_path: false,
                exit_init: state.local_init,
                exit_delta: None,
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
    }
}
