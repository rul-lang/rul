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
    ElemType, NodeFrame, Opcode, ProofRuleKind, RulProgram, SimdWidth, TokenMode, TypeLane,
    CAP_DIVERGE, CAP_NET_RING, MAX_SECTION_ITEMS,
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
}

impl NodeMeta {
    const VOID: Self = Self {
        shape: ValueShape::Scalar,
        elem: ElemType::I64,
        origin_arena: None,
        is_void: true,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ControlKind {
    Block,
    Loop,
    If { has_else: bool },
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
    };
    let mut max_fusion_group: u8 = 0;
    let mut has_gather = false;

    for (idx, frame) in program.nodes.iter().enumerate() {
        let node_idx = u32::try_from(idx).unwrap_or(u32::MAX);

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

    if let Some(unclosed) = state.control_stack.last() {
        return Err(VerifyDiagnostic {
            node_idx: unclosed.header_node,
            kind: VerifyErrorKind::InvalidControlFlow(
                "unclosed structured control block at end of program",
            ),
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
        zero_runtime_bounds_checks: !has_gather,
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
    node_visible: &[bool],
) -> Result<NodeMeta, VerifyDiagnostic> {
    if input_idx >= node_idx {
        return Err(VerifyDiagnostic {
            node_idx,
            kind: VerifyErrorKind::NonTopologicalEdge {
                input_node: input_idx,
            },
        });
    }
    let idx = input_idx as usize;
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
    if !node_visible.get(idx).copied().unwrap_or(false) {
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
            Ok(NodeMeta {
                shape: ValueShape::Column(arena.element_count),
                elem: arena.type_lane.elem,
                origin_arena: Some(expected_id),
                is_void: false,
            })
        }
        Opcode::ConstF64 => Ok(NodeMeta {
            shape: ValueShape::Scalar,
            elem: ElemType::F64,
            origin_arena: None,
            is_void: false,
        }),
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
            Ok(NodeMeta {
                shape: ValueShape::Scalar,
                elem: ElemType::I64,
                origin_arena: None,
                is_void: false,
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
            Ok(NodeMeta {
                shape: ValueShape::Scalar,
                elem,
                origin_arena: None,
                is_void: false,
            })
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
            let val_meta = get_prior_meta(node_idx, frame.in_b, metas, &state.node_visible)?;
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
            Ok(NodeMeta {
                shape: ValueShape::Scalar,
                elem: val_meta.elem,
                origin_arena: None,
                is_void: false,
            })
        }
        Opcode::ArenaSlice => {
            let src_meta = get_prior_meta(node_idx, frame.in_a, metas, &state.node_visible)?;
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
            Ok(NodeMeta {
                shape: ValueShape::Column(slice_len),
                elem: src_meta.elem,
                origin_arena: src_meta.origin_arena,
                is_void: false,
            })
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
            let val_meta = get_prior_meta(node_idx, frame.in_b, metas, &state.node_visible)?;
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
            Ok(NodeMeta {
                shape: unified,
                elem: val_meta.elem,
                origin_arena: Some(expected_id),
                is_void: false,
            })
        }
        Opcode::Add | Opcode::Sub | Opcode::Mul => {
            let meta_a = get_prior_meta(node_idx, frame.in_a, metas, &state.node_visible)?;
            let meta_b = get_prior_meta(node_idx, frame.in_b, metas, &state.node_visible)?;
            if meta_a.elem == ElemType::I64 && meta_b.elem == ElemType::I64 {
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
                return Ok(NodeMeta {
                    shape: ValueShape::Scalar,
                    elem: ElemType::I64,
                    origin_arena: None,
                    is_void: false,
                });
            }
            if meta_a.elem != ElemType::F64 || meta_b.elem != ElemType::F64 {
                let (expected, actual) = if meta_a.elem == ElemType::I64 {
                    (ElemType::I64, meta_b.elem)
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
            Ok(NodeMeta {
                shape,
                elem: ElemType::F64,
                origin_arena: None,
                is_void: false,
            })
        }
        Opcode::Div | Opcode::Min | Opcode::Max => {
            let meta_a = get_prior_meta(node_idx, frame.in_a, metas, &state.node_visible)?;
            let meta_b = get_prior_meta(node_idx, frame.in_b, metas, &state.node_visible)?;
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
            Ok(NodeMeta {
                shape,
                elem: ElemType::F64,
                origin_arena: None,
                is_void: false,
            })
        }
        Opcode::Fma => {
            let (in_b, in_c) = frame.decode_fma_bc();
            let meta_a = get_prior_meta(node_idx, frame.in_a, metas, &state.node_visible)?;
            let meta_b = get_prior_meta(node_idx, in_b, metas, &state.node_visible)?;
            let meta_c = get_prior_meta(node_idx, in_c, metas, &state.node_visible)?;
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
            Ok(NodeMeta {
                shape,
                elem: ElemType::F64,
                origin_arena: None,
                is_void: false,
            })
        }
        Opcode::Abs | Opcode::Sqrt | Opcode::Floor | Opcode::Ceil | Opcode::Ln | Opcode::Exp => {
            let meta_a = get_prior_meta(node_idx, frame.in_a, metas, &state.node_visible)?;
            if meta_a.elem != ElemType::F64 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::F64,
                        actual: meta_a.elem,
                    },
                });
            }
            Ok(NodeMeta {
                shape: meta_a.shape,
                elem: ElemType::F64,
                origin_arena: None,
                is_void: false,
            })
        }
        Opcode::CmpGt | Opcode::CmpLt | Opcode::CmpEq => {
            let meta_a = get_prior_meta(node_idx, frame.in_a, metas, &state.node_visible)?;
            let meta_b = get_prior_meta(node_idx, frame.in_b, metas, &state.node_visible)?;
            if meta_a.elem == ElemType::I64 && meta_b.elem == ElemType::I64 {
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
                return Ok(NodeMeta {
                    shape: ValueShape::Scalar,
                    elem: ElemType::BoolMask,
                    origin_arena: None,
                    is_void: false,
                });
            }
            if meta_a.elem != ElemType::F64 || meta_b.elem != ElemType::F64 {
                let (expected, actual) = if meta_a.elem == ElemType::I64 {
                    (ElemType::I64, meta_b.elem)
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
            Ok(NodeMeta {
                shape,
                elem: ElemType::BoolMask,
                origin_arena: None,
                is_void: false,
            })
        }
        Opcode::MaskAnd | Opcode::MaskOr => {
            let meta_a = get_prior_meta(node_idx, frame.in_a, metas, &state.node_visible)?;
            let meta_b = get_prior_meta(node_idx, frame.in_b, metas, &state.node_visible)?;
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
            Ok(NodeMeta {
                shape,
                elem: ElemType::BoolMask,
                origin_arena: None,
                is_void: false,
            })
        }
        Opcode::SelectZero => {
            let mask_meta = get_prior_meta(node_idx, frame.in_a, metas, &state.node_visible)?;
            let val_meta = get_prior_meta(node_idx, frame.in_b, metas, &state.node_visible)?;
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
            Ok(NodeMeta {
                shape,
                elem: val_meta.elem,
                origin_arena: None,
                is_void: false,
            })
        }
        Opcode::FilterCompact => {
            let mask_meta = get_prior_meta(node_idx, frame.in_a, metas, &state.node_visible)?;
            let val_meta = get_prior_meta(node_idx, frame.in_b, metas, &state.node_visible)?;
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
            Ok(NodeMeta {
                shape: ValueShape::Compacted {
                    max_len,
                    mask_node: frame.in_a,
                },
                elem: val_meta.elem,
                origin_arena: None,
                is_void: false,
            })
        }
        Opcode::ReduceSum | Opcode::ReduceMax | Opcode::ReduceMin => {
            let src_meta = get_prior_meta(node_idx, frame.in_a, metas, &state.node_visible)?;
            if src_meta.elem != ElemType::F64 {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::ElementTypeMismatch {
                        expected: ElemType::F64,
                        actual: src_meta.elem,
                    },
                });
            }
            Ok(NodeMeta {
                shape: ValueShape::Scalar,
                elem: ElemType::F64,
                origin_arena: None,
                is_void: false,
            })
        }
        Opcode::Gather => {
            let target_meta = get_prior_meta(node_idx, frame.in_a, metas, &state.node_visible)?;
            let idx_meta = get_prior_meta(node_idx, frame.in_b, metas, &state.node_visible)?;
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
            Ok(NodeMeta {
                shape: idx_meta.shape,
                elem: target_meta.elem,
                origin_arena: None,
                is_void: false,
            })
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
            let src_meta = get_prior_meta(node_idx, frame.in_a, metas, &state.node_visible)?;
            Ok(src_meta)
        }
        Opcode::AssertContract => {
            let src_meta = get_prior_meta(node_idx, frame.in_a, metas, &state.node_visible)?;
            Ok(src_meta)
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
            let cond_meta = get_prior_meta(node_idx, frame.in_a, metas, &state.node_visible)?;
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
                    ControlKind::Block | ControlKind::Loop => {
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
                let then_meta = get_prior_meta(node_idx, frame.in_a, metas, &state.node_visible)?;
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

            let out_meta = if let Some(res_tl) = result_tl_opt {
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
                let res_meta = get_prior_meta(node_idx, frame.in_a, metas, &state.node_visible)?;
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
                NodeMeta {
                    shape: ValueShape::Scalar,
                    elem: res_tl.elem,
                    origin_arena: None,
                    is_void: false,
                }
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
            let cond_meta = get_prior_meta(node_idx, frame.in_a, metas, &state.node_visible)?;
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
                }
            }
            Ok(NodeMeta::VOID)
        }
    }
}
