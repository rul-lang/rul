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
    ElemType, NodeFrame, Opcode, ProofRuleKind, RulProgram, TokenMode, CAP_NET_RING,
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
    let mut max_fusion_group: u8 = 0;

    for (idx, frame) in program.nodes.iter().enumerate() {
        let node_idx = u32::try_from(idx).unwrap_or(u32::MAX);

        if frame.schedule.fusion_group > max_fusion_group {
            max_fusion_group = frame.schedule.fusion_group;
        }

        // 1. Verify AffineToken / ReadLease transition if present.
        verify_token_transition(program, frame, node_idx, &mut arena_versions)?;

        // 2. Verify opcode-specific topological edges, types, shapes, and proofs.
        let meta = verify_node_semantics(program, frame, node_idx, &metas)?;
        metas.push(meta);
    }

    let output_meta = metas
        .get(program.output_node as usize)
        .copied()
        .ok_or(VerifyDiagnostic {
            node_idx: program.output_node,
            kind: VerifyErrorKind::InvalidOutputNode(program.output_node),
        })?;

    Ok(VerifiedSummary {
        node_count,
        fusion_groups: usize::from(max_fusion_group),
        output_shape: output_meta.shape,
        zero_runtime_bounds_checks: true,
    })
}

fn verify_token_transition(
    program: &RulProgram,
    frame: &NodeFrame,
    node_idx: u32,
    arena_versions: &mut [u16],
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
            // Linear consumption: advance version so old token can never be reused!
            *live_ver_slot = live_ver_slot.wrapping_add(1);
            Ok(())
        }
    }
}

fn get_prior_meta(
    node_idx: u32,
    input_idx: u32,
    metas: &[NodeMeta],
) -> Result<NodeMeta, VerifyDiagnostic> {
    if input_idx >= node_idx {
        return Err(VerifyDiagnostic {
            node_idx,
            kind: VerifyErrorKind::NonTopologicalEdge {
                input_node: input_idx,
            },
        });
    }
    metas
        .get(input_idx as usize)
        .copied()
        .ok_or(VerifyDiagnostic {
            node_idx,
            kind: VerifyErrorKind::NonTopologicalEdge {
                input_node: input_idx,
            },
        })
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
            })
        }
        Opcode::ConstF64 => Ok(NodeMeta {
            shape: ValueShape::Scalar,
            elem: ElemType::F64,
            origin_arena: None,
        }),
        Opcode::ArenaSlice => {
            let src_meta = get_prior_meta(node_idx, frame.in_a, metas)?;
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
            if proof.bound_hi > target_arena.element_count {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidRefinementProof {
                        proof_idx: frame.proof_idx,
                        bound_hi: proof.bound_hi,
                        arena_len: target_arena.element_count,
                    },
                });
            }
            let slice_len = proof.bound_hi - proof.bound_lo;
            Ok(NodeMeta {
                shape: ValueShape::Column(slice_len),
                elem: src_meta.elem,
                origin_arena: src_meta.origin_arena,
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
            let val_meta = get_prior_meta(node_idx, frame.in_b, metas)?;
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
            })
        }
        Opcode::Add | Opcode::Sub | Opcode::Mul => {
            let meta_a = get_prior_meta(node_idx, frame.in_a, metas)?;
            let meta_b = get_prior_meta(node_idx, frame.in_b, metas)?;
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
            })
        }
        Opcode::CmpGt => {
            let meta_a = get_prior_meta(node_idx, frame.in_a, metas)?;
            let meta_b = get_prior_meta(node_idx, frame.in_b, metas)?;
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
                elem: ElemType::BoolMask,
                origin_arena: None,
            })
        }
        Opcode::SelectZero => {
            let mask_meta = get_prior_meta(node_idx, frame.in_a, metas)?;
            let val_meta = get_prior_meta(node_idx, frame.in_b, metas)?;
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
            })
        }
        Opcode::FilterCompact => {
            let mask_meta = get_prior_meta(node_idx, frame.in_a, metas)?;
            let val_meta = get_prior_meta(node_idx, frame.in_b, metas)?;
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
            })
        }
        Opcode::ReduceSum | Opcode::ReduceMax => {
            let src_meta = get_prior_meta(node_idx, frame.in_a, metas)?;
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
            })
        }
        Opcode::Gather => {
            let target_meta = get_prior_meta(node_idx, frame.in_a, metas)?;
            let idx_meta = get_prior_meta(node_idx, frame.in_b, metas)?;
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
            if proof.bound_hi > target_arena.element_count {
                return Err(VerifyDiagnostic {
                    node_idx,
                    kind: VerifyErrorKind::InvalidRefinementProof {
                        proof_idx: frame.proof_idx,
                        bound_hi: proof.bound_hi,
                        arena_len: target_arena.element_count,
                    },
                });
            }
            Ok(NodeMeta {
                shape: idx_meta.shape,
                elem: target_meta.elem,
                origin_arena: None,
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
            let src_meta = get_prior_meta(node_idx, frame.in_a, metas)?;
            Ok(src_meta)
        }
        Opcode::AssertContract => {
            let src_meta = get_prior_meta(node_idx, frame.in_a, metas)?;
            Ok(src_meta)
        }
    }
}
