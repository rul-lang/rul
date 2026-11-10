//! # Rul (`.rl`) — AI-Native Binary Programming Language & Execution Engine
//!
//! Rul replaces human-centric textual source code with a flat, memory-mapped
//! 128-bit instruction graph (`.rl`) governed by the First-Principles equation:
//!
//! $$\text{Artifact (}\mathtt{.rl}\text{)} = \text{Algorithm} + \text{Schedule} + \text{Proof}$$
//! $$\text{Compiler} = \text{Linear Verifier } O(N) + \text{Tiered Stencil/Tile Engine } O(N)$$

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![deny(clippy::indexing_slicing)]

pub mod aot;
pub mod bytecode;
pub mod cas;
pub mod engine;
pub mod ffi;
pub mod mcp;
pub mod suite10;
pub mod transpiler;
pub mod verifier;

#[cfg(test)]
mod tests {
    use crate::bytecode::{
        ArenaDescriptor, ExecPolicy, NodeFrame, Opcode, ProofEntry, ProofRuleKind, RulProgram,
        ScheduleHint, TokenOp, TypeLane, CAP_NET_RING, CAP_NONE,
    };
    use crate::engine::{
        execute_tier0_interpreter, execute_tier1_stencil, execute_tier2_fused,
        execute_tier2_with_zonemap, ArenaZoneMap, ExecError, RuntimeValue,
    };
    use crate::transpiler::{disassemble_rl, SymbolicGraphBuilder};
    use crate::verifier::{verify_program, ValueShape, VerifyErrorKind};

    fn build_canonical_risk_pipeline(elem_count: u32, threshold: f64) -> RulProgram {
        let mut b = SymbolicGraphBuilder::new(CAP_NONE);
        let arena0 = b.add_arena(elem_count, TypeLane::F64X8, false);
        let sched = ScheduleHint::fused_simd(1);

        let _ = b.emit_arena_input("$amounts", arena0, 1, sched);
        let _ = b.emit_const_f64("$threshold", threshold, sched);
        let _ = b.emit_binary_op(
            "$mask",
            Opcode::CmpGt,
            TypeLane::MASKX8,
            "$amounts",
            "$threshold",
            sched,
        );
        let _ = b.emit_binary_op(
            "$squared",
            Opcode::Mul,
            TypeLane::F64X8,
            "$amounts",
            "$amounts",
            sched,
        );
        let _ = b.emit_binary_op(
            "$selected",
            Opcode::SelectZero,
            TypeLane::F64X8,
            "$mask",
            "$squared",
            sched,
        );
        let _ = b.emit_unary_op(
            "$sum",
            Opcode::ReduceSum,
            TypeLane::F64_SCALAR,
            "$selected",
            sched,
        );
        let _ = b.emit_unary_op(
            "$verified_out",
            Opcode::AssertContract,
            TypeLane::F64_SCALAR,
            "$sum",
            sched,
        );
        b.finish("$verified_out").unwrap_or_else(|_| RulProgram {
            capability_mask: CAP_NONE,
            arenas: Vec::new(),
            proofs: Vec::new(),
            nodes: Vec::new(),
            output_node: 0,
        })
    }

    #[test]
    fn test_rl_binary_roundtrip_and_compact_size() {
        let prog = build_canonical_risk_pipeline(1024, 1000.0);
        let bytes = prog.to_rl_bytes();
        // 64B header + 1*16B arena + 0*16B proof + 7*16B nodes = 192 bytes total!
        assert_eq!(bytes.len(), 64 + 16 + 7 * 16);

        let decoded = RulProgram::from_rl_bytes(&bytes);
        assert_eq!(decoded, Ok(prog.clone()));

        let audit_text = disassemble_rl(&prog);
        assert!(audit_text.contains("RUL (.rl) BINARY GRAPH AUDIT"));
        assert!(audit_text.contains("ReduceSum"));
    }

    #[test]
    fn test_linear_verifier_accepts_valid_pipeline() {
        let prog = build_canonical_risk_pipeline(1024, 1000.0);
        let summary = verify_program(&prog);
        assert!(summary.is_ok());
        if let Ok(s) = summary {
            assert_eq!(s.node_count, 7);
            assert_eq!(s.fusion_groups, 1);
            assert_eq!(s.output_shape, ValueShape::Scalar);
            assert!(s.zero_runtime_bounds_checks);
        }
    }

    #[test]
    fn test_verifier_rejects_use_after_consume_affine_token() {
        let sched = ScheduleHint::UNFUSED;
        let prog = RulProgram {
            capability_mask: CAP_NONE,
            arenas: vec![ArenaDescriptor {
                arena_id: 0,
                element_count: 64,
                type_lane: TypeLane::F64X8,
                mutable: true,
                alignment: 64,
                initial_version: 1,
            }],
            proofs: Vec::new(),
            nodes: vec![
                // Node 0: ReadLease Arena 0 at v1 (OK)
                NodeFrame {
                    opcode: Opcode::ArenaInput,
                    type_lane: TypeLane::F64X8,
                    schedule: sched,
                    proof_idx: 0,
                    token_op: TokenOp::read_lease(0, 1),
                    in_a: 0,
                    in_b: 0,
                },
                // Node 1: AffineConsume Arena 0 at v1 -> advances Arena 0 to v2 (OK)
                NodeFrame {
                    opcode: Opcode::ArenaMutate,
                    type_lane: TypeLane::F64X8,
                    schedule: sched,
                    proof_idx: 0,
                    token_op: TokenOp::affine_consume(0, 1),
                    in_a: 0,
                    in_b: 0,
                },
                // Node 2: Attempt to reuse stale Token v1 after consumption -> MUST FAIL!
                NodeFrame {
                    opcode: Opcode::ArenaInput,
                    type_lane: TypeLane::F64X8,
                    schedule: sched,
                    proof_idx: 0,
                    token_op: TokenOp::read_lease(0, 1),
                    in_a: 0,
                    in_b: 0,
                },
            ],
            output_node: 2,
        };

        let res = verify_program(&prog);
        assert!(res.is_err());
        if let Err(diag) = res {
            assert_eq!(diag.node_idx, 2);
            assert_eq!(
                diag.kind,
                VerifyErrorKind::StaleOrConsumedToken {
                    arena_id: 0,
                    provided_version: 1,
                    expected_version: 2,
                }
            );
        }
    }

    #[test]
    fn test_verifier_rejects_shape_mismatch_and_out_of_bounds_proof() {
        let sched = ScheduleHint::UNFUSED;
        // 1. Shape mismatch between Arena(0) len=64 and Arena(1) len=128
        let prog_mismatch = RulProgram {
            capability_mask: CAP_NONE,
            arenas: vec![
                ArenaDescriptor {
                    arena_id: 0,
                    element_count: 64,
                    type_lane: TypeLane::F64X8,
                    mutable: false,
                    alignment: 64,
                    initial_version: 1,
                },
                ArenaDescriptor {
                    arena_id: 1,
                    element_count: 128,
                    type_lane: TypeLane::F64X8,
                    mutable: false,
                    alignment: 64,
                    initial_version: 1,
                },
            ],
            proofs: Vec::new(),
            nodes: vec![
                NodeFrame {
                    opcode: Opcode::ArenaInput,
                    type_lane: TypeLane::F64X8,
                    schedule: sched,
                    proof_idx: 0,
                    token_op: TokenOp::read_lease(0, 1),
                    in_a: 0,
                    in_b: 0,
                },
                NodeFrame {
                    opcode: Opcode::ArenaInput,
                    type_lane: TypeLane::F64X8,
                    schedule: sched,
                    proof_idx: 0,
                    token_op: TokenOp::read_lease(1, 1),
                    in_a: 1,
                    in_b: 0,
                },
                NodeFrame {
                    opcode: Opcode::Add,
                    type_lane: TypeLane::F64X8,
                    schedule: sched,
                    proof_idx: 0,
                    token_op: TokenOp::PURE,
                    in_a: 0,
                    in_b: 1,
                },
            ],
            output_node: 2,
        };
        let res = verify_program(&prog_mismatch);
        assert!(matches!(
            res,
            Err(d) if d.node_idx == 2 && matches!(d.kind, VerifyErrorKind::ShapeMismatch { .. })
        ));

        // 2. Out-of-bounds ArenaSlice proof (bound_hi = 100 > arena_len = 64)
        let prog_bad_proof = RulProgram {
            capability_mask: CAP_NONE,
            arenas: vec![ArenaDescriptor {
                arena_id: 0,
                element_count: 64,
                type_lane: TypeLane::F64X8,
                mutable: false,
                alignment: 64,
                initial_version: 1,
            }],
            proofs: vec![ProofEntry {
                rule_kind: ProofRuleKind::SliceBounds,
                target_arena: 0,
                bound_lo: 0,
                bound_hi: 100,
                witness_node: 0,
            }],
            nodes: vec![
                NodeFrame {
                    opcode: Opcode::ArenaInput,
                    type_lane: TypeLane::F64X8,
                    schedule: sched,
                    proof_idx: 0,
                    token_op: TokenOp::read_lease(0, 1),
                    in_a: 0,
                    in_b: 0,
                },
                NodeFrame {
                    opcode: Opcode::ArenaSlice,
                    type_lane: TypeLane::F64X8,
                    schedule: sched,
                    proof_idx: 1,
                    token_op: TokenOp::PURE,
                    in_a: 0,
                    in_b: 0,
                },
            ],
            output_node: 1,
        };
        let res_proof = verify_program(&prog_bad_proof);
        assert!(matches!(
            res_proof,
            Err(d) if d.node_idx == 1 && matches!(d.kind, VerifyErrorKind::InvalidRefinementProof { .. })
        ));
    }

    #[test]
    fn test_verifier_rejects_unauthorized_capability() {
        let sched = ScheduleHint::UNFUSED;
        let prog = RulProgram {
            capability_mask: CAP_NONE, // Network capability NOT granted!
            arenas: Vec::new(),
            proofs: Vec::new(),
            nodes: vec![
                NodeFrame::const_f64(42.0, sched),
                NodeFrame {
                    opcode: Opcode::RingEnqueue,
                    type_lane: TypeLane::F64_SCALAR,
                    schedule: sched,
                    proof_idx: 0,
                    token_op: TokenOp::PURE,
                    in_a: 0,
                    in_b: 0,
                },
            ],
            output_node: 1,
        };
        let res = verify_program(&prog);
        assert_eq!(
            res.err().map(|d| d.kind),
            Some(VerifyErrorKind::CapabilityViolation {
                required: CAP_NET_RING,
                granted: CAP_NONE,
            })
        );
    }

    #[test]
    fn test_differential_equivalence_tier0_vs_tier2_fused() {
        let data: Vec<f64> = (0..1024).map(|i| f64::from(i) * 2.5).collect();
        let prog = build_canonical_risk_pipeline(1024, 1000.0);

        let rep0 = execute_tier0_interpreter(&prog, &[&data]);
        let rep1 = execute_tier1_stencil(&prog, &[&data]);
        let rep2 = execute_tier2_fused(&prog, &[&data]);
        assert!(rep0.is_ok());
        assert!(rep1.is_ok());
        assert!(rep2.is_ok());

        if let (Ok(r0), Ok(r1), Ok(r2)) = (rep0, rep1, rep2) {
            assert_eq!(r0.output, r1.output);
            assert_eq!(r0.output, r2.output);
            // Tier 2 Fused Kernel must allocate ZERO intermediate heap bytes!
            assert_eq!(r2.telemetry.intermediate_bytes_allocated, 0);
            assert!(r0.telemetry.intermediate_bytes_allocated > 0);
        }

        // Verify AssertContract catches NaN values in input.
        let mut bad_data = data;
        if let Some(first) = bad_data.first_mut() {
            *first = f64::INFINITY;
        }
        let err_rep = execute_tier2_fused(&prog, &[&bad_data]);
        assert!(matches!(
            err_rep,
            Err(ExecError::ContractAssertionFailed { .. })
        ));
    }

    #[test]
    fn test_adversarial_cases_a_b_c_d() {
        let elem_count = 100_000_u32;
        let col_a: Vec<f64> = (0..elem_count).map(|i| f64::from(i % 2000) + 1.0).collect();

        // =========================================================================
        // CASE A: P02 with `$rate` (0.15) declared BEFORE `$thr` (500.0)
        // Verifies that Tier 2 traces exact `(in_a, in_b)` SSA operand edges rather
        // than guessing the first `ConstF64` in the node array.
        // =========================================================================
        {
            let mut b = SymbolicGraphBuilder::new(CAP_NONE);
            let a0 = b.add_arena(elem_count, TypeLane::F64X8, false);
            let s = ScheduleHint::fused_simd(1);
            assert!(b.emit_arena_input("$a", a0, 1, s).is_ok());
            // Declare $rate BEFORE $thr!
            assert!(b.emit_const_f64("$rate", 0.15, s).is_ok());
            assert!(b.emit_const_f64("$thr", 500.0, s).is_ok());
            assert!(b
                .emit_binary_op("$m", Opcode::CmpGt, TypeLane::MASKX8, "$a", "$thr", s)
                .is_ok());
            assert!(b
                .emit_binary_op("$tax", Opcode::Mul, TypeLane::F64X8, "$a", "$rate", s)
                .is_ok());
            assert!(b
                .emit_binary_op("$sel", Opcode::SelectZero, TypeLane::F64X8, "$m", "$tax", s)
                .is_ok());
            assert!(b
                .emit_unary_op("$sum", Opcode::ReduceSum, TypeLane::F64_SCALAR, "$sel", s)
                .is_ok());
            let prog = b.finish("$sum");
            assert!(prog.is_ok());
            if let Ok(p) = prog {
                let r0 = execute_tier0_interpreter(&p, &[&col_a]);
                let r2 = execute_tier2_fused(&p, &[&col_a]);
                assert!(r0.is_ok() && r2.is_ok());
                if let (Ok(rep0), Ok(rep2)) = (r0, r2) {
                    let (RuntimeValue::Scalar(v0), RuntimeValue::Scalar(v2)) =
                        (rep0.output, rep2.output)
                    else {
                        return;
                    };
                    assert!((v0 - v2).abs() < 1e-5);
                    assert!(
                        (v2 - 14_068_125.0).abs() < 1e-5,
                        "Case A: Tier 2 must match 14,068,125.0 even when $rate precedes $thr"
                    );
                }
            }
        }

        // =========================================================================
        // CASE B: Dead `Mul($a, $a)` node in the graph while `SelectZero` selects `$a`
        // Verifies backward liveness reachability from `output_node`.
        // =========================================================================
        {
            let mut b = SymbolicGraphBuilder::new(CAP_NONE);
            let a0 = b.add_arena(elem_count, TypeLane::F64X8, false);
            let s = ScheduleHint::fused_simd(1);
            assert!(b.emit_arena_input("$a", a0, 1, s).is_ok());
            assert!(b.emit_const_f64("$thr", 1000.0, s).is_ok());
            assert!(b
                .emit_binary_op("$m", Opcode::CmpGt, TypeLane::MASKX8, "$a", "$thr", s)
                .is_ok());
            // Dead node: $dead_sq = $a * $a (never referenced by $sel!)
            assert!(b
                .emit_binary_op("$dead_sq", Opcode::Mul, TypeLane::F64X8, "$a", "$a", s)
                .is_ok());
            assert!(b
                .emit_binary_op("$sel", Opcode::SelectZero, TypeLane::F64X8, "$m", "$a", s)
                .is_ok());
            assert!(b
                .emit_unary_op("$sum", Opcode::ReduceSum, TypeLane::F64_SCALAR, "$sel", s)
                .is_ok());
            let prog = b.finish("$sum");
            assert!(prog.is_ok());
            if let Ok(p) = prog {
                let r0 = execute_tier0_interpreter(&p, &[&col_a]);
                let r2 = execute_tier2_fused(&p, &[&col_a]);
                assert!(r0.is_ok() && r2.is_ok());
                if let (Ok(rep0), Ok(rep2)) = (r0, r2) {
                    assert_eq!(rep0.output, rep2.output);
                    assert_eq!(
                        rep2.output,
                        RuntimeValue::Scalar(75_025_000.0),
                        "Case B: Dead Mul($a, $a) node must not corrupt Identity SelectZero"
                    );
                }
            }
        }

        // =========================================================================
        // CASE C: Real `Gather` semantics (idx = 0.0 -> 1.0e5; idx = 1e9 -> Error)
        // + Static Verifier check on `ProofEntry.witness_node == frame.in_b`
        // =========================================================================
        {
            let mut b = SymbolicGraphBuilder::new(CAP_NONE);
            let a0 = b.add_arena(elem_count, TypeLane::F64X8, false);
            let a1 = b.add_arena(elem_count, TypeLane::F64X8, false);
            let p_gather = b.add_proof(ProofEntry {
                rule_kind: ProofRuleKind::GatherIndexInBounds,
                target_arena: 0,
                bound_lo: 0,
                bound_hi: elem_count,
                witness_node: 1,
            });
            let s = ScheduleHint::UNFUSED;
            assert!(b.emit_arena_input("$target", a0, 1, s).is_ok());
            assert!(b.emit_arena_input("$idx", a1, 1, s).is_ok());
            assert!(b
                .emit_gather("$gathered", "$target", "$idx", p_gather, s)
                .is_ok());
            assert!(b
                .emit_unary_op(
                    "$sum",
                    Opcode::ReduceSum,
                    TypeLane::F64_SCALAR,
                    "$gathered",
                    s
                )
                .is_ok());
            let prog = b.finish("$sum");
            assert!(prog.is_ok());
            if let Ok(p) = prog {
                // 1) All indices = 0.0 -> col_a[0] == 1.0 -> sum across 100,000 elements == 100,000.0
                let zeros = vec![0.0_f64; elem_count as usize];
                let r_ok = execute_tier2_fused(&p, &[&col_a, &zeros]);
                assert!(r_ok.is_ok());
                if let Ok(rep) = r_ok {
                    assert_eq!(rep.output, RuntimeValue::Scalar(100_000.0));
                }

                // 2) Out-of-bounds index = 1e9 -> MUST fail with GatherIndexOutOfBounds!
                let oob_indices = vec![1.0e9_f64; elem_count as usize];
                let r_oob = execute_tier2_fused(&p, &[&col_a, &oob_indices]);
                assert!(matches!(
                    r_oob,
                    Err(ExecError::GatherIndexOutOfBounds { .. })
                ));
            }
        }

        // =========================================================================
        // CASE D: ProofEntry targeting Arena(0) (len 100k) used to slice Arena(1) (len 10)
        // Verifies that `verify_program` rejects forged cross-arena proofs in O(N).
        // =========================================================================
        {
            let s = ScheduleHint::UNFUSED;
            let prog_forged = RulProgram {
                capability_mask: CAP_NONE,
                arenas: vec![
                    ArenaDescriptor {
                        arena_id: 0,
                        element_count: 100_000,
                        type_lane: TypeLane::F64X8,
                        mutable: false,
                        alignment: 64,
                        initial_version: 1,
                    },
                    ArenaDescriptor {
                        arena_id: 1,
                        element_count: 10,
                        type_lane: TypeLane::F64X8,
                        mutable: false,
                        alignment: 64,
                        initial_version: 1,
                    },
                ],
                proofs: vec![ProofEntry {
                    rule_kind: ProofRuleKind::SliceBounds,
                    target_arena: 0, // Proof claims Arena 0 (len 100,000)
                    bound_lo: 0,
                    bound_hi: 50_000,
                    witness_node: 0,
                }],
                nodes: vec![
                    // Node 0: Read Arena 1 (len 10)
                    NodeFrame {
                        opcode: Opcode::ArenaInput,
                        type_lane: TypeLane::F64X8,
                        schedule: s,
                        proof_idx: 0,
                        token_op: TokenOp::read_lease(1, 1),
                        in_a: 1,
                        in_b: 0,
                    },
                    // Node 1: Slice Node 0 (Arena 1) using Proof 1 (which targets Arena 0!)
                    NodeFrame {
                        opcode: Opcode::ArenaSlice,
                        type_lane: TypeLane::F64X8,
                        schedule: s,
                        proof_idx: 1,
                        token_op: TokenOp::PURE,
                        in_a: 0,
                        in_b: 0,
                    },
                ],
                output_node: 1,
            };

            let res = verify_program(&prog_forged);
            assert!(matches!(
                res,
                Err(diag) if diag.node_idx == 1 && diag.kind == VerifyErrorKind::ProofTargetArenaMismatch {
                    proof_target: 0,
                    actual_arena: Some(1),
                }
            ));
        }
    }

    #[test]
    fn test_multicore_and_zonemap_acceleration() {
        let elem_count = 65_536_u32;
        // Partitioned dataset: only the final 4,096 elements exceed threshold 1000.0
        let col_a: Vec<f64> = (0..elem_count)
            .map(|i| if i >= 61_440 { 1500.0 } else { 250.0 })
            .collect();

        let mut b = SymbolicGraphBuilder::new(CAP_NONE);
        let a0 = b.add_arena(elem_count, TypeLane::F64X8, false);
        let s = ScheduleHint {
            fusion_group: 1,
            policy: ExecPolicy::MultiCore,
        };
        assert!(b.emit_arena_input("$a", a0, 1, s).is_ok());
        assert!(b.emit_const_f64("$thr", 1000.0, s).is_ok());
        assert!(b
            .emit_binary_op("$m", Opcode::CmpGt, TypeLane::MASKX8, "$a", "$thr", s)
            .is_ok());
        assert!(b
            .emit_binary_op("$sq", Opcode::Mul, TypeLane::F64X8, "$a", "$a", s)
            .is_ok());
        assert!(b
            .emit_binary_op("$sel", Opcode::SelectZero, TypeLane::F64X8, "$m", "$sq", s)
            .is_ok());
        assert!(b
            .emit_unary_op("$sum", Opcode::ReduceSum, TypeLane::F64_SCALAR, "$sel", s)
            .is_ok());
        let prog = b.finish("$sum");
        assert!(prog.is_ok());

        if let Ok(p) = prog {
            let zmap = ArenaZoneMap::build(&col_a);
            let r0 = execute_tier0_interpreter(&p, &[&col_a]);
            let r_mc = execute_tier2_fused(&p, &[&col_a]);
            let r_zm = execute_tier2_with_zonemap(&p, &[&col_a], Some(&zmap));

            assert!(r0.is_ok() && r_mc.is_ok() && r_zm.is_ok());
            if let (Ok(rep0), Ok(rep_mc), Ok(rep_zm)) = (r0, r_mc, r_zm) {
                assert_eq!(rep0.output, rep_mc.output);
                assert_eq!(rep0.output, rep_zm.output);
                assert_eq!(rep_zm.telemetry.intermediate_bytes_allocated, 0);
            }
        }
    }

    #[test]
    fn test_10_pipeline_suite_and_rl_file_roundtrip() {
        let reports = crate::suite10::run_10_pipeline_suite(2048);
        assert!(reports.is_ok());
        if let Ok(list) = reports {
            assert_eq!(list.len(), 10);
            for r in &list {
                assert!(r.tiers_matched);
            }
        }

        let prog = build_canonical_risk_pipeline(1024, 1000.0);
        let tmp_path = std::env::temp_dir().join("rul_test_pipeline.rl");
        let written = crate::suite10::save_rl_file(&tmp_path, &prog);
        assert_eq!(written.ok(), Some(192));

        let loaded = crate::suite10::load_rl_file(&tmp_path);
        assert_eq!(loaded.ok(), Some(prog));
        let _ = std::fs::remove_file(&tmp_path);
    }

    #[test]
    fn test_rls_symbolic_roundtrip_and_shorthand_assembly() {
        let prog = build_canonical_risk_pipeline(1024, 1000.0);
        let rls_text = crate::transpiler::program_to_rls(&prog);
        assert!(rls_text.contains("ArenaInput"));
        assert!(rls_text.contains("ReduceSum"));

        let reassembled = crate::transpiler::assemble_rls(&rls_text);
        assert_eq!(reassembled, Ok(prog.clone()));

        // Verify concise AI shorthand syntax (.rls without verbose defaults) produces the exact same binary!
        let shorthand = "\
            # Concise AI Agent .rls program
            arena 0 len=1024 lane=f64x8
            node $amounts = ArenaInput f64x8 a=0
            node $threshold = ConstF64 f64x1 const=1000.0
            node $mask = CmpGt maskx8 a=$amounts b=$threshold
            node $squared = Mul f64x8 a=$amounts b=$amounts
            node $selected = SelectZero f64x8 a=$mask b=$squared
            node $sum = ReduceSum f64x1 a=$selected
            node $out = AssertContract f64x1 a=$sum
            output $out
        ";
        let short_prog = crate::transpiler::assemble_rls(shorthand);
        assert_eq!(short_prog, Ok(prog));
    }

    #[test]
    fn test_aot_standalone_elf64_zero_runtime_execution() {
        let data: Vec<f64> = (0..1024).map(|i| f64::from(i) * 2.5).collect();
        let prog = build_canonical_risk_pipeline(1024, 1000.0);

        let exe_path = std::env::temp_dir().join("rul_test_aot_risk_bin");
        let data_path = std::env::temp_dir().join("rul_test_aot_risk_data.bin");

        let report = crate::aot::compile_rl_to_executable(&prog, &exe_path);
        assert!(report.is_ok());
        if let Ok(rep) = report {
            assert!(rep.avx2_vectorized);
            assert!(rep.elf_bytes > 120 && rep.elf_bytes < 2048);
            // Sub-millisecond compile latency (< 5 ms even under debug builds)
            assert!(rep.total_compile_ns < 5_000_000);
        }

        assert!(crate::aot::write_arena_data_file(&data_path, &[&data]).is_ok());
        let aot_val = crate::aot::run_compiled_executable(&exe_path, &data_path);
        let tier2_rep = execute_tier2_fused(&prog, &[&data]);
        assert!(
            aot_val.is_ok(),
            "AOT execution failed with: {:?}",
            aot_val.as_ref().err()
        );
        assert!(tier2_rep.is_ok());
        if let (Ok(aot_out), Ok(t2)) = (aot_val, tier2_rep) {
            assert_eq!(RuntimeValue::Scalar(aot_out), t2.output);
        }

        // Verify standalone ELF binary enforces AssertContract on NaN/Infinity without a runtime!
        let mut bad_data = data;
        if let Some(first) = bad_data.first_mut() {
            *first = f64::INFINITY;
        }
        assert!(crate::aot::write_arena_data_file(&data_path, &[&bad_data]).is_ok());
        let bad_res = crate::aot::run_compiled_executable(&exe_path, &data_path);
        assert!(matches!(
            bad_res,
            Err(crate::aot::AotError::ContractAssertionFailed)
        ));

        let _ = std::fs::remove_file(&exe_path);
        let _ = std::fs::remove_file(&data_path);
    }

    #[test]
    fn test_aot_avx2_256bit_ymm_and_multi_arena_execution() {
        // Odd element count (1,027 = 256 * 4 + 3) exercises BOTH the 256-bit AVX2 YMM vector loop
        // AND the 3-element scalar tail cleanup across 2 simultaneous columnar Arenas!
        let elem_count = 1027_u32;
        let col_a: Vec<f64> = (0..elem_count)
            .map(|i| f64::from(i) * 1.25 + 10.0)
            .collect();
        let col_b: Vec<f64> = (0..elem_count).map(|i| f64::from(i % 50) * 0.5).collect();

        let mut b = SymbolicGraphBuilder::new(CAP_NONE);
        let a0 = b.add_arena(elem_count, TypeLane::F64X8, false);
        let a1 = b.add_arena(elem_count, TypeLane::F64X8, false);
        let s = ScheduleHint::fused_simd(1);
        assert!(b.emit_arena_input("$a", a0, 1, s).is_ok());
        assert!(b.emit_arena_input("$b", a1, 1, s).is_ok());
        assert!(b.emit_const_f64("$thr", 400.0, s).is_ok());
        assert!(b
            .emit_binary_op("$diff", Opcode::Sub, TypeLane::F64X8, "$a", "$b", s)
            .is_ok());
        assert!(b
            .emit_binary_op("$mask", Opcode::CmpGt, TypeLane::MASKX8, "$diff", "$thr", s)
            .is_ok());
        assert!(b
            .emit_binary_op("$prod", Opcode::Mul, TypeLane::F64X8, "$diff", "$b", s)
            .is_ok());
        assert!(b
            .emit_binary_op(
                "$sel",
                Opcode::SelectZero,
                TypeLane::F64X8,
                "$mask",
                "$prod",
                s
            )
            .is_ok());
        assert!(b
            .emit_unary_op("$sum", Opcode::ReduceSum, TypeLane::F64_SCALAR, "$sel", s)
            .is_ok());
        let prog = b.finish("$sum");
        assert!(prog.is_ok());

        if let Ok(p) = prog {
            let exe_path = std::env::temp_dir().join("rul_test_aot_avx2_multi_arena_bin");
            let data_path = std::env::temp_dir().join("rul_test_aot_avx2_multi_arena.bin");

            let rep = crate::aot::compile_rl_to_executable(&p, &exe_path);
            assert!(rep.is_ok());
            if let Ok(r) = rep {
                assert!(r.avx2_vectorized);
            }

            assert!(crate::aot::write_arena_data_file(&data_path, &[&col_a, &col_b]).is_ok());
            let aot_out = crate::aot::run_compiled_executable(&exe_path, &data_path);
            let tier0_out = execute_tier0_interpreter(&p, &[&col_a, &col_b]);
            assert!(aot_out.is_ok() && tier0_out.is_ok());
            if let (Ok(v_aot), Ok(r0)) = (aot_out, tier0_out) {
                let RuntimeValue::Scalar(v0) = r0.output else {
                    return;
                };
                assert!(
                    (v_aot - v0).abs() < 1e-6,
                    "AVX2 Multi-Arena AOT mismatch: aot={v_aot}, tier0={v0}"
                );
            }

            let _ = std::fs::remove_file(&exe_path);
            let _ = std::fs::remove_file(&data_path);
        }
    }

    #[test]
    fn test_autotune_schedule_and_in_memory_mcp_server() {
        let data: Vec<f64> = (0..4096).map(|i| f64::from(i) * 1.5).collect();
        let mut prog = build_canonical_risk_pipeline(4096, 1000.0);
        let zmap = ArenaZoneMap::build(&data);

        let tune_res = crate::engine::autotune_schedule(&mut prog, &[&data], Some(&zmap));
        assert!(tune_res.is_ok());
        if let Ok(rep) = tune_res {
            assert_eq!(rep.candidates.len(), 4);
            assert_eq!(rep.best_telemetry.intermediate_bytes_allocated, 0);
            for node in &prog.nodes {
                assert_eq!(node.schedule, rep.best_schedule);
            }
        }

        // Verify MCP JSON-RPC 2.0 protocol & all 4 tools in-memory
        let server = crate::mcp::RulMcpServer::new();
        let init_resp =
            server.handle_line(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#);
        assert!(init_resp.contains(r#""name":"rul-mcp""#));

        let list_resp = server.handle_line(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#);
        assert!(list_resp.contains("rul_assemble_and_verify"));
        assert!(list_resp.contains("rul_test_differential"));
        assert!(list_resp.contains("rul_autotune_schedule"));
        assert!(list_resp.contains("rul_compile_standalone"));

        let verify_req = r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"rul_assemble_and_verify","arguments":{"rls_source":"arena 0 len=1024 lane=f64x8\nnode $a = ArenaInput f64x8 a=0\nnode $s = ReduceSum f64x1 a=$a\noutput $s"}}}"#;
        let verify_resp = server.handle_line(verify_req);
        assert!(verify_resp.contains("STATUS: VERIFIED_OK"));
        assert!(verify_resp.contains(r#""isError":false"#));

        let diff_req = r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"rul_test_differential","arguments":{"rls_source":"arena 0 len=512 lane=f64x8\nnode $a = ArenaInput f64x8 a=0\nnode $s = ReduceSum f64x1 a=$a\noutput $s","sample_size":512}}}"#;
        let diff_resp = server.handle_line(diff_req);
        assert!(diff_resp.contains("STATUS: DIFFERENTIAL_EQUIVALENCE_PASS"));
        assert!(diff_resp.contains("DATA_SOURCE: SYNTHETIC_PATTERN"));

        let tune_req = r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"rul_autotune_schedule","arguments":{"rls_source":"arena 0 len=1024 lane=f64x8\nnode $a = ArenaInput f64x8 a=0\nnode $s = ReduceSum f64x1 a=$a\noutput $s","sample_size":1024}}}"#;
        let tune_resp = server.handle_line(tune_req);
        assert!(tune_resp.contains("STATUS: AUTOTUNE_COMPLETE"));
        assert!(tune_resp.contains("DATA_SOURCE: SYNTHETIC_PATTERN"));
        assert!(tune_resp.contains("OPTIMIZED_RLS"));

        let tmp_elf = std::env::temp_dir().join("rul_mcp_test_out.elf");
        let compile_req = format!(
            r#"{{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{{"name":"rul_compile_standalone","arguments":{{"rls_source":"arena 0 len=1024 lane=f64x8\nnode $a = ArenaInput f64x8 a=0\nnode $s = ReduceSum f64x1 a=$a\noutput $s","output_path":"{}"}}}}}}"#,
            tmp_elf.display()
        );
        let compile_resp = server.handle_line(&compile_req);
        assert!(compile_resp.contains("STATUS: AOT_ELF64_COMPILED"));
        let _ = std::fs::remove_file(&tmp_elf);

        let cas_req = r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"rul_cas_intrinsics","arguments":{}}}"#;
        let cas_resp = server.handle_line(cas_req);
        assert!(cas_resp.contains("STATUS: RUL_CAS_INTRINSICS_CATALOG"));
        assert!(cas_resp.contains("std::hash_join_sum_i64"));
    }

    #[test]
    fn test_cas_content_addressed_intrinsics_and_multi_type_ops() {
        use crate::cas::{
            builtin_intrinsics, exec_group_by_bucket_sum_i64, exec_hash_join_sum_i64,
            exec_simd_scan_count_byte_u8, exec_sort_i64_asc, lookup_intrinsic_by_hash,
            lookup_intrinsic_by_name, CasHash, IntrinsicKind, RulCasStore,
        };

        assert_eq!(builtin_intrinsics().len(), 5);
        let join_desc = lookup_intrinsic_by_name("std::hash_join_sum_i64");
        assert!(join_desc.is_some());
        if let Some(d) = join_desc {
            assert_eq!(d.kind, IntrinsicKind::HashJoinSumI64);
            assert_eq!(lookup_intrinsic_by_hash(d.hash), Some(d));
        }

        // 1. I64 Sort
        let unsorted = [42_i64, -7, 100, 0, 19];
        assert_eq!(exec_sort_i64_asc(&unsorted), vec![-7, 0, 19, 42, 100]);

        // 2. I64 HashJoinSum
        let keys_l = [101_i64, 202, 303, 101];
        let vals_l = [50_i64, 75, 120, 25]; // key 101 -> 75, key 202 -> 75, key 303 -> 120
        let keys_r = [101_i64, 303, 999]; // matches 101 (75) + 303 (120) = 195
        assert_eq!(exec_hash_join_sum_i64(&keys_l, &vals_l, &keys_r), 195);

        // 3. I64 GroupByBucketSum
        let g_keys = [0_i64, 1, 2, 3, 4, 5];
        let g_vals = [10_i64, 20, 30, 40, 50, 60];
        // Bucket 0 (keys 0, 3) = 50; Bucket 1 (keys 1, 4) = 70; Bucket 2 (keys 2, 5) = 90
        assert_eq!(
            exec_group_by_bucket_sum_i64(&g_keys, &g_vals, 3),
            vec![50, 70, 90]
        );

        // 4. U8 Byte-Stream SIMD Scan & FNV-1a Hash
        let json_payload = br#"{"id":1,"amt":100},{"id":2,"amt":250},{"id":3,"amt":500}"#;
        assert_eq!(exec_simd_scan_count_byte_u8(json_payload, b':'), 6);
        let h1 = CasHash::from_bytes(json_payload);
        let h2 = CasHash::from_bytes(json_payload);
        assert_eq!(h1, h2);

        // 5. Verified .rl Content-Addressed Store admission
        let mut store = RulCasStore::new();
        let prog = build_canonical_risk_pipeline(1024, 1000.0);
        let inserted = store.insert_verified(prog.clone());
        assert!(inserted.is_ok());
        if let Ok(hash) = inserted {
            assert_eq!(store.len(), 1);
            assert_eq!(store.get(hash).map(|e| &e.program), Some(&prog));
        }
    }

    #[test]
    fn test_ffi_embed_bridge_zero_panic_execution() {
        use crate::ffi::{
            rul_embed_autotune_1col, rul_embed_compile_elf64_to_buffer, rul_embed_exec_tier2_1col,
            rul_embed_verify_bytes, RUL_ERR_FORMAT, RUL_OK,
        };

        let prog = build_canonical_risk_pipeline(1024, 1000.0);
        let rl_bytes = prog.to_rl_bytes();
        let data: Vec<f64> = (0..1024).map(|i| f64::from(i) * 2.5).collect();

        let ver = rul_embed_verify_bytes(&rl_bytes);
        assert_eq!(ver.status, RUL_OK);
        assert_eq!(ver.node_count_or_err_idx, 7);
        assert_eq!(ver.zero_runtime_bounds_checks, 1);

        let bad_ver = rul_embed_verify_bytes(&[0x00, 0x01, 0x02]);
        assert_eq!(bad_ver.status, RUL_ERR_FORMAT);

        let exec_res = rul_embed_exec_tier2_1col(&rl_bytes, &data);
        assert_eq!(exec_res.status, RUL_OK);
        assert_eq!(exec_res.intermediate_bytes, 0);
        assert!(exec_res.scalar_output > 0.0);

        let mut tuned_buf = vec![0u8; rl_bytes.len()];
        let tune_res = rul_embed_autotune_1col(&rl_bytes, &data, &mut tuned_buf);
        assert_eq!(tune_res.status, RUL_OK);
        assert!((tune_res.scalar_output - exec_res.scalar_output).abs() < 1e-6);

        let mut elf_buf = vec![0u8; 4096];
        let aot_res = rul_embed_compile_elf64_to_buffer(&rl_bytes, &mut elf_buf);
        assert_eq!(aot_res.status, RUL_OK);
        assert!(aot_res.elf_bytes > 120);
        assert_eq!(elf_buf.get(0..4), Some(&[0x7F, b'E', b'L', b'F'][..]));
    }

    #[test]
    fn test_new_opcodes_div_min_max_cmplt_cmpeq_all_tiers_and_aot() -> Result<(), String> {
        // Odd length (1,027 = 256 * 4 + 3) exercises both AVX2 256-bit YMM vectors and SSE2 scalar tail
        let rls_src = "\
            arena 0 len=1027 lane=f64x8
            arena 1 len=1027 lane=f64x8
            node $a = ArenaInput f64x8 a=0
            node $b = ArenaInput f64x8 a=1
            node $c2 = ConstF64 f64x1 const=2.0
            node $c50 = ConstF64 f64x1 const=50.0
            node $quot = Div f64x8 a=$a b=$c2
            node $mn = Min f64x8 a=$quot b=$b
            node $mx = Max f64x8 a=$mn b=$c2
            node $lt = CmpLt maskx8 a=$mx b=$c50
            node $sel1 = SelectZero f64x8 a=$lt b=$mx
            node $eq = CmpEq maskx8 a=$b b=$c2
            node $bonus = SelectZero f64x8 a=$eq b=$c50
            node $comb = Add f64x8 a=$sel1 b=$bonus
            node $sum = ReduceSum f64x1 a=$comb
            output $sum
        ";
        let p = crate::transpiler::assemble_rls(rls_src).map_err(|e| e.to_string())?;

        // Verify lossless .rls roundtrip for all 5 new opcodes
        let decompiled = crate::transpiler::program_to_rls(&p);
        assert!(decompiled.contains("Div"));
        assert!(decompiled.contains("Min"));
        assert!(decompiled.contains("Max"));
        assert!(decompiled.contains("CmpLt"));
        assert!(decompiled.contains("CmpEq"));
        assert_eq!(crate::transpiler::assemble_rls(&decompiled), Ok(p.clone()));

        let col_a: Vec<f64> = (0..1027_u32).map(|i| f64::from(i % 120) + 1.0).collect();
        let col_b: Vec<f64> = (0..1027_u32).map(|i| f64::from(i % 40) + 0.5).collect();

        let rep0 = execute_tier0_interpreter(&p, &[&col_a, &col_b]).map_err(|e| e.to_string())?;
        let rep1 = execute_tier1_stencil(&p, &[&col_a, &col_b]).map_err(|e| e.to_string())?;
        let rep2 = execute_tier2_fused(&p, &[&col_a, &col_b]).map_err(|e| e.to_string())?;
        assert_eq!(rep0.output, rep1.output);
        assert_eq!(rep0.output, rep2.output);
        assert_eq!(rep2.telemetry.intermediate_bytes_allocated, 0);

        // Verify standalone Linux x86_64 AVX2 ELF64 AOT binary matches Tier 0
        let exe_path = std::env::temp_dir().join("rul_test_new_opcodes_aot_bin");
        let data_path = std::env::temp_dir().join("rul_test_new_opcodes_aot_data.bin");
        crate::aot::compile_rl_to_executable(&p, &exe_path).map_err(|e| e.to_string())?;
        crate::aot::write_arena_data_file(&data_path, &[&col_a, &col_b])
            .map_err(|e| e.to_string())?;
        let v_aot = crate::aot::run_compiled_executable(&exe_path, &data_path)
            .map_err(|e| e.to_string())?;
        let RuntimeValue::Scalar(v0) = rep0.output else {
            return Err("expected scalar output from Tier 0".to_owned());
        };
        assert!(
            (v_aot - v0).abs() < 1e-6,
            "New opcodes AOT mismatch: aot={v_aot}, tier0={v0}"
        );
        let _ = std::fs::remove_file(&exe_path);
        let _ = std::fs::remove_file(&data_path);
        Ok(())
    }

    #[test]
    fn test_community_ieee754_challenge_suite_and_edge_cases() -> Result<(), String> {
        use crate::engine::{format_trace_table, trace_program};

        // 1. Community Challenge Test 1: Column * Scalar (0.1)
        {
            let rls_t1 = "\
                cap 0x0
                arena 0 len=5 lane=f64x8 mut=false align=64 ver=1
                node $n0 = ArenaInput f64x8 fg=1 pol=simd proof=0 tok=read(0,1) a=0 b=0
                node $n1 = ConstF64 f64x1 fg=1 pol=simd proof=0 tok=pure const=0.1
                node $n2 = Mul f64x8 fg=1 pol=simd proof=0 tok=pure a=$n0 b=$n1
                output $n2
            ";
            let p1 = crate::transpiler::assemble_rls(rls_t1).map_err(|e| e.to_string())?;
            let in1 = [100.0_f64, 250.5, 99.9, 1000.0, 0.0];
            let rep1 = execute_tier2_fused(&p1, &[&in1]).map_err(|e| e.to_string())?;
            let RuntimeValue::ColumnF64(out1) = rep1.output else {
                return Err("expected ColumnF64 output".to_owned());
            };
            let expected1 = [10.0_f64, 25.05, 9.99, 100.0, 0.0];
            for (got, exp) in out1.iter().zip(expected1.iter()) {
                assert!((got - exp).abs() < 1e-12);
            }
        }

        // 2. Community Challenge Test 2: Multi-Arena Mul + ReduceMax (-3.0 * -2.0 = 6.0)
        {
            let rls_t2 = "\
                cap 0x0
                arena 0 len=4 lane=f64x8 mut=false align=64 ver=1
                arena 1 len=4 lane=f64x8 mut=false align=64 ver=1
                node $n0 = ArenaInput f64x8 fg=1 pol=simd proof=0 tok=read(0,1) a=0 b=0
                node $n1 = ArenaInput f64x8 fg=1 pol=simd proof=0 tok=read(1,1) a=1 b=0
                node $n2 = Mul f64x8 fg=1 pol=simd proof=0 tok=pure a=$n0 b=$n1
                node $n3 = ReduceMax f64x1 fg=1 pol=simd proof=0 tok=pure a=$n2 b=0
                output $n3
            ";
            let p2 = crate::transpiler::assemble_rls(rls_t2).map_err(|e| e.to_string())?;
            let a0 = [1.5_f64, 2.0, -3.0, 4.0];
            let a1 = [2.0_f64, 0.5, -2.0, 0.25];
            let rep2 = execute_tier2_fused(&p2, &[&a0, &a1]).map_err(|e| e.to_string())?;
            assert_eq!(rep2.output, RuntimeValue::Scalar(6.0));
        }

        // 3. Community Challenge Test 3: IEEE-754 Edge Cases (0*Inf=NaN, -0.0*1.0=-0.0, Overflow, Subnormal, 0.1*0.2)
        {
            let rls_t3 = "\
                cap 0x0
                arena 0 len=8 lane=f64x8 mut=false align=64 ver=1
                arena 1 len=8 lane=f64x8 mut=false align=64 ver=1
                node $n0 = ArenaInput f64x8 fg=1 pol=simd proof=0 tok=read(0,1) a=0 b=0
                node $n1 = ArenaInput f64x8 fg=1 pol=simd proof=0 tok=read(1,1) a=1 b=0
                node $n2 = Mul f64x8 fg=1 pol=simd proof=0 tok=pure a=$n0 b=$n1
                output $n2
            ";
            let p3 = crate::transpiler::assemble_rls(rls_t3).map_err(|e| e.to_string())?;
            let a0 = [
                0.0_f64,
                -0.0_f64,
                1.0e308,
                1.0e-308,
                5.0e-324,
                f64::NAN,
                0.1,
                1.0,
            ];
            let a1 = [f64::INFINITY, 1.0, 10.0, 1.0e-308, 0.5, 1.0, 0.2, 1.0];
            let rep3 = execute_tier2_fused(&p3, &[&a0, &a1]).map_err(|e| e.to_string())?;
            let RuntimeValue::ColumnF64(ref out3) = rep3.output else {
                return Err("expected ColumnF64 output".to_owned());
            };
            assert_eq!(out3.len(), 8);
            assert!(out3.first().is_some_and(|v| v.is_nan())); // 0.0 * Inf == NaN
            assert!(out3
                .get(1)
                .is_some_and(|v| *v == 0.0 && v.is_sign_negative())); // -0.0 * 1.0 == -0.0
            assert_eq!(out3.get(2).copied(), Some(f64::INFINITY)); // 1e308 * 10.0 == +Inf
            assert!(out3
                .get(3)
                .is_some_and(|v| *v == 0.0 && v.is_sign_positive())); // 1e-308 * 1e-308 == +0.0
            assert!(out3
                .get(4)
                .is_some_and(|v| *v == 0.0 && v.is_sign_positive())); // 5e-324 * 0.5 == +0.0
            assert!(out3.get(5).is_some_and(|v| v.is_nan())); // NaN * 1.0 == NaN
            assert_eq!(
                out3.get(6).map(|v| v.to_bits()),
                Some((0.1_f64 * 0.2_f64).to_bits())
            );
            assert_eq!(out3.get(7).copied(), Some(1.0));

            // Verify White-Box Execution Tracer (`trace_program`) detects the 3 non-finite values (`NaN`, `Inf`, `NaN`)
            let trace_rep = trace_program(&p3, &[&a0, &a1]).map_err(|e| e.to_string())?;
            assert_eq!(trace_rep.entries.len(), 3);
            assert_eq!(
                trace_rep.entries.get(2).map(|e| e.nan_or_inf_count),
                Some(3)
            );
            let table_str = format_trace_table(&trace_rep);
            assert!(table_str.contains("WHITE-BOX EXECUTION TRACE"));
            assert!(table_str.contains("Mul"));
        }

        // 4. Verifier Hardening: u16::MAX TokenVersionOverflow & Gather zero_runtime_bounds_checks == false
        {
            let sched = ScheduleHint::UNFUSED;
            let prog_overflow = RulProgram {
                capability_mask: CAP_NONE,
                arenas: vec![ArenaDescriptor {
                    arena_id: 0,
                    element_count: 8,
                    type_lane: TypeLane::F64X8,
                    mutable: true,
                    alignment: 64,
                    initial_version: u16::MAX,
                }],
                proofs: Vec::new(),
                nodes: vec![
                    NodeFrame::const_f64(1.0, sched),
                    NodeFrame {
                        opcode: Opcode::ArenaMutate,
                        type_lane: TypeLane::F64X8,
                        schedule: sched,
                        proof_idx: 0,
                        token_op: TokenOp::affine_consume(0, u16::MAX),
                        in_a: 0,
                        in_b: 0,
                    },
                ],
                output_node: 1,
            };
            assert_eq!(
                verify_program(&prog_overflow).err().map(|d| d.kind),
                Some(VerifyErrorKind::TokenVersionOverflow(0))
            );

            // Fractional Gather index (2.7) must be rejected with GatherIndexOutOfBounds
            let rls_gather = "\
                arena 0 len=8 lane=f64x8
                arena 1 len=8 lane=f64x8
                proof 1 rule=gather_bounds target=0 lo=0 hi=8 witness=1
                node $t = ArenaInput f64x8 a=0
                node $idx = ArenaInput f64x8 a=1
                node $g = Gather f64x8 proof=1 a=$t b=$idx
                output $g
            ";
            let pg = crate::transpiler::assemble_rls(rls_gather).map_err(|e| e.to_string())?;
            let summary = verify_program(&pg).map_err(|e| e.to_string())?;
            assert!(!summary.zero_runtime_bounds_checks);
            let target = [10.0_f64; 8];
            let frac_idx = [0.0_f64, 1.0, 2.7, 3.0, 0.0, 0.0, 0.0, 0.0];
            assert!(matches!(
                execute_tier0_interpreter(&pg, &[&target, &frac_idx]),
                Err(ExecError::GatherIndexOutOfBounds { .. })
            ));
        }

        // 5. MCP `rul_trace` tool test
        let server = crate::mcp::RulMcpServer::new();
        let trace_req = r#"{"jsonrpc":"2.0","id":99,"method":"tools/call","params":{"name":"rul_trace","arguments":{"rls_source":"arena 0 len=16 lane=f64x8\nnode $a = ArenaInput f64x8 a=0\nnode $s = ReduceSum f64x1 a=$a\noutput $s","sample_size":16}}}"#;
        let trace_resp = server.handle_line(trace_req);
        assert!(trace_resp.contains("STATUS: WHITE_BOX_TRACE_OK"));
        assert!(trace_resp.contains("DATA_SOURCE: SYNTHETIC_PATTERN"));
        Ok(())
    }

    #[test]
    fn test_v040_math_and_logic_isa_all_tiers_and_aot() -> Result<(), String> {
        // Odd length (1,027 = 256 * 4 + 3) exercises both AVX2 256-bit YMM vectors and scalar tail
        // across all 9 v0.4.0 opcodes: Abs, Sqrt, Floor, Ceil, Ln, Exp, MaskAnd, MaskOr, ReduceMin
        let rls_src = "\
            arena 0 len=1027 lane=f64x8
            arena 1 len=1027 lane=f64x8
            node $a = ArenaInput f64x8 a=0
            node $b = ArenaInput f64x8 a=1
            node $c_neg = ConstF64 f64x1 const=-3.25
            node $c_pos = ConstF64 f64x1 const=1.0
            node $c_scale = ConstF64 f64x1 const=0.05
            node $c_hi = ConstF64 f64x1 const=90.0
            node $c_lo = ConstF64 f64x1 const=10.0
            node $shifted = Add f64x8 a=$a b=$c_neg
            node $abs_a = Abs f64x8 a=$shifted
            node $fl = Floor f64x8 a=$abs_a
            node $cl = Ceil f64x8 a=$abs_a
            node $avg_round = Add f64x8 a=$fl b=$cl
            node $plus1 = Add f64x8 a=$avg_round b=$c_pos
            node $rt = Sqrt f64x8 a=$plus1
            node $log_v = Ln f64x8 a=$plus1
            node $scaled = Mul f64x8 a=$b b=$c_scale
            node $exp_v = Exp f64x8 a=$scaled
            node $math_sum = Add f64x8 a=$rt b=$log_v
            node $math_tot = Mul f64x8 a=$math_sum b=$exp_v
            node $m_gt = CmpGt maskx8 a=$a b=$c_lo
            node $m_lt = CmpLt maskx8 a=$a b=$c_hi
            node $m_and = MaskAnd maskx8 a=$m_gt b=$m_lt
            node $m_eq = CmpEq maskx8 a=$b b=$c_pos
            node $m_or = MaskOr maskx8 a=$m_and b=$m_eq
            node $sel = SelectZero f64x8 a=$m_or b=$math_tot
            node $sum = ReduceSum f64x1 a=$sel
            output $sum
        ";
        let p = crate::transpiler::assemble_rls(rls_src).map_err(|e| e.to_string())?;

        let decompiled = crate::transpiler::program_to_rls(&p);
        for op_name in [
            "Abs", "Sqrt", "Floor", "Ceil", "Ln", "Exp", "MaskAnd", "MaskOr",
        ] {
            assert!(decompiled.contains(op_name), "missing {op_name} in .rls");
        }
        assert_eq!(crate::transpiler::assemble_rls(&decompiled), Ok(p.clone()));

        let col_a: Vec<f64> = (0..1027_u32)
            .map(|i| f64::from(i % 100) * 1.15 - 15.0)
            .collect();
        let col_b: Vec<f64> = (0..1027_u32).map(|i| f64::from(i % 25) * 0.5).collect();

        let rep0 = execute_tier0_interpreter(&p, &[&col_a, &col_b]).map_err(|e| e.to_string())?;
        let rep1 = execute_tier1_stencil(&p, &[&col_a, &col_b]).map_err(|e| e.to_string())?;
        let rep2 = execute_tier2_fused(&p, &[&col_a, &col_b]).map_err(|e| e.to_string())?;
        assert_eq!(rep0.output, rep1.output);
        let RuntimeValue::Scalar(v0) = rep0.output else {
            return Err("expected Scalar from Tier 0".to_owned());
        };
        let RuntimeValue::Scalar(v2) = rep2.output else {
            return Err("expected Scalar from Tier 2".to_owned());
        };
        assert!((v0 - v2).abs() < 1e-9, "tier0={v0} vs tier2={v2}");
        assert_eq!(rep2.telemetry.intermediate_bytes_allocated, 0);

        // Verify standalone Linux x86_64 AVX2 ELF64 AOT binary matches Tier 0
        let exe_path = std::env::temp_dir().join("rul_test_v040_math_aot_bin");
        let data_path = std::env::temp_dir().join("rul_test_v040_math_aot_data.bin");
        crate::aot::compile_rl_to_executable(&p, &exe_path).map_err(|e| e.to_string())?;
        crate::aot::write_arena_data_file(&data_path, &[&col_a, &col_b])
            .map_err(|e| e.to_string())?;
        let v_aot = crate::aot::run_compiled_executable(&exe_path, &data_path)
            .map_err(|e| e.to_string())?;
        let rel_err = (v_aot - v0).abs() / v0.abs().max(1.0);
        assert!(
            rel_err < 1e-9,
            "v0.4.0 Math AOT mismatch: aot={v_aot}, tier0={v0}, rel_err={rel_err}"
        );
        let _ = std::fs::remove_file(&exe_path);
        let _ = std::fs::remove_file(&data_path);

        // Also test ReduceMin across Tier 0, Tier 1, Tier 2, and standalone AOT
        let rls_min = "\
            arena 0 len=1027 lane=f64x8
            node $a = ArenaInput f64x8 a=0
            node $abs = Abs f64x8 a=$a
            node $rt = Sqrt f64x8 a=$abs
            node $min = ReduceMin f64x1 a=$rt
            output $min
        ";
        let p_min = crate::transpiler::assemble_rls(rls_min).map_err(|e| e.to_string())?;
        let decomp_min = crate::transpiler::program_to_rls(&p_min);
        assert!(decomp_min.contains("ReduceMin"));
        assert_eq!(
            crate::transpiler::assemble_rls(&decomp_min),
            Ok(p_min.clone())
        );

        let m0 = execute_tier0_interpreter(&p_min, &[&col_a]).map_err(|e| e.to_string())?;
        let m1 = execute_tier1_stencil(&p_min, &[&col_a]).map_err(|e| e.to_string())?;
        let m2 = execute_tier2_fused(&p_min, &[&col_a]).map_err(|e| e.to_string())?;
        assert_eq!(m0.output, m1.output);
        assert_eq!(m0.output, m2.output);
        assert_eq!(m2.telemetry.intermediate_bytes_allocated, 0);

        let exe_min = std::env::temp_dir().join("rul_test_v040_reducemin_aot_bin");
        let data_min = std::env::temp_dir().join("rul_test_v040_reducemin_aot_data.bin");
        crate::aot::compile_rl_to_executable(&p_min, &exe_min).map_err(|e| e.to_string())?;
        crate::aot::write_arena_data_file(&data_min, &[&col_a]).map_err(|e| e.to_string())?;
        let vm_aot =
            crate::aot::run_compiled_executable(&exe_min, &data_min).map_err(|e| e.to_string())?;
        let RuntimeValue::Scalar(vm0) = m0.output else {
            return Err("expected Scalar from ReduceMin Tier 0".to_owned());
        };
        assert!((vm_aot - vm0).abs() < 1e-12);
        let _ = std::fs::remove_file(&exe_min);
        let _ = std::fs::remove_file(&data_min);

        // Verify post-reduction scalar operations (`Sqrt(ReduceSum)`) and multi-reduction scalar DAGs
        // (`(ReduceSum + ReduceMin) / ReduceMax`) execute in Tier 2 (`0 bytes` intermediate allocation)
        // and compile to standalone `ELF64` AOT binaries matching Tier 0!
        let rls_post_reduce = "\
            arena 0 len=1027 lane=f64x8
            node $a = ArenaInput f64x8 a=0
            node $abs = Abs f64x8 a=$a
            node $s = ReduceSum f64x1 a=$abs
            node $rt = Sqrt f64x1 a=$s
            output $rt
        ";
        let p_post = crate::transpiler::assemble_rls(rls_post_reduce).map_err(|e| e.to_string())?;
        let pr0 = execute_tier0_interpreter(&p_post, &[&col_a]).map_err(|e| e.to_string())?;
        let pr2 = execute_tier2_fused(&p_post, &[&col_a]).map_err(|e| e.to_string())?;
        assert_eq!(pr2.telemetry.intermediate_bytes_allocated, 0);
        let RuntimeValue::Scalar(v_pr0) = pr0.output else {
            return Err("expected Scalar from Sqrt(ReduceSum)".to_owned());
        };
        let RuntimeValue::Scalar(v_pr2) = pr2.output else {
            return Err("expected Scalar from Tier 2 Sqrt(ReduceSum)".to_owned());
        };
        assert!((v_pr0 - v_pr2).abs() < 1e-12);
        let exe_pr = std::env::temp_dir().join("rul_test_post_reduce_aot_bin");
        let data_pr = std::env::temp_dir().join("rul_test_post_reduce_aot_data.bin");
        crate::aot::compile_rl_to_executable(&p_post, &exe_pr).map_err(|e| e.to_string())?;
        crate::aot::write_arena_data_file(&data_pr, &[&col_a]).map_err(|e| e.to_string())?;
        let v_pr_aot =
            crate::aot::run_compiled_executable(&exe_pr, &data_pr).map_err(|e| e.to_string())?;
        assert!((v_pr_aot - v_pr0).abs() < 1e-12);
        let _ = std::fs::remove_file(&exe_pr);
        let _ = std::fs::remove_file(&data_pr);

        let rls_multi_reduce = "\
            arena 0 len=1027 lane=f64x8
            node $a = ArenaInput f64x8 a=0
            node $abs = Abs f64x8 a=$a
            node $s1 = ReduceSum f64x1 a=$abs
            node $s2 = ReduceMax f64x1 a=$abs
            node $s3 = ReduceMin f64x1 a=$abs
            node $num = Add f64x1 a=$s1 b=$s3
            node $q = Div f64x1 a=$num b=$s2
            output $q
        ";
        let p_multi =
            crate::transpiler::assemble_rls(rls_multi_reduce).map_err(|e| e.to_string())?;
        let mr0 = execute_tier0_interpreter(&p_multi, &[&col_a]).map_err(|e| e.to_string())?;
        let mr2 = execute_tier2_fused(&p_multi, &[&col_a]).map_err(|e| e.to_string())?;
        assert_eq!(mr2.telemetry.intermediate_bytes_allocated, 0);
        let RuntimeValue::Scalar(v_mr0) = mr0.output else {
            return Err("expected Scalar from multi-reduce Tier 0".to_owned());
        };
        let RuntimeValue::Scalar(v_mr2) = mr2.output else {
            return Err("expected Scalar from multi-reduce Tier 2".to_owned());
        };
        assert!((v_mr0 - v_mr2).abs() < 1e-12);
        let exe_mr = std::env::temp_dir().join("rul_test_multi_reduce_aot_bin");
        let data_mr = std::env::temp_dir().join("rul_test_multi_reduce_aot_data.bin");
        crate::aot::compile_rl_to_executable(&p_multi, &exe_mr).map_err(|e| e.to_string())?;
        crate::aot::write_arena_data_file(&data_mr, &[&col_a]).map_err(|e| e.to_string())?;
        let v_mr_aot =
            crate::aot::run_compiled_executable(&exe_mr, &data_mr).map_err(|e| e.to_string())?;
        assert!((v_mr_aot - v_mr0).abs() < 1e-12);
        let _ = std::fs::remove_file(&exe_mr);
        let _ = std::fs::remove_file(&data_mr);

        // Verify `ReduceMin(Ln(a))` and `ReduceMax(Ln(a))` ignore `NaN` across both AVX2 vector
        // lanes and scalar cleanup tail (matching `f64::min` / `f64::max` in Tier 0/1/2).
        let nan_col: Vec<f64> = vec![
            -3.5, 2.043, -1.0, 5.5, // AVX2 vector 0 (contains NaNs after Ln)
            10.0, -9.2, 3.1, -0.5, // AVX2 vector 1 (contains NaNs after Ln)
            -2.0, 4.2, -7.0, // Scalar tail (contains NaNs after Ln)
        ];
        let rls_nan_minmax = "\
            arena 0 len=11 lane=f64x8
            node $a = ArenaInput f64x8 a=0
            node $ln = Ln f64x8 a=$a
            node $min = ReduceMin f64x1 a=$ln
            node $max = ReduceMax f64x1 a=$ln
            node $sum = Add f64x1 a=$min b=$max
            output $sum
        ";
        let p_nan_mm =
            crate::transpiler::assemble_rls(rls_nan_minmax).map_err(|e| e.to_string())?;
        let nm0 = execute_tier0_interpreter(&p_nan_mm, &[&nan_col]).map_err(|e| e.to_string())?;
        let nm1 = execute_tier1_stencil(&p_nan_mm, &[&nan_col]).map_err(|e| e.to_string())?;
        let nm2 = execute_tier2_fused(&p_nan_mm, &[&nan_col]).map_err(|e| e.to_string())?;
        let RuntimeValue::Scalar(v_nm0) = nm0.output else {
            return Err("expected Scalar from NaN ReduceMin/ReduceMax Tier 0".to_owned());
        };
        let RuntimeValue::Scalar(v_nm1) = nm1.output else {
            return Err("expected Scalar from NaN ReduceMin/ReduceMax Tier 1".to_owned());
        };
        let RuntimeValue::Scalar(v_nm2) = nm2.output else {
            return Err("expected Scalar from NaN ReduceMin/ReduceMax Tier 2".to_owned());
        };
        assert!(v_nm0.is_finite());
        assert!((v_nm0 - v_nm1).abs() < 1e-12);
        assert!((v_nm0 - v_nm2).abs() < 1e-12);
        let exe_nm = std::env::temp_dir().join("rul_test_nan_minmax_aot_bin");
        let data_nm = std::env::temp_dir().join("rul_test_nan_minmax_aot_data.bin");
        crate::aot::compile_rl_to_executable(&p_nan_mm, &exe_nm).map_err(|e| e.to_string())?;
        crate::aot::write_arena_data_file(&data_nm, &[&nan_col]).map_err(|e| e.to_string())?;
        let v_nm_aot =
            crate::aot::run_compiled_executable(&exe_nm, &data_nm).map_err(|e| e.to_string())?;
        assert!(
            v_nm_aot.is_finite() && (v_nm_aot - v_nm0).abs() < 1e-12,
            "NaN ReduceMin/ReduceMax AOT mismatch: aot={v_nm_aot}, tier0={v_nm0}"
        );
        let _ = std::fs::remove_file(&exe_nm);
        let _ = std::fs::remove_file(&data_nm);

        // Verify AOT fails closed (`UnsupportedGraph`) when a Column node depends on a Reduce result
        // (2-pass graph such as `x - sum(x)`) or when FilterCompact is combined with multiple reductions.
        let rls_two_pass = "\
            arena 0 len=16 lane=f64x8
            node $a = ArenaInput f64x8 a=0
            node $s = ReduceSum f64x1 a=$a
            node $centered = Sub f64x8 a=$a b=$s
            node $out = ReduceSum f64x1 a=$centered
            output $out
        ";
        let p_two_pass =
            crate::transpiler::assemble_rls(rls_two_pass).map_err(|e| e.to_string())?;
        assert!(matches!(
            crate::aot::compile_to_elf64_bytes(&p_two_pass),
            Err(crate::aot::AotError::UnsupportedGraph(_))
        ));

        let rls_filter_multi = "\
            arena 0 len=16 lane=f64x8
            node $a = ArenaInput f64x8 a=0
            node $zero = ConstF64 f64x1 const=0.0
            node $m = CmpGt maskx8 a=$a b=$zero
            node $fc = FilterCompact f64x8 a=$m b=$a
            node $s1 = ReduceSum f64x1 a=$fc
            node $s2 = ReduceMax f64x1 a=$fc
            node $out = Add f64x1 a=$s1 b=$s2
            output $out
        ";
        let p_filter_multi =
            crate::transpiler::assemble_rls(rls_filter_multi).map_err(|e| e.to_string())?;
        assert!(matches!(
            crate::aot::compile_to_elf64_bytes(&p_filter_multi),
            Err(crate::aot::AotError::UnsupportedGraph(_))
        ));

        // Verify static type-checker rejects invalid operand element types on MaskAnd and Sqrt
        let bad_mask_rls = "\
            arena 0 len=8 lane=f64x8
            node $a = ArenaInput f64x8 a=0
            node $bad = MaskAnd maskx8 a=$a b=$a
            output $bad
        ";
        let p_bad_mask =
            crate::transpiler::assemble_rls(bad_mask_rls).map_err(|e| e.to_string())?;
        assert_eq!(
            verify_program(&p_bad_mask).err().map(|d| d.kind),
            Some(VerifyErrorKind::ElementTypeMismatch {
                expected: crate::bytecode::ElemType::BoolMask,
                actual: crate::bytecode::ElemType::F64,
            })
        );

        let bad_sqrt_rls = "\
            arena 0 len=8 lane=f64x8
            node $a = ArenaInput f64x8 a=0
            node $c = ConstF64 f64x1 const=1.0
            node $m = CmpGt maskx8 a=$a b=$c
            node $bad = Sqrt f64x8 a=$m
            output $bad
        ";
        let p_bad_sqrt =
            crate::transpiler::assemble_rls(bad_sqrt_rls).map_err(|e| e.to_string())?;
        assert_eq!(
            verify_program(&p_bad_sqrt).err().map(|d| d.kind),
            Some(VerifyErrorKind::ElementTypeMismatch {
                expected: crate::bytecode::ElemType::F64,
                actual: crate::bytecode::ElemType::BoolMask,
            })
        );
        Ok(())
    }

    #[test]
    fn test_memory_engine_4factor_hybrid_scoring_and_cosine_similarity() -> Result<(), String> {
        // Part 1: 4-Factor Hybrid Scoring Pipeline across all 4 Memory Tiers from MEMORY_ENGINE_DESIGN.md:
        //   - core:       lambda = 0.0   (infinite half-life)
        //   - semantic:   lambda = 0.005 (~139 days half-life)
        //   - procedural: lambda = 0.02  (~35 days half-life)
        //   - episodic:   lambda = 0.15  (~4.6 days half-life)
        // Formula:
        //   final_score[i] = rrf[i] * (1.0 + 0.15 * ln(1.0 + access[i])) * (0.5 + feedback[i]) * exp(-lambda * age_days[i])
        let rrf: Vec<f64> = (0..1024_u32)
            .map(|i| 1.0 / (60.0 + f64::from(i % 50)))
            .collect();
        let access: Vec<f64> = (0..1024_u32).map(|i| f64::from(i % 30)).collect();
        let feedback: Vec<f64> = (0..1024_u32)
            .map(|i| (f64::from(i % 10) * 0.1).clamp(0.05, 0.95))
            .collect();
        let age: Vec<f64> = (0..1024_u32).map(|i| f64::from(i % 90)).collect();

        for lambda in [0.0_f64, 0.005, 0.02, 0.15] {
            let neg_lambda = -lambda;
            let rls_scorer = format!(
                "\
                arena 0 len=1024 lane=f64x8
                arena 1 len=1024 lane=f64x8
                arena 2 len=1024 lane=f64x8
                arena 3 len=1024 lane=f64x8
                node $rrf = ArenaInput f64x8 a=0
                node $access = ArenaInput f64x8 a=1
                node $feedback = ArenaInput f64x8 a=2
                node $age = ArenaInput f64x8 a=3
                node $c1 = ConstF64 f64x1 const=1.0
                node $c015 = ConstF64 f64x1 const=0.15
                node $c05 = ConstF64 f64x1 const=0.5
                node $neg_lambda = ConstF64 f64x1 const={neg_lambda:.6}
                node $acc_p1 = Add f64x8 a=$access b=$c1
                node $ln_acc = Ln f64x8 a=$acc_p1
                node $heb_scaled = Mul f64x8 a=$ln_acc b=$c015
                node $hebbian = Add f64x8 a=$heb_scaled b=$c1
                node $fb_boost = Add f64x8 a=$feedback b=$c05
                node $exp_arg = Mul f64x8 a=$age b=$neg_lambda
                node $decay = Exp f64x8 a=$exp_arg
                node $s1 = Mul f64x8 a=$rrf b=$hebbian
                node $s2 = Mul f64x8 a=$s1 b=$fb_boost
                node $final_scores = Mul f64x8 a=$s2 b=$decay
                node $best_score = ReduceMax f64x1 a=$final_scores
                output $best_score
            "
            );
            let prog = crate::transpiler::assemble_rls(&rls_scorer).map_err(|e| e.to_string())?;
            let rep0 = execute_tier0_interpreter(&prog, &[&rrf, &access, &feedback, &age])
                .map_err(|e| e.to_string())?;
            let rep1 = execute_tier1_stencil(&prog, &[&rrf, &access, &feedback, &age])
                .map_err(|e| e.to_string())?;
            let rep2 = execute_tier2_fused(&prog, &[&rrf, &access, &feedback, &age])
                .map_err(|e| e.to_string())?;
            assert_eq!(rep0.output, rep1.output);
            assert_eq!(rep2.telemetry.intermediate_bytes_allocated, 0);

            let RuntimeValue::Scalar(v0) = rep0.output else {
                return Err("expected Scalar from Tier 0 scorer".to_owned());
            };
            let RuntimeValue::Scalar(v2) = rep2.output else {
                return Err("expected Scalar from Tier 2 scorer".to_owned());
            };
            assert!((v0 - v2).abs() < 1e-12);

            let exe_scorer = std::env::temp_dir().join("rul_test_memory_scorer_aot_bin");
            let data_scorer = std::env::temp_dir().join("rul_test_memory_scorer_aot_data.bin");
            let compile_rep = crate::aot::compile_rl_to_executable(&prog, &exe_scorer)
                .map_err(|e| e.to_string())?;
            assert!(compile_rep.avx2_vectorized);
            crate::aot::write_arena_data_file(&data_scorer, &[&rrf, &access, &feedback, &age])
                .map_err(|e| e.to_string())?;
            let v_aot = crate::aot::run_compiled_executable(&exe_scorer, &data_scorer)
                .map_err(|e| e.to_string())?;
            assert!((v_aot - v0).abs() < 1e-12);
            let _ = std::fs::remove_file(&exe_scorer);
            let _ = std::fs::remove_file(&data_scorer);
        }

        // Part 2: L2-Normalized Cosine Similarity: sum(q_i * d_i) / (sqrt(sum(q_i^2)) * sqrt(sum(d_i^2)))
        let rls_cosine = "\
            arena 0 len=384 lane=f64x8
            arena 1 len=384 lane=f64x8
            node $q = ArenaInput f64x8 a=0
            node $d = ArenaInput f64x8 a=1
            node $qd = Mul f64x8 a=$q b=$d
            node $qq = Mul f64x8 a=$q b=$q
            node $dd = Mul f64x8 a=$d b=$d
            node $dot = ReduceSum f64x1 a=$qd
            node $sum_qq = ReduceSum f64x1 a=$qq
            node $sum_dd = ReduceSum f64x1 a=$dd
            node $norm_q = Sqrt f64x1 a=$sum_qq
            node $norm_d = Sqrt f64x1 a=$sum_dd
            node $denom = Mul f64x1 a=$norm_q b=$norm_d
            node $cos = Div f64x1 a=$dot b=$denom
            output $cos
        ";
        let prog_cos = crate::transpiler::assemble_rls(rls_cosine).map_err(|e| e.to_string())?;
        let vec_q: Vec<f64> = (0..384_u32)
            .map(|i| (f64::from(i % 19) - 9.0) * 0.15)
            .collect();
        let vec_d: Vec<f64> = (0..384_u32)
            .map(|i| (f64::from((i * 3) % 23) - 10.0) * 0.12)
            .collect();

        let expected_dot: f64 = vec_q.iter().zip(vec_d.iter()).map(|(a, b)| a * b).sum();
        let expected_nq: f64 = vec_q.iter().map(|a| a * a).sum::<f64>().sqrt();
        let expected_nd: f64 = vec_d.iter().map(|b| b * b).sum::<f64>().sqrt();
        let expected_cos = expected_dot / (expected_nq * expected_nd);

        let cos0 =
            execute_tier0_interpreter(&prog_cos, &[&vec_q, &vec_d]).map_err(|e| e.to_string())?;
        let cos1 =
            execute_tier1_stencil(&prog_cos, &[&vec_q, &vec_d]).map_err(|e| e.to_string())?;
        let cos2 = execute_tier2_fused(&prog_cos, &[&vec_q, &vec_d]).map_err(|e| e.to_string())?;
        assert_eq!(cos0.output, cos1.output);
        assert_eq!(cos2.telemetry.intermediate_bytes_allocated, 0);
        let RuntimeValue::Scalar(v_cos0) = cos0.output else {
            return Err("expected Scalar from Cosine Similarity Tier 0".to_owned());
        };
        let RuntimeValue::Scalar(v_cos2) = cos2.output else {
            return Err("expected Scalar from Cosine Similarity Tier 2".to_owned());
        };
        assert!((v_cos0 - expected_cos).abs() < 1e-12);
        assert!((v_cos2 - expected_cos).abs() < 1e-12);

        let exe_cos = std::env::temp_dir().join("rul_test_cosine_aot_bin");
        let data_cos = std::env::temp_dir().join("rul_test_cosine_aot_data.bin");
        let cos_rep =
            crate::aot::compile_rl_to_executable(&prog_cos, &exe_cos).map_err(|e| e.to_string())?;
        assert!(cos_rep.avx2_vectorized);
        crate::aot::write_arena_data_file(&data_cos, &[&vec_q, &vec_d])
            .map_err(|e| e.to_string())?;
        let v_cos_aot =
            crate::aot::run_compiled_executable(&exe_cos, &data_cos).map_err(|e| e.to_string())?;
        assert!((v_cos_aot - expected_cos).abs() < 1e-12);
        let _ = std::fs::remove_file(&exe_cos);
        let _ = std::fs::remove_file(&data_cos);

        // Part 3: Interleaved Reduce + Post-Reduce Scalar order (`Sqrt($sum_qq)` placed before `$sum_dd`)
        let rls_cosine_interleaved = "\
            arena 0 len=384 lane=f64x8
            arena 1 len=384 lane=f64x8
            node $q = ArenaInput f64x8 a=0
            node $d = ArenaInput f64x8 a=1
            node $qq = Mul f64x8 a=$q b=$q
            node $sum_qq = ReduceSum f64x1 a=$qq
            node $norm_q = Sqrt f64x1 a=$sum_qq
            node $dd = Mul f64x8 a=$d b=$d
            node $sum_dd = ReduceSum f64x1 a=$dd
            node $norm_d = Sqrt f64x1 a=$sum_dd
            node $qd = Mul f64x8 a=$q b=$d
            node $dot = ReduceSum f64x1 a=$qd
            node $denom = Mul f64x1 a=$norm_q b=$norm_d
            node $cos = Div f64x1 a=$dot b=$denom
            output $cos
        ";
        let prog_interleaved =
            crate::transpiler::assemble_rls(rls_cosine_interleaved).map_err(|e| e.to_string())?;
        let ic2 =
            execute_tier2_fused(&prog_interleaved, &[&vec_q, &vec_d]).map_err(|e| e.to_string())?;
        assert_eq!(ic2.telemetry.intermediate_bytes_allocated, 0);
        let RuntimeValue::Scalar(v_ic2) = ic2.output else {
            return Err("expected Scalar from interleaved Cosine Tier 2".to_owned());
        };
        assert!((v_ic2 - expected_cos).abs() < 1e-12);

        let exe_ic = std::env::temp_dir().join("rul_test_cosine_interleaved_aot_bin");
        let data_ic = std::env::temp_dir().join("rul_test_cosine_interleaved_aot_data.bin");
        crate::aot::compile_rl_to_executable(&prog_interleaved, &exe_ic)
            .map_err(|e| e.to_string())?;
        crate::aot::write_arena_data_file(&data_ic, &[&vec_q, &vec_d])
            .map_err(|e| e.to_string())?;
        let v_ic_aot =
            crate::aot::run_compiled_executable(&exe_ic, &data_ic).map_err(|e| e.to_string())?;
        assert!((v_ic_aot - expected_cos).abs() < 1e-12);
        let _ = std::fs::remove_file(&exe_ic);
        let _ = std::fs::remove_file(&data_ic);
        Ok(())
    }

    #[test]
    fn test_p0_bug1_aot_gather_on_arena_slice_multi_arena_no_oob() -> Result<(), String> {
        let rls = "\
            arena 0 len=16 lane=f64x8
            arena 1 len=4 lane=f64x4
            arena 2 len=4 lane=f64x4
            arena 3 len=16 lane=f64x8
            proof 1 rule=slice_bounds target=0 lo=8 hi=16 witness=0
            proof 2 rule=gather_bounds target=0 lo=0 hi=8 witness=1
            node $a0 = ArenaInput f64x8 a=0
            node $idx = ArenaInput f64x4 a=1
            node $w = ArenaInput f64x4 a=2
            node $slice = ArenaSlice f64x8 proof=1 a=$a0
            node $gat = Gather f64x4 proof=2 a=$slice b=$idx
            node $comb = Add f64x4 a=$gat b=$w
            node $sum = ReduceSum f64x1 a=$comb
            output $sum
        ";
        let prog = crate::transpiler::assemble_rls(rls).map_err(|e| e.to_string())?;
        let a0: Vec<f64> = vec![
            10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 70.0, 80.0, // 0..8
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 131.0, // 8..16
        ];
        let a1: Vec<f64> = vec![0.0, 2.0, 5.0, 7.0];
        let a2: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        // Canary arena 3 directly adjacent to arena 2 in the flat mmap file:
        let a3: Vec<f64> = vec![999_999.0; 16];

        let rep0 =
            execute_tier0_interpreter(&prog, &[&a0, &a1, &a2, &a3]).map_err(|e| e.to_string())?;
        let rep1 =
            execute_tier1_stencil(&prog, &[&a0, &a1, &a2, &a3]).map_err(|e| e.to_string())?;
        let rep2 = execute_tier2_fused(&prog, &[&a0, &a1, &a2, &a3]).map_err(|e| e.to_string())?;
        assert_eq!(rep0.output, RuntimeValue::Scalar(448.0));
        assert_eq!(rep1.output, RuntimeValue::Scalar(448.0));
        assert_eq!(rep2.output, RuntimeValue::Scalar(448.0));

        let pid = std::process::id();
        let exe_path = std::env::temp_dir().join(format!("rul_test_p0_bug1_bin_{pid}"));
        let data_path = std::env::temp_dir().join(format!("rul_test_p0_bug1_data_{pid}.bin"));
        crate::aot::compile_rl_to_executable(&prog, &exe_path).map_err(|e| e.to_string())?;
        crate::aot::write_arena_data_file(&data_path, &[&a0, &a1, &a2, &a3])
            .map_err(|e| e.to_string())?;
        let v_aot = crate::aot::run_compiled_executable(&exe_path, &data_path)
            .map_err(|e| e.to_string())?;
        let _ = std::fs::remove_file(&exe_path);
        let _ = std::fs::remove_file(&data_path);
        assert!(
            (v_aot - 448.0).abs() < 1e-12,
            "expected AOT output 448.0 matching Tier 0 without cross-arena OOB read, got {v_aot}"
        );
        Ok(())
    }

    #[test]
    fn test_p0_verifier_rejects_gather_or_slice_exceeding_sliced_target_len() -> Result<(), String>
    {
        let rls_bad_gather = "\
            arena 0 len=16 lane=f64x8
            arena 1 len=4 lane=f64x4
            proof 1 rule=slice_bounds target=0 lo=8 hi=12 witness=0
            proof 2 rule=gather_bounds target=0 lo=0 hi=8 witness=1
            node $a0 = ArenaInput f64x8 a=0
            node $idx = ArenaInput f64x4 a=1
            node $slice = ArenaSlice f64x4 proof=1 a=$a0
            node $gat = Gather f64x4 proof=2 a=$slice b=$idx
            node $sum = ReduceSum f64x1 a=$gat
            output $sum
        ";
        let prog = crate::transpiler::assemble_rls(rls_bad_gather).map_err(|e| e.to_string())?;
        let ver = verify_program(&prog);
        assert!(
            matches!(
                &ver,
                Err(diag) if diag.kind == VerifyErrorKind::InvalidRefinementProof {
                    proof_idx: 2,
                    bound_hi: 8,
                    arena_len: 4,
                }
            ),
            "expected verifier to reject Gather bound_hi=8 > sliced target len=4, got {ver:?}"
        );
        Ok(())
    }

    #[test]
    fn test_p0_bug2_aot_short_file_exits_cleanly_without_sigbus() -> Result<(), String> {
        let rls = "\
            arena 0 len=4096 lane=f64x8
            node $x = ArenaInput f64x8 a=0
            node $s = ReduceSum f64x1 a=$x
            output $s
        ";
        let prog = crate::transpiler::assemble_rls(rls).map_err(|e| e.to_string())?;
        let pid = std::process::id();
        let exe_path = std::env::temp_dir().join(format!("rul_test_p0_bug2_short_bin_{pid}"));
        let data_path = std::env::temp_dir().join(format!("rul_test_p0_bug2_short_data_{pid}.bin"));
        crate::aot::compile_rl_to_executable(&prog, &exe_path).map_err(|e| e.to_string())?;
        // Write only 2 elements (16 bytes) instead of 4096 elements (32,768 bytes)
        crate::aot::write_arena_data_file(&data_path, &[&[1.0_f64, 2.0_f64]])
            .map_err(|e| e.to_string())?;
        let res = crate::aot::run_compiled_executable(&exe_path, &data_path);
        let _ = std::fs::remove_file(&exe_path);
        let _ = std::fs::remove_file(&data_path);
        match res {
            Err(crate::aot::AotError::ExecutionFailed(msg)) => {
                assert!(
                    msg.contains("Some(66)"),
                    "expected clean exit code Some(66) on short arena file, got: {msg}"
                );
                Ok(())
            }
            other => Err(format!(
                "expected ExecutionFailed(Some(66)) on short file, got: {other:?}"
            )),
        }
    }

    #[test]
    fn test_p0_bug3_aot_gather_fractional_and_neg_half_rejected() -> Result<(), String> {
        let rls = "\
            arena 0 len=8 lane=f64x8
            arena 1 len=4 lane=f64x4
            proof 1 rule=gather_bounds target=0 lo=0 hi=8 witness=1
            node $a0 = ArenaInput f64x8 a=0
            node $idx = ArenaInput f64x4 a=1
            node $gat = Gather f64x4 proof=1 a=$a0 b=$idx
            node $sum = ReduceSum f64x1 a=$gat
            output $sum
        ";
        let prog = crate::transpiler::assemble_rls(rls).map_err(|e| e.to_string())?;
        let a0: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let pid = std::process::id();
        let exe_path = std::env::temp_dir().join(format!("rul_test_p0_bug3_frac_bin_{pid}"));
        let data_path = std::env::temp_dir().join(format!("rul_test_p0_bug3_frac_data_{pid}.bin"));
        crate::aot::compile_rl_to_executable(&prog, &exe_path).map_err(|e| e.to_string())?;

        for bad_indices in [[0.0, 1.5, 2.0, 3.0], [0.0, -0.5, 2.0, 3.0]] {
            let t0 = execute_tier0_interpreter(&prog, &[&a0, &bad_indices]);
            assert!(
                matches!(
                    t0,
                    Err(crate::engine::ExecError::GatherIndexOutOfBounds { .. })
                ),
                "Tier 0 must reject non-integer/negative index {bad_indices:?}, got {t0:?}"
            );
            crate::aot::write_arena_data_file(&data_path, &[&a0, &bad_indices])
                .map_err(|e| e.to_string())?;
            let aot_res = crate::aot::run_compiled_executable(&exe_path, &data_path);
            assert!(
                matches!(aot_res, Err(crate::aot::AotError::GatherOutOfBounds)),
                "AOT must reject non-integer/negative index {bad_indices:?} with GatherOutOfBounds, got {aot_res:?}"
            );
        }
        let _ = std::fs::remove_file(&exe_path);
        let _ = std::fs::remove_file(&data_path);
        Ok(())
    }

    #[test]
    fn test_p0_differential_fuzz_random_verified_graphs_all_tiers_and_aot() -> Result<(), String> {
        let mut seed: u64 = 0x5EED_CAFE_2026_0401;
        let mut next_u64 = || -> u64 {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            seed
        };

        let elem_count = 36usize; // 36 = 8*4 + 4 (exercises both AVX2 8-lane loop and scalar tail)
        let col0: Vec<f64> = (0..elem_count).map(|i| (i as f64) * 0.25 + 1.5).collect();
        let col1: Vec<f64> = (0..elem_count).map(|i| ((i % 7) as f64) + 0.75).collect();
        let col2: Vec<f64> = (0..elem_count)
            .map(|i| ((elem_count - i) as f64) * 0.1 + 0.5)
            .collect();

        let pid = std::process::id();
        let exe_path = std::env::temp_dir().join(format!("rul_test_p0_fuzz_bin_{pid}"));
        let data_path = std::env::temp_dir().join(format!("rul_test_p0_fuzz_data_{pid}.bin"));
        crate::aot::write_arena_data_file(&data_path, &[&col0, &col1, &col2])
            .map_err(|e| e.to_string())?;

        for case_idx in 0..50usize {
            let mut b = SymbolicGraphBuilder::new(CAP_NONE);
            let a0 = b.add_arena(elem_count as u32, TypeLane::F64X8, false);
            let a1 = b.add_arena(elem_count as u32, TypeLane::F64X8, false);
            let a2 = b.add_arena(elem_count as u32, TypeLane::F64X8, false);
            let s = ScheduleHint::UNFUSED;

            b.emit_arena_input("$c0", a0, 1, s)
                .map_err(|e| e.to_string())?;
            b.emit_arena_input("$c1", a1, 1, s)
                .map_err(|e| e.to_string())?;
            b.emit_arena_input("$c2", a2, 1, s)
                .map_err(|e| e.to_string())?;
            b.emit_const_f64("$k0", 1.25, s)
                .map_err(|e| e.to_string())?;
            b.emit_const_f64("$k1", 0.5, s).map_err(|e| e.to_string())?;

            let mut col_vars = vec!["$c0".to_owned(), "$c1".to_owned(), "$c2".to_owned()];
            let const_vars = ["$k0", "$k1"];
            let step_count = 4 + ((next_u64() as usize) % 12);

            for step in 0..step_count {
                let name = format!("$v{step}");
                let r = next_u64() % 12;
                let lhs_idx = (next_u64() as usize) % col_vars.len();
                let lhs = col_vars.get(lhs_idx).cloned().unwrap_or_default();
                match r {
                    0..=4 => {
                        let op = match r {
                            0 => Opcode::Add,
                            1 => Opcode::Sub,
                            2 => Opcode::Mul,
                            3 => Opcode::Min,
                            _ => Opcode::Max,
                        };
                        let use_const = (next_u64() % 3) == 0;
                        let rhs = if use_const {
                            const_vars
                                .get((next_u64() as usize) % const_vars.len())
                                .copied()
                                .unwrap_or("$k0")
                                .to_owned()
                        } else {
                            let rhs_idx = (next_u64() as usize) % col_vars.len();
                            col_vars.get(rhs_idx).cloned().unwrap_or_default()
                        };
                        b.emit_binary_op(&name, op, TypeLane::F64X8, &lhs, &rhs, s)
                            .map_err(|e| e.to_string())?;
                    }
                    5 => {
                        // Safe non-zero denominator via `Abs(rhs) + 1.25`
                        let rhs_idx = (next_u64() as usize) % col_vars.len();
                        let rhs = col_vars.get(rhs_idx).cloned().unwrap_or_default();
                        let abs_rhs = format!("$absd{step}");
                        let den = format!("$den{step}");
                        b.emit_unary_op(&abs_rhs, Opcode::Abs, TypeLane::F64X8, &rhs, s)
                            .map_err(|e| e.to_string())?;
                        b.emit_binary_op(&den, Opcode::Add, TypeLane::F64X8, &abs_rhs, "$k0", s)
                            .map_err(|e| e.to_string())?;
                        b.emit_binary_op(&name, Opcode::Div, TypeLane::F64X8, &lhs, &den, s)
                            .map_err(|e| e.to_string())?;
                    }
                    6 => {
                        b.emit_unary_op(&name, Opcode::Abs, TypeLane::F64X8, &lhs, s)
                            .map_err(|e| e.to_string())?;
                    }
                    7 => {
                        let abs_name = format!("$abs{step}");
                        b.emit_unary_op(&abs_name, Opcode::Abs, TypeLane::F64X8, &lhs, s)
                            .map_err(|e| e.to_string())?;
                        b.emit_unary_op(&name, Opcode::Sqrt, TypeLane::F64X8, &abs_name, s)
                            .map_err(|e| e.to_string())?;
                    }
                    8 => {
                        let op = if (next_u64() & 1) == 0 {
                            Opcode::Floor
                        } else {
                            Opcode::Ceil
                        };
                        b.emit_unary_op(&name, op, TypeLane::F64X8, &lhs, s)
                            .map_err(|e| e.to_string())?;
                    }
                    9 => {
                        // Positive domain `Abs(lhs) + 0.5` for `Ln`
                        let abs_l = format!("$absl{step}");
                        let pos_l = format!("$posl{step}");
                        b.emit_unary_op(&abs_l, Opcode::Abs, TypeLane::F64X8, &lhs, s)
                            .map_err(|e| e.to_string())?;
                        b.emit_binary_op(&pos_l, Opcode::Add, TypeLane::F64X8, &abs_l, "$k1", s)
                            .map_err(|e| e.to_string())?;
                        b.emit_unary_op(&name, Opcode::Ln, TypeLane::F64X8, &pos_l, s)
                            .map_err(|e| e.to_string())?;
                    }
                    10 => {
                        // Bounded domain `Min(Max(lhs, 0.5), 1.25)` for `Exp`
                        let lo_b = format!("$lob{step}");
                        let cl_b = format!("$clb{step}");
                        b.emit_binary_op(&lo_b, Opcode::Max, TypeLane::F64X8, &lhs, "$k1", s)
                            .map_err(|e| e.to_string())?;
                        b.emit_binary_op(&cl_b, Opcode::Min, TypeLane::F64X8, &lo_b, "$k0", s)
                            .map_err(|e| e.to_string())?;
                        b.emit_unary_op(&name, Opcode::Exp, TypeLane::F64X8, &cl_b, s)
                            .map_err(|e| e.to_string())?;
                    }
                    _ => {
                        let m_name = format!("$m{step}");
                        let cmp_op = match next_u64() % 3 {
                            0 => Opcode::CmpGt,
                            1 => Opcode::CmpLt,
                            _ => Opcode::CmpEq,
                        };
                        b.emit_binary_op(&m_name, cmp_op, TypeLane::MASKX8, &lhs, "$k0", s)
                            .map_err(|e| e.to_string())?;
                        b.emit_binary_op(
                            &name,
                            Opcode::SelectZero,
                            TypeLane::F64X8,
                            &m_name,
                            &lhs,
                            s,
                        )
                        .map_err(|e| e.to_string())?;
                    }
                }
                col_vars.push(name);
            }

            let last_col = col_vars.last().cloned().unwrap_or_default();
            let red_op = match case_idx % 3 {
                0 => Opcode::ReduceSum,
                1 => Opcode::ReduceMax,
                _ => Opcode::ReduceMin,
            };
            b.emit_unary_op("$out", red_op, TypeLane::F64_SCALAR, &last_col, s)
                .map_err(|e| e.to_string())?;
            let prog = b.finish("$out").map_err(|e| e.to_string())?;

            let rep0 = execute_tier0_interpreter(&prog, &[&col0, &col1, &col2])
                .map_err(|e| e.to_string())?;
            let rep1 =
                execute_tier1_stencil(&prog, &[&col0, &col1, &col2]).map_err(|e| e.to_string())?;
            let rep2 =
                execute_tier2_fused(&prog, &[&col0, &col1, &col2]).map_err(|e| e.to_string())?;
            let RuntimeValue::Scalar(v0) = rep0.output else {
                return Err("expected scalar output".to_owned());
            };
            let RuntimeValue::Scalar(v1) = rep1.output else {
                return Err("expected scalar output".to_owned());
            };
            let RuntimeValue::Scalar(v2) = rep2.output else {
                return Err("expected scalar output".to_owned());
            };

            crate::aot::compile_rl_to_executable(&prog, &exe_path).map_err(|e| e.to_string())?;
            let vaot = crate::aot::run_compiled_executable(&exe_path, &data_path)
                .map_err(|e| e.to_string())?;

            let tol = v0.abs().max(1.0) * 1e-11;
            assert!(
                (v0 - v1).abs() <= tol && (v0 - v2).abs() <= tol && (v0 - vaot).abs() <= tol,
                "fuzz case {case_idx} mismatch: Tier0={v0}, Tier1={v1}, Tier2={v2}, AOT={vaot}"
            );
        }

        let _ = std::fs::remove_file(&exe_path);
        let _ = std::fs::remove_file(&data_path);
        Ok(())
    }

    #[test]
    fn test_p1_horner50_register_allocated_aot_equivalence() -> Result<(), String> {
        let elem_count = 1024usize;
        let col_x: Vec<f64> = (0..elem_count)
            .map(|i| 0.5 + ((i % 16) as f64) * 0.025)
            .collect();

        let mut b = SymbolicGraphBuilder::new(CAP_NONE);
        let a0 = b.add_arena(elem_count as u32, TypeLane::F64X8, false);
        let s = ScheduleHint {
            fusion_group: 1,
            policy: ExecPolicy::SimdStream,
        };
        b.emit_arena_input("$x", a0, 1, s)
            .map_err(|e| e.to_string())?;
        b.emit_const_f64("$c0", 1.000_001, s)
            .map_err(|e| e.to_string())?;
        b.emit_const_f64("$c1", 0.999_999, s)
            .map_err(|e| e.to_string())?;

        let mut acc_var = "$c0".to_owned();
        for step in 0..50usize {
            let mul_name = format!("$m{step}");
            let add_name = format!("$a{step}");
            let c_ref = if (step & 1) == 0 { "$c1" } else { "$c0" };
            b.emit_binary_op(&mul_name, Opcode::Mul, TypeLane::F64X8, &acc_var, "$x", s)
                .map_err(|e| e.to_string())?;
            b.emit_binary_op(&add_name, Opcode::Add, TypeLane::F64X8, &mul_name, c_ref, s)
                .map_err(|e| e.to_string())?;
            acc_var = add_name;
        }
        b.emit_unary_op("$sum", Opcode::ReduceSum, TypeLane::F64_SCALAR, &acc_var, s)
            .map_err(|e| e.to_string())?;
        let prog = b.finish("$sum").map_err(|e| e.to_string())?;
        assert_eq!(prog.nodes.len(), 104);

        let rep0 = execute_tier0_interpreter(&prog, &[&col_x]).map_err(|e| e.to_string())?;
        let rep2 = execute_tier2_fused(&prog, &[&col_x]).map_err(|e| e.to_string())?;
        let RuntimeValue::Scalar(v0) = rep0.output else {
            return Err("expected scalar output".to_owned());
        };
        let RuntimeValue::Scalar(v2) = rep2.output else {
            return Err("expected scalar output".to_owned());
        };

        let pid = std::process::id();
        let exe_path = std::env::temp_dir().join(format!("rul_test_p1_horner50_bin_{pid}"));
        let data_path = std::env::temp_dir().join(format!("rul_test_p1_horner50_data_{pid}.bin"));
        let report =
            crate::aot::compile_rl_to_executable(&prog, &exe_path).map_err(|e| e.to_string())?;
        assert!(report.avx2_vectorized);
        crate::aot::write_arena_data_file(&data_path, &[&col_x]).map_err(|e| e.to_string())?;
        let vaot = crate::aot::run_compiled_executable(&exe_path, &data_path)
            .map_err(|e| e.to_string())?;
        let _ = std::fs::remove_file(&exe_path);
        let _ = std::fs::remove_file(&data_path);

        let tol = v0.abs().max(1.0) * 1e-11;
        assert!(
            (v0 - v2).abs() <= tol && (v0 - vaot).abs() <= tol,
            "Horner-50 mismatch: Tier0={v0}, Tier2={v2}, AOT={vaot}"
        );
        Ok(())
    }

    #[test]
    fn test_p1_register_allocator_spill_and_edge_cases_torture_suite() -> Result<(), String> {
        let pid = std::process::id();
        let exe_path = std::env::temp_dir().join(format!("rul_test_p1_torture_bin_{pid}"));
        let data_path = std::env::temp_dir().join(format!("rul_test_p1_torture_data_{pid}.bin"));
        let s = ScheduleHint::UNFUSED;

        let check_all_tiers =
            |label: &str, prog: &RulProgram, cols: &[&[f64]]| -> Result<(), String> {
                let rep0 = execute_tier0_interpreter(prog, cols).map_err(|e| e.to_string())?;
                let rep1 = execute_tier1_stencil(prog, cols).map_err(|e| e.to_string())?;
                let rep2 = execute_tier2_fused(prog, cols).map_err(|e| e.to_string())?;
                let RuntimeValue::Scalar(v0) = rep0.output else {
                    return Err(format!("{label}: expected scalar in Tier 0"));
                };
                let RuntimeValue::Scalar(v1) = rep1.output else {
                    return Err(format!("{label}: expected scalar in Tier 1"));
                };
                let RuntimeValue::Scalar(v2) = rep2.output else {
                    return Err(format!("{label}: expected scalar in Tier 2"));
                };
                crate::aot::compile_rl_to_executable(prog, &exe_path).map_err(|e| e.to_string())?;
                crate::aot::write_arena_data_file(&data_path, cols).map_err(|e| e.to_string())?;
                let vaot = crate::aot::run_compiled_executable(&exe_path, &data_path)
                    .map_err(|e| e.to_string())?;
                let tol = v0.abs().max(1.0) * 1e-11;
                if (v0 - v1).abs() > tol || (v0 - v2).abs() > tol || (v0 - vaot).abs() > tol {
                    return Err(format!(
                        "{label} mismatch: Tier0={v0}, Tier1={v1}, Tier2={v2}, AOT={vaot}"
                    ));
                }
                Ok(())
            };

        // 1. Spill torture (> 12 live columns): 16 live `Sub` columns consumed in reverse order via `Div`/`Add`
        {
            let elem_count = 20usize; // 2*8 + 4 scalar tail
            let col_x: Vec<f64> = (0..elem_count).map(|i| 10.0 + (i as f64) * 0.5).collect();
            let mut b = SymbolicGraphBuilder::new(CAP_NONE);
            let a0 = b.add_arena(elem_count as u32, TypeLane::F64X8, false);
            b.emit_arena_input("$x", a0, 1, s)
                .map_err(|e| e.to_string())?;
            let mut live_cols = Vec::with_capacity(16);
            for k in 0..16usize {
                let cname = format!("$c{k}");
                let sname = format!("$sub{k}");
                b.emit_const_f64(&cname, (k as f64) * 0.25 + 1.0, s)
                    .map_err(|e| e.to_string())?;
                b.emit_binary_op(&sname, Opcode::Sub, TypeLane::F64X8, "$x", &cname, s)
                    .map_err(|e| e.to_string())?;
                live_cols.push(sname);
            }
            let mut acc = live_cols.get(15).cloned().unwrap_or_default();
            for k in (0..15usize).rev() {
                let rhs = live_cols.get(k).cloned().unwrap_or_default();
                let dname = format!("$div{k}");
                let aname = format!("$acc{k}");
                b.emit_binary_op(&dname, Opcode::Div, TypeLane::F64X8, &acc, &rhs, s)
                    .map_err(|e| e.to_string())?;
                b.emit_binary_op(&aname, Opcode::Add, TypeLane::F64X8, &dname, &rhs, s)
                    .map_err(|e| e.to_string())?;
                acc = aname;
            }
            b.emit_unary_op("$out", Opcode::ReduceSum, TypeLane::F64_SCALAR, &acc, s)
                .map_err(|e| e.to_string())?;
            let prog = b.finish("$out").map_err(|e| e.to_string())?;
            check_all_tiers("case1_spill_16_live_reverse_div", &prog, &[&col_x])?;
        }

        // 2. Unary (`Ln`, `Exp`, `Abs`, `Sqrt`, `Floor`, `Ceil`) with register operand reused later
        {
            let elem_count = 20usize;
            let col_x: Vec<f64> = (0..elem_count).map(|i| -1.75 + (i as f64) * 0.2).collect();
            let mut b = SymbolicGraphBuilder::new(CAP_NONE);
            let a0 = b.add_arena(elem_count as u32, TypeLane::F64X8, false);
            b.emit_arena_input("$x", a0, 1, s)
                .map_err(|e| e.to_string())?;
            b.emit_const_f64("$two", 2.0, s)
                .map_err(|e| e.to_string())?;
            b.emit_unary_op("$abs", Opcode::Abs, TypeLane::F64X8, "$x", s)
                .map_err(|e| e.to_string())?;
            b.emit_binary_op("$pos", Opcode::Add, TypeLane::F64X8, "$abs", "$two", s)
                .map_err(|e| e.to_string())?;
            // `$pos` stays in a YMM register and is used by `Ln`, `Exp`, `Sqrt`, `Floor`, `Ceil`, AND a trailing `Add`!
            b.emit_unary_op("$ln", Opcode::Ln, TypeLane::F64X8, "$pos", s)
                .map_err(|e| e.to_string())?;
            b.emit_unary_op("$exp", Opcode::Exp, TypeLane::F64X8, "$ln", s)
                .map_err(|e| e.to_string())?;
            b.emit_unary_op("$sq", Opcode::Sqrt, TypeLane::F64X8, "$pos", s)
                .map_err(|e| e.to_string())?;
            b.emit_unary_op("$fl", Opcode::Floor, TypeLane::F64X8, "$pos", s)
                .map_err(|e| e.to_string())?;
            b.emit_unary_op("$cl", Opcode::Ceil, TypeLane::F64X8, "$x", s)
                .map_err(|e| e.to_string())?;
            b.emit_binary_op("$s1", Opcode::Add, TypeLane::F64X8, "$exp", "$sq", s)
                .map_err(|e| e.to_string())?;
            b.emit_binary_op("$s2", Opcode::Add, TypeLane::F64X8, "$fl", "$cl", s)
                .map_err(|e| e.to_string())?;
            b.emit_binary_op("$s3", Opcode::Add, TypeLane::F64X8, "$s1", "$s2", s)
                .map_err(|e| e.to_string())?;
            b.emit_binary_op("$s4", Opcode::Add, TypeLane::F64X8, "$s3", "$pos", s)
                .map_err(|e| e.to_string())?;
            b.emit_unary_op("$out", Opcode::ReduceSum, TypeLane::F64_SCALAR, "$s4", s)
                .map_err(|e| e.to_string())?;
            let prog = b.finish("$out").map_err(|e| e.to_string())?;
            check_all_tiers("case2_unary_x87_and_abs_with_reuse", &prog, &[&col_x])?;
        }

        // 3. Dual `ArenaSlice` with distinct `bound_lo` offsets (`lo=2..14` and `lo=4..16`, both len 12)
        {
            let col_a: Vec<f64> = (0..16).map(|i| (i as f64) * 1.5 + 1.0).collect();
            let col_b: Vec<f64> = (0..16).map(|i| 100.0 - (i as f64) * 2.0).collect();
            let rls = "\
                arena 0 len=16 lane=f64x8
                arena 1 len=16 lane=f64x8
                proof 1 rule=slice_bounds target=0 lo=2 hi=14 witness=0
                proof 2 rule=slice_bounds target=1 lo=4 hi=16 witness=0
                node $a = ArenaInput f64x8 a=0
                node $b = ArenaInput f64x8 a=1
                node $sa = ArenaSlice f64x8 proof=1 a=$a
                node $sb = ArenaSlice f64x8 proof=2 a=$b
                node $m = Mul f64x8 a=$sa b=$sb
                node $out = ReduceSum f64x1 a=$m
                output $out
            ";
            let prog = crate::transpiler::assemble_rls(rls).map_err(|e| e.to_string())?;
            check_all_tiers(
                "case3_dual_arena_slice_distinct_offsets",
                &prog,
                &[&col_a, &col_b],
            )?;
        }

        // 4. Nested `ArenaSlice(ArenaSlice)`: base len 16 -> slice1 lo=2..14 (len 12) -> slice2 lo=3..11 (len 8)
        {
            let col_a: Vec<f64> = (0..16).map(|i| ((i + 1) as f64) * 3.25).collect();
            let rls = "\
                arena 0 len=16 lane=f64x8
                proof 1 rule=slice_bounds target=0 lo=2 hi=14 witness=0
                proof 2 rule=slice_bounds target=0 lo=3 hi=11 witness=0
                node $a = ArenaInput f64x8 a=0
                node $s1 = ArenaSlice f64x8 proof=1 a=$a
                node $s2 = ArenaSlice f64x8 proof=2 a=$s1
                node $out = ReduceSum f64x1 a=$s2
                output $out
            ";
            let prog = crate::transpiler::assemble_rls(rls).map_err(|e| e.to_string())?;
            check_all_tiers("case4_nested_arena_slice", &prog, &[&col_a])?;
        }

        // 5. `Gather(ArenaSlice)` valid + runtime OOB when index >= slice_len (even if < base_arena_len)
        {
            let col_a: Vec<f64> = (0..16).map(|i| (i as f64) * 10.0).collect();
            let idx_ok: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 2.0];
            let idx_oob: Vec<f64> = vec![0.0, 1.0, 8.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 2.0]; // 8.0 >= slice len 8!
            let rls = "\
                arena 0 len=16 lane=f64x8
                arena 1 len=10 lane=f64x8
                proof 1 rule=slice_bounds target=0 lo=4 hi=12 witness=0
                proof 2 rule=gather_bounds target=0 lo=0 hi=8 witness=1
                node $a = ArenaInput f64x8 a=0
                node $idx = ArenaInput f64x8 a=1
                node $slice = ArenaSlice f64x8 proof=1 a=$a
                node $gat = Gather f64x8 proof=2 a=$slice b=$idx
                node $out = ReduceSum f64x1 a=$gat
                output $out
            ";
            let prog = crate::transpiler::assemble_rls(rls).map_err(|e| e.to_string())?;
            check_all_tiers(
                "case5_gather_on_arena_slice_valid",
                &prog,
                &[&col_a, &idx_ok],
            )?;

            assert!(matches!(
                execute_tier0_interpreter(&prog, &[&col_a, &idx_oob]),
                Err(crate::engine::ExecError::GatherIndexOutOfBounds { .. })
            ));
            assert!(matches!(
                execute_tier2_fused(&prog, &[&col_a, &idx_oob]),
                Err(crate::engine::ExecError::GatherIndexOutOfBounds { .. })
            ));
            crate::aot::write_arena_data_file(&data_path, &[&col_a, &idx_oob])
                .map_err(|e| e.to_string())?;
            assert!(matches!(
                crate::aot::run_compiled_executable(&exe_path, &data_path),
                Err(crate::aot::AotError::GatherOutOfBounds)
            ));
        }

        // 6. Multi-reduction (`ReduceSum` + `ReduceMax` + `ReduceMin`) + post-reduction scalar DAG
        {
            let col_a: Vec<f64> = (0..20).map(|i| (i as f64) * 1.25 + 2.0).collect();
            let rls = "\
                arena 0 len=20 lane=f64x8
                node $a = ArenaInput f64x8 a=0
                node $sum = ReduceSum f64x1 a=$a
                node $max = ReduceMax f64x1 a=$a
                node $min = ReduceMin f64x1 a=$a
                node $diff = Sub f64x1 a=$max b=$min
                node $num = Add f64x1 a=$sum b=$diff
                node $out = Div f64x1 a=$num b=$min
                output $out
            ";
            let prog = crate::transpiler::assemble_rls(rls).map_err(|e| e.to_string())?;
            check_all_tiers("case6_multi_reduction_post_scalar", &prog, &[&col_a])?;
        }

        // 7. `a == b` at `last_use` (`Mul($v, $v)`) + `CmpGt`/`CmpLt`/`CmpEq`/`MaskAnd`/`MaskOr`/`SelectZero`
        {
            let col_a: Vec<f64> = (0..20).map(|i| (i as f64) - 8.0).collect();
            let rls = "\
                arena 0 len=20 lane=f64x8
                node $a = ArenaInput f64x8 a=0
                node $c1 = ConstF64 f64x1 const=2.0
                node $c2 = ConstF64 f64x1 const=25.0
                node $v = Add f64x8 a=$a b=$c1
                node $sq = Mul f64x8 a=$v b=$v
                node $mgt = CmpGt maskx8 a=$sq b=$c1
                node $mlt = CmpLt maskx8 a=$sq b=$c2
                node $meq = CmpEq maskx8 a=$sq b=$c2
                node $mand = MaskAnd maskx8 a=$mgt b=$mlt
                node $mor = MaskOr maskx8 a=$mand b=$meq
                node $sel = SelectZero f64x8 a=$mor b=$sq
                node $out = ReduceSum f64x1 a=$sel
                output $out
            ";
            let prog = crate::transpiler::assemble_rls(rls).map_err(|e| e.to_string())?;
            check_all_tiers("case7_self_mul_last_use_and_masks", &prog, &[&col_a])?;
        }

        // 8. `Gather` float edge cases: `-0.0` (accepted as index 0), `1e20` (rejected), `NaN` (rejected)
        {
            let col_a: Vec<f64> = vec![42.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
            let idx_neg_zero: Vec<f64> = vec![-0.0, 1.0, -0.0, 2.0];
            let rls = "\
                arena 0 len=8 lane=f64x8
                arena 1 len=4 lane=f64x4
                proof 1 rule=gather_bounds target=0 lo=0 hi=8 witness=1
                node $a = ArenaInput f64x8 a=0
                node $idx = ArenaInput f64x4 a=1
                node $gat = Gather f64x4 proof=1 a=$a b=$idx
                node $out = ReduceSum f64x1 a=$gat
                output $out
            ";
            let prog = crate::transpiler::assemble_rls(rls).map_err(|e| e.to_string())?;
            check_all_tiers(
                "case8_gather_neg_zero_valid",
                &prog,
                &[&col_a, &idx_neg_zero],
            )?;

            for bad_idx in [[0.0, 1e20, 1.0, 2.0], [0.0, f64::NAN, 1.0, 2.0]] {
                assert!(matches!(
                    execute_tier0_interpreter(&prog, &[&col_a, &bad_idx]),
                    Err(crate::engine::ExecError::GatherIndexOutOfBounds { .. })
                ));
                crate::aot::write_arena_data_file(&data_path, &[&col_a, &bad_idx])
                    .map_err(|e| e.to_string())?;
                assert!(matches!(
                    crate::aot::run_compiled_executable(&exe_path, &data_path),
                    Err(crate::aot::AotError::GatherOutOfBounds)
                ));
            }
        }

        let _ = std::fs::remove_file(&exe_path);
        let _ = std::fs::remove_file(&data_path);
        Ok(())
    }

    #[test]
    fn test_p1b_explicit_fma_opcode_all_tiers_and_aot_horner50() -> Result<(), String> {
        let pid = std::process::id();
        let exe_path = std::env::temp_dir().join(format!("rul_test_p1b_fma_{pid}"));
        let data_path = std::env::temp_dir().join(format!("rul_test_p1b_fma_{pid}.bin"));

        // 1. Verify `.rls` <-> `.rl` round-trip and single-rounded IEEE-754 FMA precision
        //    on `(1 + 2^-27) * (1 + 2^-27) - 1`, where un-fused `Mul + Add` loses the `2^-54` term!
        let eps = f64::from_bits(0x3E40_0000_0000_0000); // 2^-27
        let a_val = 1.0_f64 + eps;
        let col_x: Vec<f64> = vec![a_val; 11]; // 2 AVX2 vectors (8) + 3 scalar tail
        let rls_single_round = "\
            arena 0 len=11 lane=f64x8
            node $x = ArenaInput f64x8 a=0
            node $neg1 = ConstF64 f64x1 const=-1.0
            node $fma = Fma f64x8 a=$x b=$x c=$neg1
            node $sum = ReduceSum f64x1 a=$fma
            node $scale = ConstF64 f64x1 const=2.0
            node $bias = ConstF64 f64x1 const=0.25
            node $out = Fma f64x1 a=$sum b=$scale c=$bias
            output $out
        ";
        let prog_sr =
            crate::transpiler::assemble_rls(rls_single_round).map_err(|e| e.to_string())?;
        let rls_decompiled = crate::transpiler::program_to_rls(&prog_sr);
        assert!(rls_decompiled.contains("Fma"));
        let prog_re =
            crate::transpiler::assemble_rls(&rls_decompiled).map_err(|e| e.to_string())?;
        assert_eq!(prog_sr, prog_re);
        let disasm = crate::transpiler::disassemble_rl(&prog_sr);
        assert!(disasm.contains("Fma"));

        let expected_elem = a_val.mul_add(a_val, -1.0);
        let unfused_elem = (a_val * a_val) - 1.0;
        assert_ne!(
            expected_elem.to_bits(),
            unfused_elem.to_bits(),
            "test input must distinguish single-rounded FMA from two-rounding Mul+Add"
        );
        let expected_out = (expected_elem * 11.0).mul_add(2.0, 0.25);

        let RuntimeValue::Scalar(s0) = execute_tier0_interpreter(&prog_sr, &[&col_x])
            .map_err(|e| e.to_string())?
            .output
        else {
            return Err("Tier 0 did not return Scalar".to_owned());
        };
        let RuntimeValue::Scalar(s1) = execute_tier1_stencil(&prog_sr, &[&col_x])
            .map_err(|e| e.to_string())?
            .output
        else {
            return Err("Tier 1 did not return Scalar".to_owned());
        };
        let RuntimeValue::Scalar(s2) = execute_tier2_fused(&prog_sr, &[&col_x])
            .map_err(|e| e.to_string())?
            .output
        else {
            return Err("Tier 2 did not return Scalar".to_owned());
        };
        let _rep =
            crate::aot::compile_rl_to_executable(&prog_sr, &exe_path).map_err(|e| e.to_string())?;
        crate::aot::write_arena_data_file(&data_path, &[&col_x]).map_err(|e| e.to_string())?;
        let s_aot = crate::aot::run_compiled_executable(&exe_path, &data_path)
            .map_err(|e| e.to_string())?;

        assert_eq!(s0.to_bits(), expected_out.to_bits());
        assert_eq!(s1.to_bits(), expected_out.to_bits());
        assert_eq!(s2.to_bits(), expected_out.to_bits());
        assert_eq!(s_aot.to_bits(), expected_out.to_bits());

        // 2. Verify all register allocation operand-placement branches in `Opcode::Fma`:
        //    - `dst_reg == reg_a` (`acc` in `a`)
        //    - `dst_reg == reg_b` (`acc` in `b`)
        //    - `Some(dst_reg) == opt_rc` (`acc` in `c`, emitting `vfmadd231pd`)
        //    - `else` (`a`, `b`, `c` all live after `Fma`, emitting `vmovapd` + `vfmadd213pd`)
        let rls_branches = "\
            arena 0 len=11 lane=f64x8
            node $x = ArenaInput f64x8 a=0
            node $c1 = ConstF64 f64x1 const=1.25
            node $c2 = ConstF64 f64x1 const=-0.75
            node $u = Add f64x8 a=$x b=$c1
            node $v = Sub f64x8 a=$x b=$c2
            node $w = Mul f64x8 a=$x b=$c1
            node $f_else = Fma f64x8 a=$u b=$v c=$w
            node $keep_uvw = Add f64x8 a=$u b=$v
            node $keep_all = Add f64x8 a=$keep_uvw b=$w
            node $f_in_a = Fma f64x8 a=$f_else b=$x c=$c1
            node $f_in_b = Fma f64x8 a=$x b=$f_in_a c=$c2
            node $f_in_c = Fma f64x8 a=$x b=$keep_all c=$f_in_b
            node $out = ReduceSum f64x1 a=$f_in_c
            output $out
        ";
        let prog_br = crate::transpiler::assemble_rls(rls_branches).map_err(|e| e.to_string())?;
        let RuntimeValue::Scalar(br_t0) = execute_tier0_interpreter(&prog_br, &[&col_x])
            .map_err(|e| e.to_string())?
            .output
        else {
            return Err("Tier 0 did not return Scalar".to_owned());
        };
        let RuntimeValue::Scalar(br_t2) = execute_tier2_fused(&prog_br, &[&col_x])
            .map_err(|e| e.to_string())?
            .output
        else {
            return Err("Tier 2 did not return Scalar".to_owned());
        };
        crate::aot::compile_rl_to_executable(&prog_br, &exe_path).map_err(|e| e.to_string())?;
        let br_aot = crate::aot::run_compiled_executable(&exe_path, &data_path)
            .map_err(|e| e.to_string())?;
        assert!((br_t2 - br_t0).abs() <= 1e-12 * br_t0.abs().max(1.0));
        assert!((br_aot - br_t0).abs() <= 1e-12 * br_t0.abs().max(1.0));

        // 3. Verify 50-Step Horner Polynomial using explicit `Opcode::Fma` (`54` nodes vs `104` nodes)
        let elem_count: u32 = 1_027;
        let sched = ScheduleHint {
            fusion_group: 1,
            policy: ExecPolicy::SimdStream,
        };
        let mut b_fma = SymbolicGraphBuilder::new(CAP_NONE);
        let a0 = b_fma.add_arena(elem_count, TypeLane::F64X8, false);
        b_fma
            .emit_arena_input("$x", a0, 1, sched)
            .map_err(|e| e.to_string())?;
        b_fma
            .emit_const_f64("$c0", 1.000_001, sched)
            .map_err(|e| e.to_string())?;
        b_fma
            .emit_const_f64("$c1", 0.999_999, sched)
            .map_err(|e| e.to_string())?;
        let mut acc = "$c0".to_owned();
        for step in 0..50usize {
            let f = format!("$f{step}");
            let c = if (step & 1) == 0 { "$c1" } else { "$c0" };
            b_fma
                .emit_fma_op(&f, TypeLane::F64X8, &acc, "$x", c, sched)
                .map_err(|e| e.to_string())?;
            acc = f;
        }
        b_fma
            .emit_unary_op("$sum", Opcode::ReduceSum, TypeLane::F64_SCALAR, &acc, sched)
            .map_err(|e| e.to_string())?;
        let prog_h50_fma = b_fma.finish("$sum").map_err(|e| e.to_string())?;
        assert_eq!(prog_h50_fma.nodes.len(), 54);

        let xs: Vec<f64> = (0..elem_count)
            .map(|i| 0.5 + (f64::from(i % 16) * 0.025))
            .collect();
        let RuntimeValue::Scalar(h_t0) = execute_tier0_interpreter(&prog_h50_fma, &[&xs])
            .map_err(|e| e.to_string())?
            .output
        else {
            return Err("Tier 0 did not return Scalar".to_owned());
        };
        let RuntimeValue::Scalar(h_t2) = execute_tier2_fused(&prog_h50_fma, &[&xs])
            .map_err(|e| e.to_string())?
            .output
        else {
            return Err("Tier 2 did not return Scalar".to_owned());
        };
        let rep_h50_fma = crate::aot::compile_rl_to_executable(&prog_h50_fma, &exe_path)
            .map_err(|e| e.to_string())?;
        crate::aot::write_arena_data_file(&data_path, &[&xs]).map_err(|e| e.to_string())?;
        let h_aot = crate::aot::run_compiled_executable(&exe_path, &data_path)
            .map_err(|e| e.to_string())?;

        assert!(rep_h50_fma.avx2_vectorized);
        // 50 single-instruction `vfmadd213pd` steps must produce a much smaller ELF64 (< 3,000B)
        // than the 104-node `Mul+Add` Horner-50 (`3,713B`).
        assert!(
            rep_h50_fma.machine_code_bytes < 3000,
            "expected compact FMA machine code < 3000B, got {}B",
            rep_h50_fma.machine_code_bytes
        );
        assert!((h_t2 - h_t0).abs() / h_t0.abs().max(1.0) < 1e-12);
        assert!((h_aot - h_t0).abs() / h_t0.abs().max(1.0) < 1e-12);

        // 4. Verify `Opcode::Fma` 16-bit operand `b`/`c` index boundary (`u16::MAX = 65,535` succeeds; `> 65,535` errors)
        let mut b_large = SymbolicGraphBuilder::new(CAP_NONE);
        let a_large = b_large.add_arena(4, TypeLane::F64X8, false);
        b_large
            .emit_arena_input("$x", a_large, 1, sched)
            .map_err(|e| e.to_string())?;
        for idx in 1..=65_536usize {
            b_large
                .emit_const_f64(&format!("$c{idx}"), 1.0, sched)
                .map_err(|e| e.to_string())?;
        }
        // Node `#65535` (`$c65535`) fits in `u16::MAX` and must succeed for both `b` and `c`:
        b_large
            .emit_fma_op(
                "$ok_65535",
                TypeLane::F64X8,
                "$x",
                "$c65535",
                "$c65535",
                sched,
            )
            .map_err(|e| e.to_string())?;
        // Node `#65536` (`$c65536`) exceeds `u16::MAX` and must be rejected for `b` and `c`:
        let err_b = b_large.emit_fma_op("$bad_b", TypeLane::F64X8, "$x", "$c65536", "$c1", sched);
        assert_eq!(
            err_b,
            Err(crate::transpiler::TranspileError::FmaOperandIndexOverflow {
                symbol: "$c65536".to_owned(),
                node_idx: 65_536,
            })
        );
        let err_c = b_large.emit_fma_op("$bad_c", TypeLane::F64X8, "$x", "$c1", "$c65536", sched);
        assert_eq!(
            err_c,
            Err(crate::transpiler::TranspileError::FmaOperandIndexOverflow {
                symbol: "$c65536".to_owned(),
                node_idx: 65_536,
            })
        );
        // Numeric index `> 65,535` in `.rls` text assembly must also return `FmaOperandIndexOverflow`:
        let rls_overflow = "\
            arena 0 len=4 lane=f64x4
            node $x = ArenaInput f64x4 a=0
            node $c = ConstF64 f64x1 const=1.0
            node $fma = Fma f64x4 a=$x b=$c c=69999
            output $fma
        ";
        assert_eq!(
            crate::transpiler::assemble_rls(rls_overflow),
            Err(crate::transpiler::RlsError::Symbol(
                crate::transpiler::TranspileError::FmaOperandIndexOverflow {
                    symbol: "69999".to_owned(),
                    node_idx: 69_999,
                }
            ))
        );
        // Missing `c=` operand in `.rls` text assembly must be rejected rather than defaulting to node `#0`:
        let rls_missing_c = "\
            arena 0 len=4 lane=f64x4
            node $x = ArenaInput f64x4 a=0
            node $c = ConstF64 f64x1 const=1.0
            node $fma = Fma f64x4 a=$x b=$c
            output $fma
        ";
        assert!(matches!(
            crate::transpiler::assemble_rls(rls_missing_c),
            Err(crate::transpiler::RlsError::LineParse { line: 4, .. })
        ));

        let _ = std::fs::remove_file(&exe_path);
        let _ = std::fs::remove_file(&data_path);
        Ok(())
    }

    #[test]
    fn test_p1b_vectorized_avx2_ln_and_exp_precision_and_ieee754_edge_cases() -> Result<(), String>
    {
        let pid = std::process::id();
        let exe_path = std::env::temp_dir().join(format!("rul_test_p1b_math_{pid}"));
        let data_path = std::env::temp_dir().join(format!("rul_test_p1b_math_{pid}.bin"));

        // 1. Precision across wide dynamic range (1,027 elements = 256 AVX2 vectors + 3 scalar tail)
        //    with live `ymm2`/`ymm3` columns across `Ln` and `Exp` to verify scratch save/restore.
        let n = 1_027usize;
        let col_ln_in: Vec<f64> = (0..n)
            .map(|i| {
                let frac = (i as f64 + 0.5) / (n as f64);
                // Span exponents from 2^-400 (~3.8e-121) to 2^+400 (~2.58e+120) including near-1.0 values
                let exp2 = (frac - 0.5) * 800.0;
                2.0_f64.powf(exp2) * (0.7 + 0.6 * ((i % 17) as f64 / 16.0))
            })
            .collect();
        let col_exp_in: Vec<f64> = (0..n)
            .map(|i| {
                let frac = (i as f64 + 0.5) / (n as f64);
                // Span [-700.0, +700.0]
                (frac - 0.5) * 1400.0
            })
            .collect();

        let rls_ln = format!(
            "\
            arena 0 len={n} lane=f64x8
            node $x = ArenaInput f64x8 a=0
            node $c1 = ConstF64 f64x1 const=1.5
            node $c2 = ConstF64 f64x1 const=0.25
            node $keep2 = Add f64x8 a=$x b=$c1
            node $keep3 = Sub f64x8 a=$x b=$c2
            node $lnx = Ln f64x8 a=$x
            node $comb = Add f64x8 a=$keep2 b=$keep3
            node $ratio = Div f64x8 a=$keep2 b=$comb
            node $tot = Add f64x8 a=$lnx b=$ratio
            node $out = ReduceSum f64x1 a=$tot
            output $out
            "
        );
        let prog_ln = crate::transpiler::assemble_rls(&rls_ln).map_err(|e| e.to_string())?;
        let RuntimeValue::Scalar(ln_t0) = execute_tier0_interpreter(&prog_ln, &[&col_ln_in])
            .map_err(|e| e.to_string())?
            .output
        else {
            return Err("Tier 0 did not return Scalar".to_owned());
        };
        let rep_ln =
            crate::aot::compile_rl_to_executable(&prog_ln, &exe_path).map_err(|e| e.to_string())?;
        assert!(rep_ln.avx2_vectorized);
        crate::aot::write_arena_data_file(&data_path, &[&col_ln_in]).map_err(|e| e.to_string())?;
        let ln_aot = crate::aot::run_compiled_executable(&exe_path, &data_path)
            .map_err(|e| e.to_string())?;
        let ln_rel_err = (ln_aot - ln_t0).abs() / ln_t0.abs().max(1.0);
        assert!(
            ln_rel_err < 1e-13,
            "AVX2 Ln relative error too large: t0={ln_t0}, aot={ln_aot}, rel_err={ln_rel_err:e}"
        );

        let rls_exp = format!(
            "\
            arena 0 len={n} lane=f64x8
            node $x = ArenaInput f64x8 a=0
            node $c1 = ConstF64 f64x1 const=1.5
            node $c2 = ConstF64 f64x1 const=0.25
            node $keep2 = Add f64x8 a=$x b=$c1
            node $keep3 = Sub f64x8 a=$x b=$c2
            node $expx = Exp f64x8 a=$x
            node $comb = Add f64x8 a=$keep2 b=$keep3
            node $prod = Mul f64x8 a=$expx b=$comb
            node $out = ReduceMax f64x1 a=$expx
            node $s2 = ReduceMin f64x1 a=$comb
            node $final = Add f64x1 a=$out b=$s2
            output $final
            "
        );
        let prog_exp = crate::transpiler::assemble_rls(&rls_exp).map_err(|e| e.to_string())?;
        let RuntimeValue::Scalar(exp_t0) = execute_tier0_interpreter(&prog_exp, &[&col_exp_in])
            .map_err(|e| e.to_string())?
            .output
        else {
            return Err("Tier 0 did not return Scalar".to_owned());
        };
        crate::aot::compile_rl_to_executable(&prog_exp, &exe_path).map_err(|e| e.to_string())?;
        crate::aot::write_arena_data_file(&data_path, &[&col_exp_in]).map_err(|e| e.to_string())?;
        let exp_aot = crate::aot::run_compiled_executable(&exe_path, &data_path)
            .map_err(|e| e.to_string())?;
        let exp_rel_err = (exp_aot - exp_t0).abs() / exp_t0.abs().max(1.0);
        assert!(
            exp_rel_err < 1e-13,
            "AVX2 Exp relative error too large: t0={exp_t0}, aot={exp_aot}, rel_err={exp_rel_err:e}"
        );

        // Also test ReduceSum of Exp on [-10.0, +10.0] across 1,027 elements
        let col_exp_mid: Vec<f64> = (0..n)
            .map(|i| -10.0 + 20.0 * ((i as f64) / (n as f64)))
            .collect();
        let rls_exp_sum = format!(
            "\
            arena 0 len={n} lane=f64x8
            node $x = ArenaInput f64x8 a=0
            node $ex = Exp f64x8 a=$x
            node $out = ReduceSum f64x1 a=$ex
            output $out
            "
        );
        let prog_exp_sum =
            crate::transpiler::assemble_rls(&rls_exp_sum).map_err(|e| e.to_string())?;
        let RuntimeValue::Scalar(es_t0) = execute_tier0_interpreter(&prog_exp_sum, &[&col_exp_mid])
            .map_err(|e| e.to_string())?
            .output
        else {
            return Err("Tier 0 did not return Scalar".to_owned());
        };
        crate::aot::compile_rl_to_executable(&prog_exp_sum, &exe_path)
            .map_err(|e| e.to_string())?;
        crate::aot::write_arena_data_file(&data_path, &[&col_exp_mid])
            .map_err(|e| e.to_string())?;
        let es_aot = crate::aot::run_compiled_executable(&exe_path, &data_path)
            .map_err(|e| e.to_string())?;
        assert!(
            (es_aot - es_t0).abs() / es_t0.abs().max(1.0) < 1e-13,
            "AVX2 Exp sum mismatch: t0={es_t0}, aot={es_aot}"
        );

        // 2. IEEE-754 special cases inside the 4-element 256-bit AVX2 loop (`len = 4`)
        let rls_ln_4 = "\
            arena 0 len=4 lane=f64x4
            node $x = ArenaInput f64x4 a=0
            node $y = Ln f64x4 a=$x
            node $out = ReduceSum f64x1 a=$y
            output $out
        ";
        let prog_ln_4 = crate::transpiler::assemble_rls(rls_ln_4).map_err(|e| e.to_string())?;
        crate::aot::compile_rl_to_executable(&prog_ln_4, &exe_path).map_err(|e| e.to_string())?;

        for (label, vec4, check_fn) in [
            (
                "ln_pos_zero",
                [0.0, 1.0, 1.0, 1.0],
                (|v: f64| v == f64::NEG_INFINITY) as fn(f64) -> bool,
            ),
            (
                "ln_neg_zero",
                [1.0, -0.0, 1.0, 1.0],
                (|v: f64| v == f64::NEG_INFINITY) as fn(f64) -> bool,
            ),
            (
                "ln_pos_inf",
                [1.0, 1.0, f64::INFINITY, 1.0],
                (|v: f64| v == f64::INFINITY) as fn(f64) -> bool,
            ),
            (
                "ln_negative",
                [1.0, 1.0, 1.0, -2.5],
                f64::is_nan as fn(f64) -> bool,
            ),
            (
                "ln_neg_inf",
                [f64::NEG_INFINITY, 1.0, 1.0, 1.0],
                f64::is_nan as fn(f64) -> bool,
            ),
            (
                "ln_nan",
                [1.0, f64::NAN, 1.0, 1.0],
                f64::is_nan as fn(f64) -> bool,
            ),
        ] {
            crate::aot::write_arena_data_file(&data_path, &[&vec4]).map_err(|e| e.to_string())?;
            let obs = crate::aot::run_compiled_executable(&exe_path, &data_path)
                .map_err(|e| e.to_string())?;
            assert!(
                check_fn(obs),
                "AVX2 Ln special case `{label}` failed: observed={obs}"
            );
        }

        let rls_exp_4 = "\
            arena 0 len=4 lane=f64x4
            node $x = ArenaInput f64x4 a=0
            node $y = Exp f64x4 a=$x
            node $out = ReduceSum f64x1 a=$y
            output $out
        ";
        let prog_exp_4 = crate::transpiler::assemble_rls(rls_exp_4).map_err(|e| e.to_string())?;
        crate::aot::compile_rl_to_executable(&prog_exp_4, &exe_path).map_err(|e| e.to_string())?;

        for (label, vec4, check_fn) in [
            (
                "exp_underflow_zero",
                [-800.0, -900.0, f64::NEG_INFINITY, -750.0],
                (|v: f64| v == 0.0 && !v.is_sign_negative()) as fn(f64) -> bool,
            ),
            (
                "exp_overflow_inf",
                [0.0, 800.0, 0.0, 0.0],
                (|v: f64| v == f64::INFINITY) as fn(f64) -> bool,
            ),
            (
                "exp_overflow_first_inf_boundary",
                [0.0, f64::from_bits(0x4086_2E42_FEFA_39F0), 0.0, 0.0],
                (|v: f64| v == f64::INFINITY) as fn(f64) -> bool,
            ),
            (
                "exp_pos_inf",
                [0.0, 0.0, f64::INFINITY, 0.0],
                (|v: f64| v == f64::INFINITY) as fn(f64) -> bool,
            ),
            (
                "exp_nan",
                [0.0, 0.0, 0.0, f64::NAN],
                f64::is_nan as fn(f64) -> bool,
            ),
        ] {
            crate::aot::write_arena_data_file(&data_path, &[&vec4]).map_err(|e| e.to_string())?;
            let obs = crate::aot::run_compiled_executable(&exe_path, &data_path)
                .map_err(|e| e.to_string())?;
            assert!(
                check_fn(obs),
                "AVX2 Exp special case `{label}` failed: observed={obs}"
            );
        }

        // 3. Subnormal `Ln`, near-overflow `Exp` (`k = 1024`), and gradual-underflow subnormal `Exp`
        //    inside the 4-lane AVX2 loop (`len = 4`) compared against Tier 0 (`libm`)
        crate::aot::compile_rl_to_executable(&prog_ln_4, &exe_path).map_err(|e| e.to_string())?;
        for ln_sub_vec in [
            [
                1e-310_f64,
                f64::from_bits(1), // 5e-324 (min positive subnormal)
                f64::from_bits(0x000F_FFFF_FFFF_FFFF), // max subnormal
                f64::MIN_POSITIVE, // min normal (`2^-1022`)
            ],
            [1e-320_f64, 1e-315_f64, 1e-309_f64, 1.0_f64],
        ] {
            let RuntimeValue::Scalar(t0_sub) =
                execute_tier0_interpreter(&prog_ln_4, &[&ln_sub_vec])
                    .map_err(|e| e.to_string())?
                    .output
            else {
                return Err("Tier 0 did not return Scalar".to_owned());
            };
            crate::aot::write_arena_data_file(&data_path, &[&ln_sub_vec])
                .map_err(|e| e.to_string())?;
            let aot_sub = crate::aot::run_compiled_executable(&exe_path, &data_path)
                .map_err(|e| e.to_string())?;
            let rel = (aot_sub - t0_sub).abs() / t0_sub.abs().max(1.0);
            assert!(
                rel < 1e-13,
                "AVX2 Ln subnormal mismatch for {ln_sub_vec:?}: t0={t0_sub}, aot={aot_sub}, rel={rel:e}"
            );
        }

        crate::aot::compile_rl_to_executable(&prog_exp_4, &exe_path).map_err(|e| e.to_string())?;
        for exp_edge_vec in [
            // `k = 1024` finite near-overflow range `(709.436, 709.78271289338397]`
            [709.5_f64, 0.0, 0.0, 0.0],
            [f64::from_bits(0x4086_2E42_FEFA_39EF), 0.0, 0.0, 0.0],
            // Gradual-underflow subnormal outputs `[-745.1332191019411, -708.3964185322641)`
            [-708.5_f64, 0.0, 0.0, 0.0],
            [-710.0_f64, -720.0_f64, -730.0_f64, -740.0_f64],
            [
                -744.440_071_921_381_2_f64,
                f64::from_bits(0xC087_4910_D52D_3051), // last non-zero subnormal `5e-324`
                f64::from_bits(0xC087_4910_D52D_3052), // first exact `+0.0`
                -800.0,
            ],
        ] {
            let RuntimeValue::Scalar(t0_edge) =
                execute_tier0_interpreter(&prog_exp_4, &[&exp_edge_vec])
                    .map_err(|e| e.to_string())?
                    .output
            else {
                return Err("Tier 0 did not return Scalar".to_owned());
            };
            crate::aot::write_arena_data_file(&data_path, &[&exp_edge_vec])
                .map_err(|e| e.to_string())?;
            let aot_edge = crate::aot::run_compiled_executable(&exe_path, &data_path)
                .map_err(|e| e.to_string())?;
            assert!(
                aot_edge.is_finite() && t0_edge.is_finite(),
                "expected finite Exp for {exp_edge_vec:?}: t0={t0_edge}, aot={aot_edge}"
            );
            let rel = (aot_edge - t0_edge).abs() / t0_edge.abs().max(f64::MIN_POSITIVE);
            assert!(
                rel < 1e-13,
                "AVX2 Exp boundary mismatch for {exp_edge_vec:?}: t0={t0_edge:e}, aot={aot_edge:e}, rel={rel:e}"
            );
        }

        // Also check single-lane subnormal `exp(-744.4400719213812) == 5e-324` without `0.0` lanes masking it:
        let tiny_vec = [
            f64::from_bits(0xC087_4910_D52D_3051),
            -745.133_219_101_941_2_f64,
            -800.0,
            -900.0,
        ];
        crate::aot::write_arena_data_file(&data_path, &[&tiny_vec]).map_err(|e| e.to_string())?;
        let aot_tiny = crate::aot::run_compiled_executable(&exe_path, &data_path)
            .map_err(|e| e.to_string())?;
        assert_eq!(
            aot_tiny.to_bits(),
            1,
            "expected smallest subnormal 5e-324 (bits=1), got {aot_tiny:e} (bits={})",
            aot_tiny.to_bits()
        );

        let _ = std::fs::remove_file(&exe_path);
        let _ = std::fs::remove_file(&data_path);
        Ok(())
    }
}
