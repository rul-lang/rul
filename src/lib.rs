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
    fn test_new_opcodes_div_min_max_cmplt_cmpeq_all_tiers_and_aot() {
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
        let prog = crate::transpiler::assemble_rls(rls_src);
        assert!(prog.is_ok());
        let Ok(p) = prog else { return };

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

        let r0 = execute_tier0_interpreter(&p, &[&col_a, &col_b]);
        let r1 = execute_tier1_stencil(&p, &[&col_a, &col_b]);
        let r2 = execute_tier2_fused(&p, &[&col_a, &col_b]);
        assert!(r0.is_ok() && r1.is_ok() && r2.is_ok());
        let (Ok(rep0), Ok(rep1), Ok(rep2)) = (r0, r1, r2) else {
            return;
        };
        assert_eq!(rep0.output, rep1.output);
        assert_eq!(rep0.output, rep2.output);
        assert_eq!(rep2.telemetry.intermediate_bytes_allocated, 0);

        // Verify standalone Linux x86_64 AVX2 ELF64 AOT binary matches Tier 0
        let exe_path = std::env::temp_dir().join("rul_test_new_opcodes_aot_bin");
        let data_path = std::env::temp_dir().join("rul_test_new_opcodes_aot_data.bin");
        let aot_rep = crate::aot::compile_rl_to_executable(&p, &exe_path);
        assert!(aot_rep.is_ok());
        assert!(crate::aot::write_arena_data_file(&data_path, &[&col_a, &col_b]).is_ok());
        let aot_val = crate::aot::run_compiled_executable(&exe_path, &data_path);
        assert!(aot_val.is_ok());
        if let (Ok(v_aot), RuntimeValue::Scalar(v0)) = (aot_val, rep0.output) {
            assert!(
                (v_aot - v0).abs() < 1e-6,
                "New opcodes AOT mismatch: aot={v_aot}, tier0={v0}"
            );
        }
        let _ = std::fs::remove_file(&exe_path);
        let _ = std::fs::remove_file(&data_path);
    }

    #[test]
    fn test_community_ieee754_challenge_suite_and_edge_cases() {
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
            let Ok(p1) = crate::transpiler::assemble_rls(rls_t1) else {
                return;
            };
            let in1 = [100.0_f64, 250.5, 99.9, 1000.0, 0.0];
            let Ok(rep1) = execute_tier2_fused(&p1, &[&in1]) else {
                return;
            };
            let RuntimeValue::ColumnF64(out1) = rep1.output else {
                return;
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
            let Ok(p2) = crate::transpiler::assemble_rls(rls_t2) else {
                return;
            };
            let a0 = [1.5_f64, 2.0, -3.0, 4.0];
            let a1 = [2.0_f64, 0.5, -2.0, 0.25];
            let Ok(rep2) = execute_tier2_fused(&p2, &[&a0, &a1]) else {
                return;
            };
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
            let Ok(p3) = crate::transpiler::assemble_rls(rls_t3) else {
                return;
            };
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
            let Ok(rep3) = execute_tier2_fused(&p3, &[&a0, &a1]) else {
                return;
            };
            let RuntimeValue::ColumnF64(ref out3) = rep3.output else {
                return;
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
            let Ok(trace_rep) = trace_program(&p3, &[&a0, &a1]) else {
                return;
            };
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
            let Ok(pg) = crate::transpiler::assemble_rls(rls_gather) else {
                return;
            };
            let Ok(summary) = verify_program(&pg) else {
                return;
            };
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
    }
}
