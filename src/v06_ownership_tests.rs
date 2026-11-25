//! Milestone 2a (`v0.6`) tests: Statically named region typestate, frame-rule ownership
//! across control-flow joins and calls, Tier 0 == ELF64 AOT equivalence, `.rls` lossless
//! round-trip, and single-pass `StaticCostSummary`.

use crate::aot::{compile_rl_to_executable, run_compiled_scalar_executable};
use crate::bytecode::{
    func_frame_units, ArenaDescriptor, ElemType, NodeFrame, Opcode, ProofRuleKind, RulProgram,
    ScheduleHint, TokenOp, TypeLane, CAP_NONE,
};
use crate::engine::{
    execute_tier0_interpreter, execute_tier1_stencil, execute_tier2_fused, ExecError, RuntimeValue,
    TypedScalar,
};
use crate::transpiler::{assemble_rls, disassemble_rl, program_to_rls};
use crate::verifier::{
    verify_program, verify_program_with_join_work, StaticCostSummary, ValueShape, VerifyErrorKind,
};

fn verify_rls(src: &str) -> Result<crate::verifier::VerifiedSummary, VerifyErrorKind> {
    let prog = assemble_rls(src).unwrap_or_else(|e| {
        // Test helper: avoid unwrap/panic lints by returning a sentinel via Err on parse failure.
        let _ = e;
        RulProgram {
            capability_mask: 0,
            arenas: Vec::new(),
            proofs: Vec::new(),
            nodes: Vec::new(),
            output_node: 0,
        }
    });
    assert!(
        !prog.nodes.is_empty(),
        "test .rls program failed to assemble:\n{src}"
    );
    verify_program(&prog).map_err(|d| d.kind)
}

#[test]
fn test_v06_ownership_verifier() -> Result<(), String> {
    // 1. Frame-rule composition: caller leases slot 0, calls pure empty-footprint helper (f0)
    //    AND calls slot-1 helper (f1, dom={1}) while slot 0 remains leased, then ends lease
    //    and mutates slot 0.
    let frame_rule_ok = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=16 witness=i64:plain
proof 2 rule=region_decl target=1 lo=0 hi=16 witness=i64:plain
proof 3 rule=func_sig target=0 lo=1 hi=i64 witness=i64
proof 4 rule=func_sig target=1 lo=1 hi=i64 witness=i64
proof 5 rule=func_sig target=2 lo=0 hi=0 witness=i64
proof func_region_effect func=0 slot=none
proof func_region_effect func=1 slot=1 pre=owned post=owned
proof func_region_effect func=2 slot=0 pre=owned post=owned
proof func_region_effect func=2 slot=1 pre=owned post=owned
node $f0 = Func i64x1 a=0 b=0
node $p0 = LocalGet i64x1 a=0 b=0
node $k1 = ConstI64 i64x1 const_i64=1
node $r0 = Add i64x1 a=$p0 b=$k1
node $e0 = End i64x1 a=$r0 b=1
node $f1 = Func i64x1 a=1 b=0
node $p1 = LocalGet i64x1 a=0 b=0
node $z1 = ConstI64 i64x1 const_i64=0
node $n1 = ConstI64 i64x1 const_i64=2
node $rn1 = RegionNew i64x1 a=1 b=$n1
node $st1 = RegionStore i64x1 a=1 b=$z1 c=$p1
node $ld1 = RegionLoad i64x1 a=1 b=$z1
node $e1 = End i64x1 a=$ld1 b=1
node $f2 = Func i64x1 a=2 b=0
node $z = ConstI64 i64x1 const_i64=0
node $sz = ConstI64 i64x1 const_i64=4
node $v = ConstI64 i64x1 const_i64=41
node $rn0 = RegionNew i64x1 a=0 b=$sz
node $st0 = RegionStore i64x1 a=0 b=$z c=$v
node $ls0 = RegionLease i64x1 a=0 b=0
node $ld0 = RegionLoad i64x1 a=0 b=$z
node $c0 = Call i64x1 a=0 b=$ld0
node $c1 = Call i64x1 a=1 b=$c0
node $el0 = RegionEndLease i64x1 a=0 b=0
node $st0b = RegionStore i64x1 a=0 b=$z c=$c1
node $out = RegionLoad i64x1 a=0 b=$z
node $e2 = End i64x1 a=$out b=1
output $f2
";
    assert!(verify_rls(frame_rule_ok).is_ok());

    // 2. Entry function precondition MUST be Owned on dom(entry).
    let entry_pre_leased = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
proof 2 rule=func_sig target=0 lo=0 hi=0 witness=i64
proof func_region_effect func=0 slot=0 pre=leased post=leased
node $f0 = Func i64x1 a=0 b=0
node $z = ConstI64 i64x1 const_i64=0
node $e0 = End i64x1 a=$z b=1
output $f0
";
    assert!(matches!(
        verify_rls(entry_pre_leased),
        Err(VerifyErrorKind::InvalidRegionEffect { func_id: 0, .. })
    ));

    let entry_pre_moved = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
proof 2 rule=func_sig target=0 lo=0 hi=0 witness=i64
proof func_region_effect func=0 slot=0 pre=moved post=moved
node $f0 = Func i64x1 a=0 b=0
node $z = ConstI64 i64x1 const_i64=0
node $e0 = End i64x1 a=$z b=1
output $f0
";
    assert!(matches!(
        verify_rls(entry_pre_moved),
        Err(VerifyErrorKind::InvalidRegionEffect { func_id: 0, .. })
    ));

    // 3. All mutation & nested lease rejections under live read lease.
    for (label, bad_op) in [
        ("RegionStore", "node $bad = RegionStore i64x1 a=0 b=$z c=$v"),
        ("RegionGrow", "node $bad = RegionGrow i64x1 a=0 b=$one"),
        ("RegionNew", "node $bad = RegionNew i64x1 a=0 b=$one"),
        ("RegionFree", "node $bad = RegionFree i64x1 a=0 b=0"),
        ("RegionDrop", "node $bad = RegionDrop i64x1 a=0 b=0"),
        (
            "nested RegionLease",
            "node $bad = RegionLease i64x1 a=0 b=0",
        ),
        ("RegionMove src", "node $bad = RegionMove i64x1 a=0 b=1"),
    ] {
        let src = format!(
            "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
proof 2 rule=region_decl target=1 lo=0 hi=8 witness=i64:plain
node $z = ConstI64 i64x1 const_i64=0
node $one = ConstI64 i64x1 const_i64=1
node $v = ConstI64 i64x1 const_i64=7
node $dr1 = RegionDrop i64x1 a=1 b=0
node $rn0 = RegionNew i64x1 a=0 b=$one
node $ls = RegionLease i64x1 a=0 b=0
{bad_op}
node $el = RegionEndLease i64x1 a=0 b=0
output $v
"
        );
        assert_eq!(
            verify_rls(&src),
            Err(VerifyErrorKind::MutationUnderLiveLease { slot: 0 }),
            "expected MutationUnderLiveLease for {label}"
        );
    }

    // RegionMove into a Leased destination must also fail with MutationUnderLiveLease { slot: 1 }.
    let move_into_leased = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
proof 2 rule=region_decl target=1 lo=0 hi=8 witness=i64:plain
node $v = ConstI64 i64x1 const_i64=7
node $ls1 = RegionLease i64x1 a=1 b=0
node $mv = RegionMove i64x1 a=0 b=1
output $v
";
    assert_eq!(
        verify_rls(move_into_leased),
        Err(VerifyErrorKind::MutationUnderLiveLease { slot: 1 })
    );

    // 4. All accesses on a Moved slot rejected with UseOfMovedRegion { slot: 0 }.
    for (label, bad_op) in [
        ("RegionLen", "node $bad = RegionLen i64x1 a=0 b=0"),
        ("RegionLoad", "node $bad = RegionLoad i64x1 a=0 b=$z"),
        ("RegionStore", "node $bad = RegionStore i64x1 a=0 b=$z c=$v"),
        ("RegionGrow", "node $bad = RegionGrow i64x1 a=0 b=$one"),
        ("RegionFree", "node $bad = RegionFree i64x1 a=0 b=0"),
        ("RegionLease", "node $bad = RegionLease i64x1 a=0 b=0"),
        ("RegionEndLease", "node $bad = RegionEndLease i64x1 a=0 b=0"),
        ("RegionDrop", "node $bad = RegionDrop i64x1 a=0 b=0"),
        ("RegionMove src", "node $bad = RegionMove i64x1 a=0 b=1"),
    ] {
        let src = format!(
            "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
proof 2 rule=region_decl target=1 lo=0 hi=8 witness=i64:plain
node $z = ConstI64 i64x1 const_i64=0
node $one = ConstI64 i64x1 const_i64=1
node $v = ConstI64 i64x1 const_i64=7
node $dr0 = RegionDrop i64x1 a=0 b=0
node $dr1 = RegionDrop i64x1 a=1 b=0
{bad_op}
output $v
"
        );
        assert_eq!(
            verify_rls(&src),
            Err(VerifyErrorKind::UseOfMovedRegion { slot: 0 }),
            "expected UseOfMovedRegion for {label}"
        );
    }

    // 5. Restoring Owned on a Moved Plain slot via RegionNew and RegionMove dst succeeds.
    let restore_moved_ok = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
proof 2 rule=region_decl target=1 lo=0 hi=8 witness=i64:plain
node $one = ConstI64 i64x1 const_i64=1
node $dr0 = RegionDrop i64x1 a=0 b=0
node $rn0 = RegionNew i64x1 a=0 b=$one
node $dr1 = RegionDrop i64x1 a=1 b=0
node $mv = RegionMove i64x1 a=0 b=1
node $rn0b = RegionNew i64x1 a=0 b=$one
node $l0 = RegionLen i64x1 a=0 b=0
node $l1 = RegionLen i64x1 a=1 b=0
node $sum = Add i64x1 a=$l0 b=$l1
output $sum
";
    assert!(verify_rls(restore_moved_ok).is_ok());

    // 6. RegionEndLease on Owned slot, RegionMove into Owned slot, mismatched elem/role/max_len/min_len, RegionDrop non-plain/min_len>0.
    let end_lease_owned = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
node $el = RegionEndLease i64x1 a=0 b=0
node $z = ConstI64 i64x1 const_i64=0
output $z
";
    assert!(matches!(
        verify_rls(end_lease_owned),
        Err(VerifyErrorKind::RegionTypestateMismatch { .. })
    ));

    let move_into_owned = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
proof 2 rule=region_decl target=1 lo=0 hi=8 witness=i64:plain
node $mv = RegionMove i64x1 a=0 b=1
node $z = ConstI64 i64x1 const_i64=0
output $z
";
    assert!(matches!(
        verify_rls(move_into_owned),
        Err(VerifyErrorKind::RegionTypestateMismatch { .. })
    ));

    let move_elem_mismatch = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
proof 2 rule=region_decl target=1 lo=0 hi=8 witness=u8:plain
node $dr1 = RegionDrop u8x1 a=1 b=0
node $mv = RegionMove i64x1 a=0 b=1
node $z = ConstI64 i64x1 const_i64=0
output $z
";
    assert_eq!(
        verify_rls(move_elem_mismatch),
        Err(VerifyErrorKind::ElementTypeMismatch {
            expected: ElemType::I64,
            actual: ElemType::U8,
        })
    );

    let move_max_len_exceeded = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=16 witness=i64:plain
proof 2 rule=region_decl target=1 lo=0 hi=8 witness=i64:plain
node $dr1 = RegionDrop i64x1 a=1 b=0
node $mv = RegionMove i64x1 a=0 b=1
node $z = ConstI64 i64x1 const_i64=0
output $z
";
    assert!(matches!(
        verify_rls(move_max_len_exceeded),
        Err(VerifyErrorKind::InvalidRegionDecl { slot: 1, .. })
    ));

    let move_dst_min_len_nonzero = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
proof 2 rule=region_decl target=1 lo=2 hi=8 witness=i64:plain
node $mv = RegionMove i64x1 a=0 b=1
node $z = ConstI64 i64x1 const_i64=0
output $z
";
    assert!(matches!(
        verify_rls(move_dst_min_len_nonzero),
        Err(VerifyErrorKind::InvalidRegionDecl { slot: 1, .. })
    ));

    let drop_non_plain = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=u8:output
node $dr = RegionDrop u8x1 a=0 b=0
node $z = ConstI64 i64x1 const_i64=0
output $z
";
    assert!(matches!(
        verify_rls(drop_non_plain),
        Err(VerifyErrorKind::InvalidRegionDecl { slot: 0, .. })
    ));

    let drop_min_len_nonzero = "\
cap NONE
proof 1 rule=region_decl target=0 lo=2 hi=8 witness=i64:plain
node $dr = RegionDrop i64x1 a=0 b=0
node $z = ConstI64 i64x1 const_i64=0
output $z
";
    assert!(matches!(
        verify_rls(drop_min_len_nonzero),
        Err(VerifyErrorKind::InvalidRegionDecl { slot: 0, .. })
    ));

    // 7. Footprint violations: RegionOutsideFootprint and CalleeFootprintExceedsCaller.
    let outside_footprint = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
proof 2 rule=func_sig target=0 lo=0 hi=0 witness=i64
proof func_region_effect func=0 slot=none
node $f0 = Func i64x1 a=0 b=0
node $l = RegionLen i64x1 a=0 b=0
node $e0 = End i64x1 a=$l b=1
output $f0
";
    assert_eq!(
        verify_rls(outside_footprint),
        Err(VerifyErrorKind::RegionOutsideFootprint { slot: 0 })
    );

    let callee_exceeds_caller = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
proof 2 rule=region_decl target=1 lo=0 hi=8 witness=i64:plain
proof 3 rule=func_sig target=0 lo=0 hi=0 witness=i64
proof 4 rule=func_sig target=1 lo=0 hi=0 witness=i64
proof func_region_effect func=0 slot=1 pre=owned post=owned
proof func_region_effect func=1 slot=0 pre=owned post=owned
node $f0 = Func i64x1 a=0 b=0
node $l = RegionLen i64x1 a=1 b=0
node $e0 = End i64x1 a=$l b=1
node $f1 = Func i64x1 a=1 b=0
node $c = Call i64x1 a=0
node $e1 = End i64x1 a=$c b=1
output $f1
";
    assert_eq!(
        verify_rls(callee_exceeds_caller),
        Err(VerifyErrorKind::CalleeFootprintExceedsCaller {
            callee_func_id: 0,
            extra_mask: 1u64 << 1,
        })
    );

    // 8. Unreachable bottom typestate (⊥): dead arm after Return/Br does not cause false
    //    typestate error and does not constrain Else/End joins.
    let unreachable_bottom_ok = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
proof 2 rule=func_sig target=0 lo=0 hi=0 witness=i64
proof func_region_effect func=0 slot=0 pre=owned post=owned
node $f0 = Func i64x1 a=0 b=0
node $z = ConstI64 i64x1 const_i64=0
node $one = ConstI64 i64x1 const_i64=1
node $cond = CmpEq maskx1 a=$z b=$one
node $if = If i64x1 a=$cond b=0
node $dr = RegionDrop i64x1 a=0 b=0
node $rn = RegionNew i64x1 a=0 b=$one
node $ret = Return i64x1 a=$one b=0
node $dead_drop = RegionDrop i64x1 a=0 b=0
node $dead_store = RegionStore i64x1 a=0 b=$z c=$one
node $el = Else i64x1 a=0 b=0
node $ls = RegionLease i64x1 a=0 b=0
node $en = RegionEndLease i64x1 a=0 b=0
node $end_if = End i64x1 a=0 b=0
node $len = RegionLen i64x1 a=0 b=0
node $e0 = End i64x1 a=$len b=1
output $f0
";
    assert!(verify_rls(unreachable_bottom_ok).is_ok());

    // 9. Branch join mismatch (If/Else and If without Else) -> RegionTypestateMismatch.
    let if_else_mismatch = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
node $z = ConstI64 i64x1 const_i64=0
node $one = ConstI64 i64x1 const_i64=1
node $cond = CmpEq maskx1 a=$z b=$one
node $if = If i64x1 a=$cond b=0
node $dr = RegionDrop i64x1 a=0 b=0
node $el = Else i64x1 a=0 b=0
node $end_if = End i64x1 a=0 b=0
output $z
";
    assert!(matches!(
        verify_rls(if_else_mismatch),
        Err(VerifyErrorKind::RegionTypestateMismatch { .. })
    ));

    let if_no_else_mismatch = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
node $z = ConstI64 i64x1 const_i64=0
node $one = ConstI64 i64x1 const_i64=1
node $cond = CmpEq maskx1 a=$z b=$one
node $if = If i64x1 a=$cond b=0
node $ls = RegionLease i64x1 a=0 b=0
node $end_if = End i64x1 a=0 b=0
output $z
";
    assert!(matches!(
        verify_rls(if_no_else_mismatch),
        Err(VerifyErrorKind::RegionTypestateMismatch { .. })
    ));

    // 10. Loop typestate invariant violation (fall-through and back-edge BrIf).
    let loop_fallthrough_violation = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
proof 2 rule=loop_bound target=0 lo=0 hi=4 witness=0
node $z = ConstI64 i64x1 const_i64=0
node $lp = Loop i64x1 proof=2 a=0 b=0
node $ls = RegionLease i64x1 a=0 b=0
node $end_lp = End i64x1 a=0 b=0
output $z
";
    assert!(matches!(
        verify_rls(loop_fallthrough_violation),
        Err(VerifyErrorKind::LoopTypestateInvariantViolation { .. })
    ));

    let loop_backedge_violation = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
proof 2 rule=loop_bound target=0 lo=0 hi=4 witness=0
node $z = ConstI64 i64x1 const_i64=0
node $one = ConstI64 i64x1 const_i64=1
node $cond = CmpEq maskx1 a=$z b=$one
node $lp = Loop i64x1 proof=2 a=0 b=0
node $dr = RegionDrop i64x1 a=0 b=0
node $br = BrIf maskx1 a=$cond b=0
node $rn = RegionNew i64x1 a=0 b=$one
node $end_lp = End i64x1 a=0 b=0
output $z
";
    assert!(matches!(
        verify_rls(loop_backedge_violation),
        Err(VerifyErrorKind::LoopTypestateInvariantViolation { .. })
    ));

    // 11. Multi-arena AffineToken joins across If/Else/End and Br/BrIf.
    let multi_arena_join_ok = "\
cap NONE
arena 0 len=4 lane=f64x8 mut=true align=64 ver=1
arena 1 len=4 lane=f64x8 mut=true align=64 ver=1
node $a0 = ArenaInput f64x8 tok=read(0,1) a=0 b=0
node $a1 = ArenaInput f64x8 tok=read(1,1) a=1 b=0
node $k1 = ConstF64 f64x1 const=1.0
node $k2 = ConstF64 f64x1 const=2.0
node $cond = CmpGt maskx1 a=$k2 b=$k1
node $if = If i64x1 a=$cond b=0
node $m0a = ArenaMutate f64x8 tok=consume(0,1) a=0 b=$a0
node $m1a = ArenaMutate f64x8 tok=consume(1,1) a=1 b=$a1
node $el = Else i64x1 a=0 b=0
node $m1b = ArenaMutate f64x8 tok=consume(1,1) a=1 b=$a1
node $m0b = ArenaMutate f64x8 tok=consume(0,1) a=0 b=$a0
node $end_if = End i64x1 a=0 b=0
node $r0 = ArenaInput f64x8 tok=read(0,2) a=0 b=0
node $r1 = ArenaInput f64x8 tok=read(1,2) a=1 b=0
node $sum = Add f64x8 a=$r0 b=$r1
node $out = ReduceSum f64x1 a=$sum b=0
output $out
";
    assert!(verify_rls(multi_arena_join_ok).is_ok());

    let multi_arena_join_mismatch = "\
cap NONE
arena 0 len=4 lane=f64x8 mut=true align=64 ver=1
arena 1 len=4 lane=f64x8 mut=true align=64 ver=1
node $a0 = ArenaInput f64x8 tok=read(0,1) a=0 b=0
node $a1 = ArenaInput f64x8 tok=read(1,1) a=1 b=0
node $k1 = ConstF64 f64x1 const=1.0
node $k2 = ConstF64 f64x1 const=2.0
node $cond = CmpGt maskx1 a=$k2 b=$k1
node $if = If i64x1 a=$cond b=0
node $m0a = ArenaMutate f64x8 tok=consume(0,1) a=0 b=$a0
node $m1a = ArenaMutate f64x8 tok=consume(1,1) a=1 b=$a1
node $el = Else i64x1 a=0 b=0
node $m0b = ArenaMutate f64x8 tok=consume(0,1) a=0 b=$a0
node $end_if = End i64x1 a=0 b=0
output $k1
";
    assert_eq!(
        verify_rls(multi_arena_join_mismatch),
        Err(VerifyErrorKind::StaleOrConsumedToken {
            arena_id: 1,
            provided_version: 1,
            expected_version: 2,
        })
    );

    // 12. Multi-arena AffineToken join + region typestate join across Block + BrIf.
    let brif_multi_arena_ok = "\
cap NONE
arena 0 len=4 lane=f64x8 mut=true align=64 ver=1
arena 1 len=4 lane=f64x8 mut=true align=64 ver=1
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
node $a0 = ArenaInput f64x8 tok=read(0,1) a=0 b=0
node $a1 = ArenaInput f64x8 tok=read(1,1) a=1 b=0
node $k1 = ConstF64 f64x1 const=1.0
node $k2 = ConstF64 f64x1 const=2.0
node $cond = CmpGt maskx1 a=$k2 b=$k1
node $blk = Block i64x1 a=0 b=0
node $m0 = ArenaMutate f64x8 tok=consume(0,1) a=0 b=$a0
node $m1 = ArenaMutate f64x8 tok=consume(1,1) a=1 b=$a1
node $dr = RegionDrop i64x1 a=0 b=0
node $br = BrIf maskx1 a=$cond b=0
node $end_blk = End i64x1 a=0 b=0
node $r0 = ArenaInput f64x8 tok=read(0,2) a=0 b=0
node $r1 = ArenaInput f64x8 tok=read(1,2) a=1 b=0
node $s = Add f64x8 a=$r0 b=$r1
node $out = ReduceSum f64x1 a=$s b=0
output $out
";
    assert!(verify_rls(brif_multi_arena_ok).is_ok());

    let brif_region_mismatch = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
node $z = ConstI64 i64x1 const_i64=0
node $one = ConstI64 i64x1 const_i64=1
node $cond = CmpEq maskx1 a=$z b=$one
node $blk = Block i64x1 a=0 b=0
node $br = BrIf maskx1 a=$cond b=0
node $dr = RegionDrop i64x1 a=0 b=0
node $end_blk = End i64x1 a=0 b=0
output $z
";
    assert!(matches!(
        verify_rls(brif_region_mismatch),
        Err(VerifyErrorKind::RegionTypestateMismatch { .. })
    ));

    // 13. Call precondition mismatch, Return/End postcondition mismatch, and malformed FuncRegionEffect.
    let call_pre_mismatch = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
proof 2 rule=func_sig target=0 lo=0 hi=0 witness=i64
proof 3 rule=func_sig target=1 lo=0 hi=0 witness=i64
proof func_region_effect func=0 slot=0 pre=leased post=leased
proof func_region_effect func=1 slot=0 pre=owned post=owned
node $f0 = Func i64x1 a=0 b=0
node $l = RegionLen i64x1 a=0 b=0
node $e0 = End i64x1 a=$l b=1
node $f1 = Func i64x1 a=1 b=0
node $c = Call i64x1 a=0
node $e1 = End i64x1 a=$c b=1
output $f1
";
    assert!(matches!(
        verify_rls(call_pre_mismatch),
        Err(VerifyErrorKind::RegionTypestateMismatch { .. })
    ));

    let func_post_mismatch = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
proof 2 rule=func_sig target=0 lo=0 hi=0 witness=i64
proof func_region_effect func=0 slot=0 pre=owned post=owned
node $f0 = Func i64x1 a=0 b=0
node $dr = RegionDrop i64x1 a=0 b=0
node $z = ConstI64 i64x1 const_i64=0
node $ret = Return i64x1 a=$z b=0
node $e0 = End i64x1 a=0 b=0
output $f0
";
    assert!(matches!(
        verify_rls(func_post_mismatch),
        Err(VerifyErrorKind::RegionTypestateMismatch { .. })
    ));

    let dup_effect = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
proof 2 rule=func_sig target=0 lo=0 hi=0 witness=i64
proof func_region_effect func=0 slot=0 pre=owned post=owned
proof func_region_effect func=0 slot=0 pre=owned post=owned
node $f0 = Func i64x1 a=0 b=0
node $z = ConstI64 i64x1 const_i64=0
node $e0 = End i64x1 a=$z b=1
output $f0
";
    assert!(matches!(
        verify_rls(dup_effect),
        Err(VerifyErrorKind::InvalidRegionEffect { func_id: 0, .. })
    ));

    let sentinel_plus_slot = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
proof 2 rule=func_sig target=0 lo=0 hi=0 witness=i64
proof func_region_effect func=0 slot=none
proof func_region_effect func=0 slot=0 pre=owned post=owned
node $f0 = Func i64x1 a=0 b=0
node $z = ConstI64 i64x1 const_i64=0
node $e0 = End i64x1 a=$z b=1
output $f0
";
    assert!(matches!(
        verify_rls(sentinel_plus_slot),
        Err(VerifyErrorKind::InvalidRegionEffect { func_id: 0, .. })
    ));

    // 14. Multi-arena edge cases for MRU undo tracker:
    //     (a) If with empty `then` arm and arena mutation in `else` arm -> StaleOrConsumedToken.
    let then_empty_else_mut = "\
cap NONE
arena 0 len=4 lane=f64x8 mut=true align=64 ver=1
arena 1 len=4 lane=f64x8 mut=true align=64 ver=1
node $a1 = ArenaInput f64x8 tok=read(1,1) a=1 b=0
node $k1 = ConstF64 f64x1 const=1.0
node $k2 = ConstF64 f64x1 const=2.0
node $cond = CmpGt maskx1 a=$k2 b=$k1
node $if = If i64x1 a=$cond b=0
node $el = Else i64x1 a=0 b=0
node $m1 = ArenaMutate f64x8 tok=consume(1,1) a=1 b=$a1
node $end_if = End i64x1 a=0 b=0
output $k1
";
    assert_eq!(
        verify_rls(then_empty_else_mut),
        Err(VerifyErrorKind::StaleOrConsumedToken {
            arena_id: 1,
            provided_version: 2,
            expected_version: 1,
        })
    );

    //     (b) Nested Block replaying exit_deltas into outer Block that already mutated the same
    //         and another arena, plus Else rollback + reversed-order re-mutation of 3 arenas.
    let nested_replay_3arenas_ok = "\
cap NONE
arena 0 len=4 lane=f64x8 mut=true align=64 ver=1
arena 1 len=4 lane=f64x8 mut=true align=64 ver=1
arena 2 len=4 lane=f64x8 mut=true align=64 ver=1
node $a0 = ArenaInput f64x8 tok=read(0,1) a=0 b=0
node $a1 = ArenaInput f64x8 tok=read(1,1) a=1 b=0
node $a2 = ArenaInput f64x8 tok=read(2,1) a=2 b=0
node $k1 = ConstF64 f64x1 const=1.0
node $k2 = ConstF64 f64x1 const=2.0
node $cond = CmpGt maskx1 a=$k2 b=$k1
node $if = If i64x1 a=$cond b=0
node $m0a = ArenaMutate f64x8 tok=consume(0,1) a=0 b=$a0
node $m1a = ArenaMutate f64x8 tok=consume(1,1) a=1 b=$a1
node $blk = Block i64x1 a=0 b=0
node $m0b = ArenaMutate f64x8 tok=consume(0,2) a=0 b=$a0
node $m2a = ArenaMutate f64x8 tok=consume(2,1) a=2 b=$a2
node $br = BrIf maskx1 a=$cond b=0
node $eblk = End i64x1 a=0 b=0
node $el = Else i64x1 a=0 b=0
node $m2b = ArenaMutate f64x8 tok=consume(2,1) a=2 b=$a2
node $m1b = ArenaMutate f64x8 tok=consume(1,1) a=1 b=$a1
node $m0c = ArenaMutate f64x8 tok=consume(0,1) a=0 b=$a0
node $m0d = ArenaMutate f64x8 tok=consume(0,2) a=0 b=$a0
node $eif = End i64x1 a=0 b=0
node $r0 = ArenaInput f64x8 tok=read(0,3) a=0 b=0
node $r1 = ArenaInput f64x8 tok=read(1,2) a=1 b=0
node $r2 = ArenaInput f64x8 tok=read(2,2) a=2 b=0
node $s01 = Add f64x8 a=$r0 b=$r1
node $s012 = Add f64x8 a=$s01 b=$r2
node $out = ReduceSum f64x1 a=$s012 b=0
output $out
";
    assert!(verify_rls(nested_replay_3arenas_ok).is_ok());

    // 15. Linear-time O(N) join regression checks (pinning O(1) per mutated arena across exits
    //     and per-scope undo deduplication for large mutation & branch counts).
    let build_block_mut_brif = |num_mutations: u16, num_brifs: usize| -> RulProgram {
        let sched = ScheduleHint::UNFUSED;
        let pure = TokenOp::PURE;
        let mut nodes = Vec::with_capacity(6 + usize::from(num_mutations) + num_brifs);
        nodes.push(NodeFrame {
            opcode: Opcode::ArenaInput,
            type_lane: TypeLane::F64X8,
            schedule: sched,
            proof_idx: 0,
            token_op: TokenOp::read_lease(0, 1),
            in_a: 0,
            in_b: 0,
        });
        nodes.push(NodeFrame {
            opcode: Opcode::ConstI64,
            type_lane: TypeLane::I64_SCALAR,
            schedule: sched,
            proof_idx: 0,
            token_op: pure,
            in_a: 0,
            in_b: 0,
        });
        nodes.push(NodeFrame {
            opcode: Opcode::CmpEq,
            type_lane: TypeLane::MASK_SCALAR,
            schedule: sched,
            proof_idx: 0,
            token_op: pure,
            in_a: 1,
            in_b: 1,
        });
        nodes.push(NodeFrame {
            opcode: Opcode::Block,
            type_lane: TypeLane::I64_SCALAR,
            schedule: sched,
            proof_idx: 0,
            token_op: pure,
            in_a: 0,
            in_b: 0,
        });
        for v in 1..=num_mutations {
            nodes.push(NodeFrame {
                opcode: Opcode::ArenaMutate,
                type_lane: TypeLane::F64X8,
                schedule: sched,
                proof_idx: 0,
                token_op: TokenOp::affine_consume(0, v),
                in_a: 0,
                in_b: 0,
            });
        }
        for _ in 0..num_brifs {
            nodes.push(NodeFrame {
                opcode: Opcode::BrIf,
                type_lane: TypeLane::MASK_SCALAR,
                schedule: sched,
                proof_idx: 0,
                token_op: pure,
                in_a: 2,
                in_b: 0,
            });
        }
        nodes.push(NodeFrame {
            opcode: Opcode::End,
            type_lane: TypeLane::I64_SCALAR,
            schedule: sched,
            proof_idx: 0,
            token_op: pure,
            in_a: 0,
            in_b: 0,
        });
        RulProgram {
            capability_mask: CAP_NONE,
            arenas: vec![ArenaDescriptor {
                arena_id: 0,
                element_count: 4,
                type_lane: TypeLane::F64X8,
                mutable: true,
                alignment: 64,
                initial_version: 1,
            }],
            proofs: Vec::new(),
            nodes,
            output_node: 1,
        }
    };

    // 15a. Single-arena Block: 60k mutations + 60k BrIf exits must perform exact O(1) work per exit
    //      (work == num_brifs + 1, identical to the 1-mutation baseline).
    let base_1mut = build_block_mut_brif(1, 60_000);
    let stress_60k = build_block_mut_brif(60_000, 60_000);
    let (_, base_work) = verify_program_with_join_work(&base_1mut).map_err(|e| e.to_string())?;
    let t_stress = std::time::Instant::now();
    let (_, stress_work) = verify_program_with_join_work(&stress_60k).map_err(|e| e.to_string())?;
    let stress_ms = t_stress.elapsed().as_millis();
    assert_eq!(base_work, 60_001);
    assert_eq!(
        stress_work, 60_001,
        "join work must depend on distinct mutated arenas (1), not mutation count (60k)"
    );
    assert!(
        stress_ms < 2_000,
        "60k mutations x 60k BrIf took {stress_ms} ms (expected linear time well under 2000 ms)"
    );

    // 15b. Nested first exits (`Block^D; M mutations; BrIf 0..D-1; End^D`): work == 2 * D.
    let d_nest: usize = 8_000;
    let m_nest: u16 = 8_000;
    let sched = ScheduleHint::UNFUSED;
    let pure = TokenOp::PURE;
    let mut nested_nodes = Vec::with_capacity(4 + 3 * d_nest + usize::from(m_nest));
    nested_nodes.push(NodeFrame {
        opcode: Opcode::ArenaInput,
        type_lane: TypeLane::F64X8,
        schedule: sched,
        proof_idx: 0,
        token_op: TokenOp::read_lease(0, 1),
        in_a: 0,
        in_b: 0,
    });
    nested_nodes.push(NodeFrame {
        opcode: Opcode::ConstI64,
        type_lane: TypeLane::I64_SCALAR,
        schedule: sched,
        proof_idx: 0,
        token_op: pure,
        in_a: 0,
        in_b: 0,
    });
    nested_nodes.push(NodeFrame {
        opcode: Opcode::CmpEq,
        type_lane: TypeLane::MASK_SCALAR,
        schedule: sched,
        proof_idx: 0,
        token_op: pure,
        in_a: 1,
        in_b: 1,
    });
    for _ in 0..d_nest {
        nested_nodes.push(NodeFrame {
            opcode: Opcode::Block,
            type_lane: TypeLane::I64_SCALAR,
            schedule: sched,
            proof_idx: 0,
            token_op: pure,
            in_a: 0,
            in_b: 0,
        });
    }
    for v in 1..=m_nest {
        nested_nodes.push(NodeFrame {
            opcode: Opcode::ArenaMutate,
            type_lane: TypeLane::F64X8,
            schedule: sched,
            proof_idx: 0,
            token_op: TokenOp::affine_consume(0, v),
            in_a: 0,
            in_b: 0,
        });
    }
    for depth in 0..d_nest {
        nested_nodes.push(NodeFrame {
            opcode: Opcode::BrIf,
            type_lane: TypeLane::MASK_SCALAR,
            schedule: sched,
            proof_idx: 0,
            token_op: pure,
            in_a: 2,
            in_b: u32::try_from(depth).unwrap_or(0),
        });
    }
    for _ in 0..d_nest {
        nested_nodes.push(NodeFrame {
            opcode: Opcode::End,
            type_lane: TypeLane::I64_SCALAR,
            schedule: sched,
            proof_idx: 0,
            token_op: pure,
            in_a: 0,
            in_b: 0,
        });
    }
    let nested_prog = RulProgram {
        capability_mask: CAP_NONE,
        arenas: vec![ArenaDescriptor {
            arena_id: 0,
            element_count: 4,
            type_lane: TypeLane::F64X8,
            mutable: true,
            alignment: 64,
            initial_version: 1,
        }],
        proofs: Vec::new(),
        nodes: nested_nodes,
        output_node: 1,
    };
    let (_, nested_work) =
        verify_program_with_join_work(&nested_prog).map_err(|e| e.to_string())?;
    assert_eq!(nested_work, (2 * d_nest) as u64);

    // 15c. Interleaved nested mutations (`Block_0; Mut(0); ...; Block_{D-1}; Mut(0); BrIf 0..D-1; End^D`).
    let mut inter_nodes = Vec::with_capacity(4 + 4 * d_nest);
    inter_nodes.push(NodeFrame {
        opcode: Opcode::ArenaInput,
        type_lane: TypeLane::F64X8,
        schedule: sched,
        proof_idx: 0,
        token_op: TokenOp::read_lease(0, 1),
        in_a: 0,
        in_b: 0,
    });
    inter_nodes.push(NodeFrame {
        opcode: Opcode::ConstI64,
        type_lane: TypeLane::I64_SCALAR,
        schedule: sched,
        proof_idx: 0,
        token_op: pure,
        in_a: 0,
        in_b: 0,
    });
    inter_nodes.push(NodeFrame {
        opcode: Opcode::CmpEq,
        type_lane: TypeLane::MASK_SCALAR,
        schedule: sched,
        proof_idx: 0,
        token_op: pure,
        in_a: 1,
        in_b: 1,
    });
    for i in 0..d_nest {
        inter_nodes.push(NodeFrame {
            opcode: Opcode::Block,
            type_lane: TypeLane::I64_SCALAR,
            schedule: sched,
            proof_idx: 0,
            token_op: pure,
            in_a: 0,
            in_b: 0,
        });
        let ver = u16::try_from(i + 1).unwrap_or(1);
        inter_nodes.push(NodeFrame {
            opcode: Opcode::ArenaMutate,
            type_lane: TypeLane::F64X8,
            schedule: sched,
            proof_idx: 0,
            token_op: TokenOp::affine_consume(0, ver),
            in_a: 0,
            in_b: 0,
        });
    }
    for depth in 0..d_nest {
        inter_nodes.push(NodeFrame {
            opcode: Opcode::BrIf,
            type_lane: TypeLane::MASK_SCALAR,
            schedule: sched,
            proof_idx: 0,
            token_op: pure,
            in_a: 2,
            in_b: u32::try_from(depth).unwrap_or(0),
        });
    }
    for _ in 0..d_nest {
        inter_nodes.push(NodeFrame {
            opcode: Opcode::End,
            type_lane: TypeLane::I64_SCALAR,
            schedule: sched,
            proof_idx: 0,
            token_op: pure,
            in_a: 0,
            in_b: 0,
        });
    }
    let inter_prog = RulProgram {
        capability_mask: CAP_NONE,
        arenas: vec![ArenaDescriptor {
            arena_id: 0,
            element_count: 4,
            type_lane: TypeLane::F64X8,
            mutable: true,
            alignment: 64,
            initial_version: 1,
        }],
        proofs: Vec::new(),
        nodes: inter_nodes,
        output_node: 1,
    };
    let (_, inter_work) = verify_program_with_join_work(&inter_prog).map_err(|e| e.to_string())?;
    assert_eq!(inter_work, (2 * d_nest) as u64);

    // 15d. Shape B (K > 1): `Block; K x (ArenaInput k, ArenaMutate k); M x BrIf(0); End`
    //      at K = M = 16_000 must stay strictly O(N + A) in both work and wall-clock time.
    let k_b: usize = 16_000;
    let m_b: usize = 16_000;
    let arenas_b: Vec<ArenaDescriptor> = (0..k_b)
        .map(|k| ArenaDescriptor {
            arena_id: u32::try_from(k).unwrap_or(0),
            element_count: 4,
            type_lane: TypeLane::F64X8,
            mutable: true,
            alignment: 64,
            initial_version: 1,
        })
        .collect();
    let mut nodes_b = Vec::with_capacity(4 + 2 * k_b + m_b);
    nodes_b.push(NodeFrame {
        opcode: Opcode::ConstI64,
        type_lane: TypeLane::I64_SCALAR,
        schedule: sched,
        proof_idx: 0,
        token_op: pure,
        in_a: 0,
        in_b: 0,
    });
    nodes_b.push(NodeFrame {
        opcode: Opcode::CmpEq,
        type_lane: TypeLane::MASK_SCALAR,
        schedule: sched,
        proof_idx: 0,
        token_op: pure,
        in_a: 0,
        in_b: 0,
    });
    nodes_b.push(NodeFrame {
        opcode: Opcode::Block,
        type_lane: TypeLane::I64_SCALAR,
        schedule: sched,
        proof_idx: 0,
        token_op: pure,
        in_a: 0,
        in_b: 0,
    });
    for k in 0..k_b {
        let a_id = u16::try_from(k).unwrap_or(0);
        let a_u32 = u32::from(a_id);
        let inp_idx = u32::try_from(nodes_b.len()).unwrap_or(0);
        nodes_b.push(NodeFrame {
            opcode: Opcode::ArenaInput,
            type_lane: TypeLane::F64X8,
            schedule: sched,
            proof_idx: 0,
            token_op: TokenOp::read_lease(a_id, 1),
            in_a: a_u32,
            in_b: 0,
        });
        nodes_b.push(NodeFrame {
            opcode: Opcode::ArenaMutate,
            type_lane: TypeLane::F64X8,
            schedule: sched,
            proof_idx: 0,
            token_op: TokenOp::affine_consume(a_id, 1),
            in_a: a_u32,
            in_b: inp_idx,
        });
    }
    for _ in 0..m_b {
        nodes_b.push(NodeFrame {
            opcode: Opcode::BrIf,
            type_lane: TypeLane::MASK_SCALAR,
            schedule: sched,
            proof_idx: 0,
            token_op: pure,
            in_a: 1,
            in_b: 0,
        });
    }
    nodes_b.push(NodeFrame {
        opcode: Opcode::End,
        type_lane: TypeLane::I64_SCALAR,
        schedule: sched,
        proof_idx: 0,
        token_op: pure,
        in_a: 0,
        in_b: 0,
    });
    let prog_b = RulProgram {
        capability_mask: CAP_NONE,
        arenas: arenas_b,
        proofs: Vec::new(),
        nodes: nodes_b,
        output_node: 0,
    };
    let n_plus_a_b = (prog_b.nodes.len() + prog_b.arenas.len()) as u64;
    let t_b = std::time::Instant::now();
    let (_, work_b) = verify_program_with_join_work(&prog_b).map_err(|e| e.to_string())?;
    let ms_b = t_b.elapsed().as_millis();
    assert!(
        work_b <= 16 * n_plus_a_b,
        "Shape B (K={k_b}, M={m_b}) join work {work_b} exceeded linear bound 16*(N+A)={}",
        16 * n_plus_a_b
    );
    assert!(
        ms_b < 2_000,
        "Shape B (K={k_b}, M={m_b}) took {ms_b} ms (expected < 2000 ms in linear time)"
    );

    // 15e. Shape C (K > 1): `Block^D; K x (ArenaInput k, ArenaMutate k); BrIf 0..D-1; End^D`
    //      at K = D = 16_000 must stay strictly O(N + A) in time and memory.
    let k_c: usize = 16_000;
    let d_c: usize = 16_000;
    let arenas_c: Vec<ArenaDescriptor> = (0..k_c)
        .map(|k| ArenaDescriptor {
            arena_id: u32::try_from(k).unwrap_or(0),
            element_count: 4,
            type_lane: TypeLane::F64X8,
            mutable: true,
            alignment: 64,
            initial_version: 1,
        })
        .collect();
    let mut nodes_c = Vec::with_capacity(2 + 3 * d_c + 2 * k_c);
    nodes_c.push(NodeFrame {
        opcode: Opcode::ConstI64,
        type_lane: TypeLane::I64_SCALAR,
        schedule: sched,
        proof_idx: 0,
        token_op: pure,
        in_a: 0,
        in_b: 0,
    });
    nodes_c.push(NodeFrame {
        opcode: Opcode::CmpEq,
        type_lane: TypeLane::MASK_SCALAR,
        schedule: sched,
        proof_idx: 0,
        token_op: pure,
        in_a: 0,
        in_b: 0,
    });
    for _ in 0..d_c {
        nodes_c.push(NodeFrame {
            opcode: Opcode::Block,
            type_lane: TypeLane::I64_SCALAR,
            schedule: sched,
            proof_idx: 0,
            token_op: pure,
            in_a: 0,
            in_b: 0,
        });
    }
    for k in 0..k_c {
        let a_id = u16::try_from(k).unwrap_or(0);
        let a_u32 = u32::from(a_id);
        let inp_idx = u32::try_from(nodes_c.len()).unwrap_or(0);
        nodes_c.push(NodeFrame {
            opcode: Opcode::ArenaInput,
            type_lane: TypeLane::F64X8,
            schedule: sched,
            proof_idx: 0,
            token_op: TokenOp::read_lease(a_id, 1),
            in_a: a_u32,
            in_b: 0,
        });
        nodes_c.push(NodeFrame {
            opcode: Opcode::ArenaMutate,
            type_lane: TypeLane::F64X8,
            schedule: sched,
            proof_idx: 0,
            token_op: TokenOp::affine_consume(a_id, 1),
            in_a: a_u32,
            in_b: inp_idx,
        });
    }
    for depth in 0..d_c {
        nodes_c.push(NodeFrame {
            opcode: Opcode::BrIf,
            type_lane: TypeLane::MASK_SCALAR,
            schedule: sched,
            proof_idx: 0,
            token_op: pure,
            in_a: 1,
            in_b: u32::try_from(depth).unwrap_or(0),
        });
    }
    for _ in 0..d_c {
        nodes_c.push(NodeFrame {
            opcode: Opcode::End,
            type_lane: TypeLane::I64_SCALAR,
            schedule: sched,
            proof_idx: 0,
            token_op: pure,
            in_a: 0,
            in_b: 0,
        });
    }
    let prog_c = RulProgram {
        capability_mask: CAP_NONE,
        arenas: arenas_c,
        proofs: Vec::new(),
        nodes: nodes_c,
        output_node: 0,
    };
    let n_plus_a_c = (prog_c.nodes.len() + prog_c.arenas.len()) as u64;
    let t_c = std::time::Instant::now();
    let (_, work_c) = verify_program_with_join_work(&prog_c).map_err(|e| e.to_string())?;
    let ms_c = t_c.elapsed().as_millis();
    assert!(
        work_c <= 16 * n_plus_a_c,
        "Shape C (K={k_c}, D={d_c}) join work {work_c} exceeded linear bound 16*(N+A)={}",
        16 * n_plus_a_c
    );
    assert!(
        ms_c < 2_000,
        "Shape C (K={k_c}, D={d_c}) took {ms_c} ms (expected < 2000 ms in linear time)"
    );

    // 15f. Shape F (order-independent If/Else multi-arena join + nested outer blocks, K = D = 4_000):
    //      `then` mutates 0..K-1 ascending, `else` mutates K-1..0 descending, both branch out to
    //      all D outer blocks; also verify dropping 1 mutation in `else` pinpoints the exact arena.
    let build_shape_f = |k_f: usize, d_f: usize, skip_else_arena: Option<usize>| -> RulProgram {
        let arenas_f: Vec<ArenaDescriptor> = (0..k_f)
            .map(|k| ArenaDescriptor {
                arena_id: u32::try_from(k).unwrap_or(0),
                element_count: 4,
                type_lane: TypeLane::F64X8,
                mutable: true,
                alignment: 64,
                initial_version: 1,
            })
            .collect();
        let mut nodes_f = Vec::with_capacity(6 + 4 * d_f + 4 * k_f);
        nodes_f.push(NodeFrame {
            opcode: Opcode::ConstI64,
            type_lane: TypeLane::I64_SCALAR,
            schedule: sched,
            proof_idx: 0,
            token_op: pure,
            in_a: 0,
            in_b: 0,
        });
        nodes_f.push(NodeFrame {
            opcode: Opcode::CmpEq,
            type_lane: TypeLane::MASK_SCALAR,
            schedule: sched,
            proof_idx: 0,
            token_op: pure,
            in_a: 0,
            in_b: 0,
        });
        for _ in 0..d_f {
            nodes_f.push(NodeFrame {
                opcode: Opcode::Block,
                type_lane: TypeLane::I64_SCALAR,
                schedule: sched,
                proof_idx: 0,
                token_op: pure,
                in_a: 0,
                in_b: 0,
            });
        }
        nodes_f.push(NodeFrame {
            opcode: Opcode::If,
            type_lane: TypeLane::I64_SCALAR,
            schedule: sched,
            proof_idx: 0,
            token_op: pure,
            in_a: 1,
            in_b: 0,
        });
        for k in 0..k_f {
            let a_id = u16::try_from(k).unwrap_or(0);
            let a_u32 = u32::from(a_id);
            let inp_idx = u32::try_from(nodes_f.len()).unwrap_or(0);
            nodes_f.push(NodeFrame {
                opcode: Opcode::ArenaInput,
                type_lane: TypeLane::F64X8,
                schedule: sched,
                proof_idx: 0,
                token_op: TokenOp::read_lease(a_id, 1),
                in_a: a_u32,
                in_b: 0,
            });
            nodes_f.push(NodeFrame {
                opcode: Opcode::ArenaMutate,
                type_lane: TypeLane::F64X8,
                schedule: sched,
                proof_idx: 0,
                token_op: TokenOp::affine_consume(a_id, 1),
                in_a: a_u32,
                in_b: inp_idx,
            });
        }
        for depth in 1..=d_f {
            nodes_f.push(NodeFrame {
                opcode: Opcode::BrIf,
                type_lane: TypeLane::MASK_SCALAR,
                schedule: sched,
                proof_idx: 0,
                token_op: pure,
                in_a: 1,
                in_b: u32::try_from(depth).unwrap_or(0),
            });
        }
        nodes_f.push(NodeFrame {
            opcode: Opcode::Else,
            type_lane: TypeLane::I64_SCALAR,
            schedule: sched,
            proof_idx: 0,
            token_op: pure,
            in_a: 0,
            in_b: 0,
        });
        for k in (0..k_f).rev() {
            if skip_else_arena == Some(k) {
                continue;
            }
            let a_id = u16::try_from(k).unwrap_or(0);
            let a_u32 = u32::from(a_id);
            let inp_idx = u32::try_from(nodes_f.len()).unwrap_or(0);
            nodes_f.push(NodeFrame {
                opcode: Opcode::ArenaInput,
                type_lane: TypeLane::F64X8,
                schedule: sched,
                proof_idx: 0,
                token_op: TokenOp::read_lease(a_id, 1),
                in_a: a_u32,
                in_b: 0,
            });
            nodes_f.push(NodeFrame {
                opcode: Opcode::ArenaMutate,
                type_lane: TypeLane::F64X8,
                schedule: sched,
                proof_idx: 0,
                token_op: TokenOp::affine_consume(a_id, 1),
                in_a: a_u32,
                in_b: inp_idx,
            });
        }
        for depth in 1..=d_f {
            nodes_f.push(NodeFrame {
                opcode: Opcode::BrIf,
                type_lane: TypeLane::MASK_SCALAR,
                schedule: sched,
                proof_idx: 0,
                token_op: pure,
                in_a: 1,
                in_b: u32::try_from(depth).unwrap_or(0),
            });
        }
        nodes_f.push(NodeFrame {
            opcode: Opcode::End,
            type_lane: TypeLane::I64_SCALAR,
            schedule: sched,
            proof_idx: 0,
            token_op: pure,
            in_a: 0,
            in_b: 0,
        });
        for _ in 0..d_f {
            nodes_f.push(NodeFrame {
                opcode: Opcode::End,
                type_lane: TypeLane::I64_SCALAR,
                schedule: sched,
                proof_idx: 0,
                token_op: pure,
                in_a: 0,
                in_b: 0,
            });
        }
        RulProgram {
            capability_mask: CAP_NONE,
            arenas: arenas_f,
            proofs: Vec::new(),
            nodes: nodes_f,
            output_node: 0,
        }
    };
    let prog_f_ok = build_shape_f(4_000, 4_000, None);
    let n_plus_a_f = (prog_f_ok.nodes.len() + prog_f_ok.arenas.len()) as u64;
    let t_f = std::time::Instant::now();
    let (_, work_f) = verify_program_with_join_work(&prog_f_ok).map_err(|e| e.to_string())?;
    let ms_f = t_f.elapsed().as_millis();
    assert!(
        work_f <= 16 * n_plus_a_f,
        "Shape F (K=4000, D=4000) join work {work_f} exceeded linear bound 16*(N+A)={}",
        16 * n_plus_a_f
    );
    assert!(
        ms_f < 2_000,
        "Shape F (K=4000, D=4000) took {ms_f} ms (expected < 2000 ms in linear time)"
    );
    let prog_f_bad = build_shape_f(4_000, 4_000, Some(2_718));
    let err_f = verify_program(&prog_f_bad)
        .err()
        .ok_or_else(|| "expected mismatch when else skips arena 2718".to_string())?;
    assert_eq!(
        err_f.kind,
        VerifyErrorKind::StaleOrConsumedToken {
            arena_id: 2_718,
            provided_version: 1,
            expected_version: 2,
        }
    );

    Ok(())
}

#[test]
fn test_v06_ownership_tier0_aot_and_rls_roundtrip() -> Result<(), String> {
    // Program exercising RegionLease, RegionEndLease, RegionMove, RegionDrop, and FuncRegionEffect
    // (including `slot=none` pure helper, leased-region reader, and ownership-transferring callee)
    // across a bounded loop, branches, and nested calls.
    let rls = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=64 witness=i64:plain
proof 2 rule=region_decl target=1 lo=0 hi=64 witness=i64:plain
proof 3 rule=loop_bound target=0 lo=0 hi=32 witness=0
proof 4 rule=func_sig target=0 lo=2 hi=i64,i64 witness=i64
proof 5 rule=func_sig target=1 lo=1 hi=i64 witness=i64
proof 6 rule=func_sig target=2 lo=0 hi=0 witness=i64
proof 7 rule=func_region_effect func=0 slot=none pre=owned post=owned
proof 8 rule=func_region_effect func=1 slot=0 pre=leased post=leased
proof 9 rule=func_region_effect func=2 slot=0 pre=owned post=moved
proof 10 rule=func_region_effect func=2 slot=1 pre=owned post=owned
node $f0 = Func i64x1 a=0 b=0
node $a = LocalGet i64x1 a=0 b=0
node $b = LocalGet i64x1 a=1 b=0
node $s = Add i64x1 a=$a b=$b
node $e0 = End i64x1 a=$s b=1
node $f1 = Func i64x1 a=1 b=0
node $idx = LocalGet i64x1 a=0 b=0
node $val = RegionLoad i64x1 a=0 b=$idx
node $c10 = ConstI64 i64x1 const_i64=10
node $adj = Call i64x1 a=0 b=$val c=$c10
node $e1 = End i64x1 a=$adj b=1
node $f2 = Func i64x1 a=2 b=0
node $c0 = ConstI64 i64x1 const_i64=0
node $c1 = ConstI64 i64x1 const_i64=1
node $c8 = ConstI64 i64x1 const_i64=8
node $rn0 = RegionNew i64x1 a=0 b=$c8
node $si = LocalSet i64x1 a=0 b=$c0
node $sacc = LocalSet i64x1 a=1 b=$c0
node $blk = Block i64x1 a=0 b=0
node $lp = Loop i64x1 proof=3 a=0 b=0
node $i = LocalGet i64x1 a=0 b=0
node $done = CmpEq maskx1 a=$i b=$c8
node $brk = BrIf maskx1 a=$done b=1
node $sq = Mul i64x1 a=$i b=$i
node $st = RegionStore i64x1 a=0 b=$i c=$sq
node $ls = RegionLease i64x1 a=0 b=0
node $got = Call i64x1 a=1 b=$i
node $el = RegionEndLease i64x1 a=0 b=0
node $acc = LocalGet i64x1 a=1 b=0
node $nacc = Add i64x1 a=$acc b=$got
node $uacc = LocalSet i64x1 a=1 b=$nacc
node $ni = Add i64x1 a=$i b=$c1
node $ui = LocalSet i64x1 a=0 b=$ni
node $back = Br i64x1 a=0 b=0
node $elp = End i64x1 a=0 b=0
node $eblk = End i64x1 a=0 b=0
node $dr1 = RegionDrop i64x1 a=1 b=0
node $mv = RegionMove i64x1 a=0 b=1
node $c7 = ConstI64 i64x1 const_i64=7
node $last = RegionLoad i64x1 a=1 b=$c7
node $len1 = RegionLen i64x1 a=1 b=0
node $accf = LocalGet i64x1 a=1 b=0
node $t1 = Add i64x1 a=$accf b=$last
node $t2 = Add i64x1 a=$t1 b=$len1
node $e2 = End i64x1 a=$t2 b=1
output $f2
";

    let prog = assemble_rls(rls).map_err(|e| e.to_string())?;
    let summary = verify_program(&prog).map_err(|e| e.to_string())?;
    assert!(summary.static_cost.is_some());

    // 1. Lossless binary + `.rls` round-trip check (including `slot=none` and all 4 new opcodes).
    let bin = prog.to_rl_bytes();
    let decoded = RulProgram::from_rl_bytes(&bin).map_err(|e| e.to_string())?;
    assert_eq!(prog, decoded);
    let rls_out = program_to_rls(&decoded);
    assert!(rls_out.contains("rule=func_region_effect"));
    assert!(rls_out.contains("slot=none"));
    assert!(rls_out.contains("RegionLease"));
    assert!(rls_out.contains("RegionEndLease"));
    assert!(rls_out.contains("RegionMove"));
    assert!(rls_out.contains("RegionDrop"));
    let reassembled = assemble_rls(&rls_out).map_err(|e| e.to_string())?;
    assert_eq!(prog, reassembled);
    assert!(prog
        .proofs
        .iter()
        .any(|p| p.rule_kind == ProofRuleKind::FuncRegionEffect));

    let disasm = disassemble_rl(&prog);
    assert!(disasm.contains("FuncRegionEffect"));
    assert!(disasm.contains("RegionMove"));

    // 2. Columnar tiers (Tier 1 / Tier 2) must fail closed.
    assert_eq!(
        execute_tier1_stencil(&prog, &[]).err(),
        Some(ExecError::UnsupportedInColumnarTier)
    );
    assert_eq!(
        execute_tier2_fused(&prog, &[]).err(),
        Some(ExecError::UnsupportedInColumnarTier)
    );

    // 3. Expected mathematical result:
    //    sum_{i=0..8} (i*i + 10) = (0+1+4+9+16+25+36+49) + 80 = 140 + 80 = 220
    //    plus last element (7*7 = 49) plus len(slot 1) = 8 => 277.
    let expected: i64 = 277;

    let t0_report = execute_tier0_interpreter(&prog, &[]).map_err(|e| e.to_string())?;
    assert_eq!(
        t0_report.output,
        RuntimeValue::TypedScalar(TypedScalar::i64(expected))
    );

    let exe_path =
        std::env::temp_dir().join(format!("rul_v06_ownership_aot_test_{}", std::process::id()));
    let _ = std::fs::remove_file(&exe_path);
    let _report = compile_rl_to_executable(&prog, &exe_path).map_err(|e| e.to_string())?;
    let aot_f64 = run_compiled_scalar_executable(&exe_path).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&exe_path);
    let aot_i64 = i64::from_ne_bytes(aot_f64.to_bits().to_ne_bytes());
    assert_eq!(aot_i64, expected, "Tier 0 == ELF64 AOT mismatch");

    // 4. Verify zero runtime instructions for RegionLease/RegionEndLease in ELF64 AOT,
    //    and verify RegionMove + RegionNew re-allocation has zero aliasing between src and dst.
    let base_no_lease = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
node $c0 = ConstI64 i64x1 const_i64=0
node $c2 = ConstI64 i64x1 const_i64=2
node $v = ConstI64 i64x1 const_i64=123
node $rn = RegionNew i64x1 a=0 b=$c2
node $st = RegionStore i64x1 a=0 b=$c0 c=$v
node $ld = RegionLoad i64x1 a=0 b=$c0
output $ld
";
    let with_lease = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
node $c0 = ConstI64 i64x1 const_i64=0
node $c2 = ConstI64 i64x1 const_i64=2
node $v = ConstI64 i64x1 const_i64=123
node $rn = RegionNew i64x1 a=0 b=$c2
node $st = RegionStore i64x1 a=0 b=$c0 c=$v
node $ls = RegionLease i64x1 a=0 b=0
node $ld = RegionLoad i64x1 a=0 b=$c0
node $el = RegionEndLease i64x1 a=0 b=0
output $ld
";
    let p_base = assemble_rls(base_no_lease).map_err(|e| e.to_string())?;
    let p_lease = assemble_rls(with_lease).map_err(|e| e.to_string())?;
    let (elf_base, _) = crate::aot::compile_to_elf64_bytes(&p_base).map_err(|e| e.to_string())?;
    let (elf_lease, _) = crate::aot::compile_to_elf64_bytes(&p_lease).map_err(|e| e.to_string())?;
    assert_eq!(
        elf_base, elf_lease,
        "RegionLease/RegionEndLease must emit zero runtime instructions"
    );

    // Non-aliasing check: move slot 0 -> slot 1, then re-allocate slot 0 with different data.
    let move_reinit_rls = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=16 witness=i64:plain
proof 2 rule=region_decl target=1 lo=0 hi=16 witness=i64:plain
node $c0 = ConstI64 i64x1 const_i64=0
node $c1 = ConstI64 i64x1 const_i64=1
node $c4 = ConstI64 i64x1 const_i64=4
node $v10 = ConstI64 i64x1 const_i64=1000
node $v99 = ConstI64 i64x1 const_i64=99
node $rn0 = RegionNew i64x1 a=0 b=$c4
node $st0 = RegionStore i64x1 a=0 b=$c1 c=$v10
node $dr1 = RegionDrop i64x1 a=1 b=0
node $mv = RegionMove i64x1 a=0 b=1
node $rn0b = RegionNew i64x1 a=0 b=$c4
node $st0b = RegionStore i64x1 a=0 b=$c1 c=$v99
node $a = RegionLoad i64x1 a=1 b=$c1
node $b = RegionLoad i64x1 a=0 b=$c1
node $out = Add i64x1 a=$a b=$b
output $out
";
    let p_mv = assemble_rls(move_reinit_rls).map_err(|e| e.to_string())?;
    let t0_mv = execute_tier0_interpreter(&p_mv, &[]).map_err(|e| e.to_string())?;
    assert_eq!(
        t0_mv.output,
        RuntimeValue::TypedScalar(TypedScalar::i64(1099))
    );
    let exe_mv = std::env::temp_dir().join(format!(
        "rul_v06_ownership_move_reinit_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&exe_mv);
    let _rep_mv = compile_rl_to_executable(&p_mv, &exe_mv).map_err(|e| e.to_string())?;
    let aot_mv_f64 = run_compiled_scalar_executable(&exe_mv).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&exe_mv);
    let aot_mv_i64 = i64::from_ne_bytes(aot_mv_f64.to_bits().to_ne_bytes());
    assert_eq!(aot_mv_i64, 1099);

    // 5. Tier 0 == ELF64 AOT parity for RegionLease/RegionEndLease/RegionMove/RegionDrop across If/Else arms.
    for (take_then, expected_branch) in [(1i64, 507i64), (0i64, 809i64)] {
        let if_else_rls = format!(
            "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=8 witness=i64:plain
proof 2 rule=region_decl target=1 lo=0 hi=8 witness=i64:plain
node $c0 = ConstI64 i64x1 const_i64=0
node $c1 = ConstI64 i64x1 const_i64=1
node $c2 = ConstI64 i64x1 const_i64=2
node $flag = ConstI64 i64x1 const_i64={take_then}
node $v_then = ConstI64 i64x1 const_i64=500
node $v_else = ConstI64 i64x1 const_i64=800
node $a_then = ConstI64 i64x1 const_i64=7
node $a_else = ConstI64 i64x1 const_i64=9
node $cond = CmpEq maskx1 a=$flag b=$c1
node $rn0 = RegionNew i64x1 a=0 b=$c2
node $dr1 = RegionDrop i64x1 a=1 b=0
node $if = If i64x1 a=$cond b=0
node $st_t = RegionStore i64x1 a=0 b=$c0 c=$v_then
node $ls_t = RegionLease i64x1 a=0 b=0
node $ld_t = RegionLoad i64x1 a=0 b=$c0
node $el_t = RegionEndLease i64x1 a=0 b=0
node $mv_t = RegionMove i64x1 a=0 b=1
node $st_t2 = RegionStore i64x1 a=1 b=$c1 c=$a_then
node $el = Else i64x1 a=0 b=0
node $st_e = RegionStore i64x1 a=0 b=$c0 c=$v_else
node $mv_e = RegionMove i64x1 a=0 b=1
node $st_e2 = RegionStore i64x1 a=1 b=$c1 c=$a_else
node $eif = End i64x1 a=0 b=0
node $x0 = RegionLoad i64x1 a=1 b=$c0
node $x1 = RegionLoad i64x1 a=1 b=$c1
node $sum = Add i64x1 a=$x0 b=$x1
output $sum
"
        );
        let p_br = assemble_rls(&if_else_rls).map_err(|e| e.to_string())?;
        let t0_br = execute_tier0_interpreter(&p_br, &[]).map_err(|e| e.to_string())?;
        assert_eq!(
            t0_br.output,
            RuntimeValue::TypedScalar(TypedScalar::i64(expected_branch))
        );
        let exe_br = std::env::temp_dir().join(format!(
            "rul_v06_ownership_ifelse_{}_{}",
            take_then,
            std::process::id()
        ));
        let _ = std::fs::remove_file(&exe_br);
        let _rep_br = compile_rl_to_executable(&p_br, &exe_br).map_err(|e| e.to_string())?;
        let aot_br_f64 = run_compiled_scalar_executable(&exe_br).map_err(|e| e.to_string())?;
        let _ = std::fs::remove_file(&exe_br);
        let aot_br_i64 = i64::from_ne_bytes(aot_br_f64.to_bits().to_ne_bytes());
        assert_eq!(
            aot_br_i64, expected_branch,
            "Tier 0 == ELF64 AOT mismatch for If/Else branch take_then={take_then}"
        );
    }

    Ok(())
}

#[test]
fn test_v06_static_cost_summary() -> Result<(), String> {
    // 1. Columnar DAG (`len = 1024`): columnar ops add `1024` (NOT `1`), `ReduceSum` adds `1024`,
    //    `max_region_bytes == 1024 * 8`.
    let columnar_rls = "\
cap NONE
arena 0 len=1024 lane=f64x8 mut=false align=64 ver=1
node $a = ArenaInput f64x8 tok=read(0,1) a=0 b=0
node $k = ConstF64 f64x1 const=2.0
node $p = Mul f64x8 a=$a b=$k
node $s = ReduceSum f64x1 a=$p b=0
output $s
";
    let col_prog = assemble_rls(columnar_rls).map_err(|e| e.to_string())?;
    let col_sum = verify_program(&col_prog).map_err(|e| e.to_string())?;
    assert_eq!(col_sum.output_shape, ValueShape::Scalar);
    assert_eq!(
        col_sum.static_cost,
        Some(StaticCostSummary {
            worst_case_steps: 1024 + 1 + 1024 + 1024,
            max_call_depth: 1,
            max_stack_units: 0,
            max_stack_bytes: 0,
            max_region_bytes: 1024 * 8,
        })
    );

    // 2. Nested If/Else (asymmetric arms: then=4 ops, else=1 op) + nested bounded Loops
    //    (outer bound_hi=3 -> 4 iters; inner bound_hi=4 -> 5 iters).
    let control_rls = "\
cap NONE
proof 1 rule=loop_bound target=0 lo=0 hi=3 witness=0
proof 2 rule=loop_bound target=0 lo=0 hi=4 witness=0
node $c0 = ConstI64 i64x1 const_i64=0
node $c1 = ConstI64 i64x1 const_i64=1
node $cond = CmpEq maskx1 a=$c0 b=$c1
node $if = If i64x1 a=$cond b=0
node $t1 = Add i64x1 a=$c0 b=$c1
node $t2 = Add i64x1 a=$t1 b=$c1
node $t3 = Add i64x1 a=$t2 b=$c1
node $t4 = Add i64x1 a=$t3 b=$c1
node $el = Else i64x1 a=0 b=0
node $e1 = Sub i64x1 a=$c1 b=$c0
node $end_if = End i64x1 a=0 b=0
node $lp_out = Loop i64x1 proof=1 a=0 b=0
node $lp_in = Loop i64x1 proof=2 a=0 b=0
node $x = Add i64x1 a=$c0 b=$c1
node $end_in = End i64x1 a=0 b=0
node $end_out = End i64x1 a=0 b=0
output $c1
";
    let ctrl_prog = assemble_rls(control_rls).map_err(|e| e.to_string())?;
    let ctrl_sum = verify_program(&ctrl_prog).map_err(|e| e.to_string())?;
    // Before If: 3 steps ($c0, $c1, $cond) + 1 ($if) + max(4, 1) = 8.
    // Outer loop: + 1 ($lp_out) + 4 * (1 ($lp_in) + 5 * 1 ($x)) = 1 + 4 * 6 = 25.
    // Total: 8 + 25 = 33.
    assert_eq!(
        ctrl_sum.static_cost,
        Some(StaticCostSummary {
            worst_case_steps: 33,
            max_call_depth: 1,
            max_stack_units: 0,
            max_stack_bytes: 0,
            max_region_bytes: 0,
        })
    );

    // 3. Multi-function acyclic call DAG + page-rounded region bytes + descriptor/guard overhead.
    //    Region 0: u8, max_len=1 -> align_up_4096(1) = 4096
    //    Region 1: i64, max_len=513 -> align_up_4096(4104) = 8192
    //    Region 2: u8, max_len=0 -> align_up_4096(0) = 0
    //    r15 descriptor page: + 4096
    //    Total max_region_bytes = 4096 + 8192 + 0 + 4096 = 16384.
    let dag_rls = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=1 witness=u8:plain
proof 2 rule=region_decl target=1 lo=0 hi=513 witness=i64:plain
proof 3 rule=region_decl target=2 lo=0 hi=0 witness=u8:plain
proof 4 rule=func_sig target=0 lo=1 hi=i64 witness=i64
proof 5 rule=func_sig target=1 lo=1 hi=i64 witness=i64
proof 6 rule=func_sig target=2 lo=0 hi=0 witness=i64
node $f0 = Func i64x1 a=0 b=0
node $p0 = LocalGet i64x1 a=0 b=0
node $k1 = ConstI64 i64x1 const_i64=1
node $r0 = Add i64x1 a=$p0 b=$k1
node $e0 = End i64x1 a=$r0 b=1
node $f1 = Func i64x1 a=1 b=0
node $p1 = LocalGet i64x1 a=0 b=0
node $c0 = Call i64x1 a=0 b=$p1
node $r1 = Add i64x1 a=$c0 b=$p1
node $e1 = End i64x1 a=$r1 b=1
node $f2 = Func i64x1 a=2 b=0
node $v = ConstI64 i64x1 const_i64=5
node $a1 = Call i64x1 a=1 b=$v
node $a0 = Call i64x1 a=0 b=$a1
node $e2 = End i64x1 a=$a0 b=1
output $f2
";
    let dag_prog = assemble_rls(dag_rls).map_err(|e| e.to_string())?;
    let dag_sum = verify_program(&dag_prog).map_err(|e| e.to_string())?;
    // f0: 5 nodes, steps = 3 ($p0, $k1, $r0), depth = 1, stack = func_frame_units(5) = 77
    // f1: 5 nodes, steps = 1 ($p1) + (1 + 3) ($c0) + 1 ($r1) = 6, depth = 2, stack = 77 + 77 = 154
    // f2: 5 nodes, steps = 1 ($v) + (1 + 6) ($a1) + (1 + 3) ($a0) = 12, depth = 3, stack = 77 + 154 = 231
    let expected_stack_units = func_frame_units(5) + func_frame_units(5) + func_frame_units(5);
    assert_eq!(expected_stack_units, 231);
    assert_eq!(
        dag_sum.static_cost,
        Some(StaticCostSummary {
            worst_case_steps: 12,
            max_call_depth: 3,
            max_stack_units: expected_stack_units,
            max_stack_bytes: u64::from(expected_stack_units) * 8 + 4096,
            max_region_bytes: 4096 + 8192 + 4096,
        })
    );

    // 4. CAP_DIVERGE returns static_cost == None.
    let diverge_rls = "\
cap DIVERGE
node $c = ConstI64 i64x1 const_i64=42
output $c
";
    let div_prog = assemble_rls(diverge_rls).map_err(|e| e.to_string())?;
    let div_sum = verify_program(&div_prog).map_err(|e| e.to_string())?;
    assert_eq!(div_sum.static_cost, None);

    Ok(())
}
