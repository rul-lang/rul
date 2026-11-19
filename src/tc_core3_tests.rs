//! Milestone 1c (`v0-5-tc-core-3`) test suite:
//! - AC-1: Function table, `FuncSig`, arity `0..=4`, `CAP_DIVERGE`-gated recursion, and `MAX_CALL_DEPTH = 4096` guard.
//! - AC-2: Brainf*ck interpreter Turing-completeness proof (`Tier 0 == ELF64 AOT`, microsecond compile).
//! - AC-3: Exit programs suite (`fib(20)`, in-place `quicksort`, prime `sieve(10_000)`, `sha256`) with `Tier 0 == AOT` and lossless `.rls` round-trip.
//! - AC-5: Function table >255 functions & >65,536 total nodes + all layout/verification rejection cases.
//! - AC-6: Host-independent stack budget (`MAX_STACK_UNITS = 1 << 20`) verified under `ulimit -s 64`.

use std::fmt::Write as _;
use std::process::Command;

use crate::aot::{
    compile_rl_to_executable, run_compiled_structured_executable, AotError, AOT_EXIT_STACK_OVERFLOW,
};
use crate::bytecode::{
    func_frame_units, ElemType, NodeFrame, Opcode, ProofEntry, RulProgram, ScheduleHint, TokenOp,
    TypeLane, CALL_ARG_SENTINEL, CAP_DIVERGE, CAP_NONE, MAX_CALL_DEPTH, MAX_FUNC_NODES,
    MAX_STACK_UNITS,
};
use crate::engine::{
    execute_tier0_interpreter, execute_tier0_interpreter_with_input, ExecError, RuntimeValue,
    TypedScalar,
};
use crate::transpiler::{assemble_rls, program_to_rls};
use crate::verifier::{verify_program, VerifyErrorKind};

/// Lightweight `.rls` emitter for constructing readable, modular multi-function Rul programs in tests.
struct RlsBuilder {
    cap: String,
    proofs: Vec<String>,
    body: String,
    next_id: usize,
    last_func_label: Option<String>,
}

impl RlsBuilder {
    fn new(cap: &str) -> Self {
        Self {
            cap: cap.to_owned(),
            proofs: Vec::new(),
            body: String::new(),
            next_id: 0,
            last_func_label: None,
        }
    }

    fn fresh(&mut self, prefix: &str) -> String {
        let id = self.next_id;
        self.next_id += 1;
        format!("${prefix}_{id}")
    }

    fn add_region(&mut self, slot: u16, min_len: u32, max_len: u32, elem: &str, role: &str) {
        let idx = self.proofs.len() + 1;
        self.proofs.push(format!(
            "proof {idx} rule=region_decl target={slot} lo={min_len} hi={max_len} witness={elem}:{role}"
        ));
    }

    fn add_func_sig(&mut self, func_id: u16, params: &[&str], ret: &str) {
        let idx = self.proofs.len() + 1;
        let hi = if params.is_empty() {
            "0".to_owned()
        } else {
            params.join(",")
        };
        self.proofs.push(format!(
            "proof {idx} rule=func_sig target={func_id} lo={} hi={hi} witness={ret}",
            params.len()
        ));
    }

    fn func(&mut self, func_id: u16, ret_lane: &str) -> String {
        let lbl = format!("$fn_{func_id}");
        let _ = writeln!(
            self.body,
            "node {lbl} = Func {ret_lane} fg=0 pol=l1 proof=0 tok=pure a={func_id} b=0"
        );
        self.last_func_label = Some(lbl.clone());
        lbl
    }

    fn end_void(&mut self) {
        let lbl = self.fresh("end");
        let _ = writeln!(
            self.body,
            "node {lbl} = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0"
        );
    }

    fn end_val(&mut self, lane: &str, val: &str) -> String {
        let lbl = self.fresh("end");
        let _ = writeln!(
            self.body,
            "node {lbl} = End {lane} fg=0 pol=l1 proof=0 tok=pure a={val} b=1"
        );
        lbl
    }

    fn ret_void(&mut self) {
        let lbl = self.fresh("ret");
        let _ = writeln!(
            self.body,
            "node {lbl} = Return i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0"
        );
    }

    fn ret_val(&mut self, lane: &str, val: &str) {
        let lbl = self.fresh("ret");
        let _ = writeln!(
            self.body,
            "node {lbl} = Return {lane} fg=0 pol=l1 proof=0 tok=pure a={val} b=0"
        );
    }

    fn const_i64(&mut self, val: i64) -> String {
        let lbl = self.fresh("ci");
        let _ = writeln!(
            self.body,
            "node {lbl} = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64={val}"
        );
        lbl
    }

    fn const_f64(&mut self, val: f64) -> String {
        let lbl = self.fresh("cf");
        let _ = writeln!(
            self.body,
            "node {lbl} = ConstF64 f64x1 fg=0 pol=l1 proof=0 tok=pure const={val:?}"
        );
        lbl
    }

    fn const_u8(&mut self, val: u8) -> String {
        let ci = self.const_i64(i64::from(val));
        self.cast("u8x1", &ci)
    }

    fn lget(&mut self, lane: &str, slot: u32) -> String {
        let lbl = self.fresh("lg");
        let _ = writeln!(
            self.body,
            "node {lbl} = LocalGet {lane} fg=0 pol=l1 proof=0 tok=pure a={slot} b=0"
        );
        lbl
    }

    fn lset(&mut self, lane: &str, slot: u32, val: &str) {
        let lbl = self.fresh("ls");
        let _ = writeln!(
            self.body,
            "node {lbl} = LocalSet {lane} fg=0 pol=l1 proof=0 tok=pure a={slot} b={val}"
        );
    }

    fn bin(&mut self, op: &str, lane: &str, a: &str, b: &str) -> String {
        let lbl = self.fresh("op");
        let _ = writeln!(
            self.body,
            "node {lbl} = {op} {lane} fg=0 pol=l1 proof=0 tok=pure a={a} b={b}"
        );
        lbl
    }

    fn cast(&mut self, lane: &str, a: &str) -> String {
        let lbl = self.fresh("cast");
        let _ = writeln!(
            self.body,
            "node {lbl} = Cast {lane} fg=0 pol=l1 proof=0 tok=pure a={a} b=0"
        );
        lbl
    }

    fn mask_not(&mut self, a: &str) -> String {
        let lbl = self.fresh("not");
        let _ = writeln!(
            self.body,
            "node {lbl} = MaskNot maskx1 fg=0 pol=l1 proof=0 tok=pure a={a} b=0"
        );
        lbl
    }

    fn block(&mut self) {
        let lbl = self.fresh("blk");
        let _ = writeln!(
            self.body,
            "node {lbl} = Block i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0"
        );
    }

    fn loop_open(&mut self) {
        let lbl = self.fresh("lp");
        let _ = writeln!(
            self.body,
            "node {lbl} = Loop i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0"
        );
    }

    fn if_open(&mut self, cond: &str) {
        let lbl = self.fresh("if");
        let _ = writeln!(
            self.body,
            "node {lbl} = If i64x1 fg=0 pol=l1 proof=0 tok=pure a={cond} b=0"
        );
    }

    fn else_open(&mut self) {
        let lbl = self.fresh("el");
        let _ = writeln!(
            self.body,
            "node {lbl} = Else i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0"
        );
    }

    fn br(&mut self, depth: u32) {
        let lbl = self.fresh("br");
        let _ = writeln!(
            self.body,
            "node {lbl} = Br i64x1 fg=0 pol=l1 proof=0 tok=pure a={depth} b=0"
        );
    }

    fn br_if(&mut self, cond: &str, depth: u32) {
        let lbl = self.fresh("bri");
        let _ = writeln!(
            self.body,
            "node {lbl} = BrIf i64x1 fg=0 pol=l1 proof=0 tok=pure a={cond} b={depth}"
        );
    }

    fn reg_new(&mut self, slot: u16, lane: &str, len: &str) {
        let lbl = self.fresh("rnew");
        let _ = writeln!(
            self.body,
            "node {lbl} = RegionNew {lane} fg=0 pol=l1 proof=0 tok=pure a={slot} b={len}"
        );
    }

    fn reg_grow(&mut self, slot: u16, add: &str) -> String {
        let lbl = self.fresh("rgrow");
        let _ = writeln!(
            self.body,
            "node {lbl} = RegionGrow i64x1 fg=0 pol=l1 proof=0 tok=pure a={slot} b={add}"
        );
        lbl
    }

    fn reg_len(&mut self, slot: u16) -> String {
        let lbl = self.fresh("rlen");
        let _ = writeln!(
            self.body,
            "node {lbl} = RegionLen i64x1 fg=0 pol=l1 proof=0 tok=pure a={slot} b=0"
        );
        lbl
    }

    fn reg_load(&mut self, slot: u16, lane: &str, idx: &str) -> String {
        let lbl = self.fresh("rld");
        let _ = writeln!(
            self.body,
            "node {lbl} = RegionLoad {lane} fg=0 pol=l1 proof=0 tok=pure a={slot} b={idx}"
        );
        lbl
    }

    fn reg_store(&mut self, slot: u16, lane: &str, idx: &str, val: &str) {
        let lbl = self.fresh("rst");
        let _ = writeln!(
            self.body,
            "node {lbl} = RegionStore {lane} fg=0 pol=l1 proof=0 tok=pure a={slot} b={idx} c={val}"
        );
    }

    fn call(&mut self, target_id: u16, ret_lane: &str, args: &[&str]) -> String {
        if args.len() >= 3 {
            let ca = self.fresh("carg");
            let a2 = args.get(2).copied().unwrap_or("none");
            let a3 = args.get(3).copied().unwrap_or("none");
            let _ = writeln!(
                self.body,
                "node {ca} = CallArg i64x1 fg=0 pol=l1 proof=0 tok=pure a={a2} b={a3}"
            );
        }
        let lbl = self.fresh("call");
        let a0 = args.first().copied().unwrap_or("none");
        let a1 = args.get(1).copied().unwrap_or("none");
        let _ = writeln!(
            self.body,
            "node {lbl} = Call {ret_lane} fg=0 pol=l1 proof=0 tok=pure a={target_id} b={a0} c={a1}"
        );
        lbl
    }

    fn finish(self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "cap {}", self.cap);
        for p in &self.proofs {
            let _ = writeln!(out, "{p}");
        }
        out.push_str(&self.body);
        if let Some(entry) = self.last_func_label {
            let _ = writeln!(out, "output {entry}");
        }
        out
    }
}

fn verify_rls_roundtrip(prog: &RulProgram, tag: &str) -> Result<(), String> {
    let rls = program_to_rls(prog);
    let reparsed = assemble_rls(&rls).map_err(|e| format!("{tag} roundtrip assemble: {e:?}"))?;
    if reparsed != *prog {
        return Err(format!("{tag} program_to_rls -> assemble_rls mismatch"));
    }
    Ok(())
}

fn run_tier0_and_aot_scalar(
    prog: &RulProgram,
    tag: &str,
) -> Result<(TypedScalar, crate::aot::AotCompileReport), String> {
    let t0 = execute_tier0_interpreter(prog, &[]).map_err(|e| format!("{tag} tier0: {e:?}"))?;
    let t0_scalar = t0
        .output
        .as_typed_scalar()
        .ok_or_else(|| format!("{tag} expected scalar output"))?;
    let exe = std::env::temp_dir().join(format!("rul_tc_core3_{}_{tag}", std::process::id()));
    let report =
        compile_rl_to_executable(prog, &exe).map_err(|e| format!("{tag} compile: {e:?}"))?;
    let stdout =
        run_compiled_structured_executable(&exe, &[]).map_err(|e| format!("{tag} aot: {e:?}"))?;
    let _ = std::fs::remove_file(&exe);
    let raw: [u8; 8] = stdout
        .as_slice()
        .try_into()
        .map_err(|_e| format!("{tag} expected 8 stdout bytes, got {}", stdout.len()))?;
    let aot_bits = u64::from_le_bytes(raw);
    assert_eq!(
        aot_bits, t0_scalar.bits,
        "{tag}: Tier 0 ({:#x}) != AOT ({:#x})",
        t0_scalar.bits, aot_bits
    );
    Ok((t0_scalar, report))
}

fn run_tier0_and_aot_io(
    prog: &RulProgram,
    input: &[u8],
    tag: &str,
) -> Result<(Vec<u8>, crate::aot::AotCompileReport), String> {
    let t0 = execute_tier0_interpreter_with_input(prog, &[], input)
        .map_err(|e| format!("{tag} tier0: {e:?}"))?;
    let t0_bytes = t0
        .output_bytes
        .ok_or_else(|| format!("{tag} missing output_bytes"))?;
    let exe = std::env::temp_dir().join(format!("rul_tc_core3_{}_{tag}", std::process::id()));
    let report =
        compile_rl_to_executable(prog, &exe).map_err(|e| format!("{tag} compile: {e:?}"))?;
    let aot_bytes =
        run_compiled_structured_executable(&exe, input).map_err(|e| format!("{tag} aot: {e:?}"))?;
    let _ = std::fs::remove_file(&exe);
    assert_eq!(
        aot_bytes, t0_bytes,
        "{tag}: Tier 0 output bytes != AOT output bytes"
    );
    Ok((t0_bytes, report))
}

// =========================================================================
// AC-1: Functions (arity 0..=4), CAP_DIVERGE-gated recursion & MAX_CALL_DEPTH
// =========================================================================
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "covers arity 0..=4 calls, recursion capability gate, exact MAX_CALL_DEPTH guard, and all AC-1 verifier type-rule rejections"
)]
fn test_tc_core3_functions_recursion_and_stack_guard() -> Result<(), String> {
    // 1. Multi-function non-recursive program with arity 0..=4, mixed scalar types (i64, f64, u8, bool),
    //    void helper writing to region, early Return, and End-return under CAP_NONE.
    let mut b = RlsBuilder::new("NONE");
    b.add_region(0, 1, 1, "i64", "plain");
    // func 0: const_seven() -> i64 (arity 0)
    b.add_func_sig(0, &[], "i64");
    // func 1: inc(x: i64) -> i64 (arity 1)
    b.add_func_sig(1, &["i64"], "i64");
    // func 2: store_add(a: i64, b: i64) -> void (arity 2)
    b.add_func_sig(2, &["i64", "i64"], "void");
    // func 3: mix3(a: i64, b: u8, flag: bool) -> i64 (arity 3 -> uses CallArg with a3=none)
    b.add_func_sig(3, &["i64", "u8", "bool"], "i64");
    // func 4: mix4(a: i64, b: f64, c: u8, d: bool) -> i64 (arity 4 -> uses CallArg with a2 & a3)
    b.add_func_sig(4, &["i64", "f64", "u8", "bool"], "i64");
    // func 5 (entry): main() -> i64 (arity 0)
    b.add_func_sig(5, &[], "i64");

    // fn 0: const_seven
    b.func(0, "i64x1");
    let c7 = b.const_i64(7);
    let _ = b.end_val("i64x1", &c7);

    // fn 1: inc(x)
    b.func(1, "i64x1");
    let x = b.lget("i64x1", 0);
    let c1 = b.const_i64(1);
    let xp1 = b.bin("Add", "i64x1", &x, &c1);
    b.ret_val("i64x1", &xp1);
    b.end_void();

    // fn 2: store_add(a, b) -> void
    b.func(2, "i64x1");
    let a = b.lget("i64x1", 0);
    let bv = b.lget("i64x1", 1);
    let s = b.bin("Add", "i64x1", &a, &bv);
    let z0 = b.const_i64(0);
    b.reg_store(0, "i64x1", &z0, &s);
    b.ret_void();
    b.end_void();

    // fn 3: mix3(a: i64, b: u8, flag: bool) -> i64
    b.func(3, "i64x1");
    let flag = b.lget("maskx1", 2);
    b.if_open(&flag);
    let a0 = b.lget("i64x1", 0);
    let b0 = b.lget("u8x1", 1);
    let b0_i = b.cast("i64x1", &b0);
    let sum3 = b.bin("Add", "i64x1", &a0, &b0_i);
    b.ret_val("i64x1", &sum3);
    b.end_void();
    let a0_f = b.lget("i64x1", 0);
    let b0_f = b.lget("u8x1", 1);
    let b0_fi = b.cast("i64x1", &b0_f);
    let diff3 = b.bin("Sub", "i64x1", &a0_f, &b0_fi);
    let _ = b.end_val("i64x1", &diff3);

    // fn 4: mix4(a: i64, b: f64, c: u8, d: bool) -> i64
    b.func(4, "i64x1");
    let a4 = b.lget("i64x1", 0);
    let b4 = b.lget("f64x1", 1);
    let b4_i = b.cast("i64x1", &b4);
    let c4 = b.lget("u8x1", 2);
    let d4 = b.lget("maskx1", 3);
    let ab = b.bin("Add", "i64x1", &a4, &b4_i);
    let r3 = b.call(3, "i64x1", &[&ab, &c4, &d4]);
    let _ = b.end_val("i64x1", &r3);

    // fn 5: main() -> i64
    b.func(5, "i64x1");
    let seven = b.call(0, "i64x1", &[]);
    let eight = b.call(1, "i64x1", &[&seven]);
    let c10 = b.const_i64(10);
    let _ = b.call(2, "i64x1", &[&eight, &c10]); // region[0] = 18
    let z0m = b.const_i64(0);
    let reg_v = b.reg_load(0, "i64x1", &z0m); // 18
    let f12 = b.const_f64(12.0);
    let u5 = b.const_u8(5);
    let t_cond = b.bin("CmpGt", "maskx1", &eight, &seven); // true
    let res = b.call(4, "i64x1", &[&reg_v, &f12, &u5, &t_cond]); // (18 + 12) + 5 = 35
    let _ = b.end_val("i64x1", &res);

    let prog = assemble_rls(&b.finish()).map_err(|e| format!("arity0..4 assemble: {e:?}"))?;
    verify_program(&prog).map_err(|e| format!("arity0..4 verify: {e:?}"))?;
    verify_rls_roundtrip(&prog, "arity0..4")?;
    let (scalar, _) = run_tier0_and_aot_scalar(&prog, "arity0_4")?;
    assert_eq!(scalar.as_i64(), Some(35));

    // 2. Self-recursion rejected without CAP_DIVERGE, admitted with CAP_DIVERGE.
    let build_countdown = |cap: &str, start_n: i64| -> Result<RulProgram, String> {
        let mut cb = RlsBuilder::new(cap);
        cb.add_func_sig(0, &["i64"], "i64");
        cb.add_func_sig(1, &[], "i64");
        cb.func(0, "i64x1");
        let n = cb.lget("i64x1", 0);
        let c0 = cb.const_i64(0);
        let is_zero = cb.bin("CmpEq", "maskx1", &n, &c0);
        cb.if_open(&is_zero);
        let ret0 = cb.const_i64(0);
        cb.ret_val("i64x1", &ret0);
        cb.end_void();
        let n2 = cb.lget("i64x1", 0);
        let c1 = cb.const_i64(1);
        let nm1 = cb.bin("Sub", "i64x1", &n2, &c1);
        let sub = cb.call(0, "i64x1", &[&nm1]);
        let ans = cb.bin("Add", "i64x1", &sub, &c1);
        let _ = cb.end_val("i64x1", &ans);

        cb.func(1, "i64x1");
        let sn = cb.const_i64(start_n);
        let out = cb.call(0, "i64x1", &[&sn]);
        let _ = cb.end_val("i64x1", &out);

        assemble_rls(&cb.finish()).map_err(|e| format!("countdown assemble: {e:?}"))
    };

    let self_rec_no_cap = build_countdown("NONE", 5)?;
    assert!(
        matches!(
            verify_program(&self_rec_no_cap),
            Err(crate::verifier::VerifyDiagnostic {
                kind: VerifyErrorKind::MissingCapability {
                    required: CAP_DIVERGE,
                    ..
                },
                ..
            })
        ),
        "self recursion must fail without CAP_DIVERGE"
    );

    // 3. Mutual recursion (`is_even` <-> `is_odd`, forward call target_func_id > caller_func_id)
    //    rejected without CAP_DIVERGE and admitted with CAP_DIVERGE.
    let build_mutual = |cap: &str, n_val: i64| -> Result<RulProgram, String> {
        let mut mb = RlsBuilder::new(cap);
        mb.add_func_sig(0, &["i64"], "i64"); // is_even
        mb.add_func_sig(1, &["i64"], "i64"); // is_odd
        mb.add_func_sig(2, &[], "i64"); // main

        mb.func(0, "i64x1");
        let n = mb.lget("i64x1", 0);
        let c0 = mb.const_i64(0);
        let eq0 = mb.bin("CmpEq", "maskx1", &n, &c0);
        mb.if_open(&eq0);
        let c1 = mb.const_i64(1);
        mb.ret_val("i64x1", &c1);
        mb.end_void();
        let n2 = mb.lget("i64x1", 0);
        let one = mb.const_i64(1);
        let nm1 = mb.bin("Sub", "i64x1", &n2, &one);
        let odd_r = mb.call(1, "i64x1", &[&nm1]); // forward call 0 -> 1
        let _ = mb.end_val("i64x1", &odd_r);

        mb.func(1, "i64x1");
        let m = mb.lget("i64x1", 0);
        let m0 = mb.const_i64(0);
        let meq0 = mb.bin("CmpEq", "maskx1", &m, &m0);
        mb.if_open(&meq0);
        let z = mb.const_i64(0);
        mb.ret_val("i64x1", &z);
        mb.end_void();
        let m2 = mb.lget("i64x1", 0);
        let mone = mb.const_i64(1);
        let mm1 = mb.bin("Sub", "i64x1", &m2, &mone);
        let even_r = mb.call(0, "i64x1", &[&mm1]); // back call 1 -> 0
        let _ = mb.end_val("i64x1", &even_r);

        mb.func(2, "i64x1");
        let nv = mb.const_i64(n_val);
        let res = mb.call(0, "i64x1", &[&nv]);
        let _ = mb.end_val("i64x1", &res);

        assemble_rls(&mb.finish()).map_err(|e| format!("mutual assemble: {e:?}"))
    };

    let mutual_no_cap = build_mutual("NONE", 10)?;
    assert!(
        matches!(
            verify_program(&mutual_no_cap),
            Err(crate::verifier::VerifyDiagnostic {
                kind: VerifyErrorKind::MissingCapability {
                    required: CAP_DIVERGE,
                    ..
                },
                ..
            })
        ),
        "mutual recursion / forward call must fail without CAP_DIVERGE"
    );

    let mutual_ok = build_mutual("DIVERGE", 42)?;
    verify_program(&mutual_ok).map_err(|e| format!("mutual verify: {e:?}"))?;
    verify_rls_roundtrip(&mutual_ok, "mutual_rec")?;
    let (m_scalar, _) = run_tier0_and_aot_scalar(&mutual_ok, "mutual_42")?;
    assert_eq!(m_scalar.as_i64(), Some(1));

    // 4. Exact MAX_CALL_DEPTH = 4096 boundary:
    //    `main` (1 activation) + `countdown(4094)` (4095 activations: 4094..=0) = 4096 live activations -> OK.
    //    `main` (1 activation) + `countdown(4095)` (4096 activations: 4095..=0) = 4097 live activations -> fail closed (75).
    let at_limit = build_countdown("DIVERGE", i64::from(MAX_CALL_DEPTH) - 2)?;
    let (at_scalar, _) = run_tier0_and_aot_scalar(&at_limit, "depth_4096_ok")?;
    assert_eq!(at_scalar.as_i64(), Some(i64::from(MAX_CALL_DEPTH) - 2));

    let over_limit = build_countdown("DIVERGE", i64::from(MAX_CALL_DEPTH) - 1)?;
    let t0_err = execute_tier0_interpreter(&over_limit, &[]);
    assert!(
        matches!(
            t0_err,
            Err(ExecError::CallDepthExceeded {
                call_depth: 4097,
                ..
            })
        ),
        "expected Tier 0 CallDepthExceeded at 4097, got {t0_err:?}"
    );
    let exe = std::env::temp_dir().join(format!("rul_tc_core3_{}_depth_4097", std::process::id()));
    compile_rl_to_executable(&over_limit, &exe)
        .map_err(|e| format!("depth_4097 compile: {e:?}"))?;
    let aot_err = run_compiled_structured_executable(&exe, &[]);
    let _ = std::fs::remove_file(&exe);
    assert!(
        matches!(aot_err, Err(AotError::StackOverflow)),
        "expected AOT StackOverflow (exit 75) at 4097, got {aot_err:?}"
    );

    // 5. Negative verifier type-rule tests for every AC-1 invariant (matching exact VerifyErrorKind):
    // 5a. Call argument count mismatch (too few / too many) -> InvalidControlFlow
    let call_too_few = assemble_rls(
        "\
cap NONE
proof 1 rule=func_sig target=0 lo=2 hi=i64,i64 witness=i64
proof 2 rule=func_sig target=1 lo=0 hi=0 witness=i64
node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $a0 = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$a0 b=1
node $f1 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
node $c1 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1
node $bad = Call i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$c1 c=none
node $e1 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$bad b=1
output $f1
",
    )
    .map_err(|e| format!("call_too_few assemble: {e:?}"))?;
    assert!(matches!(
        verify_program(&call_too_few),
        Err(crate::verifier::VerifyDiagnostic {
            kind: VerifyErrorKind::InvalidControlFlow(_),
            ..
        })
    ));

    let call_too_many = assemble_rls(
        "\
cap NONE
proof 1 rule=func_sig target=0 lo=1 hi=i64 witness=i64
proof 2 rule=func_sig target=1 lo=0 hi=0 witness=i64
node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $a0 = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$a0 b=1
node $f1 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
node $c1 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1
node $bad = Call i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$c1 c=$c1
node $e1 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$bad b=1
output $f1
",
    )
    .map_err(|e| format!("call_too_many assemble: {e:?}"))?;
    assert!(matches!(
        verify_program(&call_too_many),
        Err(crate::verifier::VerifyDiagnostic {
            kind: VerifyErrorKind::InvalidControlFlow(_),
            ..
        })
    ));

    // 5b. Call argument type mismatch (passing f64 to i64 param) -> ElementTypeMismatch
    let call_bad_arg_type = assemble_rls(
        "\
cap NONE
proof 1 rule=func_sig target=0 lo=1 hi=i64 witness=i64
proof 2 rule=func_sig target=1 lo=0 hi=0 witness=i64
node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $a0 = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$a0 b=1
node $f1 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
node $cf = ConstF64 f64x1 fg=0 pol=l1 proof=0 tok=pure const=1.5
node $bad = Call i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$cf c=none
node $e1 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$bad b=1
output $f1
",
    )
    .map_err(|e| format!("call_bad_arg_type assemble: {e:?}"))?;
    assert!(matches!(
        verify_program(&call_bad_arg_type),
        Err(crate::verifier::VerifyDiagnostic {
            kind: VerifyErrorKind::ElementTypeMismatch {
                expected: ElemType::I64,
                actual: ElemType::F64,
            },
            ..
        })
    ));

    // 5c. Call result type_lane mismatch (Call f64x1 calling -> i64) -> ElementTypeMismatch
    let call_bad_ret_lane = assemble_rls(
        "\
cap NONE
proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64
proof 2 rule=func_sig target=1 lo=0 hi=0 witness=i64
node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $c0 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=7
node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c0 b=1
node $f1 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
node $bad = Call f64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=none c=none
node $c1 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
node $e1 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c1 b=1
output $f1
",
    )
    .map_err(|e| format!("call_bad_ret_lane assemble: {e:?}"))?;
    assert!(matches!(
        verify_program(&call_bad_ret_lane),
        Err(crate::verifier::VerifyDiagnostic {
            kind: VerifyErrorKind::ElementTypeMismatch {
                expected: ElemType::I64,
                actual: ElemType::F64,
            },
            ..
        })
    ));

    // 5d. Unknown callee func_id -> UnknownFunction
    let call_unknown = assemble_rls(
        "\
cap NONE
proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64
node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $bad = Call i64x1 fg=0 pol=l1 proof=0 tok=pure a=99 b=none c=none
node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$bad b=1
output $f0
",
    )
    .map_err(|e| format!("call_unknown assemble: {e:?}"))?;
    assert!(matches!(
        verify_program(&call_unknown),
        Err(crate::verifier::VerifyDiagnostic {
            kind: VerifyErrorKind::UnknownFunction(99),
            ..
        })
    ));

    // 5e. Return value type mismatch -> ElementTypeMismatch; void Return with value -> InvalidControlFlow
    let ret_bad_type = assemble_rls(
        "\
cap NONE
proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64
node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $cf = ConstF64 f64x1 fg=0 pol=l1 proof=0 tok=pure const=2.5
node $r = Return i64x1 fg=0 pol=l1 proof=0 tok=pure a=$cf b=0
node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
output $f0
",
    )
    .map_err(|e| format!("ret_bad_type assemble: {e:?}"))?;
    assert!(matches!(
        verify_program(&ret_bad_type),
        Err(crate::verifier::VerifyDiagnostic {
            kind: VerifyErrorKind::ElementTypeMismatch {
                expected: ElemType::I64,
                actual: ElemType::F64,
            },
            ..
        })
    ));

    let ret_void_with_val = assemble_rls(
        "\
cap NONE
proof 1 rule=func_sig target=0 lo=0 hi=0 witness=void
proof 2 rule=func_sig target=1 lo=0 hi=0 witness=i64
node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $c = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1
node $r = Return i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c b=0
node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $f1 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
node $c1 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
node $e1 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c1 b=1
output $f1
",
    )
    .map_err(|e| format!("ret_void_with_val assemble: {e:?}"))?;
    assert!(matches!(
        verify_program(&ret_void_with_val),
        Err(crate::verifier::VerifyDiagnostic {
            kind: VerifyErrorKind::InvalidControlFlow(_),
            ..
        })
    ));

    // 5f. Closing End of Func: wrong value type -> ElementTypeMismatch; reachable fallthrough -> InvalidControlFlow
    let end_bad_type = assemble_rls(
        "\
cap NONE
proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64
node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $cf = ConstF64 f64x1 fg=0 pol=l1 proof=0 tok=pure const=3.0
node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$cf b=1
output $f0
",
    )
    .map_err(|e| format!("end_bad_type assemble: {e:?}"))?;
    assert!(matches!(
        verify_program(&end_bad_type),
        Err(crate::verifier::VerifyDiagnostic {
            kind: VerifyErrorKind::ElementTypeMismatch {
                expected: ElemType::I64,
                actual: ElemType::F64,
            },
            ..
        })
    ));

    let end_fallthrough = assemble_rls(
        "\
cap NONE
proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64
node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $c = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1
node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
output $f0
",
    )
    .map_err(|e| format!("end_fallthrough assemble: {e:?}"))?;
    assert!(matches!(
        verify_program(&end_fallthrough),
        Err(crate::verifier::VerifyDiagnostic {
            kind: VerifyErrorKind::InvalidControlFlow(_),
            ..
        })
    ));

    // 5g. Local initialization and types reset across function boundaries:
    //     fn0 initializes slot 5; fn1 reading slot 5 must fail with UninitializedLocalRead { slot: 5 }.
    let local_cross_fn = assemble_rls(
        "\
cap NONE
proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64
proof 2 rule=func_sig target=1 lo=0 hi=0 witness=i64
node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $c0 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=99
node $s5 = LocalSet i64x1 fg=0 pol=l1 proof=0 tok=pure a=5 b=$c0
node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$s5 b=1
node $f1 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
node $g5 = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=5 b=0
node $e1 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$g5 b=1
output $f1
",
    )
    .map_err(|e| format!("local_cross_fn assemble: {e:?}"))?;
    assert!(matches!(
        verify_program(&local_cross_fn),
        Err(crate::verifier::VerifyDiagnostic {
            kind: VerifyErrorKind::UninitializedLocalRead { slot: 5 },
            ..
        })
    ));

    // 5h. CallArg not followed by Call -> InvalidControlFlow; CallArg with <2 args on Call -> InvalidControlFlow;
    //     Call arg1 without arg0 -> InvalidControlFlow
    let callarg_not_followed = assemble_rls(
        "\
cap NONE
proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64
node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $c0 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1
node $ca = CallArg i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c0 b=none
node $c1 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=2
node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c1 b=1
output $f0
",
    )
    .map_err(|e| format!("callarg_not_followed assemble: {e:?}"))?;
    assert!(matches!(
        verify_program(&callarg_not_followed),
        Err(crate::verifier::VerifyDiagnostic {
            kind: VerifyErrorKind::InvalidControlFlow(_),
            ..
        })
    ));

    let callarg_lt2 = assemble_rls(
        "\
cap NONE
proof 1 rule=func_sig target=0 lo=1 hi=i64 witness=i64
proof 2 rule=func_sig target=1 lo=0 hi=0 witness=i64
node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $a0 = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$a0 b=1
node $f1 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
node $c0 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1
node $ca = CallArg i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c0 b=none
node $bad = Call i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=$c0 c=none
node $e1 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$bad b=1
output $f1
",
    )
    .map_err(|e| format!("callarg_lt2 assemble: {e:?}"))?;
    assert!(matches!(
        verify_program(&callarg_lt2),
        Err(crate::verifier::VerifyDiagnostic {
            kind: VerifyErrorKind::InvalidControlFlow(_),
            ..
        })
    ));

    let call_arg1_no_arg0 = assemble_rls(
        "\
cap NONE
proof 1 rule=func_sig target=0 lo=1 hi=i64 witness=i64
proof 2 rule=func_sig target=1 lo=0 hi=0 witness=i64
node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $a0 = LocalGet i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$a0 b=1
node $f1 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=1 b=0
node $c0 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=1
node $bad = Call i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=none c=$c0
node $e1 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$bad b=1
output $f1
",
    )
    .map_err(|e| format!("call_arg1_no_arg0 assemble: {e:?}"))?;
    assert!(matches!(
        verify_program(&call_arg1_no_arg0),
        Err(crate::verifier::VerifyDiagnostic {
            kind: VerifyErrorKind::InvalidControlFlow(_),
            ..
        })
    ));

    // 5i. Malformed FuncSig (arity > 4, non-zero high nibble, invalid return tag) -> InvalidFuncSig
    for bad_sig in [
        ProofEntry {
            rule_kind: crate::bytecode::ProofRuleKind::FuncSig,
            target_arena: 0,
            bound_lo: 5, // arity > 4
            bound_hi: 0x2222,
            witness_node: 2,
        },
        ProofEntry {
            rule_kind: crate::bytecode::ProofRuleKind::FuncSig,
            target_arena: 0,
            bound_lo: 1,
            bound_hi: 0x0022, // non-zero unused param nibble
            witness_node: 2,
        },
        ProofEntry {
            rule_kind: crate::bytecode::ProofRuleKind::FuncSig,
            target_arena: 0,
            bound_lo: 0,
            bound_hi: 0,
            witness_node: 99, // unknown ElemType
        },
    ] {
        let p_bad_sig = RulProgram {
            capability_mask: CAP_NONE,
            arenas: Vec::new(),
            proofs: vec![bad_sig],
            nodes: vec![
                NodeFrame {
                    opcode: Opcode::Func,
                    type_lane: TypeLane::I64_SCALAR,
                    schedule: ScheduleHint::UNFUSED,
                    proof_idx: 0,
                    token_op: TokenOp::PURE,
                    in_a: 0,
                    in_b: 0,
                },
                NodeFrame::const_i64(0, ScheduleHint::UNFUSED),
                NodeFrame {
                    opcode: Opcode::End,
                    type_lane: TypeLane::I64_SCALAR,
                    schedule: ScheduleHint::UNFUSED,
                    proof_idx: 0,
                    token_op: TokenOp::PURE,
                    in_a: 1,
                    in_b: 1,
                },
            ],
            output_node: 0,
        };
        assert!(matches!(
            verify_program(&p_bad_sig),
            Err(crate::verifier::VerifyDiagnostic {
                kind: VerifyErrorKind::InvalidFuncSig { .. },
                ..
            })
        ));
    }

    // 5j. Br and BrIf targeting the Func scope -> InvalidControlFlow
    let br_to_func = assemble_rls(
        "\
cap NONE
proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64
node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $br = Br i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
output $f0
",
    )
    .map_err(|e| format!("br_to_func assemble: {e:?}"))?;
    assert!(matches!(
        verify_program(&br_to_func),
        Err(crate::verifier::VerifyDiagnostic {
            kind: VerifyErrorKind::InvalidControlFlow(_),
            ..
        })
    ));

    let brif_to_func = assemble_rls(
        "\
cap NONE
proof 1 rule=func_sig target=0 lo=0 hi=0 witness=i64
node $f0 = Func i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0
node $c0 = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
node $cond = CmpEq maskx1 fg=0 pol=l1 proof=0 tok=pure a=$c0 b=$c0
node $bri = BrIf i64x1 fg=0 pol=l1 proof=0 tok=pure a=$cond b=0
node $e0 = End i64x1 fg=0 pol=l1 proof=0 tok=pure a=$c0 b=1
output $f0
",
    )
    .map_err(|e| format!("brif_to_func assemble: {e:?}"))?;
    assert!(matches!(
        verify_program(&brif_to_func),
        Err(crate::verifier::VerifyDiagnostic {
            kind: VerifyErrorKind::InvalidControlFlow(_),
            ..
        })
    ));

    Ok(())
}

// =========================================================================
// AC-2: Brainf*ck interpreter Turing-completeness proof
// =========================================================================
fn build_brainfuck_rls() -> String {
    let mut b = RlsBuilder::new("DIVERGE");
    // Slot 0: input (BF source, optional '!' + stdin)
    b.add_region(0, 0, 65_536, "u8", "input");
    // Slot 1: output (bytes emitted by '.')
    b.add_region(1, 0, 65_536, "u8", "output");
    // Slot 2: 30,000-byte wrapping U8 tape
    b.add_region(2, 30_000, 30_000, "u8", "plain");
    // Slot 3: jump table (i64, length = code_len)
    b.add_region(3, 0, 65_536, "i64", "plain");
    // Slot 4: bracket stack (i64, length = code_len)
    b.add_region(4, 0, 65_536, "i64", "plain");

    // func 0: find_code_end(total_len: i64) -> i64
    b.add_func_sig(0, &["i64"], "i64");
    // func 1: build_jumps(code_len: i64) -> void
    b.add_func_sig(1, &["i64"], "void");
    // func 2: emit_byte(ch: u8) -> void
    b.add_func_sig(2, &["u8"], "void");
    // func 3 (entry): main() -> i64
    b.add_func_sig(3, &[], "i64");

    // --- func 0: find_code_end(total_len: local 0) ---
    b.func(0, "i64x1");
    let z0 = b.const_i64(0);
    b.lset("i64x1", 1, &z0); // i = 0
    b.block();
    b.loop_open();
    let i = b.lget("i64x1", 1);
    let tot = b.lget("i64x1", 0);
    let lt = b.bin("CmpLt", "maskx1", &i, &tot);
    let done = b.mask_not(&lt);
    b.br_if(&done, 1);
    let ch = b.reg_load(0, "u8x1", &i);
    let bang = b.const_u8(b'!');
    let is_bang = b.bin("CmpEq", "maskx1", &ch, &bang);
    b.if_open(&is_bang);
    let i_ret = b.lget("i64x1", 1);
    b.ret_val("i64x1", &i_ret);
    b.end_void();
    let c1 = b.const_i64(1);
    let i_cur = b.lget("i64x1", 1);
    let i_next = b.bin("Add", "i64x1", &i_cur, &c1);
    b.lset("i64x1", 1, &i_next);
    b.br(0);
    b.end_void(); // end loop
    b.end_void(); // end block
    let tot_ret = b.lget("i64x1", 0);
    let _ = b.end_val("i64x1", &tot_ret);

    // --- func 1: build_jumps(code_len: local 0) -> void ---
    b.func(1, "i64x1");
    let clen = b.lget("i64x1", 0);
    b.reg_new(3, "i64x1", &clen);
    b.reg_new(4, "i64x1", &clen);
    let z = b.const_i64(0);
    b.lset("i64x1", 1, &z); // i = 0
    b.lset("i64x1", 2, &z); // sp = 0
    b.block();
    b.loop_open();
    let i = b.lget("i64x1", 1);
    let n = b.lget("i64x1", 0);
    let lt = b.bin("CmpLt", "maskx1", &i, &n);
    let done = b.mask_not(&lt);
    b.br_if(&done, 1);

    let ch = b.reg_load(0, "u8x1", &i);
    let lb = b.const_u8(b'[');
    let is_lb = b.bin("CmpEq", "maskx1", &ch, &lb);
    b.if_open(&is_lb);
    let sp = b.lget("i64x1", 2);
    let ic = b.lget("i64x1", 1);
    b.reg_store(4, "i64x1", &sp, &ic);
    let c1 = b.const_i64(1);
    let sp1 = b.bin("Add", "i64x1", &sp, &c1);
    b.lset("i64x1", 2, &sp1);
    b.end_void();

    let rb = b.const_u8(b']');
    let is_rb = b.bin("CmpEq", "maskx1", &ch, &rb);
    b.if_open(&is_rb);
    let sp = b.lget("i64x1", 2);
    let c1 = b.const_i64(1);
    let spm1 = b.bin("Sub", "i64x1", &sp, &c1);
    b.lset("i64x1", 2, &spm1);
    let open_pos = b.reg_load(4, "i64x1", &spm1);
    let ic = b.lget("i64x1", 1);
    b.reg_store(3, "i64x1", &open_pos, &ic);
    b.reg_store(3, "i64x1", &ic, &open_pos);
    b.end_void();

    let c1 = b.const_i64(1);
    let ic = b.lget("i64x1", 1);
    let inext = b.bin("Add", "i64x1", &ic, &c1);
    b.lset("i64x1", 1, &inext);
    b.br(0);
    b.end_void(); // end loop
    b.end_void(); // end block
    b.end_void(); // end func 1

    // --- func 2: emit_byte(ch: u8 in local 0) -> void ---
    b.func(2, "i64x1");
    let c1 = b.const_i64(1);
    let new_len = b.reg_grow(1, &c1);
    let idx = b.bin("Sub", "i64x1", &new_len, &c1);
    let ch = b.lget("u8x1", 0);
    b.reg_store(1, "u8x1", &idx, &ch);
    b.end_void();

    // --- func 3: main() -> i64 ---
    // locals: 0 = total_len, 1 = code_len, 2 = pc, 3 = dp, 4 = in_pos
    b.func(3, "i64x1");
    let total_len = b.reg_len(0);
    b.lset("i64x1", 0, &total_len);
    let code_len = b.call(0, "i64x1", &[&total_len]);
    b.lset("i64x1", 1, &code_len);
    let _ = b.call(1, "i64x1", &[&code_len]);
    let z = b.const_i64(0);
    let c1 = b.const_i64(1);
    b.lset("i64x1", 2, &z); // pc = 0
    b.lset("i64x1", 3, &z); // dp = 0
    let in_start = b.bin("Add", "i64x1", &code_len, &c1);
    b.lset("i64x1", 4, &in_start); // in_pos = code_len + 1

    b.block();
    b.loop_open();
    let pc = b.lget("i64x1", 2);
    let cl = b.lget("i64x1", 1);
    let lt = b.bin("CmpLt", "maskx1", &pc, &cl);
    let done = b.mask_not(&lt);
    b.br_if(&done, 1);

    let cmd = b.reg_load(0, "u8x1", &pc);

    // '>'
    let c_gt = b.const_u8(b'>');
    let is_gt = b.bin("CmpEq", "maskx1", &cmd, &c_gt);
    b.if_open(&is_gt);
    let dp = b.lget("i64x1", 3);
    let one = b.const_i64(1);
    let dp1 = b.bin("Add", "i64x1", &dp, &one);
    let tape_len = b.const_i64(30_000);
    let wrap = b.bin("CmpEq", "maskx1", &dp1, &tape_len);
    b.if_open(&wrap);
    let z0 = b.const_i64(0);
    b.lset("i64x1", 3, &z0);
    b.else_open();
    b.lset("i64x1", 3, &dp1);
    b.end_void();
    b.end_void();

    // '<'
    let c_lt = b.const_u8(b'<');
    let is_lt = b.bin("CmpEq", "maskx1", &cmd, &c_lt);
    b.if_open(&is_lt);
    let dp = b.lget("i64x1", 3);
    let z0 = b.const_i64(0);
    let at_z = b.bin("CmpEq", "maskx1", &dp, &z0);
    b.if_open(&at_z);
    let last = b.const_i64(29_999);
    b.lset("i64x1", 3, &last);
    b.else_open();
    let one = b.const_i64(1);
    let dpm1 = b.bin("Sub", "i64x1", &dp, &one);
    b.lset("i64x1", 3, &dpm1);
    b.end_void();
    b.end_void();

    // '+'
    let c_plus = b.const_u8(b'+');
    let is_plus = b.bin("CmpEq", "maskx1", &cmd, &c_plus);
    b.if_open(&is_plus);
    let dp = b.lget("i64x1", 3);
    let cur = b.reg_load(2, "u8x1", &dp);
    let u1 = b.const_u8(1);
    let next_v = b.bin("Add", "u8x1", &cur, &u1);
    b.reg_store(2, "u8x1", &dp, &next_v);
    b.end_void();

    // '-'
    let c_minus = b.const_u8(b'-');
    let is_minus = b.bin("CmpEq", "maskx1", &cmd, &c_minus);
    b.if_open(&is_minus);
    let dp = b.lget("i64x1", 3);
    let cur = b.reg_load(2, "u8x1", &dp);
    let u1 = b.const_u8(1);
    let next_v = b.bin("Sub", "u8x1", &cur, &u1);
    b.reg_store(2, "u8x1", &dp, &next_v);
    b.end_void();

    // '.'
    let c_dot = b.const_u8(b'.');
    let is_dot = b.bin("CmpEq", "maskx1", &cmd, &c_dot);
    b.if_open(&is_dot);
    let dp = b.lget("i64x1", 3);
    let cur = b.reg_load(2, "u8x1", &dp);
    let _ = b.call(2, "i64x1", &[&cur]);
    b.end_void();

    // ','
    let c_comma = b.const_u8(b',');
    let is_comma = b.bin("CmpEq", "maskx1", &cmd, &c_comma);
    b.if_open(&is_comma);
    let in_pos = b.lget("i64x1", 4);
    let tot = b.lget("i64x1", 0);
    let dp = b.lget("i64x1", 3);
    let has_in = b.bin("CmpLt", "maskx1", &in_pos, &tot);
    b.if_open(&has_in);
    let byte = b.reg_load(0, "u8x1", &in_pos);
    b.reg_store(2, "u8x1", &dp, &byte);
    let one = b.const_i64(1);
    let in_next = b.bin("Add", "i64x1", &in_pos, &one);
    b.lset("i64x1", 4, &in_next);
    b.else_open();
    let u0 = b.const_u8(0);
    b.reg_store(2, "u8x1", &dp, &u0);
    b.end_void();
    b.end_void();

    // '['
    let c_lb = b.const_u8(b'[');
    let is_lb = b.bin("CmpEq", "maskx1", &cmd, &c_lb);
    b.if_open(&is_lb);
    let dp = b.lget("i64x1", 3);
    let cur = b.reg_load(2, "u8x1", &dp);
    let u0 = b.const_u8(0);
    let is_z = b.bin("CmpEq", "maskx1", &cur, &u0);
    b.if_open(&is_z);
    let pc_cur = b.lget("i64x1", 2);
    let tgt = b.reg_load(3, "i64x1", &pc_cur);
    b.lset("i64x1", 2, &tgt);
    b.end_void();
    b.end_void();

    // ']'
    let c_rb = b.const_u8(b']');
    let is_rb = b.bin("CmpEq", "maskx1", &cmd, &c_rb);
    b.if_open(&is_rb);
    let dp = b.lget("i64x1", 3);
    let cur = b.reg_load(2, "u8x1", &dp);
    let u0 = b.const_u8(0);
    let is_z = b.bin("CmpEq", "maskx1", &cur, &u0);
    let is_nz = b.mask_not(&is_z);
    b.if_open(&is_nz);
    let pc_cur = b.lget("i64x1", 2);
    let tgt = b.reg_load(3, "i64x1", &pc_cur);
    b.lset("i64x1", 2, &tgt);
    b.end_void();
    b.end_void();

    // pc += 1
    let pc_now = b.lget("i64x1", 2);
    let one = b.const_i64(1);
    let pc_next = b.bin("Add", "i64x1", &pc_now, &one);
    b.lset("i64x1", 2, &pc_next);
    b.br(0);
    b.end_void(); // end loop
    b.end_void(); // end block

    let out_len = b.reg_len(1);
    let _ = b.end_val("i64x1", &out_len);

    b.finish()
}

#[test]
fn test_tc_core3_brainfuck_turing_completeness() -> Result<(), String> {
    let rls = build_brainfuck_rls();
    let prog = assemble_rls(&rls).map_err(|e| format!("bf assemble: {e:?}"))?;
    verify_program(&prog).map_err(|e| format!("bf verify: {e:?}"))?;
    verify_rls_roundtrip(&prog, "brainfuck")?;

    // 1. Classic nested-loop "Hello World!\n"
    let hello_bf = b"++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.";
    let (hello_out, report) = run_tier0_and_aot_io(&prog, hello_bf, "bf_hello")?;
    assert_eq!(hello_out.as_slice(), b"Hello World!\n");
    // Verify microsecond-class native compilation latency (< 10 ms even in debug builds).
    assert!(
        report.total_compile_ns < 10_000_000,
        "BF compile took {} ns (expected microsecond-class)",
        report.total_compile_ns
    );

    // 2. Interactive `,` input with '!' separator, nested loops, wrapping u8, and EOF -> 0:
    //    Reads input bytes until EOF (0), shifts each byte by +1, and prints it.
    let caesar_bf = b",[+.,]!Rul05";
    let (caesar_out, _) = run_tier0_and_aot_io(&prog, caesar_bf, "bf_caesar")?;
    assert_eq!(caesar_out.as_slice(), b"Svm16");

    // 3. Nested loop multiplication: read two digits via `,`, multiply via nested loops, output ASCII.
    //    Input "!6x7" -> reads '6' (54) - 48 = 6, '7' (55) - 48 = 7, computes 6*7 + 48 = 90 ('Z').
    let mul_bf = b",>++++++[<-------->-],>++++++[<-------->-]<<[>[>+>+<<-]>>[<<+>>-]<<<-]>>>++++++[<++++++++>-]<.!67";
    let (mul_out, _) = run_tier0_and_aot_io(&prog, mul_bf, "bf_mul")?;
    assert_eq!(mul_out.as_slice(), b"Z");

    Ok(())
}

// =========================================================================
// AC-3: Exit programs suite (fib(20), quicksort, sieve(10_000), sha256)
// =========================================================================
fn build_fib_rls(n_val: i64) -> String {
    let mut b = RlsBuilder::new("DIVERGE");
    b.add_func_sig(0, &["i64"], "i64");
    b.add_func_sig(1, &[], "i64");

    b.func(0, "i64x1");
    let n = b.lget("i64x1", 0);
    let c2 = b.const_i64(2);
    let lt2 = b.bin("CmpLt", "maskx1", &n, &c2);
    b.if_open(&lt2);
    let n_ret = b.lget("i64x1", 0);
    b.ret_val("i64x1", &n_ret);
    b.end_void();
    let n_a = b.lget("i64x1", 0);
    let c1 = b.const_i64(1);
    let nm1 = b.bin("Sub", "i64x1", &n_a, &c1);
    let f1 = b.call(0, "i64x1", &[&nm1]);
    let n_b = b.lget("i64x1", 0);
    let c2b = b.const_i64(2);
    let nm2 = b.bin("Sub", "i64x1", &n_b, &c2b);
    let f2 = b.call(0, "i64x1", &[&nm2]);
    let sum = b.bin("Add", "i64x1", &f1, &f2);
    let _ = b.end_val("i64x1", &sum);

    b.func(1, "i64x1");
    let nv = b.const_i64(n_val);
    let res = b.call(0, "i64x1", &[&nv]);
    let _ = b.end_val("i64x1", &res);

    b.finish()
}

fn build_quicksort_rls(values: &[i64]) -> String {
    let n = u32::try_from(values.len()).unwrap_or(0);
    let mut b = RlsBuilder::new("DIVERGE");
    b.add_region(0, n, n, "i64", "plain");
    b.add_region(1, 0, n.saturating_mul(8), "u8", "output");

    // func 0: swap(i: i64, j: i64) -> void
    b.add_func_sig(0, &["i64", "i64"], "void");
    // func 1: partition(lo: i64, hi: i64) -> i64
    b.add_func_sig(1, &["i64", "i64"], "i64");
    // func 2: qsort(lo: i64, hi: i64) -> void
    b.add_func_sig(2, &["i64", "i64"], "void");
    // func 3: emit_i64_le(val: i64) -> void
    b.add_func_sig(3, &["i64"], "void");
    // func 4 (entry): main() -> i64
    b.add_func_sig(4, &[], "i64");

    // func 0: swap
    b.func(0, "i64x1");
    let i = b.lget("i64x1", 0);
    let j = b.lget("i64x1", 1);
    let vi = b.reg_load(0, "i64x1", &i);
    let vj = b.reg_load(0, "i64x1", &j);
    b.reg_store(0, "i64x1", &i, &vj);
    b.reg_store(0, "i64x1", &j, &vi);
    b.end_void();

    // func 1: partition(lo: 0, hi: 1) -> i64
    // locals: 2 = pivot, 3 = i, 4 = j
    b.func(1, "i64x1");
    let lo = b.lget("i64x1", 0);
    let hi = b.lget("i64x1", 1);
    let pivot = b.reg_load(0, "i64x1", &hi);
    b.lset("i64x1", 2, &pivot);
    b.lset("i64x1", 3, &lo);
    b.lset("i64x1", 4, &lo);
    b.block();
    b.loop_open();
    let j = b.lget("i64x1", 4);
    let h = b.lget("i64x1", 1);
    let lt = b.bin("CmpLt", "maskx1", &j, &h);
    let done = b.mask_not(&lt);
    b.br_if(&done, 1);
    let aj = b.reg_load(0, "i64x1", &j);
    let piv = b.lget("i64x1", 2);
    let gt = b.bin("CmpGt", "maskx1", &aj, &piv);
    let le = b.mask_not(&gt);
    b.if_open(&le);
    let ii = b.lget("i64x1", 3);
    let jj = b.lget("i64x1", 4);
    let _ = b.call(0, "i64x1", &[&ii, &jj]);
    let c1 = b.const_i64(1);
    let ii1 = b.bin("Add", "i64x1", &ii, &c1);
    b.lset("i64x1", 3, &ii1);
    b.end_void();
    let jcur = b.lget("i64x1", 4);
    let c1 = b.const_i64(1);
    let jnext = b.bin("Add", "i64x1", &jcur, &c1);
    b.lset("i64x1", 4, &jnext);
    b.br(0);
    b.end_void();
    b.end_void();
    let ifin = b.lget("i64x1", 3);
    let hfin = b.lget("i64x1", 1);
    let _ = b.call(0, "i64x1", &[&ifin, &hfin]);
    let _ = b.end_val("i64x1", &ifin);

    // func 2: qsort(lo: 0, hi: 1) -> void
    b.func(2, "i64x1");
    let lo = b.lget("i64x1", 0);
    let hi = b.lget("i64x1", 1);
    let ok = b.bin("CmpLt", "maskx1", &lo, &hi);
    let stop = b.mask_not(&ok);
    b.if_open(&stop);
    b.ret_void();
    b.end_void();
    let p = b.call(1, "i64x1", &[&lo, &hi]);
    b.lset("i64x1", 2, &p);
    let c1 = b.const_i64(1);
    let pm1 = b.bin("Sub", "i64x1", &p, &c1);
    let lo2 = b.lget("i64x1", 0);
    let _ = b.call(2, "i64x1", &[&lo2, &pm1]);
    let p2 = b.lget("i64x1", 2);
    let c1b = b.const_i64(1);
    let pp1 = b.bin("Add", "i64x1", &p2, &c1b);
    let hi2 = b.lget("i64x1", 1);
    let _ = b.call(2, "i64x1", &[&pp1, &hi2]);
    b.end_void();

    // func 3: emit_i64_le(val: 0) -> void
    b.func(3, "i64x1");
    let c8 = b.const_i64(8);
    let new_len = b.reg_grow(1, &c8);
    let base = b.bin("Sub", "i64x1", &new_len, &c8);
    b.lset("i64x1", 1, &base);
    let z0 = b.const_i64(0);
    b.lset("i64x1", 2, &z0); // k = 0..8
    b.block();
    b.loop_open();
    let k = b.lget("i64x1", 2);
    let l8 = b.const_i64(8);
    let lt = b.bin("CmpLt", "maskx1", &k, &l8);
    let done = b.mask_not(&lt);
    b.br_if(&done, 1);
    let val = b.lget("i64x1", 0);
    let sh = b.bin("Mul", "i64x1", &k, &l8);
    let shifted = b.bin("ShrU", "i64x1", &val, &sh);
    let byte = b.cast("u8x1", &shifted);
    let bpos = b.lget("i64x1", 1);
    let dst = b.bin("Add", "i64x1", &bpos, &k);
    b.reg_store(1, "u8x1", &dst, &byte);
    let c1 = b.const_i64(1);
    let kn = b.bin("Add", "i64x1", &k, &c1);
    b.lset("i64x1", 2, &kn);
    b.br(0);
    b.end_void();
    b.end_void();
    b.end_void();

    // func 4: main() -> i64
    b.func(4, "i64x1");
    for (idx, &v) in values.iter().enumerate() {
        let ci = b.const_i64(i64::try_from(idx).unwrap_or(0));
        let cv = b.const_i64(v);
        b.reg_store(0, "i64x1", &ci, &cv);
    }
    let lo0 = b.const_i64(0);
    let hi0 = b.const_i64(i64::from(n).saturating_sub(1));
    let _ = b.call(2, "i64x1", &[&lo0, &hi0]);
    let z = b.const_i64(0);
    b.lset("i64x1", 0, &z);
    b.block();
    b.loop_open();
    let idx = b.lget("i64x1", 0);
    let lim = b.const_i64(i64::from(n));
    let lt = b.bin("CmpLt", "maskx1", &idx, &lim);
    let done = b.mask_not(&lt);
    b.br_if(&done, 1);
    let elem = b.reg_load(0, "i64x1", &idx);
    let _ = b.call(3, "i64x1", &[&elem]);
    let c1 = b.const_i64(1);
    let next = b.bin("Add", "i64x1", &idx, &c1);
    b.lset("i64x1", 0, &next);
    b.br(0);
    b.end_void();
    b.end_void();
    let ret_n = b.const_i64(i64::from(n));
    let _ = b.end_val("i64x1", &ret_n);

    b.finish()
}

fn build_sieve_rls(limit: u32) -> String {
    let mut b = RlsBuilder::new("DIVERGE");
    let size = limit.saturating_add(1);
    b.add_region(0, size, size, "u8", "plain");

    // func 0: mark_multiples(p: i64, limit: i64) -> void
    b.add_func_sig(0, &["i64", "i64"], "void");
    // func 1: count_primes(limit: i64) -> i64
    b.add_func_sig(1, &["i64"], "i64");
    // func 2 (entry): main() -> i64
    b.add_func_sig(2, &[], "i64");

    // func 0: mark_multiples(p: 0, limit: 1) -> void
    b.func(0, "i64x1");
    let p = b.lget("i64x1", 0);
    let m0 = b.bin("Mul", "i64x1", &p, &p);
    b.lset("i64x1", 2, &m0); // m = p * p
    b.block();
    b.loop_open();
    let m = b.lget("i64x1", 2);
    let lim = b.lget("i64x1", 1);
    let gt = b.bin("CmpGt", "maskx1", &m, &lim);
    b.br_if(&gt, 1);
    let u1 = b.const_u8(1);
    b.reg_store(0, "u8x1", &m, &u1);
    let p_step = b.lget("i64x1", 0);
    let mnext = b.bin("Add", "i64x1", &m, &p_step);
    b.lset("i64x1", 2, &mnext);
    b.br(0);
    b.end_void();
    b.end_void();
    b.end_void();

    // func 1: count_primes(limit: 0) -> i64
    b.func(1, "i64x1");
    let c2 = b.const_i64(2);
    let z0 = b.const_i64(0);
    b.lset("i64x1", 1, &c2); // i = 2
    b.lset("i64x1", 2, &z0); // cnt = 0
    b.block();
    b.loop_open();
    let i = b.lget("i64x1", 1);
    let lim = b.lget("i64x1", 0);
    let gt = b.bin("CmpGt", "maskx1", &i, &lim);
    b.br_if(&gt, 1);
    let flag = b.reg_load(0, "u8x1", &i);
    let u0 = b.const_u8(0);
    let is_prime = b.bin("CmpEq", "maskx1", &flag, &u0);
    b.if_open(&is_prime);
    let cnt = b.lget("i64x1", 2);
    let c1 = b.const_i64(1);
    let cnt1 = b.bin("Add", "i64x1", &cnt, &c1);
    b.lset("i64x1", 2, &cnt1);
    b.end_void();
    let c1 = b.const_i64(1);
    let inext = b.bin("Add", "i64x1", &i, &c1);
    b.lset("i64x1", 1, &inext);
    b.br(0);
    b.end_void();
    b.end_void();
    let ans = b.lget("i64x1", 2);
    let _ = b.end_val("i64x1", &ans);

    // func 2: main() -> i64
    b.func(2, "i64x1");
    let lim = b.const_i64(i64::from(limit));
    let c2 = b.const_i64(2);
    b.lset("i64x1", 0, &lim);
    b.lset("i64x1", 1, &c2); // p = 2
    b.block();
    b.loop_open();
    let p = b.lget("i64x1", 1);
    let pp = b.bin("Mul", "i64x1", &p, &p);
    let l = b.lget("i64x1", 0);
    let gt = b.bin("CmpGt", "maskx1", &pp, &l);
    b.br_if(&gt, 1);
    let flag = b.reg_load(0, "u8x1", &p);
    let u0 = b.const_u8(0);
    let is_prime = b.bin("CmpEq", "maskx1", &flag, &u0);
    b.if_open(&is_prime);
    let _ = b.call(0, "i64x1", &[&p, &l]);
    b.end_void();
    let c1 = b.const_i64(1);
    let pnext = b.bin("Add", "i64x1", &p, &c1);
    b.lset("i64x1", 1, &pnext);
    b.br(0);
    b.end_void();
    b.end_void();
    let l2 = b.lget("i64x1", 0);
    let count = b.call(1, "i64x1", &[&l2]);
    let _ = b.end_val("i64x1", &count);

    b.finish()
}

#[expect(
    clippy::too_many_lines,
    reason = "complete FIPS 180-4 SHA-256 implementation emitted as modular .rls"
)]
fn build_sha256_rls() -> String {
    const K: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];
    const H0: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];

    let mut b = RlsBuilder::new("DIVERGE");
    b.add_region(0, 0, 65_536, "u8", "input"); // message
    b.add_region(1, 32, 32, "u8", "output"); // 32-byte digest
    b.add_region(2, 64, 64, "i64", "plain"); // K[0..64]
    b.add_region(3, 64, 64, "i64", "plain"); // W[0..64]
    b.add_region(4, 8, 8, "i64", "plain"); // H[0..8]
    b.add_region(5, 0, 65_664, "u8", "plain"); // padded message

    // func 0: rotr32(x: i64, n: i64) -> i64
    b.add_func_sig(0, &["i64", "i64"], "i64");
    // func 1: init_k_and_h() -> void
    b.add_func_sig(1, &[], "void");
    // func 2: compress_block(block_off: i64) -> void
    b.add_func_sig(2, &["i64"], "void");
    // func 3 (entry): main() -> i64
    b.add_func_sig(3, &[], "i64");

    // --- func 0: rotr32(x: 0, n: 1) -> i64 ---
    b.func(0, "i64x1");
    let x = b.lget("i64x1", 0);
    let n = b.lget("i64x1", 1);
    let c32 = b.const_i64(32);
    let mask32 = b.const_i64(0xFFFF_FFFF);
    let xm = b.bin("BitAnd", "i64x1", &x, &mask32);
    let lo = b.bin("ShrU", "i64x1", &xm, &n);
    let lsh = b.bin("Sub", "i64x1", &c32, &n);
    let hi = b.bin("Shl", "i64x1", &xm, &lsh);
    let comb = b.bin("BitOr", "i64x1", &lo, &hi);
    let res = b.bin("BitAnd", "i64x1", &comb, &mask32);
    let _ = b.end_val("i64x1", &res);

    // --- func 1: init_k_and_h() -> void ---
    b.func(1, "i64x1");
    for (idx, &kv) in K.iter().enumerate() {
        let ci = b.const_i64(i64::try_from(idx).unwrap_or(0));
        let cv = b.const_i64(i64::from(kv));
        b.reg_store(2, "i64x1", &ci, &cv);
    }
    for (idx, &hv) in H0.iter().enumerate() {
        let ci = b.const_i64(i64::try_from(idx).unwrap_or(0));
        let cv = b.const_i64(i64::from(hv));
        b.reg_store(4, "i64x1", &ci, &cv);
    }
    b.end_void();

    // --- func 2: compress_block(block_off: 0) -> void ---
    // locals: 0 = block_off, 1 = i, 2..9 = a..h, 10 = temp1, 11 = temp2
    b.func(2, "i64x1");
    let mask32 = b.const_i64(0xFFFF_FFFF);
    b.lset("i64x1", 12, &mask32); // local 12 = 0xFFFF_FFFF

    // Build W[0..16] from big-endian bytes in region 5
    let z0 = b.const_i64(0);
    b.lset("i64x1", 1, &z0);
    b.block();
    b.loop_open();
    let i = b.lget("i64x1", 1);
    let c16 = b.const_i64(16);
    let lt = b.bin("CmpLt", "maskx1", &i, &c16);
    let done = b.mask_not(&lt);
    b.br_if(&done, 1);

    let boff = b.lget("i64x1", 0);
    let c4 = b.const_i64(4);
    let i4 = b.bin("Mul", "i64x1", &i, &c4);
    let p0 = b.bin("Add", "i64x1", &boff, &i4);
    let c1 = b.const_i64(1);
    let c2 = b.const_i64(2);
    let c3 = b.const_i64(3);
    let p1 = b.bin("Add", "i64x1", &p0, &c1);
    let p2 = b.bin("Add", "i64x1", &p0, &c2);
    let p3 = b.bin("Add", "i64x1", &p0, &c3);
    let b0u = b.reg_load(5, "u8x1", &p0);
    let b1u = b.reg_load(5, "u8x1", &p1);
    let b2u = b.reg_load(5, "u8x1", &p2);
    let b3u = b.reg_load(5, "u8x1", &p3);
    let b0 = b.cast("i64x1", &b0u);
    let b1 = b.cast("i64x1", &b1u);
    let b2 = b.cast("i64x1", &b2u);
    let b3 = b.cast("i64x1", &b3u);
    let s24 = b.const_i64(24);
    let s16 = b.const_i64(16);
    let s8 = b.const_i64(8);
    let w0 = b.bin("Shl", "i64x1", &b0, &s24);
    let w1 = b.bin("Shl", "i64x1", &b1, &s16);
    let w2 = b.bin("Shl", "i64x1", &b2, &s8);
    let w01 = b.bin("BitOr", "i64x1", &w0, &w1);
    let w012 = b.bin("BitOr", "i64x1", &w01, &w2);
    let word = b.bin("BitOr", "i64x1", &w012, &b3);
    b.reg_store(3, "i64x1", &i, &word);

    let inext = b.bin("Add", "i64x1", &i, &c1);
    b.lset("i64x1", 1, &inext);
    b.br(0);
    b.end_void();
    b.end_void();

    // Expand W[16..64]
    let c16s = b.const_i64(16);
    b.lset("i64x1", 1, &c16s);
    b.block();
    b.loop_open();
    let i = b.lget("i64x1", 1);
    let c64 = b.const_i64(64);
    let lt = b.bin("CmpLt", "maskx1", &i, &c64);
    let done = b.mask_not(&lt);
    b.br_if(&done, 1);

    let c15 = b.const_i64(15);
    let i15 = b.bin("Sub", "i64x1", &i, &c15);
    let w15 = b.reg_load(3, "i64x1", &i15);
    let c7 = b.const_i64(7);
    let c18 = b.const_i64(18);
    let c3 = b.const_i64(3);
    let r7 = b.call(0, "i64x1", &[&w15, &c7]);
    let r18 = b.call(0, "i64x1", &[&w15, &c18]);
    let sh3 = b.bin("ShrU", "i64x1", &w15, &c3);
    let s0a = b.bin("BitXor", "i64x1", &r7, &r18);
    let s0 = b.bin("BitXor", "i64x1", &s0a, &sh3);

    let c2 = b.const_i64(2);
    let i2 = b.bin("Sub", "i64x1", &i, &c2);
    let w2 = b.reg_load(3, "i64x1", &i2);
    let c17 = b.const_i64(17);
    let c19 = b.const_i64(19);
    let c10 = b.const_i64(10);
    let r17 = b.call(0, "i64x1", &[&w2, &c17]);
    let r19 = b.call(0, "i64x1", &[&w2, &c19]);
    let sh10 = b.bin("ShrU", "i64x1", &w2, &c10);
    let s1a = b.bin("BitXor", "i64x1", &r17, &r19);
    let s1 = b.bin("BitXor", "i64x1", &s1a, &sh10);

    let c16k = b.const_i64(16);
    let i16 = b.bin("Sub", "i64x1", &i, &c16k);
    let w16 = b.reg_load(3, "i64x1", &i16);
    let i7 = b.bin("Sub", "i64x1", &i, &c7);
    let w7 = b.reg_load(3, "i64x1", &i7);

    let t1 = b.bin("Add", "i64x1", &w16, &s0);
    let t2 = b.bin("Add", "i64x1", &t1, &w7);
    let t3 = b.bin("Add", "i64x1", &t2, &s1);
    let m32 = b.lget("i64x1", 12);
    let wi = b.bin("BitAnd", "i64x1", &t3, &m32);
    b.reg_store(3, "i64x1", &i, &wi);

    let c1 = b.const_i64(1);
    let inext = b.bin("Add", "i64x1", &i, &c1);
    b.lset("i64x1", 1, &inext);
    b.br(0);
    b.end_void();
    b.end_void();

    // Load H[0..8] into locals 2..=9 (a..=h)
    for k in 0..8_u32 {
        let idx = b.const_i64(i64::from(k));
        let hv = b.reg_load(4, "i64x1", &idx);
        b.lset("i64x1", 2 + k, &hv);
    }

    // 64 compression rounds
    let z0r = b.const_i64(0);
    b.lset("i64x1", 1, &z0r);
    b.block();
    b.loop_open();
    let i = b.lget("i64x1", 1);
    let c64 = b.const_i64(64);
    let lt = b.bin("CmpLt", "maskx1", &i, &c64);
    let done = b.mask_not(&lt);
    b.br_if(&done, 1);

    let la = b.lget("i64x1", 2);
    let lb = b.lget("i64x1", 3);
    let lc = b.lget("i64x1", 4);
    let ld = b.lget("i64x1", 5);
    let le = b.lget("i64x1", 6);
    let lf = b.lget("i64x1", 7);
    let lg = b.lget("i64x1", 8);
    let lh = b.lget("i64x1", 9);
    let m32 = b.lget("i64x1", 12);

    // S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25)
    let c6 = b.const_i64(6);
    let c11 = b.const_i64(11);
    let c25 = b.const_i64(25);
    let e6 = b.call(0, "i64x1", &[&le, &c6]);
    let e11 = b.call(0, "i64x1", &[&le, &c11]);
    let e25 = b.call(0, "i64x1", &[&le, &c25]);
    let s1x = b.bin("BitXor", "i64x1", &e6, &e11);
    let s1 = b.bin("BitXor", "i64x1", &s1x, &e25);

    // ch = (e & f) ^ ((e ^ 0xFFFF_FFFF) & g)
    let ef = b.bin("BitAnd", "i64x1", &le, &lf);
    let ne = b.bin("BitXor", "i64x1", &le, &m32);
    let neg = b.bin("BitAnd", "i64x1", &ne, &lg);
    let ch = b.bin("BitXor", "i64x1", &ef, &neg);

    let ki = b.reg_load(2, "i64x1", &i);
    let wi = b.reg_load(3, "i64x1", &i);
    let t1a = b.bin("Add", "i64x1", &lh, &s1);
    let t1b = b.bin("Add", "i64x1", &t1a, &ch);
    let t1c = b.bin("Add", "i64x1", &t1b, &ki);
    let t1d = b.bin("Add", "i64x1", &t1c, &wi);
    let temp1 = b.bin("BitAnd", "i64x1", &t1d, &m32);
    b.lset("i64x1", 10, &temp1);

    // S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22)
    let c2 = b.const_i64(2);
    let c13 = b.const_i64(13);
    let c22 = b.const_i64(22);
    let a2 = b.call(0, "i64x1", &[&la, &c2]);
    let a13 = b.call(0, "i64x1", &[&la, &c13]);
    let a22 = b.call(0, "i64x1", &[&la, &c22]);
    let s0x = b.bin("BitXor", "i64x1", &a2, &a13);
    let s0 = b.bin("BitXor", "i64x1", &s0x, &a22);

    // maj = (a & b) ^ (a & c) ^ (b & c)
    let ab = b.bin("BitAnd", "i64x1", &la, &lb);
    let ac = b.bin("BitAnd", "i64x1", &la, &lc);
    let bc = b.bin("BitAnd", "i64x1", &lb, &lc);
    let m1 = b.bin("BitXor", "i64x1", &ab, &ac);
    let maj = b.bin("BitXor", "i64x1", &m1, &bc);

    let t2a = b.bin("Add", "i64x1", &s0, &maj);
    let temp2 = b.bin("BitAnd", "i64x1", &t2a, &m32);
    b.lset("i64x1", 11, &temp2);

    // Update a..h
    let t1_v = b.lget("i64x1", 10);
    let t2_v = b.lget("i64x1", 11);
    let new_e_raw = b.bin("Add", "i64x1", &ld, &t1_v);
    let new_e = b.bin("BitAnd", "i64x1", &new_e_raw, &m32);
    let new_a_raw = b.bin("Add", "i64x1", &t1_v, &t2_v);
    let new_a = b.bin("BitAnd", "i64x1", &new_a_raw, &m32);

    b.lset("i64x1", 9, &lg);
    b.lset("i64x1", 8, &lf);
    b.lset("i64x1", 7, &le);
    b.lset("i64x1", 6, &new_e);
    b.lset("i64x1", 5, &lc);
    b.lset("i64x1", 4, &lb);
    b.lset("i64x1", 3, &la);
    b.lset("i64x1", 2, &new_a);

    let c1 = b.const_i64(1);
    let inext = b.bin("Add", "i64x1", &i, &c1);
    b.lset("i64x1", 1, &inext);
    b.br(0);
    b.end_void();
    b.end_void();

    // Add working vars back to H[0..8]
    let m32_end = b.lget("i64x1", 12);
    for k in 0..8_u32 {
        let idx = b.const_i64(i64::from(k));
        let cur_h = b.reg_load(4, "i64x1", &idx);
        let wk = b.lget("i64x1", 2 + k);
        let sum = b.bin("Add", "i64x1", &cur_h, &wk);
        let masked = b.bin("BitAnd", "i64x1", &sum, &m32_end);
        b.reg_store(4, "i64x1", &idx, &masked);
    }
    b.end_void();

    // --- func 3: main() -> i64 ---
    b.func(3, "i64x1");
    let _ = b.call(1, "i64x1", &[]); // init_k_and_h
    let msg_len = b.reg_len(0);
    b.lset("i64x1", 0, &msg_len);

    // padded_len = align_up_64(msg_len + 9)
    let c9 = b.const_i64(9);
    let c64 = b.const_i64(64);
    let base_len = b.bin("Add", "i64x1", &msg_len, &c9);
    let rem = b.bin("Rem", "i64x1", &base_len, &c64);
    let z0 = b.const_i64(0);
    let is_aligned = b.bin("CmpEq", "maskx1", &rem, &z0);
    b.if_open(&is_aligned);
    b.lset("i64x1", 1, &base_len);
    b.else_open();
    let pad_extra = b.bin("Sub", "i64x1", &c64, &rem);
    let padded = b.bin("Add", "i64x1", &base_len, &pad_extra);
    b.lset("i64x1", 1, &padded);
    b.end_void();

    let plen = b.lget("i64x1", 1);
    b.reg_new(5, "u8x1", &plen);

    // Copy msg[0..msg_len] to padded buffer
    b.lset("i64x1", 2, &z0);
    b.block();
    b.loop_open();
    let i = b.lget("i64x1", 2);
    let ml = b.lget("i64x1", 0);
    let lt = b.bin("CmpLt", "maskx1", &i, &ml);
    let done = b.mask_not(&lt);
    b.br_if(&done, 1);
    let byte = b.reg_load(0, "u8x1", &i);
    b.reg_store(5, "u8x1", &i, &byte);
    let c1 = b.const_i64(1);
    let inext = b.bin("Add", "i64x1", &i, &c1);
    b.lset("i64x1", 2, &inext);
    b.br(0);
    b.end_void();
    b.end_void();

    // Append 0x80 at msg_len
    let ml = b.lget("i64x1", 0);
    let b80 = b.const_u8(0x80);
    b.reg_store(5, "u8x1", &ml, &b80);

    // Write 64-bit big-endian bit_len at plen - 8 .. plen
    let c8 = b.const_i64(8);
    let bit_len = b.bin("Mul", "i64x1", &ml, &c8);
    b.lset("i64x1", 3, &bit_len);
    for byte_i in 0..8_i64 {
        let shift = b.const_i64((7 - byte_i) * 8);
        let bl = b.lget("i64x1", 3);
        let sh = b.bin("ShrU", "i64x1", &bl, &shift);
        let ub = b.cast("u8x1", &sh);
        let pl = b.lget("i64x1", 1);
        let off = b.const_i64(8 - byte_i);
        let pos = b.bin("Sub", "i64x1", &pl, &off);
        b.reg_store(5, "u8x1", &pos, &ub);
    }

    // Compress each 64-byte block
    b.lset("i64x1", 2, &z0);
    b.block();
    b.loop_open();
    let off = b.lget("i64x1", 2);
    let pl = b.lget("i64x1", 1);
    let lt = b.bin("CmpLt", "maskx1", &off, &pl);
    let done = b.mask_not(&lt);
    b.br_if(&done, 1);
    let _ = b.call(2, "i64x1", &[&off]);
    let c64b = b.const_i64(64);
    let off_next = b.bin("Add", "i64x1", &off, &c64b);
    b.lset("i64x1", 2, &off_next);
    b.br(0);
    b.end_void();
    b.end_void();

    // Emit H[0..8] as 32 big-endian bytes into output region 1
    for k in 0..8_i64 {
        let ki = b.const_i64(k);
        let word = b.reg_load(4, "i64x1", &ki);
        for j in 0..4_i64 {
            let shift = b.const_i64((3 - j) * 8);
            let sh = b.bin("ShrU", "i64x1", &word, &shift);
            let ub = b.cast("u8x1", &sh);
            let dst = b.const_i64(k * 4 + j);
            b.reg_store(1, "u8x1", &dst, &ub);
        }
    }

    let c32 = b.const_i64(32);
    let _ = b.end_val("i64x1", &c32);
    b.finish()
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[test]
fn test_tc_core3_exit_programs_suite() -> Result<(), String> {
    // 1. Recursive fib(20) == 6765
    let fib_prog = assemble_rls(&build_fib_rls(20)).map_err(|e| format!("fib assemble: {e:?}"))?;
    verify_program(&fib_prog).map_err(|e| format!("fib verify: {e:?}"))?;
    verify_rls_roundtrip(&fib_prog, "fib20")?;
    let (fib_scalar, _) = run_tier0_and_aot_scalar(&fib_prog, "fib20")?;
    assert_eq!(fib_scalar.as_i64(), Some(6765));

    // 2. In-place quicksort on I64 region (negatives, duplicates, extremes)
    let unsorted: Vec<i64> = vec![
        42, -17, 0, 999_999, -17, 5, 1, 100, -500, 42, 7, -1, 88, 3, -999_999, 12,
    ];
    let mut expected_sorted = unsorted.clone();
    expected_sorted.sort_unstable();
    let mut expected_bytes = Vec::with_capacity(expected_sorted.len() * 8);
    for &v in &expected_sorted {
        expected_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let qsort_prog = assemble_rls(&build_quicksort_rls(&unsorted))
        .map_err(|e| format!("qsort assemble: {e:?}"))?;
    verify_program(&qsort_prog).map_err(|e| format!("qsort verify: {e:?}"))?;
    verify_rls_roundtrip(&qsort_prog, "quicksort")?;
    let (qsort_out, _) = run_tier0_and_aot_io(&qsort_prog, &[], "quicksort")?;
    assert_eq!(qsort_out, expected_bytes);

    // 3. Prime sieve(N = 10_000) == 1229
    let sieve_prog =
        assemble_rls(&build_sieve_rls(10_000)).map_err(|e| format!("sieve assemble: {e:?}"))?;
    verify_program(&sieve_prog).map_err(|e| format!("sieve verify: {e:?}"))?;
    verify_rls_roundtrip(&sieve_prog, "sieve_10000")?;
    let (sieve_scalar, _) = run_tier0_and_aot_scalar(&sieve_prog, "sieve_10000")?;
    assert_eq!(sieve_scalar.as_i64(), Some(1229));

    // 4. SHA-256 against NIST FIPS 180-4 reference test vectors
    let sha_prog =
        assemble_rls(&build_sha256_rls()).map_err(|e| format!("sha256 assemble: {e:?}"))?;
    verify_program(&sha_prog).map_err(|e| format!("sha256 verify: {e:?}"))?;
    verify_rls_roundtrip(&sha_prog, "sha256")?;

    let vectors: &[(&str, &[u8], &str)] = &[
        (
            "empty",
            b"",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        (
            "abc",
            b"abc",
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        ),
        (
            "two_block_56",
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        ),
    ];
    for &(name, msg, expected_hex) in vectors {
        let (digest, _) = run_tier0_and_aot_io(&sha_prog, msg, &format!("sha256_{name}"))?;
        assert_eq!(digest.len(), 32);
        assert_eq!(
            bytes_to_hex(&digest),
            expected_hex,
            "sha256 vector '{name}' mismatch"
        );
    }

    Ok(())
}

// =========================================================================
// AC-5: Function table >255 functions & >65,536 total nodes + layout rejections
// =========================================================================
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "comprehensive scale test (>255 funcs, >65,536 nodes) plus all 11 AC-5 verifier rejection checks"
)]
fn test_tc_core3_function_table_limits_and_layout() -> Result<(), String> {
    // 1. Positive scale test: 260 functions (> 255) and > 65,536 total nodes (each function ~260 nodes).
    let num_helpers: u16 = 259; // ids 0..=258; entry is id 259 (total = 260 functions)
    let pad_consts: usize = 255; // each helper has 1 Func + 255 ConstI64 + 1 End = 257 nodes -> 259 * 257 = 66,563 > 65,536!
    let mut proofs: Vec<ProofEntry> = Vec::with_capacity(usize::from(num_helpers) + 1);
    let mut nodes: Vec<NodeFrame> = Vec::with_capacity(70_000);

    for fid in 0..num_helpers {
        proofs.push(ProofEntry::func_sig(fid, &[], Some(ElemType::I64)));
        nodes.push(NodeFrame {
            opcode: Opcode::Func,
            type_lane: TypeLane::I64_SCALAR,
            schedule: ScheduleHint::UNFUSED,
            proof_idx: 0,
            token_op: TokenOp::PURE,
            in_a: u32::from(fid),
            in_b: 0,
        });
        for k in 0..pad_consts {
            let val = if k + 1 == pad_consts {
                i64::from(fid) + 1
            } else {
                i64::try_from(k).unwrap_or(0)
            };
            nodes.push(NodeFrame::const_i64(val, ScheduleHint::UNFUSED));
        }
        // rel index of last ConstI64 is `pad_consts`
        let last_rel = u32::try_from(pad_consts).unwrap_or(1);
        nodes.push(NodeFrame {
            opcode: Opcode::End,
            type_lane: TypeLane::I64_SCALAR,
            schedule: ScheduleHint::UNFUSED,
            proof_idx: 0,
            token_op: TokenOp::PURE,
            in_a: last_rel,
            in_b: 1,
        });
    }

    // Entry function (id = 259): calls helpers 0, 128, 255, 256, 258 and sums their return values.
    let entry_id = num_helpers;
    proofs.push(ProofEntry::func_sig(entry_id, &[], Some(ElemType::I64)));
    let entry_start = u32::try_from(nodes.len()).unwrap_or(0);
    nodes.push(NodeFrame {
        opcode: Opcode::Func,
        type_lane: TypeLane::I64_SCALAR,
        schedule: ScheduleHint::UNFUSED,
        proof_idx: 0,
        token_op: TokenOp::PURE,
        in_a: u32::from(entry_id),
        in_b: 0,
    });
    let call_targets: [u16; 5] = [0, 128, 255, 256, 258];
    let no_args = NodeFrame::encode_u16_pair(CALL_ARG_SENTINEL, CALL_ARG_SENTINEL);
    let mut acc_rel: Option<u32> = None;
    for &tgt in &call_targets {
        let call_rel = u32::try_from(nodes.len()).unwrap_or(0) - entry_start;
        nodes.push(NodeFrame {
            opcode: Opcode::Call,
            type_lane: TypeLane::I64_SCALAR,
            schedule: ScheduleHint::UNFUSED,
            proof_idx: 0,
            token_op: TokenOp::PURE,
            in_a: u32::from(tgt),
            in_b: no_args,
        });
        acc_rel = if let Some(prev) = acc_rel {
            let add_rel = u32::try_from(nodes.len()).unwrap_or(0) - entry_start;
            nodes.push(NodeFrame {
                opcode: Opcode::Add,
                type_lane: TypeLane::I64_SCALAR,
                schedule: ScheduleHint::UNFUSED,
                proof_idx: 0,
                token_op: TokenOp::PURE,
                in_a: prev,
                in_b: call_rel,
            });
            Some(add_rel)
        } else {
            Some(call_rel)
        };
    }
    nodes.push(NodeFrame {
        opcode: Opcode::End,
        type_lane: TypeLane::I64_SCALAR,
        schedule: ScheduleHint::UNFUSED,
        proof_idx: 0,
        token_op: TokenOp::PURE,
        in_a: acc_rel.unwrap_or(1),
        in_b: 1,
    });

    assert!(proofs.len() > 255, "must have > 255 FuncSig proofs");
    assert!(nodes.len() > 65_536, "must have > 65,536 total nodes");

    let big_prog = RulProgram {
        capability_mask: CAP_NONE,
        arenas: Vec::new(),
        proofs,
        nodes,
        output_node: entry_start,
    };
    verify_program(&big_prog).map_err(|e| format!("big_prog verify: {e:?}"))?;
    let expected_sum: i64 = call_targets.iter().map(|&t| i64::from(t) + 1).sum();
    let (big_scalar, _) = run_tier0_and_aot_scalar(&big_prog, "scale_260funcs_67knodes")?;
    assert_eq!(big_scalar.as_i64(), Some(expected_sum));

    // Helper for constructing a valid minimal 1-function program to mutate for negative tests.
    let valid_min = || RulProgram {
        capability_mask: CAP_NONE,
        arenas: Vec::new(),
        proofs: vec![ProofEntry::func_sig(0, &[], Some(ElemType::I64))],
        nodes: vec![
            NodeFrame {
                opcode: Opcode::Func,
                type_lane: TypeLane::I64_SCALAR,
                schedule: ScheduleHint::UNFUSED,
                proof_idx: 0,
                token_op: TokenOp::PURE,
                in_a: 0,
                in_b: 0,
            },
            NodeFrame::const_i64(42, ScheduleHint::UNFUSED),
            NodeFrame {
                opcode: Opcode::End,
                type_lane: TypeLane::I64_SCALAR,
                schedule: ScheduleHint::UNFUSED,
                proof_idx: 0,
                token_op: TokenOp::PURE,
                in_a: 1,
                in_b: 1,
            },
        ],
        output_node: 0,
    };

    // Case 1: Func with proof_idx != 0 -> rejected with InvalidFuncSig
    let mut p = valid_min();
    if let Some(f) = p.nodes.first_mut() {
        f.proof_idx = 1;
    }
    assert!(
        matches!(
            verify_program(&p),
            Err(crate::verifier::VerifyDiagnostic {
                kind: VerifyErrorKind::InvalidFuncSig { func_id: 0, .. },
                ..
            })
        ),
        "Func with proof_idx != 0 must fail with InvalidFuncSig"
    );

    // Case 2a: Missing FuncSig for func_id -> rejected with UnknownFunction
    let mut p = valid_min();
    p.proofs.clear();
    assert!(
        matches!(
            verify_program(&p),
            Err(crate::verifier::VerifyDiagnostic {
                kind: VerifyErrorKind::UnknownFunction(0),
                ..
            })
        ),
        "missing FuncSig must fail with UnknownFunction"
    );

    // Case 2b: Extra FuncSig with no Func -> rejected with InvalidFuncSig
    let mut p = valid_min();
    p.proofs
        .push(ProofEntry::func_sig(1, &[], Some(ElemType::I64)));
    assert!(
        matches!(
            verify_program(&p),
            Err(crate::verifier::VerifyDiagnostic {
                kind: VerifyErrorKind::InvalidFuncSig { func_id: 1, .. },
                ..
            })
        ),
        "extra FuncSig without Func must fail with InvalidFuncSig"
    );

    // Case 3: Duplicate FuncSig for func_id -> rejected with InvalidFuncSig
    let mut p = valid_min();
    p.proofs
        .push(ProofEntry::func_sig(0, &[], Some(ElemType::I64)));
    assert!(
        matches!(
            verify_program(&p),
            Err(crate::verifier::VerifyDiagnostic {
                kind: VerifyErrorKind::InvalidFuncSig { func_id: 0, .. },
                ..
            })
        ),
        "duplicate FuncSig must fail with InvalidFuncSig"
    );

    // Case 4a: Non-dense func_id (func_id = 1 without 0) -> rejected with InvalidFuncSig
    let mut p = valid_min();
    p.proofs = vec![ProofEntry::func_sig(1, &[], Some(ElemType::I64))];
    if let Some(f) = p.nodes.first_mut() {
        f.in_a = 1;
    }
    assert!(
        matches!(
            verify_program(&p),
            Err(crate::verifier::VerifyDiagnostic {
                kind: VerifyErrorKind::InvalidFuncSig { func_id: 1, .. },
                ..
            })
        ),
        "non-dense func_id must fail with InvalidFuncSig"
    );

    // Case 4b: Out-of-order Func declarations (1 then 0) -> rejected with InvalidFuncSig
    let p_ooo = RulProgram {
        capability_mask: CAP_NONE,
        arenas: Vec::new(),
        proofs: vec![
            ProofEntry::func_sig(0, &[], Some(ElemType::I64)),
            ProofEntry::func_sig(1, &[], Some(ElemType::I64)),
        ],
        nodes: vec![
            NodeFrame {
                opcode: Opcode::Func,
                type_lane: TypeLane::I64_SCALAR,
                schedule: ScheduleHint::UNFUSED,
                proof_idx: 0,
                token_op: TokenOp::PURE,
                in_a: 1,
                in_b: 0,
            },
            NodeFrame::const_i64(1, ScheduleHint::UNFUSED),
            NodeFrame {
                opcode: Opcode::End,
                type_lane: TypeLane::I64_SCALAR,
                schedule: ScheduleHint::UNFUSED,
                proof_idx: 0,
                token_op: TokenOp::PURE,
                in_a: 1,
                in_b: 1,
            },
            NodeFrame {
                opcode: Opcode::Func,
                type_lane: TypeLane::I64_SCALAR,
                schedule: ScheduleHint::UNFUSED,
                proof_idx: 0,
                token_op: TokenOp::PURE,
                in_a: 0,
                in_b: 0,
            },
            NodeFrame::const_i64(2, ScheduleHint::UNFUSED),
            NodeFrame {
                opcode: Opcode::End,
                type_lane: TypeLane::I64_SCALAR,
                schedule: ScheduleHint::UNFUSED,
                proof_idx: 0,
                token_op: TokenOp::PURE,
                in_a: 1,
                in_b: 1,
            },
        ],
        output_node: 0,
    };
    assert!(
        matches!(
            verify_program(&p_ooo),
            Err(crate::verifier::VerifyDiagnostic {
                kind: VerifyErrorKind::InvalidFuncSig { func_id: 1, .. },
                ..
            })
        ),
        "out-of-order Func must fail with InvalidFuncSig"
    );

    // Case 5: Func nested inside open Func scope -> rejected with InvalidControlFlow
    let p_nested = RulProgram {
        capability_mask: CAP_NONE,
        arenas: Vec::new(),
        proofs: vec![
            ProofEntry::func_sig(0, &[], Some(ElemType::I64)),
            ProofEntry::func_sig(1, &[], Some(ElemType::I64)),
        ],
        nodes: vec![
            NodeFrame {
                opcode: Opcode::Func,
                type_lane: TypeLane::I64_SCALAR,
                schedule: ScheduleHint::UNFUSED,
                proof_idx: 0,
                token_op: TokenOp::PURE,
                in_a: 0,
                in_b: 0,
            },
            NodeFrame {
                opcode: Opcode::Func,
                type_lane: TypeLane::I64_SCALAR,
                schedule: ScheduleHint::UNFUSED,
                proof_idx: 0,
                token_op: TokenOp::PURE,
                in_a: 1,
                in_b: 0,
            },
            NodeFrame::const_i64(1, ScheduleHint::UNFUSED),
            NodeFrame {
                opcode: Opcode::End,
                type_lane: TypeLane::I64_SCALAR,
                schedule: ScheduleHint::UNFUSED,
                proof_idx: 0,
                token_op: TokenOp::PURE,
                in_a: 1,
                in_b: 1,
            },
            NodeFrame::const_i64(2, ScheduleHint::UNFUSED),
            NodeFrame {
                opcode: Opcode::End,
                type_lane: TypeLane::I64_SCALAR,
                schedule: ScheduleHint::UNFUSED,
                proof_idx: 0,
                token_op: TokenOp::PURE,
                in_a: 4,
                in_b: 1,
            },
        ],
        output_node: 0,
    };
    assert!(
        matches!(
            verify_program(&p_nested),
            Err(crate::verifier::VerifyDiagnostic {
                kind: VerifyErrorKind::InvalidControlFlow(_),
                ..
            })
        ),
        "nested Func must fail with InvalidControlFlow"
    );

    // Case 6: Node outside Func scope (before and after Func) -> rejected with InvalidControlFlow
    let mut p_before = valid_min();
    p_before
        .nodes
        .insert(0, NodeFrame::const_i64(1, ScheduleHint::UNFUSED));
    p_before.output_node = 1;
    assert!(
        matches!(
            verify_program(&p_before),
            Err(crate::verifier::VerifyDiagnostic {
                kind: VerifyErrorKind::InvalidControlFlow(_),
                ..
            })
        ),
        "node before Func must fail with InvalidControlFlow"
    );
    let mut p_after = valid_min();
    p_after
        .nodes
        .push(NodeFrame::const_i64(1, ScheduleHint::UNFUSED));
    assert!(
        matches!(
            verify_program(&p_after),
            Err(crate::verifier::VerifyDiagnostic {
                kind: VerifyErrorKind::InvalidControlFlow(_),
                ..
            })
        ),
        "node after Func must fail with InvalidControlFlow"
    );

    // Case 7: Operand referring outside enclosing Func scope (rel 0 -> InvalidControlFlow, rel >= len -> NonTopologicalEdge)
    let mut p_rel0 = valid_min();
    if let Some(end) = p_rel0.nodes.get_mut(2) {
        end.in_a = 0; // relative 0 names Func header
        end.in_b = 1;
    }
    assert!(
        matches!(
            verify_program(&p_rel0),
            Err(crate::verifier::VerifyDiagnostic {
                kind: VerifyErrorKind::InvalidControlFlow(_),
                ..
            })
        ),
        "relative operand 0 must fail with InvalidControlFlow"
    );
    let mut p_oob = valid_min();
    if let Some(end) = p_oob.nodes.get_mut(2) {
        end.in_a = 99;
        end.in_b = 1;
    }
    assert!(
        matches!(
            verify_program(&p_oob),
            Err(crate::verifier::VerifyDiagnostic {
                kind: VerifyErrorKind::NonTopologicalEdge { input_node: 99 },
                ..
            })
        ),
        "out-of-scope relative operand must fail with NonTopologicalEdge"
    );

    // Case 8: Function scope > MAX_FUNC_NODES (65,536 nodes) -> rejected with FunctionTooLarge;
    //         exact MAX_FUNC_NODES (65,535 nodes) -> accepted!
    let build_single_func_len = |total_len: usize| -> RulProgram {
        let mut fn_nodes = Vec::with_capacity(total_len);
        fn_nodes.push(NodeFrame {
            opcode: Opcode::Func,
            type_lane: TypeLane::I64_SCALAR,
            schedule: ScheduleHint::UNFUSED,
            proof_idx: 0,
            token_op: TokenOp::PURE,
            in_a: 0,
            in_b: 0,
        });
        for _ in 0..(total_len - 2) {
            fn_nodes.push(NodeFrame::const_i64(7, ScheduleHint::UNFUSED));
        }
        fn_nodes.push(NodeFrame {
            opcode: Opcode::End,
            type_lane: TypeLane::I64_SCALAR,
            schedule: ScheduleHint::UNFUSED,
            proof_idx: 0,
            token_op: TokenOp::PURE,
            in_a: 1,
            in_b: 1,
        });
        RulProgram {
            capability_mask: CAP_NONE,
            arenas: Vec::new(),
            proofs: vec![ProofEntry::func_sig(0, &[], Some(ElemType::I64))],
            nodes: fn_nodes,
            output_node: 0,
        }
    };
    let ok_max = build_single_func_len(MAX_FUNC_NODES);
    verify_program(&ok_max).map_err(|e| format!("MAX_FUNC_NODES verify: {e:?}"))?;
    let too_large = build_single_func_len(MAX_FUNC_NODES + 1);
    assert!(
        matches!(
            verify_program(&too_large),
            Err(crate::verifier::VerifyDiagnostic {
                kind: VerifyErrorKind::FunctionTooLarge { .. },
                ..
            })
        ),
        "function scope > 65,535 nodes must fail with FunctionTooLarge"
    );

    // Case 9: Entry function with nonzero arity or void return -> rejected with InvalidFuncSig
    let mut p_nonzero = valid_min();
    p_nonzero.proofs = vec![ProofEntry::func_sig(
        0,
        &[ElemType::I64],
        Some(ElemType::I64),
    )];
    assert!(
        matches!(
            verify_program(&p_nonzero),
            Err(crate::verifier::VerifyDiagnostic {
                kind: VerifyErrorKind::InvalidFuncSig { func_id: 0, .. },
                ..
            })
        ),
        "entry function with arity > 0 must fail with InvalidFuncSig"
    );
    let p_void_entry = RulProgram {
        capability_mask: CAP_NONE,
        arenas: Vec::new(),
        proofs: vec![ProofEntry::func_sig(0, &[], None)],
        nodes: vec![
            NodeFrame {
                opcode: Opcode::Func,
                type_lane: TypeLane::I64_SCALAR,
                schedule: ScheduleHint::UNFUSED,
                proof_idx: 0,
                token_op: TokenOp::PURE,
                in_a: 0,
                in_b: 0,
            },
            NodeFrame {
                opcode: Opcode::End,
                type_lane: TypeLane::I64_SCALAR,
                schedule: ScheduleHint::UNFUSED,
                proof_idx: 0,
                token_op: TokenOp::PURE,
                in_a: 0,
                in_b: 0,
            },
        ],
        output_node: 0,
    };
    assert!(
        matches!(
            verify_program(&p_void_entry),
            Err(crate::verifier::VerifyDiagnostic {
                kind: VerifyErrorKind::InvalidFuncSig { func_id: 0, .. },
                ..
            })
        ),
        "void entry function must fail with InvalidFuncSig"
    );

    // Case 10: header.output_node not pointing to entry Func node -> rejected with InvalidOutputNode
    let mut p_bad_out = valid_min();
    p_bad_out.output_node = 1;
    assert!(
        matches!(
            verify_program(&p_bad_out),
            Err(crate::verifier::VerifyDiagnostic {
                kind: VerifyErrorKind::InvalidOutputNode(1),
                ..
            })
        ),
        "output_node != entry Func must fail"
    );

    // Case 11: Non-pure token_op in a program containing Func -> rejected with InvalidControlFlow
    let mut p_tok = valid_min();
    if let Some(c) = p_tok.nodes.get_mut(1) {
        c.token_op = TokenOp::read_lease(0, 1);
    }
    assert!(
        matches!(
            verify_program(&p_tok),
            Err(crate::verifier::VerifyDiagnostic {
                kind: VerifyErrorKind::InvalidControlFlow(_),
                ..
            })
        ),
        "non-pure token_op in function program must fail with InvalidControlFlow"
    );

    Ok(())
}

// =========================================================================
// AC-6: Host-independent stack budget (`MAX_STACK_UNITS = 1 << 20`, `ulimit -s 64`)
// =========================================================================
#[test]
fn test_tc_core3_stack_budget_is_host_independent() -> Result<(), String> {
    // Build a recursive function `big_rec(n: i64) -> i64` with `pad_nodes = 1079` so:
    //   `main_nodes = 4` -> `main_units = 76`
    //   `big_rec_nodes = 14 + 1079 = 1093` -> `big_units = 1165`
    //   `main_units + 900 * big_units = 76 + 1_048_500 = 1_048_576 == MAX_STACK_UNITS` EXACTLY!
    // This tests the exact `== MAX_STACK_UNITS` boundary (must succeed) vs `> MAX_STACK_UNITS` (must exit 75)
    // well below `MAX_CALL_DEPTH = 4096` and under `ulimit -s 64` (64 KiB host stack).
    let pad_nodes = 1079usize;
    let build_big_rec = |n_val: i64| -> Result<(RulProgram, u32, u32), String> {
        let mut b = RlsBuilder::new("DIVERGE");
        b.add_func_sig(0, &["i64"], "i64");
        b.add_func_sig(1, &[], "i64");

        // func 0: big_rec(n)
        b.func(0, "i64x1");
        let n = b.lget("i64x1", 0);
        let c0 = b.const_i64(0);
        let eq0 = b.bin("CmpEq", "maskx1", &n, &c0);
        b.if_open(&eq0);
        let z = b.const_i64(0);
        b.ret_val("i64x1", &z);
        b.end_void();
        for k in 0..pad_nodes {
            let _ = b.const_i64(i64::try_from(k).unwrap_or(0));
        }
        let n2 = b.lget("i64x1", 0);
        let c1 = b.const_i64(1);
        let nm1 = b.bin("Sub", "i64x1", &n2, &c1);
        let sub = b.call(0, "i64x1", &[&nm1]);
        let ans = b.bin("Add", "i64x1", &sub, &c1);
        let _ = b.end_val("i64x1", &ans);

        // func 1: main()
        b.func(1, "i64x1");
        let nv = b.const_i64(n_val);
        let out = b.call(0, "i64x1", &[&nv]);
        let _ = b.end_val("i64x1", &out);

        let prog = assemble_rls(&b.finish()).map_err(|e| format!("big_rec assemble: {e:?}"))?;
        verify_program(&prog).map_err(|e| format!("big_rec verify: {e:?}"))?;

        let entry_start = prog.output_node as usize;
        let big_rec_nodes = entry_start; // func 0 spans 0..entry_start
        let main_nodes = prog.nodes.len() - entry_start;
        Ok((
            prog,
            func_frame_units(big_rec_nodes),
            func_frame_units(main_nodes),
        ))
    };

    let (_, big_units, main_units) = build_big_rec(0)?;
    let max_big_frames = (MAX_STACK_UNITS - main_units) / big_units;
    assert_eq!(
        main_units + max_big_frames * big_units,
        MAX_STACK_UNITS,
        "within-budget test case must land on exact MAX_STACK_UNITS boundary"
    );
    assert!(
        max_big_frames + 1 < MAX_CALL_DEPTH,
        "test must hit MAX_STACK_UNITS before MAX_CALL_DEPTH (got {max_big_frames})"
    );
    // `n = max_big_frames - 1` produces `max_big_frames` live activations of `big_rec` + 1 of `main`
    // (total units <= MAX_STACK_UNITS).
    let ok_n = i64::from(max_big_frames - 1);
    let over_n = i64::from(max_big_frames);

    // 1. Within budget: Tier 0 == AOT (default stack) == AOT (`ulimit -s 64`)
    let (ok_prog, _, _) = build_big_rec(ok_n)?;
    let t0_ok =
        execute_tier0_interpreter(&ok_prog, &[]).map_err(|e| format!("budget_ok tier0: {e:?}"))?;
    assert_eq!(
        t0_ok.output,
        RuntimeValue::TypedScalar(TypedScalar::i64(ok_n))
    );
    let exe_ok =
        std::env::temp_dir().join(format!("rul_tc_core3_{}_budget_ok", std::process::id()));
    compile_rl_to_executable(&ok_prog, &exe_ok).map_err(|e| format!("budget_ok compile: {e:?}"))?;

    let default_out = run_compiled_structured_executable(&exe_ok, &[])
        .map_err(|e| format!("budget_ok default aot: {e:?}"))?;
    let default_val = i64::from_le_bytes(
        default_out
            .as_slice()
            .try_into()
            .map_err(|_e| "expected 8 bytes".to_owned())?,
    );
    assert_eq!(default_val, ok_n);

    let limited_ok = Command::new("bash")
        .arg("-c")
        .arg("ulimit -s 64 && exec \"$1\"")
        .arg("_")
        .arg(&exe_ok)
        .output()
        .map_err(|e| format!("ulimit -s 64 ok exec: {e}"))?;
    let _ = std::fs::remove_file(&exe_ok);
    assert!(
        limited_ok.status.success(),
        "under ulimit -s 64, within-budget binary must exit 0, got {:?}",
        limited_ok.status
    );
    assert_eq!(
        limited_ok.stdout, default_out,
        "output under ulimit -s 64 must match default"
    );

    // 2. Exceeding budget by 1 frame: Tier 0 CallDepthExceeded == AOT exit 75 (both default and `ulimit -s 64`)
    let (over_prog, _, _) = build_big_rec(over_n)?;
    let t0_over = execute_tier0_interpreter(&over_prog, &[]);
    assert!(
        matches!(t0_over, Err(ExecError::CallDepthExceeded { .. })),
        "Tier 0 expected CallDepthExceeded on stack budget overflow, got {t0_over:?}"
    );

    let exe_over =
        std::env::temp_dir().join(format!("rul_tc_core3_{}_budget_over", std::process::id()));
    compile_rl_to_executable(&over_prog, &exe_over)
        .map_err(|e| format!("budget_over compile: {e:?}"))?;

    let aot_over = run_compiled_structured_executable(&exe_over, &[]);
    assert!(
        matches!(aot_over, Err(AotError::StackOverflow)),
        "AOT default stack expected StackOverflow (75), got {aot_over:?}"
    );

    let limited_over = Command::new("bash")
        .arg("-c")
        .arg("ulimit -s 64 && exec \"$1\"")
        .arg("_")
        .arg(&exe_over)
        .output()
        .map_err(|e| format!("ulimit -s 64 over exec: {e}"))?;
    let _ = std::fs::remove_file(&exe_over);
    assert_eq!(
        limited_over.status.code(),
        Some(AOT_EXIT_STACK_OVERFLOW),
        "under ulimit -s 64, over-budget binary must fail closed with exit 75 (not SIGSEGV), got {:?}",
        limited_over.status
    );
    assert!(
        limited_over.stdout.is_empty(),
        "over-budget binary must emit zero stdout bytes"
    );

    Ok(())
}
