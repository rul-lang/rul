//! Milestone 1d (`v0-5-tc-core-4`) test suite:
//! - AC-1: Deterministic 500+-program differential torture suite (`Tier 0 == ELF64 AOT`, byte-identical
//!   recompilation, >16 live GPRs and >16 live XMMs across loops, `If`/`Else`, and `Call`).
//! - AC-2: Spilled recursive frames (>40 live `I64` values across recursive `Call`) staying strictly
//!   within the ISA stack-unit budget (`MAX_STACK_UNITS = 1 << 20`) under `ulimit -s 64`.
//! - AC-3 (`#[ignore]`): Performance gate comparing all five `v0.5` exit programs (`fib`, `quicksort`,
//!   `sieve`, `sha256`, `brainfuck`) against `rustc -O` references (`<= 2.0x` min wall time).
//! - AC-4 (`#[ignore]`): Linear $O(N)$ codegen latency gate (`codegen_ns <= 20_000 + 50 * node_count`,
//!   60,000-node time `<= 15x` 6,000-node time).

use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::aot::{
    compile_rl_to_executable, compile_to_elf64_bytes, run_compiled_structured_executable, AotError,
    AOT_EXIT_STACK_OVERFLOW,
};
use crate::bytecode::{func_frame_units, RulProgram, MAX_CALL_DEPTH, MAX_STACK_UNITS};
use crate::engine::{
    execute_tier0_interpreter, execute_tier0_interpreter_with_input, ExecError, RuntimeValue,
    TypedScalar,
};
use crate::transpiler::assemble_rls;
use crate::verifier::verify_program;

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_range(&mut self, upper: usize) -> usize {
        if upper == 0 {
            return 0;
        }
        usize::try_from(self.next_u64() % (upper as u64)).unwrap_or(0)
    }

    #[expect(
        clippy::cast_possible_wrap,
        reason = "deterministic signed integer sampling in differential test generator"
    )]
    fn next_small_i64(&mut self) -> i64 {
        (self.next_u64() % 201) as i64 - 100
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "deterministic finite f64 sampling in differential test generator"
    )]
    fn next_small_f64(&mut self) -> f64 {
        let raw = (self.next_u64() % 401) as i64 - 200;
        (raw as f64) * 0.25 + 0.125
    }
}

struct RlsGen {
    cap: String,
    proofs: Vec<String>,
    body: String,
    next_id: usize,
    last_func: Option<String>,
}

impl RlsGen {
    fn new(cap: &str) -> Self {
        Self {
            cap: cap.to_owned(),
            proofs: Vec::new(),
            body: String::new(),
            next_id: 0,
            last_func: None,
        }
    }

    fn fresh(&mut self, prefix: &str) -> String {
        let id = self.next_id;
        self.next_id += 1;
        format!("${prefix}_{id}")
    }

    fn add_block_sig(&mut self, result_lane: &str) -> usize {
        let idx = self.proofs.len() + 1;
        let witness_byte: u8 = match result_lane {
            "f64x1" => 0x11,
            "i64x1" => 0x21,
            "maskx1" => 0x31,
            "u8x1" => 0x51,
            _ => 0,
        };
        self.proofs.push(format!(
            "proof {idx} rule=block_sig target=0 lo=0 hi=0 witness={witness_byte}"
        ));
        idx
    }

    fn add_loop_bound(&mut self, bound_hi: u32) -> usize {
        let idx = self.proofs.len() + 1;
        self.proofs.push(format!(
            "proof {idx} rule=loop_bound target=0 lo=0 hi={bound_hi} witness=0"
        ));
        idx
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
        self.last_func = Some(lbl.clone());
        lbl
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

    fn lset(&mut self, lane: &str, slot: u32, val: &str) -> String {
        let lbl = self.fresh("ls");
        let _ = writeln!(
            self.body,
            "node {lbl} = LocalSet {lane} fg=0 pol=l1 proof=0 tok=pure a={slot} b={val}"
        );
        lbl
    }

    fn bin(&mut self, op: &str, lane: &str, a: &str, b: &str) -> String {
        let lbl = self.fresh("op");
        let _ = writeln!(
            self.body,
            "node {lbl} = {op} {lane} fg=0 pol=l1 proof=0 tok=pure a={a} b={b}"
        );
        lbl
    }

    fn unary(&mut self, op: &str, lane: &str, a: &str) -> String {
        let lbl = self.fresh("un");
        let _ = writeln!(
            self.body,
            "node {lbl} = {op} {lane} fg=0 pol=l1 proof=0 tok=pure a={a} b=0"
        );
        lbl
    }

    fn fma(&mut self, a: &str, b: &str, c: &str) -> String {
        let lbl = self.fresh("fma");
        let _ = writeln!(
            self.body,
            "node {lbl} = Fma f64x1 fg=0 pol=l1 proof=0 tok=pure a={a} b={b} c={c}"
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
        self.unary("MaskNot", "maskx1", a)
    }

    fn block(&mut self, proof: usize) {
        let lbl = self.fresh("blk");
        let _ = writeln!(
            self.body,
            "node {lbl} = Block i64x1 fg=0 pol=l1 proof={proof} tok=pure a=0 b=0"
        );
    }

    fn loop_open(&mut self, proof: usize) {
        let lbl = self.fresh("lp");
        let _ = writeln!(
            self.body,
            "node {lbl} = Loop i64x1 fg=0 pol=l1 proof={proof} tok=pure a=0 b=0"
        );
    }

    fn if_open(&mut self, cond: &str, proof: usize) {
        self.if_open_lane("i64x1", cond, proof);
    }

    fn if_open_lane(&mut self, lane: &str, cond: &str, proof: usize) {
        let lbl = self.fresh("if");
        let _ = writeln!(
            self.body,
            "node {lbl} = If {lane} fg=0 pol=l1 proof={proof} tok=pure a={cond} b=0"
        );
    }

    fn else_void(&mut self) {
        let lbl = self.fresh("el");
        let _ = writeln!(
            self.body,
            "node {lbl} = Else i64x1 fg=0 pol=l1 proof=0 tok=pure a=0 b=0"
        );
    }

    fn else_val(&mut self, lane: &str, val: &str) {
        let lbl = self.fresh("el");
        let _ = writeln!(
            self.body,
            "node {lbl} = Else {lane} fg=0 pol=l1 proof=0 tok=pure a={val} b=1"
        );
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

    fn ret_val(&mut self, lane: &str, val: &str) {
        let lbl = self.fresh("ret");
        let _ = writeln!(
            self.body,
            "node {lbl} = Return {lane} fg=0 pol=l1 proof=0 tok=pure a={val} b=0"
        );
    }

    fn reg_new(&mut self, slot: u16, len: &str) {
        let lbl = self.fresh("rnew");
        let _ = writeln!(
            self.body,
            "node {lbl} = RegionNew i64x1 fg=0 pol=l1 proof=0 tok=pure a={slot} b={len}"
        );
    }

    fn reg_grow(&mut self, slot: u16, delta: &str) {
        let lbl = self.fresh("rgrow");
        let _ = writeln!(
            self.body,
            "node {lbl} = RegionGrow i64x1 fg=0 pol=l1 proof=0 tok=pure a={slot} b={delta}"
        );
    }

    fn reg_len(&mut self, slot: u16) -> String {
        let lbl = self.fresh("rlen");
        let _ = writeln!(
            self.body,
            "node {lbl} = RegionLen i64x1 fg=0 pol=l1 proof=0 tok=pure a={slot} b=0"
        );
        lbl
    }

    fn reg_free(&mut self, slot: u16) {
        let lbl = self.fresh("rfree");
        let _ = writeln!(
            self.body,
            "node {lbl} = RegionFree i64x1 fg=0 pol=l1 proof=0 tok=pure a={slot} b=0"
        );
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

    fn call(&mut self, func_id: u16, ret_lane: &str, args: &[&str]) -> String {
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
            "node {lbl} = Call {ret_lane} fg=0 pol=l1 proof=0 tok=pure a={func_id} b={a0} c={a1}"
        );
        lbl
    }

    fn finish(self, output_label: Option<&str>) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "cap {}", self.cap);
        for p in &self.proofs {
            let _ = writeln!(out, "{p}");
        }
        out.push_str(&self.body);
        let final_out = output_label
            .or(self.last_func.as_deref())
            .unwrap_or("$fn_0");
        let _ = writeln!(out, "output {final_out}");
        out
    }
}

/// Runs both `Tier 0` and `ELF64` AOT (checking byte-identical recompilation) and asserts identical output bytes.
fn check_tier0_eq_aot(prog: &RulProgram, label: &str) -> Result<(), String> {
    verify_program(prog).map_err(|e| {
        let nf = prog.nodes.get(e.node_idx as usize);
        format!("{label}: verify failed: {e:?} (node={nf:?})")
    })?;

    let (elf1, _) =
        compile_to_elf64_bytes(prog).map_err(|e| format!("{label}: compile #1 failed: {e:?}"))?;
    let (elf2, _) =
        compile_to_elf64_bytes(prog).map_err(|e| format!("{label}: compile #2 failed: {e:?}"))?;
    if elf1 != elf2 {
        return Err(format!(
            "{label}: non-deterministic ELF64 output ({} vs {} bytes)",
            elf1.len(),
            elf2.len()
        ));
    }

    let exe_path = std::env::temp_dir().join(format!("rul_tc4_{}_{label}", std::process::id()));
    compile_rl_to_executable(prog, &exe_path)
        .map_err(|e| format!("{label}: write exe failed: {e:?}"))?;
    let aot_res = run_compiled_structured_executable(&exe_path, &[]);
    let _ = fs::remove_file(&exe_path);

    let t0_res = execute_tier0_interpreter_with_input(prog, &[], &[]);
    match (t0_res, aot_res) {
        (Ok(t0), Ok(aot_bytes)) => {
            let t0_bytes = if let Some(out_b) = t0.output_bytes {
                out_b
            } else {
                match t0.output {
                    RuntimeValue::TypedScalar(ts) => ts.bits.to_le_bytes().to_vec(),
                    RuntimeValue::Scalar(v) => v.to_bits().to_le_bytes().to_vec(),
                    _ => {
                        return Err(format!("{label}: unexpected non-scalar output"));
                    }
                }
            };
            if t0_bytes != aot_bytes {
                return Err(format!(
                    "{label}: Tier 0 ({t0_bytes:02x?}) != AOT ({aot_bytes:02x?})"
                ));
            }
            Ok(())
        }
        (Err(e_t0), Err(e_aot)) => {
            let matched = matches!(
                (&e_t0, &e_aot),
                (
                    ExecError::LoopBoundExceeded { .. },
                    AotError::ContractAssertionFailed
                ) | (
                    ExecError::ContractAssertionFailed { .. },
                    AotError::ContractAssertionFailed
                ) | (
                    ExecError::RegionBoundsOutOfRange { .. },
                    AotError::GatherOutOfBounds
                ) | (
                    ExecError::IntegerDivisionError { .. },
                    AotError::IntegerDivisionError
                ) | (ExecError::CastOutOfRange { .. }, AotError::CastOutOfRange)
                    | (
                        ExecError::RegionAllocationFailed { .. },
                        AotError::RegionAllocationFailed
                    )
                    | (ExecError::CallDepthExceeded { .. }, AotError::StackOverflow)
            );
            if !matched {
                return Err(format!(
                    "{label}: fail-closed mismatch: Tier 0 = {e_t0:?}, AOT = {e_aot:?}"
                ));
            }
            Ok(())
        }
        (Ok(t0), Err(e_aot)) => Err(format!(
            "{label}: Tier 0 succeeded ({:?}) but AOT failed ({e_aot:?})",
            t0.output
        )),
        (Err(e_t0), Ok(aot_bytes)) => Err(format!(
            "{label}: Tier 0 failed ({e_t0:?}) but AOT succeeded ({aot_bytes:02x?})"
        )),
    }
}

/// Generates a high-pressure multi-function program with >16 live integer-class values AND
/// >16 live `F64` values across a loop back-edge, both arms of an `If`/`Else`, and a `Call`.
#[expect(
    clippy::too_many_lines,
    reason = "self-contained randomized high-register-pressure program generator for AC-1"
)]
fn gen_high_pressure_program(rng: &mut SplitMix64) -> String {
    let mut g = RlsGen::new("NONE");
    g.add_region(0, 16, 16, "i64", "plain");
    g.add_region(1, 16, 16, "u8", "plain");
    g.add_region(2, 16, 16, "f64", "plain");

    let p_if_i64 = g.add_block_sig("i64x1");
    let p_if_f64 = g.add_block_sig("f64x1");
    let p_blk_i64 = g.add_block_sig("i64x1");
    let p_loop = g.add_loop_bound(8);

    // 4 helpers with arity 0..=4 mixing I64, U8, BoolMask, F64
    g.add_func_sig(0, &[], "i64");
    g.add_func_sig(1, &["i64"], "f64");
    g.add_func_sig(2, &["i64", "u8"], "i64");
    g.add_func_sig(3, &["i64", "f64", "u8", "mask"], "i64");
    g.add_func_sig(4, &[], "i64");

    // func 0: 0-arg helper touching region 0
    g.func(0, "i64x1");
    let idx0 = g.const_i64(i64::try_from(rng.next_range(16)).unwrap_or(0));
    let v0 = g.reg_load(0, "i64x1", &idx0);
    let k0 = g.const_i64(rng.next_small_i64());
    let r0 = g.bin("Add", "i64x1", &v0, &k0);
    let _ = g.end_val("i64x1", &r0);

    // func 1: 1-arg I64 -> F64 helper with FMA, Sqrt, Floor, Ceil, Abs
    g.func(1, "f64x1");
    let p0 = g.lget("i64x1", 0);
    let f0 = g.cast("f64x1", &p0);
    let fa = g.const_f64(rng.next_small_f64().abs() + 1.0);
    let fb = g.const_f64(rng.next_small_f64());
    let fm = g.fma(&f0, &fa, &fb);
    let ab = g.unary("Abs", "f64x1", &fm);
    let c1 = g.const_f64(1.0);
    let pos = g.bin("Add", "f64x1", &ab, &c1);
    let sq = g.unary("Sqrt", "f64x1", &pos);
    let fl = if rng.next_range(2) == 0 {
        g.unary("Floor", "f64x1", &sq)
    } else {
        g.unary("Ceil", "f64x1", &sq)
    };
    let _ = g.end_val("f64x1", &fl);

    // func 2: 2-arg (i64, u8) -> i64 helper using div/rem/shifts/u8 ops
    g.func(2, "i64x1");
    let a = g.lget("i64x1", 0);
    let b = g.lget("u8x1", 1);
    let u_add = g.const_u8(u8::try_from(rng.next_range(250) + 1).unwrap_or(1));
    let b2 = g.bin("Add", "u8x1", &b, &u_add);
    let b_i64 = g.cast("i64x1", &b2);
    let d_val = g.const_i64(i64::try_from(rng.next_range(17) + 1).unwrap_or(3));
    let q = g.bin("Div", "i64x1", &a, &d_val);
    let r = g.bin("Rem", "i64x1", &a, &d_val);
    let sh = g.const_i64(i64::try_from(rng.next_range(5) + 1).unwrap_or(1));
    let s1 = g.bin("Shl", "i64x1", &q, &sh);
    let s2 = g.bin("ShrS", "i64x1", &r, &sh);
    let res2 = g.bin("BitXor", "i64x1", &s1, &s2);
    let res2b = g.bin("Add", "i64x1", &res2, &b_i64);
    let _ = g.end_val("i64x1", &res2b);

    // func 3: 4-arg (i64, f64, u8, mask) -> i64
    g.func(3, "i64x1");
    let x0 = g.lget("i64x1", 0);
    let x1 = g.lget("f64x1", 1);
    let x2 = g.lget("u8x1", 2);
    let x3 = g.lget("maskx1", 3);
    let x1_fl = g.unary("Floor", "f64x1", &x1);
    let lim_lo = g.const_f64(-1000.0);
    let lim_hi = g.const_f64(1000.0);
    let clamped = g.bin("Max", "f64x1", &x1_fl, &lim_lo);
    let clamped2 = g.bin("Min", "f64x1", &clamped, &lim_hi);
    let f_as_i = g.cast("i64x1", &clamped2);
    let u_as_i = g.cast("i64x1", &x2);
    let m_as_i = g.cast("i64x1", &x3);
    let sum1 = g.bin("Add", "i64x1", &x0, &f_as_i);
    let sum2 = g.bin("Add", "i64x1", &u_as_i, &m_as_i);
    let sel = g.bin("SelectZero", "i64x1", &x3, &sum2);
    let out3 = g.bin("BitXor", "i64x1", &sum1, &sel);
    let _ = g.end_val("i64x1", &out3);

    // func 4 (entry): >16 live I64/U8/BoolMask AND >16 live F64 across Loop, If/Else, and Call!
    g.func(4, "i64x1");
    let num_ints = 18 + rng.next_range(5); // 18..=22 live integer values (> 16)
    let num_floats = 18 + rng.next_range(5); // 18..=22 live F64 values (> 16)

    // Initialize regions 0, 1, 2
    for slot_i in 0..4_i64 {
        let ci = g.const_i64(slot_i);
        let iv = g.const_i64(rng.next_small_i64());
        g.reg_store(0, "i64x1", &ci, &iv);
        let uv = g.const_u8(u8::try_from(rng.next_range(256)).unwrap_or(0));
        g.reg_store(1, "u8x1", &ci, &uv);
        let fv = g.const_f64(rng.next_small_f64());
        g.reg_store(2, "f64x1", &ci, &fv);
    }

    // Create >16 non-constant integer values (via region loads + arithmetic so they cannot fold!)
    let mut live_ints: Vec<String> = Vec::with_capacity(num_ints);
    let seed_idx = g.const_i64(0);
    let base_i = g.reg_load(0, "i64x1", &seed_idx);
    for k in 0..num_ints {
        let ck = g.const_i64(rng.next_small_i64() + i64::try_from(k).unwrap_or(0));
        let v = match k % 4 {
            0 => g.bin("Add", "i64x1", &base_i, &ck),
            1 => g.bin("Sub", "i64x1", &base_i, &ck),
            2 => g.bin("BitXor", "i64x1", &base_i, &ck),
            _ => g.bin("Mul", "i64x1", &base_i, &ck),
        };
        live_ints.push(v);
    }

    // Create >16 non-constant F64 values
    let mut live_floats: Vec<String> = Vec::with_capacity(num_floats);
    let base_f = g.reg_load(2, "f64x1", &seed_idx);
    for k in 0..num_floats {
        let ck = g.const_f64(rng.next_small_f64() + 0.5 * (k as f64));
        let v = match k % 3 {
            0 => g.bin("Add", "f64x1", &base_f, &ck),
            1 => g.bin("Sub", "f64x1", &base_f, &ck),
            _ => g.bin("Mul", "f64x1", &base_f, &ck),
        };
        live_floats.push(v);
    }

    // Typed locals (mix of I64, F64, U8, BoolMask)
    let z0 = g.const_i64(0);
    g.lset("i64x1", 0, &z0); // loop counter
    g.lset("i64x1", 1, &z0); // int accumulator
    let fz = g.const_f64(0.0);
    g.lset("f64x1", 2, &fz); // float accumulator
    let uz = g.const_u8(u8::try_from(rng.next_range(256)).unwrap_or(0));
    g.lset("u8x1", 3, &uz);

    // Bounded Loop using several outer live_ints and live_floats across back-edge AND calling helpers
    let iters = i64::try_from(2 + rng.next_range(4)).unwrap_or(3);
    g.block(0);
    g.loop_open(p_loop);
    let lc = g.lget("i64x1", 0);
    let lim = g.const_i64(iters);
    let lt = g.bin("CmpLt", "maskx1", &lc, &lim);
    let stop = g.mask_not(&lt);
    g.br_if(&stop, 1);

    let pick_i = rng.next_range(num_ints);
    let pick_f = rng.next_range(num_floats);
    let li = live_ints.get(pick_i).cloned().unwrap_or_default();
    let lf = live_floats.get(pick_f).cloned().unwrap_or_default();

    let cur_acc = g.lget("i64x1", 1);
    let cur_u = g.lget("u8x1", 3);
    let c0_res = g.call(0, "i64x1", &[]);
    let c1_res = g.call(1, "f64x1", &[&li]);
    let c2_res = g.call(2, "i64x1", &[&c0_res, &cur_u]);
    let cond_m = g.bin("CmpLt", "maskx1", &lc, &li);
    let c3_res = g.call(3, "i64x1", &[&c2_res, &lf, &cur_u, &cond_m]);

    // Value-returning If/Else inside loop
    g.if_open(&cond_m, p_if_i64);
    let t_branch = g.bin("Add", "i64x1", &cur_acc, &c3_res);
    g.else_val("i64x1", &t_branch);
    let e_branch = g.bin("BitXor", "i64x1", &cur_acc, &c3_res);
    let if_res = g.end_val("i64x1", &e_branch);
    g.lset("i64x1", 1, &if_res);

    // Value-returning F64 If/Else inside loop
    let cur_facc = g.lget("f64x1", 2);
    g.if_open_lane("f64x1", &cond_m, p_if_f64);
    let tf = g.bin("Add", "f64x1", &cur_facc, &c1_res);
    g.else_val("f64x1", &tf);
    let ef = g.bin("Sub", "f64x1", &cur_facc, &lf);
    let if_fres = g.end_val("f64x1", &ef);
    g.lset("f64x1", 2, &if_fres);

    let u1 = g.const_u8(7);
    let next_u = g.bin("Add", "u8x1", &cur_u, &u1);
    g.lset("u8x1", 3, &next_u);

    let one = g.const_i64(1);
    let lc_next = g.bin("Add", "i64x1", &lc, &one);
    g.lset("i64x1", 0, &lc_next);
    g.br(0);
    g.end_void(); // end loop
    g.end_void(); // end block

    // Outer If/Else consuming ALL >16 live_ints and ALL >16 live_floats in BOTH arms!
    let final_acc_pre = g.lget("i64x1", 1);
    let cmp_zero = g.const_i64(0);
    let branch_cond = g.bin("CmpGt", "maskx1", &final_acc_pre, &cmp_zero);

    g.if_open(&branch_cond, p_if_i64);
    let mut sum_then = final_acc_pre.clone();
    for v in &live_ints {
        sum_then = g.bin("Add", "i64x1", &sum_then, v);
    }
    let mut fsum_then = g.lget("f64x1", 2);
    for fv in &live_floats {
        fsum_then = g.bin("Add", "f64x1", &fsum_then, fv);
    }
    let fsum_then_clamped = {
        let lo = g.const_f64(-1e9);
        let hi = g.const_f64(1e9);
        let a = g.bin("Max", "f64x1", &fsum_then, &lo);
        g.bin("Min", "f64x1", &a, &hi)
    };
    let fti = g.cast("i64x1", &fsum_then_clamped);
    let then_total = g.bin("BitXor", "i64x1", &sum_then, &fti);
    g.else_val("i64x1", &then_total);
    let mut sum_else = final_acc_pre;
    for v in live_ints.iter().rev() {
        sum_else = g.bin("Sub", "i64x1", &sum_else, v);
    }
    let mut fsum_else = g.lget("f64x1", 2);
    for fv in live_floats.iter().rev() {
        fsum_else = g.bin("Sub", "f64x1", &fsum_else, fv);
    }
    let fsum_else_clamped = {
        let lo = g.const_f64(-1e9);
        let hi = g.const_f64(1e9);
        let a = g.bin("Max", "f64x1", &fsum_else, &lo);
        g.bin("Min", "f64x1", &a, &hi)
    };
    let fei = g.cast("i64x1", &fsum_else_clamped);
    let else_total = g.bin("Add", "i64x1", &sum_else, &fei);
    let outer_if_res = g.end_val("i64x1", &else_total);

    // Value-returning Block wrapping final combination + RegionStore/Load round-trip
    g.block(p_blk_i64);
    let store_slot = g.const_i64(5);
    g.reg_store(0, "i64x1", &store_slot, &outer_if_res);
    let reloaded = g.reg_load(0, "i64x1", &store_slot);
    let mask32 = g.const_i64(0xFFFF_FFFF);
    let lo32 = g.bin("BitAnd", "i64x1", &reloaded, &mask32);
    let final_v = g.bin("BitXor", "i64x1", &reloaded, &lo32);
    let blk_res = g.end_val("i64x1", &final_v);

    let _ = g.end_val("i64x1", &blk_res);
    g.finish(None)
}

/// Generates a randomized mixed-type structured program (both single-unit and multi-function).
#[expect(
    clippy::too_many_lines,
    reason = "randomized mixed-type structured program generator for AC-1"
)]
fn gen_mixed_program(rng: &mut SplitMix64, use_funcs: bool) -> String {
    let mut g = RlsGen::new("NONE");
    let use_regions = rng.next_range(2) == 0;
    if use_regions {
        g.add_region(0, 8, 8, "i64", "plain");
        g.add_region(1, 8, 8, "u8", "plain");
        g.add_region(2, 8, 8, "f64", "plain");
    }
    let p_if_i = g.add_block_sig("i64x1");
    let p_blk_i = g.add_block_sig("i64x1");
    let p_loop = g.add_loop_bound(12);

    if use_funcs {
        let arity = rng.next_range(5); // 0..=4
        let param_types: Vec<&str> = (0..arity)
            .map(|k| match (k + rng.next_range(4)) % 4 {
                0 => "i64",
                1 => "u8",
                2 => "f64",
                _ => "mask",
            })
            .collect();
        g.add_func_sig(0, &param_types, "i64");
        g.add_func_sig(1, &[], "i64");

        // Helper func 0
        g.func(0, "i64x1");
        let mut acc = g.const_i64(rng.next_small_i64());
        for (k, &pt) in param_types.iter().enumerate() {
            let k32 = u32::try_from(k).unwrap_or(0);
            let term = match pt {
                "i64" => g.lget("i64x1", k32),
                "u8" => {
                    let u = g.lget("u8x1", k32);
                    g.cast("i64x1", &u)
                }
                "f64" => {
                    let f = g.lget("f64x1", k32);
                    let lo = g.const_f64(-1000.0);
                    let hi = g.const_f64(1000.0);
                    let c = g.bin("Max", "f64x1", &f, &lo);
                    let c2 = g.bin("Min", "f64x1", &c, &hi);
                    g.cast("i64x1", &c2)
                }
                _ => {
                    let m = g.lget("maskx1", k32);
                    g.cast("i64x1", &m)
                }
            };
            acc = match rng.next_range(3) {
                0 => g.bin("Add", "i64x1", &acc, &term),
                1 => g.bin("Sub", "i64x1", &acc, &term),
                _ => g.bin("BitXor", "i64x1", &acc, &term),
            };
        }
        if rng.next_range(3) == 0 {
            // Early return inside an If
            let c0 = g.const_i64(0);
            let is_lt = g.bin("CmpLt", "maskx1", &acc, &c0);
            g.if_open(&is_lt, 0);
            let alt = g.const_i64(rng.next_small_i64());
            let sum = g.bin("Add", "i64x1", &acc, &alt);
            g.ret_val("i64x1", &sum);
            g.end_void();
        }
        let _ = g.end_val("i64x1", &acc);

        // Entry func 1
        g.func(1, "i64x1");
        let mut ints: Vec<String> = Vec::new();
        let mut u8s: Vec<String> = Vec::new();
        let mut floats: Vec<String> = Vec::new();
        let mut masks: Vec<String> = Vec::new();

        for _ in 0..(4 + rng.next_range(8)) {
            ints.push(g.const_i64(rng.next_small_i64()));
        }
        for _ in 0..3 {
            u8s.push(g.const_u8(u8::try_from(rng.next_range(256)).unwrap_or(0)));
        }
        for _ in 0..4 {
            floats.push(g.const_f64(rng.next_small_f64()));
        }
        let m0 = g.bin(
            "CmpLt",
            "maskx1",
            ints.first().map_or("", String::as_str),
            ints.get(1).map_or("", String::as_str),
        );
        masks.push(m0);

        // Random ALU chain
        for _ in 0..(8 + rng.next_range(12)) {
            match rng.next_range(6) {
                0 => {
                    let a = ints
                        .get(rng.next_range(ints.len()))
                        .cloned()
                        .unwrap_or_default();
                    let b = ints
                        .get(rng.next_range(ints.len()))
                        .cloned()
                        .unwrap_or_default();
                    let op = match rng.next_range(8) {
                        0 => "Add",
                        1 => "Sub",
                        2 => "Mul",
                        3 => "BitAnd",
                        4 => "BitOr",
                        5 => "BitXor",
                        6 => "Shl",
                        _ => "ShrU",
                    };
                    let v = g.bin(op, "i64x1", &a, &b);
                    ints.push(v);
                }
                1 => {
                    let a = u8s
                        .get(rng.next_range(u8s.len()))
                        .cloned()
                        .unwrap_or_default();
                    let b = u8s
                        .get(rng.next_range(u8s.len()))
                        .cloned()
                        .unwrap_or_default();
                    let op = match rng.next_range(5) {
                        0 => "Add",
                        1 => "Sub",
                        2 => "Mul",
                        3 => "BitXor",
                        _ => "ShrU",
                    };
                    let v = g.bin(op, "u8x1", &a, &b);
                    u8s.push(v);
                }
                2 => {
                    let a = floats
                        .get(rng.next_range(floats.len()))
                        .cloned()
                        .unwrap_or_default();
                    let b = floats
                        .get(rng.next_range(floats.len()))
                        .cloned()
                        .unwrap_or_default();
                    let v = match rng.next_range(5) {
                        0 => g.bin("Add", "f64x1", &a, &b),
                        1 => g.bin("Sub", "f64x1", &a, &b),
                        2 => g.bin("Min", "f64x1", &a, &b),
                        3 => g.bin("Max", "f64x1", &a, &b),
                        _ => g.unary("Abs", "f64x1", &a),
                    };
                    floats.push(v);
                }
                3 => {
                    let a = ints
                        .get(rng.next_range(ints.len()))
                        .cloned()
                        .unwrap_or_default();
                    let b = ints
                        .get(rng.next_range(ints.len()))
                        .cloned()
                        .unwrap_or_default();
                    let op = match rng.next_range(4) {
                        0 => "CmpGt",
                        1 => "CmpLt",
                        2 => "CmpEq",
                        _ => "CmpLtU",
                    };
                    let m = g.bin(op, "maskx1", &a, &b);
                    masks.push(m);
                }
                4 => {
                    let a = ints
                        .get(rng.next_range(ints.len()))
                        .cloned()
                        .unwrap_or_default();
                    let d = g.const_i64(i64::try_from(rng.next_range(19) + 1).unwrap_or(5));
                    let op = if rng.next_range(2) == 0 { "Div" } else { "Rem" };
                    let v = g.bin(op, "i64x1", &a, &d);
                    ints.push(v);
                }
                _ => {
                    let m = masks
                        .get(rng.next_range(masks.len()))
                        .cloned()
                        .unwrap_or_default();
                    let v = ints
                        .get(rng.next_range(ints.len()))
                        .cloned()
                        .unwrap_or_default();
                    let s = g.bin("SelectZero", "i64x1", &m, &v);
                    ints.push(s);
                }
            }
        }

        // Bounded loop with locals + optional region traffic
        let z0 = g.const_i64(0);
        let init_a = ints.last().cloned().unwrap_or_else(|| z0.clone());
        g.lset("i64x1", 0, &z0);
        g.lset("i64x1", 1, &init_a);
        let lim_val = i64::try_from(1 + rng.next_range(6)).unwrap_or(3);
        g.block(0);
        g.loop_open(p_loop);
        let iv = g.lget("i64x1", 0);
        let lim = g.const_i64(lim_val);
        let cond = g.bin("CmpLt", "maskx1", &iv, &lim);
        let done = g.mask_not(&cond);
        g.br_if(&done, 1);

        let cur = g.lget("i64x1", 1);
        let arg_strs: Vec<String> = param_types
            .iter()
            .map(|&pt| match pt {
                "i64" => ints
                    .get(rng.next_range(ints.len()))
                    .cloned()
                    .unwrap_or_default(),
                "u8" => u8s
                    .get(rng.next_range(u8s.len()))
                    .cloned()
                    .unwrap_or_default(),
                "f64" => floats
                    .get(rng.next_range(floats.len()))
                    .cloned()
                    .unwrap_or_default(),
                _ => masks
                    .get(rng.next_range(masks.len()))
                    .cloned()
                    .unwrap_or_default(),
            })
            .collect();
        let arg_refs: Vec<&str> = arg_strs.iter().map(String::as_str).collect();
        let call_r = g.call(0, "i64x1", &arg_refs);
        if use_regions {
            g.reg_store(0, "i64x1", &iv, &call_r);
            let loaded = g.reg_load(0, "i64x1", &iv);
            let upd = g.bin("Add", "i64x1", &cur, &loaded);
            g.lset("i64x1", 1, &upd);
        } else {
            let zero_c = g.const_i64(0);
            let is_pos = g.bin("CmpGt", "maskx1", &call_r, &zero_c);
            g.if_open(&is_pos, 0);
            let upd1 = g.bin("Add", "i64x1", &cur, &call_r);
            g.lset("i64x1", 1, &upd1);
            g.else_void();
            let upd2 = g.bin("Sub", "i64x1", &cur, &call_r);
            g.lset("i64x1", 1, &upd2);
            g.end_void();
        }
        let one = g.const_i64(1);
        let iv_next = g.bin("Add", "i64x1", &iv, &one);
        g.lset("i64x1", 0, &iv_next);
        g.br(0);
        g.end_void();
        g.end_void();

        let fin = g.lget("i64x1", 1);
        let c_mask = masks.last().cloned().unwrap_or_default();
        g.if_open(&c_mask, p_if_i);
        let k_t = g.const_i64(rng.next_small_i64());
        let r_t = g.bin("Add", "i64x1", &fin, &k_t);
        g.else_val("i64x1", &r_t);
        let k_e = g.const_i64(rng.next_small_i64());
        let r_e = g.bin("BitXor", "i64x1", &fin, &k_e);
        let final_if = g.end_val("i64x1", &r_e);

        g.block(p_blk_i);
        let u_last = u8s.last().cloned().unwrap_or_default();
        let u_i64 = g.cast("i64x1", &u_last);
        let combined = g.bin("Add", "i64x1", &final_if, &u_i64);
        let blk_out = g.end_val("i64x1", &combined);
        let _ = g.end_val("i64x1", &blk_out);
        g.finish(None)
    } else {
        // Non-function top-level structured program (exercises `!func_mode` emitter path!)
        let z0 = g.const_i64(0);
        let start_acc = g.const_i64(rng.next_small_i64());
        g.lset("i64x1", 0, &z0);
        g.lset("i64x1", 1, &start_acc);
        let f_init = g.const_f64(rng.next_small_f64().abs() + 1.5);
        g.lset("f64x1", 2, &f_init);

        let iters = i64::try_from(1 + rng.next_range(8)).unwrap_or(4);
        g.block(0);
        g.loop_open(p_loop);
        let i = g.lget("i64x1", 0);
        let lim = g.const_i64(iters);
        let lt = g.bin("CmpLt", "maskx1", &i, &lim);
        let done = g.mask_not(&lt);
        g.br_if(&done, 1);

        let acc = g.lget("i64x1", 1);
        let mul_k = g.const_i64(rng.next_small_i64());
        let prod = g.bin("Mul", "i64x1", &i, &mul_k);
        let next_a = g.bin("Add", "i64x1", &acc, &prod);
        if use_regions {
            g.reg_store(0, "i64x1", &i, &next_a);
            let rld = g.reg_load(0, "i64x1", &i);
            g.lset("i64x1", 1, &rld);
        } else {
            g.lset("i64x1", 1, &next_a);
        }

        // Exercise x87 Ln/Exp + F64 locals in non-func mode
        let fc = g.lget("f64x1", 2);
        let fln = g.unary("Ln", "f64x1", &fc);
        let fone = g.const_f64(1.0);
        let fabs = g.unary("Abs", "f64x1", &fln);
        let fnext = g.bin("Add", "f64x1", &fabs, &fone);
        let _ = g.unary("AssertContract", "f64x1", &fnext);
        g.lset("f64x1", 2, &fnext);

        let c1 = g.const_i64(1);
        let inext = g.bin("Add", "i64x1", &i, &c1);
        g.lset("i64x1", 0, &inext);
        g.br(0);
        g.end_void();
        g.end_void();

        let fin_i = g.lget("i64x1", 1);
        let fin_f = g.lget("f64x1", 2);
        let fin_fi = g.cast("i64x1", &fin_f);
        let out_v = g.bin("BitXor", "i64x1", &fin_i, &fin_fi);
        g.finish(Some(&out_v))
    }
}

/// Builds a targeted `Call -> Add/Sub(LocalGet, ConstI64) -> LocalSet` window program inside
/// nested result `Block`s or `If`s with `num_locals` pinned locals and optional call-spanning values.
/// Exercises free-GPR exhaustion inside `keep_in_rax_local_set`.
fn build_keep_in_rax_window_program(
    nested_scopes: usize,
    use_if_wrapper: bool,
    num_locals: usize,
    spanning_vals: usize,
    window_ops: usize,
    use_sub: bool,
    call_ret_val: i64,
) -> String {
    let mut g = RlsGen::new("NONE");
    let p_blk = g.add_block_sig("i64x1");
    g.add_func_sig(0, &[], "i64");
    g.add_func_sig(1, &[], "i64");

    // func 0: returns `call_ret_val`
    g.func(0, "i64x1");
    let rv = g.const_i64(call_ret_val);
    let _ = g.end_val("i64x1", &rv);

    // func 1 (entry): initialize `num_locals` locals so they are pinned to GPRs,
    // open `nested_scopes` result `Block`/`If` scopes to reserve result GPRs,
    // keep `spanning_vals` values live across `Call 0`, then emit `window_ops`
    // `Add`/`Sub(LocalGet, ConstI64)` nodes between `Call 0` and `LocalSet`.
    g.func(1, "i64x1");
    for s in 0..num_locals {
        let init = g.const_i64(i64::try_from(s).unwrap_or(0));
        g.lset("i64x1", u32::try_from(s).unwrap_or(0), &init);
    }

    // Create `LocalGet`s before the call so window `Add`/`Sub` nodes can use them directly
    // within the 4-node window `(idx + 1)..eff_lu`.
    let mut pre_gets: Vec<String> = Vec::with_capacity(num_locals);
    for s in 0..num_locals {
        pre_gets.push(g.lget("i64x1", u32::try_from(s).unwrap_or(0)));
    }

    // Open nested value-producing scopes to consume caller-saved GPRs as result homes.
    let c1_cond = g.const_i64(1);
    let c0_cond = g.const_i64(0);
    let always_true = g.bin("CmpGt", "maskx1", &c1_cond, &c0_cond);
    for d in 0..nested_scopes {
        if use_if_wrapper && d == nested_scopes.saturating_sub(1) {
            g.if_open(&always_true, p_blk);
        } else {
            g.block(p_blk);
        }
    }

    // Optional call-spanning non-constant values.
    let mut spans: Vec<String> = Vec::with_capacity(spanning_vals);
    for k in 0..spanning_vals {
        let base_g = pre_gets
            .get(k % num_locals.max(1))
            .cloned()
            .unwrap_or_else(|| c1_cond.clone());
        let ck = g.const_i64(i64::try_from(100 + k).unwrap_or(100));
        spans.push(g.bin("Add", "i64x1", &base_g, &ck));
    }

    // Constants used inside the window (defined BEFORE the Call so the window stays within 4 nodes).
    let mut win_consts: Vec<String> = Vec::with_capacity(window_ops);
    for k in 0..window_ops {
        win_consts.push(g.const_i64(i64::try_from((k + 1) * 11).unwrap_or(11)));
    }

    let c = g.call(0, "i64x1", &[]);
    let mut win_results: Vec<String> = Vec::with_capacity(window_ops);
    let target_local = u32::try_from(num_locals.saturating_sub(1)).unwrap_or(0);
    for k in 0..window_ops {
        // Pick a pinned local other than `target_local` (if available) so its `LocalGet` aliases its GPR.
        let src_slot = if num_locals > 1 {
            (k + 1) % (num_locals - 1)
        } else {
            0
        };
        let lg = pre_gets
            .get(src_slot)
            .cloned()
            .unwrap_or_else(|| c1_cond.clone());
        let kc = win_consts
            .get(k)
            .cloned()
            .unwrap_or_else(|| c1_cond.clone());
        let op = if use_sub && k % 2 == 1 { "Sub" } else { "Add" };
        win_results.push(g.bin(op, "i64x1", &lg, &kc));
    }
    g.lset("i64x1", target_local, &c);

    let mut acc = g.lget("i64x1", target_local);
    for wr in &win_results {
        acc = g.bin("Add", "i64x1", &acc, wr);
    }
    for sp in &spans {
        acc = g.bin("Add", "i64x1", &acc, sp);
    }
    // Keep all locals live/referenced so the local scorer pins locals 0..num_locals.
    for s in 0..num_locals {
        let g_s = g.lget("i64x1", u32::try_from(s).unwrap_or(0));
        acc = g.bin("Add", "i64x1", &acc, &g_s);
    }

    for d in (0..nested_scopes).rev() {
        if use_if_wrapper && d == nested_scopes.saturating_sub(1) {
            g.else_val("i64x1", &acc);
            let zero_alt = g.const_i64(0);
            acc = g.end_val("i64x1", &zero_alt);
        } else {
            acc = g.end_val("i64x1", &acc);
        }
    }
    let _ = g.end_val("i64x1", &acc);
    g.finish(None)
}

#[test]
fn test_tc_core4_keep_in_rax_window_exhaustion_regression() -> Result<(), String> {
    // 1. Exact minimal reproducer from code review:
    //    func0 returns 1007; entry pins locals 0..=4, opens 4 nested `Block i64x1`
    //    (reserving rdi, rsi, rdx, rcx), then runs:
    //      c = Call 0; a1 = Add(LocalGet 1, 11); a2 = Add(LocalGet 2, 22);
    //      LocalSet 4 = c; r = Add(Add(LocalGet 4, a1), a2); End x5.
    //    Tier 0 and ELF64 AOT must both return 1031 (not 24).
    let mut g = RlsGen::new("NONE");
    let p_blk = g.add_block_sig("i64x1");
    g.add_func_sig(0, &[], "i64");
    g.add_func_sig(1, &[], "i64");

    g.func(0, "i64x1");
    let c1007 = g.const_i64(1007);
    let _ = g.end_val("i64x1", &c1007);

    g.func(1, "i64x1");
    for s in 0u32..5 {
        let z = g.const_i64(0);
        g.lset("i64x1", s, &z);
    }
    let lg1 = g.lget("i64x1", 1);
    let lg2 = g.lget("i64x1", 2);
    let k11 = g.const_i64(11);
    let k22 = g.const_i64(22);
    for _ in 0..4 {
        g.block(p_blk);
    }
    let c = g.call(0, "i64x1", &[]);
    let a1 = g.bin("Add", "i64x1", &lg1, &k11);
    let a2 = g.bin("Add", "i64x1", &lg2, &k22);
    g.lset("i64x1", 4, &c);
    let lg4 = g.lget("i64x1", 4);
    let s1 = g.bin("Add", "i64x1", &lg4, &a1);
    let mut r = g.bin("Add", "i64x1", &s1, &a2);
    // Reference locals 0 and 3 after the window so all 5 locals are used.
    let lg0 = g.lget("i64x1", 0);
    let lg3 = g.lget("i64x1", 3);
    let s03 = g.bin("Add", "i64x1", &lg0, &lg3);
    r = g.bin("Add", "i64x1", &r, &s03);
    for _ in 0..4 {
        r = g.end_val("i64x1", &r);
    }
    let _ = g.end_val("i64x1", &r);

    let exact_prog =
        assemble_rls(&g.finish(None)).map_err(|e| format!("keep_in_rax exact assemble: {e:?}"))?;
    check_tier0_eq_aot(&exact_prog, "keep_in_rax_exact_repro")?;
    let t0 = execute_tier0_interpreter(&exact_prog, &[])
        .map_err(|e| format!("keep_in_rax exact tier0: {e:?}"))?;
    let RuntimeValue::TypedScalar(TypedScalar { bits, .. }) = t0.output else {
        return Err("expected TypedScalar from Tier 0".to_owned());
    };
    assert_eq!(
        i64::from_le_bytes(bits.to_le_bytes()),
        1040,
        "sanity check on exact reproducer value (1007 + 11 + 22 = 1040)"
    );

    // Also test the exact minimal form without touching locals 0/3 after the window (where result is 1007 + 11 + 22 = 1040, or with local 1 = -9 -> 1031):
    let mut g2 = RlsGen::new("NONE");
    let p_blk2 = g2.add_block_sig("i64x1");
    g2.add_func_sig(0, &[], "i64");
    g2.add_func_sig(1, &[], "i64");
    g2.func(0, "i64x1");
    let c1007_2 = g2.const_i64(1007);
    let _ = g2.end_val("i64x1", &c1007_2);
    g2.func(1, "i64x1");
    let init_vals = [0i64, -9, 0, 0, 0];
    for (s, &v) in init_vals.iter().enumerate() {
        let cv = g2.const_i64(v);
        g2.lset("i64x1", u32::try_from(s).unwrap_or(0), &cv);
    }
    let lg1_2 = g2.lget("i64x1", 1);
    let lg2_2 = g2.lget("i64x1", 2);
    let k11_2 = g2.const_i64(11);
    let k22_2 = g2.const_i64(22);
    for _ in 0..4 {
        g2.block(p_blk2);
    }
    let c_2 = g2.call(0, "i64x1", &[]);
    let a1_2 = g2.bin("Add", "i64x1", &lg1_2, &k11_2);
    let a2_2 = g2.bin("Add", "i64x1", &lg2_2, &k22_2);
    g2.lset("i64x1", 4, &c_2);
    let lg4_2 = g2.lget("i64x1", 4);
    let s1_2 = g2.bin("Add", "i64x1", &lg4_2, &a1_2);
    let mut r_2 = g2.bin("Add", "i64x1", &s1_2, &a2_2);
    for _ in 0..4 {
        r_2 = g2.end_val("i64x1", &r_2);
    }
    let _ = g2.end_val("i64x1", &r_2);
    let exact_prog2 = assemble_rls(&g2.finish(None))
        .map_err(|e| format!("keep_in_rax exact2 assemble: {e:?}"))?;
    check_tier0_eq_aot(&exact_prog2, "keep_in_rax_exact_1031")?;
    let t0_2 = execute_tier0_interpreter(&exact_prog2, &[])
        .map_err(|e| format!("keep_in_rax exact2 tier0: {e:?}"))?;
    let RuntimeValue::TypedScalar(TypedScalar { bits: bits2, .. }) = t0_2.output else {
        return Err("expected TypedScalar from Tier 0".to_owned());
    };
    assert_eq!(i64::from_le_bytes(bits2.to_le_bytes()), 1031);

    // 2. Sweep across nested result Blocks/Ifs (2..=4), locals (2..=7), spanning values (0..=2),
    //    and window Add/Sub counts (1..=3).
    let mut case_idx = 0usize;
    for nested_scopes in 2..=4 {
        for use_if in [false, true] {
            for num_locals in [2usize, 4, 5, 6, 7] {
                for spanning_vals in 0..=2 {
                    for window_ops in 1..=3 {
                        let rls = build_keep_in_rax_window_program(
                            nested_scopes,
                            use_if,
                            num_locals,
                            spanning_vals,
                            window_ops,
                            case_idx % 2 == 1,
                            1007 + i64::try_from(case_idx).unwrap_or(0),
                        );
                        let prog = assemble_rls(&rls)
                            .map_err(|e| format!("keep_in_rax_{case_idx} assemble: {e:?}"))?;
                        check_tier0_eq_aot(&prog, &format!("keep_in_rax_{case_idx}"))?;
                        case_idx += 1;
                    }
                }
            }
        }
    }
    Ok(())
}

// =========================================================================
// AC-1: 500+ randomized differential programs + byte-identical ELF64 check
// =========================================================================
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "comprehensive AC-1 differential suite covering randomized, permutation, barrier, nested-loop, and fail-closed tests"
)]
fn test_tc_core4_regalloc_differential() -> Result<(), String> {
    let mut rng = SplitMix64::new(0xC0DE_4040_2026_0504);
    let total_programs = 520usize;
    let high_pressure_count = 120usize;

    for idx in 0..total_programs {
        let rls = if idx < high_pressure_count {
            gen_high_pressure_program(&mut rng)
        } else if idx < high_pressure_count + 40 {
            // Randomized Call -> Add/Sub(LocalGet, ConstI64) -> LocalSet windows inside
            // nested result Blocks/Ifs with 5+ pinned locals.
            let nested = 2 + rng.next_range(3);
            let use_if = rng.next_range(2) == 0;
            let num_locals = 5 + rng.next_range(3);
            let spans = rng.next_range(3);
            let win_ops = 1 + rng.next_range(3);
            let use_sub = rng.next_range(2) == 0;
            let ret_v = 1000 + rng.next_small_i64();
            build_keep_in_rax_window_program(
                nested, use_if, num_locals, spans, win_ops, use_sub, ret_v,
            )
        } else {
            gen_mixed_program(&mut rng, idx % 3 != 0)
        };
        let prog = assemble_rls(&rls).map_err(|e| format!("prog_{idx} assemble: {e:?}"))?;
        check_tier0_eq_aot(&prog, &format!("diff_{idx}"))?;
    }

    // Explicitly verify all 24 permutations of 4-arg Call parallel moves (including 2-cycles & 4-cycles)
    // plus duplicated arguments.
    let perms: [[usize; 4]; 24] = [
        [0, 1, 2, 3],
        [0, 1, 3, 2],
        [0, 2, 1, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2],
        [0, 3, 2, 1],
        [1, 0, 2, 3],
        [1, 0, 3, 2],
        [1, 2, 0, 3],
        [1, 2, 3, 0],
        [1, 3, 0, 2],
        [1, 3, 2, 0],
        [2, 0, 1, 3],
        [2, 0, 3, 1],
        [2, 1, 0, 3],
        [2, 1, 3, 0],
        [2, 3, 0, 1],
        [2, 3, 1, 0],
        [3, 0, 1, 2],
        [3, 0, 2, 1],
        [3, 1, 0, 2],
        [3, 1, 2, 0],
        [3, 2, 0, 1],
        [3, 2, 1, 0],
    ];
    let mut perm_g = RlsGen::new("NONE");
    perm_g.add_func_sig(0, &["i64", "i64", "i64", "i64"], "i64");
    perm_g.add_func_sig(1, &["i64", "i64", "i64", "i64"], "i64");
    perm_g.add_func_sig(2, &[], "i64");
    // func 0: weighted sum a0 + 10*a1 + 100*a2 + 1000*a3
    perm_g.func(0, "i64x1");
    let p0 = perm_g.lget("i64x1", 0);
    let p1 = perm_g.lget("i64x1", 1);
    let p2 = perm_g.lget("i64x1", 2);
    let p3 = perm_g.lget("i64x1", 3);
    let c10 = perm_g.const_i64(10);
    let c100 = perm_g.const_i64(100);
    let c1000 = perm_g.const_i64(1000);
    let t1 = perm_g.bin("Mul", "i64x1", &p1, &c10);
    let t2 = perm_g.bin("Mul", "i64x1", &p2, &c100);
    let t3 = perm_g.bin("Mul", "i64x1", &p3, &c1000);
    let s01 = perm_g.bin("Add", "i64x1", &p0, &t1);
    let s23 = perm_g.bin("Add", "i64x1", &t2, &t3);
    let tot = perm_g.bin("Add", "i64x1", &s01, &s23);
    let _ = perm_g.end_val("i64x1", &tot);
    // func 1: invokes func 0 with all 24 permutations + duplicate argument patterns
    perm_g.func(1, "i64x1");
    let qs = [
        perm_g.lget("i64x1", 0),
        perm_g.lget("i64x1", 1),
        perm_g.lget("i64x1", 2),
        perm_g.lget("i64x1", 3),
    ];
    let mut perm_acc = perm_g.const_i64(0);
    for (p_idx, p) in perms.iter().enumerate() {
        let a0 = qs.get(p[0]).map_or("", String::as_str);
        let a1 = qs.get(p[1]).map_or("", String::as_str);
        let a2 = qs.get(p[2]).map_or("", String::as_str);
        let a3 = qs.get(p[3]).map_or("", String::as_str);
        let res = perm_g.call(0, "i64x1", &[a0, a1, a2, a3]);
        let weight = perm_g.const_i64(i64::try_from(p_idx + 1).unwrap_or(1));
        let weighted = perm_g.bin("Mul", "i64x1", &res, &weight);
        perm_acc = perm_g.bin("Add", "i64x1", &perm_acc, &weighted);
    }
    let dup_a = qs.get(3).map_or("", String::as_str);
    let dup_b = qs.first().map_or("", String::as_str);
    let dup_res = perm_g.call(0, "i64x1", &[dup_a, dup_a, dup_b, dup_b]);
    perm_acc = perm_g.bin("Add", "i64x1", &perm_acc, &dup_res);
    let _ = perm_g.end_val("i64x1", &perm_acc);
    // func 2: entry
    perm_g.func(2, "i64x1");
    let a0 = perm_g.const_i64(1);
    let a1 = perm_g.const_i64(2);
    let a2 = perm_g.const_i64(3);
    let a3 = perm_g.const_i64(4);
    let out = perm_g.call(1, "i64x1", &[&a0, &a1, &a2, &a3]);
    let _ = perm_g.end_val("i64x1", &out);
    let perm_prog = assemble_rls(&perm_g.finish(None)).map_err(|e| format!("perm: {e:?}"))?;
    check_tier0_eq_aot(&perm_prog, "call_arg_all_24_perms")?;

    // Dynamic register Div/Rem, RegionNew/Grow/Len/Free barriers with live GPR + F64 values,
    // and 2-level nested loops with outer values live across inner loops.
    {
        let mut g = RlsGen::new("NONE");
        g.add_region(0, 0, 64, "i64", "plain");
        let p_outer = g.add_loop_bound(10);
        let p_inner = g.add_loop_bound(10);
        g.add_func_sig(0, &["i64", "i64"], "i64");
        g.add_func_sig(1, &[], "i64");

        // func 0: dynamic register Div & Rem on register operands
        g.func(0, "i64x1");
        let x = g.lget("i64x1", 0);
        let y = g.lget("i64x1", 1);
        let q = g.bin("Div", "i64x1", &x, &y);
        let r = g.bin("Rem", "i64x1", &x, &y);
        let qr = g.bin("Add", "i64x1", &q, &r);
        let _ = g.end_val("i64x1", &qr);

        // func 1: keep 14 I64 + 6 F64 values live across RegionNew/Grow/Len/Free and 2-level nested loops
        g.func(1, "i64x1");
        let mut live_i: Vec<String> = Vec::new();
        let mut live_f: Vec<String> = Vec::new();
        for k in 0..14_i64 {
            let ck = g.const_i64(k + 3);
            let c2 = g.const_i64(2);
            live_i.push(g.bin("Add", "i64x1", &ck, &c2));
        }
        for k in 0..6 {
            let cf = g.const_f64(f64::from(k + 1) * 1.5);
            let c1 = g.const_f64(0.5);
            live_f.push(g.bin("Add", "f64x1", &cf, &c1));
        }
        let init_len = g.const_i64(8);
        g.reg_new(0, &init_len);
        let delta = g.const_i64(8);
        g.reg_grow(0, &delta);
        let rlen = g.reg_len(0);

        let z0 = g.const_i64(0);
        g.lset("i64x1", 0, &z0);
        g.lset("i64x1", 1, &rlen);
        g.block(0);
        g.loop_open(p_outer);
        let oi = g.lget("i64x1", 0);
        let olim = g.const_i64(3);
        let ocond = g.bin("CmpLt", "maskx1", &oi, &olim);
        let odone = g.mask_not(&ocond);
        g.br_if(&odone, 1);

        let outer_val = g.bin("Add", "i64x1", &oi, live_i.first().map_or(&z0, |s| s));
        g.lset("i64x1", 2, &z0);
        g.block(0);
        g.loop_open(p_inner);
        let ii = g.lget("i64x1", 2);
        let ilim = g.const_i64(4);
        let icond = g.bin("CmpLt", "maskx1", &ii, &ilim);
        let idone = g.mask_not(&icond);
        g.br_if(&idone, 1);

        let c1 = g.const_i64(1);
        let denom = g.bin("Add", "i64x1", &ii, &c1);
        let num = g.bin("Add", "i64x1", &outer_val, &rlen);
        let dr = g.call(0, "i64x1", &[&num, &denom]);
        g.reg_store(0, "i64x1", &ii, &dr);
        let ld = g.reg_load(0, "i64x1", &ii);
        let acc = g.lget("i64x1", 1);
        let acc2 = g.bin("Add", "i64x1", &acc, &ld);
        g.lset("i64x1", 1, &acc2);

        let ii_next = g.bin("Add", "i64x1", &ii, &c1);
        g.lset("i64x1", 2, &ii_next);
        g.br(0);
        g.end_void();
        g.end_void();

        let c1_o = g.const_i64(1);
        let oi_next = g.bin("Add", "i64x1", &oi, &c1_o);
        g.lset("i64x1", 0, &oi_next);
        g.br(0);
        g.end_void();
        g.end_void();

        g.reg_free(0);

        let mut total = g.lget("i64x1", 1);
        for v in &live_i {
            total = g.bin("Add", "i64x1", &total, v);
        }
        for fv in &live_f {
            let fi = g.cast("i64x1", fv);
            total = g.bin("Add", "i64x1", &total, &fi);
        }
        let _ = g.end_val("i64x1", &total);
        let prog =
            assemble_rls(&g.finish(None)).map_err(|e| format!("barriers assemble: {e:?}"))?;
        check_tier0_eq_aot(&prog, "region_barriers_nested_loops_dyn_div")?;
    }

    // Fail-closed parity checks (exit 70, 71, 72, 73): Tier 0 and ELF64 AOT must trap identically.
    // (a) Register divide-by-zero (exit 72)
    {
        let mut g = RlsGen::new("NONE");
        g.add_func_sig(0, &["i64", "i64"], "i64");
        g.add_func_sig(1, &[], "i64");
        g.func(0, "i64x1");
        let a = g.lget("i64x1", 0);
        let b = g.lget("i64x1", 1);
        let q = g.bin("Div", "i64x1", &a, &b);
        let _ = g.end_val("i64x1", &q);
        g.func(1, "i64x1");
        let c42 = g.const_i64(42);
        let c0 = g.const_i64(0);
        let r = g.call(0, "i64x1", &[&c42, &c0]);
        let _ = g.end_val("i64x1", &r);
        let prog = assemble_rls(&g.finish(None)).map_err(|e| format!("div0 assemble: {e:?}"))?;
        check_tier0_eq_aot(&prog, "fail_closed_div_zero")?;
    }
    // (b) Signed i64::MIN / -1 overflow (exit 72)
    {
        let mut g = RlsGen::new("NONE");
        g.add_func_sig(0, &["i64", "i64"], "i64");
        g.add_func_sig(1, &[], "i64");
        g.func(0, "i64x1");
        let a = g.lget("i64x1", 0);
        let b = g.lget("i64x1", 1);
        let q = g.bin("Div", "i64x1", &a, &b);
        let _ = g.end_val("i64x1", &q);
        g.func(1, "i64x1");
        let cmin = g.const_i64(i64::MIN);
        let cm1 = g.const_i64(-1);
        let r = g.call(0, "i64x1", &[&cmin, &cm1]);
        let _ = g.end_val("i64x1", &r);
        let prog = assemble_rls(&g.finish(None)).map_err(|e| format!("divmin assemble: {e:?}"))?;
        check_tier0_eq_aot(&prog, "fail_closed_div_min_neg1")?;
    }
    // (c) Region out-of-bounds load (exit 71)
    {
        let mut g = RlsGen::new("NONE");
        g.add_region(0, 4, 4, "i64", "plain");
        let oob = g.const_i64(4);
        let v = g.reg_load(0, "i64x1", &oob);
        let prog = assemble_rls(&g.finish(Some(&v))).map_err(|e| format!("oob assemble: {e:?}"))?;
        check_tier0_eq_aot(&prog, "fail_closed_region_oob")?;
    }
    // (d) Bounded-loop overrun (exit 70)
    {
        let mut g = RlsGen::new("NONE");
        let p_lb = g.add_loop_bound(3);
        let z = g.const_i64(0);
        g.lset("i64x1", 0, &z);
        g.block(0);
        g.loop_open(p_lb);
        let i = g.lget("i64x1", 0);
        let c1 = g.const_i64(1);
        let n = g.bin("Add", "i64x1", &i, &c1);
        g.lset("i64x1", 0, &n);
        g.br(0);
        g.end_void();
        g.end_void();
        let fin = g.lget("i64x1", 0);
        let prog = assemble_rls(&g.finish(Some(&fin)))
            .map_err(|e| format!("lb overrun assemble: {e:?}"))?;
        check_tier0_eq_aot(&prog, "fail_closed_loop_bound")?;
    }
    // (e) Out-of-range F64 -> I64 Cast (exit 73)
    {
        let mut g = RlsGen::new("NONE");
        let big = g.const_f64(1.0e20);
        let c = g.cast("i64x1", &big);
        let prog =
            assemble_rls(&g.finish(Some(&c))).map_err(|e| format!("cast oor assemble: {e:?}"))?;
        check_tier0_eq_aot(&prog, "fail_closed_cast_oor")?;
    }

    Ok(())
}

// =========================================================================
// AC-2: Spilled recursive frames (>40 live I64s across recursive Call)
//       stay strictly within MAX_STACK_UNITS under `ulimit -s 64`
// =========================================================================
#[test]
fn test_tc_core4_spilled_frames_stay_within_stack_budget() -> Result<(), String> {
    let live_count = 48usize; // > 40 I64 values simultaneously live across recursive Call
    let pad_nodes = 320usize; // guarantees MAX_STACK_UNITS hits well before MAX_CALL_DEPTH (4096)

    let build_heavy_rec = |n_val: i64| -> Result<(RulProgram, u32, u32), String> {
        let mut g = RlsGen::new("DIVERGE");
        g.add_func_sig(0, &["i64"], "i64");
        g.add_func_sig(1, &[], "i64");

        // func 0: heavy_rec(n: local 0) -> i64
        g.func(0, "i64x1");
        let n = g.lget("i64x1", 0);
        let c0 = g.const_i64(0);
        let eq0 = g.bin("CmpEq", "maskx1", &n, &c0);
        g.if_open(&eq0, 0);
        let z = g.const_i64(0);
        g.ret_val("i64x1", &z);
        g.end_void();

        // Compute 48 distinct non-constant I64 values derived from `n`, all live across the recursive Call!
        let mut live_vals = Vec::with_capacity(live_count);
        for k in 0..live_count {
            let ck = g.const_i64(i64::try_from(k + 1).unwrap_or(1));
            let vk = g.bin("Add", "i64x1", &n, &ck);
            live_vals.push(vk);
        }
        // Additional padding nodes to push `frame_units(f)` so `MAX_STACK_UNITS` triggers before `MAX_CALL_DEPTH`.
        for k in 0..pad_nodes {
            let _ = g.const_i64(i64::try_from(k).unwrap_or(0));
        }

        let c1 = g.const_i64(1);
        let nm1 = g.bin("Sub", "i64x1", &n, &c1);
        let sub = g.call(0, "i64x1", &[&nm1]);

        // Fold all 48 live values with `sub` after the recursive call returns!
        let mut acc = sub;
        for v in &live_vals {
            acc = g.bin("Add", "i64x1", &acc, v);
        }
        let _ = g.end_val("i64x1", &acc);

        // func 1 (entry): main() -> i64
        g.func(1, "i64x1");
        let nv = g.const_i64(n_val);
        let res = g.call(0, "i64x1", &[&nv]);
        let _ = g.end_val("i64x1", &res);

        let prog =
            assemble_rls(&g.finish(None)).map_err(|e| format!("heavy_rec assemble: {e:?}"))?;
        verify_program(&prog).map_err(|e| format!("heavy_rec verify: {e:?}"))?;
        let entry_start = prog.output_node as usize;
        let rec_nodes = entry_start;
        let main_nodes = prog.nodes.len() - entry_start;
        Ok((
            prog,
            func_frame_units(rec_nodes),
            func_frame_units(main_nodes),
        ))
    };

    let (_, rec_units, main_units) = build_heavy_rec(0)?;
    let max_rec_frames = (MAX_STACK_UNITS - main_units) / rec_units;
    assert!(
        max_rec_frames + 1 < MAX_CALL_DEPTH,
        "stack budget must be hit before MAX_CALL_DEPTH (got {max_rec_frames} frames)"
    );
    assert!(
        main_units + max_rec_frames * rec_units <= MAX_STACK_UNITS
            && main_units + (max_rec_frames + 1) * rec_units > MAX_STACK_UNITS,
        "exact MAX_STACK_UNITS boundary calculation failed"
    );

    let ok_n = i64::from(max_rec_frames - 1);
    let over_n = i64::from(max_rec_frames);

    // 1. Deepest valid level (`ok_n`): Tier 0 == AOT (default) == AOT (`ulimit -s 64`)
    let (ok_prog, _, _) = build_heavy_rec(ok_n)?;
    let t0_ok = execute_tier0_interpreter(&ok_prog, &[])
        .map_err(|e| format!("heavy_rec ok tier0: {e:?}"))?;
    let RuntimeValue::TypedScalar(TypedScalar {
        bits: expected_bits,
        ..
    }) = t0_ok.output
    else {
        return Err("expected TypedScalar output from Tier 0".to_owned());
    };
    let expected_val = i64::from_le_bytes(expected_bits.to_le_bytes());

    let exe_ok = std::env::temp_dir().join(format!("rul_tc4_{}_spilled_ok", std::process::id()));
    compile_rl_to_executable(&ok_prog, &exe_ok)
        .map_err(|e| format!("heavy_rec ok compile: {e:?}"))?;
    let aot_ok = run_compiled_structured_executable(&exe_ok, &[])
        .map_err(|e| format!("heavy_rec ok aot: {e:?}"))?;
    let aot_ok_val = i64::from_le_bytes(
        aot_ok
            .as_slice()
            .try_into()
            .map_err(|_e| "expected 8 bytes".to_owned())?,
    );
    assert_eq!(
        aot_ok_val, expected_val,
        "Tier 0 != AOT at deepest valid spilled recursion"
    );

    let lim_ok = Command::new("bash")
        .arg("-c")
        .arg("ulimit -s 64 && exec \"$1\"")
        .arg("_")
        .arg(&exe_ok)
        .output()
        .map_err(|e| format!("ulimit -s 64 ok exec: {e}"))?;
    let _ = fs::remove_file(&exe_ok);
    assert!(
        lim_ok.status.success(),
        "expected success under ulimit -s 64, got status={:?}",
        lim_ok.status
    );
    let lim_ok_val = i64::from_le_bytes(
        lim_ok
            .stdout
            .as_slice()
            .try_into()
            .map_err(|_e| "expected 8 bytes under ulimit -s 64".to_owned())?,
    );
    assert_eq!(lim_ok_val, expected_val);

    // 2. One level deeper (`over_n`): fail closed with CallDepthExceeded / exit 75 (never a signal!)
    let (over_prog, _, _) = build_heavy_rec(over_n)?;
    let t0_over = execute_tier0_interpreter(&over_prog, &[]);
    assert!(
        matches!(t0_over, Err(ExecError::CallDepthExceeded { .. })),
        "expected Tier 0 CallDepthExceeded at over_n={over_n}, got {t0_over:?}"
    );

    let exe_over =
        std::env::temp_dir().join(format!("rul_tc4_{}_spilled_over", std::process::id()));
    compile_rl_to_executable(&over_prog, &exe_over)
        .map_err(|e| format!("heavy_rec over compile: {e:?}"))?;
    let aot_over = run_compiled_structured_executable(&exe_over, &[]);
    assert!(
        matches!(aot_over, Err(AotError::StackOverflow)),
        "expected AOT StackOverflow (exit 75), got {aot_over:?}"
    );

    let lim_over = Command::new("bash")
        .arg("-c")
        .arg("ulimit -s 64 && exec \"$1\"")
        .arg("_")
        .arg(&exe_over)
        .output()
        .map_err(|e| format!("ulimit -s 64 over exec: {e}"))?;
    let _ = fs::remove_file(&exe_over);
    assert_eq!(
        lim_over.status.code(),
        Some(AOT_EXIT_STACK_OVERFLOW),
        "expected exit code 75 (never a signal) under ulimit -s 64, got {:?}",
        lim_over.status
    );
    assert!(
        lim_over.stdout.is_empty(),
        "fail-closed exit 75 must emit zero stdout bytes"
    );

    Ok(())
}

// =========================================================================
// AC-3: Performance Gate (`<= 2.0x` of `rustc -O` across all 5 exit programs)
// =========================================================================
fn run_binary_timed(exe: &Path, input_file: &Path) -> Result<(Duration, Vec<u8>), String> {
    let stdin_file =
        File::open(input_file).map_err(|e| format!("open input {}: {e}", input_file.display()))?;
    let t0 = Instant::now();
    let output = Command::new(exe)
        .stdin(Stdio::from(stdin_file))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("spawn {}: {e}", exe.display()))?;
    let elapsed = t0.elapsed();
    if !output.status.success() {
        return Err(format!(
            "{} exited with {:?}, stderr={}",
            exe.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok((elapsed, output.stdout))
}

#[test]
#[ignore = "AC-3 release benchmark gate against rustc -O; run with cargo test --release test_tc_core4_performance_gate -- --ignored --nocapture --test-threads=1"]
fn test_tc_core4_performance_gate() -> Result<(), String> {
    // Verify `rustc` is available (must fail instead of skipping).
    let rustc_check = Command::new("rustc")
        .arg("--version")
        .output()
        .map_err(|e| format!("rustc is required for AC-3 performance gate: {e}"))?;
    if !rustc_check.status.success() {
        return Err("rustc --version failed".to_owned());
    }

    let perf_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("perf/v0_5_exit");
    let tmp_dir = std::env::temp_dir().join(format!("rul_tc4_perf_{}", std::process::id()));
    fs::create_dir_all(&tmp_dir).map_err(|e| format!("create tmp_dir: {e}"))?;

    // Build benchmark inputs matching the approved specification table.
    let fib_input = 40i64.to_le_bytes().to_vec();
    let mut qsort_input = Vec::with_capacity(16);
    qsort_input.extend_from_slice(&2_000_000_i64.to_le_bytes());
    qsort_input.extend_from_slice(&0x1234_5678_9ABC_DEF0_i64.to_le_bytes());
    let sieve_input = 50_000_000_i64.to_le_bytes().to_vec();
    let mut sha_input = vec![0u8; 64 * 1024 * 1024];
    for (i, slot) in sha_input.iter_mut().enumerate() {
        *slot = u8::try_from((i.wrapping_mul(31).wrapping_add(7)) & 0xFF).unwrap_or(0);
    }
    let bf_input = b"-[>-[>-[>+<-]<-]<-]-[>-[>-[>+<-]<-]<-]-[>-[>-[>+<-]<-]<-]>>>.".to_vec();

    let benchmarks: [(&str, &[u8]); 5] = [
        ("fib", &fib_input),
        ("quicksort", &qsort_input),
        ("sieve", &sieve_input),
        ("sha256", &sha_input),
        ("brainfuck", &bf_input),
    ];

    let mut failures: Vec<String> = Vec::new();
    println!(
        "\n{:<12} {:>10} {:>10} {:>8} {:>10} {:>10}",
        "Benchmark", "rustc -O", "Rul AOT", "Ratio", "ELF (B)", "Codegen"
    );
    println!("{}", "-".repeat(66));

    for (name, input_bytes) in benchmarks {
        let rls_path = perf_dir.join(format!("{name}.rls"));
        let rs_path = perf_dir.join(format!("{name}.rs"));
        let rls_src = fs::read_to_string(&rls_path)
            .map_err(|e| format!("read {}: {e}", rls_path.display()))?;
        let prog = assemble_rls(&rls_src).map_err(|e| format!("{name} assemble: {e:?}"))?;
        verify_program(&prog).map_err(|e| format!("{name} verify: {e:?}"))?;

        let rul_exe = tmp_dir.join(format!("{name}_rul"));
        let ref_exe = tmp_dir.join(format!("{name}_rs"));
        let in_path = tmp_dir.join(format!("{name}.in"));

        let report = compile_rl_to_executable(&prog, &rul_exe)
            .map_err(|e| format!("{name} aot compile: {e:?}"))?;

        let rustc_out = Command::new("rustc")
            .arg("-O")
            .arg(&rs_path)
            .arg("-o")
            .arg(&ref_exe)
            .output()
            .map_err(|e| format!("{name} rustc invoke: {e}"))?;
        if !rustc_out.status.success() {
            let _ = fs::remove_dir_all(&tmp_dir);
            return Err(format!(
                "{name} rustc -O failed: {}",
                String::from_utf8_lossy(&rustc_out.stderr)
            ));
        }

        {
            let mut f =
                File::create(&in_path).map_err(|e| format!("create {}: {e}", in_path.display()))?;
            f.write_all(input_bytes)
                .map_err(|e| format!("write {}: {e}", in_path.display()))?;
        }

        let mut ref_min = Duration::MAX;
        let mut rul_min = Duration::MAX;
        let mut expected_stdout: Option<Vec<u8>> = None;

        for iter in 0..5usize {
            let (d_ref, out_ref) = run_binary_timed(&ref_exe, &in_path)?;
            if d_ref < ref_min {
                ref_min = d_ref;
            }
            if let Some(ref exp) = expected_stdout {
                if exp != &out_ref {
                    failures.push(format!(
                        "{name}: rustc ref non-deterministic on iter {iter}"
                    ));
                }
            } else {
                expected_stdout = Some(out_ref.clone());
            }

            let (d_rul, out_rul) = run_binary_timed(&rul_exe, &in_path)?;
            if d_rul < rul_min {
                rul_min = d_rul;
            }
            if out_rul != out_ref {
                failures.push(format!(
                    "{name}: stdout mismatch on iter {iter}: rul={out_rul:02x?} vs ref={out_ref:02x?}"
                ));
            }
        }

        if name == "brainfuck" && expected_stdout.as_deref() != Some(&[0xFD]) {
            failures.push(format!(
                "brainfuck: expected [0xFD], got {expected_stdout:02x?}"
            ));
        }

        let ref_ms = ref_min.as_secs_f64() * 1000.0;
        let rul_ms = rul_min.as_secs_f64() * 1000.0;
        let ratio = rul_ms / ref_ms;
        let cg_us = (report.codegen_ns as f64) / 1000.0;

        println!(
            "{name:<12} {ref_ms:>8.2}ms {rul_ms:>8.2}ms {ratio:>7.2}x {:>10} {cg_us:>8.2}us",
            report.elf_bytes
        );

        if ref_min < Duration::from_millis(50) {
            failures.push(format!(
                "{name}: reference min wall time {ref_ms:.2}ms is below 50ms minimum"
            ));
        }
        if ratio > 2.0 {
            failures.push(format!(
                "{name}: ratio {ratio:.3}x exceeds 2.0x gate (rul={rul_ms:.2}ms, ref={ref_ms:.2}ms)"
            ));
        }

        let _ = fs::remove_file(&in_path);
        let _ = fs::remove_file(&rul_exe);
        let _ = fs::remove_file(&ref_exe);
    }

    let _ = fs::remove_dir_all(&tmp_dir);

    if !failures.is_empty() {
        return Err(format!(
            "AC-3 performance gate failed:\n  - {}",
            failures.join("\n  - ")
        ));
    }
    Ok(())
}

// =========================================================================
// AC-4: Linear O(N) Codegen Latency Gate
// =========================================================================
fn build_linear_codegen_stress_program(target_nodes: usize) -> Result<RulProgram, String> {
    let mut g = RlsGen::new("NONE");
    let live_count = 64usize;
    let mut live_vals: Vec<String> = Vec::with_capacity(live_count);
    let seed_a = g.const_i64(7);
    let seed_b = g.const_i64(13);
    for k in 0..live_count {
        let ck = g.const_i64(i64::try_from(k + 1).unwrap_or(1));
        let v = if k % 2 == 0 {
            g.bin("Add", "i64x1", &seed_a, &ck)
        } else {
            g.bin("BitXor", "i64x1", &seed_b, &ck)
        };
        live_vals.push(v);
    }

    // Emit chain of ALU operations in the middle so total node count reaches `target_nodes`,
    // while all 64 `live_vals` remain live across the entire body until the final fold.
    let fold_nodes = live_count;
    let current_nodes = 2 + live_count * 2;
    let body_ops = target_nodes.saturating_sub(current_nodes + fold_nodes);
    let mut chain = live_vals.first().cloned().unwrap_or_else(|| seed_a.clone());
    for k in 0..body_ops {
        let partner = live_vals
            .get(k % live_count)
            .cloned()
            .unwrap_or_else(|| seed_a.clone());
        chain = match k % 3 {
            0 => g.bin("Add", "i64x1", &chain, &partner),
            1 => g.bin("BitXor", "i64x1", &chain, &partner),
            _ => g.bin("Sub", "i64x1", &chain, &partner),
        };
    }

    // Consume all 64 live values at the very end so all 64 stay live across the entire body.
    for v in &live_vals {
        chain = g.bin("Add", "i64x1", &chain, v);
    }

    let rls = g.finish(Some(&chain));
    let prog = assemble_rls(&rls).map_err(|e| format!("stress {target_nodes} assemble: {e:?}"))?;
    verify_program(&prog).map_err(|e| format!("stress {target_nodes} verify: {e:?}"))?;
    Ok(prog)
}

#[test]
#[ignore = "AC-4 release codegen linearity gate; run with cargo test --release test_tc_core4_codegen_time_is_linear -- --ignored --nocapture --test-threads=1"]
fn test_tc_core4_codegen_time_is_linear() -> Result<(), String> {
    let prog_6k = build_linear_codegen_stress_program(6_000)?;
    let prog_60k = build_linear_codegen_stress_program(60_000)?;

    let perf_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("perf/v0_5_exit");
    let mut cases: Vec<(String, RulProgram)> = vec![
        ("stress_6k".to_owned(), prog_6k),
        ("stress_60k".to_owned(), prog_60k),
    ];
    for name in ["fib", "quicksort", "sieve", "sha256", "brainfuck"] {
        let rls_path = perf_dir.join(format!("{name}.rls"));
        let rls_src = fs::read_to_string(&rls_path)
            .map_err(|e| format!("read {}: {e}", rls_path.display()))?;
        let prog = assemble_rls(&rls_src).map_err(|e| format!("{name} assemble: {e:?}"))?;
        cases.push((name.to_owned(), prog));
    }

    println!(
        "\n{:<12} {:>8} {:>12} {:>12} {:>10}",
        "Program", "Nodes", "Min cg (ns)", "Limit (ns)", "ns/node"
    );
    println!("{}", "-".repeat(60));

    let mut min_6k: u64 = 0;
    let mut min_60k: u64 = 0;

    for (name, prog) in &cases {
        let mut best_ns = u64::MAX;
        for _ in 0..5 {
            let (_, report) = compile_to_elf64_bytes(prog)
                .map_err(|e| format!("{name} compile failed: {e:?}"))?;
            if report.codegen_ns < best_ns {
                best_ns = report.codegen_ns;
            }
        }
        let n = prog.nodes.len() as u64;
        let limit_ns = 20_000u64.saturating_add(50u64.saturating_mul(n));
        let ns_per_node = (best_ns as f64) / (n.max(1) as f64);
        println!("{name:<12} {n:>8} {best_ns:>12} {limit_ns:>12} {ns_per_node:>9.2}");

        assert!(
            best_ns <= limit_ns,
            "{name}: codegen_ns ({best_ns}) exceeded 20_000 + 50 * {n} ({limit_ns})"
        );
        if name == "stress_6k" {
            min_6k = best_ns;
        } else if name == "stress_60k" {
            min_60k = best_ns;
        }
    }

    let ratio_60_6 = (min_60k as f64) / (min_6k.max(1) as f64);
    println!("\n60k / 6k codegen ratio: {ratio_60_6:.2}x (limit <= 15.0x)");
    assert!(
        min_60k <= min_6k.saturating_mul(15),
        "60,000-node codegen ({min_60k} ns) exceeded 15x of 6,000-node ({min_6k} ns)"
    );
    Ok(())
}
