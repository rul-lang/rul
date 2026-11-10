//! Rul (`.rl`) CLI Inspector & First-Principles Production Benchmark Suite.
//!
//! Compares Rul (`.rl`) across:
//! - `[A]` Legacy Imperative AoS + `Vec::collect()`
//! - `[B]` Idiomatic Rust SoA (`iter().copied().filter().map().sum()`)
//! - `[C]` Hand-Written C/Rust 8-Accumulator SoA Loop (Single-Core DRAM Ceiling)
//! - `[D]` Rul (`.rl`) Tier 2 Single-Core Fused SIMD (`ExecPolicy::SimdStream`)
//! - `[E]` Rul (`.rl`) Tier 2 Auto-MultiCore (`ExecPolicy::MultiCore`)
//! - `[F]` Rul (`.rl`) Tier 2 + `ArenaZoneMap` Block Skipping (Clustered Column)

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![deny(clippy::indexing_slicing)]

use std::fs::File;
use std::hint::black_box;
use std::io::Read;
use std::time::Instant;

use rul::aot::{
    compile_rl_to_executable, compile_to_elf64_bytes, run_compiled_executable,
    write_arena_data_file,
};
use rul::bytecode::{
    ArenaDescriptor, ExecPolicy, NodeFrame, Opcode, RulProgram, ScheduleHint, TokenOp, TypeLane,
    CAP_NONE,
};
use rul::engine::{
    execute_tier2_fused, execute_tier2_with_zonemap, format_trace_table, trace_program,
    ArenaZoneMap, RuntimeValue,
};
use rul::suite10::{
    format_suite10_table, load_rl_file, run_10_pipeline_suite, save_rl_file, MAX_RL_FILE_BYTES,
};
use rul::transpiler::{assemble_rls, disassemble_rl, program_to_rls, SymbolicGraphBuilder};
use rul::verifier::verify_program;

/// Traditional human-written Array-of-Structs (AoS) record (64 bytes per record,
/// polluting CPU L1/L2 cache lines when only `amount` is needed).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct LegacyTransactionAoS {
    tx_id: u64,
    account_id: u64,
    merchant_id: u64,
    timestamp_ms: u64,
    region_code: u64,
    status_flags: u64,
    fee_rate: f64,
    amount: f64,
}

#[inline(never)]
fn run_legacy_aos_pipeline(records: &[LegacyTransactionAoS], threshold: f64) -> f64 {
    let filtered_squares: Vec<f64> = records
        .iter()
        .filter(|r| r.amount > threshold)
        .map(|r| r.amount * r.amount)
        .collect();
    filtered_squares.iter().copied().sum()
}

#[inline(never)]
fn run_idiomatic_rust_soa(amounts: &[f64], threshold: f64) -> f64 {
    amounts
        .iter()
        .copied()
        .filter(|&x| x > threshold)
        .map(|x| x * x)
        .sum()
}

#[inline(never)]
fn run_handwritten_c_8acc_soa(amounts: &[f64], threshold: f64) -> f64 {
    let mut acc = [0.0_f64; 8];
    let (chunks, remainder) = amounts.as_chunks::<8>();
    for chunk in chunks {
        for (lane, &x) in acc.iter_mut().zip(chunk.iter()) {
            let mask = f64::from(u8::from(x > threshold));
            *lane += (x * x) * mask;
        }
    }
    let mut tail = 0.0_f64;
    for &x in remainder {
        if x > threshold {
            tail += x * x;
        }
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + (acc[4] + acc[5]) + (acc[6] + acc[7]) + tail
}

fn build_rl_pipeline(
    elem_count: u32,
    threshold: f64,
    policy: ExecPolicy,
) -> Result<RulProgram, String> {
    let mut builder = SymbolicGraphBuilder::new(CAP_NONE);
    let arena0 = builder.add_arena(elem_count, TypeLane::F64X8, false);
    let sched = ScheduleHint {
        fusion_group: u8::from(policy != ExecPolicy::AutoL1Tile),
        policy,
    };

    builder
        .emit_arena_input("$amounts", arena0, 1, sched)
        .map_err(|e| e.to_string())?;
    builder
        .emit_const_f64("$threshold", threshold, sched)
        .map_err(|e| e.to_string())?;
    builder
        .emit_binary_op(
            "$mask",
            Opcode::CmpGt,
            TypeLane::MASKX8,
            "$amounts",
            "$threshold",
            sched,
        )
        .map_err(|e| e.to_string())?;
    builder
        .emit_binary_op(
            "$squared",
            Opcode::Mul,
            TypeLane::F64X8,
            "$amounts",
            "$amounts",
            sched,
        )
        .map_err(|e| e.to_string())?;
    builder
        .emit_binary_op(
            "$selected",
            Opcode::SelectZero,
            TypeLane::F64X8,
            "$mask",
            "$squared",
            sched,
        )
        .map_err(|e| e.to_string())?;
    builder
        .emit_unary_op(
            "$sum",
            Opcode::ReduceSum,
            TypeLane::F64_SCALAR,
            "$selected",
            sched,
        )
        .map_err(|e| e.to_string())?;
    builder
        .emit_unary_op(
            "$contract",
            Opcode::AssertContract,
            TypeLane::F64_SCALAR,
            "$sum",
            sched,
        )
        .map_err(|e| e.to_string())?;
    builder.finish("$contract").map_err(|e| e.to_string())
}

fn build_horner50_pipeline(elem_count: u32, policy: ExecPolicy) -> Result<RulProgram, String> {
    let mut b = SymbolicGraphBuilder::new(CAP_NONE);
    let a0 = b.add_arena(elem_count, TypeLane::F64X8, false);
    let s = ScheduleHint {
        fusion_group: 1,
        policy,
    };
    b.emit_arena_input("$x", a0, 1, s)
        .map_err(|e| e.to_string())?;
    b.emit_const_f64("$c0", 1.000_001, s)
        .map_err(|e| e.to_string())?;
    b.emit_const_f64("$c1", 0.999_999, s)
        .map_err(|e| e.to_string())?;
    let mut acc = "$c0".to_owned();
    for step in 0..50usize {
        let m = format!("$m{step}");
        let a = format!("$a{step}");
        let c = if (step & 1) == 0 { "$c1" } else { "$c0" };
        b.emit_binary_op(&m, Opcode::Mul, TypeLane::F64X8, &acc, "$x", s)
            .map_err(|e| e.to_string())?;
        b.emit_binary_op(&a, Opcode::Add, TypeLane::F64X8, &m, c, s)
            .map_err(|e| e.to_string())?;
        acc = a;
    }
    b.emit_unary_op("$sum", Opcode::ReduceSum, TypeLane::F64_SCALAR, &acc, s)
        .map_err(|e| e.to_string())?;
    b.finish("$sum").map_err(|e| e.to_string())
}

fn build_horner50_fma_pipeline(elem_count: u32, policy: ExecPolicy) -> Result<RulProgram, String> {
    let mut b = SymbolicGraphBuilder::new(CAP_NONE);
    let a0 = b.add_arena(elem_count, TypeLane::F64X8, false);
    let s = ScheduleHint {
        fusion_group: 1,
        policy,
    };
    b.emit_arena_input("$x", a0, 1, s)
        .map_err(|e| e.to_string())?;
    b.emit_const_f64("$c0", 1.000_001, s)
        .map_err(|e| e.to_string())?;
    b.emit_const_f64("$c1", 0.999_999, s)
        .map_err(|e| e.to_string())?;
    let mut acc = "$c0".to_owned();
    for step in 0..50usize {
        let f = format!("$f{step}");
        let c = if (step & 1) == 0 { "$c1" } else { "$c0" };
        b.emit_fma_op(&f, TypeLane::F64X8, &acc, "$x", c, s)
            .map_err(|e| e.to_string())?;
        acc = f;
    }
    b.emit_unary_op("$sum", Opcode::ReduceSum, TypeLane::F64_SCALAR, &acc, s)
        .map_err(|e| e.to_string())?;
    b.finish("$sum").map_err(|e| e.to_string())
}

#[inline(never)]
fn run_handwritten_c_horner50(xs: &[f64]) -> f64 {
    let c0 = 1.000_001_f64;
    let c1 = 0.999_999_f64;
    let mut sum0 = 0.0_f64;
    let mut sum1 = 0.0_f64;
    let mut sum2 = 0.0_f64;
    let mut sum3 = 0.0_f64;
    let mut sum4 = 0.0_f64;
    let mut sum5 = 0.0_f64;
    let mut sum6 = 0.0_f64;
    let mut sum7 = 0.0_f64;
    let (chunks, rem) = xs.as_chunks::<8>();
    for &[x0, x1, x2, x3, x4, x5, x6, x7] in chunks {
        let mut a0 = c0;
        let mut a1 = c0;
        let mut a2 = c0;
        let mut a3 = c0;
        let mut a4 = c0;
        let mut a5 = c0;
        let mut a6 = c0;
        let mut a7 = c0;
        for step in 0..50usize {
            let k = if (step & 1) == 0 { c1 } else { c0 };
            a0 = a0 * x0 + k;
            a1 = a1 * x1 + k;
            a2 = a2 * x2 + k;
            a3 = a3 * x3 + k;
            a4 = a4 * x4 + k;
            a5 = a5 * x5 + k;
            a6 = a6 * x6 + k;
            a7 = a7 * x7 + k;
        }
        sum0 += a0;
        sum1 += a1;
        sum2 += a2;
        sum3 += a3;
        sum4 += a4;
        sum5 += a5;
        sum6 += a6;
        sum7 += a7;
    }
    let mut tail_sum = 0.0_f64;
    for &x in rem {
        let mut a = c0;
        for step in 0..50usize {
            let k = if (step & 1) == 0 { c1 } else { c0 };
            a = a * x + k;
        }
        tail_sum += a;
    }
    ((sum0 + sum1) + (sum2 + sum3)) + ((sum4 + sum5) + (sum6 + sum7)) + tail_sum
}

fn build_large_synthetic_graph(node_count: usize) -> RulProgram {
    let sched = ScheduleHint::fused_simd(1);
    let mut nodes = Vec::with_capacity(node_count);
    nodes.push(NodeFrame {
        opcode: Opcode::ArenaInput,
        type_lane: TypeLane::F64X8,
        schedule: sched,
        proof_idx: 0,
        token_op: TokenOp::read_lease(0, 1),
        in_a: 0,
        in_b: 0,
    });
    for i in 1..node_count {
        let prev = u32::try_from(i - 1).unwrap_or(0);
        nodes.push(NodeFrame {
            opcode: Opcode::Add,
            type_lane: TypeLane::F64X8,
            schedule: sched,
            proof_idx: 0,
            token_op: TokenOp::PURE,
            in_a: prev,
            in_b: 0,
        });
    }
    RulProgram {
        capability_mask: CAP_NONE,
        arenas: vec![ArenaDescriptor {
            arena_id: 0,
            element_count: 1024,
            type_lane: TypeLane::F64X8,
            mutable: false,
            alignment: 64,
            initial_version: 1,
        }],
        proofs: Vec::new(),
        nodes,
        output_node: u32::try_from(node_count.saturating_sub(1)).unwrap_or(0),
    }
}

#[derive(Debug, Clone, Copy)]
struct TimingStats {
    median_ms: f64,
    min_ms: f64,
    max_ms: f64,
    gbps: f64,
}

fn bench_runs<F>(
    warmup: usize,
    iters: usize,
    bytes_processed: usize,
    mut f: F,
) -> (f64, TimingStats)
where
    F: FnMut() -> f64,
{
    let mut last_val = 0.0_f64;
    for _ in 0..warmup {
        last_val = black_box(f());
    }
    let mut samples_ms = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        last_val = black_box(f());
        samples_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    samples_ms.sort_by(f64::total_cmp);
    let min_ms = samples_ms.first().copied().unwrap_or(0.0);
    let max_ms = samples_ms.last().copied().unwrap_or(0.0);
    let median_ms = samples_ms.get(samples_ms.len() / 2).copied().unwrap_or(0.0);
    let gbps = (bytes_processed as f64) / (median_ms * 1e-3) / 1e9;
    (
        last_val,
        TimingStats {
            median_ms,
            min_ms,
            max_ms,
            gbps,
        },
    )
}

fn read_bounded_utf8(path: &str) -> Result<String, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut s = String::new();
    file.take(MAX_RL_FILE_BYTES)
        .read_to_string(&mut s)
        .map_err(|e| e.to_string())?;
    Ok(s)
}

fn load_rl_or_rls(path: &str) -> Result<RulProgram, String> {
    if path.ends_with(".rls") {
        let text = read_bounded_utf8(path)?;
        assemble_rls(&text).map_err(|e| e.to_string())
    } else {
        load_rl_file(path).map_err(|e| e.to_string())
    }
}

fn handle_cli(args: &[String]) -> Result<bool, String> {
    let Some(cmd) = args.get(1).map(String::as_str) else {
        return Ok(false);
    };
    let Some(path) = args.get(2).map(String::as_str) else {
        return Ok(false);
    };

    match cmd {
        "inspect" => {
            let prog = load_rl_or_rls(path)?;
            println!("{}", disassemble_rl(&prog));
            Ok(true)
        }
        "trace" => {
            let prog = load_rl_or_rls(path)?;
            let mut cols = Vec::with_capacity(prog.arenas.len());
            for (idx, desc) in prog.arenas.iter().enumerate() {
                let len = desc.element_count as usize;
                let col: Vec<f64> = (0..len)
                    .map(|i| {
                        let v = ((i.saturating_mul(idx.saturating_add(1))) % 250) as f64;
                        v + 0.5
                    })
                    .collect();
                cols.push(col);
            }
            let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
            let rep = trace_program(&prog, &refs).map_err(|e| e.to_string())?;
            print!("{}", format_trace_table(&rep));
            Ok(true)
        }
        "decompile" => {
            let prog = load_rl_file(path).map_err(|e| e.to_string())?;
            print!("{}", program_to_rls(&prog));
            Ok(true)
        }
        "verify" => {
            let prog = load_rl_or_rls(path)?;
            let t0 = Instant::now();
            let summary = verify_program(&prog).map_err(|e| e.to_string())?;
            let ns = t0.elapsed().as_nanos();
            println!(
                "[VERIFIED OK] File={path} | Nodes={} | FusionGroups={} | OutputShape={:?} | ZeroRuntimeBoundsChecks={} | Time={ns} ns",
                summary.node_count,
                summary.fusion_groups,
                summary.output_shape,
                summary.zero_runtime_bounds_checks
            );
            Ok(true)
        }
        "pack" => {
            let out_path = args
                .get(3)
                .map(String::as_str)
                .ok_or_else(|| "usage: rul_bench pack <input.rls> <output.rl>".to_owned())?;
            let text = read_bounded_utf8(path)?;
            let t0 = Instant::now();
            let prog = assemble_rls(&text).map_err(|e| e.to_string())?;
            let summary = verify_program(&prog).map_err(|e| e.to_string())?;
            let pack_ns = t0.elapsed().as_nanos();
            let bytes = save_rl_file(out_path, &prog).map_err(|e| e.to_string())?;
            println!(
                "[PACKED OK] {path} -> {out_path} ({bytes} bytes) | Nodes={} | Verify+Pack={pack_ns} ns (0 rustc invocations)",
                summary.node_count
            );
            Ok(true)
        }
        "compile" => {
            let out_exe = args.get(3).map(String::as_str).ok_or_else(|| {
                "usage: rul_bench compile <input.rl|.rls> <output_exe>".to_owned()
            })?;
            let prog = load_rl_or_rls(path)?;
            let rep = compile_rl_to_executable(&prog, out_exe).map_err(|e| e.to_string())?;
            println!(
                "[AOT COMPILED OK] {path} -> {out_exe} | ELF64={0} B (MachineCode={1} B) | Verify={2} ns | Codegen={3} ns | Total={4} ns (< 1 ms, Zero Runtime)",
                rep.elf_bytes,
                rep.machine_code_bytes,
                rep.verify_ns,
                rep.codegen_ns,
                rep.total_compile_ns
            );
            Ok(true)
        }
        "exec-aot" => {
            let data_path = args.get(3).map(String::as_str).ok_or_else(|| {
                "usage: rul_bench exec-aot <compiled_exe> <arena_data.bin>".to_owned()
            })?;
            let t0 = Instant::now();
            let val = run_compiled_executable(path, data_path).map_err(|e| e.to_string())?;
            let us = t0.elapsed().as_secs_f64() * 1e6;
            println!("[AOT EXEC OK] Binary={path} | Output={val:.6e} | Time={us:.2} us");
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if handle_cli(&args)? {
        return Ok(());
    }

    let record_count: u32 = 10_000_000;
    let threshold = 1000.0_f64;
    let soa_bytes = (record_count as usize) * std::mem::size_of::<f64>();

    // 1. Emit, save to disk (.rl & .rls), reload, and inspect the canonical .rl binary program
    let simd_prog = build_rl_pipeline(record_count, threshold, ExecPolicy::SimdStream)?;
    let multicore_prog = build_rl_pipeline(record_count, threshold, ExecPolicy::MultiCore)?;
    let rl_path = "target/risk_pipeline.rl";
    let rls_path = "target/risk_pipeline.rls";
    let aot_exe_path = "target/risk_pipeline_aot";
    std::fs::create_dir_all("target")
        .map_err(|e| format!("failed to create target/ for benchmark artifacts: {e}"))?;
    let written_bytes = save_rl_file(rl_path, &simd_prog).map_err(|e| e.to_string())?;
    let rls_text = program_to_rls(&simd_prog);
    std::fs::write(rls_path, &rls_text).map_err(|e| e.to_string())?;
    let reloaded_prog = load_rl_file(rl_path).map_err(|e| e.to_string())?;

    // Measure .rls -> .rl assembly speed and Zero-Runtime ELF64 AOT compilation speed
    let t_rls = Instant::now();
    let rls_assembled = assemble_rls(black_box(&rls_text)).map_err(|e| e.to_string())?;
    let rls_assemble_ns = t_rls.elapsed().as_nanos();
    if rls_assembled != simd_prog {
        return Err("rls roundtrip mismatch".to_owned());
    }
    let (_elf_img, aot_mem_rep) =
        compile_to_elf64_bytes(black_box(&simd_prog)).map_err(|e| e.to_string())?;
    let aot_disk_rep =
        compile_rl_to_executable(&simd_prog, aot_exe_path).map_err(|e| e.to_string())?;

    println!(
        "Saved & reloaded canonical binary: {rl_path} ({written_bytes} B) | {rls_path} (Assemble={rls_assemble_ns} ns) | Standalone ELF64: {aot_exe_path} ({} B, In-RAM Compile={} ns)",
        aot_disk_rep.elf_bytes, aot_mem_rep.total_compile_ns
    );
    println!("{}", disassemble_rl(&reloaded_prog));

    // 2. Benchmark O(N) Verifier throughput on a 10,000-node .rl graph
    let large_graph = build_large_synthetic_graph(10_000);
    let large_bytes = large_graph.to_rl_bytes().len();
    let verify_iters = 200u32;
    let t_ver = Instant::now();
    for _ in 0..verify_iters {
        let summary = verify_program(black_box(&large_graph)).map_err(|e| e.to_string())?;
        black_box(summary);
    }
    let ver_elapsed = t_ver.elapsed();
    let ver_per_call_us = (ver_elapsed.as_secs_f64() * 1e6) / f64::from(verify_iters);
    let ver_ns_per_node = (ver_elapsed.as_secs_f64() * 1e9) / (f64::from(verify_iters) * 10_000.0);
    let ver_gbps =
        ((large_bytes as f64) * f64::from(verify_iters)) / ver_elapsed.as_secs_f64() / 1e9;

    println!("=== 1. O(N) LINEAR VERIFIER BENCHMARK (10,000 Nodes / {large_bytes} Bytes) ===");
    println!(
        "  Verify Latency     : {ver_per_call_us:.2} us ({ver_ns_per_node:.2} ns / 128-bit node)"
    );
    println!("  Verify Throughput  : {ver_gbps:.2} GB/s of .rl instruction graph\n");

    // 3. Prepare 10,000,000 records (Uniform + Clustered/Partitioned for Zone-Map)
    println!(
        "=== 2. PRODUCTION RUNTIME BENCHMARK ({record_count} Records = 80 MB SoA, 3 Warmup + 15 Runs) ==="
    );
    let soa_amounts: Vec<f64> = (0..record_count)
        .map(|i| f64::from(i % 2000) + 0.5)
        .collect();
    // Clustered/partitioned analytical column with identical elements sorted by value bucket
    // (typical of time-series / parquet / analytical arenas where Zone-Maps skip cold tiles).
    let mut soa_clustered = soa_amounts.clone();
    soa_clustered.sort_by(f64::total_cmp);
    let clustered_zmap = ArenaZoneMap::build(&soa_clustered);

    let aos_records: Vec<LegacyTransactionAoS> = soa_amounts
        .iter()
        .enumerate()
        .map(|(idx, &amt)| LegacyTransactionAoS {
            tx_id: idx as u64,
            account_id: (idx as u64) ^ 0x55AA,
            merchant_id: (idx as u64) & 0xFF,
            timestamp_ms: 1_700_000_000_000 + (idx as u64),
            region_code: 84,
            status_flags: 1,
            fee_rate: 0.015,
            amount: amt,
        })
        .collect();

    let (aos_sum, stats_aos) = bench_runs(2, 7, soa_bytes, || {
        run_legacy_aos_pipeline(black_box(&aos_records), black_box(threshold))
    });
    let (idiom_sum, stats_idiom) = bench_runs(3, 15, soa_bytes, || {
        run_idiomatic_rust_soa(black_box(&soa_amounts), black_box(threshold))
    });
    let (c8_sum, stats_c8) = bench_runs(3, 15, soa_bytes, || {
        run_handwritten_c_8acc_soa(black_box(&soa_amounts), black_box(threshold))
    });
    let (rl_simd_sum, stats_rl_simd) = bench_runs(3, 15, soa_bytes, || {
        let rep = execute_tier2_fused(black_box(&simd_prog), &[black_box(&soa_amounts)]);
        match rep.map(|r| r.output) {
            Ok(RuntimeValue::Scalar(v)) => v,
            _ => f64::NAN,
        }
    });
    let (rl_mc_sum, stats_rl_mc) = bench_runs(3, 15, soa_bytes, || {
        let rep = execute_tier2_fused(black_box(&multicore_prog), &[black_box(&soa_amounts)]);
        match rep.map(|r| r.output) {
            Ok(RuntimeValue::Scalar(v)) => v,
            _ => f64::NAN,
        }
    });
    let (rl_zm_sum, stats_rl_zm) = bench_runs(3, 15, soa_bytes, || {
        let rep = execute_tier2_with_zonemap(
            black_box(&multicore_prog),
            &[black_box(&soa_clustered)],
            Some(&clustered_zmap),
        );
        match rep.map(|r| r.output) {
            Ok(RuntimeValue::Scalar(v)) => v,
            _ => f64::NAN,
        }
    });

    for (label, obs) in [
        ("idiomatic_rust", idiom_sum),
        ("c_8acc", c8_sum),
        ("rl_simd", rl_simd_sum),
        ("rl_multicore", rl_mc_sum),
        ("rl_zonemap", rl_zm_sum),
    ] {
        let rel_diff = (obs - aos_sum).abs() / aos_sum.max(1.0);
        if rel_diff > 1e-9 {
            return Err(format!(
                "result mismatch on {label}: expected={aos_sum}, observed={obs}"
            ));
        }
    }

    println!(
        "  IMPLEMENTATION                                  | MEDIAN (ms) | MIN..MAX (ms)     | EFF. BW     | SPEEDUP vs C-8ACC | SPEEDUP vs AoS"
    );
    println!(
        "  ---------------------------------------------------------------------------------------------------------------------------------------"
    );
    for (name, st) in [
        ("[A] Legacy Imperative AoS + Vec::collect()", stats_aos),
        ("[B] Idiomatic Rust SoA (filter().map().sum())", stats_idiom),
        ("[C] Hand-Written C/Rust 8-Acc SoA (1-Core)", stats_c8),
        ("[D] Rul (.rl) Tier 2 Fused SIMD (1-Core)", stats_rl_simd),
        ("[E] Rul (.rl) Tier 2 Auto-MultiCore (All)", stats_rl_mc),
        ("[F] Rul (.rl) Tier 2 MultiCore + ZoneMap", stats_rl_zm),
    ] {
        println!(
            "  {name:<47} | {:>8.2} ms | {:>6.2}..{:<6.2} ms | {:>6.2} GB/s | {:>16.2}x | {:>13.2}x",
            st.median_ms,
            st.min_ms,
            st.max_ms,
            st.gbps,
            stats_c8.median_ms / st.median_ms.max(1e-6),
            stats_aos.median_ms / st.median_ms.max(1e-6),
        );
    }
    println!("  Verified Output Sum : {rl_mc_sum:.6e}\n");

    // 4. Print AI Agent Hardware Telemetry Reward Vector
    let rep_mc =
        execute_tier2_fused(&multicore_prog, &[&soa_amounts]).map_err(|e| e.to_string())?;
    println!("=== 3. AI AGENT HARDWARE TELEMETRY REWARD VECTOR ===");
    println!(
        "  {{ \"verify_ns\": {}, \"exec_ns\": {}, \"bandwidth_gbps\": {:.2}, \"interm_bytes\": {}, \"reward_score\": {:.2} }}\n",
        rep_mc.telemetry.verify_ns,
        rep_mc.telemetry.exec_ns,
        rep_mc.telemetry.effective_bandwidth_gbps,
        rep_mc.telemetry.intermediate_bytes_allocated,
        rep_mc.telemetry.reward_score
    );

    // 5. Run the 10-Pipeline Differential Suite across diverse operations & proof certificates
    let suite_reports = run_10_pipeline_suite(100_000).map_err(|e| e.to_string())?;
    print!("{}", format_suite10_table(&suite_reports));

    // 6. Run BM-HORNER-50 (50-Step Horner Polynomial: Mul+Add 104 Nodes vs Explicit FMA 54 Nodes, 10,000,000 Elements = 80 MB SoA)
    let horner_elems: u32 = 10_000_000;
    let horner_bytes = (horner_elems as usize) * 8;
    let horner_prog = build_horner50_pipeline(horner_elems, ExecPolicy::SimdStream)?;
    let horner_fma_prog = build_horner50_fma_pipeline(horner_elems, ExecPolicy::SimdStream)?;
    let horner_x: Vec<f64> = (0..horner_elems)
        .map(|i| 0.5 + (f64::from(i % 16) * 0.025))
        .collect();
    let horner_exe = "target/horner50_aot";
    let horner_fma_exe = "target/horner50_fma_aot";
    let horner_data = "target/horner50_data.bin";
    let horner_aot_rep =
        compile_rl_to_executable(&horner_prog, horner_exe).map_err(|e| e.to_string())?;
    let horner_fma_aot_rep =
        compile_rl_to_executable(&horner_fma_prog, horner_fma_exe).map_err(|e| e.to_string())?;
    write_arena_data_file(horner_data, &[&horner_x]).map_err(|e| e.to_string())?;
    let (h_c_val, h_c_stats) = bench_runs(2, 7, horner_bytes, || {
        run_handwritten_c_horner50(black_box(&horner_x))
    });
    let (h_t2_val, h_t2_stats) = bench_runs(2, 7, horner_bytes, || {
        match execute_tier2_fused(black_box(&horner_prog), &[black_box(&horner_x)])
            .map(|r| r.output)
        {
            Ok(RuntimeValue::Scalar(v)) => v,
            _ => f64::NAN,
        }
    });
    let (h_aot_val, h_aot_stats) = bench_runs(2, 7, horner_bytes, || {
        run_compiled_executable(horner_exe, horner_data).unwrap_or(f64::NAN)
    });
    let (h_fma_t2_val, h_fma_t2_stats) = bench_runs(2, 7, horner_bytes, || {
        match execute_tier2_fused(black_box(&horner_fma_prog), &[black_box(&horner_x)])
            .map(|r| r.output)
        {
            Ok(RuntimeValue::Scalar(v)) => v,
            _ => f64::NAN,
        }
    });
    let (h_fma_aot_val, h_fma_aot_stats) = bench_runs(2, 7, horner_bytes, || {
        run_compiled_executable(horner_fma_exe, horner_data).unwrap_or(f64::NAN)
    });
    for (label, obs) in [
        ("horner_tier2", h_t2_val),
        ("horner_aot", h_aot_val),
        ("horner_fma_tier2", h_fma_t2_val),
        ("horner_fma_aot", h_fma_aot_val),
    ] {
        let rel_diff = (obs - h_c_val).abs() / h_c_val.abs().max(1.0);
        if rel_diff > 1e-9 {
            return Err(format!(
                "BM-HORNER-50 mismatch on {label}: expected={h_c_val}, observed={obs}"
            ));
        }
    }
    println!(
        "\n=== 5. BM-HORNER-50 (50-STEP HORNER POLYNOMIAL, {horner_elems} ELEMS = 80 MB SoA) ==="
    );
    println!(
        "  AOT ELF64 (Mul+Add 104 Nodes) : {} B (MachineCode={} B, Compile={} ns)",
        horner_aot_rep.elf_bytes,
        horner_aot_rep.machine_code_bytes,
        horner_aot_rep.total_compile_ns
    );
    println!(
        "  AOT ELF64 (FMA 54 Nodes)      : {} B (MachineCode={} B, Compile={} ns)",
        horner_fma_aot_rep.elf_bytes,
        horner_fma_aot_rep.machine_code_bytes,
        horner_fma_aot_rep.total_compile_ns
    );
    println!(
        "  IMPLEMENTATION                                  | MEDIAN (ms) | MIN..MAX (ms)     | EFF. BW     | RATIO vs C-8ACC"
    );
    println!(
        "  --------------------------------------------------------------------------------------------------------------------"
    );
    for (name, st) in [
        (
            "[A] Hand-Written C/Rust 8-Acc Horner-50 (1-Core)",
            h_c_stats,
        ),
        ("[B] Rul (.rl) Tier 2 Fused Mul+Add (1-Core)", h_t2_stats),
        (
            "[C] Rul (.rl) AOT Native ELF64 Mul+Add (1-Core)",
            h_aot_stats,
        ),
        (
            "[D] Rul (.rl) Tier 2 Fused Explicit FMA (1-Core)",
            h_fma_t2_stats,
        ),
        (
            "[E] Rul (.rl) AOT Native ELF64 FMA (1-Core)",
            h_fma_aot_stats,
        ),
    ] {
        println!(
            "  {name:<47} | {:>8.2} ms | {:>6.2}..{:<6.2} ms | {:>6.2} GB/s | {:>14.2}x",
            st.median_ms,
            st.min_ms,
            st.max_ms,
            st.gbps,
            h_c_stats.median_ms / st.median_ms.max(1e-6),
        );
    }
    println!(
        "  Verified Horner-50 Output Sum : FMA_AOT={h_fma_aot_val:.6e} (MulAdd_AOT={h_aot_val:.6e}, C8={h_c_val:.6e})"
    );

    // 7. Run BM-TRANSCENDENTAL-LN-EXP (Vectorized 256-Bit AVX2 Ln + Exp, 1,000,000 Elements = 8 MB SoA)
    let trans_elems: u32 = 1_000_000;
    let trans_bytes = (trans_elems as usize) * 8;
    let trans_rls = format!(
        "\
        arena 0 len={trans_elems} lane=f64x8
        node $x = ArenaInput f64x8 a=0
        node $half = ConstF64 f64x1 const=0.5
        node $lnx = Ln f64x8 a=$x
        node $scaled = Mul f64x8 a=$lnx b=$half
        node $expx = Exp f64x8 a=$scaled
        node $sum = ReduceSum f64x1 a=$expx
        output $sum
        "
    );
    let trans_prog = assemble_rls(&trans_rls).map_err(|e| e.to_string())?;
    let trans_x: Vec<f64> = (0..trans_elems)
        .map(|i| 0.25 + (f64::from(i % 1024) * 0.01))
        .collect();
    let trans_exe = "target/transcendental_ln_exp_aot";
    let trans_data = "target/transcendental_ln_exp_data.bin";
    let trans_aot_rep =
        compile_rl_to_executable(&trans_prog, trans_exe).map_err(|e| e.to_string())?;
    write_arena_data_file(trans_data, &[&trans_x]).map_err(|e| e.to_string())?;

    let (tr_c_val, tr_c_stats) = bench_runs(3, 11, trans_bytes, || {
        let mut s = 0.0_f64;
        for &x in black_box(&trans_x) {
            s += (x.ln() * 0.5).exp();
        }
        s
    });
    let (tr_t2_val, tr_t2_stats) = bench_runs(3, 11, trans_bytes, || {
        match execute_tier2_fused(black_box(&trans_prog), &[black_box(&trans_x)]).map(|r| r.output)
        {
            Ok(RuntimeValue::Scalar(v)) => v,
            _ => f64::NAN,
        }
    });
    let (tr_aot_val, tr_aot_stats) = bench_runs(3, 11, trans_bytes, || {
        run_compiled_executable(trans_exe, trans_data).unwrap_or(f64::NAN)
    });
    for (label, obs) in [("trans_tier2", tr_t2_val), ("trans_aot", tr_aot_val)] {
        let rel_diff = (obs - tr_c_val).abs() / tr_c_val.abs().max(1.0);
        if rel_diff > 1e-11 {
            return Err(format!(
                "BM-TRANSCENDENTAL-LN-EXP mismatch on {label}: expected={tr_c_val}, observed={obs}"
            ));
        }
    }
    println!(
        "\n=== 6. BM-TRANSCENDENTAL-LN-EXP (VECTORIZED AVX2 LN + EXP, {trans_elems} ELEMS = 8 MB SoA) ==="
    );
    println!(
        "  AOT ELF64 Size     : {} B (MachineCode={} B, Compile={} ns)",
        trans_aot_rep.elf_bytes, trans_aot_rep.machine_code_bytes, trans_aot_rep.total_compile_ns
    );
    println!(
        "  IMPLEMENTATION                                  | MEDIAN (ms) | MIN..MAX (ms)     | EFF. BW     | SPEEDUP vs LIBM"
    );
    println!(
        "  --------------------------------------------------------------------------------------------------------------------"
    );
    for (name, st) in [
        ("[A] Rust/glibc Scalar x.ln() + exp() (1-Core)", tr_c_stats),
        ("[B] Rul (.rl) Tier 2 Fused Ln + Exp (1-Core)", tr_t2_stats),
        (
            "[C] Rul (.rl) AOT AVX2 Vectorized Ln+Exp (1-Core)",
            tr_aot_stats,
        ),
    ] {
        println!(
            "  {name:<47} | {:>8.2} ms | {:>6.2}..{:<6.2} ms | {:>6.2} GB/s | {:>14.2}x",
            st.median_ms,
            st.min_ms,
            st.max_ms,
            st.gbps,
            tr_c_stats.median_ms / st.median_ms.max(1e-6),
        );
    }
    println!(
        "  Verified Transcendental Sum : AOT={tr_aot_val:.9e} (Tier2={tr_t2_val:.9e}, libm={tr_c_val:.9e})"
    );

    Ok(())
}
