//! `rul` — Official Standalone Toolchain CLI for the Rul (`.rl`) Programming Language.
//!
//! Provides a complete Cargo-free / `rustc`-free project workflow for Rul (`.rl` / `.rls`):
//! - `rul new <dir>`      : Create a pure Rul project (`rul.toml`, `src/main.rls`, `data/arena_0.bin`)
//! - `rul check [path]`   : Run the single-pass O(N) static verifier
//! - `rul build [path]`   : Assemble `.rls` -> `.rl` and compile a standalone Linux `x86_64` AVX2 `ELF64` binary
//! - `rul run [path]`     : Execute `.rl`/`.rls` via Tier 2 Fused SIMD + native `ELF64` AOT with telemetry
//! - `rul autotune [path]`: Sweep hardware schedules (`AutoL1Tile`, `SimdStream`, `MultiCore`) for max reward
//! - `rul inspect <path>` : Disassemble a `.rl` or `.rls` program into a 128-bit `NodeFrame` table
//! - `rul decompile <rl>` : Decompile a `.rl` binary into lossless flat symbolic `.rls` text

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![deny(clippy::indexing_slicing)]

use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rul::aot::{compile_rl_to_executable, run_compiled_executable, write_arena_data_file};
use rul::bytecode::RulProgram;
use rul::engine::{
    autotune_schedule, execute_tier0_interpreter, execute_tier2_fused, RuntimeValue,
};
use rul::suite10::{load_rl_file, save_rl_file, MAX_RL_FILE_BYTES};
use rul::transpiler::{assemble_rls, disassemble_rl, program_to_rls};
use rul::verifier::verify_program;

const RUL_VERSION: &str = env!("CARGO_PKG_VERSION");

const DEFAULT_MAIN_RLS: &str = r#"# Rul Flat Symbolic IR (.rls) — Zero Rust / Zero Cargo
# Computes: sum_{x > 100.0} (x * x) on a 4,096-element f64 Columnar Arena
cap 0x0
arena 0 len=4096 lane=f64x8 mut=false align=64 ver=1
node $n0 = ArenaInput f64x8 fg=1 pol=simd proof=0 tok=read(0,1) a=0 b=0
node $n1 = ConstF64 f64x1 fg=1 pol=simd proof=0 tok=pure const=100.0
node $n2 = CmpGt maskx8 fg=1 pol=simd proof=0 tok=pure a=$n0 b=$n1
node $n3 = Mul f64x8 fg=1 pol=simd proof=0 tok=pure a=$n0 b=$n0
node $n4 = SelectZero f64x8 fg=1 pol=simd proof=0 tok=pure a=$n2 b=$n3
node $n5 = ReduceSum f64x1 fg=1 pol=simd proof=0 tok=pure a=$n4 b=0
node $n6 = AssertContract f64x1 fg=1 pol=simd proof=0 tok=pure a=$n5 b=0
output $n6
"#;

fn print_help() {
    println!(
        "\
rul {RUL_VERSION} — The AI-Native 128-Bit Binary Programming Language Toolchain (.rl)

USAGE:
    rul <COMMAND> [ARGS]

COMMANDS:
    new <dir>           Create a new pure Rul project (rul.toml, src/main.rls, data/arena_0.bin)
    check [file]        Run the single-pass O(N) static verifier (defaults to src/main.rls)
    build [file]        Pack .rls -> build/*.rl and compile standalone x86_64 AVX2 ELF64 binary
    run [file]          Verify, execute (Tier 2 SIMD + ELF64 AOT), and print HardwareTelemetry
    autotune [file]     Autotune hardware ScheduleHint (AutoL1Tile, SimdStream, MultiCore)
    inspect <file>      Disassemble a .rl or .rls file into a 128-bit NodeFrame table
    decompile <file.rl> Decompile a binary .rl graph into lossless .rls symbolic text
    version             Print the Rul toolchain version"
    );
}

fn read_bounded_utf8(path: &Path) -> Result<String, String> {
    let file = File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let mut buf = Vec::new();
    let max_limit = MAX_RL_FILE_BYTES;
    file.take(max_limit.saturating_add(1))
        .read_to_end(&mut buf)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    if u64::try_from(buf.len()).unwrap_or(u64::MAX) > max_limit {
        return Err(format!(
            "file {} exceeds maximum size limit ({MAX_RL_FILE_BYTES} bytes)",
            path.display()
        ));
    }
    String::from_utf8(buf).map_err(|e| format!("invalid UTF-8 in {}: {e}", path.display()))
}

fn read_bounded_f64_column(path: &Path, expected_len: usize) -> Result<Vec<f64>, String> {
    let file = File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let mut raw = Vec::new();
    let max_limit = MAX_RL_FILE_BYTES;
    file.take(max_limit.saturating_add(1))
        .read_to_end(&mut raw)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let expected_bytes = expected_len.saturating_mul(8);
    if raw.len() < expected_bytes {
        return Err(format!(
            "arena file {} has {} bytes, expected at least {} bytes ({} x f64)",
            path.display(),
            raw.len(),
            expected_bytes,
            expected_len
        ));
    }
    let mut out = Vec::with_capacity(expected_len);
    for &bytes in raw.as_chunks::<8>().0.iter().take(expected_len) {
        out.push(f64::from_le_bytes(bytes));
    }
    Ok(out)
}

fn resolve_entry_path(arg: Option<&str>) -> Result<PathBuf, String> {
    if let Some(p) = arg {
        return Ok(PathBuf::from(p));
    }
    let candidates = ["src/main.rls", "src/main.rl", "main.rls", "main.rl"];
    for c in candidates {
        let path = PathBuf::from(c);
        if path.exists() {
            return Ok(path);
        }
    }
    Err(
        "no entry file found (expected `src/main.rls` or pass a `.rls`/`.rl` path explicitly)"
            .to_owned(),
    )
}

fn load_program(path: &Path) -> Result<RulProgram, String> {
    let is_rls = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("rls"));
    if is_rls {
        let text = read_bounded_utf8(path)?;
        assemble_rls(&text).map_err(|e| e.to_string())
    } else {
        let path_str = path
            .to_str()
            .ok_or_else(|| format!("non-UTF8 path: {}", path.display()))?;
        load_rl_file(path_str).map_err(|e| e.to_string())
    }
}

fn cmd_new(project_arg: &str) -> Result<(), String> {
    let root = PathBuf::from(project_arg);
    if root.exists() {
        return Err(format!("directory `{}` already exists", root.display()));
    }
    let src_dir = root.join("src");
    let data_dir = root.join("data");
    let build_dir = root.join("build");
    fs::create_dir_all(&src_dir).map_err(|e| format!("failed to create src/: {e}"))?;
    fs::create_dir_all(&data_dir).map_err(|e| format!("failed to create data/: {e}"))?;
    fs::create_dir_all(&build_dir).map_err(|e| format!("failed to create build/: {e}"))?;

    let project_name = root
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("rul_app");

    let rul_toml = format!(
        "# Rul (.rl) Project Manifest — Zero Rust / Zero Cargo\n\
         [project]\n\
         name = \"{project_name}\"\n\
         version = \"0.1.0\"\n\
         entry = \"src/main.rls\"\n\
         binary_out = \"build/{project_name}.rl\"\n\
         native_out = \"build/{project_name}\"\n\
         arena_0 = \"data/arena_0.bin\"\n"
    );
    fs::write(root.join("rul.toml"), rul_toml)
        .map_err(|e| format!("failed to write rul.toml: {e}"))?;
    fs::write(src_dir.join("main.rls"), DEFAULT_MAIN_RLS)
        .map_err(|e| format!("failed to write src/main.rls: {e}"))?;
    fs::write(root.join(".gitignore"), "build/\n")
        .map_err(|e| format!("failed to write .gitignore: {e}"))?;

    // Generate deterministic 4,096-element f64 sample column in `data/arena_0.bin`
    let sample_col: Vec<f64> = (0..4096_u32).map(|i| f64::from(i % 250) + 0.5).collect();
    let arena_path = data_dir.join("arena_0.bin");
    let arena_bytes = write_arena_data_file(&arena_path, &[&sample_col])
        .map_err(|e| format!("failed to write data/arena_0.bin: {e}"))?;

    println!(
        "[RUL NEW] Created pure Rul (.rl) project `{project_name}` at `{}`",
        root.display()
    );
    println!("  ├── rul.toml          (Rul project manifest)");
    println!("  ├── src/main.rls      (7-node 128-bit symbolic SSA-DAG program)");
    println!("  └── data/arena_0.bin  ({arena_bytes} bytes = 4,096 x f64 columnar arena)");
    println!("\nNext steps:");
    println!("  cd {}", root.display());
    println!("  rul check");
    println!("  rul build");
    println!("  rul run");
    Ok(())
}

fn cmd_check(arg: Option<&str>) -> Result<(), String> {
    let entry = resolve_entry_path(arg)?;
    let t0 = Instant::now();
    let prog = load_program(&entry)?;
    let summary = verify_program(&prog).map_err(|e| e.to_string())?;
    let elapsed_ns = t0.elapsed().as_nanos();
    println!(
        "[RUL CHECK OK] {} | Nodes={} | FusionGroups={} | OutputShape={:?} | ZeroBoundsChecks={} | Time={elapsed_ns} ns",
        entry.display(),
        summary.node_count,
        summary.fusion_groups,
        summary.output_shape,
        summary.zero_runtime_bounds_checks
    );
    Ok(())
}

fn cmd_build(arg: Option<&str>) -> Result<(PathBuf, PathBuf, RulProgram), String> {
    let entry = resolve_entry_path(arg)?;
    let t0 = Instant::now();
    let prog = load_program(&entry)?;
    let summary = verify_program(&prog).map_err(|e| e.to_string())?;

    let stem = entry
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("program");
    let build_dir = PathBuf::from("build");
    fs::create_dir_all(&build_dir).map_err(|e| format!("failed to create build/: {e}"))?;

    let rl_out = build_dir.join(format!("{stem}.rl"));
    let exe_out = build_dir.join(stem);
    let rl_out_str = rl_out
        .to_str()
        .ok_or_else(|| "invalid UTF-8 output path".to_owned())?;
    let rl_bytes = save_rl_file(rl_out_str, &prog).map_err(|e| e.to_string())?;
    let aot_rep = compile_rl_to_executable(&prog, &exe_out).map_err(|e| e.to_string())?;
    let total_us = t0.elapsed().as_secs_f64() * 1e6;

    println!(
        "[RUL BUILD OK] {} -> {} ({rl_bytes} B) + {} ({} B ELF64, AVX2={}) | Nodes={} | Verify={} ns | Codegen={} ns | Total={total_us:.1} us",
        entry.display(),
        rl_out.display(),
        exe_out.display(),
        aot_rep.elf_bytes,
        aot_rep.avx2_vectorized,
        summary.node_count,
        aot_rep.verify_ns,
        aot_rep.codegen_ns
    );
    Ok((rl_out, exe_out, prog))
}

fn load_or_synthesize_arenas(prog: &RulProgram) -> Result<(Vec<Vec<f64>>, PathBuf), String> {
    let data_dir = PathBuf::from("data");
    let default_arena_file = data_dir.join("arena_0.bin");

    if prog.arenas.len() == 1 {
        if let Some(first_desc) = prog.arenas.first() {
            let len = first_desc.element_count as usize;
            if default_arena_file.exists() {
                if let Ok(col) = read_bounded_f64_column(&default_arena_file, len) {
                    return Ok((vec![col], default_arena_file));
                }
            }
        }
    }

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

    let build_dir = PathBuf::from("build");
    fs::create_dir_all(&build_dir).map_err(|e| format!("failed to create build/: {e}"))?;
    let synth_path = build_dir.join("arena_input.bin");
    let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
    write_arena_data_file(&synth_path, &refs).map_err(|e| e.to_string())?;
    Ok((cols, synth_path))
}

fn cmd_run(arg: Option<&str>) -> Result<(), String> {
    let (_rl_path, exe_path, prog) = cmd_build(arg)?;
    let (cols, arena_file) = load_or_synthesize_arenas(&prog)?;
    let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();

    let tier0 = execute_tier0_interpreter(&prog, &refs).map_err(|e| e.to_string())?;
    let tier2 = execute_tier2_fused(&prog, &refs).map_err(|e| e.to_string())?;

    let t_aot = Instant::now();
    let aot_val = run_compiled_executable(&exe_path, &arena_file).map_err(|e| e.to_string())?;
    let aot_us = t_aot.elapsed().as_secs_f64() * 1e6;

    let tier0_scalar = match tier0.output {
        RuntimeValue::Scalar(v) => v,
        RuntimeValue::ColumnF64(ref c) => c.iter().copied().sum(),
        RuntimeValue::ColumnMask(ref m) => m.iter().filter(|&&b| b).count() as f64,
    };
    let tier2_scalar = match tier2.output {
        RuntimeValue::Scalar(v) => v,
        RuntimeValue::ColumnF64(ref c) => c.iter().copied().sum(),
        RuntimeValue::ColumnMask(ref m) => m.iter().filter(|&&b| b).count() as f64,
    };

    println!(
        "[RUL RUN OK] Output={tier2_scalar:.6} (Tier0 Oracle={tier0_scalar:.6}, Native ELF64={aot_val:.6} in {aot_us:.1} us)"
    );
    println!(
        "[TELEMETRY]  verify_ns={} | tier2_exec_ns={} | bw={:.2} GB/s | interm_alloc={} B | reward={:.2}",
        tier2.telemetry.verify_ns,
        tier2.telemetry.exec_ns,
        tier2.telemetry.effective_bandwidth_gbps,
        tier2.telemetry.intermediate_bytes_allocated,
        tier2.telemetry.reward_score
    );
    Ok(())
}

fn cmd_autotune(arg: Option<&str>) -> Result<(), String> {
    let entry = resolve_entry_path(arg)?;
    let mut prog = load_program(&entry)?;
    let (cols, _arena_file) = load_or_synthesize_arenas(&prog)?;
    let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
    let report = autotune_schedule(&mut prog, &refs, None).map_err(|e| e.to_string())?;
    println!(
        "[RUL AUTOTUNE OK] BestPolicy={:?} (group={}) | ExecNs={} | Bandwidth={:.2} GB/s | RewardScore={:.2}",
        report.best_schedule.policy,
        report.best_schedule.fusion_group,
        report.best_telemetry.exec_ns,
        report.best_telemetry.effective_bandwidth_gbps,
        report.best_telemetry.reward_score
    );
    for c in &report.candidates {
        println!(
            "  - Policy={:<12} | ExecNs={:<8} | BW={:>6.2} GB/s | Reward={:.2}",
            format!("{:?}", c.policy),
            c.telemetry.exec_ns,
            c.telemetry.effective_bandwidth_gbps,
            c.telemetry.reward_score
        );
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let Some(cmd) = args.get(1).map(String::as_str) else {
        print_help();
        return Ok(());
    };

    match cmd {
        "new" | "init" => {
            let dir = args
                .get(2)
                .map(String::as_str)
                .ok_or_else(|| "usage: rul new <project_dir>".to_owned())?;
            cmd_new(dir)
        }
        "check" | "verify" => cmd_check(args.get(2).map(String::as_str)),
        "build" | "compile" => {
            let _ = cmd_build(args.get(2).map(String::as_str))?;
            Ok(())
        }
        "run" => cmd_run(args.get(2).map(String::as_str)),
        "autotune" => cmd_autotune(args.get(2).map(String::as_str)),
        "inspect" => {
            let path = resolve_entry_path(args.get(2).map(String::as_str))?;
            let prog = load_program(&path)?;
            print!("{}", disassemble_rl(&prog));
            Ok(())
        }
        "decompile" => {
            let path_str = args
                .get(2)
                .map(String::as_str)
                .ok_or_else(|| "usage: rul decompile <file.rl>".to_owned())?;
            let prog = load_rl_file(path_str).map_err(|e| e.to_string())?;
            print!("{}", program_to_rls(&prog));
            Ok(())
        }
        "version" | "--version" | "-V" => {
            println!("rul {RUL_VERSION} (128-bit schedule- and proof-carrying binary toolchain)");
            Ok(())
        }
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        other => Err(format!(
            "unknown subcommand `{other}` (run `rul help` for available commands)"
        )),
    }
}
