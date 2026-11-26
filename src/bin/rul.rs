//! `rul` — Official Standalone Toolchain CLI for the Rul (`.rl`) Programming Language.
//!
//! Provides a complete Cargo-free / `rustc`-free project workflow for Rul (`.rl` / `.rls`):
//! - `rul new <dir>`      : Create a pure Rul project (`rul.toml`, `src/main.rls`, `data/arena_0.bin`)
//! - `rul check [path]`   : Run the single-pass O(N) static verifier
//! - `rul link <mods...> -o <out.rl>`: Verify and link modular `.rl`/`.rls` libraries into one verified `.rl`
//! - `rul build [path]`   : Assemble `.rls` -> `.rl` and compile a standalone Linux `x86_64` AVX2 `ELF64` binary
//! - `rul run [path]`     : Execute `.rl`/`.rls` via Tier 2 Fused SIMD + native `ELF64` AOT with telemetry
//! - `rul autotune [path]`: Sweep hardware schedules (`AutoL1Tile`, `SimdStream`, `MultiCore`) for max reward
//! - `rul trace [path]`   : Run white-box node-by-node execution trace (`min`, `max`, `sum`, `NaN/Inf`, samples)
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
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use rul::aot::{
    compile_rl_to_executable, run_compiled_executable, run_compiled_scalar_executable,
    run_compiled_structured_executable, write_arena_data_file,
};
use rul::bytecode::{ElemType, RulProgram};
use rul::engine::{
    autotune_schedule, execute_tier0_interpreter, execute_tier0_interpreter_with_input,
    execute_tier2_fused, format_trace_table, trace_program, RuntimeValue,
};
use rul::linker::{link_modules, LinkOptions};
use rul::mcp::escape_json_string;
use rul::suite10::{load_rl_file, save_rl_file, MAX_RL_FILE_BYTES};
use rul::transpiler::{assemble_rls, disassemble_rl, program_to_rls};
use rul::verifier::{verify_program, StaticCostSummary, VerifiedSummary};

const RUL_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Maximum allowed byte size for an input `.bin` columnar arena data file (`1 GiB`),
/// allowing 10M+ `f64` element benchmark arenas (`80 MB+`) while bounding memory usage.
const MAX_ARENA_DATA_BYTES: u64 = 1024 * 1024 * 1024;

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
    new <dir>             Create a new pure Rul project (rul.toml, src/main.rls, data/arena_0.bin)
    check [--json] [file] Run the single-pass O(N) static verifier (defaults to src/main.rls)
    link <mods...> -o <out.rl> [--entry <name>] [--cap <hex>]
                          Verify and link modular .rl/.rls files into a single verified .rl executable
    build [file]        Pack .rls -> build/*.rl and compile standalone x86_64 AVX2 ELF64 binary
    run [file]          Verify, execute (Tier 2 SIMD + ELF64 AOT), and print HardwareTelemetry
    autotune [file]     Autotune hardware ScheduleHint (AutoL1Tile, SimdStream, MultiCore)
    trace [file]        Execute in white-box trace mode (inspect per-node min/max/sum/NaN/samples)
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
    let max_limit = MAX_ARENA_DATA_BYTES;
    file.take(max_limit.saturating_add(1))
        .read_to_end(&mut raw)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    if u64::try_from(raw.len()).unwrap_or(u64::MAX) > max_limit {
        return Err(format!(
            "arena file {} exceeds maximum size limit ({MAX_ARENA_DATA_BYTES} bytes)",
            path.display()
        ));
    }
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

fn format_static_cost_inline(static_cost: Option<StaticCostSummary>) -> String {
    match static_cost {
        Some(c) => format!(
            "Cost(steps={}, depth={}, stack_units={}, stack_bytes={}, region_bytes={})",
            c.worst_case_steps,
            c.max_call_depth,
            c.max_stack_units,
            c.max_stack_bytes,
            c.max_region_bytes
        ),
        None => "Cost=unbounded(CAP_DIVERGE)".to_owned(),
    }
}

fn format_check_json_ok(entry: &Path, summary: &VerifiedSummary, verify_ns: u128) -> String {
    let file_esc = escape_json_string(&entry.display().to_string());
    let shape_esc = escape_json_string(&format!("{:?}", summary.output_shape));
    let elem_esc = escape_json_string(&format!("{:?}", summary.output_elem));
    let cost_json = match summary.static_cost {
        Some(c) => format!(
            "{{\"worst_case_steps\":{},\"max_call_depth\":{},\"max_stack_units\":{},\"max_stack_bytes\":{},\"max_region_bytes\":{}}}",
            c.worst_case_steps,
            c.max_call_depth,
            c.max_stack_units,
            c.max_stack_bytes,
            c.max_region_bytes
        ),
        None => "null".to_owned(),
    };
    format!(
        "{{\"status\":\"ok\",\"file\":\"{file_esc}\",\"node_count\":{},\"fusion_groups\":{},\"output_shape\":\"{shape_esc}\",\"output_elem\":\"{elem_esc}\",\"zero_runtime_bounds_checks\":{},\"verify_ns\":{verify_ns},\"static_cost\":{cost_json}}}",
        summary.node_count,
        summary.fusion_groups,
        summary.zero_runtime_bounds_checks
    )
}

fn format_check_json_err(entry_label: &str, message: &str) -> String {
    let file_esc = escape_json_string(entry_label);
    let err_esc = escape_json_string(message);
    format!("{{\"status\":\"error\",\"file\":\"{file_esc}\",\"error\":\"{err_esc}\"}}")
}

fn parse_check_args(args: &[String]) -> Result<(bool, Option<&str>), String> {
    let mut json_mode = false;
    let mut file_arg: Option<&str> = None;
    for raw in args {
        let a = raw.as_str();
        if a == "--json" {
            json_mode = true;
        } else if a.starts_with('-') {
            return Err(format!("unknown flag `{a}` for `rul check`"));
        } else if file_arg.is_none() {
            file_arg = Some(a);
        } else {
            return Err(format!("unexpected extra argument `{a}` for `rul check`"));
        }
    }
    Ok((json_mode, file_arg))
}

fn cmd_check(args: &[String]) -> Result<ExitCode, String> {
    let json_hint = args.iter().any(|a| a == "--json");
    let (json_mode, file_arg) = match parse_check_args(args) {
        Ok(parsed) => parsed,
        Err(e) => {
            if json_hint {
                println!("{}", format_check_json_err("", &e));
                return Ok(ExitCode::FAILURE);
            }
            return Err(e);
        }
    };
    let entry = match resolve_entry_path(file_arg) {
        Ok(p) => p,
        Err(e) => {
            if json_mode {
                println!("{}", format_check_json_err(file_arg.unwrap_or(""), &e));
                return Ok(ExitCode::FAILURE);
            }
            return Err(e);
        }
    };
    let entry_label = entry.display().to_string();
    let t0 = Instant::now();
    let prog = match load_program(&entry) {
        Ok(p) => p,
        Err(e) => {
            if json_mode {
                println!("{}", format_check_json_err(&entry_label, &e));
                return Ok(ExitCode::FAILURE);
            }
            return Err(e);
        }
    };
    let summary = match verify_program(&prog) {
        Ok(s) => s,
        Err(e) => {
            let msg = e.to_string();
            if json_mode {
                println!("{}", format_check_json_err(&entry_label, &msg));
                return Ok(ExitCode::FAILURE);
            }
            return Err(msg);
        }
    };
    let elapsed_ns = t0.elapsed().as_nanos();
    if json_mode {
        println!("{}", format_check_json_ok(&entry, &summary, elapsed_ns));
    } else {
        let cost_inline = format_static_cost_inline(summary.static_cost);
        println!(
            "[RUL CHECK OK] {} | Nodes={} | FusionGroups={} | OutputShape={:?} | ZeroBoundsChecks={} | {cost_inline} | Time={elapsed_ns} ns",
            entry.display(),
            summary.node_count,
            summary.fusion_groups,
            summary.output_shape,
            summary.zero_runtime_bounds_checks
        );
    }
    Ok(ExitCode::SUCCESS)
}

fn parse_cap_flag(raw: &str) -> Result<u64, String> {
    if let Some(hex) = raw.strip_prefix("0x").or_else(|| raw.strip_prefix("0X")) {
        if hex.is_empty() {
            return Err(format!("invalid --cap hex value `{raw}`"));
        }
        u64::from_str_radix(hex, 16).map_err(|_e| format!("invalid --cap hex value `{raw}`"))
    } else {
        raw.parse::<u64>()
            .map_err(|_e| format!("invalid --cap value `{raw}` (expected hex like 0x0 or 0x8)"))
    }
}

fn cmd_link(args: &[String]) -> Result<(), String> {
    let mut input_paths: Vec<PathBuf> = Vec::new();
    let mut out_path: Option<PathBuf> = None;
    let mut entry_export: Option<String> = None;
    let mut capability_mask: Option<u64> = None;

    let mut i = 0usize;
    while i < args.len() {
        let Some(tok) = args.get(i).map(String::as_str) else {
            break;
        };
        match tok {
            "-o" | "--output" => {
                if out_path.is_some() {
                    return Err("duplicate `-o` output path for `rul link`".to_owned());
                }
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "missing output path after `-o`".to_owned())?;
                out_path = Some(PathBuf::from(val));
            }
            "--entry" => {
                if entry_export.is_some() {
                    return Err("duplicate `--entry` flag for `rul link`".to_owned());
                }
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "missing export name after `--entry`".to_owned())?;
                entry_export = Some(val.clone());
            }
            "--cap" => {
                if capability_mask.is_some() {
                    return Err("duplicate `--cap` flag for `rul link`".to_owned());
                }
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "missing hex mask after `--cap`".to_owned())?;
                capability_mask = Some(parse_cap_flag(val)?);
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown flag `{other}` for `rul link`"));
            }
            path_str => {
                input_paths.push(PathBuf::from(path_str));
            }
        }
        i += 1;
    }

    if input_paths.is_empty() {
        return Err(
            "usage: rul link <mod1> <mod2> ... -o <out.rl> [--entry <name>] [--cap <hex>]"
                .to_owned(),
        );
    }
    let out_file = out_path.ok_or_else(|| {
        "missing required `-o <out.rl>` output path (usage: rul link <mod1> <mod2> ... -o <out.rl> [--entry <name>] [--cap <hex>])"
            .to_owned()
    })?;

    let t0 = Instant::now();
    let mut modules = Vec::with_capacity(input_paths.len());
    for path in &input_paths {
        let prog = load_program(path).map_err(|e| format!("{}: {e}", path.display()))?;
        modules.push(prog);
    }

    let options = LinkOptions {
        entry_export,
        capability_mask,
    };
    let report = link_modules(&modules, &options).map_err(|e| e.to_string())?;

    if let Some(parent) = out_file.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
        }
    }
    let out_str = out_file
        .to_str()
        .ok_or_else(|| format!("non-UTF8 output path: {}", out_file.display()))?;
    let out_bytes = save_rl_file(out_str, &report.program).map_err(|e| e.to_string())?;
    let elapsed_us = t0.elapsed().as_secs_f64() * 1e6;

    println!(
        "[RUL LINK OK] -> {} ({out_bytes} B) | Cap={:#x} | Modules={} | Funcs={} | Regions={} | ConstData={} B | Time={elapsed_us:.1} us",
        out_file.display(),
        report.capability_mask,
        report.module_count,
        report.func_count,
        report.region_count,
        report.const_data_bytes,
    );
    Ok(())
}

fn cmd_build(arg: Option<&str>) -> Result<(PathBuf, PathBuf, RulProgram, VerifiedSummary), String> {
    build_program(arg, false)
}

fn build_program(
    arg: Option<&str>,
    status_to_stderr_if_output_region: bool,
) -> Result<(PathBuf, PathBuf, RulProgram, VerifiedSummary), String> {
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

    if status_to_stderr_if_output_region && prog.has_output_region() {
        eprintln!(
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
    } else {
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
    }
    Ok((rl_out, exe_out, prog, summary))
}

fn load_or_synthesize_arenas_in_dir(
    prog: &RulProgram,
    data_dir: &Path,
) -> Result<(Vec<Vec<f64>>, bool), String> {
    if prog.arenas.is_empty() {
        return Ok((Vec::new(), false));
    }
    // Load every declared arena from `<data_dir>/arena_<idx>.bin`, for any arena
    // count. Real user data must never be silently replaced by synthetic
    // data: a program that declares N arenas runs on the user's N columns,
    // fails loudly, or — only when no arena files exist at all — runs on
    // clearly-labelled synthetic demo data.
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(prog.arenas.len());
    let mut loaded = 0usize;
    for (idx, desc) in prog.arenas.iter().enumerate() {
        let path = data_dir.join(format!("arena_{idx}.bin"));
        if path.exists() {
            let col = read_bounded_f64_column(&path, desc.element_count as usize)
                .map_err(|e| format!("arena #{idx}: {e}"))?;
            cols.push(col);
            loaded += 1;
        }
    }
    if !prog.arenas.is_empty() && loaded == prog.arenas.len() {
        return Ok((cols, false));
    }
    if loaded > 0 {
        return Err(format!(
            "found data files for {loaded} of {} declared arenas; refusing to mix real and synthetic data — provide data/arena_<idx>.bin for every arena, or remove all arena files to run on synthetic demo data",
            prog.arenas.len()
        ));
    }

    eprintln!(
        "[RUL WARN] No data/arena_*.bin files found — executing on SYNTHETIC demo data; results are NOT computed from user data."
    );
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

    Ok((cols, true))
}

fn load_or_synthesize_arenas(prog: &RulProgram) -> Result<(Vec<Vec<f64>>, bool), String> {
    load_or_synthesize_arenas_in_dir(prog, Path::new("data"))
}

fn prepare_aot_arena_file(
    cols: &[Vec<f64>],
    data_dir: &Path,
    is_synthetic: bool,
) -> Result<PathBuf, String> {
    // Fast zero-copy path: when a single real arena file is exactly sized on disk,
    // pass `data/arena_0.bin` directly to the standalone ELF64 binary without rewriting `build/arena_input.bin`.
    if !is_synthetic && cols.len() == 1 {
        if let Some(first_col) = cols.first() {
            let direct_path = data_dir.join("arena_0.bin");
            let expected_bytes = u64::try_from(first_col.len().saturating_mul(8)).unwrap_or(0);
            if let Ok(meta) = fs::metadata(&direct_path) {
                if meta.len() == expected_bytes {
                    return Ok(direct_path);
                }
            }
        }
    }

    let build_dir = PathBuf::from("build");
    fs::create_dir_all(&build_dir).map_err(|e| format!("failed to create build/: {e}"))?;
    let combined_path = build_dir.join("arena_input.bin");
    let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
    write_arena_data_file(&combined_path, &refs).map_err(|e| e.to_string())?;
    Ok(combined_path)
}

fn scalars_equivalent(expected: f64, actual: f64) -> bool {
    if expected.is_nan() && actual.is_nan() {
        return true;
    }
    if expected.is_nan() || actual.is_nan() {
        return false;
    }
    if expected.is_infinite() || actual.is_infinite() {
        return expected.is_infinite()
            && actual.is_infinite()
            && expected.is_sign_positive() == actual.is_sign_positive();
    }
    let diff = (expected - actual).abs();
    let scale = expected.abs().max(actual.abs()).max(1.0);
    diff <= 1e-9 * scale
}

fn scalar_outputs_equivalent(output_elem: ElemType, expected_bits: u64, actual_bits: u64) -> bool {
    match output_elem {
        ElemType::F64 => {
            scalars_equivalent(f64::from_bits(expected_bits), f64::from_bits(actual_bits))
        }
        ElemType::I64 | ElemType::U8 | ElemType::BoolMask => expected_bits == actual_bits,
    }
}

fn read_bounded_input_bytes(reader: impl Read, max_len: u32) -> Result<Vec<u8>, String> {
    let cap_bytes = u64::from(max_len).saturating_add(1);
    let mut buf = Vec::new();
    reader
        .take(cap_bytes)
        .read_to_end(&mut buf)
        .map_err(|e| format!("failed to read stdin for input region: {e}"))?;
    Ok(buf)
}

fn cmd_run(arg: Option<&str>) -> Result<(), String> {
    let (_rl_path, exe_path, prog, summary) = build_program(arg, true)?;
    let (cols, is_synthetic) = load_or_synthesize_arenas(&prog)?;
    let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();

    if prog.uses_structured_control_or_locals() {
        let stdin_bytes = if let Some(max_len) = prog.input_region_max_len() {
            read_bounded_input_bytes(io::stdin(), max_len)?
        } else {
            Vec::new()
        };
        let tier0 = execute_tier0_interpreter_with_input(&prog, &refs, &stdin_bytes)
            .map_err(|e| e.to_string())?;
        let t_aot = Instant::now();
        if prog.has_output_region() {
            let aot_bytes = run_compiled_structured_executable(&exe_path, &stdin_bytes)
                .map_err(|e| e.to_string())?;
            let t0_bytes = tier0.output_bytes.as_deref().unwrap_or(&[]);
            if t0_bytes != aot_bytes.as_slice() {
                return Err(format!(
                    "[RUL RUN ERROR] differential equivalence failed (OutputRegion): Tier0 len={} != Native ELF64 len={}",
                    t0_bytes.len(),
                    aot_bytes.len()
                ));
            }
            io::stdout()
                .write_all(&aot_bytes)
                .map_err(|e| format!("failed to write output region to stdout: {e}"))?;
            return Ok(());
        }
        let aot_bytes = if prog.has_io_regions() {
            run_compiled_structured_executable(&exe_path, &stdin_bytes)
                .map_err(|e| e.to_string())?
        } else {
            run_compiled_scalar_executable(&exe_path)
                .map_err(|e| e.to_string())?
                .to_le_bytes()
                .to_vec()
        };
        let aot_us = t_aot.elapsed().as_secs_f64() * 1e6;
        let aot_bits = u64::from_le_bytes(aot_bytes.as_slice().try_into().map_err(|_e| {
            format!(
                "[RUL RUN ERROR] expected 8-byte scalar output, got {} bytes",
                aot_bytes.len()
            )
        })?);
        let tier0_bits = match tier0.output.as_typed_scalar() {
            Some(ts) => ts.bits,
            None => match tier0.output {
                RuntimeValue::ColumnF64(ref c) => c.iter().copied().sum::<f64>().to_bits(),
                RuntimeValue::ColumnMask(ref m) => {
                    (m.iter().filter(|&&b| b).count() as f64).to_bits()
                }
                RuntimeValue::Scalar(v) => v.to_bits(),
                RuntimeValue::TypedScalar(ts) => ts.bits,
            },
        };
        if !scalar_outputs_equivalent(summary.output_elem, tier0_bits, aot_bits) {
            return Err(format!(
                "[RUL RUN ERROR] differential equivalence failed ({:?}): Tier0 Oracle=0x{tier0_bits:016x}, Native ELF64=0x{aot_bits:016x}",
                summary.output_elem,
            ));
        }
        match summary.output_elem {
            ElemType::I64 => {
                let out_i64 = i64::from_ne_bytes(tier0_bits.to_ne_bytes());
                println!(
                    "[RUL RUN OK] Output_i64={out_i64} (bits=0x{tier0_bits:016x}, Native ELF64 in {aot_us:.1} us)"
                );
            }
            ElemType::U8 => {
                let out_u8 = (tier0_bits & 0xFF) as u8;
                println!(
                    "[RUL RUN OK] Output_u8={out_u8} (bits=0x{tier0_bits:016x}, Native ELF64 in {aot_us:.1} us)"
                );
            }
            ElemType::BoolMask => {
                let out_bool = tier0_bits != 0;
                println!(
                    "[RUL RUN OK] Output_bool={out_bool} (bits=0x{tier0_bits:016x}, Native ELF64 in {aot_us:.1} us)"
                );
            }
            ElemType::F64 => {
                let tier0_scalar = f64::from_bits(tier0_bits);
                println!(
                    "[RUL RUN OK] Output={tier0_scalar:.6} (bits=0x{tier0_bits:016x}, Native ELF64 in {aot_us:.1} us)"
                );
            }
        }
        return Ok(());
    }

    let arena_file = prepare_aot_arena_file(&cols, Path::new("data"), is_synthetic)?;
    let tier0 = execute_tier0_interpreter(&prog, &refs).map_err(|e| e.to_string())?;
    let tier2 = execute_tier2_fused(&prog, &refs).map_err(|e| e.to_string())?;

    let t_aot = Instant::now();
    let aot_val = run_compiled_executable(&exe_path, &arena_file).map_err(|e| e.to_string())?;
    let aot_us = t_aot.elapsed().as_secs_f64() * 1e6;

    let tier0_scalar = match tier0.output {
        RuntimeValue::Scalar(v) => v,
        RuntimeValue::TypedScalar(ts) => f64::from_bits(ts.bits),
        RuntimeValue::ColumnF64(ref c) => c.iter().copied().sum(),
        RuntimeValue::ColumnMask(ref m) => m.iter().filter(|&&b| b).count() as f64,
    };
    let tier2_scalar = match tier2.output {
        RuntimeValue::Scalar(v) => v,
        RuntimeValue::TypedScalar(ts) => f64::from_bits(ts.bits),
        RuntimeValue::ColumnF64(ref c) => c.iter().copied().sum(),
        RuntimeValue::ColumnMask(ref m) => m.iter().filter(|&&b| b).count() as f64,
    };

    if !scalars_equivalent(tier0_scalar, tier2_scalar) || !scalars_equivalent(tier0_scalar, aot_val)
    {
        return Err(format!(
            "[RUL RUN ERROR] differential equivalence failed: Tier0 Oracle={tier0_scalar:.6}, Tier2={tier2_scalar:.6}, Native ELF64={aot_val:.6}"
        ));
    }

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
    let (cols, _is_synthetic) = load_or_synthesize_arenas(&prog)?;
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

fn cmd_trace(arg: Option<&str>) -> Result<(), String> {
    let entry = resolve_entry_path(arg)?;
    let prog = load_program(&entry)?;
    let (cols, _is_synthetic) = load_or_synthesize_arenas(&prog)?;
    let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
    let report = trace_program(&prog, &refs).map_err(|e| e.to_string())?;
    print!("{}", format_trace_table(&report));
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let Some(cmd) = args.get(1).map(String::as_str) else {
        print_help();
        return ExitCode::SUCCESS;
    };

    let result: Result<ExitCode, String> = match cmd {
        "new" | "init" => {
            let dir = args
                .get(2)
                .map(String::as_str)
                .ok_or_else(|| "usage: rul new <project_dir>".to_owned());
            dir.and_then(cmd_new).map(|()| ExitCode::SUCCESS)
        }
        "check" | "verify" => cmd_check(args.get(2..).unwrap_or(&[])),
        "link" => cmd_link(args.get(2..).unwrap_or(&[])).map(|()| ExitCode::SUCCESS),
        "build" | "compile" => {
            cmd_build(args.get(2).map(String::as_str)).map(|_| ExitCode::SUCCESS)
        }
        "run" => cmd_run(args.get(2).map(String::as_str)).map(|()| ExitCode::SUCCESS),
        "autotune" => cmd_autotune(args.get(2).map(String::as_str)).map(|()| ExitCode::SUCCESS),
        "trace" => cmd_trace(args.get(2).map(String::as_str)).map(|()| ExitCode::SUCCESS),
        "inspect" => resolve_entry_path(args.get(2).map(String::as_str))
            .and_then(|path| load_program(&path))
            .map(|prog| {
                print!("{}", disassemble_rl(&prog));
                ExitCode::SUCCESS
            }),
        "decompile" => args
            .get(2)
            .map(String::as_str)
            .ok_or_else(|| "usage: rul decompile <file.rl>".to_owned())
            .and_then(|path_str| load_rl_file(path_str).map_err(|e| e.to_string()))
            .map(|prog| {
                print!("{}", program_to_rls(&prog));
                ExitCode::SUCCESS
            }),
        "version" | "--version" | "-V" => {
            println!("rul {RUL_VERSION} (128-bit schedule- and proof-carrying binary toolchain)");
            Ok(ExitCode::SUCCESS)
        }
        "help" | "--help" | "-h" => {
            print_help();
            Ok(ExitCode::SUCCESS)
        }
        other => Err(format!(
            "unknown subcommand `{other}` (run `rul help` for available commands)"
        )),
    };

    match result {
        Ok(code) => code,
        Err(msg) => {
            eprintln!("Error: {msg}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_arena_loading_partial_rejection_and_zero_copy_aot_path() -> Result<(), String> {
        let rls_two = "\
            arena 0 len=4 lane=f64x8
            arena 1 len=4 lane=f64x8
            node $a = ArenaInput f64x8 a=0
            node $b = ArenaInput f64x8 a=1
            node $p = Mul f64x8 a=$a b=$b
            node $s = ReduceSum f64x1 a=$p
            output $s
        ";
        let prog_two = assemble_rls(rls_two).map_err(|e| e.to_string())?;

        let tmp_dir =
            std::env::temp_dir().join(format!("rul_cli_test_arenas_dir_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp_dir);
        fs::create_dir_all(&tmp_dir).map_err(|e| e.to_string())?;

        // 1. No arena files -> returns synthetic demo columns (`is_synthetic == true`)
        let (cols, is_synth) = load_or_synthesize_arenas_in_dir(&prog_two, &tmp_dir)?;
        assert!(is_synth);
        assert_eq!(cols.len(), 2);

        // 2. Partial arena files (`arena_0.bin` present, `arena_1.bin` missing) -> hard error
        let col0 = [10.0_f64, 20.0, 30.0, 40.0];
        let mut bytes0 = Vec::with_capacity(32);
        for v in col0 {
            bytes0.extend_from_slice(&v.to_le_bytes());
        }
        let f0 = tmp_dir.join("arena_0.bin");
        fs::write(&f0, &bytes0).map_err(|e| e.to_string())?;

        let partial_res = load_or_synthesize_arenas_in_dir(&prog_two, &tmp_dir);
        assert!(partial_res.is_err());
        let Err(msg) = partial_res else {
            return Err("expected partial arena error".to_owned());
        };
        assert!(msg.contains("refusing to mix real and synthetic data"));

        // 3. Single-arena real data -> `prepare_aot_arena_file` returns `arena_0.bin` directly (zero-copy)
        let rls_one = "\
            arena 0 len=4 lane=f64x8
            node $a = ArenaInput f64x8 a=0
            node $s = ReduceSum f64x1 a=$a
            output $s
        ";
        let prog_one = assemble_rls(rls_one).map_err(|e| e.to_string())?;
        let (cols1, is_synth1) = load_or_synthesize_arenas_in_dir(&prog_one, &tmp_dir)?;
        assert!(!is_synth1);
        assert_eq!(cols1, vec![col0.to_vec()]);
        let aot_path = prepare_aot_arena_file(&cols1, &tmp_dir, is_synth1)?;
        assert_eq!(aot_path, f0);

        // 4. All 2 arena files present -> returns exact user columns (`is_synthetic == false`)
        let col1 = [1.0_f64, 2.0, 3.0, 4.0];
        let mut bytes1 = Vec::with_capacity(32);
        for v in col1 {
            bytes1.extend_from_slice(&v.to_le_bytes());
        }
        let f1 = tmp_dir.join("arena_1.bin");
        fs::write(&f1, &bytes1).map_err(|e| e.to_string())?;

        let (cols2, is_synth2) = load_or_synthesize_arenas_in_dir(&prog_two, &tmp_dir)?;
        assert!(!is_synth2);
        assert_eq!(cols2, vec![col0.to_vec(), col1.to_vec()]);

        let _ = fs::remove_dir_all(&tmp_dir);
        Ok(())
    }

    #[test]
    fn test_scalars_equivalent_ieee754_and_tolerance() {
        assert!(scalars_equivalent(f64::NAN, f64::NAN));
        assert!(!scalars_equivalent(f64::NAN, 0.0));
        assert!(!scalars_equivalent(0.0, f64::NAN));
        assert!(scalars_equivalent(f64::INFINITY, f64::INFINITY));
        assert!(scalars_equivalent(f64::NEG_INFINITY, f64::NEG_INFINITY));
        assert!(!scalars_equivalent(f64::INFINITY, f64::NEG_INFINITY));
        assert!(!scalars_equivalent(f64::INFINITY, 1.0));
        assert!(scalars_equivalent(1000.0, 1000.0 + 1e-7));
        assert!(!scalars_equivalent(0.874_586, 20_726_104.0));
    }

    #[test]
    fn test_scalar_outputs_equivalent_strict_i64_and_mask_bits() {
        let neg_one = u64::from_ne_bytes((-1_i64).to_ne_bytes());
        let neg_two = u64::from_ne_bytes((-2_i64).to_ne_bytes());
        assert!(scalar_outputs_equivalent(ElemType::I64, neg_one, neg_one));
        assert!(!scalar_outputs_equivalent(ElemType::I64, neg_one, neg_two));

        assert!(scalar_outputs_equivalent(ElemType::I64, 1_u64, 1_u64));
        assert!(!scalar_outputs_equivalent(ElemType::I64, 1_u64, 2_u64));
        assert!(!scalar_outputs_equivalent(
            ElemType::I64,
            1_u64,
            999_999_999_u64
        ));

        assert!(scalar_outputs_equivalent(ElemType::U8, 255_u64, 255_u64));
        assert!(!scalar_outputs_equivalent(ElemType::U8, 0_u64, 255_u64));

        assert!(scalar_outputs_equivalent(ElemType::BoolMask, 1_u64, 1_u64));
        assert!(!scalar_outputs_equivalent(ElemType::BoolMask, 0_u64, 1_u64));

        assert!(scalar_outputs_equivalent(
            ElemType::F64,
            100.0_f64.to_bits(),
            (100.0_f64 + 1e-11).to_bits()
        ));
        assert!(scalar_outputs_equivalent(
            ElemType::F64,
            f64::NAN.to_bits(),
            f64::NAN.to_bits()
        ));
    }

    #[test]
    fn test_p0_bug4_max_arena_data_bytes_1gib_limit() -> Result<(), String> {
        assert_eq!(MAX_ARENA_DATA_BYTES, 1024 * 1024 * 1024);
        // Verify that an arena data file exceeding the 16 MiB `.rl` bytecode limit
        // (`MAX_RL_FILE_BYTES = 16,777,216` bytes) — here 2,100,000 f64 elements (16,800,000 bytes) —
        // is cleanly accepted and read back by `read_bounded_f64_column`.
        let elem_count = 2_100_000usize;
        let byte_len = elem_count * 8;
        assert!(byte_len > MAX_RL_FILE_BYTES as usize);
        let tmp_path = std::env::temp_dir().join(format!(
            "rul_cli_test_large_arena_{}.bin",
            std::process::id()
        ));
        let mut raw = vec![0u8; byte_len];
        if let Some(first8) = raw.get_mut(0..8) {
            first8.copy_from_slice(&42.5_f64.to_le_bytes());
        }
        if let Some(last8) = raw.get_mut((byte_len - 8)..byte_len) {
            last8.copy_from_slice(&99.25_f64.to_le_bytes());
        }
        fs::write(&tmp_path, &raw).map_err(|e| e.to_string())?;
        let col_res = read_bounded_f64_column(&tmp_path, elem_count);
        let _ = fs::remove_file(&tmp_path);
        let col = col_res?;
        assert_eq!(col.len(), elem_count);
        assert_eq!(col.first().copied(), Some(42.5));
        assert_eq!(col.last().copied(), Some(99.25));
        Ok(())
    }

    #[test]
    fn test_read_bounded_input_bytes_caps_at_max_len_plus_one() -> Result<(), String> {
        let large = vec![0x42_u8; 1024];
        let bounded = read_bounded_input_bytes(large.as_slice(), 16)?;
        assert_eq!(bounded.len(), 17);

        let exact = vec![0x41_u8; 16];
        let ok_exact = read_bounded_input_bytes(exact.as_slice(), 16)?;
        assert_eq!(ok_exact.len(), 16);

        let out_only_rls = "\
cap NONE
proof 1 rule=region_decl target=0 lo=0 hi=16 witness=u8:output
node $c = ConstI64 i64x1 fg=0 pol=l1 proof=0 tok=pure const_i64=0
output $c
";
        let out_only_prog = assemble_rls(out_only_rls).map_err(|e| e.to_string())?;
        assert_eq!(out_only_prog.input_region_max_len(), None);
        assert!(out_only_prog.has_output_region());
        Ok(())
    }
}
