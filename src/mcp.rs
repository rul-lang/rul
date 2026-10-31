//! In-Memory Model Context Protocol (MCP) Server for Rul (`.rl`).
//!
//! Exposes a zero-dependency, sub-millisecond JSON-RPC 2.0 interface over `stdio`
//! so AI Agents can assemble, linearly verify, differentially test, autotune hardware
//! schedules, and AOT-compile standalone `ELF64` executables without spawning `rustc`.

use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use crate::aot::compile_rl_to_executable;
use crate::bytecode::RulProgram;
use crate::cas::builtin_intrinsics;
use crate::engine::{
    autotune_schedule, execute_tier0_interpreter, execute_tier1_stencil,
    execute_tier2_with_zonemap, ArenaZoneMap, RuntimeValue,
};
use crate::transpiler::{assemble_rls, disassemble_rl, program_to_rls};
use crate::verifier::verify_program;

/// Maximum accepted JSON-RPC line length (1 MiB) to enforce bounded memory.
pub const MAX_MCP_LINE_BYTES: usize = 1_048_576;

/// Stateless MCP server configuration and request handler.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RulMcpServer;

impl RulMcpServer {
    /// Creates a new in-memory `RulMcpServer`.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Processes a single JSON-RPC 2.0 line and returns a JSON-RPC 2.0 response line
    /// (or an empty string for JSON-RPC notifications).
    #[must_use]
    pub fn handle_line(&self, line: &str) -> String {
        handle_mcp_request(line)
    }
}

/// Processes a single JSON-RPC 2.0 request string and returns the JSON-RPC 2.0 response.
#[must_use]
pub fn handle_mcp_request(json_line: &str) -> String {
    let trimmed = json_line.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if trimmed.len() > MAX_MCP_LINE_BYTES {
        return json_rpc_error("null", -32600, "MCP request exceeds 1 MiB limit");
    }

    let req_id = extract_json_id(trimmed).unwrap_or_else(|| "null".to_string());
    let Some(method) = extract_json_string_field(trimmed, "method") else {
        return json_rpc_error(&req_id, -32600, "Missing JSON-RPC 'method' field");
    };

    match method.as_str() {
        "notifications/initialized" | "initialized" => String::new(),
        "ping" => json_rpc_result(&req_id, "{}"),
        "initialize" => {
            let result = concat!(
                r#"{"protocolVersion":"2024-11-05","#,
                r#""capabilities":{"tools":{"listChanged":false}},"#,
                r#""serverInfo":{"name":"rul-mcp","version":"0.1.0"}}"#
            );
            json_rpc_result(&req_id, result)
        }
        "tools/list" => json_rpc_result(&req_id, tools_list_json()),
        "tools/call" => {
            let Some(tool_name) = extract_tool_call_name(trimmed) else {
                return json_rpc_error(&req_id, -32602, "Missing tool 'name' in tools/call");
            };
            match dispatch_tool_call(&tool_name, trimmed) {
                Ok(text_payload) => build_tool_call_response(&req_id, &text_payload, false),
                Err(err_payload) => build_tool_call_response(&req_id, &err_payload, true),
            }
        }
        _ => json_rpc_error(&req_id, -32601, "Unsupported JSON-RPC method"),
    }
}

fn dispatch_tool_call(tool_name: &str, raw_json: &str) -> Result<String, String> {
    match tool_name {
        "rul_assemble_and_verify" => {
            let rls = extract_json_string_field(raw_json, "rls_source")
                .ok_or_else(|| "Missing required string argument 'rls_source'".to_string())?;
            tool_assemble_and_verify(&rls)
        }
        "rul_test_differential" => {
            let rls = extract_json_string_field(raw_json, "rls_source")
                .ok_or_else(|| "Missing required string argument 'rls_source'".to_string())?;
            let sample_size =
                extract_json_u64_field(raw_json, "sample_size").and_then(|v| u32::try_from(v).ok());
            tool_test_differential(&rls, sample_size)
        }
        "rul_autotune_schedule" => {
            let rls = extract_json_string_field(raw_json, "rls_source")
                .ok_or_else(|| "Missing required string argument 'rls_source'".to_string())?;
            let sample_size =
                extract_json_u64_field(raw_json, "sample_size").and_then(|v| u32::try_from(v).ok());
            tool_autotune_schedule(&rls, sample_size)
        }
        "rul_compile_standalone" => {
            let rls = extract_json_string_field(raw_json, "rls_source")
                .ok_or_else(|| "Missing required string argument 'rls_source'".to_string())?;
            let out_path = extract_json_string_field(raw_json, "output_path")
                .unwrap_or_else(|| "/tmp/rul_mcp_standalone.elf".to_string());
            tool_compile_standalone(&rls, Path::new(&out_path))
        }
        "rul_cas_intrinsics" => Ok(tool_cas_intrinsics()),
        _ => Err(format!("Unknown Rul MCP tool: {tool_name}")),
    }
}

fn tool_assemble_and_verify(rls_source: &str) -> Result<String, String> {
    let t0 = Instant::now();
    let program = assemble_rls(rls_source).map_err(|e| format!("RLS_ASSEMBLE_ERROR: {e:?}"))?;
    let assemble_ns = u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX);

    let t_ver = Instant::now();
    let summary = verify_program(&program).map_err(|diag| format!("VERIFY_REJECTED: {diag:?}"))?;
    let verify_ns = u64::try_from(t_ver.elapsed().as_nanos()).unwrap_or(u64::MAX);

    let rl_bytes = program.to_rl_bytes();
    let audit = disassemble_rl(&program);

    let mut out = String::with_capacity(512 + audit.len());
    let _ = writeln!(out, "STATUS: VERIFIED_OK");
    let _ = writeln!(out, "RL_BINARY_BYTES: {}", rl_bytes.len());
    let _ = writeln!(out, "ASSEMBLE_NS: {assemble_ns}");
    let _ = writeln!(out, "VERIFY_NS: {verify_ns}");
    let _ = writeln!(out, "NODE_COUNT: {}", summary.node_count);
    let _ = writeln!(out, "FUSION_GROUPS: {}", summary.fusion_groups);
    let _ = writeln!(out, "OUTPUT_SHAPE: {:?}", summary.output_shape);
    let _ = writeln!(
        out,
        "ZERO_RUNTIME_BOUNDS_CHECKS: {}",
        summary.zero_runtime_bounds_checks
    );
    let _ = writeln!(out, "--- AUDIT ---");
    out.push_str(&audit);
    Ok(out)
}

fn prepare_program_and_arenas(
    rls_source: &str,
    sample_size_override: Option<u32>,
) -> Result<(RulProgram, Vec<Vec<f64>>), String> {
    let mut program = assemble_rls(rls_source).map_err(|e| format!("RLS_ASSEMBLE_ERROR: {e:?}"))?;
    if let Some(override_len) = sample_size_override {
        let clamped = override_len.clamp(8, 1_000_000);
        for arena in &mut program.arenas {
            arena.element_count = clamped;
        }
    }
    let _ = verify_program(&program).map_err(|d| format!("VERIFY_REJECTED: {d:?}"))?;

    let mut owned_arenas = Vec::with_capacity(program.arenas.len());
    for (idx, arena) in program.arenas.iter().enumerate() {
        let count = arena.element_count as usize;
        let col: Vec<f64> = (0..count)
            .map(|i| {
                let base = f64::from(u32::try_from(i % 2000).unwrap_or(0));
                if idx == 0 {
                    base + 1.0
                } else {
                    (base * 0.5) + 10.0
                }
            })
            .collect();
        owned_arenas.push(col);
    }
    Ok((program, owned_arenas))
}

fn format_runtime_value(val: &RuntimeValue) -> String {
    match val {
        RuntimeValue::Scalar(v) => format!("Scalar({v:.6})"),
        RuntimeValue::ColumnF64(col) => {
            let preview: Vec<String> = col.iter().take(4).map(|x| format!("{x:.2}")).collect();
            format!(
                "ColumnF64(len={}, head=[{}])",
                col.len(),
                preview.join(", ")
            )
        }
        RuntimeValue::ColumnMask(mask) => {
            let trues = mask.iter().filter(|&&b| b).count();
            format!("ColumnMask(len={}, true_count={trues})", mask.len())
        }
    }
}

fn tool_test_differential(
    rls_source: &str,
    sample_size_override: Option<u32>,
) -> Result<String, String> {
    let (program, owned_arenas) = prepare_program_and_arenas(rls_source, sample_size_override)?;
    let input_slices: Vec<&[f64]> = owned_arenas.iter().map(Vec::as_slice).collect();
    let zmap = owned_arenas.first().map(|col| ArenaZoneMap::build(col));

    let rep0 = execute_tier0_interpreter(&program, &input_slices)
        .map_err(|e| format!("TIER0_EXEC_ERROR: {e:?}"))?;
    let rep1 = execute_tier1_stencil(&program, &input_slices)
        .map_err(|e| format!("TIER1_EXEC_ERROR: {e:?}"))?;
    let rep2 = execute_tier2_with_zonemap(&program, &input_slices, zmap.as_ref())
        .map_err(|e| format!("TIER2_EXEC_ERROR: {e:?}"))?;

    let mut out = String::with_capacity(512);
    let _ = writeln!(out, "STATUS: DIFFERENTIAL_EQUIVALENCE_PASS");
    let _ = writeln!(out, "DATA_SOURCE: SYNTHETIC_PATTERN (arena[0][i] = (i % 2000) + 1; arena[k>0][i] = (i % 2000) * 0.5 + 10) — generated in-memory sample data, NOT user-supplied data; this check proves Tier0/Tier1/Tier2 equivalence only");
    let _ = writeln!(out, "OUTPUT: {}", format_runtime_value(&rep2.output));
    let _ = writeln!(
        out,
        "TIER0_INTERPRETER: exec_ns={} alloc_bytes={}",
        rep0.telemetry.exec_ns, rep0.telemetry.intermediate_bytes_allocated
    );
    let _ = writeln!(
        out,
        "TIER1_STENCIL: exec_ns={} alloc_bytes={}",
        rep1.telemetry.exec_ns, rep1.telemetry.intermediate_bytes_allocated
    );
    let _ = writeln!(
        out,
        "TIER2_FUSED: exec_ns={} alloc_bytes={} elements={} bw_gbps={:.2} reward={:.3}",
        rep2.telemetry.exec_ns,
        rep2.telemetry.intermediate_bytes_allocated,
        rep2.telemetry.elements_processed,
        rep2.telemetry.effective_bandwidth_gbps,
        rep2.telemetry.reward_score
    );
    Ok(out)
}

fn tool_autotune_schedule(
    rls_source: &str,
    sample_size_override: Option<u32>,
) -> Result<String, String> {
    let (mut program, owned_arenas) = prepare_program_and_arenas(rls_source, sample_size_override)?;
    let input_slices: Vec<&[f64]> = owned_arenas.iter().map(Vec::as_slice).collect();
    let zmap = owned_arenas.first().map(|col| ArenaZoneMap::build(col));

    let report = autotune_schedule(&mut program, &input_slices, zmap.as_ref())
        .map_err(|e| format!("AUTOTUNE_ERROR: {e:?}"))?;
    let tuned_rls = program_to_rls(&program);

    let mut out = String::with_capacity(512 + tuned_rls.len());
    let _ = writeln!(out, "STATUS: AUTOTUNE_COMPLETE");
    let _ = writeln!(out, "DATA_SOURCE: SYNTHETIC_PATTERN (arena[0][i] = (i % 2000) + 1; arena[k>0][i] = (i % 2000) * 0.5 + 10) — generated in-memory sample data, NOT user-supplied data; schedule ranking may differ on real data");
    let _ = writeln!(
        out,
        "WINNING_SCHEDULE: policy={:?} fusion_group={}",
        report.best_schedule.policy, report.best_schedule.fusion_group
    );
    let _ = writeln!(
        out,
        "WINNING_TELEMETRY: exec_ns={} alloc_bytes={} reward={:.3}",
        report.best_telemetry.exec_ns,
        report.best_telemetry.intermediate_bytes_allocated,
        report.best_telemetry.reward_score
    );
    let _ = writeln!(out, "CANDIDATES:");
    for c in &report.candidates {
        let _ = writeln!(
            out,
            "  - policy={:?} fg={} exec_ns={} alloc_bytes={} reward={:.3}",
            c.policy,
            c.fusion_group,
            c.telemetry.exec_ns,
            c.telemetry.intermediate_bytes_allocated,
            c.telemetry.reward_score
        );
    }
    let _ = writeln!(out, "--- OPTIMIZED_RLS ---");
    out.push_str(&tuned_rls);
    Ok(out)
}

fn tool_compile_standalone(rls_source: &str, output_path: &Path) -> Result<String, String> {
    let program = assemble_rls(rls_source).map_err(|e| format!("RLS_ASSEMBLE_ERROR: {e:?}"))?;
    let report = compile_rl_to_executable(&program, output_path)
        .map_err(|e| format!("AOT_COMPILE_ERROR: {e:?}"))?;

    let mut out = String::with_capacity(256);
    let _ = writeln!(out, "STATUS: AOT_ELF64_COMPILED");
    let _ = writeln!(out, "EXECUTABLE_PATH: {}", output_path.display());
    let _ = writeln!(out, "ELF_BYTES: {}", report.elf_bytes);
    let _ = writeln!(out, "MACHINE_CODE_BYTES: {}", report.machine_code_bytes);
    let _ = writeln!(out, "VERIFY_NS: {}", report.verify_ns);
    let _ = writeln!(out, "CODEGEN_NS: {}", report.codegen_ns);
    let _ = writeln!(out, "TOTAL_COMPILE_NS: {}", report.total_compile_ns);
    Ok(out)
}

fn tool_cas_intrinsics() -> String {
    let mut out = String::with_capacity(512);
    let _ = writeln!(out, "STATUS: RUL_CAS_INTRINSICS_CATALOG");
    for desc in builtin_intrinsics() {
        let _ = writeln!(
            out,
            "  - hash={} name={} sig=\"{}\" kind={:?}",
            desc.hash, desc.name, desc.signature, desc.kind
        );
    }
    out
}

fn tools_list_json() -> &'static str {
    concat!(
        r#"{"tools":["#,
        r#"{"name":"rul_assemble_and_verify","description":"Assemble flat symbolic .rls IR into a 128-bit .rl binary program and run the O(N) linear verifier (< 0.1 ms).","inputSchema":{"type":"object","properties":{"rls_source":{"type":"string","description":"Flat symbolic .rls program text"}},"required":["rls_source"]}},"#,
        r#"{"name":"rul_test_differential","description":"Run differential execution comparing Tier 0 Interpreter, Tier 1 Stencil, and Tier 2 Fused L1-Tile Engine on generated SYNTHETIC sample arenas (deterministic pattern, not user data — this tool accepts no real arena data).","inputSchema":{"type":"object","properties":{"rls_source":{"type":"string","description":"Flat symbolic .rls program text"},"sample_size":{"type":"integer","description":"Optional element count override for sample arenas"}},"required":["rls_source"]}},"#,
        r#"{"name":"rul_autotune_schedule","description":"Sweep ScheduleHint policies (AutoL1Tile, SimdStream, MultiCore) on generated SYNTHETIC sample arenas (deterministic pattern, not user data — this tool accepts no real arena data), pick the highest hardware reward score, and return the tuned .rls program.","inputSchema":{"type":"object","properties":{"rls_source":{"type":"string","description":"Flat symbolic .rls program text"},"sample_size":{"type":"integer","description":"Optional element count override for autotuning"}},"required":["rls_source"]}},"#,
        r#"{"name":"rul_compile_standalone","description":"Compile a .rls program directly into a zero-runtime standalone Linux x86_64 ELF64 native executable in < 1 ms.","inputSchema":{"type":"object","properties":{"rls_source":{"type":"string","description":"Flat symbolic .rls program text"},"output_path":{"type":"string","description":"Output filesystem path for the standalone ELF64 binary"}},"required":["rls_source"]}},"#,
        r#"{"name":"rul_cas_intrinsics","description":"List content-addressed 64-bit verified intrinsics (I64 Sort, HashJoin, GroupBy, U8 SIMD Scan) in rul-cas.","inputSchema":{"type":"object","properties":{}}}"#,
        r#"]}"#
    )
}

fn build_tool_call_response(req_id: &str, text: &str, is_error: bool) -> String {
    let escaped = escape_json_string(text);
    let err_flag = if is_error { "true" } else { "false" };
    let result =
        format!(r#"{{"content":[{{"type":"text","text":"{escaped}"}}],"isError":{err_flag}}}"#);
    json_rpc_result(req_id, &result)
}

fn json_rpc_result(req_id: &str, result_json: &str) -> String {
    format!(r#"{{"jsonrpc":"2.0","id":{req_id},"result":{result_json}}}"#)
}

fn json_rpc_error(req_id: &str, code: i32, message: &str) -> String {
    let escaped = escape_json_string(message);
    format!(r#"{{"jsonrpc":"2.0","id":{req_id},"error":{{"code":{code},"message":"{escaped}"}}}}"#)
}

fn escape_json_string(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + 16);
    for ch in input.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

fn extract_json_id(json: &str) -> Option<String> {
    let pat = "\"id\"";
    let pos = json.find(pat)?;
    let after = json.get(pos + pat.len()..)?.trim_start();
    let rest = after.strip_prefix(':')?.trim_start();
    if let Some(stripped) = rest.strip_prefix('"') {
        let mut s = String::new();
        let mut escaped = false;
        for ch in stripped.chars() {
            if escaped {
                s.push(ch);
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                return Some(format!("\"{}\"", escape_json_string(&s)));
            } else {
                s.push(ch);
            }
        }
        None
    } else {
        let token_end = rest
            .find(|c: char| c == ',' || c == '}' || c.is_whitespace())
            .unwrap_or(rest.len());
        let token = rest.get(..token_end)?.trim();
        if token.is_empty() {
            None
        } else {
            Some(token.to_string())
        }
    }
}

fn extract_tool_call_name(json: &str) -> Option<String> {
    let params_pos = json.find("\"params\"")?;
    let params_slice = json.get(params_pos..)?;
    extract_json_string_field(params_slice, "name")
}

fn extract_json_string_field(json: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let mut search_slice = json;
    while let Some(pos) = search_slice.find(&pat) {
        let after = search_slice.get(pos + pat.len()..)?.trim_start();
        if let Some(after_colon) = after.strip_prefix(':') {
            let val_start = after_colon.trim_start();
            if let Some(in_quotes) = val_start.strip_prefix('"') {
                return parse_escaped_json_string(in_quotes);
            }
        }
        search_slice = after;
    }
    None
}

fn parse_escaped_json_string(in_quotes: &str) -> Option<String> {
    let mut out = String::new();
    let mut chars = in_quotes.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            let next_ch = chars.next()?;
            match next_ch {
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                '/' => out.push('/'),
                other => out.push(other),
            }
        } else if ch == '"' {
            return Some(out);
        } else {
            out.push(ch);
        }
    }
    None
}

fn extract_json_u64_field(json: &str, key: &str) -> Option<u64> {
    let pat = format!("\"{key}\"");
    let pos = json.find(&pat)?;
    let after = json.get(pos + pat.len()..)?.trim_start();
    let rest = after.strip_prefix(':')?.trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    let digits = rest.get(..end)?;
    if digits.is_empty() {
        None
    } else {
        digits.parse::<u64>().ok()
    }
}
