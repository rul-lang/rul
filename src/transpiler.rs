//! Bridge transpiler, symbolic emitter, and human-auditable disassembler for `.rl` binaries.
//!
//! While the Rul Core Engine operates exclusively on flat 128-bit `.rl` binaries,
//! current text-trained LLMs benefit from semantic attention anchors (`"$prices"`, `"$mask"`)
//! during emission. [`SymbolicGraphBuilder`] lets an AI Agent emit symbolic node names that
//! are stripped into pure `u32` flat indices when packed into `.rl`.

use std::collections::HashMap;
use std::fmt::{self, Write as _};

use crate::bytecode::{
    ArenaDescriptor, ElemType, ExecPolicy, NodeFrame, Opcode, ProofEntry, ProofRuleKind,
    RulProgram, ScheduleHint, SimdWidth, TokenMode, TokenOp, TypeLane,
};

/// Error returned when resolving symbolic labels in [`SymbolicGraphBuilder`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranspileError {
    /// Referenced symbolic node label was never defined.
    UnknownSymbol(String),
    /// Duplicate symbolic label definition.
    DuplicateSymbol(String),
}

impl fmt::Display for TranspileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownSymbol(sym) => write!(f, "unknown symbolic label: {sym}"),
            Self::DuplicateSymbol(sym) => write!(f, "duplicate symbolic label: {sym}"),
        }
    }
}

impl std::error::Error for TranspileError {}

/// Symbolic builder that allows an AI Agent to use short semantic anchors (`"$tx"`, `"$mask"`)
/// while emitting a pure binary `.rl` program with zero ASCII strings in the final image.
#[derive(Debug, Default)]
pub struct SymbolicGraphBuilder {
    capability_mask: u64,
    arenas: Vec<ArenaDescriptor>,
    proofs: Vec<ProofEntry>,
    nodes: Vec<NodeFrame>,
    symbols: HashMap<String, u32>,
}

impl SymbolicGraphBuilder {
    /// Creates a new symbolic builder with the given hardware `capability_mask`.
    #[must_use]
    pub fn new(capability_mask: u64) -> Self {
        Self {
            capability_mask,
            arenas: Vec::new(),
            proofs: Vec::new(),
            nodes: Vec::new(),
            symbols: HashMap::new(),
        }
    }

    /// Declares a columnar `Arena` and returns its `arena_id`.
    pub fn add_arena(&mut self, element_count: u32, type_lane: TypeLane, mutable: bool) -> u32 {
        let arena_id = u32::try_from(self.arenas.len()).unwrap_or(0);
        self.arenas.push(ArenaDescriptor {
            arena_id,
            element_count,
            type_lane,
            mutable,
            alignment: 64,
            initial_version: 1,
        });
        arena_id
    }

    /// Adds a sidecar [`ProofEntry`] and returns its 1-based `proof_idx`.
    pub fn add_proof(&mut self, entry: ProofEntry) -> u8 {
        self.proofs.push(entry);
        u8::try_from(self.proofs.len()).unwrap_or(0)
    }

    fn register_symbol(&mut self, label: &str, frame: NodeFrame) -> Result<u32, TranspileError> {
        if self.symbols.contains_key(label) {
            return Err(TranspileError::DuplicateSymbol(label.to_owned()));
        }
        let idx = u32::try_from(self.nodes.len()).unwrap_or(0);
        self.nodes.push(frame);
        self.symbols.insert(label.to_owned(), idx);
        Ok(idx)
    }

    /// Resolves a symbolic label to its flat `u32` node index.
    ///
    /// # Errors
    /// Returns [`TranspileError::UnknownSymbol`] if `label` has not been emitted.
    pub fn resolve(&self, label: &str) -> Result<u32, TranspileError> {
        self.symbols
            .get(label)
            .copied()
            .ok_or_else(|| TranspileError::UnknownSymbol(label.to_owned()))
    }

    /// Emits an [`Opcode::ArenaInput`] node with a read lease and binds it to `label`.
    ///
    /// # Errors
    /// Returns [`TranspileError`] if `label` is duplicated.
    pub fn emit_arena_input(
        &mut self,
        label: &str,
        arena_id: u32,
        version: u16,
        schedule: ScheduleHint,
    ) -> Result<u32, TranspileError> {
        let arena_u16 = u16::try_from(arena_id).unwrap_or(0);
        self.register_symbol(
            label,
            NodeFrame {
                opcode: Opcode::ArenaInput,
                type_lane: TypeLane::F64X8,
                schedule,
                proof_idx: 0,
                token_op: TokenOp::read_lease(arena_u16, version),
                in_a: arena_id,
                in_b: 0,
            },
        )
    }

    /// Emits an [`Opcode::ConstF64`] scalar node and binds it to `label`.
    ///
    /// # Errors
    /// Returns [`TranspileError`] if `label` is duplicated.
    pub fn emit_const_f64(
        &mut self,
        label: &str,
        value: f64,
        schedule: ScheduleHint,
    ) -> Result<u32, TranspileError> {
        self.register_symbol(label, NodeFrame::const_f64(value, schedule))
    }

    /// Emits a binary operation node referencing two prior symbolic labels (`in_a_sym`, `in_b_sym`).
    ///
    /// # Errors
    /// Returns [`TranspileError`] if either input symbol is unknown or `label` is duplicated.
    pub fn emit_binary_op(
        &mut self,
        label: &str,
        opcode: Opcode,
        type_lane: TypeLane,
        in_a_sym: &str,
        in_b_sym: &str,
        schedule: ScheduleHint,
    ) -> Result<u32, TranspileError> {
        let in_a = self.resolve(in_a_sym)?;
        let in_b = self.resolve(in_b_sym)?;
        self.register_symbol(
            label,
            NodeFrame {
                opcode,
                type_lane,
                schedule,
                proof_idx: 0,
                token_op: TokenOp::PURE,
                in_a,
                in_b,
            },
        )
    }

    /// Emits a unary reduction or assertion node referencing `in_a_sym`.
    ///
    /// # Errors
    /// Returns [`TranspileError`] if `in_a_sym` is unknown or `label` is duplicated.
    pub fn emit_unary_op(
        &mut self,
        label: &str,
        opcode: Opcode,
        type_lane: TypeLane,
        in_a_sym: &str,
        schedule: ScheduleHint,
    ) -> Result<u32, TranspileError> {
        let in_a = self.resolve(in_a_sym)?;
        self.register_symbol(
            label,
            NodeFrame {
                opcode,
                type_lane,
                schedule,
                proof_idx: 0,
                token_op: TokenOp::PURE,
                in_a,
                in_b: 0,
            },
        )
    }

    /// Emits a bounds-proven [`Opcode::ArenaSlice`] node referencing `in_a_sym` and `proof_idx`.
    ///
    /// # Errors
    /// Returns [`TranspileError`] if `in_a_sym` is unknown or `label` is duplicated.
    pub fn emit_slice(
        &mut self,
        label: &str,
        in_a_sym: &str,
        proof_idx: u8,
        schedule: ScheduleHint,
    ) -> Result<u32, TranspileError> {
        let in_a = self.resolve(in_a_sym)?;
        self.register_symbol(
            label,
            NodeFrame {
                opcode: Opcode::ArenaSlice,
                type_lane: TypeLane::F64X8,
                schedule,
                proof_idx,
                token_op: TokenOp::PURE,
                in_a,
                in_b: 0,
            },
        )
    }

    /// Emits an indirect [`Opcode::Gather`] node (`target_sym[indices_sym]`) proven by `proof_idx`.
    ///
    /// # Errors
    /// Returns [`TranspileError`] if either symbol is unknown or `label` is duplicated.
    pub fn emit_gather(
        &mut self,
        label: &str,
        target_sym: &str,
        indices_sym: &str,
        proof_idx: u8,
        schedule: ScheduleHint,
    ) -> Result<u32, TranspileError> {
        let in_a = self.resolve(target_sym)?;
        let in_b = self.resolve(indices_sym)?;
        self.register_symbol(
            label,
            NodeFrame {
                opcode: Opcode::Gather,
                type_lane: TypeLane::F64X8,
                schedule,
                proof_idx,
                token_op: TokenOp::PURE,
                in_a,
                in_b,
            },
        )
    }

    /// Emits an in-place [`Opcode::ArenaMutate`] node that consumes `AffineToken(arena_id, version)`.
    ///
    /// # Errors
    /// Returns [`TranspileError`] if `val_sym` is unknown or `label` is duplicated.
    pub fn emit_mutate(
        &mut self,
        label: &str,
        arena_id: u32,
        version: u16,
        val_sym: &str,
        schedule: ScheduleHint,
    ) -> Result<u32, TranspileError> {
        let in_b = self.resolve(val_sym)?;
        let arena_u16 = u16::try_from(arena_id).unwrap_or(0);
        self.register_symbol(
            label,
            NodeFrame {
                opcode: Opcode::ArenaMutate,
                type_lane: TypeLane::F64X8,
                schedule,
                proof_idx: 0,
                token_op: TokenOp::affine_consume(arena_u16, version),
                in_a: arena_id,
                in_b,
            },
        )
    }

    /// Finalizes the builder into a pure binary [`RulProgram`] with `output_sym` as primary output.
    ///
    /// # Errors
    /// Returns [`TranspileError`] if `output_sym` is unknown.
    pub fn finish(self, output_sym: &str) -> Result<RulProgram, TranspileError> {
        let output_node = self.resolve(output_sym)?;
        Ok(RulProgram {
            capability_mask: self.capability_mask,
            arenas: self.arenas,
            proofs: self.proofs,
            nodes: self.nodes,
            output_node,
        })
    }
}

/// Disassembles a binary [`RulProgram`] into a human-auditable table (`rul inspect`).
#[must_use]
pub fn disassemble_rl(program: &RulProgram) -> String {
    let mut out = String::new();
    let total_bytes = program.to_rl_bytes().len();
    let hash = program.compute_graph_hash();

    let _ = writeln!(
        out,
        "=== RUL (.rl) BINARY GRAPH AUDIT ({total_bytes} bytes total) ==="
    );
    let _ = writeln!(
        out,
        "HEADER : Magic=\"RUL\\x00\\x01\\x00\\x00\\x00\" | Hash={hash:#018x} | CapMask={:#06x} | Output=Node({})",
        program.capability_mask, program.output_node
    );

    for arena in &program.arenas {
        let _ = writeln!(
            out,
            "ARENA[{}]: Len={} | Type={:?}/{:?} | Mutable={} | Align={}B | InitToken=v{}",
            arena.arena_id,
            arena.element_count,
            arena.type_lane.elem,
            arena.type_lane.width,
            arena.mutable,
            arena.alignment,
            arena.initial_version
        );
    }

    for (idx, proof) in program.proofs.iter().enumerate() {
        let _ = writeln!(
            out,
            "PROOF[{}]: Rule={:?} | Target=Arena({}) | Bounds=[{}..{}]",
            idx + 1,
            proof.rule_kind,
            proof.target_arena,
            proof.bound_lo,
            proof.bound_hi
        );
    }

    let _ = writeln!(
        out,
        "--------------------------------------------------------------------------------"
    );
    let _ = writeln!(
        out,
        "NODE | OPCODE         | TYPE/LANE   | SCHEDULE (GRP/POL) | PROOF | TOKEN          | INPUTS"
    );
    let _ = writeln!(
        out,
        "--------------------------------------------------------------------------------"
    );

    for (idx, frame) in program.nodes.iter().enumerate() {
        let inputs = if frame.opcode == Opcode::ConstF64 {
            format!("const={:.4}", frame.decode_const_f64())
        } else {
            format!("in_a={}, in_b={}", frame.in_a, frame.in_b)
        };
        let _ = writeln!(
            out,
            "#{idx:<3} | {:<14} | {:<4?}/{:<6?} | Grp={} / {:<10?} | #{:<4} | {:<14} | {inputs}",
            format!("{:?}", frame.opcode),
            frame.type_lane.elem,
            frame.type_lane.width,
            frame.schedule.fusion_group,
            frame.schedule.policy,
            frame.proof_idx,
            format!(
                "{:?}(A{},v{})",
                frame.token_op.mode, frame.token_op.arena_id, frame.token_op.version
            ),
        );
    }

    out
}

/// Error returned when parsing or assembling a flat `.rls` symbolic text program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RlsError {
    /// Syntax or token error on a specific 1-based line number.
    LineParse {
        /// 1-based line index.
        line: usize,
        /// Diagnostic description.
        message: String,
    },
    /// Symbol resolution error.
    Symbol(TranspileError),
    /// Program contains zero instruction nodes.
    EmptyProgram,
}

impl fmt::Display for RlsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LineParse { line, message } => {
                write!(f, ".rls parse error at line {line}: {message}")
            }
            Self::Symbol(e) => write!(f, ".rls symbol error: {e}"),
            Self::EmptyProgram => write!(f, ".rls program contains no instruction nodes"),
        }
    }
}

impl std::error::Error for RlsError {}

/// Decompiles a binary [`RulProgram`] into the canonical flat `.rls` text representation.
///
/// The emitted `.rls` text is 100% lossless and round-trips through [`assemble_rls`].
#[must_use]
pub fn program_to_rls(program: &RulProgram) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# Rul Flat Symbolic IR (.rls)");
    let _ = writeln!(out, "cap {:#x}", program.capability_mask);

    for arena in &program.arenas {
        let _ = writeln!(
            out,
            "arena {} len={} lane={} mut={} align={} ver={}",
            arena.arena_id,
            arena.element_count,
            format_type_lane(arena.type_lane),
            arena.mutable,
            arena.alignment,
            arena.initial_version
        );
    }

    for (idx, proof) in program.proofs.iter().enumerate() {
        let rule_str = match proof.rule_kind {
            ProofRuleKind::SliceBounds => "slice_bounds",
            ProofRuleKind::GatherIndexInBounds => "gather_bounds",
        };
        let _ = writeln!(
            out,
            "proof {} rule={} target={} lo={} hi={} witness={}",
            idx + 1,
            rule_str,
            proof.target_arena,
            proof.bound_lo,
            proof.bound_hi,
            proof.witness_node
        );
    }

    for (idx, frame) in program.nodes.iter().enumerate() {
        let op_str = format_opcode(frame.opcode);
        let lane_str = format_type_lane(frame.type_lane);
        let pol_str = format_policy(frame.schedule.policy);
        let tok_str = format_token_op(frame.token_op);

        if frame.opcode == Opcode::ConstF64 {
            let val = frame.decode_const_f64();
            let _ = writeln!(
                out,
                "node $n{idx} = {op_str} {lane_str} fg={} pol={pol_str} proof={} tok={tok_str} const={val:?}",
                frame.schedule.fusion_group, frame.proof_idx
            );
        } else if matches!(frame.opcode, Opcode::ArenaInput | Opcode::ArenaMutate) {
            let b_ref = format_node_ref(frame.opcode, false, frame.in_b);
            let _ = writeln!(
                out,
                "node $n{idx} = {op_str} {lane_str} fg={} pol={pol_str} proof={} tok={tok_str} a={} b={b_ref}",
                frame.schedule.fusion_group, frame.proof_idx, frame.in_a
            );
        } else {
            let a_ref = format_node_ref(frame.opcode, true, frame.in_a);
            let b_ref = format_node_ref(frame.opcode, false, frame.in_b);
            let _ = writeln!(
                out,
                "node $n{idx} = {op_str} {lane_str} fg={} pol={pol_str} proof={} tok={tok_str} a={a_ref} b={b_ref}",
                frame.schedule.fusion_group, frame.proof_idx
            );
        }
    }

    let _ = writeln!(out, "output $n{}", program.output_node);
    out
}

/// Assembles a flat `.rls` symbolic text source directly into a binary [`RulProgram`]
/// in a single $O(N)$ pass without invoking `rustc`.
///
/// # Errors
/// Returns [`RlsError`] if any directive, opcode, type lane, or symbol reference is invalid.
pub fn assemble_rls(source: &str) -> Result<RulProgram, RlsError> {
    let mut capability_mask: u64 = 0;
    let mut arenas: Vec<ArenaDescriptor> = Vec::new();
    let mut proofs: Vec<ProofEntry> = Vec::new();
    let mut nodes: Vec<NodeFrame> = Vec::new();
    let mut symbols: HashMap<String, u32> = HashMap::new();
    let mut explicit_output: Option<String> = None;

    for (line_idx, raw_line) in source.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }

        let mut parts = line.split_whitespace();
        let Some(directive) = parts.next() else {
            continue;
        };

        match directive {
            "cap" => {
                let raw_val = parts.next().ok_or_else(|| RlsError::LineParse {
                    line: line_no,
                    message: "missing capability mask value".to_owned(),
                })?;
                capability_mask = parse_u64_auto(raw_val).ok_or_else(|| RlsError::LineParse {
                    line: line_no,
                    message: format!("invalid capability mask: {raw_val}"),
                })?;
            }
            "arena" => {
                let arena = parse_arena_line(line_no, parts, arenas.len())?;
                arenas.push(arena);
            }
            "proof" => {
                let proof = parse_proof_line(line_no, parts)?;
                proofs.push(proof);
            }
            "node" => {
                parse_node_line(line_no, parts, &mut nodes, &mut symbols)?;
            }
            "output" => {
                let sym = parts.next().ok_or_else(|| RlsError::LineParse {
                    line: line_no,
                    message: "missing output node symbol or index".to_owned(),
                })?;
                explicit_output = Some(sym.to_owned());
            }
            other => {
                return Err(RlsError::LineParse {
                    line: line_no,
                    message: format!("unknown directive '{other}'"),
                });
            }
        }
    }

    if nodes.is_empty() {
        return Err(RlsError::EmptyProgram);
    }

    let output_node = if let Some(sym) = explicit_output {
        resolve_sym_or_u32(0, &sym, &symbols)?
    } else {
        u32::try_from(nodes.len().saturating_sub(1)).unwrap_or(0)
    };

    Ok(RulProgram {
        capability_mask,
        arenas,
        proofs,
        nodes,
        output_node,
    })
}

fn parse_arena_line<'a>(
    line_no: usize,
    mut parts: impl Iterator<Item = &'a str>,
    default_id: usize,
) -> Result<ArenaDescriptor, RlsError> {
    let id_token = parts.next().ok_or_else(|| RlsError::LineParse {
        line: line_no,
        message: "missing arena id".to_owned(),
    })?;
    let arena_id = id_token
        .parse::<u32>()
        .unwrap_or_else(|_| u32::try_from(default_id).unwrap_or(0));

    let mut element_count: Option<u32> = None;
    let mut type_lane = TypeLane::F64X8;
    let mut mutable = false;
    let mut alignment: u16 = 64;
    let mut initial_version: u16 = 1;

    for kv in parts {
        let Some((k, v)) = kv.split_once('=') else {
            return Err(RlsError::LineParse {
                line: line_no,
                message: format!("expected key=value in arena directive, got '{kv}'"),
            });
        };
        match k {
            "len" => {
                element_count = Some(v.parse::<u32>().map_err(|_e| RlsError::LineParse {
                    line: line_no,
                    message: format!("invalid arena len '{v}'"),
                })?);
            }
            "lane" => {
                type_lane = parse_type_lane(line_no, v)?;
            }
            "mut" => {
                mutable = match v {
                    "true" | "1" => true,
                    "false" | "0" => false,
                    _ => {
                        return Err(RlsError::LineParse {
                            line: line_no,
                            message: format!("invalid arena mutability '{v}'"),
                        })
                    }
                };
            }
            "align" => {
                alignment = v.parse::<u16>().map_err(|_e| RlsError::LineParse {
                    line: line_no,
                    message: format!("invalid arena alignment '{v}'"),
                })?;
            }
            "ver" => {
                initial_version = v.parse::<u16>().map_err(|_e| RlsError::LineParse {
                    line: line_no,
                    message: format!("invalid arena initial version '{v}'"),
                })?;
            }
            _ => {
                return Err(RlsError::LineParse {
                    line: line_no,
                    message: format!("unknown arena attribute '{k}'"),
                });
            }
        }
    }

    let len = element_count.ok_or_else(|| RlsError::LineParse {
        line: line_no,
        message: "arena directive requires len=<u32>".to_owned(),
    })?;

    Ok(ArenaDescriptor {
        arena_id,
        element_count: len,
        type_lane,
        mutable,
        alignment,
        initial_version,
    })
}

fn parse_proof_line<'a>(
    line_no: usize,
    mut parts: impl Iterator<Item = &'a str>,
) -> Result<ProofEntry, RlsError> {
    let _idx_tok = parts.next().ok_or_else(|| RlsError::LineParse {
        line: line_no,
        message: "missing proof index".to_owned(),
    })?;

    let mut rule_kind = ProofRuleKind::SliceBounds;
    let mut target_arena: u16 = 0;
    let mut bound_lo: u32 = 0;
    let mut bound_hi: Option<u32> = None;
    let mut witness_node: u32 = 0;

    for kv in parts {
        let Some((k, v)) = kv.split_once('=') else {
            return Err(RlsError::LineParse {
                line: line_no,
                message: format!("expected key=value in proof directive, got '{kv}'"),
            });
        };
        match k {
            "rule" => {
                rule_kind = match v {
                    "slice_bounds" | "SliceBounds" | "1" => ProofRuleKind::SliceBounds,
                    "gather_bounds" | "GatherIndexInBounds" | "2" => {
                        ProofRuleKind::GatherIndexInBounds
                    }
                    _ => {
                        return Err(RlsError::LineParse {
                            line: line_no,
                            message: format!("unknown proof rule '{v}'"),
                        })
                    }
                };
            }
            "target" => {
                target_arena = v.parse::<u16>().map_err(|_e| RlsError::LineParse {
                    line: line_no,
                    message: format!("invalid target arena '{v}'"),
                })?;
            }
            "lo" => {
                bound_lo = v.parse::<u32>().map_err(|_e| RlsError::LineParse {
                    line: line_no,
                    message: format!("invalid bound_lo '{v}'"),
                })?;
            }
            "hi" => {
                bound_hi = Some(v.parse::<u32>().map_err(|_e| RlsError::LineParse {
                    line: line_no,
                    message: format!("invalid bound_hi '{v}'"),
                })?);
            }
            "witness" => {
                witness_node = v.parse::<u32>().map_err(|_e| RlsError::LineParse {
                    line: line_no,
                    message: format!("invalid witness_node '{v}'"),
                })?;
            }
            _ => {
                return Err(RlsError::LineParse {
                    line: line_no,
                    message: format!("unknown proof attribute '{k}'"),
                });
            }
        }
    }

    let hi = bound_hi.ok_or_else(|| RlsError::LineParse {
        line: line_no,
        message: "proof directive requires hi=<u32>".to_owned(),
    })?;

    Ok(ProofEntry {
        rule_kind,
        target_arena,
        bound_lo,
        bound_hi: hi,
        witness_node,
    })
}

fn parse_node_line<'a>(
    line_no: usize,
    mut parts: impl Iterator<Item = &'a str>,
    nodes: &mut Vec<NodeFrame>,
    symbols: &mut HashMap<String, u32>,
) -> Result<(), RlsError> {
    let label = parts.next().ok_or_else(|| RlsError::LineParse {
        line: line_no,
        message: "missing node label".to_owned(),
    })?;
    let eq = parts.next().ok_or_else(|| RlsError::LineParse {
        line: line_no,
        message: "expected '=' after node label".to_owned(),
    })?;
    if eq != "=" {
        return Err(RlsError::LineParse {
            line: line_no,
            message: format!("expected '=' after '{label}', got '{eq}'"),
        });
    }

    let op_tok = parts.next().ok_or_else(|| RlsError::LineParse {
        line: line_no,
        message: "missing opcode".to_owned(),
    })?;
    let opcode = parse_opcode(line_no, op_tok)?;

    let lane_tok = parts.next().ok_or_else(|| RlsError::LineParse {
        line: line_no,
        message: "missing type lane".to_owned(),
    })?;
    let type_lane = parse_type_lane(line_no, lane_tok)?;

    let mut fusion_group: u8 = 1;
    let mut policy = ExecPolicy::SimdStream;
    let mut proof_idx: u8 = 0;
    let mut explicit_token: Option<TokenOp> = None;
    let mut const_val: Option<f64> = None;
    let mut raw_a: Option<&str> = None;
    let mut raw_b: Option<&str> = None;

    for kv in parts {
        let Some((k, v)) = kv.split_once('=') else {
            return Err(RlsError::LineParse {
                line: line_no,
                message: format!("expected key=value in node directive, got '{kv}'"),
            });
        };
        match k {
            "fg" => {
                fusion_group = v.parse::<u8>().map_err(|_e| RlsError::LineParse {
                    line: line_no,
                    message: format!("invalid fusion_group '{v}'"),
                })?;
            }
            "pol" => {
                policy = parse_policy(line_no, v)?;
            }
            "proof" => {
                proof_idx = v.parse::<u8>().map_err(|_e| RlsError::LineParse {
                    line: line_no,
                    message: format!("invalid proof_idx '{v}'"),
                })?;
            }
            "tok" => {
                explicit_token = Some(parse_token_op(line_no, v)?);
            }
            "const" => {
                const_val = Some(v.parse::<f64>().map_err(|_e| RlsError::LineParse {
                    line: line_no,
                    message: format!("invalid f64 constant '{v}'"),
                })?);
            }
            "a" => {
                raw_a = Some(v);
            }
            "b" => {
                raw_b = Some(v);
            }
            _ => {
                return Err(RlsError::LineParse {
                    line: line_no,
                    message: format!("unknown node attribute '{k}'"),
                });
            }
        }
    }

    let schedule = ScheduleHint {
        fusion_group: fusion_group & 0x0F,
        policy,
    };

    let frame = if opcode == Opcode::ConstF64 {
        let c = const_val.ok_or_else(|| RlsError::LineParse {
            line: line_no,
            message: "ConstF64 requires const=<f64>".to_owned(),
        })?;
        let mut f = NodeFrame::const_f64(c, schedule);
        f.type_lane = type_lane;
        f.proof_idx = proof_idx;
        if let Some(tok) = explicit_token {
            f.token_op = tok;
        }
        f
    } else {
        let in_a = if let Some(a_str) = raw_a {
            if matches!(opcode, Opcode::ArenaInput | Opcode::ArenaMutate) {
                a_str.parse::<u32>().map_err(|_e| RlsError::LineParse {
                    line: line_no,
                    message: format!("expected numeric arena_id for a=, got '{a_str}'"),
                })?
            } else {
                resolve_sym_or_u32(line_no, a_str, symbols)?
            }
        } else {
            0
        };

        let in_b = if let Some(b_str) = raw_b {
            resolve_sym_or_u32(line_no, b_str, symbols)?
        } else {
            0
        };

        let token_op = if let Some(tok) = explicit_token {
            tok
        } else {
            let arena_u16 = u16::try_from(in_a).unwrap_or(0);
            match opcode {
                Opcode::ArenaInput => TokenOp::read_lease(arena_u16, 1),
                Opcode::ArenaMutate => TokenOp::affine_consume(arena_u16, 1),
                _ => TokenOp::PURE,
            }
        };

        NodeFrame {
            opcode,
            type_lane,
            schedule,
            proof_idx,
            token_op,
            in_a,
            in_b,
        }
    };

    if symbols.contains_key(label) {
        return Err(RlsError::Symbol(TranspileError::DuplicateSymbol(
            label.to_owned(),
        )));
    }
    let idx = u32::try_from(nodes.len()).unwrap_or(0);
    nodes.push(frame);
    symbols.insert(label.to_owned(), idx);
    Ok(())
}

fn resolve_sym_or_u32(
    line_no: usize,
    raw: &str,
    symbols: &HashMap<String, u32>,
) -> Result<u32, RlsError> {
    if let Some(idx) = symbols.get(raw).copied() {
        return Ok(idx);
    }
    if let Ok(num) = raw.parse::<u32>() {
        return Ok(num);
    }
    if line_no == 0 {
        Err(RlsError::Symbol(TranspileError::UnknownSymbol(
            raw.to_owned(),
        )))
    } else {
        Err(RlsError::LineParse {
            line: line_no,
            message: format!("unknown node symbol or index '{raw}'"),
        })
    }
}

fn parse_u64_auto(raw: &str) -> Option<u64> {
    if let Some(hex) = raw.strip_prefix("0x").or_else(|| raw.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok()
    } else {
        raw.parse::<u64>().ok()
    }
}

fn format_type_lane(lane: TypeLane) -> &'static str {
    match (lane.elem, lane.width) {
        (ElemType::F64, SimdWidth::Scalar) => "f64x1",
        (ElemType::F64, SimdWidth::Lane4) => "f64x4",
        (ElemType::F64, SimdWidth::Lane8) => "f64x8",
        (ElemType::I64, SimdWidth::Scalar) => "i64x1",
        (ElemType::I64, SimdWidth::Lane4) => "i64x4",
        (ElemType::I64, SimdWidth::Lane8) => "i64x8",
        (ElemType::BoolMask, SimdWidth::Scalar) => "maskx1",
        (ElemType::BoolMask, SimdWidth::Lane4) => "maskx4",
        (ElemType::BoolMask, SimdWidth::Lane8) => "maskx8",
    }
}

fn parse_type_lane(line_no: usize, raw: &str) -> Result<TypeLane, RlsError> {
    let lane = match raw {
        "f64x1" | "f64" | "scalar" => TypeLane::F64_SCALAR,
        "f64x4" => TypeLane {
            elem: ElemType::F64,
            width: SimdWidth::Lane4,
        },
        "f64x8" => TypeLane::F64X8,
        "i64x1" | "i64" => TypeLane {
            elem: ElemType::I64,
            width: SimdWidth::Scalar,
        },
        "i64x4" => TypeLane {
            elem: ElemType::I64,
            width: SimdWidth::Lane4,
        },
        "i64x8" => TypeLane::I64X8,
        "maskx1" | "bool" => TypeLane {
            elem: ElemType::BoolMask,
            width: SimdWidth::Scalar,
        },
        "maskx4" => TypeLane {
            elem: ElemType::BoolMask,
            width: SimdWidth::Lane4,
        },
        "maskx8" => TypeLane::MASKX8,
        _ => {
            return Err(RlsError::LineParse {
                line: line_no,
                message: format!("unknown type lane '{raw}'"),
            })
        }
    };
    Ok(lane)
}

fn format_opcode(op: Opcode) -> &'static str {
    match op {
        Opcode::ArenaInput => "ArenaInput",
        Opcode::ConstF64 => "ConstF64",
        Opcode::ArenaSlice => "ArenaSlice",
        Opcode::ArenaMutate => "ArenaMutate",
        Opcode::Add => "Add",
        Opcode::Sub => "Sub",
        Opcode::Mul => "Mul",
        Opcode::CmpGt => "CmpGt",
        Opcode::SelectZero => "SelectZero",
        Opcode::FilterCompact => "FilterCompact",
        Opcode::ReduceSum => "ReduceSum",
        Opcode::ReduceMax => "ReduceMax",
        Opcode::Gather => "Gather",
        Opcode::RingEnqueue => "RingEnqueue",
        Opcode::AssertContract => "AssertContract",
    }
}

fn parse_opcode(line_no: usize, raw: &str) -> Result<Opcode, RlsError> {
    let op = match raw {
        "ArenaInput" => Opcode::ArenaInput,
        "ConstF64" => Opcode::ConstF64,
        "ArenaSlice" => Opcode::ArenaSlice,
        "ArenaMutate" => Opcode::ArenaMutate,
        "Add" => Opcode::Add,
        "Sub" => Opcode::Sub,
        "Mul" => Opcode::Mul,
        "CmpGt" => Opcode::CmpGt,
        "SelectZero" => Opcode::SelectZero,
        "FilterCompact" => Opcode::FilterCompact,
        "ReduceSum" => Opcode::ReduceSum,
        "ReduceMax" => Opcode::ReduceMax,
        "Gather" => Opcode::Gather,
        "RingEnqueue" => Opcode::RingEnqueue,
        "AssertContract" => Opcode::AssertContract,
        _ => {
            return Err(RlsError::LineParse {
                line: line_no,
                message: format!("unknown opcode '{raw}'"),
            })
        }
    };
    Ok(op)
}

fn format_policy(policy: ExecPolicy) -> &'static str {
    match policy {
        ExecPolicy::AutoL1Tile => "l1",
        ExecPolicy::SimdStream => "simd",
        ExecPolicy::MultiCore => "multicore",
        ExecPolicy::GpuDma => "gpudma",
    }
}

fn parse_policy(line_no: usize, raw: &str) -> Result<ExecPolicy, RlsError> {
    match raw {
        "l1" | "AutoL1Tile" => Ok(ExecPolicy::AutoL1Tile),
        "simd" | "SimdStream" => Ok(ExecPolicy::SimdStream),
        "multicore" | "MultiCore" => Ok(ExecPolicy::MultiCore),
        "gpudma" | "GpuDma" => Ok(ExecPolicy::GpuDma),
        _ => Err(RlsError::LineParse {
            line: line_no,
            message: format!("unknown execution policy '{raw}'"),
        }),
    }
}

fn format_token_op(tok: TokenOp) -> String {
    match tok.mode {
        TokenMode::PureValue => "pure".to_owned(),
        TokenMode::ReadLease => format!("read({},{})", tok.arena_id, tok.version),
        TokenMode::AffineConsume => format!("consume({},{})", tok.arena_id, tok.version),
    }
}

fn parse_token_op(line_no: usize, raw: &str) -> Result<TokenOp, RlsError> {
    if raw == "pure" {
        return Ok(TokenOp::PURE);
    }
    if let Some(inner) = raw.strip_prefix("read(").and_then(|s| s.strip_suffix(')')) {
        let (a_str, v_str) = inner.split_once(',').ok_or_else(|| RlsError::LineParse {
            line: line_no,
            message: format!("invalid read token '{raw}'"),
        })?;
        let arena_id = a_str.parse::<u16>().map_err(|_e| RlsError::LineParse {
            line: line_no,
            message: format!("invalid arena_id in '{raw}'"),
        })?;
        let version = v_str.parse::<u16>().map_err(|_e| RlsError::LineParse {
            line: line_no,
            message: format!("invalid version in '{raw}'"),
        })?;
        return Ok(TokenOp::read_lease(arena_id, version));
    }
    if let Some(inner) = raw
        .strip_prefix("consume(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let (a_str, v_str) = inner.split_once(',').ok_or_else(|| RlsError::LineParse {
            line: line_no,
            message: format!("invalid consume token '{raw}'"),
        })?;
        let arena_id = a_str.parse::<u16>().map_err(|_e| RlsError::LineParse {
            line: line_no,
            message: format!("invalid arena_id in '{raw}'"),
        })?;
        let version = v_str.parse::<u16>().map_err(|_e| RlsError::LineParse {
            line: line_no,
            message: format!("invalid version in '{raw}'"),
        })?;
        return Ok(TokenOp::affine_consume(arena_id, version));
    }
    Err(RlsError::LineParse {
        line: line_no,
        message: format!("unknown token op '{raw}'"),
    })
}

fn format_node_ref(opcode: Opcode, is_a: bool, idx: u32) -> String {
    if !is_a
        && matches!(
            opcode,
            Opcode::ArenaInput
                | Opcode::ArenaSlice
                | Opcode::ReduceSum
                | Opcode::ReduceMax
                | Opcode::RingEnqueue
                | Opcode::AssertContract
        )
        && idx == 0
    {
        "0".to_owned()
    } else {
        format!("$n{idx}")
    }
}
