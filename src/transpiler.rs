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
    RegionRole, RegionState, RulProgram, ScheduleHint, SimdWidth, TokenMode, TokenOp, TypeLane,
    CALL_ARG_SENTINEL, CAP_DIVERGE, CAP_GPU_DISPATCH, CAP_NET_RING, CAP_NONE, CAP_STORAGE_DMA,
    REGION_FOOTPRINT_NONE,
};

/// Error returned when resolving symbolic labels in [`SymbolicGraphBuilder`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranspileError {
    /// Referenced symbolic node label was never defined.
    UnknownSymbol(String),
    /// Duplicate symbolic label definition.
    DuplicateSymbol(String),
    /// `Opcode::Fma` operand `b` or `c` node index exceeds the 16-bit packed limit (`65,535`).
    FmaOperandIndexOverflow {
        /// Symbolic label or operand token.
        symbol: String,
        /// Resolved 32-bit node index (`> 65,535`).
        node_idx: u32,
    },
}

impl fmt::Display for TranspileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownSymbol(sym) => write!(f, "unknown symbolic label: {sym}"),
            Self::DuplicateSymbol(sym) => write!(f, "duplicate symbolic label: {sym}"),
            Self::FmaOperandIndexOverflow { symbol, node_idx } => write!(
                f,
                "Fma operand '{symbol}' resolved to node index {node_idx}, exceeding u16::MAX (65535)"
            ),
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

    /// Emits a ternary [`Opcode::Fma`] node (`in_a_sym * in_b_sym + in_c_sym`).
    ///
    /// # Errors
    /// Returns [`TranspileError`] if any input symbol is unknown or `label` is duplicated.
    pub fn emit_fma_op(
        &mut self,
        label: &str,
        type_lane: TypeLane,
        in_a_sym: &str,
        in_b_sym: &str,
        in_c_sym: &str,
        schedule: ScheduleHint,
    ) -> Result<u32, TranspileError> {
        let in_a = self.resolve(in_a_sym)?;
        let in_b = self.resolve(in_b_sym)?;
        let in_c = self.resolve(in_c_sym)?;
        let b16 = u16::try_from(in_b).map_err(|_e| TranspileError::FmaOperandIndexOverflow {
            symbol: in_b_sym.to_owned(),
            node_idx: in_b,
        })?;
        let c16 = u16::try_from(in_c).map_err(|_e| TranspileError::FmaOperandIndexOverflow {
            symbol: in_c_sym.to_owned(),
            node_idx: in_c,
        })?;
        self.register_symbol(label, NodeFrame::fma(type_lane, in_a, b16, c16, schedule))
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
            const_data: Vec::new(),
            link_meta: Vec::new(),
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
        "HEADER : Magic=\"RUL\\x00\\x01\\x00\\x00\\x00\" | Hash={hash:#018x} | CapMask={:#06x} | Output=Node({}) | ConstData={}B | LinkMeta={}B",
        program.capability_mask,
        program.output_node,
        program.const_data.len(),
        program.link_meta.len()
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
        } else if frame.opcode == Opcode::ConstI64 {
            format!("const_i64={}", frame.decode_const_i64())
        } else if frame.opcode == Opcode::Fma {
            let (b, c) = frame.decode_fma_bc();
            format!("in_a={}, in_b={b}, in_c={c}", frame.in_a)
        } else if frame.opcode == Opcode::RegionStore {
            let (idx_node, val_node) = frame.decode_u16_pair();
            format!("slot={}, idx={idx_node}, val={val_node}", frame.in_a)
        } else if frame.opcode == Opcode::RegionMove {
            format!("src={}, dst={}", frame.in_a, frame.in_b)
        } else if matches!(
            frame.opcode,
            Opcode::RegionLease | Opcode::RegionEndLease | Opcode::RegionDrop
        ) {
            format!("slot={}", frame.in_a)
        } else if frame.opcode == Opcode::Func {
            format!("func_id={}", frame.in_a)
        } else if frame.opcode == Opcode::Return {
            format!("ret_val={}", frame.in_a)
        } else if frame.opcode == Opcode::Call {
            let (arg0, arg1) = frame.decode_u16_pair();
            format!("func_id={}, arg0={arg0}, arg1={arg1}", frame.in_a)
        } else if frame.opcode == Opcode::CallArg {
            format!("arg2={}, arg3={}", frame.in_a, frame.in_b)
        } else if frame.opcode == Opcode::ConstDataLen {
            format!("seg={}", frame.in_a)
        } else if frame.opcode == Opcode::ConstDataLoad {
            format!("seg={}, idx={}", frame.in_a, frame.in_b)
        } else if frame.opcode == Opcode::BrTable {
            format!("sel={}, table={}", frame.in_a, frame.in_b)
        } else if frame.opcode == Opcode::CallIndirect {
            let (table_id, idx_node) = frame.decode_in_a_u16_pair();
            let (arg0, arg1) = frame.decode_u16_pair();
            format!("table={table_id}, idx={idx_node}, arg0={arg0}, arg1={arg1}")
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

    let contiguous_const = const_data_is_contiguous_partition(program);
    if !program.const_data.is_empty() && !contiguous_const {
        let _ = writeln!(
            out,
            "const_data hex \"{}\"",
            encode_hex(&program.const_data)
        );
    }
    if !program.link_meta.is_empty() {
        let _ = writeln!(out, "link_meta hex \"{}\"", encode_hex(&program.link_meta));
    }

    for (idx, proof) in program.proofs.iter().enumerate() {
        if contiguous_const {
            match proof.rule_kind {
                ProofRuleKind::ConstDataSeg => {
                    let lo = proof.bound_lo as usize;
                    let hi = lo.saturating_add(proof.bound_hi as usize);
                    if let Some(slice) = program.const_data.get(lo..hi) {
                        if is_printable_rls_string(slice) {
                            let _ = writeln!(
                                out,
                                "data seg {} \"{}\"",
                                proof.target_arena,
                                escape_rls_string(slice)
                            );
                        } else {
                            let _ = writeln!(
                                out,
                                "data seg {} hex \"{}\"",
                                proof.target_arena,
                                encode_hex(slice)
                            );
                        }
                        continue;
                    }
                }
                ProofRuleKind::BrTableSpec => {
                    let lo = proof.bound_lo as usize;
                    let byte_len = (proof.bound_hi as usize).saturating_mul(4);
                    if let Some(slice) = program.const_data.get(lo..lo.saturating_add(byte_len)) {
                        let mut targets = String::new();
                        for (i, chunk) in slice.as_chunks::<4>().0.iter().enumerate() {
                            if i > 0 {
                                targets.push(',');
                            }
                            let d = u32::from_le_bytes(*chunk);
                            let _ = write!(targets, "{d}");
                        }
                        let _ = writeln!(
                            out,
                            "br_table {} default={} targets={targets}",
                            proof.witness_node, proof.target_arena
                        );
                        continue;
                    }
                }
                ProofRuleKind::CallTable => {
                    let lo = proof.bound_lo as usize;
                    let byte_len = (proof.bound_hi as usize).saturating_mul(4);
                    if let Some(slice) = program.const_data.get(lo..lo.saturating_add(byte_len)) {
                        let mut funcs = String::new();
                        for (i, chunk) in slice.as_chunks::<4>().0.iter().enumerate() {
                            if i > 0 {
                                funcs.push(',');
                            }
                            let fid = u32::from_le_bytes(*chunk);
                            let _ = write!(funcs, "{fid}");
                        }
                        let _ = writeln!(out, "call_table {} funcs={funcs}", proof.target_arena);
                        continue;
                    }
                }
                _ => {}
            }
        }
        if proof.rule_kind == ProofRuleKind::FuncRegionEffect {
            if let Some(eff) = proof.decode_func_region_effect() {
                let slot_s = eff
                    .slot
                    .map_or_else(|| "none".to_owned(), |s| s.to_string());
                let pre_s = format_region_state(eff.pre);
                let post_s = format_region_state(eff.post);
                let _ = writeln!(
                    out,
                    "proof {} rule=func_region_effect func={} slot={slot_s} pre={pre_s} post={post_s}",
                    idx + 1,
                    eff.func_id,
                );
            } else {
                let _ = writeln!(
                    out,
                    "proof {} rule=func_region_effect target={} lo={} hi={} witness={}",
                    idx + 1,
                    proof.target_arena,
                    proof.bound_lo,
                    proof.bound_hi,
                    proof.witness_node,
                );
            }
            continue;
        }
        let rule_str = match proof.rule_kind {
            ProofRuleKind::SliceBounds => "slice_bounds",
            ProofRuleKind::GatherIndexInBounds => "gather_bounds",
            ProofRuleKind::BlockSig => "block_sig",
            ProofRuleKind::LoopBound => "loop_bound",
            ProofRuleKind::FuncSig => "func_sig",
            ProofRuleKind::RegionDecl => "region_decl",
            ProofRuleKind::RegionIndexInBounds => "region_index_in_bounds",
            ProofRuleKind::FuncRegionEffect => "func_region_effect",
            ProofRuleKind::ConstDataSeg => "const_data_seg",
            ProofRuleKind::ConstDataInBounds => "const_data_in_bounds",
            ProofRuleKind::BrTableSpec => "br_table_spec",
            ProofRuleKind::CallTable => "call_table",
        };
        let (hi_str, witness_str) = if proof.rule_kind == ProofRuleKind::FuncSig {
            if let Some(sig) = proof.decode_func_sig() {
                let hi = if sig.arity == 0 {
                    "0".to_owned()
                } else {
                    let mut s = String::new();
                    for i in 0..usize::from(sig.arity) {
                        if i > 0 {
                            s.push(',');
                        }
                        if let Some(&el) = sig.params.get(i) {
                            s.push_str(format_elem_short(el));
                        }
                    }
                    s
                };
                let wit = sig
                    .ret_elem
                    .map_or_else(|| "void".to_owned(), |el| format_elem_short(el).to_owned());
                (hi, wit)
            } else {
                (proof.bound_hi.to_string(), proof.witness_node.to_string())
            }
        } else if proof.rule_kind == ProofRuleKind::RegionDecl {
            let wit = if let Some((elem, role)) =
                ProofEntry::decode_region_decl_witness(proof.witness_node)
            {
                let elem_s = format_elem_short(elem);
                let role_s = match role {
                    RegionRole::Plain => "plain",
                    RegionRole::Input => "input",
                    RegionRole::Output => "output",
                };
                format!("{elem_s}:{role_s}")
            } else {
                proof.witness_node.to_string()
            };
            (proof.bound_hi.to_string(), wit)
        } else {
            (proof.bound_hi.to_string(), proof.witness_node.to_string())
        };
        let _ = writeln!(
            out,
            "proof {} rule={} target={} lo={} hi={hi_str} witness={witness_str}",
            idx + 1,
            rule_str,
            proof.target_arena,
            proof.bound_lo,
        );
    }

    let total_nodes = program.nodes.len();
    let mut current_func_start: Option<u32> = None;
    let mut ctrl_depth: usize = 0;

    for (idx, frame) in program.nodes.iter().enumerate() {
        let op_str = format_opcode(frame.opcode);
        let lane_str = format_type_lane(frame.type_lane);
        let pol_str = format_policy(frame.schedule.policy);
        let tok_str = format_token_op(frame.token_op);

        if frame.opcode == Opcode::Func && ctrl_depth == 0 {
            current_func_start = Some(u32::try_from(idx).unwrap_or(0));
            ctrl_depth = 1;
        } else if matches!(frame.opcode, Opcode::Block | Opcode::Loop | Opcode::If) {
            ctrl_depth = ctrl_depth.saturating_add(1);
        }

        if frame.opcode == Opcode::ConstF64 {
            let val = frame.decode_const_f64();
            let _ = writeln!(
                out,
                "node $n{idx} = {op_str} {lane_str} fg={} pol={pol_str} proof={} tok={tok_str} const={val:?}",
                frame.schedule.fusion_group, frame.proof_idx
            );
        } else if frame.opcode == Opcode::ConstI64 {
            let val = frame.decode_const_i64();
            let _ = writeln!(
                out,
                "node $n{idx} = {op_str} {lane_str} fg={} pol={pol_str} proof={} tok={tok_str} const_i64={val}",
                frame.schedule.fusion_group, frame.proof_idx
            );
        } else if frame.opcode == Opcode::Fma {
            let (b, c) = frame.decode_fma_bc();
            let a_ref = format_node_ref(
                current_func_start,
                total_nodes,
                frame.opcode,
                true,
                frame.in_a,
            );
            let b_ref = format_node_ref(current_func_start, total_nodes, frame.opcode, false, b);
            let c_ref = format_node_ref(current_func_start, total_nodes, frame.opcode, false, c);
            let _ = writeln!(
                out,
                "node $n{idx} = {op_str} {lane_str} fg={} pol={pol_str} proof={} tok={tok_str} a={a_ref} b={b_ref} c={c_ref}",
                frame.schedule.fusion_group, frame.proof_idx
            );
        } else if frame.opcode == Opcode::RegionStore {
            let (idx_node, val_node) = frame.decode_u16_pair();
            let b_ref = format_node_ref(
                current_func_start,
                total_nodes,
                frame.opcode,
                false,
                idx_node,
            );
            let c_ref = format_node_ref(
                current_func_start,
                total_nodes,
                frame.opcode,
                false,
                val_node,
            );
            let _ = writeln!(
                out,
                "node $n{idx} = {op_str} {lane_str} fg={} pol={pol_str} proof={} tok={tok_str} a={} b={b_ref} c={c_ref}",
                frame.schedule.fusion_group, frame.proof_idx, frame.in_a
            );
        } else if frame.opcode == Opcode::Call {
            let (arg0, arg1) = frame.decode_u16_pair();
            let sentinel = u32::from(CALL_ARG_SENTINEL);
            let b_ref = if arg0 == sentinel {
                "none".to_owned()
            } else {
                format_node_ref(current_func_start, total_nodes, frame.opcode, false, arg0)
            };
            let c_ref = if arg1 == sentinel {
                "none".to_owned()
            } else {
                format_node_ref(current_func_start, total_nodes, frame.opcode, false, arg1)
            };
            let _ = writeln!(
                out,
                "node $n{idx} = {op_str} {lane_str} fg={} pol={pol_str} proof={} tok={tok_str} a={} b={b_ref} c={c_ref}",
                frame.schedule.fusion_group, frame.proof_idx, frame.in_a
            );
        } else if frame.opcode == Opcode::CallIndirect {
            let (table_id, idx_node) = frame.decode_in_a_u16_pair();
            let (arg0, arg1) = frame.decode_u16_pair();
            let sentinel = u32::from(CALL_ARG_SENTINEL);
            let idx_ref = format_node_ref(
                current_func_start,
                total_nodes,
                frame.opcode,
                true,
                idx_node,
            );
            let b_ref = if arg0 == sentinel {
                "none".to_owned()
            } else {
                format_node_ref(current_func_start, total_nodes, frame.opcode, false, arg0)
            };
            let c_ref = if arg1 == sentinel {
                "none".to_owned()
            } else {
                format_node_ref(current_func_start, total_nodes, frame.opcode, false, arg1)
            };
            let _ = writeln!(
                out,
                "node $n{idx} = {op_str} {lane_str} fg={} pol={pol_str} proof={} tok={tok_str} table={table_id} idx={idx_ref} b={b_ref} c={c_ref}",
                frame.schedule.fusion_group, frame.proof_idx
            );
        } else if frame.opcode == Opcode::CallArg {
            let sentinel = u32::from(CALL_ARG_SENTINEL);
            let a_ref = if frame.in_a == sentinel {
                "none".to_owned()
            } else {
                format_node_ref(
                    current_func_start,
                    total_nodes,
                    frame.opcode,
                    true,
                    frame.in_a,
                )
            };
            let b_ref = if frame.in_b == sentinel {
                "none".to_owned()
            } else {
                format_node_ref(
                    current_func_start,
                    total_nodes,
                    frame.opcode,
                    false,
                    frame.in_b,
                )
            };
            let _ = writeln!(
                out,
                "node $n{idx} = {op_str} {lane_str} fg={} pol={pol_str} proof={} tok={tok_str} a={a_ref} b={b_ref}",
                frame.schedule.fusion_group, frame.proof_idx
            );
        } else if frame.opcode == Opcode::Return {
            let a_ref = if frame.in_a == 0 {
                "0".to_owned()
            } else {
                format_node_ref(
                    current_func_start,
                    total_nodes,
                    frame.opcode,
                    true,
                    frame.in_a,
                )
            };
            let _ = writeln!(
                out,
                "node $n{idx} = {op_str} {lane_str} fg={} pol={pol_str} proof={} tok={tok_str} a={a_ref} b={}",
                frame.schedule.fusion_group, frame.proof_idx, frame.in_b
            );
        } else if matches!(
            frame.opcode,
            Opcode::ArenaInput
                | Opcode::ArenaMutate
                | Opcode::LocalSet
                | Opcode::ConstDataLoad
                | Opcode::RegionNew
                | Opcode::RegionGrow
                | Opcode::RegionLoad
        ) {
            let b_ref = format_node_ref(
                current_func_start,
                total_nodes,
                frame.opcode,
                false,
                frame.in_b,
            );
            let _ = writeln!(
                out,
                "node $n{idx} = {op_str} {lane_str} fg={} pol={pol_str} proof={} tok={tok_str} a={} b={b_ref}",
                frame.schedule.fusion_group, frame.proof_idx, frame.in_a
            );
        } else if matches!(
            frame.opcode,
            Opcode::LocalGet
                | Opcode::ConstDataLen
                | Opcode::Block
                | Opcode::Loop
                | Opcode::Br
                | Opcode::RegionLen
                | Opcode::RegionFree
                | Opcode::RegionLease
                | Opcode::RegionEndLease
                | Opcode::RegionMove
                | Opcode::RegionDrop
                | Opcode::Func
        ) {
            let _ = writeln!(
                out,
                "node $n{idx} = {op_str} {lane_str} fg={} pol={pol_str} proof={} tok={tok_str} a={} b={}",
                frame.schedule.fusion_group, frame.proof_idx, frame.in_a, frame.in_b
            );
        } else if matches!(frame.opcode, Opcode::If | Opcode::BrIf | Opcode::BrTable) {
            let a_ref = format_node_ref(
                current_func_start,
                total_nodes,
                frame.opcode,
                true,
                frame.in_a,
            );
            let _ = writeln!(
                out,
                "node $n{idx} = {op_str} {lane_str} fg={} pol={pol_str} proof={} tok={tok_str} a={a_ref} b={}",
                frame.schedule.fusion_group, frame.proof_idx, frame.in_b
            );
        } else if matches!(frame.opcode, Opcode::Else | Opcode::End) {
            let a_ref = if frame.in_b == 0 {
                frame.in_a.to_string()
            } else {
                format_node_ref(
                    current_func_start,
                    total_nodes,
                    frame.opcode,
                    true,
                    frame.in_a,
                )
            };
            let _ = writeln!(
                out,
                "node $n{idx} = {op_str} {lane_str} fg={} pol={pol_str} proof={} tok={tok_str} a={a_ref} b={}",
                frame.schedule.fusion_group, frame.proof_idx, frame.in_b
            );
        } else {
            let a_ref = format_node_ref(
                current_func_start,
                total_nodes,
                frame.opcode,
                true,
                frame.in_a,
            );
            let b_ref = format_node_ref(
                current_func_start,
                total_nodes,
                frame.opcode,
                false,
                frame.in_b,
            );
            let _ = writeln!(
                out,
                "node $n{idx} = {op_str} {lane_str} fg={} pol={pol_str} proof={} tok={tok_str} a={a_ref} b={b_ref}",
                frame.schedule.fusion_group, frame.proof_idx
            );
        }

        if frame.opcode == Opcode::End {
            ctrl_depth = ctrl_depth.saturating_sub(1);
            if ctrl_depth == 0 {
                current_func_start = None;
            }
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
    let mut const_data: Vec<u8> = Vec::new();
    let mut link_meta: Vec<u8> = Vec::new();
    let mut seg_symbols: HashMap<String, u16> = HashMap::new();
    let mut next_seg_id: u16 = 0;
    let mut global_symbols: HashMap<String, u32> = HashMap::new();
    let mut func_symbols: HashMap<String, u32> = HashMap::new();
    let mut current_func_start: Option<u32> = None;
    let mut ctrl_depth: usize = 0;
    let mut explicit_output: Option<String> = None;

    for (line_idx, raw_line) in source.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = strip_rls_comment(raw_line).trim();
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
                capability_mask = parse_cap_mask(raw_val).ok_or_else(|| RlsError::LineParse {
                    line: line_no,
                    message: format!("invalid capability mask: {raw_val}"),
                })?;
            }
            "arena" => {
                let arena = parse_arena_line(line_no, parts, arenas.len())?;
                arenas.push(arena);
            }
            "data" => {
                let proof = parse_data_seg_line(
                    line_no,
                    line,
                    &mut const_data,
                    &mut seg_symbols,
                    &mut next_seg_id,
                )?;
                proofs.push(proof);
            }
            "const_data" => {
                let bytes = parse_hex_section_line(line_no, line, "const_data")?;
                const_data.extend_from_slice(&bytes);
            }
            "link_meta" => {
                let bytes = parse_hex_section_line(line_no, line, "link_meta")?;
                link_meta.extend_from_slice(&bytes);
            }
            "br_table" => {
                let proof = parse_br_table_line(line_no, parts, &mut const_data)?;
                proofs.push(proof);
            }
            "call_table" => {
                let proof = parse_call_table_line(line_no, parts, &mut const_data)?;
                proofs.push(proof);
            }
            "proof" => {
                let proof = parse_proof_line(line_no, parts, &seg_symbols)?;
                proofs.push(proof);
            }
            "node" => {
                parse_node_line(
                    line_no,
                    parts,
                    &mut nodes,
                    &mut global_symbols,
                    &mut func_symbols,
                    &seg_symbols,
                    &mut current_func_start,
                    &mut ctrl_depth,
                )?;
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
        resolve_sym_or_u32(0, &sym, &global_symbols)?
    } else {
        u32::try_from(nodes.len().saturating_sub(1)).unwrap_or(0)
    };

    Ok(RulProgram {
        capability_mask,
        arenas,
        proofs,
        nodes,
        output_node,
        const_data,
        link_meta,
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

fn format_region_state(state: RegionState) -> &'static str {
    match state {
        RegionState::Owned => "owned",
        RegionState::Leased => "leased",
        RegionState::Moved => "moved",
    }
}

fn parse_region_state(line_no: usize, raw: &str) -> Result<u32, RlsError> {
    if let Some(num) = parse_u64_auto(raw) {
        return u32::try_from(num).map_err(|_e| RlsError::LineParse {
            line: line_no,
            message: format!("invalid region state '{raw}'"),
        });
    }
    match raw {
        "owned" | "Owned" => Ok(RegionState::Owned as u32),
        "leased" | "Leased" => Ok(RegionState::Leased as u32),
        "moved" | "Moved" => Ok(RegionState::Moved as u32),
        _ => Err(RlsError::LineParse {
            line: line_no,
            message: format!("unknown region state '{raw}'"),
        }),
    }
}

fn parse_proof_line<'a>(
    line_no: usize,
    mut parts: impl Iterator<Item = &'a str>,
    seg_symbols: &HashMap<String, u16>,
) -> Result<ProofEntry, RlsError> {
    let first_tok = parts.next().ok_or_else(|| RlsError::LineParse {
        line: line_no,
        message: "missing proof index".to_owned(),
    })?;

    let mut rule_kind = match first_tok {
        "func_region_effect" | "FuncRegionEffect" => ProofRuleKind::FuncRegionEffect,
        _ => ProofRuleKind::SliceBounds,
    };
    let mut target_arena: u16 = 0;
    let mut bound_lo: u32 = 0;
    let mut raw_hi: Option<&str> = None;
    let mut raw_witness: Option<&str> = None;

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
                    "block_sig" | "BlockSig" | "3" => ProofRuleKind::BlockSig,
                    "loop_bound" | "LoopBound" | "4" => ProofRuleKind::LoopBound,
                    "func_sig" | "FuncSig" | "5" => ProofRuleKind::FuncSig,
                    "region_decl" | "RegionDecl" | "6" => ProofRuleKind::RegionDecl,
                    "region_index_in_bounds" | "RegionIndexInBounds" | "7" => {
                        ProofRuleKind::RegionIndexInBounds
                    }
                    "func_region_effect" | "FuncRegionEffect" | "8" => {
                        ProofRuleKind::FuncRegionEffect
                    }
                    "const_data_seg" | "ConstDataSeg" | "9" => ProofRuleKind::ConstDataSeg,
                    "const_data_in_bounds" | "ConstDataInBounds" | "10" => {
                        ProofRuleKind::ConstDataInBounds
                    }
                    "br_table_spec" | "BrTableSpec" | "11" => ProofRuleKind::BrTableSpec,
                    "call_table" | "CallTable" | "12" => ProofRuleKind::CallTable,
                    _ => {
                        return Err(RlsError::LineParse {
                            line: line_no,
                            message: format!("unknown proof rule '{v}'"),
                        })
                    }
                };
            }
            "target" | "func" => {
                target_arena = if let Some(&seg_id) = seg_symbols.get(v) {
                    seg_id
                } else {
                    v.parse::<u16>().map_err(|_e| RlsError::LineParse {
                        line: line_no,
                        message: format!("invalid target arena '{v}'"),
                    })?
                };
            }
            "lo" | "slot" => {
                bound_lo = if v == "none" || v == "None" {
                    REGION_FOOTPRINT_NONE
                } else if let Some(num) = parse_u64_auto(v) {
                    u32::try_from(num).map_err(|_e| RlsError::LineParse {
                        line: line_no,
                        message: format!("invalid bound_lo '{v}'"),
                    })?
                } else {
                    return Err(RlsError::LineParse {
                        line: line_no,
                        message: format!("invalid bound_lo '{v}'"),
                    });
                };
            }
            "hi" | "pre" => {
                raw_hi = Some(v);
            }
            "witness" | "post" => {
                raw_witness = Some(v);
            }
            _ => {
                return Err(RlsError::LineParse {
                    line: line_no,
                    message: format!("unknown proof attribute '{k}'"),
                });
            }
        }
    }

    if rule_kind == ProofRuleKind::FuncRegionEffect {
        let bound_hi = match raw_hi {
            Some(hi_str) => parse_region_state(line_no, hi_str)?,
            None if bound_lo == REGION_FOOTPRINT_NONE => 0,
            None => {
                return Err(RlsError::LineParse {
                    line: line_no,
                    message: "func_region_effect requires pre=<owned|leased|moved>".to_owned(),
                });
            }
        };
        let witness_node = match raw_witness {
            Some(w_str) => parse_region_state(line_no, w_str)?,
            None if bound_lo == REGION_FOOTPRINT_NONE => 0,
            None => {
                return Err(RlsError::LineParse {
                    line: line_no,
                    message: "func_region_effect requires post=<owned|leased|moved>".to_owned(),
                });
            }
        };
        return Ok(ProofEntry {
            rule_kind,
            target_arena,
            bound_lo,
            bound_hi,
            witness_node,
        });
    }

    let hi_str = raw_hi.ok_or_else(|| RlsError::LineParse {
        line: line_no,
        message: "proof directive requires hi=<u32>".to_owned(),
    })?;
    let bound_hi = if let Some(num) = parse_u64_auto(hi_str) {
        u32::try_from(num).map_err(|_e| RlsError::LineParse {
            line: line_no,
            message: format!("invalid bound_hi '{hi_str}'"),
        })?
    } else if rule_kind == ProofRuleKind::FuncSig {
        if hi_str == "none" {
            0
        } else {
            let mut packed: u32 = 0;
            for (idx, tok) in hi_str.split(',').enumerate() {
                if idx >= 4 {
                    return Err(RlsError::LineParse {
                        line: line_no,
                        message: format!("FuncSig supports at most 4 parameters, got '{hi_str}'"),
                    });
                }
                let el = parse_elem_short(line_no, tok)?;
                packed |= (el as u32) << (idx * 4);
            }
            packed
        }
    } else {
        return Err(RlsError::LineParse {
            line: line_no,
            message: format!("invalid bound_hi '{hi_str}'"),
        });
    };

    let witness_node = if let Some(v) = raw_witness {
        if let Some(num) = parse_u64_auto(v) {
            u32::try_from(num).map_err(|_e| RlsError::LineParse {
                line: line_no,
                message: format!("invalid witness_node '{v}'"),
            })?
        } else if v == "void" {
            0
        } else if rule_kind == ProofRuleKind::FuncSig {
            parse_elem_short(line_no, v)? as u32
        } else if let Some((elem_str, role_str)) = v.split_once(':') {
            let elem = match elem_str {
                "f64" | "F64" => ElemType::F64,
                "i64" | "I64" => ElemType::I64,
                "u8" | "U8" => ElemType::U8,
                _ => {
                    return Err(RlsError::LineParse {
                        line: line_no,
                        message: format!("unknown region element type '{elem_str}'"),
                    })
                }
            };
            let role = match role_str {
                "plain" | "Plain" => RegionRole::Plain,
                "input" | "Input" => RegionRole::Input,
                "output" | "Output" => RegionRole::Output,
                _ => {
                    return Err(RlsError::LineParse {
                        line: line_no,
                        message: format!("unknown region role '{role_str}'"),
                    })
                }
            };
            (elem as u32) | ((role as u32) << 8)
        } else {
            let tl = parse_type_lane(line_no, v)?;
            u32::from(tl.to_u8())
        }
    } else {
        0
    };

    Ok(ProofEntry {
        rule_kind,
        target_arena,
        bound_lo,
        bound_hi,
        witness_node,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "single-pass .rls node line parser tracking symbol tables and control depth"
)]
fn parse_node_line<'a>(
    line_no: usize,
    mut parts: impl Iterator<Item = &'a str>,
    nodes: &mut Vec<NodeFrame>,
    global_symbols: &mut HashMap<String, u32>,
    func_symbols: &mut HashMap<String, u32>,
    seg_symbols: &HashMap<String, u16>,
    current_func_start: &mut Option<u32>,
    ctrl_depth: &mut usize,
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
    let mut const_i64_val: Option<i64> = None;
    let mut raw_a: Option<&str> = None;
    let mut raw_b: Option<&str> = None;
    let mut raw_c: Option<&str> = None;
    let mut raw_table: Option<&str> = None;
    let mut raw_idx: Option<&str> = None;

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
                if opcode == Opcode::ConstI64 {
                    const_i64_val = Some(parse_i64_auto(v).ok_or_else(|| RlsError::LineParse {
                        line: line_no,
                        message: format!("invalid i64 constant '{v}'"),
                    })?);
                } else {
                    const_val = Some(v.parse::<f64>().map_err(|_e| RlsError::LineParse {
                        line: line_no,
                        message: format!("invalid f64 constant '{v}'"),
                    })?);
                }
            }
            "const_i64" => {
                const_i64_val = Some(parse_i64_auto(v).ok_or_else(|| RlsError::LineParse {
                    line: line_no,
                    message: format!("invalid i64 constant '{v}'"),
                })?);
            }
            "a" => {
                raw_a = Some(v);
            }
            "b" => {
                raw_b = Some(v);
            }
            "c" => {
                raw_c = Some(v);
            }
            "table" => {
                raw_table = Some(v);
            }
            "idx" => {
                raw_idx = Some(v);
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

    let active_symbols: &HashMap<String, u32> = if current_func_start.is_some() {
        func_symbols
    } else {
        global_symbols
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
    } else if opcode == Opcode::ConstI64 {
        let c = const_i64_val.ok_or_else(|| RlsError::LineParse {
            line: line_no,
            message: "ConstI64 requires const_i64=<i64>".to_owned(),
        })?;
        let mut f = NodeFrame::const_i64(c, schedule);
        f.type_lane = type_lane;
        f.proof_idx = proof_idx;
        if let Some(tok) = explicit_token {
            f.token_op = tok;
        }
        f
    } else {
        let in_a = if opcode == Opcode::CallIndirect {
            let table_str = raw_table.or(raw_a).ok_or_else(|| RlsError::LineParse {
                line: line_no,
                message: "CallIndirect requires table=<u16> and idx=<node>".to_owned(),
            })?;
            let idx_str = raw_idx.ok_or_else(|| RlsError::LineParse {
                line: line_no,
                message: "CallIndirect requires table=<u16> and idx=<node>".to_owned(),
            })?;
            let table_id = table_str.parse::<u16>().map_err(|_e| RlsError::LineParse {
                line: line_no,
                message: format!("invalid CallIndirect table id '{table_str}'"),
            })?;
            let idx_node = resolve_sym_or_u32(line_no, idx_str, active_symbols)?;
            let idx16 = u16::try_from(idx_node).map_err(|_e| {
                RlsError::Symbol(TranspileError::FmaOperandIndexOverflow {
                    symbol: idx_str.to_owned(),
                    node_idx: idx_node,
                })
            })?;
            NodeFrame::encode_u16_pair(table_id, idx16)
        } else if let Some(a_str) = raw_a {
            if matches!(opcode, Opcode::ConstDataLen | Opcode::ConstDataLoad) {
                if let Some(&seg_id) = seg_symbols.get(a_str) {
                    u32::from(seg_id)
                } else {
                    a_str.parse::<u32>().map_err(|_e| RlsError::LineParse {
                        line: line_no,
                        message: format!(
                            "expected numeric or named const data segment for a=, got '{a_str}'"
                        ),
                    })?
                }
            } else if matches!(
                opcode,
                Opcode::ArenaInput
                    | Opcode::ArenaMutate
                    | Opcode::LocalGet
                    | Opcode::LocalSet
                    | Opcode::Block
                    | Opcode::Loop
                    | Opcode::Br
                    | Opcode::RegionNew
                    | Opcode::RegionGrow
                    | Opcode::RegionLen
                    | Opcode::RegionLoad
                    | Opcode::RegionStore
                    | Opcode::RegionFree
                    | Opcode::RegionLease
                    | Opcode::RegionEndLease
                    | Opcode::RegionMove
                    | Opcode::RegionDrop
                    | Opcode::Func
                    | Opcode::Call
            ) {
                a_str.parse::<u32>().map_err(|_e| RlsError::LineParse {
                    line: line_no,
                    message: format!("expected numeric operand for a=, got '{a_str}'"),
                })?
            } else if opcode == Opcode::CallArg && a_str == "none" {
                u32::from(CALL_ARG_SENTINEL)
            } else {
                resolve_sym_or_u32(line_no, a_str, active_symbols)?
            }
        } else {
            0
        };

        let in_b = if matches!(opcode, Opcode::Fma | Opcode::RegionStore) {
            if raw_a.is_none() {
                return Err(RlsError::LineParse {
                    line: line_no,
                    message: format!("{opcode:?} requires a=, b=, and c="),
                });
            }
            let b_str = raw_b.ok_or_else(|| RlsError::LineParse {
                line: line_no,
                message: format!("{opcode:?} requires a=, b=, and c="),
            })?;
            let c_str = raw_c.ok_or_else(|| RlsError::LineParse {
                line: line_no,
                message: format!("{opcode:?} requires a=, b=, and c="),
            })?;
            let b_idx = resolve_sym_or_u32(line_no, b_str, active_symbols)?;
            let c_idx = resolve_sym_or_u32(line_no, c_str, active_symbols)?;
            let b16 = u16::try_from(b_idx).map_err(|_e| {
                RlsError::Symbol(TranspileError::FmaOperandIndexOverflow {
                    symbol: b_str.to_owned(),
                    node_idx: b_idx,
                })
            })?;
            let c16 = u16::try_from(c_idx).map_err(|_e| {
                RlsError::Symbol(TranspileError::FmaOperandIndexOverflow {
                    symbol: c_str.to_owned(),
                    node_idx: c_idx,
                })
            })?;
            NodeFrame::encode_u16_pair(b16, c16)
        } else if matches!(opcode, Opcode::Call | Opcode::CallIndirect) {
            let arg0 = match raw_b {
                None | Some("none") => CALL_ARG_SENTINEL,
                Some(b_str) => {
                    let b_idx = resolve_sym_or_u32(line_no, b_str, active_symbols)?;
                    u16::try_from(b_idx).map_err(|_e| {
                        RlsError::Symbol(TranspileError::FmaOperandIndexOverflow {
                            symbol: b_str.to_owned(),
                            node_idx: b_idx,
                        })
                    })?
                }
            };
            let arg1 = match raw_c {
                None | Some("none") => CALL_ARG_SENTINEL,
                Some(c_str) => {
                    let c_idx = resolve_sym_or_u32(line_no, c_str, active_symbols)?;
                    u16::try_from(c_idx).map_err(|_e| {
                        RlsError::Symbol(TranspileError::FmaOperandIndexOverflow {
                            symbol: c_str.to_owned(),
                            node_idx: c_idx,
                        })
                    })?
                }
            };
            NodeFrame::encode_u16_pair(arg0, arg1)
        } else if opcode == Opcode::CallArg {
            match raw_b {
                None | Some("none") => u32::from(CALL_ARG_SENTINEL),
                Some(b_str) => resolve_sym_or_u32(line_no, b_str, active_symbols)?,
            }
        } else if let Some(b_str) = raw_b {
            if matches!(
                opcode,
                Opcode::LocalGet
                    | Opcode::ConstDataLen
                    | Opcode::Block
                    | Opcode::Loop
                    | Opcode::If
                    | Opcode::Else
                    | Opcode::End
                    | Opcode::Br
                    | Opcode::BrIf
                    | Opcode::BrTable
                    | Opcode::RegionLen
                    | Opcode::RegionFree
                    | Opcode::RegionLease
                    | Opcode::RegionEndLease
                    | Opcode::RegionMove
                    | Opcode::RegionDrop
                    | Opcode::Func
                    | Opcode::Return
            ) {
                b_str.parse::<u32>().map_err(|_e| RlsError::LineParse {
                    line: line_no,
                    message: format!("expected numeric operand for b=, got '{b_str}'"),
                })?
            } else {
                resolve_sym_or_u32(line_no, b_str, active_symbols)?
            }
        } else if matches!(opcode, Opcode::Else | Opcode::End) && raw_a.is_some_and(|s| s != "0") {
            1
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

    let idx = u32::try_from(nodes.len()).unwrap_or(0);
    if opcode == Opcode::Func {
        if *ctrl_depth == 0 {
            if global_symbols.contains_key(label) {
                return Err(RlsError::Symbol(TranspileError::DuplicateSymbol(
                    label.to_owned(),
                )));
            }
            global_symbols.insert(label.to_owned(), idx);
            func_symbols.clear();
            *current_func_start = Some(idx);
            *ctrl_depth = 1;
        } else {
            *ctrl_depth = ctrl_depth.saturating_add(1);
            if func_symbols.contains_key(label) {
                return Err(RlsError::Symbol(TranspileError::DuplicateSymbol(
                    label.to_owned(),
                )));
            }
            let rel = idx.saturating_sub(current_func_start.unwrap_or(0));
            func_symbols.insert(label.to_owned(), rel);
        }
    } else if let Some(fn_start) = *current_func_start {
        if func_symbols.contains_key(label) {
            return Err(RlsError::Symbol(TranspileError::DuplicateSymbol(
                label.to_owned(),
            )));
        }
        let rel = idx.saturating_sub(fn_start);
        func_symbols.insert(label.to_owned(), rel);
        if matches!(opcode, Opcode::Block | Opcode::Loop | Opcode::If) {
            *ctrl_depth = ctrl_depth.saturating_add(1);
        } else if opcode == Opcode::End {
            *ctrl_depth = ctrl_depth.saturating_sub(1);
            if *ctrl_depth == 0 {
                *current_func_start = None;
                func_symbols.clear();
            }
        }
    } else {
        if global_symbols.contains_key(label) {
            return Err(RlsError::Symbol(TranspileError::DuplicateSymbol(
                label.to_owned(),
            )));
        }
        global_symbols.insert(label.to_owned(), idx);
    }
    nodes.push(frame);
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

fn parse_i64_auto(raw: &str) -> Option<i64> {
    if let Some(neg_hex) = raw.strip_prefix("-0x").or_else(|| raw.strip_prefix("-0X")) {
        let mag = u64::from_str_radix(neg_hex, 16).ok()?;
        if mag == (1_u64 << 63) {
            Some(i64::MIN)
        } else {
            i64::try_from(mag).ok()?.checked_neg()
        }
    } else if let Some(hex) = raw.strip_prefix("0x").or_else(|| raw.strip_prefix("0X")) {
        let bits = u64::from_str_radix(hex, 16).ok()?;
        #[expect(
            clippy::cast_possible_wrap,
            reason = "hex literal in const_i64 encodes exact 64-bit two's-complement bit pattern"
        )]
        Some(bits as i64)
    } else {
        raw.parse::<i64>().ok()
    }
}

fn parse_cap_mask(raw: &str) -> Option<u64> {
    if let Some(num) = parse_u64_auto(raw) {
        return Some(num);
    }
    let mut mask: u64 = 0;
    for part in raw.split(['|', '+', ',']) {
        let tok = part.trim();
        let bit = match tok {
            "none" | "NONE" | "CAP_NONE" => CAP_NONE,
            "net_ring" | "NET_RING" | "CAP_NET_RING" => CAP_NET_RING,
            "storage_dma" | "STORAGE_DMA" | "CAP_STORAGE_DMA" => CAP_STORAGE_DMA,
            "gpu_dispatch" | "GPU_DISPATCH" | "CAP_GPU_DISPATCH" => CAP_GPU_DISPATCH,
            "diverge" | "DIVERGE" | "CAP_DIVERGE" => CAP_DIVERGE,
            other => parse_u64_auto(other)?,
        };
        mask |= bit;
    }
    Some(mask)
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
        (ElemType::U8, SimdWidth::Scalar) => "u8x1",
        (ElemType::U8, SimdWidth::Lane4) => "u8x4",
        (ElemType::U8, SimdWidth::Lane8) => "u8x8",
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
        "u8x1" | "u8" => TypeLane::U8_SCALAR,
        "u8x4" => TypeLane {
            elem: ElemType::U8,
            width: SimdWidth::Lane4,
        },
        "u8x8" => TypeLane {
            elem: ElemType::U8,
            width: SimdWidth::Lane8,
        },
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
        Opcode::ConstI64 => "ConstI64",
        Opcode::LocalGet => "LocalGet",
        Opcode::LocalSet => "LocalSet",
        Opcode::ConstDataLen => "ConstDataLen",
        Opcode::ConstDataLoad => "ConstDataLoad",
        Opcode::Add => "Add",
        Opcode::Sub => "Sub",
        Opcode::Mul => "Mul",
        Opcode::Div => "Div",
        Opcode::CmpGt => "CmpGt",
        Opcode::SelectZero => "SelectZero",
        Opcode::Min => "Min",
        Opcode::Max => "Max",
        Opcode::CmpLt => "CmpLt",
        Opcode::CmpEq => "CmpEq",
        Opcode::Abs => "Abs",
        Opcode::Sqrt => "Sqrt",
        Opcode::Floor => "Floor",
        Opcode::Ceil => "Ceil",
        Opcode::Ln => "Ln",
        Opcode::Exp => "Exp",
        Opcode::FilterCompact => "FilterCompact",
        Opcode::ReduceSum => "ReduceSum",
        Opcode::ReduceMax => "ReduceMax",
        Opcode::Gather => "Gather",
        Opcode::ReduceMin => "ReduceMin",
        Opcode::MaskAnd => "MaskAnd",
        Opcode::MaskOr => "MaskOr",
        Opcode::Fma => "Fma",
        Opcode::MaskNot => "MaskNot",
        Opcode::Cast => "Cast",
        Opcode::BitAnd => "BitAnd",
        Opcode::BitOr => "BitOr",
        Opcode::BitXor => "BitXor",
        Opcode::Shl => "Shl",
        Opcode::ShrU => "ShrU",
        Opcode::Rem => "Rem",
        Opcode::RingEnqueue => "RingEnqueue",
        Opcode::ShrS => "ShrS",
        Opcode::CmpLtU => "CmpLtU",
        Opcode::AssertContract => "AssertContract",
        Opcode::Block => "Block",
        Opcode::Loop => "Loop",
        Opcode::If => "If",
        Opcode::Else => "Else",
        Opcode::End => "End",
        Opcode::Br => "Br",
        Opcode::BrIf => "BrIf",
        Opcode::RegionNew => "RegionNew",
        Opcode::RegionGrow => "RegionGrow",
        Opcode::RegionLen => "RegionLen",
        Opcode::RegionLoad => "RegionLoad",
        Opcode::RegionStore => "RegionStore",
        Opcode::RegionFree => "RegionFree",
        Opcode::RegionLease => "RegionLease",
        Opcode::RegionEndLease => "RegionEndLease",
        Opcode::RegionMove => "RegionMove",
        Opcode::RegionDrop => "RegionDrop",
        Opcode::Func => "Func",
        Opcode::Return => "Return",
        Opcode::Call => "Call",
        Opcode::CallArg => "CallArg",
        Opcode::BrTable => "BrTable",
        Opcode::CallIndirect => "CallIndirect",
    }
}

fn format_elem_short(elem: ElemType) -> &'static str {
    match elem {
        ElemType::F64 => "f64",
        ElemType::I64 => "i64",
        ElemType::BoolMask => "bool",
        ElemType::U8 => "u8",
    }
}

fn parse_elem_short(line_no: usize, raw: &str) -> Result<ElemType, RlsError> {
    match raw.trim() {
        "f64" | "F64" => Ok(ElemType::F64),
        "i64" | "I64" => Ok(ElemType::I64),
        "bool" | "BoolMask" | "mask" => Ok(ElemType::BoolMask),
        "u8" | "U8" => Ok(ElemType::U8),
        other => Err(RlsError::LineParse {
            line: line_no,
            message: format!("unknown element type '{other}'"),
        }),
    }
}

fn parse_opcode(line_no: usize, raw: &str) -> Result<Opcode, RlsError> {
    let op = match raw {
        "ArenaInput" => Opcode::ArenaInput,
        "ConstF64" => Opcode::ConstF64,
        "ArenaSlice" => Opcode::ArenaSlice,
        "ArenaMutate" => Opcode::ArenaMutate,
        "ConstI64" => Opcode::ConstI64,
        "LocalGet" => Opcode::LocalGet,
        "LocalSet" => Opcode::LocalSet,
        "ConstDataLen" => Opcode::ConstDataLen,
        "ConstDataLoad" => Opcode::ConstDataLoad,
        "Add" => Opcode::Add,
        "Sub" => Opcode::Sub,
        "Mul" => Opcode::Mul,
        "Div" => Opcode::Div,
        "CmpGt" => Opcode::CmpGt,
        "SelectZero" => Opcode::SelectZero,
        "Min" => Opcode::Min,
        "Max" => Opcode::Max,
        "CmpLt" => Opcode::CmpLt,
        "CmpEq" => Opcode::CmpEq,
        "Abs" => Opcode::Abs,
        "Sqrt" => Opcode::Sqrt,
        "Floor" => Opcode::Floor,
        "Ceil" => Opcode::Ceil,
        "Ln" => Opcode::Ln,
        "Exp" => Opcode::Exp,
        "FilterCompact" => Opcode::FilterCompact,
        "ReduceSum" => Opcode::ReduceSum,
        "ReduceMax" => Opcode::ReduceMax,
        "Gather" => Opcode::Gather,
        "ReduceMin" => Opcode::ReduceMin,
        "MaskAnd" => Opcode::MaskAnd,
        "MaskOr" => Opcode::MaskOr,
        "Fma" => Opcode::Fma,
        "MaskNot" => Opcode::MaskNot,
        "Cast" => Opcode::Cast,
        "BitAnd" => Opcode::BitAnd,
        "BitOr" => Opcode::BitOr,
        "BitXor" => Opcode::BitXor,
        "Shl" => Opcode::Shl,
        "ShrU" => Opcode::ShrU,
        "Rem" => Opcode::Rem,
        "RingEnqueue" => Opcode::RingEnqueue,
        "ShrS" => Opcode::ShrS,
        "CmpLtU" => Opcode::CmpLtU,
        "AssertContract" => Opcode::AssertContract,
        "Block" => Opcode::Block,
        "Loop" => Opcode::Loop,
        "If" => Opcode::If,
        "Else" => Opcode::Else,
        "End" => Opcode::End,
        "Br" => Opcode::Br,
        "BrIf" => Opcode::BrIf,
        "RegionNew" => Opcode::RegionNew,
        "RegionGrow" => Opcode::RegionGrow,
        "RegionLen" => Opcode::RegionLen,
        "RegionLoad" => Opcode::RegionLoad,
        "RegionStore" => Opcode::RegionStore,
        "RegionFree" => Opcode::RegionFree,
        "RegionLease" => Opcode::RegionLease,
        "RegionEndLease" => Opcode::RegionEndLease,
        "RegionMove" => Opcode::RegionMove,
        "RegionDrop" => Opcode::RegionDrop,
        "Func" => Opcode::Func,
        "Return" => Opcode::Return,
        "Call" => Opcode::Call,
        "CallArg" => Opcode::CallArg,
        "BrTable" => Opcode::BrTable,
        "CallIndirect" => Opcode::CallIndirect,
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

fn format_node_ref(
    current_func_start: Option<u32>,
    total_nodes: usize,
    opcode: Opcode,
    is_a: bool,
    idx: u32,
) -> String {
    if !is_a
        && matches!(
            opcode,
            Opcode::ArenaInput
                | Opcode::ArenaSlice
                | Opcode::Abs
                | Opcode::Sqrt
                | Opcode::Floor
                | Opcode::Ceil
                | Opcode::Ln
                | Opcode::Exp
                | Opcode::ReduceSum
                | Opcode::ReduceMax
                | Opcode::ReduceMin
                | Opcode::MaskNot
                | Opcode::Cast
                | Opcode::RingEnqueue
                | Opcode::AssertContract
        )
        && idx == 0
    {
        "0".to_owned()
    } else if let Some(fn_start) = current_func_start {
        if idx == 0 {
            "0".to_owned()
        } else if let Some(abs) = fn_start.checked_add(idx) {
            if (abs as usize) < total_nodes {
                format!("$n{abs}")
            } else {
                idx.to_string()
            }
        } else {
            idx.to_string()
        }
    } else if (idx as usize) < total_nodes {
        format!("$n{idx}")
    } else {
        idx.to_string()
    }
}

fn strip_rls_comment(raw_line: &str) -> &str {
    let bytes = raw_line.as_bytes();
    let mut in_quotes = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_quotes {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_quotes = false;
            }
        } else if b == b'"' {
            in_quotes = true;
        } else if b == b'#' {
            return raw_line.get(..i).unwrap_or(raw_line);
        }
    }
    raw_line
}

fn const_data_is_contiguous_partition(program: &RulProgram) -> bool {
    let mut cursor: usize = 0;
    for proof in &program.proofs {
        match proof.rule_kind {
            ProofRuleKind::ConstDataSeg => {
                if proof.witness_node != 0 {
                    return false;
                }
                let lo = proof.bound_lo as usize;
                let len = proof.bound_hi as usize;
                if lo != cursor {
                    return false;
                }
                let Some(next) = cursor.checked_add(len) else {
                    return false;
                };
                if next > program.const_data.len() {
                    return false;
                }
                cursor = next;
            }
            ProofRuleKind::BrTableSpec => {
                let lo = proof.bound_lo as usize;
                let count = proof.bound_hi as usize;
                if count == 0 || lo != cursor {
                    return false;
                }
                let Some(byte_len) = count.checked_mul(4) else {
                    return false;
                };
                let Some(next) = cursor.checked_add(byte_len) else {
                    return false;
                };
                if next > program.const_data.len() {
                    return false;
                }
                cursor = next;
            }
            ProofRuleKind::CallTable => {
                if proof.witness_node != 0 {
                    return false;
                }
                let lo = proof.bound_lo as usize;
                let count = proof.bound_hi as usize;
                if count == 0 || lo != cursor {
                    return false;
                }
                let Some(byte_len) = count.checked_mul(4) else {
                    return false;
                };
                let Some(next) = cursor.checked_add(byte_len) else {
                    return false;
                };
                if next > program.const_data.len() {
                    return false;
                }
                cursor = next;
            }
            _ => {}
        }
    }
    cursor == program.const_data.len()
}

fn is_printable_rls_string(bytes: &[u8]) -> bool {
    bytes
        .iter()
        .all(|&b| matches!(b, 0x20..=0x7E | b'\n' | b'\r' | b'\t'))
}

fn escape_rls_string(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    for &b in bytes {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'"' => out.push_str("\\\""),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            0x20..=0x7E => out.push(char::from(b)),
            other => {
                let _ = write!(out, "\\x{other:02x}");
            }
        }
    }
    out
}

fn encode_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().saturating_mul(2));
    for &b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn decode_hex(line_no: usize, raw: &str) -> Result<Vec<u8>, RlsError> {
    let mut nibbles: Vec<u8> = Vec::with_capacity(raw.len());
    for b in raw.bytes() {
        if b.is_ascii_whitespace() || b == b'_' {
            continue;
        }
        let n = hex_nibble(b).ok_or_else(|| RlsError::LineParse {
            line: line_no,
            message: format!("invalid hex byte in '{raw}'"),
        })?;
        nibbles.push(n);
    }
    if (nibbles.len() & 1) != 0 {
        return Err(RlsError::LineParse {
            line: line_no,
            message: "hex literal must have an even number of hex digits".to_owned(),
        });
    }
    let mut out = Vec::with_capacity(nibbles.len() / 2);
    for &[hi, lo] in nibbles.as_chunks::<2>().0 {
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

fn parse_quoted_rls_string(line_no: usize, raw: &str) -> Result<Vec<u8>, RlsError> {
    let trimmed = raw.trim();
    let Some(inner) = trimmed.strip_prefix('"').and_then(|s| s.strip_suffix('"')) else {
        return Err(RlsError::LineParse {
            line: line_no,
            message: format!("expected double-quoted string literal, got '{trimmed}'"),
        });
    };
    let bytes = inner.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        let Some(&b) = bytes.get(i) else {
            break;
        };
        if b == b'"' {
            return Err(RlsError::LineParse {
                line: line_no,
                message: "unescaped '\"' inside string literal".to_owned(),
            });
        }
        if b == b'\\' {
            let Some(&esc) = bytes.get(i + 1) else {
                return Err(RlsError::LineParse {
                    line: line_no,
                    message: "unterminated escape sequence in string literal".to_owned(),
                });
            };
            match esc {
                b'\\' => {
                    out.push(b'\\');
                    i += 2;
                }
                b'"' => {
                    out.push(b'"');
                    i += 2;
                }
                b'n' => {
                    out.push(b'\n');
                    i += 2;
                }
                b'r' => {
                    out.push(b'\r');
                    i += 2;
                }
                b't' => {
                    out.push(b'\t');
                    i += 2;
                }
                b'0' => {
                    out.push(0);
                    i += 2;
                }
                b'x' => {
                    let (Some(&h1), Some(&h2)) = (bytes.get(i + 2), bytes.get(i + 3)) else {
                        return Err(RlsError::LineParse {
                            line: line_no,
                            message: "truncated \\xHH escape in string literal".to_owned(),
                        });
                    };
                    let (Some(n1), Some(n2)) = (hex_nibble(h1), hex_nibble(h2)) else {
                        return Err(RlsError::LineParse {
                            line: line_no,
                            message: "invalid hex digit in \\xHH escape".to_owned(),
                        });
                    };
                    out.push((n1 << 4) | n2);
                    i += 4;
                }
                other => {
                    return Err(RlsError::LineParse {
                        line: line_no,
                        message: format!("unknown string escape '\\{}'", char::from(other)),
                    });
                }
            }
        } else {
            out.push(b);
            i += 1;
        }
    }
    Ok(out)
}

fn split_first_token(s: &str) -> Option<(&str, &str)> {
    let trimmed = s.trim_start();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(pos) = trimmed.find(char::is_whitespace) {
        Some((
            trimmed.get(..pos).unwrap_or(""),
            trimmed.get(pos..).unwrap_or("").trim_start(),
        ))
    } else {
        Some((trimmed, ""))
    }
}

fn parse_data_seg_line(
    line_no: usize,
    line: &str,
    const_data: &mut Vec<u8>,
    seg_symbols: &mut HashMap<String, u16>,
    next_seg_id: &mut u16,
) -> Result<ProofEntry, RlsError> {
    let (_, rest) = split_first_token(line).ok_or_else(|| RlsError::LineParse {
        line: line_no,
        message: "malformed data directive".to_owned(),
    })?;
    let (kw, rest) = split_first_token(rest).ok_or_else(|| RlsError::LineParse {
        line: line_no,
        message: "expected 'seg' after 'data'".to_owned(),
    })?;
    if kw != "seg" {
        return Err(RlsError::LineParse {
            line: line_no,
            message: format!("expected 'data seg <id> ...', got 'data {kw}'"),
        });
    }
    let (id_tok, rest) = split_first_token(rest).ok_or_else(|| RlsError::LineParse {
        line: line_no,
        message: "missing segment id after 'data seg'".to_owned(),
    })?;
    let seg_id = if let Ok(num) = id_tok.parse::<u16>() {
        *next_seg_id = (*next_seg_id).max(num.saturating_add(1));
        num
    } else {
        if seg_symbols.contains_key(id_tok) {
            return Err(RlsError::Symbol(TranspileError::DuplicateSymbol(
                id_tok.to_owned(),
            )));
        }
        let assigned = *next_seg_id;
        *next_seg_id = assigned.saturating_add(1);
        seg_symbols.insert(id_tok.to_owned(), assigned);
        assigned
    };

    let bytes = if let Some(after_hex) = rest.strip_prefix("hex") {
        if after_hex.is_empty()
            || !after_hex
                .bytes()
                .next()
                .is_some_and(|b| b.is_ascii_whitespace())
        {
            return Err(RlsError::LineParse {
                line: line_no,
                message: "expected whitespace and quoted hex string after 'hex'".to_owned(),
            });
        }
        let trimmed = after_hex.trim();
        let Some(inner) = trimmed.strip_prefix('"').and_then(|s| s.strip_suffix('"')) else {
            return Err(RlsError::LineParse {
                line: line_no,
                message: format!("expected quoted hex string, got '{trimmed}'"),
            });
        };
        decode_hex(line_no, inner)?
    } else {
        parse_quoted_rls_string(line_no, rest)?
    };

    let offset = u32::try_from(const_data.len()).map_err(|_e| RlsError::LineParse {
        line: line_no,
        message: "const_data section offset exceeds u32::MAX".to_owned(),
    })?;
    let len = u32::try_from(bytes.len()).map_err(|_e| RlsError::LineParse {
        line: line_no,
        message: "const_data segment length exceeds u32::MAX".to_owned(),
    })?;
    const_data.extend_from_slice(&bytes);
    Ok(ProofEntry::const_data_seg(seg_id, offset, len))
}

fn parse_hex_section_line(
    line_no: usize,
    line: &str,
    directive: &str,
) -> Result<Vec<u8>, RlsError> {
    let (_, rest) = split_first_token(line).ok_or_else(|| RlsError::LineParse {
        line: line_no,
        message: format!("malformed {directive} directive"),
    })?;
    let (kw, rest) = split_first_token(rest).ok_or_else(|| RlsError::LineParse {
        line: line_no,
        message: format!("expected 'hex' after '{directive}'"),
    })?;
    if kw != "hex" {
        return Err(RlsError::LineParse {
            line: line_no,
            message: format!("expected '{directive} hex \"...\"', got '{kw}'"),
        });
    }
    let trimmed = rest.trim();
    let Some(inner) = trimmed.strip_prefix('"').and_then(|s| s.strip_suffix('"')) else {
        return Err(RlsError::LineParse {
            line: line_no,
            message: format!("expected quoted hex string in {directive}"),
        });
    };
    decode_hex(line_no, inner)
}

fn parse_br_table_line<'a>(
    line_no: usize,
    mut parts: impl Iterator<Item = &'a str>,
    const_data: &mut Vec<u8>,
) -> Result<ProofEntry, RlsError> {
    let id_tok = parts.next().ok_or_else(|| RlsError::LineParse {
        line: line_no,
        message: "missing br_table id".to_owned(),
    })?;
    let br_table_id = id_tok.parse::<u32>().map_err(|_e| RlsError::LineParse {
        line: line_no,
        message: format!("invalid br_table id '{id_tok}'"),
    })?;
    let mut default_depth: Option<u16> = None;
    let mut targets_raw: Option<&str> = None;
    for kv in parts {
        let Some((k, v)) = kv.split_once('=') else {
            return Err(RlsError::LineParse {
                line: line_no,
                message: format!("expected key=value in br_table directive, got '{kv}'"),
            });
        };
        match k {
            "default" => {
                default_depth = Some(v.parse::<u16>().map_err(|_e| RlsError::LineParse {
                    line: line_no,
                    message: format!("invalid br_table default depth '{v}'"),
                })?);
            }
            "targets" => {
                targets_raw = Some(v);
            }
            _ => {
                return Err(RlsError::LineParse {
                    line: line_no,
                    message: format!("unknown br_table attribute '{k}'"),
                });
            }
        }
    }
    let def = default_depth.ok_or_else(|| RlsError::LineParse {
        line: line_no,
        message: "br_table requires default=<u16>".to_owned(),
    })?;
    let targets_str = targets_raw.ok_or_else(|| RlsError::LineParse {
        line: line_no,
        message: "br_table requires targets=<d0,d1,...>".to_owned(),
    })?;
    let offset = u32::try_from(const_data.len()).map_err(|_e| RlsError::LineParse {
        line: line_no,
        message: "const_data section offset exceeds u32::MAX".to_owned(),
    })?;
    let mut count: u32 = 0;
    for tok in targets_str.split(',') {
        let d = tok
            .trim()
            .parse::<u32>()
            .map_err(|_e| RlsError::LineParse {
                line: line_no,
                message: format!("invalid br_table target depth '{tok}'"),
            })?;
        const_data.extend_from_slice(&d.to_le_bytes());
        count = count.saturating_add(1);
    }
    Ok(ProofEntry::br_table_spec(br_table_id, def, offset, count))
}

fn parse_call_table_line<'a>(
    line_no: usize,
    mut parts: impl Iterator<Item = &'a str>,
    const_data: &mut Vec<u8>,
) -> Result<ProofEntry, RlsError> {
    let id_tok = parts.next().ok_or_else(|| RlsError::LineParse {
        line: line_no,
        message: "missing call_table id".to_owned(),
    })?;
    let table_id = id_tok.parse::<u16>().map_err(|_e| RlsError::LineParse {
        line: line_no,
        message: format!("invalid call_table id '{id_tok}'"),
    })?;
    let mut funcs_raw: Option<&str> = None;
    for kv in parts {
        let Some((k, v)) = kv.split_once('=') else {
            return Err(RlsError::LineParse {
                line: line_no,
                message: format!("expected key=value in call_table directive, got '{kv}'"),
            });
        };
        if k == "funcs" {
            funcs_raw = Some(v);
        } else {
            return Err(RlsError::LineParse {
                line: line_no,
                message: format!("unknown call_table attribute '{k}'"),
            });
        }
    }
    let funcs_str = funcs_raw.ok_or_else(|| RlsError::LineParse {
        line: line_no,
        message: "call_table requires funcs=<f0,f1,...>".to_owned(),
    })?;
    let offset = u32::try_from(const_data.len()).map_err(|_e| RlsError::LineParse {
        line: line_no,
        message: "const_data section offset exceeds u32::MAX".to_owned(),
    })?;
    let mut count: u32 = 0;
    for tok in funcs_str.split(',') {
        let fid = tok
            .trim()
            .parse::<u32>()
            .map_err(|_e| RlsError::LineParse {
                line: line_no,
                message: format!("invalid call_table func_id '{tok}'"),
            })?;
        const_data.extend_from_slice(&fid.to_le_bytes());
        count = count.saturating_add(1);
    }
    Ok(ProofEntry::call_table(table_id, offset, count))
}
