//! Bridge transpiler, symbolic emitter, and human-auditable disassembler for `.rl` binaries.
//!
//! While the Rul Core Engine operates exclusively on flat 128-bit `.rl` binaries,
//! current text-trained LLMs benefit from semantic attention anchors (`"$prices"`, `"$mask"`)
//! during emission. [`SymbolicGraphBuilder`] lets an AI Agent emit symbolic node names that
//! are stripped into pure `u32` flat indices when packed into `.rl`.

use std::collections::HashMap;
use std::fmt::{self, Write as _};

use crate::bytecode::{
    ArenaDescriptor, NodeFrame, Opcode, ProofEntry, RulProgram, ScheduleHint, TokenOp, TypeLane,
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
