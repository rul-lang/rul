//! Binary format definitions for Rul (`.rl`) programs.
//!
//! Every Rul instruction is a fixed-size 128-bit (16-byte) [`NodeFrame`] carrying:
//! `Algorithm (Opcode + TypeLane) + Schedule (ScheduleHint) + Proof (ProofIdx + TokenOp)`.

use std::fmt;

/// Magic header bytes identifying a Rul v0.1 `.rl` binary image.
pub const RL_MAGIC: [u8; 8] = *b"RUL\x00\x01\x00\x00\x00";

/// Size in bytes of the fixed `RulHeader`.
pub const RL_HEADER_SIZE: usize = 64;

/// Size in bytes of each [`ArenaDescriptor`], [`ProofEntry`], and [`NodeFrame`].
pub const RL_FRAME_SIZE: usize = 16;

/// Maximum number of descriptors or nodes permitted in a bounded `.rl` load.
pub const MAX_SECTION_ITEMS: u32 = 1_000_000;

/// Capability bit: Pure compute only (no external I/O allowed).
pub const CAP_NONE: u64 = 0x00;
/// Capability bit: Lock-free network ring enqueue/dequeue allowed.
pub const CAP_NET_RING: u64 = 0x01;
/// Capability bit: Direct NVMe/storage DMA allowed.
pub const CAP_STORAGE_DMA: u64 = 0x02;
/// Capability bit: Heterogeneous GPU/NPU queue dispatch allowed.
pub const CAP_GPU_DISPATCH: u64 = 0x04;

/// Core opcodes of the Rul v0.1 binary instruction set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Opcode {
    /// Load a columnar input arena (`in_a = arena_id`).
    ArenaInput = 0x01,
    /// Load a 64-bit IEEE-754 floating-point scalar constant (`in_a = low32`, `in_b = high32`).
    ConstF64 = 0x02,
    /// Zero-copy sub-slice of an arena (`in_a = source_node`, verified by `proof_idx`).
    ArenaSlice = 0x03,
    /// In-place linear mutation of a mutable arena (`in_a = target_arena`, `in_b = value_node`).
    ArenaMutate = 0x04,
    /// Branchless elementwise addition (`out[i] = a[i] + b[i]`).
    Add = 0x10,
    /// Branchless elementwise subtraction (`out[i] = a[i] - b[i]`).
    Sub = 0x11,
    /// Branchless elementwise multiplication (`out[i] = a[i] * b[i]`).
    Mul = 0x12,
    /// Branchless elementwise greater-than comparison (`mask[i] = a[i] > b[i]`).
    CmpGt = 0x14,
    /// Branchless predicated selection (`out[i] = mask[i] ? val[i] : 0.0`).
    SelectZero = 0x15,
    /// Stream compaction filtering elements of `in_b` where mask `in_a` is true.
    FilterCompact = 0x20,
    /// Horizontal sum reduction of column `in_a` to a scalar `f64`.
    ReduceSum = 0x21,
    /// Horizontal maximum reduction of column `in_a` to a scalar `f64`.
    ReduceMax = 0x22,
    /// Indirect indexed gather (`out[i] = arena_node[index_node[i]]`, requires `proof_idx > 0`).
    Gather = 0x23,
    /// Enqueue scalar/column output to a hardware NIC ring (requires [`CAP_NET_RING`]).
    RingEnqueue = 0x30,
    /// Output contract assertion checking non-NaN and finite bounds on `in_a`.
    AssertContract = 0x40,
}

impl Opcode {
    /// Decodes a raw byte into an [`Opcode`].
    ///
    /// # Errors
    /// Returns [`RulFormatError::UnknownOpcode`] if the byte is not a valid v0.1 opcode.
    pub const fn from_u8(byte: u8) -> Result<Self, RulFormatError> {
        match byte {
            0x01 => Ok(Self::ArenaInput),
            0x02 => Ok(Self::ConstF64),
            0x03 => Ok(Self::ArenaSlice),
            0x04 => Ok(Self::ArenaMutate),
            0x10 => Ok(Self::Add),
            0x11 => Ok(Self::Sub),
            0x12 => Ok(Self::Mul),
            0x14 => Ok(Self::CmpGt),
            0x15 => Ok(Self::SelectZero),
            0x20 => Ok(Self::FilterCompact),
            0x21 => Ok(Self::ReduceSum),
            0x22 => Ok(Self::ReduceMax),
            0x23 => Ok(Self::Gather),
            0x30 => Ok(Self::RingEnqueue),
            0x40 => Ok(Self::AssertContract),
            other => Err(RulFormatError::UnknownOpcode(other)),
        }
    }
}

/// Primitive element type encoded in the high nibble of [`TypeLane`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ElemType {
    /// 64-bit IEEE-754 float (`f64`).
    F64 = 0x1,
    /// 64-bit signed integer (`i64` / index).
    I64 = 0x2,
    /// Predication boolean mask (`bool` / lane mask).
    BoolMask = 0x3,
}

/// SIMD vector lane width encoded in the low nibble of [`TypeLane`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SimdWidth {
    /// Scalar (1 lane).
    Scalar = 0x1,
    /// 4 lanes (256-bit AVX2 for `f64`).
    Lane4 = 0x4,
    /// 8 lanes (512-bit AVX-512 for `f64`).
    Lane8 = 0x8,
}

/// Packed 8-bit type and SIMD lane descriptor (`[ElemType: 4b | SimdWidth: 4b]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeLane {
    /// Element primitive type.
    pub elem: ElemType,
    /// Vector lane width.
    pub width: SimdWidth,
}

impl TypeLane {
    /// Standard 8-lane `f64` columnar vector type (`f64x8`).
    pub const F64X8: Self = Self {
        elem: ElemType::F64,
        width: SimdWidth::Lane8,
    };

    /// Standard scalar `f64` type (`f64x1`).
    pub const F64_SCALAR: Self = Self {
        elem: ElemType::F64,
        width: SimdWidth::Scalar,
    };

    /// Standard 8-lane boolean mask type (`maskx8`).
    pub const MASKX8: Self = Self {
        elem: ElemType::BoolMask,
        width: SimdWidth::Lane8,
    };

    /// Standard 8-lane `i64` index column type (`i64x8`).
    pub const I64X8: Self = Self {
        elem: ElemType::I64,
        width: SimdWidth::Lane8,
    };

    /// Packs the [`TypeLane`] into a single byte.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        ((self.elem as u8) << 4) | (self.width as u8)
    }

    /// Unpacks a [`TypeLane`] from a single byte.
    ///
    /// # Errors
    /// Returns [`RulFormatError::InvalidTypeLane`] if either nibble is unknown.
    pub const fn from_u8(byte: u8) -> Result<Self, RulFormatError> {
        let elem = match byte >> 4 {
            0x1 => ElemType::F64,
            0x2 => ElemType::I64,
            0x3 => ElemType::BoolMask,
            _ => return Err(RulFormatError::InvalidTypeLane(byte)),
        };
        let width = match byte & 0x0F {
            0x1 => SimdWidth::Scalar,
            0x4 => SimdWidth::Lane4,
            0x8 => SimdWidth::Lane8,
            _ => return Err(RulFormatError::InvalidTypeLane(byte)),
        };
        Ok(Self { elem, width })
    }
}

/// Hardware execution policy carried inside [`ScheduleHint`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ExecPolicy {
    /// Automatic L1 cache-tiled execution.
    AutoL1Tile = 0x0,
    /// Streaming SIMD execution (`AVX-512` / `NEON` vector lanes).
    SimdStream = 0x1,
    /// Multi-core parallel data partition.
    MultiCore = 0x2,
    /// Heterogeneous GPU DMA offload.
    GpuDma = 0x3,
}

/// Schedule-carrying byte (`[fusion_group: 4b | exec_policy: 4b]`).
///
/// The AI Agent specifies which nodes belong to the same kernel fusion group (`1..=15`)
/// so the compiler never has to run an expensive autoscheduler search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduleHint {
    /// Kernel fusion group identifier (`0` = unfused, `1..=15` = fused loop group).
    pub fusion_group: u8,
    /// Target hardware execution policy.
    pub policy: ExecPolicy,
}

impl ScheduleHint {
    /// Default unfused schedule.
    pub const UNFUSED: Self = Self {
        fusion_group: 0,
        policy: ExecPolicy::AutoL1Tile,
    };

    /// Creates a fused SIMD streaming schedule for the given group (`1..=15`).
    #[must_use]
    pub const fn fused_simd(group: u8) -> Self {
        Self {
            fusion_group: group & 0x0F,
            policy: ExecPolicy::SimdStream,
        }
    }

    /// Packs the schedule hint into 8 bits.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        ((self.fusion_group & 0x0F) << 4) | (self.policy as u8)
    }

    /// Unpacks an 8-bit schedule hint.
    ///
    /// # Errors
    /// Returns [`RulFormatError::InvalidScheduleHint`] if the policy nibble is invalid.
    pub const fn from_u8(byte: u8) -> Result<Self, RulFormatError> {
        let fusion_group = byte >> 4;
        let policy = match byte & 0x0F {
            0x0 => ExecPolicy::AutoL1Tile,
            0x1 => ExecPolicy::SimdStream,
            0x2 => ExecPolicy::MultiCore,
            0x3 => ExecPolicy::GpuDma,
            _ => return Err(RulFormatError::InvalidScheduleHint(byte)),
        };
        Ok(Self {
            fusion_group,
            policy,
        })
    }
}

/// Ownership mode encoded in the top 2 bits of [`TokenOp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TokenMode {
    /// Pure value computation; does not touch arena ownership state.
    PureValue = 0b00,
    /// Shared read-only lease (`&Arena`) at a specific token version.
    ReadLease = 0b01,
    /// Linear write/consume (`&mut Arena`), invalidating the old version and advancing to `ver + 1`.
    AffineConsume = 0b10,
}

/// 32-bit explicit ownership & borrow certificate (`[mode: 2b | arena_id: 14b | version: 16b]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenOp {
    /// Ownership transition mode.
    pub mode: TokenMode,
    /// Target arena identifier (`0..16383`).
    pub arena_id: u16,
    /// Expected token version (`0..65535`).
    pub version: u16,
}

impl TokenOp {
    /// Pure value operation with no arena ownership side effects.
    pub const PURE: Self = Self {
        mode: TokenMode::PureValue,
        arena_id: 0,
        version: 0,
    };

    /// Creates a shared read lease (`&Arena`) for `arena_id` at `version`.
    #[must_use]
    pub const fn read_lease(arena_id: u16, version: u16) -> Self {
        Self {
            mode: TokenMode::ReadLease,
            arena_id: arena_id & 0x3FFF,
            version,
        }
    }

    /// Creates an affine write/consume token operation (`&mut Arena`) for `arena_id` at `version`.
    #[must_use]
    pub const fn affine_consume(arena_id: u16, version: u16) -> Self {
        Self {
            mode: TokenMode::AffineConsume,
            arena_id: arena_id & 0x3FFF,
            version,
        }
    }

    /// Packs the token operation into a 32-bit integer.
    #[must_use]
    pub const fn to_u32(self) -> u32 {
        ((self.mode as u32) << 30)
            | (((self.arena_id & 0x3FFF) as u32) << 16)
            | (self.version as u32)
    }

    /// Unpacks a 32-bit token operation.
    ///
    /// # Errors
    /// Returns [`RulFormatError::InvalidTokenMode`] if the top 2 bits are `0b11`.
    pub const fn from_u32(raw: u32) -> Result<Self, RulFormatError> {
        let mode = match raw >> 30 {
            0b00 => TokenMode::PureValue,
            0b01 => TokenMode::ReadLease,
            0b10 => TokenMode::AffineConsume,
            _ => return Err(RulFormatError::InvalidTokenMode(raw)),
        };
        let arena_id = ((raw >> 16) & 0x3FFF) as u16;
        let version = (raw & 0xFFFF) as u16;
        Ok(Self {
            mode,
            arena_id,
            version,
        })
    }
}

/// 16-byte columnar [`ArenaDescriptor`] in the `.rl` header section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaDescriptor {
    /// Arena index (`0..arena_count`).
    pub arena_id: u32,
    /// Number of elements in this SoA column.
    pub element_count: u32,
    /// Element type and vector width.
    pub type_lane: TypeLane,
    /// Whether this arena permits in-place affine mutation.
    pub mutable: bool,
    /// Byte alignment requirement (typically `64` for cache line / AVX-512).
    pub alignment: u16,
    /// Initial [`TokenOp`] version assigned when the program starts.
    pub initial_version: u16,
}

impl ArenaDescriptor {
    /// Encodes the descriptor into 16 bytes (little-endian).
    #[must_use]
    pub fn to_bytes(self) -> [u8; RL_FRAME_SIZE] {
        let mut out = [0u8; RL_FRAME_SIZE];
        out[0..4].copy_from_slice(&self.arena_id.to_le_bytes());
        out[4..8].copy_from_slice(&self.element_count.to_le_bytes());
        out[8] = self.type_lane.to_u8();
        out[9] = u8::from(self.mutable);
        out[10..12].copy_from_slice(&self.alignment.to_le_bytes());
        out[12..16].copy_from_slice(&u32::from(self.initial_version).to_le_bytes());
        out
    }

    /// Decodes an [`ArenaDescriptor`] from a 16-byte slice.
    ///
    /// # Errors
    /// Returns [`RulFormatError`] if the slice length or encoded fields are invalid.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, RulFormatError> {
        let chunk: &[u8; RL_FRAME_SIZE] = bytes
            .try_into()
            .map_err(|_err| RulFormatError::TruncatedSection("ArenaDescriptor"))?;
        let arena_id = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let element_count = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        let type_lane = TypeLane::from_u8(chunk[8])?;
        let mutable = chunk[9] != 0;
        let alignment = u16::from_le_bytes([chunk[10], chunk[11]]);
        let initial_version_u32 = u32::from_le_bytes([chunk[12], chunk[13], chunk[14], chunk[15]]);
        let initial_version = u16::try_from(initial_version_u32)
            .map_err(|_err| RulFormatError::InvalidTokenMode(initial_version_u32))?;
        Ok(Self {
            arena_id,
            element_count,
            type_lane,
            mutable,
            alignment,
            initial_version,
        })
    }
}

/// Kind of refinement proof stored in the [`ProofEntry`] sidecar table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum ProofRuleKind {
    /// Proves that a slice `[bound_lo..bound_hi]` lies within `target_arena.element_count`.
    SliceBounds = 1,
    /// Proves that all indices in an index arena are `< target_arena.element_count`.
    GatherIndexInBounds = 2,
}

impl ProofRuleKind {
    const fn from_u16(raw: u16) -> Result<Self, RulFormatError> {
        match raw {
            1 => Ok(Self::SliceBounds),
            2 => Ok(Self::GatherIndexInBounds),
            other => Err(RulFormatError::InvalidProofRule(other)),
        }
    }
}

/// 16-byte sidecar refinement proof certificate used for `ArenaSlice` and `Gather`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProofEntry {
    /// Proof rule kind.
    pub rule_kind: ProofRuleKind,
    /// Target arena whose bounds are being proven.
    pub target_arena: u16,
    /// Lower bound (`lo`).
    pub bound_lo: u32,
    /// Upper bound (`hi`, must be `<= target_arena.element_count`).
    pub bound_hi: u32,
    /// Optional witness node index.
    pub witness_node: u32,
}

impl ProofEntry {
    /// Encodes the proof entry into 16 bytes.
    #[must_use]
    pub fn to_bytes(self) -> [u8; RL_FRAME_SIZE] {
        let mut out = [0u8; RL_FRAME_SIZE];
        out[0..2].copy_from_slice(&(self.rule_kind as u16).to_le_bytes());
        out[2..4].copy_from_slice(&self.target_arena.to_le_bytes());
        out[4..8].copy_from_slice(&self.bound_lo.to_le_bytes());
        out[8..12].copy_from_slice(&self.bound_hi.to_le_bytes());
        out[12..16].copy_from_slice(&self.witness_node.to_le_bytes());
        out
    }

    /// Decodes a [`ProofEntry`] from a 16-byte slice.
    ///
    /// # Errors
    /// Returns [`RulFormatError`] if the rule kind or slice size is invalid.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, RulFormatError> {
        let chunk: &[u8; RL_FRAME_SIZE] = bytes
            .try_into()
            .map_err(|_err| RulFormatError::TruncatedSection("ProofEntry"))?;
        let rule_raw = u16::from_le_bytes([chunk[0], chunk[1]]);
        let rule_kind = ProofRuleKind::from_u16(rule_raw)?;
        let target_arena = u16::from_le_bytes([chunk[2], chunk[3]]);
        let bound_lo = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        let bound_hi = u32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
        let witness_node = u32::from_le_bytes([chunk[12], chunk[13], chunk[14], chunk[15]]);
        Ok(Self {
            rule_kind,
            target_arena,
            bound_lo,
            bound_hi,
            witness_node,
        })
    }
}

/// Fixed-size 128-bit (16-byte) Rul instruction frame: `Algorithm + Schedule + Proof`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C, align(16))]
pub struct NodeFrame {
    /// Operation code (`Algorithm`).
    pub opcode: Opcode,
    /// Element type and SIMD lane width (`Algorithm`).
    pub type_lane: TypeLane,
    /// Kernel fusion group and hardware policy (`Schedule`).
    pub schedule: ScheduleHint,
    /// Proof certificate index (`0 = TRIVIAL_BY_SHAPE`, `>0 = Sidecar ProofEntry`).
    pub proof_idx: u8,
    /// Explicit ownership/lease certificate (`Proof`).
    pub token_op: TokenOp,
    /// First input operand (node index, arena ID, or low 32 bits of constant).
    pub in_a: u32,
    /// Second input operand (node index or high 32 bits of constant).
    pub in_b: u32,
}

impl NodeFrame {
    /// Encodes a [`Opcode::ConstF64`] frame from a 64-bit float value.
    #[must_use]
    pub fn const_f64(value: f64, schedule: ScheduleHint) -> Self {
        let bits = value.to_bits();
        let low = (bits & 0xFFFF_FFFF) as u32;
        let high = (bits >> 32) as u32;
        Self {
            opcode: Opcode::ConstF64,
            type_lane: TypeLane::F64_SCALAR,
            schedule,
            proof_idx: 0,
            token_op: TokenOp::PURE,
            in_a: low,
            in_b: high,
        }
    }

    /// Extracts the `f64` constant stored in `in_a` and `in_b`.
    #[must_use]
    pub const fn decode_const_f64(self) -> f64 {
        let bits = (self.in_a as u64) | ((self.in_b as u64) << 32);
        f64::from_bits(bits)
    }

    /// Serializes the 128-bit frame into 16 little-endian bytes.
    #[must_use]
    pub fn to_bytes(self) -> [u8; RL_FRAME_SIZE] {
        let mut out = [0u8; RL_FRAME_SIZE];
        out[0] = self.opcode as u8;
        out[1] = self.type_lane.to_u8();
        out[2] = self.schedule.to_u8();
        out[3] = self.proof_idx;
        out[4..8].copy_from_slice(&self.token_op.to_u32().to_le_bytes());
        out[8..12].copy_from_slice(&self.in_a.to_le_bytes());
        out[12..16].copy_from_slice(&self.in_b.to_le_bytes());
        out
    }

    /// Deserializes a 128-bit frame from a 16-byte slice.
    ///
    /// # Errors
    /// Returns [`RulFormatError`] if any byte field is invalid.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, RulFormatError> {
        let chunk: &[u8; RL_FRAME_SIZE] = bytes
            .try_into()
            .map_err(|_err| RulFormatError::TruncatedSection("NodeFrame"))?;
        let opcode = Opcode::from_u8(chunk[0])?;
        let type_lane = TypeLane::from_u8(chunk[1])?;
        let schedule = ScheduleHint::from_u8(chunk[2])?;
        let proof_idx = chunk[3];
        let token_raw = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        let token_op = TokenOp::from_u32(token_raw)?;
        let in_a = u32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
        let in_b = u32::from_le_bytes([chunk[12], chunk[13], chunk[14], chunk[15]]);
        Ok(Self {
            opcode,
            type_lane,
            schedule,
            proof_idx,
            token_op,
            in_a,
            in_b,
        })
    }
}

/// Complete in-memory representation of a `.rl` binary program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RulProgram {
    /// Hardware capability bitmask declared in the `.rl` header.
    pub capability_mask: u64,
    /// Columnar memory arena descriptors.
    pub arenas: Vec<ArenaDescriptor>,
    /// Sidecar refinement proofs (`1-based` indexing via `proof_idx`).
    pub proofs: Vec<ProofEntry>,
    /// Topological array of 128-bit instruction frames.
    pub nodes: Vec<NodeFrame>,
    /// Primary output node index.
    pub output_node: u32,
}

impl RulProgram {
    /// Computes the deterministic 64-bit FNV-1a content hash of the program body.
    #[must_use]
    pub fn compute_graph_hash(&self) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        let prime: u64 = 0x0100_0000_01b3;
        for arena in &self.arenas {
            for b in arena.to_bytes() {
                hash ^= u64::from(b);
                hash = hash.wrapping_mul(prime);
            }
        }
        for proof in &self.proofs {
            for b in proof.to_bytes() {
                hash ^= u64::from(b);
                hash = hash.wrapping_mul(prime);
            }
        }
        for node in &self.nodes {
            for b in node.to_bytes() {
                hash ^= u64::from(b);
                hash = hash.wrapping_mul(prime);
            }
        }
        hash
    }

    /// Encodes the entire program into the canonical `.rl` flat binary image.
    #[must_use]
    pub fn to_rl_bytes(&self) -> Vec<u8> {
        let total_bytes = RL_HEADER_SIZE
            + (self.arenas.len() + self.proofs.len() + self.nodes.len()) * RL_FRAME_SIZE;
        let mut buf = Vec::with_capacity(total_bytes);

        let graph_hash = self.compute_graph_hash();
        let arena_count = u32::try_from(self.arenas.len()).unwrap_or(u32::MAX);
        let proof_count = u32::try_from(self.proofs.len()).unwrap_or(u32::MAX);
        let node_count = u32::try_from(self.nodes.len()).unwrap_or(u32::MAX);

        buf.extend_from_slice(&RL_MAGIC);
        buf.extend_from_slice(&graph_hash.to_le_bytes());
        buf.extend_from_slice(&self.capability_mask.to_le_bytes());
        buf.extend_from_slice(&arena_count.to_le_bytes());
        buf.extend_from_slice(&proof_count.to_le_bytes());
        buf.extend_from_slice(&node_count.to_le_bytes());
        buf.extend_from_slice(&self.output_node.to_le_bytes());
        buf.extend_from_slice(&[0u8; 24]);

        for arena in &self.arenas {
            buf.extend_from_slice(&arena.to_bytes());
        }
        for proof in &self.proofs {
            buf.extend_from_slice(&proof.to_bytes());
        }
        for node in &self.nodes {
            buf.extend_from_slice(&node.to_bytes());
        }
        buf
    }

    /// Decodes a `.rl` binary image from a byte slice with zero text parsing.
    ///
    /// # Errors
    /// Returns [`RulFormatError`] if magic bytes, hash integrity, or frame sizes are invalid.
    pub fn from_rl_bytes(bytes: &[u8]) -> Result<Self, RulFormatError> {
        let header: &[u8; RL_HEADER_SIZE] = bytes
            .get(0..RL_HEADER_SIZE)
            .and_then(|s| s.try_into().ok())
            .ok_or(RulFormatError::TruncatedSection("RulHeader"))?;

        if header[0..8] != RL_MAGIC {
            return Err(RulFormatError::BadMagic);
        }

        let expected_hash = u64::from_le_bytes([
            header[8], header[9], header[10], header[11], header[12], header[13], header[14],
            header[15],
        ]);
        let capability_mask = u64::from_le_bytes([
            header[16], header[17], header[18], header[19], header[20], header[21], header[22],
            header[23],
        ]);
        let arena_count = u32::from_le_bytes([header[24], header[25], header[26], header[27]]);
        let proof_count = u32::from_le_bytes([header[28], header[29], header[30], header[31]]);
        let node_count = u32::from_le_bytes([header[32], header[33], header[34], header[35]]);
        let output_node = u32::from_le_bytes([header[36], header[37], header[38], header[39]]);

        if arena_count > MAX_SECTION_ITEMS
            || proof_count > MAX_SECTION_ITEMS
            || node_count > MAX_SECTION_ITEMS
        {
            return Err(RulFormatError::SectionTooLarge);
        }

        let total_frames = (arena_count as usize)
            .saturating_add(proof_count as usize)
            .saturating_add(node_count as usize);
        let expected_len = RL_HEADER_SIZE.saturating_add(total_frames * RL_FRAME_SIZE);
        if bytes.len() != expected_len {
            return Err(RulFormatError::LengthMismatch {
                expected: expected_len,
                actual: bytes.len(),
            });
        }

        let mut offset = RL_HEADER_SIZE;
        let mut arenas = Vec::with_capacity(arena_count as usize);
        for _ in 0..arena_count {
            let slice = bytes
                .get(offset..offset + RL_FRAME_SIZE)
                .ok_or(RulFormatError::TruncatedSection("ArenaTable"))?;
            arenas.push(ArenaDescriptor::from_bytes(slice)?);
            offset += RL_FRAME_SIZE;
        }

        let mut proofs = Vec::with_capacity(proof_count as usize);
        for _ in 0..proof_count {
            let slice = bytes
                .get(offset..offset + RL_FRAME_SIZE)
                .ok_or(RulFormatError::TruncatedSection("ProofTable"))?;
            proofs.push(ProofEntry::from_bytes(slice)?);
            offset += RL_FRAME_SIZE;
        }

        let mut nodes = Vec::with_capacity(node_count as usize);
        for _ in 0..node_count {
            let slice = bytes
                .get(offset..offset + RL_FRAME_SIZE)
                .ok_or(RulFormatError::TruncatedSection("NodeTable"))?;
            nodes.push(NodeFrame::from_bytes(slice)?);
            offset += RL_FRAME_SIZE;
        }

        let program = Self {
            capability_mask,
            arenas,
            proofs,
            nodes,
            output_node,
        };
        let actual_hash = program.compute_graph_hash();
        if actual_hash != expected_hash {
            return Err(RulFormatError::HashMismatch {
                expected: expected_hash,
                actual: actual_hash,
            });
        }

        Ok(program)
    }
}

/// Errors that can occur when decoding a binary `.rl` image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RulFormatError {
    /// Header magic bytes do not match `RUL\x00\x01\x00\x00\x00`.
    BadMagic,
    /// Binary slice ended prematurely.
    TruncatedSection(&'static str),
    /// Declared section counts exceed bounded safety limits.
    SectionTooLarge,
    /// Total byte length does not match header section counts.
    LengthMismatch {
        /// Expected byte count.
        expected: usize,
        /// Actual byte count.
        actual: usize,
    },
    /// Content-addressed 64-bit graph hash mismatch.
    HashMismatch {
        /// Hash recorded in header.
        expected: u64,
        /// Hash computed from frames.
        actual: u64,
    },
    /// Unrecognized opcode byte.
    UnknownOpcode(u8),
    /// Unrecognized `TypeLane` encoding.
    InvalidTypeLane(u8),
    /// Unrecognized `ScheduleHint` encoding.
    InvalidScheduleHint(u8),
    /// Unrecognized `TokenMode` encoding.
    InvalidTokenMode(u32),
    /// Unrecognized `ProofRuleKind` encoding.
    InvalidProofRule(u16),
}

impl fmt::Display for RulFormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => write!(f, "invalid .rl magic bytes"),
            Self::TruncatedSection(sec) => write!(f, "truncated .rl section: {sec}"),
            Self::SectionTooLarge => write!(f, ".rl section count exceeds bounded limit"),
            Self::LengthMismatch { expected, actual } => {
                write!(
                    f,
                    ".rl byte length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::HashMismatch { expected, actual } => {
                write!(
                    f,
                    ".rl content hash mismatch: expected {expected:#018x}, got {actual:#018x}"
                )
            }
            Self::UnknownOpcode(op) => write!(f, "unknown .rl opcode: {op:#04x}"),
            Self::InvalidTypeLane(tl) => write!(f, "invalid TypeLane byte: {tl:#04x}"),
            Self::InvalidScheduleHint(sh) => write!(f, "invalid ScheduleHint byte: {sh:#04x}"),
            Self::InvalidTokenMode(raw) => write!(f, "invalid TokenOp encoding: {raw:#010x}"),
            Self::InvalidProofRule(rule) => write!(f, "invalid ProofRuleKind: {rule}"),
        }
    }
}

impl std::error::Error for RulFormatError {}
