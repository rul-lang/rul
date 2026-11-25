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

/// Maximum number of statically declared growable region slots (`0..64`).
pub const MAX_REGION_SLOTS: usize = 64;

/// Maximum permitted byte capacity of a single declared region (`256 MiB`).
pub const MAX_REGION_BYTES: usize = 256 * 1024 * 1024;

/// Maximum number of live function activations (entry function included) permitted by the ISA.
pub const MAX_CALL_DEPTH: u32 = 4096;

/// Maximum sum of ISA frame units across live function activations (`1 << 20` units = `8 MiB`).
pub const MAX_STACK_UNITS: u32 = 1 << 20;

/// Maximum number of nodes permitted in a single `Func`..`End` scope (`< 0xFFFF` sentinel).
pub const MAX_FUNC_NODES: usize = 65_535;

/// Sentinel packed 16-bit operand value indicating an unused argument slot in `Call` / `CallArg`.
pub const CALL_ARG_SENTINEL: u16 = 0xFFFF;

/// Sentinel `bound_lo` value in [`ProofRuleKind::FuncRegionEffect`] declaring an empty function region footprint (`dom(f) = 0`).
pub const REGION_FOOTPRINT_NONE: u32 = 0xFFFF;

/// Computes the ISA-defined stack budget in 8-byte frame units for a function spanning `node_count` nodes.
#[must_use]
pub const fn func_frame_units(node_count: usize) -> u32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "verified function scopes contain at most MAX_FUNC_NODES (65_535) nodes"
    )]
    let count_u32 = node_count as u32;
    72_u32.saturating_add(count_u32)
}

/// Capability bit: Pure compute only (no external I/O allowed).
pub const CAP_NONE: u64 = 0x00;
/// Capability bit: Lock-free network ring enqueue/dequeue allowed.
pub const CAP_NET_RING: u64 = 0x01;
/// Capability bit: Direct NVMe/storage DMA allowed.
pub const CAP_STORAGE_DMA: u64 = 0x02;
/// Capability bit: Heterogeneous GPU/NPU queue dispatch allowed.
pub const CAP_GPU_DISPATCH: u64 = 0x04;
/// Capability bit: Unrestricted unbounded loops allowed without a static `LoopBound` proof.
pub const CAP_DIVERGE: u64 = 0x08;

/// Core opcodes of the Rul binary instruction set.
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
    /// Load a 64-bit two's-complement signed integer scalar constant (`in_a = low32`, `in_b = high32`).
    ConstI64 = 0x05,
    /// Read a definitely-initialized typed scalar local slot (`in_a = slot` in `0..64`, `in_b = 0`).
    LocalGet = 0x06,
    /// Write a scalar value into a typed local slot (`in_a = slot` in `0..64`, `in_b = value_node`).
    LocalSet = 0x07,
    /// Branchless elementwise addition (`out[i] = a[i] + b[i]`).
    Add = 0x10,
    /// Branchless elementwise subtraction (`out[i] = a[i] - b[i]`).
    Sub = 0x11,
    /// Branchless elementwise multiplication (`out[i] = a[i] * b[i]`).
    Mul = 0x12,
    /// Branchless elementwise IEEE-754 division (`out[i] = a[i] / b[i]`).
    Div = 0x13,
    /// Branchless elementwise greater-than comparison (`mask[i] = a[i] > b[i]`).
    CmpGt = 0x14,
    /// Branchless predicated selection (`out[i] = mask[i] ? val[i] : 0.0`).
    SelectZero = 0x15,
    /// Branchless elementwise minimum (`out[i] = min(a[i], b[i])`).
    Min = 0x16,
    /// Branchless elementwise maximum (`out[i] = max(a[i], b[i])`).
    Max = 0x17,
    /// Branchless elementwise less-than comparison (`mask[i] = a[i] < b[i]`).
    CmpLt = 0x18,
    /// Branchless elementwise equality comparison (`mask[i] = a[i] == b[i]`).
    CmpEq = 0x19,
    /// Branchless elementwise IEEE-754 absolute value (`out[i] = |a[i]|`).
    Abs = 0x1A,
    /// Branchless elementwise IEEE-754 square root (`out[i] = sqrt(a[i])`).
    Sqrt = 0x1B,
    /// Branchless elementwise floor rounding toward $-\infty$ (`out[i] = floor(a[i])`).
    Floor = 0x1C,
    /// Branchless elementwise ceiling rounding toward $+\infty$ (`out[i] = ceil(a[i])`).
    Ceil = 0x1D,
    /// Elementwise natural logarithm (`out[i] = ln(a[i])`).
    Ln = 0x1E,
    /// Elementwise natural exponential (`out[i] = exp(a[i])`).
    Exp = 0x1F,
    /// Stream compaction filtering elements of `in_b` where mask `in_a` is true.
    FilterCompact = 0x20,
    /// Horizontal sum reduction of column `in_a` to a scalar `f64`.
    ReduceSum = 0x21,
    /// Horizontal maximum reduction of column `in_a` to a scalar `f64`.
    ReduceMax = 0x22,
    /// Indirect indexed gather (`out[i] = arena_node[index_node[i]]`, requires `proof_idx > 0`).
    Gather = 0x23,
    /// Horizontal minimum reduction of column `in_a` to a scalar `f64`.
    ReduceMin = 0x24,
    /// Branchless elementwise boolean mask conjunction (`mask[i] = a[i] && b[i]`).
    MaskAnd = 0x25,
    /// Branchless elementwise boolean mask disjunction (`mask[i] = a[i] || b[i]`).
    MaskOr = 0x26,
    /// Branchless elementwise IEEE-754 fused multiply-add (`out[i] = a[i] * b[i] + c[i]` with single rounding; `in_a = a`, `in_b = (b & 0xFFFF) | (c << 16)`).
    Fma = 0x27,
    /// Scalar boolean mask negation (`out = !a`, `in_a = mask_node`, `in_b = 0`).
    MaskNot = 0x28,
    /// Explicit scalar primitive cast (`in_a = src_node`, `in_b = 0`, target type from `type_lane.elem`).
    Cast = 0x29,
    /// Scalar integer bitwise AND (`I64` or `U8`).
    BitAnd = 0x2A,
    /// Scalar integer bitwise OR (`I64` or `U8`).
    BitOr = 0x2B,
    /// Scalar integer bitwise XOR (`I64` or `U8`).
    BitXor = 0x2C,
    /// Scalar integer shift left (`I64` masks count `& 63`; `U8` masks count `& 7`).
    Shl = 0x2D,
    /// Scalar integer logical (unsigned) shift right (`I64` masks count `& 63`; `U8` masks count `& 7`).
    ShrU = 0x2E,
    /// Scalar integer checked remainder (`I64` signed, `U8` unsigned; traps on div-by-zero or `i64::MIN % -1`).
    Rem = 0x2F,
    /// Enqueue scalar/column output to a hardware NIC ring (requires [`CAP_NET_RING`]).
    RingEnqueue = 0x30,
    /// Scalar `I64` arithmetic (signed) shift right (count masked `& 63`).
    ShrS = 0x38,
    /// Scalar `I64` unsigned less-than comparison (`(a as u64) < (b as u64)` yielding `BoolMask`).
    CmpLtU = 0x39,
    /// Output contract assertion checking non-NaN and finite bounds on `in_a`.
    AssertContract = 0x40,
    /// Structured forward-exit block header (`optional proof_idx -> BlockSig`).
    Block = 0x50,
    /// Structured backward-branch loop header (`proof_idx -> LoopBound`, or `0` with [`CAP_DIVERGE`]).
    Loop = 0x51,
    /// Structured conditional block header (`in_a = cond_node`, `optional proof_idx -> BlockSig`).
    If = 0x52,
    /// Alternate branch delimiter for an enclosing [`Opcode::If`] (`in_a = optional_then_result`, `in_b = has_result`).
    Else = 0x53,
    /// Structured block/loop/conditional terminator (`in_a = optional_result`, `in_b = has_result`).
    End = 0x54,
    /// Unconditional relative structured branch (`in_a = relative_depth`).
    Br = 0x55,
    /// Conditional relative structured branch (`in_a = cond_node`, `in_b = relative_depth`).
    BrIf = 0x56,
    /// Function scope header (`proof_idx == 0`, `in_a = func_id`, `in_b = 0`).
    Func = 0x57,
    /// Return from enclosing function (`in_a = rel_val_node` or `0` for `void`, `in_b = 0`).
    Return = 0x58,
    /// Direct function call (`in_a = target_func_id`, `in_b = (arg0: u16) | ((arg1: u16) << 16)` with `0xFFFF` sentinel).
    Call = 0x59,
    /// Additional call arguments immediately preceding `Call` for arity 3 or 4 (`in_a = arg2`, `in_b = arg3` or `0xFFFF`).
    CallArg = 0x5A,
    /// Reset a declared region to `len` zeroed elements (`in_a = slot`, `in_b = len_node`; void).
    RegionNew = 0x60,
    /// Append `add_len` zeroed elements to a declared region (`in_a = slot`, `in_b = add_len_node`; yields new length as `I64`).
    RegionGrow = 0x61,
    /// Query current element length of a declared region (`in_a = slot`, `in_b = 0`; yields `I64`).
    RegionLen = 0x62,
    /// Load one element from a declared region (`in_a = slot`, `in_b = idx_node`; bounds-checked unless `proof_idx > 0`).
    RegionLoad = 0x63,
    /// Store one element into a declared region (`in_a = slot`, `in_b = (idx: u16) | ((val: u16) << 16)`; void).
    RegionStore = 0x64,
    /// Release a declared region's mapping and set length to `0` (`in_a = slot`, `in_b = 0`; requires `min_len == 0`; void).
    RegionFree = 0x65,
    /// Begin a compile-time read lease on a declared region (`in_a = slot`, `in_b = 0`; `Owned -> Leased`; void, zero runtime cost).
    RegionLease = 0x66,
    /// End a compile-time read lease on a declared region (`in_a = slot`, `in_b = 0`; `Leased -> Owned`; void, zero runtime cost).
    RegionEndLease = 0x67,
    /// Move ownership of a declared region descriptor (`in_a = src_slot`, `in_b = dst_slot`; `src: Owned -> Moved`, `dst: Moved -> Owned`; void).
    RegionMove = 0x68,
    /// Unmap and drop a declared region (`in_a = slot`, `in_b = 0`; `Owned -> Moved`; requires `Plain` role and `min_len == 0`; void).
    RegionDrop = 0x69,
}

impl Opcode {
    /// Decodes a raw byte into an [`Opcode`].
    ///
    /// # Errors
    /// Returns [`RulFormatError::UnknownOpcode`] if the byte is not a valid opcode.
    pub const fn from_u8(byte: u8) -> Result<Self, RulFormatError> {
        match byte {
            0x01 => Ok(Self::ArenaInput),
            0x02 => Ok(Self::ConstF64),
            0x03 => Ok(Self::ArenaSlice),
            0x04 => Ok(Self::ArenaMutate),
            0x05 => Ok(Self::ConstI64),
            0x06 => Ok(Self::LocalGet),
            0x07 => Ok(Self::LocalSet),
            0x10 => Ok(Self::Add),
            0x11 => Ok(Self::Sub),
            0x12 => Ok(Self::Mul),
            0x13 => Ok(Self::Div),
            0x14 => Ok(Self::CmpGt),
            0x15 => Ok(Self::SelectZero),
            0x16 => Ok(Self::Min),
            0x17 => Ok(Self::Max),
            0x18 => Ok(Self::CmpLt),
            0x19 => Ok(Self::CmpEq),
            0x1A => Ok(Self::Abs),
            0x1B => Ok(Self::Sqrt),
            0x1C => Ok(Self::Floor),
            0x1D => Ok(Self::Ceil),
            0x1E => Ok(Self::Ln),
            0x1F => Ok(Self::Exp),
            0x20 => Ok(Self::FilterCompact),
            0x21 => Ok(Self::ReduceSum),
            0x22 => Ok(Self::ReduceMax),
            0x23 => Ok(Self::Gather),
            0x24 => Ok(Self::ReduceMin),
            0x25 => Ok(Self::MaskAnd),
            0x26 => Ok(Self::MaskOr),
            0x27 => Ok(Self::Fma),
            0x28 => Ok(Self::MaskNot),
            0x29 => Ok(Self::Cast),
            0x2A => Ok(Self::BitAnd),
            0x2B => Ok(Self::BitOr),
            0x2C => Ok(Self::BitXor),
            0x2D => Ok(Self::Shl),
            0x2E => Ok(Self::ShrU),
            0x2F => Ok(Self::Rem),
            0x30 => Ok(Self::RingEnqueue),
            0x38 => Ok(Self::ShrS),
            0x39 => Ok(Self::CmpLtU),
            0x40 => Ok(Self::AssertContract),
            0x50 => Ok(Self::Block),
            0x51 => Ok(Self::Loop),
            0x52 => Ok(Self::If),
            0x53 => Ok(Self::Else),
            0x54 => Ok(Self::End),
            0x55 => Ok(Self::Br),
            0x56 => Ok(Self::BrIf),
            0x57 => Ok(Self::Func),
            0x58 => Ok(Self::Return),
            0x59 => Ok(Self::Call),
            0x5A => Ok(Self::CallArg),
            0x60 => Ok(Self::RegionNew),
            0x61 => Ok(Self::RegionGrow),
            0x62 => Ok(Self::RegionLen),
            0x63 => Ok(Self::RegionLoad),
            0x64 => Ok(Self::RegionStore),
            0x65 => Ok(Self::RegionFree),
            0x66 => Ok(Self::RegionLease),
            0x67 => Ok(Self::RegionEndLease),
            0x68 => Ok(Self::RegionMove),
            0x69 => Ok(Self::RegionDrop),
            other => Err(RulFormatError::UnknownOpcode(other)),
        }
    }

    /// Returns `true` if this opcode belongs to the v0.5 structured control-flow, typed scalar, or declared region ISA.
    #[must_use]
    pub const fn is_structured_or_local(self) -> bool {
        matches!(
            self,
            Self::ConstI64
                | Self::LocalGet
                | Self::LocalSet
                | Self::MaskNot
                | Self::Cast
                | Self::BitAnd
                | Self::BitOr
                | Self::BitXor
                | Self::Shl
                | Self::ShrU
                | Self::Rem
                | Self::ShrS
                | Self::CmpLtU
                | Self::Block
                | Self::Loop
                | Self::If
                | Self::Else
                | Self::End
                | Self::Br
                | Self::BrIf
                | Self::Func
                | Self::Return
                | Self::Call
                | Self::CallArg
                | Self::RegionNew
                | Self::RegionGrow
                | Self::RegionLen
                | Self::RegionLoad
                | Self::RegionStore
                | Self::RegionFree
                | Self::RegionLease
                | Self::RegionEndLease
                | Self::RegionMove
                | Self::RegionDrop
        )
    }

    /// Returns `true` if this opcode operates on a statically declared growable region.
    #[must_use]
    pub const fn is_region_op(self) -> bool {
        matches!(
            self,
            Self::RegionNew
                | Self::RegionGrow
                | Self::RegionLen
                | Self::RegionLoad
                | Self::RegionStore
                | Self::RegionFree
                | Self::RegionLease
                | Self::RegionEndLease
                | Self::RegionMove
                | Self::RegionDrop
        )
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
    /// 8-bit unsigned wrapping byte (`u8`).
    U8 = 0x5,
}

impl ElemType {
    /// Decodes a raw `u8` tag into an [`ElemType`].
    ///
    /// # Errors
    /// Returns [`RulFormatError::InvalidTypeLane`] if the tag is unrecognized.
    pub const fn from_u8(raw: u8) -> Result<Self, RulFormatError> {
        match raw {
            0x1 => Ok(Self::F64),
            0x2 => Ok(Self::I64),
            0x3 => Ok(Self::BoolMask),
            0x5 => Ok(Self::U8),
            other => Err(RulFormatError::InvalidTypeLane(other)),
        }
    }

    /// Returns the element byte width inside a declared region (`8` for `F64`/`I64`, `1` for `U8`/`BoolMask`).
    #[must_use]
    pub const fn region_elem_bytes(self) -> usize {
        match self {
            Self::F64 | Self::I64 => 8,
            Self::U8 | Self::BoolMask => 1,
        }
    }
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

    /// Standard scalar `i64` type (`i64x1`).
    pub const I64_SCALAR: Self = Self {
        elem: ElemType::I64,
        width: SimdWidth::Scalar,
    };

    /// Alias for [`Self::I64_SCALAR`] (`i64x1`).
    pub const I64X1: Self = Self::I64_SCALAR;

    /// Standard scalar `u8` type (`u8x1`).
    pub const U8_SCALAR: Self = Self {
        elem: ElemType::U8,
        width: SimdWidth::Scalar,
    };

    /// Alias for [`Self::U8_SCALAR`] (`u8x1`).
    pub const U8X1: Self = Self::U8_SCALAR;

    /// Standard scalar boolean mask type (`maskx1`).
    pub const MASK_SCALAR: Self = Self {
        elem: ElemType::BoolMask,
        width: SimdWidth::Scalar,
    };

    /// Alias for [`Self::MASK_SCALAR`] (`maskx1`).
    pub const MASKX1: Self = Self::MASK_SCALAR;

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
            0x5 => ElemType::U8,
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

/// Role of a statically declared region encoded in `ProofEntry.witness_node >> 8`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RegionRole {
    /// Ordinary program-global growable region (`0`).
    Plain = 0,
    /// Declarative byte input region pre-filled from stdin / caller buffer (`1`; must be `U8` with `min_len == 0`).
    Input = 1,
    /// Declarative byte output region flushed to stdout / returned bytes on success (`2`; must be `U8`).
    Output = 2,
}

impl RegionRole {
    /// Decodes a raw `u8` role tag into a [`RegionRole`].
    #[must_use]
    pub const fn from_u8(raw: u8) -> Option<Self> {
        match raw {
            0 => Some(Self::Plain),
            1 => Some(Self::Input),
            2 => Some(Self::Output),
            _ => None,
        }
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
    /// Declares structured block signature (`target_arena` version transition `bound_lo -> bound_hi` and optional scalar result `TypeLane` in `witness_node`).
    BlockSig = 3,
    /// Declares a static upper bound (`1 <= bound_hi <= 1_000_000`) on back-edge iterations of a [`Opcode::Loop`] without [`CAP_DIVERGE`].
    LoopBound = 4,
    /// Declares a function signature (`target_arena = func_id`, `bound_lo = arity (0..=4)`, `bound_hi = packed 4-bit param ElemTypes`, `witness_node = ret ElemType or 0 for void`).
    FuncSig = 5,
    /// Declares a statically typed, bounded growable region (`target_arena = slot`, `bound_lo = min_len`, `bound_hi = max_len`, `witness_node = elem | (role << 8)`).
    RegionDecl = 6,
    /// Certifies that a `RegionLoad` or `RegionStore` index is `< bound_hi <= min_len` for `target_arena = slot`.
    RegionIndexInBounds = 7,
    /// Declares a function's region footprint and typestate transition (`target_arena = func_id`, `bound_lo = slot or 0xFFFF`, `bound_hi = pre`, `witness_node = post`).
    FuncRegionEffect = 8,
}

impl ProofRuleKind {
    const fn from_u16(raw: u16) -> Result<Self, RulFormatError> {
        match raw {
            1 => Ok(Self::SliceBounds),
            2 => Ok(Self::GatherIndexInBounds),
            3 => Ok(Self::BlockSig),
            4 => Ok(Self::LoopBound),
            5 => Ok(Self::FuncSig),
            6 => Ok(Self::RegionDecl),
            7 => Ok(Self::RegionIndexInBounds),
            8 => Ok(Self::FuncRegionEffect),
            other => Err(RulFormatError::InvalidProofRule(other)),
        }
    }
}

/// Compile-time ownership typestate of a statically declared region slot (`0..64`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RegionState {
    /// Exclusive ownership: reads, mutations, leases, moves, and drops are permitted (`0`).
    Owned = 0,
    /// Active compile-time read lease: reads permitted, mutations and nested leases rejected (`1`).
    Leased = 1,
    /// Moved or dropped: all accesses rejected except restoring `Owned` via `RegionNew` or `RegionMove` destination (`2`).
    Moved = 2,
}

impl RegionState {
    /// Decodes a raw `u32` tag into a [`RegionState`].
    #[must_use]
    pub const fn from_u32(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Owned),
            1 => Some(Self::Leased),
            2 => Some(Self::Moved),
            _ => None,
        }
    }
}

/// Decoded per-slot function region effect from a [`ProofRuleKind::FuncRegionEffect`] entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FuncRegionEffectInfo {
    /// Function identifier (`target_arena`).
    pub func_id: u16,
    /// Region slot in `0..64`, or `None` for the empty-footprint sentinel (`0xFFFF`).
    pub slot: Option<u16>,
    /// Required region typestate on function entry (always `Owned` when `slot` is `None`).
    pub pre: RegionState,
    /// Guaranteed region typestate on function exit (always `Owned` when `slot` is `None`).
    pub post: RegionState,
}

/// Decoded function signature from a [`ProofRuleKind::FuncSig`] proof entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FuncSigInfo {
    /// Function identifier (`0..func_count`).
    pub func_id: u16,
    /// Parameter count (`0..=4`).
    pub arity: u8,
    /// Parameter element types in slots `0..arity` (unused slots are `ElemType::I64`).
    pub params: [ElemType; 4],
    /// Return element type (`None` for `void`).
    pub ret_elem: Option<ElemType>,
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
    /// Constructs a [`ProofRuleKind::FuncSig`] entry for `func_id`, `params` (`0..=4`), and optional return type.
    #[must_use]
    pub fn func_sig(func_id: u16, params: &[ElemType], ret_elem: Option<ElemType>) -> Self {
        let mut packed: u32 = 0;
        for (i, &el) in params.iter().take(4).enumerate() {
            packed |= (el as u32) << (i * 4);
        }
        Self {
            rule_kind: ProofRuleKind::FuncSig,
            target_arena: func_id,
            bound_lo: u32::try_from(params.len()).unwrap_or(4),
            bound_hi: packed,
            witness_node: ret_elem.map_or(0, |el| el as u32),
        }
    }

    /// Decodes a [`FuncSigInfo`] from a [`ProofRuleKind::FuncSig`] proof entry.
    ///
    /// Returns `None` if `rule_kind != FuncSig`, `arity > 4`, any param nibble or return tag is invalid,
    /// or unused high bits in `bound_hi` are non-zero.
    #[must_use]
    pub const fn decode_func_sig(self) -> Option<FuncSigInfo> {
        if !matches!(self.rule_kind, ProofRuleKind::FuncSig) || self.bound_lo > 4 {
            return None;
        }
        let arity = self.bound_lo as u8;
        if (self.bound_hi >> (arity * 4)) != 0 {
            return None;
        }
        let mut params = [ElemType::I64; 4];
        let mut i = 0u8;
        while i < arity {
            let nibble = ((self.bound_hi >> (i * 4)) & 0x0F) as u8;
            let elem = match ElemType::from_u8(nibble) {
                Ok(e) => e,
                Err(_) => return None,
            };
            match i {
                0 => params[0] = elem,
                1 => params[1] = elem,
                2 => params[2] = elem,
                3 => params[3] = elem,
                _ => return None,
            }
            i += 1;
        }
        let ret_elem = if self.witness_node == 0 {
            None
        } else if self.witness_node > 0xFF {
            return None;
        } else {
            match ElemType::from_u8(self.witness_node as u8) {
                Ok(e) => Some(e),
                Err(_) => return None,
            }
        };
        Some(FuncSigInfo {
            func_id: self.target_arena,
            arity,
            params,
            ret_elem,
        })
    }

    /// Constructs a [`ProofRuleKind::RegionDecl`] entry encoding `(elem, role)` in `witness_node`.
    #[must_use]
    pub const fn region_decl(
        slot: u16,
        min_len: u32,
        max_len: u32,
        elem: ElemType,
        role: RegionRole,
    ) -> Self {
        Self {
            rule_kind: ProofRuleKind::RegionDecl,
            target_arena: slot,
            bound_lo: min_len,
            bound_hi: max_len,
            witness_node: (elem as u32) | ((role as u32) << 8),
        }
    }

    /// Decodes `(ElemType, RegionRole)` from a [`ProofRuleKind::RegionDecl`] `witness_node` field.
    ///
    /// Returns `None` if upper bits are non-zero, the element type is not in `{F64, I64, U8}`, or the role is unknown.
    #[must_use]
    pub const fn decode_region_decl_witness(witness_node: u32) -> Option<(ElemType, RegionRole)> {
        if (witness_node >> 16) != 0 {
            return None;
        }
        let elem = match (witness_node & 0xFF) as u8 {
            0x1 => ElemType::F64,
            0x2 => ElemType::I64,
            0x5 => ElemType::U8,
            _ => return None,
        };
        let Some(role) = RegionRole::from_u8(((witness_node >> 8) & 0xFF) as u8) else {
            return None;
        };
        Some((elem, role))
    }

    /// Constructs a [`ProofRuleKind::FuncRegionEffect`] entry for `func_id`.
    ///
    /// Pass `slot = None` to encode the empty-footprint sentinel (`bound_lo = 0xFFFF`, `bound_hi = 0`, `witness_node = 0`).
    #[must_use]
    pub const fn func_region_effect(
        func_id: u16,
        slot: Option<u16>,
        pre: RegionState,
        post: RegionState,
    ) -> Self {
        match slot {
            Some(s) => Self {
                rule_kind: ProofRuleKind::FuncRegionEffect,
                target_arena: func_id,
                bound_lo: s as u32,
                bound_hi: pre as u32,
                witness_node: post as u32,
            },
            None => Self {
                rule_kind: ProofRuleKind::FuncRegionEffect,
                target_arena: func_id,
                bound_lo: REGION_FOOTPRINT_NONE,
                bound_hi: 0,
                witness_node: 0,
            },
        }
    }

    /// Decodes a [`FuncRegionEffectInfo`] from a [`ProofRuleKind::FuncRegionEffect`] proof entry.
    ///
    /// Returns `None` if `rule_kind != FuncRegionEffect`, if the empty-footprint sentinel (`0xFFFF`)
    /// has non-zero `bound_hi` or `witness_node`, if `slot >= MAX_REGION_SLOTS`, or if `pre`/`post` is invalid.
    #[must_use]
    pub const fn decode_func_region_effect(self) -> Option<FuncRegionEffectInfo> {
        if !matches!(self.rule_kind, ProofRuleKind::FuncRegionEffect) {
            return None;
        }
        if self.bound_lo == REGION_FOOTPRINT_NONE {
            if self.bound_hi != 0 || self.witness_node != 0 {
                return None;
            }
            return Some(FuncRegionEffectInfo {
                func_id: self.target_arena,
                slot: None,
                pre: RegionState::Owned,
                post: RegionState::Owned,
            });
        }
        if self.bound_lo >= MAX_REGION_SLOTS as u32 {
            return None;
        }
        let Some(pre) = RegionState::from_u32(self.bound_hi) else {
            return None;
        };
        let Some(post) = RegionState::from_u32(self.witness_node) else {
            return None;
        };
        Some(FuncRegionEffectInfo {
            func_id: self.target_arena,
            slot: Some(self.bound_lo as u16),
            pre,
            post,
        })
    }

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

    /// Encodes a [`Opcode::ConstI64`] frame from a 64-bit signed integer value.
    #[must_use]
    pub const fn const_i64(value: i64, schedule: ScheduleHint) -> Self {
        #[expect(
            clippy::cast_sign_loss,
            reason = "splitting 64-bit two's-complement integer into exact low/high u32 halves"
        )]
        let (low, high) = {
            let bits = value as u64;
            ((bits & 0xFFFF_FFFF) as u32, (bits >> 32) as u32)
        };
        Self {
            opcode: Opcode::ConstI64,
            type_lane: TypeLane::I64_SCALAR,
            schedule,
            proof_idx: 0,
            token_op: TokenOp::PURE,
            in_a: low,
            in_b: high,
        }
    }

    /// Extracts the `i64` constant stored in `in_a` and `in_b`.
    #[must_use]
    pub const fn decode_const_i64(self) -> i64 {
        #[expect(
            clippy::cast_possible_wrap,
            reason = "reconstructing 64-bit two's-complement integer from low/high u32 halves"
        )]
        {
            let bits = (self.in_a as u64) | ((self.in_b as u64) << 32);
            bits as i64
        }
    }

    /// Packs two 16-bit node indices `(lo, hi)` into a 32-bit operand field (`Fma` `(b, c)` or `RegionStore` `(idx, val)`).
    #[must_use]
    pub const fn encode_u16_pair(lo: u16, hi: u16) -> u32 {
        (lo as u32) | ((hi as u32) << 16)
    }

    /// Unpacks two 16-bit node indices `(lo, hi)` from `self.in_b`.
    #[must_use]
    pub const fn decode_u16_pair(self) -> (u32, u32) {
        (self.in_b & 0xFFFF, self.in_b >> 16)
    }

    /// Packs the second (`in_b`) and third (`in_c`) 16-bit node indices into the `in_b` field of an [`Opcode::Fma`] frame.
    #[must_use]
    pub const fn encode_fma_bc(in_b: u16, in_c: u16) -> u32 {
        Self::encode_u16_pair(in_b, in_c)
    }

    /// Unpacks the `(in_b, in_c)` node indices from the `in_b` field of an [`Opcode::Fma`] frame.
    #[must_use]
    pub const fn decode_fma_bc(self) -> (u32, u32) {
        self.decode_u16_pair()
    }

    /// Encodes a ternary [`Opcode::Fma`] frame (`a * b + c` with single IEEE-754 rounding).
    #[must_use]
    pub const fn fma(
        type_lane: TypeLane,
        in_a: u32,
        in_b: u16,
        in_c: u16,
        schedule: ScheduleHint,
    ) -> Self {
        Self {
            opcode: Opcode::Fma,
            type_lane,
            schedule,
            proof_idx: 0,
            token_op: TokenOp::PURE,
            in_a,
            in_b: Self::encode_fma_bc(in_b, in_c),
        }
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
    /// Returns `true` if any node or proof uses structured control flow, functions, typed scalar ops, locals, or declared regions.
    #[must_use]
    pub fn uses_structured_control_or_locals(&self) -> bool {
        self.nodes
            .iter()
            .any(|node| node.opcode.is_structured_or_local() || node.type_lane.elem == ElemType::U8)
            || self.proofs.iter().any(|proof| {
                matches!(
                    proof.rule_kind,
                    ProofRuleKind::FuncSig
                        | ProofRuleKind::RegionDecl
                        | ProofRuleKind::RegionIndexInBounds
                        | ProofRuleKind::FuncRegionEffect
                )
            })
    }

    /// Returns `true` if the program contains any `Func` node or `FuncSig` proof entry.
    #[must_use]
    pub fn has_functions(&self) -> bool {
        self.nodes.iter().any(|node| node.opcode == Opcode::Func)
            || self
                .proofs
                .iter()
                .any(|proof| proof.rule_kind == ProofRuleKind::FuncSig)
    }

    /// Returns `true` if any proof entry declares an input or output byte region.
    #[must_use]
    pub fn has_io_regions(&self) -> bool {
        self.input_region_max_len().is_some() || self.has_output_region()
    }

    /// Returns the declared `max_len` of the program's `RegionRole::Input` region, if any.
    #[must_use]
    pub fn input_region_max_len(&self) -> Option<u32> {
        self.proofs.iter().find_map(|proof| {
            if proof.rule_kind == ProofRuleKind::RegionDecl
                && matches!(
                    ProofEntry::decode_region_decl_witness(proof.witness_node),
                    Some((_, RegionRole::Input))
                )
            {
                Some(proof.bound_hi)
            } else {
                None
            }
        })
    }

    /// Returns `true` if any proof entry declares an output byte region.
    #[must_use]
    pub fn has_output_region(&self) -> bool {
        self.proofs.iter().any(|proof| {
            proof.rule_kind == ProofRuleKind::RegionDecl
                && matches!(
                    ProofEntry::decode_region_decl_witness(proof.witness_node),
                    Some((_, RegionRole::Output))
                )
        })
    }

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
