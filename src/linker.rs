//! Zero-dependency verified module linker for Rul (`.rl` / `.rls`) libraries.
//!
//! Combines two or more separately verified modules into a single self-contained, verified `.rl`
//! executable image with:
//! - Strict fail-closed capability enforcement (zero silent capability escalation).
//! - Exact cross-module `FuncSig`, remapped `FuncRegionContract`, and `RegionDecl` checking.
//! - Legacy function footprint materialization (`FuncRegionEffect`, including `0xFFFF` empty sentinel).
//! - Deduplicated node-referenced proof table layout respecting the 1-based `u8` `proof_idx <= 255` ceiling.
//! - Complete $O(\text{nodes} + \text{proofs} + \text{const\_data} + V \log V + E)$ relocation of
//!   functions, regions, constant-data segments, `BrTableSpec`s, and `CallTable`s.
//! - Deterministic min-heap Kahn topological ordering placing the entry function last (`total_funcs - 1`).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fmt;
use std::ops::Range;

use crate::bytecode::{
    FuncSigInfo, LinkDirection, LinkSymbolKind, NodeFrame, Opcode, ProofEntry, ProofRuleKind,
    RegionRole, RegionState, RulFormatError, RulProgram, CAP_DIVERGE, MAX_CONST_DATA_BYTES,
    MAX_REGION_SLOTS, MAX_SECTION_ITEMS,
};
use crate::verifier::{
    declared_region_mask, func_sig_matches, scan_func_region_contracts, scan_func_sigs,
    scan_region_decls, verify_module, verify_program, FuncRegionContract, RegionDeclInfo,
    VerifiedSummary, VerifyDiagnostic, VerifyMode,
};

/// Maximum number of deduplicated node-referenced sidecar proofs addressable by 1-based `u8` `proof_idx`.
pub const MAX_NODE_PROOF_INDEX: usize = 255;

/// Options controlling verified module linking via [`link_modules`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LinkOptions {
    /// Exported function symbol name to select as the linked executable's entry point (`--entry <name>`).
    ///
    /// When `None`, the last module in `modules` is treated as the entry module and its final
    /// locally-defined function is used as the executable entry point.
    pub entry_export: Option<String>,
    /// Explicit capability bitmask ceiling for the linked executable (`--cap <hex>`).
    ///
    /// When `None`, defaults to the entry module's `capability_mask`. Every input module must satisfy
    /// `(module.capability_mask & !final_cap) == 0`; capabilities are never silently OR-ed or escalated.
    pub capability_mask: Option<u64>,
}

/// Output produced by [`link_modules`] upon producing a verified self-contained `.rl` executable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkReport {
    /// Linked, verified, self-contained `.rl` executable program.
    pub program: RulProgram,
    /// Verification summary from [`verify_program`] on the linked executable.
    pub summary: VerifiedSummary,
    /// Number of input modules linked.
    pub module_count: usize,
    /// Total number of functions in the linked executable.
    pub func_count: usize,
    /// Total number of region slots allocated in the linked executable.
    pub region_count: usize,
    /// Total byte length of Section 5a (`const_data`) in the linked executable.
    pub const_data_bytes: usize,
    /// Final capability bitmask applied to the linked executable.
    pub capability_mask: u64,
}

/// Structured fail-closed error returned by [`link_modules`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkError {
    /// No input modules were supplied.
    EmptyInput,
    /// Input module failed single-pass verification via [`verify_module`].
    ModuleVerifyFailed {
        /// Zero-based index of the failing input module.
        module_index: usize,
        /// Verification diagnostic returned by [`verify_module`].
        diagnostic: VerifyDiagnostic,
    },
    /// Input module contains malformed Section 5b (`link_meta`) metadata.
    InvalidLinkMetadata {
        /// Zero-based index of the offending input module.
        module_index: usize,
        /// Format decoding error.
        error: RulFormatError,
    },
    /// Input module declares columnar `ArenaDescriptor`s (unsupported until the `v0.7` columnar bridge).
    UnsupportedColumnarModule {
        /// Zero-based index of the columnar input module.
        module_index: usize,
    },
    /// Input module does not define function-based module units (`FuncSig`/`Func`).
    UnsupportedNonFunctionModule {
        /// Zero-based index of the non-function input module.
        module_index: usize,
    },
    /// An input module requires capability bits not granted by the entry module or explicit `--cap`.
    CapabilityEscalation {
        /// Zero-based index of the offending input module.
        module_index: usize,
        /// Capability mask declared by the offending module.
        module_cap: u64,
        /// Allowed capability ceiling for the linked executable.
        allowed_cap: u64,
    },
    /// Two input modules exported the same symbol name for the same kind.
    DuplicateExport {
        /// Symbol kind (`Func` or `Region`).
        kind: LinkSymbolKind,
        /// Conflicting symbol name.
        name: String,
        /// Module index of the first export.
        first_module: usize,
        /// Module index of the duplicate export.
        second_module: usize,
    },
    /// An imported function or region symbol has no matching export in any input module.
    UnresolvedSymbol {
        /// Zero-based index of the importing module.
        module_index: usize,
        /// Symbol kind (`Func` or `Region`).
        kind: LinkSymbolKind,
        /// Missing symbol name.
        name: String,
    },
    /// Explicit `--entry <name>` did not match any exported function.
    UnknownEntryExport {
        /// Requested entry export symbol name.
        name: String,
    },
    /// Selected entry function is not a valid 0-arity non-void entry point.
    InvalidEntryFunction {
        /// Reason the selected entry function was rejected.
        reason: &'static str,
    },
    /// Imported function signature (`arity`, parameter types, or return type) disagrees with export.
    FuncSigMismatch {
        /// Conflicting function symbol name.
        name: String,
        /// Zero-based index of the importing module.
        importer_module: usize,
        /// Zero-based index of the exporting module.
        exporter_module: usize,
    },
    /// Imported function remapped `FuncRegionContract` disagrees with the exported function's contract.
    FuncRegionEffectMismatch {
        /// Conflicting function symbol name.
        name: String,
        /// Zero-based index of the importing module.
        importer_module: usize,
        /// Zero-based index of the exporting module.
        exporter_module: usize,
    },
    /// Imported region `(elem, role, min_len, max_len)` disagrees with the exported region.
    RegionDeclMismatch {
        /// Conflicting region symbol name.
        name: String,
        /// Zero-based index of the importing module.
        importer_module: usize,
        /// Zero-based index of the exporting module.
        exporter_module: usize,
    },
    /// Combined unique regions across linked modules exceed `MAX_REGION_SLOTS` (64).
    TooManyRegions {
        /// Total unique region slots needed.
        needed: usize,
        /// Maximum supported region slots (`64`).
        max: usize,
    },
    /// Linked modules declare multiple distinct `Input` or `Output` I/O regions.
    ConflictingIoRegion {
        /// Conflicting I/O region role (`Input` or `Output`).
        role: RegionRole,
    },
    /// Combined functions across linked modules exceed `u16` `func_id` space (`65_536`).
    TooManyFunctions {
        /// Total functions needed.
        needed: usize,
        /// Maximum supported function count (`65_536`).
        max: usize,
    },
    /// Deduplicated node-referenced proofs exceed the 1-based `u8` `proof_idx` ceiling (`255`).
    ProofIndexOverflow {
        /// Total unique node-referenced proofs required after relocation and deduplication.
        needed: usize,
        /// Maximum representable 1-based `u8` proof index (`255`).
        max: usize,
    },
    /// Inter-module or intra-module call cycle detected while `CAP_DIVERGE` is not granted.
    CycleWithoutDiverge,
    /// Combined linked binary exceeds a bounded section or identifier limit.
    ResourceLimitExceeded {
        /// Specific bounded limit exceeded.
        reason: &'static str,
    },
    /// Defense-in-depth failure if the final linked `.rl` image fails [`verify_program`].
    LinkedVerifyFailed(VerifyDiagnostic),
}

impl fmt::Display for LinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "linker requires at least one input module"),
            Self::ModuleVerifyFailed {
                module_index,
                diagnostic,
            } => write!(f, "module #{module_index} failed verification: {diagnostic}"),
            Self::InvalidLinkMetadata {
                module_index,
                error,
            } => write!(f, "module #{module_index} has invalid link_meta: {error}"),
            Self::UnsupportedColumnarModule { module_index } => write!(
                f,
                "module #{module_index} uses columnar arenas, which are unsupported by the v0.6 linker"
            ),
            Self::UnsupportedNonFunctionModule { module_index } => write!(
                f,
                "module #{module_index} contains no Func declarations; linker requires function modules"
            ),
            Self::CapabilityEscalation {
                module_index,
                module_cap,
                allowed_cap,
            } => write!(
                f,
                "capability escalation rejected: module #{module_index} requests {module_cap:#x}, exceeding allowed cap {allowed_cap:#x}"
            ),
            Self::DuplicateExport {
                kind,
                name,
                first_module,
                second_module,
            } => write!(
                f,
                "duplicate {kind:?} export '{name}' in modules #{first_module} and #{second_module}"
            ),
            Self::UnresolvedSymbol {
                module_index,
                kind,
                name,
            } => write!(
                f,
                "unresolved {kind:?} import '{name}' in module #{module_index}"
            ),
            Self::UnknownEntryExport { name } => {
                write!(f, "entry export function '{name}' not found in any module")
            }
            Self::InvalidEntryFunction { reason } => {
                write!(f, "invalid entry function: {reason}")
            }
            Self::FuncSigMismatch {
                name,
                importer_module,
                exporter_module,
            } => write!(
                f,
                "FuncSig mismatch for '{name}' between importer module #{importer_module} and exporter module #{exporter_module}"
            ),
            Self::FuncRegionEffectMismatch {
                name,
                importer_module,
                exporter_module,
            } => write!(
                f,
                "FuncRegionEffect contract mismatch for '{name}' between importer module #{importer_module} and exporter module #{exporter_module}"
            ),
            Self::RegionDeclMismatch {
                name,
                importer_module,
                exporter_module,
            } => write!(
                f,
                "RegionDecl mismatch for '{name}' between importer module #{importer_module} and exporter module #{exporter_module}"
            ),
            Self::TooManyRegions { needed, max } => {
                write!(f, "too many regions after linking: needed {needed}, max {max}")
            }
            Self::ConflictingIoRegion { role } => {
                write!(f, "conflicting multiple {role:?} regions across linked modules")
            }
            Self::TooManyFunctions { needed, max } => {
                write!(f, "too many functions after linking: needed {needed}, max {max}")
            }
            Self::ProofIndexOverflow { needed, max } => write!(
                f,
                "node-referenced proof index overflow: needed {needed} unique proofs, max {max}"
            ),
            Self::CycleWithoutDiverge => write!(
                f,
                "call graph cycle detected across linked modules without CAP_DIVERGE"
            ),
            Self::ResourceLimitExceeded { reason } => {
                write!(f, "linker resource limit exceeded: {reason}")
            }
            Self::LinkedVerifyFailed(diag) => {
                write!(f, "linked program failed final verification: {diag}")
            }
        }
    }
}

impl std::error::Error for LinkError {}

struct RawCallTable {
    offset: u32,
    entry_count: u32,
}

struct ParsedModule {
    regions: [Option<RegionDeclInfo>; MAX_REGION_SLOTS],
    func_sigs: Vec<FuncSigInfo>,
    func_contracts: Vec<FuncRegionContract>,
    imported_funcs: Vec<Option<String>>,
    exported_funcs: Vec<(String, u16)>,
    imported_regions: [Option<String>; MAX_REGION_SLOTS],
    exported_regions: Vec<(String, u16)>,
    func_bodies: Vec<Option<Range<usize>>>,
    call_tables: Vec<RawCallTable>,
    seg_count: usize,
    br_table_count: usize,
}

struct DefinedFunc {
    module_index: usize,
    orig_func_id: u16,
    sig: FuncSigInfo,
    remapped_contract: FuncRegionContract,
    body_range: Range<usize>,
}

fn parse_and_verify_module(
    module_index: usize,
    prog: &RulProgram,
) -> Result<ParsedModule, LinkError> {
    if !prog.arenas.is_empty() {
        return Err(LinkError::UnsupportedColumnarModule { module_index });
    }
    verify_module(prog).map_err(|diagnostic| LinkError::ModuleVerifyFailed {
        module_index,
        diagnostic,
    })?;
    let link_meta = prog
        .decode_link_meta()
        .map_err(|error| LinkError::InvalidLinkMetadata {
            module_index,
            error,
        })?;

    let regions = scan_region_decls(prog).map_err(|diagnostic| LinkError::ModuleVerifyFailed {
        module_index,
        diagnostic,
    })?;
    let declared_mask = declared_region_mask(&regions);
    let func_sigs = scan_func_sigs(prog).map_err(|diagnostic| LinkError::ModuleVerifyFailed {
        module_index,
        diagnostic,
    })?;
    if func_sigs.is_empty() {
        return Err(LinkError::UnsupportedNonFunctionModule { module_index });
    }
    let func_contracts = scan_func_region_contracts(
        prog,
        &regions,
        declared_mask,
        &func_sigs,
        VerifyMode::Module,
    )
    .map_err(|diagnostic| LinkError::ModuleVerifyFailed {
        module_index,
        diagnostic,
    })?;

    let mut imported_funcs: Vec<Option<String>> = vec![None; func_sigs.len()];
    let mut exported_funcs: Vec<(String, u16)> = Vec::new();
    let mut imported_regions: [Option<String>; MAX_REGION_SLOTS] = std::array::from_fn(|_| None);
    let mut exported_regions: Vec<(String, u16)> = Vec::new();

    for entry in link_meta.entries {
        match (entry.direction, entry.kind) {
            (LinkDirection::Import, LinkSymbolKind::Func) => {
                if let Some(slot) = imported_funcs.get_mut(usize::from(entry.local_id)) {
                    *slot = Some(entry.name);
                }
            }
            (LinkDirection::Export, LinkSymbolKind::Func) => {
                exported_funcs.push((entry.name, entry.local_id));
            }
            (LinkDirection::Import, LinkSymbolKind::Region) => {
                if let Some(slot) = imported_regions.get_mut(usize::from(entry.local_id)) {
                    *slot = Some(entry.name);
                }
            }
            (LinkDirection::Export, LinkSymbolKind::Region) => {
                exported_regions.push((entry.name, entry.local_id));
            }
        }
    }

    // Index each locally-defined Func body range [start..=end] in a single O(N) scan.
    let mut func_bodies: Vec<Option<Range<usize>>> = vec![None; func_sigs.len()];
    let mut ctrl_depth: usize = 0;
    let mut active_func: Option<(usize, usize)> = None;
    for (idx, frame) in prog.nodes.iter().enumerate() {
        match frame.opcode {
            Opcode::Func => {
                active_func = Some((frame.in_a as usize, idx));
                ctrl_depth = 1;
            }
            Opcode::Block | Opcode::Loop | Opcode::If => {
                ctrl_depth = ctrl_depth.saturating_add(1);
            }
            Opcode::End => {
                ctrl_depth = ctrl_depth.saturating_sub(1);
                if ctrl_depth == 0 {
                    if let Some((fid, start)) = active_func.take() {
                        if let Some(slot) = func_bodies.get_mut(fid) {
                            *slot = Some(start..(idx + 1));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // Count const_data tables and index CallTables by dense table_id in O(proofs).
    let mut seg_count: usize = 0;
    let mut br_table_count: usize = 0;
    let mut raw_call_tables: Vec<(u16, RawCallTable)> = Vec::new();
    for proof in &prog.proofs {
        match proof.rule_kind {
            ProofRuleKind::ConstDataSeg => seg_count = seg_count.saturating_add(1),
            ProofRuleKind::BrTableSpec => br_table_count = br_table_count.saturating_add(1),
            ProofRuleKind::CallTable => {
                raw_call_tables.push((
                    proof.target_arena,
                    RawCallTable {
                        offset: proof.bound_lo,
                        entry_count: proof.bound_hi,
                    },
                ));
            }
            _ => {}
        }
    }
    let mut call_tables: Vec<RawCallTable> = Vec::with_capacity(raw_call_tables.len());
    for _ in 0..raw_call_tables.len() {
        call_tables.push(RawCallTable {
            offset: 0,
            entry_count: 0,
        });
    }
    for (tid, ct) in raw_call_tables {
        if let Some(slot) = call_tables.get_mut(usize::from(tid)) {
            *slot = ct;
        }
    }

    Ok(ParsedModule {
        regions,
        func_sigs,
        func_contracts,
        imported_funcs,
        exported_funcs,
        imported_regions,
        exported_regions,
        func_bodies,
        call_tables,
        seg_count,
        br_table_count,
    })
}

fn remap_region_mask(mut mask: u64, map: &[Option<u16>; MAX_REGION_SLOTS]) -> u64 {
    let mut out: u64 = 0;
    while mask != 0 {
        let bit_idx = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        if let Some(Some(global_slot)) = map.get(bit_idx) {
            out |= 1u64 << *global_slot;
        }
    }
    out
}

fn remap_contract(
    c: FuncRegionContract,
    map: &[Option<u16>; MAX_REGION_SLOTS],
) -> FuncRegionContract {
    FuncRegionContract {
        dom_mask: remap_region_mask(c.dom_mask, map),
        pre_owned: remap_region_mask(c.pre_owned, map),
        pre_leased: remap_region_mask(c.pre_leased, map),
        post_owned: remap_region_mask(c.post_owned, map),
        post_leased: remap_region_mask(c.post_leased, map),
    }
}

fn order_functions_deterministic(
    modules: &[RulProgram],
    parsed: &[ParsedModule],
    defined_funcs: &[DefinedFunc],
    local_func_to_def: &[Vec<usize>],
    entry_def_id: usize,
    allow_diverge: bool,
) -> Result<Vec<usize>, LinkError> {
    let v_count = defined_funcs.len();
    if allow_diverge {
        let mut ordered = Vec::with_capacity(v_count);
        for def_id in 0..v_count {
            if def_id != entry_def_id {
                ordered.push(def_id);
            }
        }
        ordered.push(entry_def_id);
        return Ok(ordered);
    }

    // Build callee -> caller dependency graph in O(total nodes + total const_data + E).
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); v_count];
    let mut in_degree: Vec<usize> = vec![0; v_count];
    let mut seen_callees: HashSet<usize> = HashSet::new();
    let mut seen_tables: HashSet<u16> = HashSet::new();

    for (caller_def, df) in defined_funcs.iter().enumerate() {
        seen_callees.clear();
        seen_tables.clear();
        let m = df.module_index;
        let Some(prog) = modules.get(m) else {
            continue;
        };
        let Some(pm) = parsed.get(m) else {
            continue;
        };
        let Some(f2d) = local_func_to_def.get(m) else {
            continue;
        };
        let Some(body_nodes) = prog.nodes.get(df.body_range.clone()) else {
            continue;
        };

        for frame in body_nodes {
            match frame.opcode {
                Opcode::Call => {
                    if let Some(&callee_def) = f2d.get(frame.in_a as usize) {
                        if seen_callees.insert(callee_def) {
                            if let Some(list) = adj.get_mut(callee_def) {
                                list.push(caller_def);
                            }
                            if let Some(deg) = in_degree.get_mut(caller_def) {
                                *deg = deg.saturating_add(1);
                            }
                        }
                    }
                }
                Opcode::CallIndirect => {
                    let (table_id_u32, _) = frame.decode_in_a_u16_pair();
                    if let Ok(table_id) = u16::try_from(table_id_u32) {
                        if seen_tables.insert(table_id) {
                            if let Some(ct) = pm.call_tables.get(usize::from(table_id)) {
                                let lo = ct.offset as usize;
                                let hi =
                                    lo.saturating_add((ct.entry_count as usize).saturating_mul(4));
                                if let Some(slice) = prog.const_data.get(lo..hi) {
                                    for chunk in slice.as_chunks::<4>().0 {
                                        let orig_fid = u32::from_le_bytes(*chunk);
                                        if let Some(&callee_def) = f2d.get(orig_fid as usize) {
                                            if seen_callees.insert(callee_def) {
                                                if let Some(list) = adj.get_mut(callee_def) {
                                                    list.push(caller_def);
                                                }
                                                if let Some(deg) = in_degree.get_mut(caller_def) {
                                                    *deg = deg.saturating_add(1);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // Deterministic min-heap Kahn topological sort keyed by (module_index, orig_func_id).
    // Give `entry_def_id` the maximal sort key so it always pops last whenever no cycle exists
    // and no other function depends on `entry_def_id`.
    let make_key = |def_id: usize| -> (usize, u16, usize) {
        if def_id == entry_def_id {
            (usize::MAX, u16::MAX, def_id)
        } else if let Some(df) = defined_funcs.get(def_id) {
            (df.module_index, df.orig_func_id, def_id)
        } else {
            (usize::MAX, u16::MAX, def_id)
        }
    };

    let mut heap: BinaryHeap<Reverse<(usize, u16, usize)>> = BinaryHeap::new();
    for (def_id, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            heap.push(Reverse(make_key(def_id)));
        }
    }

    let mut ordered = Vec::with_capacity(v_count);
    while let Some(Reverse((_, _, u))) = heap.pop() {
        ordered.push(u);
        if let Some(neighbors) = adj.get(u) {
            for &v in neighbors {
                if let Some(deg) = in_degree.get_mut(v) {
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        heap.push(Reverse(make_key(v)));
                    }
                }
            }
        }
    }

    if ordered.len() != v_count {
        return Err(LinkError::CycleWithoutDiverge);
    }
    if ordered.last().copied() != Some(entry_def_id) {
        return Err(LinkError::InvalidEntryFunction {
            reason: "entry function cannot be called by another function without CAP_DIVERGE",
        });
    }
    Ok(ordered)
}

fn relocate_node_proof(
    orig: ProofEntry,
    region_map: &[Option<u16>; MAX_REGION_SLOTS],
    seg_map: &[u16],
) -> Result<ProofEntry, LinkError> {
    let mut relocated = orig;
    match orig.rule_kind {
        ProofRuleKind::RegionIndexInBounds => {
            let new_slot = region_map
                .get(usize::from(orig.target_arena))
                .and_then(|s| *s)
                .ok_or(LinkError::ResourceLimitExceeded {
                    reason: "RegionIndexInBounds target slot missing from region relocation table",
                })?;
            relocated.target_arena = new_slot;
        }
        ProofRuleKind::ConstDataInBounds => {
            let new_seg = seg_map.get(usize::from(orig.target_arena)).copied().ok_or(
                LinkError::ResourceLimitExceeded {
                    reason: "ConstDataInBounds target seg_id missing from segment relocation table",
                },
            )?;
            relocated.target_arena = new_seg;
        }
        ProofRuleKind::SliceBounds
        | ProofRuleKind::GatherIndexInBounds
        | ProofRuleKind::BlockSig
        | ProofRuleKind::LoopBound => {}
        ProofRuleKind::FuncSig
        | ProofRuleKind::RegionDecl
        | ProofRuleKind::FuncRegionEffect
        | ProofRuleKind::ConstDataSeg
        | ProofRuleKind::BrTableSpec
        | ProofRuleKind::CallTable => {
            return Err(LinkError::ResourceLimitExceeded {
                reason: "node proof_idx referenced a non-node proof rule kind",
            });
        }
    }
    Ok(relocated)
}

/// Links one or more separately verified Rul modules into a single self-contained, verified `.rl` executable.
///
/// # Errors
/// Returns a typed [`LinkError`] if any module fails verification, requests capabilities above the
/// allowed ceiling, has unresolved/duplicate/mismatched symbols, exceeds region/proof/function bounds,
/// or forms a call cycle without [`CAP_DIVERGE`].
#[expect(
    clippy::too_many_lines,
    reason = "linear multi-stage module linker with explicit fail-closed validation and relocation"
)]
pub fn link_modules(
    modules: &[RulProgram],
    options: &LinkOptions,
) -> Result<LinkReport, LinkError> {
    if modules.is_empty() {
        return Err(LinkError::EmptyInput);
    }

    // Stage 1: Verify every input module via `verify_module` and parse its interface.
    let mut parsed: Vec<ParsedModule> = Vec::with_capacity(modules.len());
    for (mod_idx, prog) in modules.iter().enumerate() {
        parsed.push(parse_and_verify_module(mod_idx, prog)?);
    }

    // Stage 2: Build global export registry (rejecting duplicate exports).
    let mut func_exports: HashMap<String, (usize, u16)> = HashMap::new();
    let mut region_exports: HashMap<String, (usize, u16)> = HashMap::new();
    for (mod_idx, pm) in parsed.iter().enumerate() {
        for (name, local_id) in &pm.exported_funcs {
            if let Some(&(first_module, _)) = func_exports.get(name) {
                return Err(LinkError::DuplicateExport {
                    kind: LinkSymbolKind::Func,
                    name: name.clone(),
                    first_module,
                    second_module: mod_idx,
                });
            }
            func_exports.insert(name.clone(), (mod_idx, *local_id));
        }
        for (name, local_slot) in &pm.exported_regions {
            if let Some(&(first_module, _)) = region_exports.get(name) {
                return Err(LinkError::DuplicateExport {
                    kind: LinkSymbolKind::Region,
                    name: name.clone(),
                    first_module,
                    second_module: mod_idx,
                });
            }
            region_exports.insert(name.clone(), (mod_idx, *local_slot));
        }
    }

    // Stage 3: Select entry function and enforce zero silent capability escalation.
    let (entry_mod_idx, entry_orig_func_id) = if let Some(ref entry_name) = options.entry_export {
        func_exports
            .get(entry_name)
            .copied()
            .ok_or_else(|| LinkError::UnknownEntryExport {
                name: entry_name.clone(),
            })?
    } else {
        let last_mod = modules.len() - 1;
        let last_pm = parsed.get(last_mod).ok_or(LinkError::EmptyInput)?;
        let mut last_local_fid: Option<u16> = None;
        for (fid, imp) in last_pm.imported_funcs.iter().enumerate() {
            if imp.is_none() {
                last_local_fid = u16::try_from(fid).ok();
            }
        }
        let fid = last_local_fid.ok_or(LinkError::InvalidEntryFunction {
            reason: "entry module defines no local function",
        })?;
        (last_mod, fid)
    };

    let entry_module_cap = modules
        .get(entry_mod_idx)
        .map_or(0, |prog| prog.capability_mask);
    let final_cap = options.capability_mask.unwrap_or(entry_module_cap);
    for (mod_idx, prog) in modules.iter().enumerate() {
        if (prog.capability_mask & !final_cap) != 0 {
            return Err(LinkError::CapabilityEscalation {
                module_index: mod_idx,
                module_cap: prog.capability_mask,
                allowed_cap: final_cap,
            });
        }
    }

    // Validate entry function arity, return type, and Owned entry precondition.
    {
        let entry_pm = parsed
            .get(entry_mod_idx)
            .ok_or(LinkError::InvalidEntryFunction {
                reason: "entry module index out of bounds",
            })?;
        let entry_sig = entry_pm
            .func_sigs
            .get(usize::from(entry_orig_func_id))
            .copied()
            .ok_or(LinkError::InvalidEntryFunction {
                reason: "entry function signature missing",
            })?;
        if entry_sig.arity != 0 {
            return Err(LinkError::InvalidEntryFunction {
                reason: "entry function must have arity 0",
            });
        }
        if entry_sig.ret_elem.is_none() {
            return Err(LinkError::InvalidEntryFunction {
                reason: "entry function must have a non-void return type",
            });
        }
        let entry_contract = entry_pm
            .func_contracts
            .get(usize::from(entry_orig_func_id))
            .copied()
            .ok_or(LinkError::InvalidEntryFunction {
                reason: "entry function region contract missing",
            })?;
        if (entry_contract.pre_owned & entry_contract.dom_mask) != entry_contract.dom_mask
            || entry_contract.pre_leased != 0
        {
            return Err(LinkError::InvalidEntryFunction {
                reason:
                    "entry function precondition must be Owned for every region in its footprint",
            });
        }
    }

    // Stage 4: Allocate global region slots for non-imported regions and resolve imported regions.
    let mut region_maps: Vec<[Option<u16>; MAX_REGION_SLOTS]> =
        vec![[None; MAX_REGION_SLOTS]; modules.len()];
    let mut global_regions: Vec<RegionDeclInfo> = Vec::new();
    let mut has_input_region = false;
    let mut has_output_region = false;

    for (mod_idx, pm) in parsed.iter().enumerate() {
        for slot in 0..MAX_REGION_SLOTS {
            let Some(decl) = pm.regions.get(slot).and_then(|r| *r) else {
                continue;
            };
            if pm
                .imported_regions
                .get(slot)
                .and_then(Option::as_ref)
                .is_some()
            {
                continue;
            }
            if global_regions.len() >= MAX_REGION_SLOTS {
                return Err(LinkError::TooManyRegions {
                    needed: global_regions.len() + 1,
                    max: MAX_REGION_SLOTS,
                });
            }
            match decl.role {
                RegionRole::Plain => {}
                RegionRole::Input => {
                    if has_input_region {
                        return Err(LinkError::ConflictingIoRegion {
                            role: RegionRole::Input,
                        });
                    }
                    has_input_region = true;
                }
                RegionRole::Output => {
                    if has_output_region {
                        return Err(LinkError::ConflictingIoRegion {
                            role: RegionRole::Output,
                        });
                    }
                    has_output_region = true;
                }
            }
            let global_slot =
                u16::try_from(global_regions.len()).map_err(|_e| LinkError::TooManyRegions {
                    needed: global_regions.len() + 1,
                    max: MAX_REGION_SLOTS,
                })?;
            global_regions.push(RegionDeclInfo {
                slot: global_slot,
                min_len: decl.min_len,
                max_len: decl.max_len,
                elem: decl.elem,
                role: decl.role,
            });
            if let Some(rmap) = region_maps.get_mut(mod_idx) {
                if let Some(entry) = rmap.get_mut(slot) {
                    *entry = Some(global_slot);
                }
            }
        }
    }

    for (mod_idx, pm) in parsed.iter().enumerate() {
        for slot in 0..MAX_REGION_SLOTS {
            let Some(imp_name) = pm.imported_regions.get(slot).and_then(Option::as_ref) else {
                continue;
            };
            let local_decl =
                pm.regions
                    .get(slot)
                    .and_then(|r| *r)
                    .ok_or(LinkError::ResourceLimitExceeded {
                        reason: "imported region slot missing RegionDecl",
                    })?;
            let &(exp_mod, exp_slot) =
                region_exports
                    .get(imp_name)
                    .ok_or_else(|| LinkError::UnresolvedSymbol {
                        module_index: mod_idx,
                        kind: LinkSymbolKind::Region,
                        name: imp_name.clone(),
                    })?;
            let global_slot = region_maps
                .get(exp_mod)
                .and_then(|rmap| rmap.get(usize::from(exp_slot)))
                .and_then(|s| *s)
                .ok_or(LinkError::ResourceLimitExceeded {
                    reason: "exported region missing global slot allocation",
                })?;
            let exp_decl = global_regions
                .get(usize::from(global_slot))
                .copied()
                .ok_or(LinkError::ResourceLimitExceeded {
                    reason: "global region slot out of range",
                })?;
            if local_decl.elem != exp_decl.elem
                || local_decl.role != exp_decl.role
                || local_decl.min_len != exp_decl.min_len
                || local_decl.max_len != exp_decl.max_len
            {
                return Err(LinkError::RegionDeclMismatch {
                    name: imp_name.clone(),
                    importer_module: mod_idx,
                    exporter_module: exp_mod,
                });
            }
            if let Some(rmap) = region_maps.get_mut(mod_idx) {
                if let Some(entry) = rmap.get_mut(slot) {
                    *entry = Some(global_slot);
                }
            }
        }
    }

    // Stage 5: Collect locally-defined functions and resolve imported functions with exact sig & contract checks.
    let mut defined_funcs: Vec<DefinedFunc> = Vec::new();
    let mut local_func_to_def: Vec<Vec<usize>> = Vec::with_capacity(modules.len());
    for (mod_idx, pm) in parsed.iter().enumerate() {
        let mut map = vec![0usize; pm.func_sigs.len()];
        let rmap = region_maps
            .get(mod_idx)
            .ok_or(LinkError::ResourceLimitExceeded {
                reason: "missing region map",
            })?;
        for (fid, &sig) in pm.func_sigs.iter().enumerate() {
            if pm
                .imported_funcs
                .get(fid)
                .and_then(Option::as_ref)
                .is_some()
            {
                continue;
            }
            let body_range = pm
                .func_bodies
                .get(fid)
                .and_then(Option::as_ref)
                .cloned()
                .ok_or(LinkError::ResourceLimitExceeded {
                    reason: "locally-defined function missing Func body",
                })?;
            let local_contract =
                pm.func_contracts
                    .get(fid)
                    .copied()
                    .ok_or(LinkError::ResourceLimitExceeded {
                        reason: "locally-defined function missing FuncRegionContract",
                    })?;
            let remapped_contract = remap_contract(local_contract, rmap);
            let orig_func_id = u16::try_from(fid).map_err(|_e| LinkError::TooManyFunctions {
                needed: fid + 1,
                max: usize::from(u16::MAX) + 1,
            })?;
            let def_id = defined_funcs.len();
            defined_funcs.push(DefinedFunc {
                module_index: mod_idx,
                orig_func_id,
                sig,
                remapped_contract,
                body_range,
            });
            if let Some(slot) = map.get_mut(fid) {
                *slot = def_id;
            }
        }
        local_func_to_def.push(map);
    }

    if defined_funcs.len() > usize::from(u16::MAX) + 1 {
        return Err(LinkError::TooManyFunctions {
            needed: defined_funcs.len(),
            max: usize::from(u16::MAX) + 1,
        });
    }

    for (mod_idx, pm) in parsed.iter().enumerate() {
        let rmap = region_maps
            .get(mod_idx)
            .ok_or(LinkError::ResourceLimitExceeded {
                reason: "missing region map",
            })?;
        for (fid, imp_opt) in pm.imported_funcs.iter().enumerate() {
            let Some(ref imp_name) = imp_opt else {
                continue;
            };
            let &(exp_mod, exp_orig_id) =
                func_exports
                    .get(imp_name)
                    .ok_or_else(|| LinkError::UnresolvedSymbol {
                        module_index: mod_idx,
                        kind: LinkSymbolKind::Func,
                        name: imp_name.clone(),
                    })?;
            let exp_def_id = local_func_to_def
                .get(exp_mod)
                .and_then(|m| m.get(usize::from(exp_orig_id)))
                .copied()
                .ok_or(LinkError::ResourceLimitExceeded {
                    reason: "exported function missing definition index",
                })?;
            let exp_df = defined_funcs
                .get(exp_def_id)
                .ok_or(LinkError::ResourceLimitExceeded {
                    reason: "exported function definition out of range",
                })?;
            let imp_sig =
                pm.func_sigs
                    .get(fid)
                    .copied()
                    .ok_or(LinkError::ResourceLimitExceeded {
                        reason: "imported function missing FuncSig",
                    })?;
            if !func_sig_matches(imp_sig, exp_df.sig) {
                return Err(LinkError::FuncSigMismatch {
                    name: imp_name.clone(),
                    importer_module: mod_idx,
                    exporter_module: exp_mod,
                });
            }
            let imp_contract =
                pm.func_contracts
                    .get(fid)
                    .copied()
                    .ok_or(LinkError::ResourceLimitExceeded {
                        reason: "imported function missing FuncRegionContract",
                    })?;
            let imp_remapped = remap_contract(imp_contract, rmap);
            if imp_remapped != exp_df.remapped_contract {
                return Err(LinkError::FuncRegionEffectMismatch {
                    name: imp_name.clone(),
                    importer_module: mod_idx,
                    exporter_module: exp_mod,
                });
            }
            if let Some(m) = local_func_to_def.get_mut(mod_idx) {
                if let Some(slot) = m.get_mut(fid) {
                    *slot = exp_def_id;
                }
            }
        }
    }

    // Stage 6: Deterministically order linked functions with entry last (`total_funcs - 1`).
    let entry_def_id = local_func_to_def
        .get(entry_mod_idx)
        .and_then(|m| m.get(usize::from(entry_orig_func_id)))
        .copied()
        .ok_or(LinkError::InvalidEntryFunction {
            reason: "entry function definition not found",
        })?;
    let allow_diverge = (final_cap & CAP_DIVERGE) != 0;
    let ordered = order_functions_deterministic(
        modules,
        &parsed,
        &defined_funcs,
        &local_func_to_def,
        entry_def_id,
        allow_diverge,
    )?;

    let mut def_to_final: Vec<u16> = vec![0; defined_funcs.len()];
    for (final_idx, &def_id) in ordered.iter().enumerate() {
        let final_u16 = u16::try_from(final_idx).map_err(|_e| LinkError::TooManyFunctions {
            needed: ordered.len(),
            max: usize::from(u16::MAX) + 1,
        })?;
        if let Some(slot) = def_to_final.get_mut(def_id) {
            *slot = final_u16;
        }
    }

    let mut func_maps: Vec<Vec<u16>> = Vec::with_capacity(modules.len());
    for f2d in &local_func_to_def {
        let mut fmap = Vec::with_capacity(f2d.len());
        for &def_id in f2d {
            let final_id =
                def_to_final
                    .get(def_id)
                    .copied()
                    .ok_or(LinkError::ResourceLimitExceeded {
                        reason: "function mapping out of range",
                    })?;
            fmap.push(final_id);
        }
        func_maps.push(fmap);
    }

    // Stage 7: Concatenate and relocate Section 5a (`const_data`), `ConstDataSeg`, `BrTableSpec`, and `CallTable`.
    let mut linked_const_data: Vec<u8> = Vec::new();
    let mut linked_const_table_proofs: Vec<ProofEntry> = Vec::new();
    let mut seg_maps: Vec<Vec<u16>> = Vec::with_capacity(modules.len());
    let mut br_table_maps: Vec<Vec<u32>> = Vec::with_capacity(modules.len());
    let mut call_table_maps: Vec<Vec<u16>> = Vec::with_capacity(modules.len());
    let mut next_seg_id: u32 = 0;
    let mut next_br_table_id: u32 = 0;
    let mut next_call_table_id: u32 = 0;

    for (mod_idx, prog) in modules.iter().enumerate() {
        let pm = parsed
            .get(mod_idx)
            .ok_or(LinkError::ResourceLimitExceeded {
                reason: "missing parsed module",
            })?;
        let fmap = func_maps
            .get(mod_idx)
            .ok_or(LinkError::ResourceLimitExceeded {
                reason: "missing function relocation map",
            })?;
        let mut seg_map = vec![0u16; pm.seg_count];
        let mut br_map = vec![0u32; pm.br_table_count];
        let mut ct_map = vec![0u16; pm.call_tables.len()];

        for proof in &prog.proofs {
            match proof.rule_kind {
                ProofRuleKind::ConstDataSeg => {
                    let new_seg = u16::try_from(next_seg_id).map_err(|_e| {
                        LinkError::ResourceLimitExceeded {
                            reason: "combined ConstDataSeg count exceeds u16::MAX",
                        }
                    })?;
                    next_seg_id = next_seg_id.saturating_add(1);
                    if let Some(slot) = seg_map.get_mut(usize::from(proof.target_arena)) {
                        *slot = new_seg;
                    }
                    let new_off = u32::try_from(linked_const_data.len()).map_err(|_e| {
                        LinkError::ResourceLimitExceeded {
                            reason: "combined const_data exceeds u32::MAX bytes",
                        }
                    })?;
                    let lo = proof.bound_lo as usize;
                    let hi = lo.saturating_add(proof.bound_hi as usize);
                    let slice =
                        prog.const_data
                            .get(lo..hi)
                            .ok_or(LinkError::ResourceLimitExceeded {
                                reason: "ConstDataSeg slice out of bounds",
                            })?;
                    linked_const_data.extend_from_slice(slice);
                    if linked_const_data.len() > (MAX_CONST_DATA_BYTES as usize) {
                        return Err(LinkError::ResourceLimitExceeded {
                            reason: "combined const_data exceeds MAX_CONST_DATA_BYTES (8 MiB)",
                        });
                    }
                    linked_const_table_proofs.push(ProofEntry::const_data_seg(
                        new_seg,
                        new_off,
                        proof.bound_hi,
                    ));
                }
                ProofRuleKind::BrTableSpec => {
                    let new_br = next_br_table_id;
                    next_br_table_id = next_br_table_id.checked_add(1).ok_or(
                        LinkError::ResourceLimitExceeded {
                            reason: "combined BrTableSpec count exceeds u32::MAX",
                        },
                    )?;
                    if let Some(slot) = br_map.get_mut(proof.witness_node as usize) {
                        *slot = new_br;
                    }
                    let new_off = u32::try_from(linked_const_data.len()).map_err(|_e| {
                        LinkError::ResourceLimitExceeded {
                            reason: "combined const_data exceeds u32::MAX bytes",
                        }
                    })?;
                    let lo = proof.bound_lo as usize;
                    let hi = lo.saturating_add((proof.bound_hi as usize).saturating_mul(4));
                    let slice =
                        prog.const_data
                            .get(lo..hi)
                            .ok_or(LinkError::ResourceLimitExceeded {
                                reason: "BrTableSpec slice out of bounds",
                            })?;
                    linked_const_data.extend_from_slice(slice);
                    if linked_const_data.len() > (MAX_CONST_DATA_BYTES as usize) {
                        return Err(LinkError::ResourceLimitExceeded {
                            reason: "combined const_data exceeds MAX_CONST_DATA_BYTES (8 MiB)",
                        });
                    }
                    linked_const_table_proofs.push(ProofEntry::br_table_spec(
                        new_br,
                        proof.target_arena,
                        new_off,
                        proof.bound_hi,
                    ));
                }
                ProofRuleKind::CallTable => {
                    let new_ct = u16::try_from(next_call_table_id).map_err(|_e| {
                        LinkError::ResourceLimitExceeded {
                            reason: "combined CallTable count exceeds u16::MAX",
                        }
                    })?;
                    next_call_table_id = next_call_table_id.saturating_add(1);
                    if let Some(slot) = ct_map.get_mut(usize::from(proof.target_arena)) {
                        *slot = new_ct;
                    }
                    let new_off = u32::try_from(linked_const_data.len()).map_err(|_e| {
                        LinkError::ResourceLimitExceeded {
                            reason: "combined const_data exceeds u32::MAX bytes",
                        }
                    })?;
                    let lo = proof.bound_lo as usize;
                    let hi = lo.saturating_add((proof.bound_hi as usize).saturating_mul(4));
                    let slice =
                        prog.const_data
                            .get(lo..hi)
                            .ok_or(LinkError::ResourceLimitExceeded {
                                reason: "CallTable slice out of bounds",
                            })?;
                    for chunk in slice.as_chunks::<4>().0 {
                        let orig_fid = u32::from_le_bytes(*chunk);
                        let new_fid = fmap.get(orig_fid as usize).copied().ok_or(
                            LinkError::ResourceLimitExceeded {
                                reason: "CallTable entry func_id missing from relocation map",
                            },
                        )?;
                        linked_const_data.extend_from_slice(&u32::from(new_fid).to_le_bytes());
                    }
                    if linked_const_data.len() > (MAX_CONST_DATA_BYTES as usize) {
                        return Err(LinkError::ResourceLimitExceeded {
                            reason: "combined const_data exceeds MAX_CONST_DATA_BYTES (8 MiB)",
                        });
                    }
                    linked_const_table_proofs.push(ProofEntry::call_table(
                        new_ct,
                        new_off,
                        proof.bound_hi,
                    ));
                }
                _ => {}
            }
        }
        seg_maps.push(seg_map);
        br_table_maps.push(br_map);
        call_table_maps.push(ct_map);
    }

    // Stage 8: Relocate function bodies in deterministic order, deduplicating node-referenced proofs.
    let mut node_proofs: Vec<ProofEntry> = Vec::new();
    let mut node_proof_dedup: HashMap<[u8; 16], usize> = HashMap::new();
    let mut linked_nodes: Vec<NodeFrame> = Vec::new();
    let mut entry_start_node: u32 = 0;

    for (final_idx, &def_id) in ordered.iter().enumerate() {
        let df = defined_funcs
            .get(def_id)
            .ok_or(LinkError::ResourceLimitExceeded {
                reason: "ordered function index out of range",
            })?;
        let m = df.module_index;
        let prog = modules.get(m).ok_or(LinkError::ResourceLimitExceeded {
            reason: "module index out of range",
        })?;
        let rmap = region_maps.get(m).ok_or(LinkError::ResourceLimitExceeded {
            reason: "region map missing",
        })?;
        let fmap = func_maps.get(m).ok_or(LinkError::ResourceLimitExceeded {
            reason: "func map missing",
        })?;
        let smap = seg_maps.get(m).ok_or(LinkError::ResourceLimitExceeded {
            reason: "seg map missing",
        })?;
        let bmap = br_table_maps
            .get(m)
            .ok_or(LinkError::ResourceLimitExceeded {
                reason: "br_table map missing",
            })?;
        let ctmap = call_table_maps
            .get(m)
            .ok_or(LinkError::ResourceLimitExceeded {
                reason: "call_table map missing",
            })?;

        let final_func_id = u32::try_from(final_idx).map_err(|_e| LinkError::TooManyFunctions {
            needed: ordered.len(),
            max: usize::from(u16::MAX) + 1,
        })?;
        if def_id == entry_def_id {
            entry_start_node = u32::try_from(linked_nodes.len()).map_err(|_e| {
                LinkError::ResourceLimitExceeded {
                    reason: "linked node count exceeds u32::MAX",
                }
            })?;
        }

        let body_nodes =
            prog.nodes
                .get(df.body_range.clone())
                .ok_or(LinkError::ResourceLimitExceeded {
                    reason: "function body slice out of bounds",
                })?;

        for &orig_frame in body_nodes {
            let mut frame = orig_frame;
            match frame.opcode {
                Opcode::Func => {
                    frame.in_a = final_func_id;
                }
                Opcode::Call => {
                    let new_fid = fmap.get(frame.in_a as usize).copied().ok_or(
                        LinkError::ResourceLimitExceeded {
                            reason: "Call target func_id missing from relocation map",
                        },
                    )?;
                    frame.in_a = u32::from(new_fid);
                }
                Opcode::CallIndirect => {
                    let (orig_tid, idx_node) = frame.decode_in_a_u16_pair();
                    let new_tid = ctmap.get(orig_tid as usize).copied().ok_or(
                        LinkError::ResourceLimitExceeded {
                            reason: "CallIndirect table_id missing from relocation map",
                        },
                    )?;
                    let idx_u16 =
                        u16::try_from(idx_node).map_err(|_e| LinkError::ResourceLimitExceeded {
                            reason: "CallIndirect index node exceeds u16::MAX",
                        })?;
                    frame.in_a = NodeFrame::encode_u16_pair(new_tid, idx_u16);
                }
                Opcode::BrTable => {
                    let new_br = bmap.get(frame.in_b as usize).copied().ok_or(
                        LinkError::ResourceLimitExceeded {
                            reason: "BrTable br_table_id missing from relocation map",
                        },
                    )?;
                    frame.in_b = new_br;
                }
                Opcode::ConstDataLen | Opcode::ConstDataLoad => {
                    let new_seg = smap.get(frame.in_a as usize).copied().ok_or(
                        LinkError::ResourceLimitExceeded {
                            reason: "ConstData seg_id missing from relocation map",
                        },
                    )?;
                    frame.in_a = u32::from(new_seg);
                }
                Opcode::RegionNew
                | Opcode::RegionGrow
                | Opcode::RegionLen
                | Opcode::RegionLoad
                | Opcode::RegionStore
                | Opcode::RegionFree
                | Opcode::RegionLease
                | Opcode::RegionEndLease
                | Opcode::RegionDrop => {
                    let new_slot = rmap.get(frame.in_a as usize).and_then(|s| *s).ok_or(
                        LinkError::ResourceLimitExceeded {
                            reason: "Region opcode slot missing from relocation map",
                        },
                    )?;
                    frame.in_a = u32::from(new_slot);
                }
                Opcode::RegionMove => {
                    let new_src = rmap.get(frame.in_a as usize).and_then(|s| *s).ok_or(
                        LinkError::ResourceLimitExceeded {
                            reason: "RegionMove source slot missing from relocation map",
                        },
                    )?;
                    let new_dst = rmap.get(frame.in_b as usize).and_then(|s| *s).ok_or(
                        LinkError::ResourceLimitExceeded {
                            reason: "RegionMove dest slot missing from relocation map",
                        },
                    )?;
                    frame.in_a = u32::from(new_src);
                    frame.in_b = u32::from(new_dst);
                }
                _ => {}
            }

            if frame.proof_idx != 0 {
                let orig_proof = prog
                    .proofs
                    .get(usize::from(frame.proof_idx - 1))
                    .copied()
                    .ok_or(LinkError::ResourceLimitExceeded {
                        reason: "node proof_idx out of bounds in source module",
                    })?;
                let relocated_proof = relocate_node_proof(orig_proof, rmap, smap)?;
                let key = relocated_proof.to_bytes();
                let assigned_idx = if let Some(&existing) = node_proof_dedup.get(&key) {
                    existing
                } else {
                    node_proofs.push(relocated_proof);
                    let idx = node_proofs.len();
                    node_proof_dedup.insert(key, idx);
                    idx
                };
                if let Ok(idx_u8) = u8::try_from(assigned_idx) {
                    frame.proof_idx = idx_u8;
                }
            }

            linked_nodes.push(frame);
            if linked_nodes.len() > (MAX_SECTION_ITEMS as usize) {
                return Err(LinkError::ResourceLimitExceeded {
                    reason: "combined node count exceeds MAX_SECTION_ITEMS",
                });
            }
        }
    }

    // Fail closed if deduplicated node-referenced proofs exceed the 1-based u8 ceiling (255).
    if node_proofs.len() > MAX_NODE_PROOF_INDEX {
        return Err(LinkError::ProofIndexOverflow {
            needed: node_proofs.len(),
            max: MAX_NODE_PROOF_INDEX,
        });
    }

    // Stage 9: Assemble final proof table:
    //   [0..M): deduplicated node-referenced proofs (M <= 255)
    //   [M..):  RegionDecl, FuncSig, materialized FuncRegionEffect, ConstDataSeg/BrTableSpec/CallTable
    let mut linked_proofs: Vec<ProofEntry> = Vec::with_capacity(
        node_proofs.len()
            + global_regions.len()
            + ordered.len() * 2
            + linked_const_table_proofs.len(),
    );
    linked_proofs.extend_from_slice(&node_proofs);

    for decl in &global_regions {
        linked_proofs.push(ProofEntry::region_decl(
            decl.slot,
            decl.min_len,
            decl.max_len,
            decl.elem,
            decl.role,
        ));
    }

    for (final_idx, &def_id) in ordered.iter().enumerate() {
        let df = defined_funcs
            .get(def_id)
            .ok_or(LinkError::ResourceLimitExceeded {
                reason: "ordered function index out of range",
            })?;
        let final_u16 = u16::try_from(final_idx).map_err(|_e| LinkError::TooManyFunctions {
            needed: ordered.len(),
            max: usize::from(u16::MAX) + 1,
        })?;
        let param_slice = df
            .sig
            .params
            .get(..usize::from(df.sig.arity))
            .unwrap_or(&[]);
        linked_proofs.push(ProofEntry::func_sig(
            final_u16,
            param_slice,
            df.sig.ret_elem,
        ));
    }

    for (final_idx, &def_id) in ordered.iter().enumerate() {
        let df = defined_funcs
            .get(def_id)
            .ok_or(LinkError::ResourceLimitExceeded {
                reason: "ordered function index out of range",
            })?;
        let final_u16 = u16::try_from(final_idx).map_err(|_e| LinkError::TooManyFunctions {
            needed: ordered.len(),
            max: usize::from(u16::MAX) + 1,
        })?;
        let c = df.remapped_contract;
        if c.dom_mask == 0 {
            linked_proofs.push(ProofEntry::func_region_effect(
                final_u16,
                None,
                RegionState::Owned,
                RegionState::Owned,
            ));
        } else {
            let mut dom = c.dom_mask;
            while dom != 0 {
                let slot_u32 = dom.trailing_zeros();
                dom &= dom - 1;
                let bit = 1u64 << slot_u32;
                let pre = if (c.pre_owned & bit) != 0 {
                    RegionState::Owned
                } else if (c.pre_leased & bit) != 0 {
                    RegionState::Leased
                } else {
                    RegionState::Moved
                };
                let post = if (c.post_owned & bit) != 0 {
                    RegionState::Owned
                } else if (c.post_leased & bit) != 0 {
                    RegionState::Leased
                } else {
                    RegionState::Moved
                };
                let slot_u16 = u16::try_from(slot_u32).unwrap_or(0);
                linked_proofs.push(ProofEntry::func_region_effect(
                    final_u16,
                    Some(slot_u16),
                    pre,
                    post,
                ));
            }
        }
    }

    linked_proofs.extend_from_slice(&linked_const_table_proofs);

    if linked_proofs.len() > (MAX_SECTION_ITEMS as usize) {
        return Err(LinkError::ResourceLimitExceeded {
            reason: "combined proof count exceeds MAX_SECTION_ITEMS",
        });
    }

    // Stage 10: Construct self-contained executable with stripped imports and verify via `verify_program`.
    let const_data_bytes = linked_const_data.len();
    let linked_program = RulProgram {
        capability_mask: final_cap,
        arenas: Vec::new(),
        proofs: linked_proofs,
        nodes: linked_nodes,
        output_node: entry_start_node,
        const_data: linked_const_data,
        link_meta: Vec::new(),
    };

    let summary = verify_program(&linked_program).map_err(LinkError::LinkedVerifyFailed)?;

    Ok(LinkReport {
        program: linked_program,
        summary,
        module_count: modules.len(),
        func_count: ordered.len(),
        region_count: global_regions.len(),
        const_data_bytes,
        capability_mask: final_cap,
    })
}
