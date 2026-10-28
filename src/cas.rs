//! Content-Addressed Store (`rul-cas`) & Multi-Type Verified Intrinsics (`I64`, `U8`, `F64`).
//!
//! Replaces traditional source-code package managers with a zero-dependency,
//! content-addressed registry of verified 64-bit hashes and high-throughput
//! multi-type columnar operators (`SortI64Asc`, `HashJoinSumI64`, `GroupByBucketSumI64`,
//! `SimdScanCountByteU8`).

use std::collections::BTreeMap;
use std::fmt;

use crate::bytecode::RulProgram;
use crate::verifier::{verify_program, VerifiedSummary, VerifyDiagnostic};

const FNV_OFFSET_BASIS_64: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME_64: u64 = 0x0100_0000_01b3;

/// Deterministic 64-bit content-addressed identifier for a verified `.rl` graph or intrinsic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CasHash(pub u64);

impl CasHash {
    /// Computes the FNV-1a 64-bit hash of an arbitrary byte slice (e.g., an intrinsic name).
    #[must_use]
    pub const fn from_bytes(mut bytes: &[u8]) -> Self {
        let mut hash = FNV_OFFSET_BASIS_64;
        while let [first, rest @ ..] = bytes {
            hash ^= *first as u64;
            hash = hash.wrapping_mul(FNV_PRIME_64);
            bytes = rest;
        }
        Self(hash)
    }

    /// Computes the deterministic 64-bit hash of a canonical intrinsic symbol name.
    #[must_use]
    pub const fn from_name(name: &str) -> Self {
        Self::from_bytes(name.as_bytes())
    }

    /// Computes the deterministic 64-bit content hash of a `.rl` binary program.
    #[must_use]
    pub fn from_program(program: &RulProgram) -> Self {
        Self(program.compute_graph_hash())
    }
}

impl fmt::Display for CasHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:016x}", self.0)
    }
}

/// Built-in verified multi-type columnar and byte-stream intrinsics (`I64`, `U8`, `F64`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntrinsicKind {
    /// Sort a columnar `i64` slice in ascending order (`std::sort_i64_asc`).
    SortI64Asc,
    /// Equi-join `keys_left: &[i64]` (with `vals_left: &[i64]`) against `keys_right: &[i64]`
    /// and accumulate the joined sum (`std::hash_join_sum_i64`).
    HashJoinSumI64,
    /// Group `vals: &[i64]` by `keys: &[i64]` modulo bucket count and sum each bucket
    /// (`std::group_by_bucket_sum_i64`).
    GroupByBucketSumI64,
    /// Scan a raw `u8` byte slice (`&[u8]`) and count occurrences of a target byte
    /// (`std::simd_scan_byte_u8`).
    SimdScanCountByteU8,
    /// Compute the deterministic 64-bit FNV-1a digest of a `u8` byte slice
    /// (`std::fnv1a_hash_u8`).
    Fnv1aHashU8,
}

/// Metadata for a content-addressed verified intrinsic in `rul-cas`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntrinsicDescriptor {
    /// Deterministic 64-bit content-addressed hash.
    pub hash: CasHash,
    /// Canonical symbol name (e.g. `"std::hash_join_sum_i64"`).
    pub name: &'static str,
    /// Type signature in Rul columnar notation.
    pub signature: &'static str,
    /// Underlying intrinsic implementation kind.
    pub kind: IntrinsicKind,
}

const BUILTIN_INTRINSICS: [IntrinsicDescriptor; 5] = [
    IntrinsicDescriptor {
        hash: CasHash::from_name("std::sort_i64_asc"),
        name: "std::sort_i64_asc",
        signature: "(col: i64x8) -> i64x8",
        kind: IntrinsicKind::SortI64Asc,
    },
    IntrinsicDescriptor {
        hash: CasHash::from_name("std::hash_join_sum_i64"),
        name: "std::hash_join_sum_i64",
        signature: "(keys_l: i64x8, vals_l: i64x8, keys_r: i64x8) -> i64x1",
        kind: IntrinsicKind::HashJoinSumI64,
    },
    IntrinsicDescriptor {
        hash: CasHash::from_name("std::group_by_bucket_sum_i64"),
        name: "std::group_by_bucket_sum_i64",
        signature: "(keys: i64x8, vals: i64x8, buckets: u32) -> i64x8",
        kind: IntrinsicKind::GroupByBucketSumI64,
    },
    IntrinsicDescriptor {
        hash: CasHash::from_name("std::simd_scan_byte_u8"),
        name: "std::simd_scan_byte_u8",
        signature: "(bytes: u8x64, needle: u8) -> i64x1",
        kind: IntrinsicKind::SimdScanCountByteU8,
    },
    IntrinsicDescriptor {
        hash: CasHash::from_name("std::fnv1a_hash_u8"),
        name: "std::fnv1a_hash_u8",
        signature: "(bytes: u8x64) -> u64x1",
        kind: IntrinsicKind::Fnv1aHashU8,
    },
];

/// Returns the static catalog of built-in content-addressed intrinsics.
#[must_use]
pub const fn builtin_intrinsics() -> &'static [IntrinsicDescriptor] {
    &BUILTIN_INTRINSICS
}

/// Looks up a built-in intrinsic by its 64-bit [`CasHash`].
#[must_use]
pub fn lookup_intrinsic_by_hash(hash: CasHash) -> Option<IntrinsicDescriptor> {
    BUILTIN_INTRINSICS.iter().copied().find(|d| d.hash == hash)
}

/// Looks up a built-in intrinsic by its canonical name (e.g., `"std::hash_join_sum_i64"`).
#[must_use]
pub fn lookup_intrinsic_by_name(name: &str) -> Option<IntrinsicDescriptor> {
    let target = CasHash::from_name(name);
    lookup_intrinsic_by_hash(target)
}

/// Executes `std::sort_i64_asc` on an `i64` column, returning a sorted column.
#[must_use]
pub fn exec_sort_i64_asc(input: &[i64]) -> Vec<i64> {
    let mut out = input.to_vec();
    out.sort_unstable();
    out
}

/// Executes `std::hash_join_sum_i64` (`SELECT SUM(l.val) FROM left l JOIN right r ON l.key = r.key`)
/// using a cache-friendly open-addressed power-of-two hash table with zero external dependencies.
#[must_use]
pub fn exec_hash_join_sum_i64(keys_left: &[i64], vals_left: &[i64], keys_right: &[i64]) -> i64 {
    let n = keys_left.len().min(vals_left.len());
    if n == 0 || keys_right.is_empty() {
        return 0;
    }

    let cap = (n.saturating_mul(2)).next_power_of_two().max(16);
    let mask = cap - 1;
    let mut occupied = vec![false; cap];
    let mut slot_keys = vec![0_i64; cap];
    let mut slot_sums = vec![0_i64; cap];

    for (&k, &v) in keys_left.iter().zip(vals_left.iter()).take(n) {
        let mut idx = (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) as usize & mask;
        for _ in 0..cap {
            let is_occ = occupied.get(idx).copied().unwrap_or(false);
            if !is_occ {
                if let Some(occ_ref) = occupied.get_mut(idx) {
                    *occ_ref = true;
                }
                if let Some(k_ref) = slot_keys.get_mut(idx) {
                    *k_ref = k;
                }
                if let Some(s_ref) = slot_sums.get_mut(idx) {
                    *s_ref = v;
                }
                break;
            } else if slot_keys.get(idx).copied() == Some(k) {
                if let Some(s_ref) = slot_sums.get_mut(idx) {
                    *s_ref = s_ref.wrapping_add(v);
                }
                break;
            }
            idx = (idx + 1) & mask;
        }
    }

    let mut total_joined_sum: i64 = 0;
    for &probe_key in keys_right {
        let mut idx = (probe_key as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) as usize & mask;
        for _ in 0..cap {
            let is_occ = occupied.get(idx).copied().unwrap_or(false);
            if !is_occ {
                break;
            }
            if slot_keys.get(idx).copied() == Some(probe_key) {
                let matched_val = slot_sums.get(idx).copied().unwrap_or(0);
                total_joined_sum = total_joined_sum.wrapping_add(matched_val);
                break;
            }
            idx = (idx + 1) & mask;
        }
    }

    total_joined_sum
}

/// Executes `std::group_by_bucket_sum_i64`, aggregating `vals` into `bucket_count` buckets
/// indexed by `key.rem_euclid(bucket_count)`.
#[must_use]
pub fn exec_group_by_bucket_sum_i64(keys: &[i64], vals: &[i64], bucket_count: usize) -> Vec<i64> {
    let buckets = bucket_count.clamp(1, 65_536);
    let mut agg = vec![0_i64; buckets];
    let buckets_i64 = i64::try_from(buckets).unwrap_or(1);

    for (&k, &v) in keys.iter().zip(vals.iter()) {
        let b = usize::try_from(k.rem_euclid(buckets_i64)).unwrap_or(0);
        if let Some(slot) = agg.get_mut(b) {
            *slot = slot.wrapping_add(v);
        }
    }
    agg
}

/// Executes `std::simd_scan_byte_u8` over a `u8` byte slice (`U8x64` 64-byte chunks),
/// counting occurrences of `needle`.
#[must_use]
pub fn exec_simd_scan_count_byte_u8(bytes: &[u8], needle: u8) -> i64 {
    let mut count: i64 = 0;
    for chunk in bytes.chunks(64) {
        let mut local: i64 = 0;
        for &b in chunk {
            local += i64::from(b == needle);
        }
        count = count.wrapping_add(local);
    }
    count
}

/// Entry stored inside [`RulCasStore`], pairing a verified `.rl` program with its static summary.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedCasEntry {
    /// Content hash of the `.rl` binary graph.
    pub hash: CasHash,
    /// Verified `.rl` program.
    pub program: RulProgram,
    /// Static verification certificate summary.
    pub summary: VerifiedSummary,
}

/// In-memory Content-Addressed Store (`rul-cas`) that only admits statically verified `.rl` graphs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RulCasStore {
    entries: BTreeMap<CasHash, VerifiedCasEntry>,
}

impl RulCasStore {
    /// Creates an empty content-addressed store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Verifies `program` in $O(N)$ and inserts it into the content-addressed store under its
    /// 64-bit graph hash.
    ///
    /// # Errors
    /// Returns [`VerifyDiagnostic`] if the program fails linear verification.
    pub fn insert_verified(&mut self, program: RulProgram) -> Result<CasHash, VerifyDiagnostic> {
        let summary = verify_program(&program)?;
        let hash = CasHash::from_program(&program);
        self.entries.insert(
            hash,
            VerifiedCasEntry {
                hash,
                program,
                summary,
            },
        );
        Ok(hash)
    }

    /// Retrieves a verified `.rl` entry by its 64-bit [`CasHash`].
    #[must_use]
    pub fn get(&self, hash: CasHash) -> Option<&VerifiedCasEntry> {
        self.entries.get(&hash)
    }

    /// Returns the number of verified `.rl` graphs stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no `.rl` graphs are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
