//! A small, self-verifying binary delta format ("TFDELTA1") used by
//! [`crate::hub`]'s delta-compressed model download path.
//!
//! An earlier revision "applied" a delta by XOR-ing the delta bytes over the
//! base file and writing whatever came out — a base/delta pair that didn't
//! line up produced a silently corrupted model file with no error and no
//! checksum check. This module replaces that with a real block-copy/insert
//! codec:
//!
//! ```text
//! magic:         8 bytes   b"TFDELTA1"
//! base_sha256:   32 bytes  SHA-256 of the base file this delta applies to
//! target_sha256: 32 bytes  SHA-256 of the file this delta reconstructs
//! target_len:    u64 LE    length of the reconstructed file
//! op_count:      u64 LE    number of ops that follow
//! ops:           op_count entries, each:
//!   tag == 0 (Copy):   base_offset: u64 LE, len: u64 LE
//!   tag == 1 (Insert): len: u64 LE, then `len` raw bytes
//! ```
//!
//! [`apply_delta`] refuses to apply a delta whose `base_sha256` doesn't match
//! the base bytes it was given — a mismatch there means every `Copy` offset
//! below would silently read the wrong file — and refuses to return a
//! reconstructed buffer whose `target_sha256` doesn't match what the delta
//! itself declares. Both checks are unconditional: there is no way to obtain
//! reconstructed bytes from this module without them having been verified
//! against the delta's own embedded checksums.

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

const DELTA_MAGIC: &[u8; 8] = b"TFDELTA1";
/// Block size used when indexing the base file for matches. Not part of the
/// wire format — a decoder never needs to know how the encoder chose its
/// copies, only how to replay them.
const BLOCK_SIZE: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
enum DeltaOp {
    Copy { base_offset: u64, len: u64 },
    Insert { bytes: Vec<u8> },
}

/// Non-cryptographic content hash used only to index candidate matching
/// blocks in-memory during encoding. Every candidate is byte-compared before
/// being accepted, so a hash collision can only cost a little performance,
/// never correctness.
fn block_hash(block: &[u8]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    block.hash(&mut hasher);
    hasher.finish()
}

/// Encode a delta that reconstructs `target` from `base`.
///
/// Always round-trips through [`apply_delta`] regardless of how similar
/// `base` and `target` are: in the worst case (no shared 64-byte blocks) the
/// whole target is carried as a single `Insert` op, which is still a valid,
/// correctly self-describing TFDELTA1 file — just not a compact one.
pub fn encode_delta(base: &[u8], target: &[u8]) -> Vec<u8> {
    let mut base_blocks: HashMap<u64, Vec<usize>> = HashMap::new();
    let mut offset = 0usize;
    while offset + BLOCK_SIZE <= base.len() {
        let hash = block_hash(&base[offset..offset + BLOCK_SIZE]);
        base_blocks.entry(hash).or_default().push(offset);
        offset += BLOCK_SIZE;
    }

    let mut ops: Vec<DeltaOp> = Vec::new();
    let mut pending_insert: Vec<u8> = Vec::new();
    let mut pos = 0usize;

    while pos < target.len() {
        let best_match = if pos + BLOCK_SIZE <= target.len() {
            find_best_match(base, target, pos, &base_blocks)
        } else {
            None
        };

        match best_match {
            Some((base_offset, len)) if len > 0 => {
                if !pending_insert.is_empty() {
                    ops.push(DeltaOp::Insert {
                        bytes: std::mem::take(&mut pending_insert),
                    });
                }
                ops.push(DeltaOp::Copy {
                    base_offset: base_offset as u64,
                    len: len as u64,
                });
                pos += len;
            },
            _ => {
                pending_insert.push(target[pos]);
                pos += 1;
            },
        }
    }
    if !pending_insert.is_empty() {
        ops.push(DeltaOp::Insert {
            bytes: pending_insert,
        });
    }

    serialize(base, target, &ops)
}

/// Among every base block whose hash matches the target block starting at
/// `pos`, pick the one that extends (via direct byte comparison) into the
/// longest verified run, to both reject hash collisions and maximize the
/// resulting `Copy` op's length.
fn find_best_match(
    base: &[u8],
    target: &[u8],
    pos: usize,
    base_blocks: &HashMap<u64, Vec<usize>>,
) -> Option<(usize, usize)> {
    let hash = block_hash(&target[pos..pos + BLOCK_SIZE]);
    let candidates = base_blocks.get(&hash)?;

    let mut best: Option<(usize, usize)> = None;
    for &base_offset in candidates {
        if base[base_offset..base_offset + BLOCK_SIZE] != target[pos..pos + BLOCK_SIZE] {
            continue; // hash collision, not a genuine match
        }
        let max_len = (base.len() - base_offset).min(target.len() - pos);
        let mut len = 0;
        while len < max_len && base[base_offset + len] == target[pos + len] {
            len += 1;
        }
        if best.is_none_or(|(_, best_len)| len > best_len) {
            best = Some((base_offset, len));
        }
    }
    best
}

fn serialize(base: &[u8], target: &[u8], ops: &[DeltaOp]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(DELTA_MAGIC);
    out.extend_from_slice(Sha256::digest(base).as_slice());
    out.extend_from_slice(Sha256::digest(target).as_slice());
    out.extend_from_slice(&(target.len() as u64).to_le_bytes());
    out.extend_from_slice(&(ops.len() as u64).to_le_bytes());
    for op in ops {
        match op {
            DeltaOp::Copy { base_offset, len } => {
                out.push(0u8);
                out.extend_from_slice(&base_offset.to_le_bytes());
                out.extend_from_slice(&len.to_le_bytes());
            },
            DeltaOp::Insert { bytes } => {
                out.push(1u8);
                out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
                out.extend_from_slice(bytes);
            },
        }
    }
    out
}

struct ByteCursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ByteCursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(n).ok_or("delta offset overflow")?;
        let slice = self.data.get(self.pos..end).ok_or("delta file is truncated")?;
        self.pos = end;
        Ok(slice)
    }

    fn take_u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn take_u64(&mut self) -> Result<u64, String> {
        let bytes = self.take(8)?;
        let mut buf = [0u8; 8];
        buf.copy_from_slice(bytes);
        Ok(u64::from_le_bytes(buf))
    }

    fn take_array32(&mut self) -> Result<[u8; 32], String> {
        let bytes = self.take(32)?;
        let mut buf = [0u8; 32];
        buf.copy_from_slice(bytes);
        Ok(buf)
    }
}

struct ParsedDelta {
    base_sha256: [u8; 32],
    target_sha256: [u8; 32],
    target_len: u64,
    ops: Vec<DeltaOp>,
}

fn parse(delta: &[u8]) -> Result<ParsedDelta, String> {
    let mut cursor = ByteCursor::new(delta);

    let magic = cursor.take(DELTA_MAGIC.len())?;
    if magic != DELTA_MAGIC {
        return Err(
            "not a TFDELTA1 delta file (bad magic) — refusing to treat arbitrary bytes as a delta"
                .to_string(),
        );
    }

    let base_sha256 = cursor.take_array32()?;
    let target_sha256 = cursor.take_array32()?;
    let target_len = cursor.take_u64()?;
    let op_count = cursor.take_u64()?;

    // Cap the eagerly-allocated capacity regardless of what a corrupt/hostile
    // header claims; each iteration below still validates against the actual
    // remaining bytes via `ByteCursor::take`.
    let mut ops = Vec::with_capacity(op_count.min(1_000_000) as usize);
    for _ in 0..op_count {
        let tag = cursor.take_u8()?;
        match tag {
            0 => {
                let base_offset = cursor.take_u64()?;
                let len = cursor.take_u64()?;
                ops.push(DeltaOp::Copy { base_offset, len });
            },
            1 => {
                let len = cursor.take_u64()?;
                let len_usize =
                    usize::try_from(len).map_err(|_| "insert op length too large".to_string())?;
                let bytes = cursor.take(len_usize)?.to_vec();
                ops.push(DeltaOp::Insert { bytes });
            },
            other => return Err(format!("unknown delta op tag {other}")),
        }
    }

    Ok(ParsedDelta {
        base_sha256,
        target_sha256,
        target_len,
        ops,
    })
}

fn replay(base: &[u8], target_len: u64, ops: &[DeltaOp]) -> Result<Vec<u8>, String> {
    let capacity = usize::try_from(target_len).unwrap_or(0);
    let mut out = Vec::with_capacity(capacity);
    for op in ops {
        match op {
            DeltaOp::Copy { base_offset, len } => {
                let start = usize::try_from(*base_offset)
                    .map_err(|_| "copy offset too large".to_string())?;
                let length =
                    usize::try_from(*len).map_err(|_| "copy length too large".to_string())?;
                let end = start.checked_add(length).ok_or("copy range overflow")?;
                let slice =
                    base.get(start..end).ok_or("copy op reads past the end of the base file")?;
                out.extend_from_slice(slice);
            },
            DeltaOp::Insert { bytes } => out.extend_from_slice(bytes),
        }
    }
    if out.len() as u64 != target_len {
        return Err(format!(
            "delta replayed to {} bytes but the header declares target_len {target_len}",
            out.len()
        ));
    }
    Ok(out)
}

/// Apply a TFDELTA1 `delta` to `base`, returning the reconstructed target
/// bytes, or a descriptive error string.
///
/// The base checksum is verified before any op is applied; the target
/// checksum is verified after. A caller that only ever accepts `Ok(_)` from
/// this function cannot end up with corrupted bytes: either both checks pass,
/// or it gets an `Err` and nothing else.
pub fn apply_delta(base: &[u8], delta: &[u8]) -> Result<Vec<u8>, String> {
    let parsed = parse(delta)?;

    let base_actual = Sha256::digest(base);
    if base_actual.as_slice() != parsed.base_sha256.as_slice() {
        return Err(format!(
            "delta base checksum mismatch: delta expects base sha256 {}, but the supplied base \
             file hashes to {} — this delta was not generated against this file",
            hex::encode(parsed.base_sha256),
            hex::encode(base_actual)
        ));
    }

    let target = replay(base, parsed.target_len, &parsed.ops)?;

    let target_actual = Sha256::digest(&target);
    if target_actual.as_slice() != parsed.target_sha256.as_slice() {
        return Err(format!(
            "delta reconstruction checksum mismatch: expected sha256 {}, got {} — refusing to \
             return a corrupted result",
            hex::encode(parsed.target_sha256),
            hex::encode(target_actual)
        ));
    }

    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn sha256_hex(data: &[u8]) -> String {
        hex::encode(Sha256::digest(data))
    }

    #[test]
    fn round_trips_identical_base_and_target() {
        let base = b"the quick brown fox jumps over the lazy dog".repeat(4);
        let delta = encode_delta(&base, &base);
        let out = apply_delta(&base, &delta).expect("apply_delta");
        assert_eq!(out, base);
    }

    #[test]
    fn round_trips_empty_base_and_target() {
        let delta = encode_delta(&[], &[]);
        let out = apply_delta(&[], &delta).expect("apply_delta");
        assert!(out.is_empty());
    }

    #[test]
    fn round_trips_empty_base_nonempty_target() {
        let target = b"brand new content with no base to copy from".to_vec();
        let delta = encode_delta(&[], &target);
        let out = apply_delta(&[], &delta).expect("apply_delta");
        assert_eq!(out, target);
    }

    #[test]
    fn round_trips_a_small_edit() {
        let base: Vec<u8> = (0..2048u32).map(|i| (i % 251) as u8).collect();
        let mut target = base.clone();
        // Insert a few bytes in the middle and change a handful of others,
        // simulating a small model update.
        target.splice(1000..1000, [9u8, 9, 9, 9].iter().copied());
        for b in target.iter_mut().take(50) {
            *b = b.wrapping_add(1);
        }

        let delta = encode_delta(&base, &target);
        // A real edit should produce a delta far smaller than storing the
        // whole target again, proving actual block matching happened rather
        // than a single big Insert.
        assert!(
            delta.len() < target.len(),
            "delta ({} bytes) should be smaller than the full target ({} bytes)",
            delta.len(),
            target.len()
        );

        let out = apply_delta(&base, &delta).expect("apply_delta");
        assert_eq!(out, target);
    }

    #[test]
    fn rejects_files_without_the_magic() {
        let err = apply_delta(b"base", b"not a delta file at all").expect_err("bad magic");
        assert!(err.contains("magic"), "{err}");
    }

    #[test]
    fn rejects_truncated_delta() {
        let base = b"some base content".to_vec();
        let target = b"some other target content, longer than the base".to_vec();
        let mut delta = encode_delta(&base, &target);
        delta.truncate(delta.len() / 2);
        let err = apply_delta(&base, &delta).expect_err("truncated delta");
        assert!(err.contains("truncated"), "{err}");
    }

    #[test]
    fn rejects_a_base_that_does_not_match_the_delta() {
        let base = b"the original base file".to_vec();
        let target = b"the reconstructed target file".to_vec();
        let delta = encode_delta(&base, &target);

        let wrong_base = b"a completely different base file!".to_vec();
        let err = apply_delta(&wrong_base, &delta).expect_err("wrong base must be rejected");
        assert!(err.contains("base checksum mismatch"), "{err}");
    }

    /// Regression test for the historical bug: XOR-ing an unrelated delta
    /// over a base file always "succeeds" and produces garbage. This format
    /// must refuse instead.
    #[test]
    fn never_silently_returns_corrupted_bytes_for_mismatched_inputs() {
        let base_a = vec![0xABu8; 512];
        let target_a = vec![0xCDu8; 512];
        let delta_a = encode_delta(&base_a, &target_a);

        let base_b = vec![0x12u8; 512];
        // Applying a delta built for (base_a -> target_a) against base_b must
        // error, not produce 512 bytes of silently wrong data.
        let result = apply_delta(&base_b, &delta_a);
        assert!(result.is_err(), "mismatched base must never apply silently");
    }

    #[test]
    fn rejects_a_corrupted_op_stream() {
        let base = b"base content for corruption test".to_vec();
        let target = b"target content for corruption test, a bit longer".to_vec();
        let mut delta = encode_delta(&base, &target);
        // Flip a byte inside the op stream (well past the 80-byte fixed
        // header) without touching the embedded checksums, so any corruption
        // here must be caught either by a malformed-op parse error or by the
        // post-apply target checksum — never silently accepted.
        let flip_at = delta.len() - 1;
        delta[flip_at] ^= 0xFF;
        let result = apply_delta(&base, &delta);
        assert!(
            result.is_err(),
            "a corrupted op stream must never apply silently"
        );
    }

    proptest! {
        /// Property: for *any* base/target byte strings, encoding then
        /// applying always reconstructs `target` exactly.
        #[test]
        fn prop_round_trip_any_bytes(
            base in proptest::collection::vec(any::<u8>(), 0..2048),
            target in proptest::collection::vec(any::<u8>(), 0..2048),
        ) {
            let delta = encode_delta(&base, &target);
            let out = apply_delta(&base, &delta).expect("apply_delta must succeed for a delta encode_delta just produced");
            prop_assert_eq!(out, target);
        }

        /// Property: a delta encoded for one (base, target) pair, when applied
        /// against a *different* base, either errors out or (in the
        /// vanishingly unlikely case of an actual SHA-256 collision) still
        /// reproduces the original target — it must never silently produce
        /// some third, wrong result.
        #[test]
        fn prop_wrong_base_never_applies_to_silently_wrong_bytes(
            base in proptest::collection::vec(any::<u8>(), 1..512),
            target in proptest::collection::vec(any::<u8>(), 0..512),
            other_base in proptest::collection::vec(any::<u8>(), 1..512),
        ) {
            prop_assume!(base != other_base);
            let delta = encode_delta(&base, &target);
            match apply_delta(&other_base, &delta) {
                Err(_) => {},
                Ok(out) => {
                    // Only reachable if `other_base` happens to hash identically
                    // to `base`, which would itself be a SHA-256 collision.
                    prop_assert_eq!(sha256_hex(&base), sha256_hex(&other_base));
                    prop_assert_eq!(out, target);
                },
            }
        }
    }
}
