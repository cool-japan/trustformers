//! Real Shamir secret sharing over GF(2^8).
//!
//! # What was here before
//!
//! The previous "Shamir's secret sharing" computed `share_i = value + i * 0.1`
//! per `f32` — every share revealed the secret to within 0.1, and
//! "reconstruction" simply returned share 0. Both are covered by regression
//! tests below.
//!
//! # What this is
//!
//! Textbook Shamir over the field GF(2^8) with the AES reduction polynomial
//! `x^8 + x^4 + x^3 + x + 1` (0x11B), operating byte-wise on the secret:
//!
//! - **Split**: for each secret byte `s`, sample a random polynomial
//!   `f(z) = s + a₁z + … + a_{k-1}z^{k-1}` with uniform coefficients, and give
//!   party `i` the evaluation `f(i)` for a distinct non-zero `i ∈ GF(2^8)`.
//! - **Reconstruct**: Lagrange-interpolate at `z = 0` from any `k` shares.
//!
//! # Properties
//!
//! - Any `k` shares reconstruct the secret exactly.
//! - Any `k - 1` shares are *information-theoretically* independent of the
//!   secret: for every candidate secret there is exactly one polynomial
//!   consistent with those shares, so the posterior equals the prior. The
//!   [`threshold_minus_one_reveals_nothing`](self) test checks this by
//!   exhaustively confirming that `k-1` shares extend to every possible secret
//!   byte.
//! - Because the field is GF(2^8), the number of parties is capped at 255.
//!
//! Field arithmetic uses branch-free table-free routines (`xtime`-style
//! multiplication), so it does not leak the operands through data-dependent
//! table lookups.

use rand_core::CryptoRng;
use std::collections::BTreeSet;
use trustformers_core::errors::{invalid_input, Result};
use zeroize::Zeroize;

/// Maximum number of shares: GF(2^8) has 255 distinct non-zero evaluation points.
pub const MAX_SHARES: usize = 255;

/// One party's share of a secret.
#[derive(Clone, PartialEq, Eq)]
pub struct Share {
    /// The evaluation point `i ∈ [1, 255]`. Distinct across shares.
    index: u8,
    /// `f(index)` for each byte of the secret.
    values: Vec<u8>,
}

impl Drop for Share {
    fn drop(&mut self) {
        self.values.zeroize();
    }
}

impl std::fmt::Debug for Share {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Share values are secret-adjacent; print only the shape.
        f.debug_struct("Share")
            .field("index", &self.index)
            .field("len", &self.values.len())
            .finish()
    }
}

impl Share {
    /// The evaluation point of this share.
    pub fn index(&self) -> u8 {
        self.index
    }

    /// The share payload, one byte per secret byte.
    pub fn values(&self) -> &[u8] {
        &self.values
    }

    /// Serialize as `index || values`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(1 + self.values.len());
        out.push(self.index);
        out.extend_from_slice(&self.values);
        out
    }

    /// Parse the encoding produced by [`Share::to_bytes`].
    ///
    /// # Errors
    /// Returns an error for an empty input or a zero index (index 0 would be
    /// the secret itself, so it is never a valid share).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let (&index, values) = bytes
            .split_first()
            .ok_or_else(|| invalid_input("Shamir share is empty".to_string()))?;
        if index == 0 {
            return Err(invalid_input(
                "Shamir share index 0 is reserved for the secret".to_string(),
            ));
        }
        Ok(Self {
            index,
            values: values.to_vec(),
        })
    }
}

/// Split `secret` into `num_shares` shares, any `threshold` of which
/// reconstruct it.
///
/// # Errors
/// Returns an error if `threshold` is 0 or 1 (1 would hand out the secret),
/// if `threshold > num_shares`, or if `num_shares` exceeds [`MAX_SHARES`].
pub fn split_with_rng<R: CryptoRng>(
    secret: &[u8],
    num_shares: usize,
    threshold: usize,
    rng: &mut R,
) -> Result<Vec<Share>> {
    if secret.is_empty() {
        return Err(invalid_input("Shamir secret must be non-empty".to_string()));
    }
    if threshold < 2 {
        return Err(invalid_input(format!(
            "Shamir threshold must be at least 2 (a threshold of {threshold} would reveal the \
             secret to a single share holder)"
        )));
    }
    if threshold > num_shares {
        return Err(invalid_input(format!(
            "Shamir threshold {threshold} exceeds the share count {num_shares}"
        )));
    }
    if num_shares > MAX_SHARES {
        return Err(invalid_input(format!(
            "Shamir over GF(2^8) supports at most {MAX_SHARES} shares, requested {num_shares}"
        )));
    }

    // Coefficients a_1..a_{k-1} for every byte position, drawn uniformly.
    let degree = threshold - 1;
    let mut coefficients = vec![0u8; secret.len() * degree];
    rng.fill_bytes(&mut coefficients);

    let mut shares = Vec::with_capacity(num_shares);
    for share_index in 1..=num_shares {
        // usize -> u8 is in range because num_shares <= 255.
        let x = u8::try_from(share_index)
            .map_err(|_| invalid_input("Shamir share index overflowed GF(2^8)".to_string()))?;
        let mut values = Vec::with_capacity(secret.len());
        for (byte_position, &secret_byte) in secret.iter().enumerate() {
            // Horner evaluation of f(x) = s + a_1 x + ... + a_{k-1} x^{k-1}.
            let mut accumulator = 0u8;
            for coefficient_index in (0..degree).rev() {
                let coefficient = coefficients[byte_position * degree + coefficient_index];
                accumulator = gf_add(gf_mul(accumulator, x), coefficient);
            }
            values.push(gf_add(gf_mul(accumulator, x), secret_byte));
        }
        shares.push(Share { index: x, values });
    }

    coefficients.zeroize();
    Ok(shares)
}

/// Split using the operating-system CSPRNG.
///
/// # Errors
/// See [`split_with_rng`].
pub fn split(secret: &[u8], num_shares: usize, threshold: usize) -> Result<Vec<Share>> {
    let mut rng = rand_core::UnwrapErr(getrandom::SysRng);
    split_with_rng(secret, num_shares, threshold, &mut rng)
}

/// Reconstruct the secret from `shares` by Lagrange interpolation at `z = 0`.
///
/// The caller must supply at least `threshold` shares; supplying fewer yields a
/// value unrelated to the secret, so this function requires the threshold to be
/// stated and enforces it.
///
/// # Errors
/// Returns an error when fewer than `threshold` shares are supplied, when share
/// indices repeat, when a share index is 0, or when share lengths disagree.
pub fn reconstruct(shares: &[Share], threshold: usize) -> Result<Vec<u8>> {
    if threshold < 2 {
        return Err(invalid_input(format!(
            "Shamir threshold must be at least 2, got {threshold}"
        )));
    }
    if shares.len() < threshold {
        return Err(invalid_input(format!(
            "Shamir reconstruction needs at least {threshold} shares, got {}",
            shares.len()
        )));
    }

    let mut seen = BTreeSet::new();
    for share in shares {
        if share.index == 0 {
            return Err(invalid_input("Shamir share index 0 is invalid".to_string()));
        }
        if !seen.insert(share.index) {
            return Err(invalid_input(format!(
                "Shamir shares must have distinct indices; index {} appears twice",
                share.index
            )));
        }
    }

    let secret_len = shares[0].values.len();
    if shares.iter().any(|s| s.values.len() != secret_len) {
        return Err(invalid_input(
            "Shamir shares have inconsistent lengths".to_string(),
        ));
    }

    // Only the first `threshold` shares are needed; using exactly that many
    // makes the cost independent of how many extra shares were supplied.
    let used = &shares[..threshold];

    let mut secret = vec![0u8; secret_len];
    for (i, share_i) in used.iter().enumerate() {
        // Lagrange basis polynomial evaluated at 0:
        //   L_i(0) = prod_{j != i} x_j / (x_j - x_i), and in GF(2^n) subtraction
        //   is XOR, so (x_j - x_i) == (x_j ^ x_i).
        let mut basis = 1u8;
        for (j, share_j) in used.iter().enumerate() {
            if i == j {
                continue;
            }
            let denominator = gf_add(share_j.index, share_i.index);
            // Distinct indices were checked above, so `denominator != 0`.
            let inverse = gf_inv(denominator).ok_or_else(|| {
                invalid_input("Shamir interpolation hit a zero denominator".to_string())
            })?;
            basis = gf_mul(basis, gf_mul(share_j.index, inverse));
        }
        for (byte_position, secret_byte) in secret.iter_mut().enumerate() {
            *secret_byte = gf_add(*secret_byte, gf_mul(basis, share_i.values[byte_position]));
        }
    }

    Ok(secret)
}

// ─── GF(2^8) arithmetic (AES field, reduction polynomial 0x11B) ──────────────

/// Addition in GF(2^8) is XOR.
#[inline]
fn gf_add(a: u8, b: u8) -> u8 {
    a ^ b
}

/// Branch-free carry-less multiplication with reduction by 0x11B.
///
/// The loop runs a fixed 8 iterations and uses arithmetic masks rather than
/// branches or table lookups, so its timing does not depend on the operands.
#[inline]
fn gf_mul(a: u8, b: u8) -> u8 {
    let mut product = 0u8;
    let mut lhs = a;
    let mut rhs = b;
    for _ in 0..8 {
        // mask = 0xFF when the low bit of rhs is set, else 0x00.
        let mask = 0u8.wrapping_sub(rhs & 1);
        product ^= lhs & mask;
        // high = 0xFF when the high bit of lhs is set, else 0x00.
        let high = 0u8.wrapping_sub((lhs >> 7) & 1);
        lhs <<= 1;
        lhs ^= 0x1B & high;
        rhs >>= 1;
    }
    product
}

/// Multiplicative inverse via `a^254 = a^-1` in GF(2^8), by square-and-multiply.
///
/// Returns `None` for 0, which has no inverse.
#[inline]
fn gf_inv(a: u8) -> Option<u8> {
    if a == 0 {
        return None;
    }
    // a^254 = a^(2+4+8+16+32+64+128)
    let mut result = 1u8;
    let mut power = a;
    // exponent 254 = 0b1111_1110
    for bit in 1..8 {
        power = gf_mul(power, power);
        if (254 >> bit) & 1 == 1 {
            result = gf_mul(result, power);
        }
    }
    // Handle bit 0 of the exponent (which is 0 for 254) explicitly for clarity:
    // nothing to multiply.
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advanced_security::test_rng::TestRng;
    use std::collections::BTreeSet;

    #[test]
    fn gf_field_axioms_hold() {
        // Multiplicative identity.
        for a in 0u8..=255 {
            assert_eq!(gf_mul(a, 1), a);
            assert_eq!(gf_mul(1, a), a);
            assert_eq!(gf_mul(a, 0), 0);
        }
        // Inverses.
        for a in 1u8..=255 {
            let inv = gf_inv(a).expect("non-zero has an inverse");
            assert_eq!(gf_mul(a, inv), 1, "{a} * inv({a}) should be 1");
        }
        assert_eq!(gf_inv(0), None);
        // Commutativity and distributivity on a sample.
        for a in [0u8, 1, 2, 3, 0x53, 0xCA, 0xFF] {
            for b in [0u8, 1, 7, 0x1B, 0x80, 0xFE] {
                assert_eq!(gf_mul(a, b), gf_mul(b, a));
                for c in [0u8, 5, 0x11, 0x9D] {
                    assert_eq!(gf_mul(a, gf_add(b, c)), gf_add(gf_mul(a, b), gf_mul(a, c)));
                }
            }
        }
    }

    /// The AES test vector: 0x57 * 0x83 = 0xC1.
    #[test]
    fn gf_mul_matches_the_aes_reference_vector() {
        assert_eq!(gf_mul(0x57, 0x83), 0xC1);
        assert_eq!(gf_mul(0x57, 0x13), 0xFE);
    }

    #[test]
    fn split_and_reconstruct_exactly() {
        let mut rng = TestRng::seeded(1);
        let secret = b"the quick brown fox jumps over the lazy dog";
        let shares = split_with_rng(secret, 5, 3, &mut rng).expect("split");
        assert_eq!(shares.len(), 5);
        let recovered = reconstruct(&shares[..3], 3).expect("reconstruct");
        assert_eq!(&recovered[..], secret);
    }

    /// Every subset of exactly `threshold` shares must give the same answer.
    #[test]
    fn any_threshold_subset_reconstructs() {
        let mut rng = TestRng::seeded(2);
        let secret = b"aggregate me";
        let shares = split_with_rng(secret, 5, 3, &mut rng).expect("split");

        for i in 0..shares.len() {
            for j in (i + 1)..shares.len() {
                for k in (j + 1)..shares.len() {
                    let subset = vec![shares[i].clone(), shares[j].clone(), shares[k].clone()];
                    let recovered = reconstruct(&subset, 3).expect("reconstruct");
                    assert_eq!(&recovered[..], secret, "subset ({i},{j},{k}) failed");
                }
            }
        }
    }

    /// Regression for `share_value = value + i * 0.1`: no share may equal or
    /// closely track the secret.
    #[test]
    fn shares_do_not_resemble_the_secret() {
        let mut rng = TestRng::seeded(3);
        let secret = vec![0x42u8; 64];
        let shares = split_with_rng(&secret, 5, 3, &mut rng).expect("split");
        for share in &shares {
            assert_ne!(share.values(), &secret[..], "a share equals the secret");
            // The old scheme produced shares within a constant of the secret;
            // a real share of a constant secret must not itself be constant.
            assert!(
                share.values().windows(2).any(|w| w[0] != w[1]),
                "share of a constant secret must not be constant"
            );
        }
    }

    /// Information-theoretic security of `k-1` shares: given `threshold - 1`
    /// shares, *every* value of a secret byte remains consistent, so those
    /// shares carry zero information about it.
    #[test]
    fn threshold_minus_one_reveals_nothing() {
        let mut rng = TestRng::seeded(4);
        let threshold = 3usize;
        let secret = [0xA5u8];
        let shares = split_with_rng(&secret, 5, threshold, &mut rng).expect("split");

        // Take k-1 = 2 shares and, for each candidate secret byte, solve for the
        // unique third share that would be consistent. If a consistent
        // completion exists for all 256 candidates, the two shares alone tell us
        // nothing.
        let known = [shares[0].clone(), shares[1].clone()];
        let mut reachable = BTreeSet::new();
        // Enumerate all possible values of a hypothetical third share at index 3.
        for candidate_value in 0u8..=255 {
            let hypothetical = Share {
                index: 3,
                values: vec![candidate_value],
            };
            let trio = vec![known[0].clone(), known[1].clone(), hypothetical];
            let recovered = reconstruct(&trio, threshold).expect("reconstruct");
            reachable.insert(recovered[0]);
        }
        assert_eq!(
            reachable.len(),
            256,
            "two shares of a 3-of-n split must be consistent with every secret byte"
        );
    }

    /// Two shares of a 3-of-5 split must not reconstruct the secret when the
    /// caller (incorrectly) tries.
    #[test]
    fn fewer_than_threshold_shares_are_refused() {
        let mut rng = TestRng::seeded(5);
        let secret = b"top secret";
        let shares = split_with_rng(secret, 5, 3, &mut rng).expect("split");
        assert!(
            reconstruct(&shares[..2], 3).is_err(),
            "reconstruction below the threshold must be refused, not silently wrong"
        );
    }

    #[test]
    fn duplicate_indices_are_refused() {
        let mut rng = TestRng::seeded(6);
        let shares = split_with_rng(b"abc", 5, 3, &mut rng).expect("split");
        let dup = vec![shares[0].clone(), shares[0].clone(), shares[1].clone()];
        assert!(reconstruct(&dup, 3).is_err());
    }

    #[test]
    fn inconsistent_share_lengths_are_refused() {
        let mut rng = TestRng::seeded(7);
        let mut shares = split_with_rng(b"abcdef", 4, 3, &mut rng).expect("split");
        shares[2].values.truncate(2);
        assert!(reconstruct(&shares[..3], 3).is_err());
    }

    #[test]
    fn split_rejects_degenerate_parameters() {
        let mut rng = TestRng::seeded(8);
        assert!(split_with_rng(b"", 3, 2, &mut rng).is_err(), "empty secret");
        assert!(split_with_rng(b"x", 3, 1, &mut rng).is_err(), "threshold 1");
        assert!(split_with_rng(b"x", 3, 0, &mut rng).is_err(), "threshold 0");
        assert!(
            split_with_rng(b"x", 2, 3, &mut rng).is_err(),
            "threshold > shares"
        );
        assert!(
            split_with_rng(b"x", 256, 2, &mut rng).is_err(),
            "too many shares"
        );
    }

    #[test]
    fn share_serialization_round_trip() {
        let mut rng = TestRng::seeded(9);
        let secret = b"round trip";
        let shares = split_with_rng(secret, 4, 2, &mut rng).expect("split");
        let restored: Vec<Share> = shares
            .iter()
            .map(|s| Share::from_bytes(&s.to_bytes()).expect("parse"))
            .collect();
        assert_eq!(
            &reconstruct(&restored[..2], 2).expect("reconstruct")[..],
            secret
        );
    }

    #[test]
    fn share_parsing_rejects_bad_input() {
        assert!(Share::from_bytes(&[]).is_err(), "empty");
        assert!(Share::from_bytes(&[0, 1, 2]).is_err(), "index 0");
    }

    /// A 2-of-2 split (the additive-sharing special case) and the maximum
    /// supported party count both work.
    #[test]
    fn boundary_thresholds_work() {
        let mut rng = TestRng::seeded(10);
        let secret = b"boundaries";
        let two_of_two = split_with_rng(secret, 2, 2, &mut rng).expect("2-of-2");
        assert_eq!(&reconstruct(&two_of_two, 2).expect("r")[..], secret);

        let many = split_with_rng(secret, MAX_SHARES, 2, &mut rng).expect("255 shares");
        assert_eq!(many.len(), MAX_SHARES);
        assert_eq!(&reconstruct(&many[..2], 2).expect("r")[..], secret);
        // Indices are distinct and non-zero.
        let indices: BTreeSet<u8> = many.iter().map(|s| s.index()).collect();
        assert_eq!(indices.len(), MAX_SHARES);
        assert!(!indices.contains(&0));
    }

    /// Splitting the same secret twice must give different shares (fresh
    /// polynomial coefficients).
    #[test]
    fn splits_are_randomized() {
        let secret = b"same secret";
        let mut rng_a = TestRng::seeded(11);
        let mut rng_b = TestRng::seeded(12);
        let a = split_with_rng(secret, 3, 2, &mut rng_a).expect("a");
        let b = split_with_rng(secret, 3, 2, &mut rng_b).expect("b");
        assert_ne!(a[0].values(), b[0].values(), "splits must be randomized");
        assert_eq!(&reconstruct(&a[..2], 2).expect("ra")[..], secret);
        assert_eq!(&reconstruct(&b[..2], 2).expect("rb")[..], secret);
    }
}
