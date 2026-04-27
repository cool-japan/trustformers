//! Additional Model Merging Algorithms
//!
//! Provides standalone, composable merge functions for combining fine-tuned model
//! weight vectors. These functions operate on flat `f32` slices and are independent
//! of the higher-level `ModelMerger` / `ModelWeights` abstractions in the parent module.
//!
//! # Algorithms
//!
//! - **DARE** (Drop And REscale): randomly drops a fraction of the delta weights and
//!   optionally rescales the survivors.  Uses a deterministic FNV-1a based pseudo-random
//!   drop decision so results are reproducible without any external PRNG dependency.
//!
//! - **TIES** (Trim, Elect Sign, disjoint Merge): trims the smallest-magnitude deltas,
//!   elects a consensus sign per parameter, then averages only the contributions that
//!   agree with the elected sign.
//!
//! - **Linear** (weighted average): straightforward weighted sum of model weight vectors.
//!
//! - **SLERP** (Spherical Linear Interpolation): geodesic interpolation between two
//!   weight vectors treated as directions in parameter space.
//!
//! # References
//!
//! - DARE: Yu et al. 2023 <https://arxiv.org/abs/2311.03099>
//! - TIES: Yadav et al. 2023 <https://arxiv.org/abs/2306.01708>
//! - SLERP: Shoemake 1985 (applied to neural network weights)

use std::fmt;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors produced by the merge algorithms in this module.
#[derive(Debug, Clone, PartialEq)]
pub enum MergeError {
    /// Two weight vectors (or a vector and a base) have different lengths.
    DimensionMismatch {
        /// Length of the first operand.
        a: usize,
        /// Length of the second operand.
        b: usize,
    },
    /// The provided weight vector does not sum to ~1.0 (for linear merge).
    InvalidWeights(String),
    /// No models were provided for a multi-model merge.
    EmptyModels,
    /// `trim_fraction` is outside [0, 1).
    TrimFractionOutOfRange,
}

impl fmt::Display for MergeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MergeError::DimensionMismatch { a, b } => {
                write!(f, "Dimension mismatch: {} vs {}", a, b)
            }
            MergeError::InvalidWeights(msg) => {
                write!(f, "Invalid weights: {}", msg)
            }
            MergeError::EmptyModels => write!(f, "No models provided for merge"),
            MergeError::TrimFractionOutOfRange => {
                write!(f, "trim_fraction must be in [0, 1)")
            }
        }
    }
}

impl std::error::Error for MergeError {}

// ─── FNV-1a pseudo-random helpers ─────────────────────────────────────────────

/// FNV-1a 64-bit hash of a u64 seed combined with an index.
/// Used as a cheap, dependency-free pseudo-random number generator.
#[inline]
fn fnv1a_mix(seed: u64, index: usize) -> u64 {
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;

    let mut hash = FNV_OFFSET;
    // Fold seed bytes
    for &byte in &seed.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    // Fold index bytes
    for &byte in &(index as u64).to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Deterministically decide whether parameter at `index` should be dropped.
///
/// Returns `true` if the parameter should be dropped (zeroed out).
/// `drop_rate` is in [0, 1).  Uses the FNV hash so that for any fixed
/// (seed, index) pair the answer is always the same.
#[inline]
fn should_drop(seed: u64, index: usize, drop_rate: f32) -> bool {
    if drop_rate <= 0.0 {
        return false;
    }
    if drop_rate >= 1.0 {
        return true;
    }
    let h = fnv1a_mix(seed, index);
    // Map to [0, 1)
    let unit = (h >> 11) as f64 / (1u64 << 53) as f64;
    unit < drop_rate as f64
}

// ─── DARE ─────────────────────────────────────────────────────────────────────

/// Configuration for DARE (Drop And REscale) model merging.
#[derive(Debug, Clone)]
pub struct DareMergeConfig {
    /// Fraction of delta weights to drop (set to 0).  Must be in [0, 1).
    pub drop_rate: f32,
    /// When `true`, surviving deltas are rescaled by `1 / (1 - drop_rate)` so
    /// the expected value is preserved.
    pub rescale: bool,
    /// Seed for deterministic parameter dropping via FNV hash.
    pub seed: u64,
    /// Weight applied to the sparse (post-drop) delta when adding to base.
    /// `result = base + merge_coefficient * sparse_delta`
    pub merge_coefficient: f32,
}

impl Default for DareMergeConfig {
    fn default() -> Self {
        Self {
            drop_rate: 0.9,
            rescale: true,
            seed: 0,
            merge_coefficient: 0.5,
        }
    }
}

/// Merge a fine-tuned model into a base model using DARE.
///
/// Steps:
/// 1. Compute `delta = finetuned - base`.
/// 2. Randomly drop `drop_rate` fraction of delta parameters (set to 0),
///    using a deterministic FNV-based decision keyed on `(config.seed, index)`.
/// 3. Optionally rescale remaining deltas by `1 / (1 - drop_rate)`.
/// 4. Return `base + config.merge_coefficient * sparse_delta`.
///
/// # Errors
///
/// Returns [`MergeError::DimensionMismatch`] when `base` and `finetuned` differ in length.
pub fn dare_merge(
    base: &[f32],
    finetuned: &[f32],
    config: &DareMergeConfig,
) -> Result<Vec<f32>, MergeError> {
    if base.len() != finetuned.len() {
        return Err(MergeError::DimensionMismatch {
            a: base.len(),
            b: finetuned.len(),
        });
    }

    let rescale_factor = if config.rescale && config.drop_rate > 0.0 && config.drop_rate < 1.0 {
        1.0 / (1.0 - config.drop_rate)
    } else {
        1.0
    };

    let result: Vec<f32> = base
        .iter()
        .zip(finetuned.iter())
        .enumerate()
        .map(|(i, (&b, &ft))| {
            let raw_delta = ft - b;
            let sparse_delta = if should_drop(config.seed, i, config.drop_rate) {
                0.0
            } else {
                raw_delta * rescale_factor
            };
            b + config.merge_coefficient * sparse_delta
        })
        .collect();

    Ok(result)
}

// ─── TIES ─────────────────────────────────────────────────────────────────────

/// Sign election method used in the TIES disjoint-merge step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TiesElectMethod {
    /// Elect the sign held by the majority of models.
    MajoritySign,
    /// Elect the sign of the model with the greatest absolute delta magnitude
    /// at each position.
    GreaterMagnitude,
}

impl Default for TiesElectMethod {
    fn default() -> Self {
        TiesElectMethod::MajoritySign
    }
}

/// Configuration for TIES (Trim, Elect Sign, disjoint Merge) model merging.
#[derive(Debug, Clone)]
pub struct TiesMergeConfig {
    /// Fraction of smallest-magnitude delta elements to trim to 0.
    /// Must be in [0, 1).  E.g. `0.2` trims the bottom 20 % by |delta|.
    pub trim_fraction: f32,
    /// How to elect the consensus sign at each parameter position.
    pub elect_method: TiesElectMethod,
    /// Scaling coefficient applied to each model's contribution in the merge step.
    pub merge_coefficient: f32,
}

impl Default for TiesMergeConfig {
    fn default() -> Self {
        Self {
            trim_fraction: 0.2,
            elect_method: TiesElectMethod::default(),
            merge_coefficient: 1.0,
        }
    }
}

/// Merge multiple fine-tuned models using TIES.
///
/// Steps:
/// 1. Compute `delta_i = models[i] - base` for each model.
/// 2. **Trim**: zero out the `trim_fraction` smallest-|delta| elements per model.
/// 3. **Elect sign**: determine a consensus sign per parameter position.
/// 4. **Disjoint merge**: average only the contributions whose sign agrees.
/// 5. Return `base + merge_coefficient * merged_delta`.
///
/// # Errors
///
/// Returns [`MergeError::EmptyModels`] when `models` is empty.
/// Returns [`MergeError::DimensionMismatch`] when any model length differs from `base`.
/// Returns [`MergeError::TrimFractionOutOfRange`] when `config.trim_fraction >= 1.0`.
pub fn ties_merge(
    base: &[f32],
    models: &[&[f32]],
    config: &TiesMergeConfig,
) -> Result<Vec<f32>, MergeError> {
    if models.is_empty() {
        return Err(MergeError::EmptyModels);
    }
    if config.trim_fraction >= 1.0 || config.trim_fraction < 0.0 {
        return Err(MergeError::TrimFractionOutOfRange);
    }

    let param_len = base.len();
    for model in models {
        if model.len() != param_len {
            return Err(MergeError::DimensionMismatch {
                a: param_len,
                b: model.len(),
            });
        }
    }

    // 1. Compute deltas
    let deltas: Vec<Vec<f32>> = models
        .iter()
        .map(|m| m.iter().zip(base.iter()).map(|(mi, bi)| mi - bi).collect())
        .collect();

    // 2. Trim: zero out the trim_fraction smallest-|delta| per model
    let trimmed: Vec<Vec<f32>> = deltas
        .iter()
        .map(|d| trim_delta(d, config.trim_fraction))
        .collect();

    // 3. Elect sign
    let elected_signs: Vec<f32> = match config.elect_method {
        TiesElectMethod::MajoritySign => elect_majority_sign(&trimmed, param_len),
        TiesElectMethod::GreaterMagnitude => elect_greater_magnitude_sign(&trimmed, param_len),
    };

    // 4. Disjoint merge: average contributions agreeing with elected sign
    let mut merged_delta = vec![0.0f32; param_len];
    for i in 0..param_len {
        let sign = elected_signs[i];
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for tv in &trimmed {
            let v = tv[i];
            // A contribution "agrees" if it has the same sign (non-zero, same direction)
            if (sign > 0.0 && v > 0.0) || (sign < 0.0 && v < 0.0) {
                sum += v;
                count += 1;
            }
        }
        merged_delta[i] = if count > 0 { sum / count as f32 } else { 0.0 };
    }

    // 5. Apply
    let result: Vec<f32> = base
        .iter()
        .zip(merged_delta.iter())
        .map(|(b, d)| b + config.merge_coefficient * d)
        .collect();

    Ok(result)
}

/// Trim the `trim_fraction` of parameters with smallest |value| to zero.
fn trim_delta(delta: &[f32], trim_fraction: f32) -> Vec<f32> {
    if delta.is_empty() || trim_fraction <= 0.0 {
        return delta.to_vec();
    }

    let n = delta.len();
    // Number of elements to zero out (smallest magnitude)
    let trim_k = ((n as f32 * trim_fraction).floor() as usize).min(n);
    if trim_k == 0 {
        return delta.to_vec();
    }

    // Find the threshold: the (trim_k)-th smallest |value|
    let mut abs_sorted: Vec<f32> = delta.iter().map(|x| x.abs()).collect();
    // Sort ascending
    abs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // threshold = the trim_k-th smallest (0-indexed: trim_k - 1)
    let threshold = abs_sorted[trim_k - 1];

    // Zero out everything strictly below threshold; ties go towards keeping
    // (we zero exactly trim_k elements from the smallest end)
    // Simple approach: count how many at exactly threshold we still need to zero
    let mut zeros_needed = trim_k;
    delta
        .iter()
        .map(|&v| {
            if v.abs() < threshold {
                zeros_needed = zeros_needed.saturating_sub(1);
                0.0
            } else if v.abs() == threshold && zeros_needed > 0 {
                zeros_needed -= 1;
                0.0
            } else {
                v
            }
        })
        .collect()
}

/// Elect sign at each position by majority vote (count of positive vs negative).
fn elect_majority_sign(trimmed: &[Vec<f32>], param_len: usize) -> Vec<f32> {
    (0..param_len)
        .map(|i| {
            let pos_count = trimmed.iter().filter(|v| v[i] > 0.0).count();
            let neg_count = trimmed.iter().filter(|v| v[i] < 0.0).count();
            if pos_count >= neg_count {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

/// Elect sign at each position by the direction with the greater summed |magnitude|.
fn elect_greater_magnitude_sign(trimmed: &[Vec<f32>], param_len: usize) -> Vec<f32> {
    (0..param_len)
        .map(|i| {
            let pos_mass: f32 = trimmed
                .iter()
                .map(|v| if v[i] > 0.0 { v[i] } else { 0.0 })
                .sum();
            let neg_mass: f32 = trimmed
                .iter()
                .map(|v| if v[i] < 0.0 { v[i].abs() } else { 0.0 })
                .sum();
            if pos_mass >= neg_mass {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

// ─── Linear ───────────────────────────────────────────────────────────────────

/// Configuration for linear (weighted average) model merging.
#[derive(Debug, Clone)]
pub struct LinearMergeConfig {
    /// Per-model weights.  Must sum to approximately 1.0 (within 1e-4).
    /// Must have the same length as the `models` slice passed to [`linear_merge`].
    pub weights: Vec<f32>,
}

impl Default for LinearMergeConfig {
    fn default() -> Self {
        Self { weights: vec![] }
    }
}

/// Merge models using a simple weighted average.
///
/// `result[i] = sum_j( weights[j] * models[j][i] )`
///
/// The weights must sum to approximately 1.0 (tolerance 1e-4).
///
/// # Errors
///
/// Returns [`MergeError::EmptyModels`] when `models` is empty.
/// Returns [`MergeError::InvalidWeights`] when weight count mismatches models or sum ≠ 1.
/// Returns [`MergeError::DimensionMismatch`] when models have different lengths.
pub fn linear_merge(
    models: &[&[f32]],
    config: &LinearMergeConfig,
) -> Result<Vec<f32>, MergeError> {
    if models.is_empty() {
        return Err(MergeError::EmptyModels);
    }
    if config.weights.len() != models.len() {
        return Err(MergeError::InvalidWeights(format!(
            "expected {} weights for {} models, got {}",
            models.len(),
            models.len(),
            config.weights.len(),
        )));
    }
    let weight_sum: f32 = config.weights.iter().sum();
    if (weight_sum - 1.0).abs() > 1e-4 {
        return Err(MergeError::InvalidWeights(format!(
            "weights must sum to 1.0, got {}",
            weight_sum
        )));
    }

    let param_len = models[0].len();
    for (idx, model) in models.iter().enumerate() {
        if model.len() != param_len {
            return Err(MergeError::DimensionMismatch {
                a: param_len,
                b: model.len(),
            });
        }
        let _ = idx;
    }

    let mut result = vec![0.0f32; param_len];
    for (model, &w) in models.iter().zip(config.weights.iter()) {
        for (r, &v) in result.iter_mut().zip(model.iter()) {
            *r += w * v;
        }
    }

    Ok(result)
}

// ─── SLERP ────────────────────────────────────────────────────────────────────

/// Configuration for Spherical Linear Interpolation (SLERP) model merging.
#[derive(Debug, Clone)]
pub struct SphericalLinearMergeConfig {
    /// Interpolation factor in [0, 1].  `t=0` → model_a, `t=1` → model_b.
    pub t: f32,
    /// Numerical stability epsilon.  When `omega < eps`, falls back to lerp.
    pub eps: f32,
}

impl Default for SphericalLinearMergeConfig {
    fn default() -> Self {
        Self { t: 0.5, eps: 1e-6 }
    }
}

/// Merge two models using Spherical Linear Interpolation (SLERP).
///
/// Computes the geodesic interpolation between `model_a` and `model_b` on the
/// unit hypersphere:
///
/// ```text
/// omega = acos( dot(a, b) / (||a|| * ||b||) )
/// result = sin((1-t)*omega)/sin(omega) * a + sin(t*omega)/sin(omega) * b
/// ```
///
/// Falls back to linear interpolation when:
/// - Either vector is a zero vector (norm < eps), or
/// - `omega < eps` (vectors are nearly parallel).
///
/// # Errors
///
/// Returns [`MergeError::DimensionMismatch`] when `model_a` and `model_b`
/// have different lengths.
pub fn slerp_merge(
    model_a: &[f32],
    model_b: &[f32],
    config: &SphericalLinearMergeConfig,
) -> Result<Vec<f32>, MergeError> {
    if model_a.len() != model_b.len() {
        return Err(MergeError::DimensionMismatch {
            a: model_a.len(),
            b: model_b.len(),
        });
    }

    let t = config.t as f64;
    let eps = config.eps as f64;

    let norm_a: f64 = model_a.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    let norm_b: f64 = model_b.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();

    // Fallback to linear when either vector is near-zero
    if norm_a < eps || norm_b < eps {
        let result: Vec<f32> = model_a
            .iter()
            .zip(model_b.iter())
            .map(|(&a, &b)| ((1.0 - t) * a as f64 + t * b as f64) as f32)
            .collect();
        return Ok(result);
    }

    let dot: f64 = model_a
        .iter()
        .zip(model_b.iter())
        .map(|(&a, &b)| a as f64 * b as f64)
        .sum();

    let cos_omega = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
    let omega = cos_omega.acos();

    if omega.abs() < eps {
        // Nearly parallel: linear fallback
        let result: Vec<f32> = model_a
            .iter()
            .zip(model_b.iter())
            .map(|(&a, &b)| ((1.0 - t) * a as f64 + t * b as f64) as f32)
            .collect();
        return Ok(result);
    }

    let sin_omega = omega.sin();
    let w_a = ((1.0 - t) * omega).sin() / sin_omega;
    let w_b = (t * omega).sin() / sin_omega;

    let result: Vec<f32> = model_a
        .iter()
        .zip(model_b.iter())
        .map(|(&a, &b)| (w_a * a as f64 + w_b * b as f64) as f32)
        .collect();

    Ok(result)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn assert_vec_approx(a: &[f32], b: &[f32], tol: f32, msg: &str) {
        assert_eq!(a.len(), b.len(), "length mismatch in {}", msg);
        for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (ai - bi).abs() <= tol,
                "{}: index {} differs: {} vs {} (tol {})",
                msg, i, ai, bi, tol
            );
        }
    }

    // ─────────────────────── DARE ─────────────────────────────────────────────

    // 1. DARE with drop_rate=0.0 → no dropping, all deltas kept
    #[test]
    fn test_dare_no_drop() {
        let base = vec![1.0f32, 2.0, 3.0, 4.0];
        let fine = vec![2.0f32, 4.0, 6.0, 8.0];
        let cfg = DareMergeConfig {
            drop_rate: 0.0,
            rescale: false,
            seed: 0,
            merge_coefficient: 1.0,
        };
        let result = dare_merge(&base, &fine, &cfg).expect("dare merge ok");
        // delta = [1,2,3,4], no drop, coeff=1 → result = fine
        assert_vec_approx(&result, &fine, 1e-6, "dare no drop");
    }

    // 2. DARE sparsity level: drop_rate=0.9 drops roughly 90% of parameters
    #[test]
    fn test_dare_sparsity_level() {
        let n = 1000usize;
        let base: Vec<f32> = vec![0.0f32; n];
        let fine: Vec<f32> = vec![1.0f32; n];
        let cfg = DareMergeConfig {
            drop_rate: 0.9,
            rescale: false,
            seed: 42,
            merge_coefficient: 1.0,
        };
        let result = dare_merge(&base, &fine, &cfg).expect("dare merge ok");

        // Count non-zero deltas (result[i] - base[i] != 0)
        let kept = result.iter().filter(|&&v| v.abs() > 1e-7).count();
        // Should be roughly 10% ± 5%
        assert!(
            kept >= 50 && kept <= 150,
            "expected ~10% kept, got {} / {}",
            kept, n
        );
    }

    // 3. DARE rescaling: when rescale=true, surviving deltas are amplified
    #[test]
    fn test_dare_rescaling() {
        let base = vec![0.0f32; 100];
        let fine = vec![1.0f32; 100];
        let cfg_no_rescale = DareMergeConfig {
            drop_rate: 0.5,
            rescale: false,
            seed: 7,
            merge_coefficient: 1.0,
        };
        let cfg_rescale = DareMergeConfig {
            rescale: true,
            ..cfg_no_rescale.clone()
        };

        let r_no = dare_merge(&base, &fine, &cfg_no_rescale).expect("ok");
        let r_rs = dare_merge(&base, &fine, &cfg_rescale).expect("ok");

        // For each kept parameter (non-zero in no_rescale result),
        // the rescaled version should be exactly 2x (1/(1-0.5)=2)
        for (&no, &rs) in r_no.iter().zip(r_rs.iter()) {
            if no.abs() > 1e-7 {
                assert!(
                    (rs - no * 2.0).abs() < 1e-5,
                    "rescaled should be 2x: {} vs {}",
                    rs, no
                );
            } else {
                // dropped position: both should be 0
                assert!(rs.abs() < 1e-7, "dropped position should stay 0: {}", rs);
            }
        }
    }

    // 4. DARE dimension mismatch
    #[test]
    fn test_dare_dimension_mismatch() {
        let cfg = DareMergeConfig::default();
        let err = dare_merge(&[1.0, 2.0], &[1.0], &cfg).expect_err("should fail");
        assert!(matches!(err, MergeError::DimensionMismatch { .. }));
    }

    // 5. DARE: deterministic — same seed → same result
    #[test]
    fn test_dare_deterministic() {
        let base = vec![0.0f32; 50];
        let fine: Vec<f32> = (0..50).map(|i| i as f32 * 0.1).collect();
        let cfg = DareMergeConfig { drop_rate: 0.7, rescale: true, seed: 123, merge_coefficient: 0.5 };
        let r1 = dare_merge(&base, &fine, &cfg).expect("ok");
        let r2 = dare_merge(&base, &fine, &cfg).expect("ok");
        assert_eq!(r1, r2, "dare must be deterministic");
    }

    // ─────────────────────── TIES ─────────────────────────────────────────────

    // 6. TIES trim: after trim_fraction=0.5, exactly 50% of |delta| are zeroed
    #[test]
    fn test_ties_trim_zeros_small_deltas() {
        // Four deltas of magnitude 1,2,3,4 → trim bottom 50% → zero 1,2
        let delta = vec![1.0f32, 2.0, 3.0, 4.0];
        let trimmed = trim_delta(&delta, 0.5);
        // Bottom 2 should be zero
        assert_eq!(trimmed[0], 0.0, "magnitude 1 should be trimmed");
        assert_eq!(trimmed[1], 0.0, "magnitude 2 should be trimmed");
        assert!((trimmed[2] - 3.0).abs() < 1e-6);
        assert!((trimmed[3] - 4.0).abs() < 1e-6);
    }

    // 7. TIES sign election (MajoritySign)
    #[test]
    fn test_ties_sign_election_majority() {
        let base = vec![0.0f32; 1];
        let m1 = vec![5.0f32];  // delta = +5
        let m2 = vec![-1.0f32]; // delta = -1
        let m3 = vec![3.0f32];  // delta = +3
        let cfg = TiesMergeConfig {
            trim_fraction: 0.0,
            elect_method: TiesElectMethod::MajoritySign,
            merge_coefficient: 1.0,
        };
        let result = ties_merge(&base, &[m1.as_slice(), m2.as_slice(), m3.as_slice()], &cfg)
            .expect("ok");
        // 2 positives vs 1 negative → elected sign positive → merged = avg(+5, +3) = +4
        assert!(result[0] > 0.0, "majority positive should win: {}", result[0]);
    }

    // 8. TIES sign election (GreaterMagnitude)
    #[test]
    fn test_ties_sign_election_greater_magnitude() {
        let base = vec![0.0f32; 1];
        let m1 = vec![-10.0f32]; // delta = -10 (dominant magnitude)
        let m2 = vec![1.0f32];   // delta = +1
        let m3 = vec![2.0f32];   // delta = +2
        let cfg = TiesMergeConfig {
            trim_fraction: 0.0,
            elect_method: TiesElectMethod::GreaterMagnitude,
            merge_coefficient: 1.0,
        };
        let result = ties_merge(&base, &[m1.as_slice(), m2.as_slice(), m3.as_slice()], &cfg)
            .expect("ok");
        // Negative mass = 10, positive mass = 3 → negative sign elected
        // Only m1 agrees → merged = -10
        assert!(result[0] < 0.0, "greater magnitude negative should win: {}", result[0]);
    }

    // 9. TIES: empty models error
    #[test]
    fn test_ties_empty_models() {
        let base = vec![1.0f32];
        let cfg = TiesMergeConfig::default();
        let err = ties_merge(&base, &[], &cfg).expect_err("should fail");
        assert!(matches!(err, MergeError::EmptyModels));
    }

    // 10. TIES: trim_fraction out of range
    #[test]
    fn test_ties_trim_fraction_out_of_range() {
        let base = vec![1.0f32];
        let model = vec![2.0f32];
        let cfg = TiesMergeConfig { trim_fraction: 1.5, ..Default::default() };
        let err = ties_merge(&base, &[model.as_slice()], &cfg).expect_err("should fail");
        assert!(matches!(err, MergeError::TrimFractionOutOfRange));
    }

    // 11. TIES dimension mismatch
    #[test]
    fn test_ties_dimension_mismatch() {
        let base = vec![0.0f32; 4];
        let model = vec![1.0f32; 3]; // wrong length
        let cfg = TiesMergeConfig::default();
        let err = ties_merge(&base, &[model.as_slice()], &cfg).expect_err("should fail");
        assert!(matches!(err, MergeError::DimensionMismatch { .. }));
    }

    // ─────────────────────── Linear ───────────────────────────────────────────

    // 12. Linear merge: weighted average correctness
    #[test]
    fn test_linear_merge_weighted() {
        let m1 = vec![0.0f32, 0.0, 0.0];
        let m2 = vec![4.0f32, 8.0, 12.0];
        let cfg = LinearMergeConfig { weights: vec![0.5, 0.5] };
        let result = linear_merge(&[m1.as_slice(), m2.as_slice()], &cfg).expect("ok");
        assert_vec_approx(&result, &[2.0, 4.0, 6.0], 1e-5, "linear 50/50");
    }

    // 13. Linear merge: three models with custom weights
    #[test]
    fn test_linear_merge_three_models() {
        let m1 = vec![6.0f32];
        let m2 = vec![0.0f32];
        let m3 = vec![3.0f32];
        // weights 0.5, 0.25, 0.25 → 0.5*6 + 0.25*0 + 0.25*3 = 3.75
        let cfg = LinearMergeConfig { weights: vec![0.5, 0.25, 0.25] };
        let result = linear_merge(&[m1.as_slice(), m2.as_slice(), m3.as_slice()], &cfg).expect("ok");
        assert!((result[0] - 3.75).abs() < 1e-5, "expected 3.75 got {}", result[0]);
    }

    // 14. Linear merge: weights don't sum to 1 → error
    #[test]
    fn test_linear_merge_weights_sum_error() {
        let m1 = vec![1.0f32];
        let m2 = vec![2.0f32];
        let cfg = LinearMergeConfig { weights: vec![0.3, 0.3] }; // sum = 0.6
        let err = linear_merge(&[m1.as_slice(), m2.as_slice()], &cfg).expect_err("should fail");
        assert!(matches!(err, MergeError::InvalidWeights(_)));
    }

    // 15. Linear merge: weight count mismatch → error
    #[test]
    fn test_linear_merge_weight_count_mismatch() {
        let m1 = vec![1.0f32];
        let m2 = vec![2.0f32];
        let cfg = LinearMergeConfig { weights: vec![1.0] }; // only 1 weight for 2 models
        let err = linear_merge(&[m1.as_slice(), m2.as_slice()], &cfg).expect_err("should fail");
        assert!(matches!(err, MergeError::InvalidWeights(_)));
    }

    // ─────────────────────── SLERP ────────────────────────────────────────────

    // 16. SLERP at t=0 returns model_a
    #[test]
    fn test_slerp_t0_returns_model_a() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let cfg = SphericalLinearMergeConfig { t: 0.0, eps: 1e-6 };
        let result = slerp_merge(&a, &b, &cfg).expect("ok");
        assert_vec_approx(&result, &a, 1e-5, "slerp t=0 should equal model_a");
    }

    // 17. SLERP at t=1 returns model_b
    #[test]
    fn test_slerp_t1_returns_model_b() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let cfg = SphericalLinearMergeConfig { t: 1.0, eps: 1e-6 };
        let result = slerp_merge(&a, &b, &cfg).expect("ok");
        assert_vec_approx(&result, &b, 1e-5, "slerp t=1 should equal model_b");
    }

    // 18. SLERP at t=0.5 between orthogonal unit vectors is at 45 degrees
    #[test]
    fn test_slerp_t05_orthogonal() {
        // a = [1, 0], b = [0, 1] → omega = π/2 → midpoint = [cos(π/4), sin(π/4)]
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let cfg = SphericalLinearMergeConfig { t: 0.5, eps: 1e-6 };
        let result = slerp_merge(&a, &b, &cfg).expect("ok");

        let expected = (std::f32::consts::PI / 4.0).cos(); // ~0.7071
        assert!((result[0] - expected).abs() < 1e-5, "x component: {}", result[0]);
        assert!((result[1] - expected).abs() < 1e-5, "y component: {}", result[1]);
    }

    // 19. SLERP dimension mismatch
    #[test]
    fn test_slerp_dimension_mismatch() {
        let a = vec![1.0f32, 0.0];
        let b = vec![1.0f32, 0.0, 0.0];
        let cfg = SphericalLinearMergeConfig::default();
        let err = slerp_merge(&a, &b, &cfg).expect_err("should fail");
        assert!(matches!(err, MergeError::DimensionMismatch { .. }));
    }

    // 20. SLERP with zero vector falls back to lerp
    #[test]
    fn test_slerp_zero_vector_fallback() {
        let a = vec![0.0f32, 0.0, 0.0]; // zero vector
        let b = vec![1.0f32, 0.0, 0.0];
        let cfg = SphericalLinearMergeConfig { t: 0.5, eps: 1e-6 };
        let result = slerp_merge(&a, &b, &cfg).expect("ok");
        // Linear interpolation: 0.5 * a + 0.5 * b = [0.5, 0, 0]
        assert_vec_approx(&result, &[0.5, 0.0, 0.0], 1e-5, "zero vector lerp fallback");
    }

    // 21. DARE merge_coefficient = 0 → result equals base
    #[test]
    fn test_dare_coeff_zero_equals_base() {
        let base = vec![1.0f32, 2.0, 3.0];
        let fine = vec![10.0f32, 20.0, 30.0];
        let cfg = DareMergeConfig {
            drop_rate: 0.0,
            rescale: false,
            seed: 0,
            merge_coefficient: 0.0,
        };
        let result = dare_merge(&base, &fine, &cfg).expect("ok");
        assert_vec_approx(&result, &base, 1e-6, "coeff=0 should return base");
    }

    // 22. TIES with trim_fraction=0 and single model: result = base + coeff*delta
    #[test]
    fn test_ties_no_trim_single_model() {
        let base = vec![0.0f32; 3];
        let model = vec![2.0f32, -3.0, 1.0];
        let cfg = TiesMergeConfig {
            trim_fraction: 0.0,
            elect_method: TiesElectMethod::MajoritySign,
            merge_coefficient: 1.0,
        };
        let result = ties_merge(&base, &[model.as_slice()], &cfg).expect("ok");
        // Single model: delta = [2, -3, 1], elected signs per position, same-sign avg = delta itself
        assert_vec_approx(&result, &model, 1e-5, "single model no-trim should equal model");
    }
}
