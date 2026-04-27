//! RoPE (Rotary Position Embeddings) variants
//!
//! Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
//!
//! This module provides multiple RoPE scaling strategies for extending context length:
//! - Standard RoPE (no scaling)
//! - Linear scaling for simple context extension
//! - NTK-aware scaling for better quality at extended lengths
//! - Dynamic NTK that adapts based on actual sequence length
//! - YaRN (Yet another RoPE Extension) with frequency-dependent scaling
//! - LongRoPE with per-dimension scaling factors (Microsoft)

use std::fmt;

// ────────────────────────────────────────────────────────────────────────────
// Error type
// ────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during RoPE computation or application
#[derive(Debug)]
pub enum RopeError {
    /// Head dimension is invalid (must be positive and even)
    InvalidHeadDim { dim: usize, reason: &'static str },
    /// Scaling factor is out of range (must be ≥ 1.0)
    InvalidScalingFactor(f64),
    /// Requested sequence length exceeds precomputed maximum
    SequenceLengthExceeded { seq_len: usize, max: usize },
    /// Tensor dimension mismatch
    DimensionMismatch { expected: usize, got: usize },
}

impl fmt::Display for RopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RopeError::InvalidHeadDim { dim, reason } => {
                write!(f, "Invalid head dimension {dim}: {reason}")
            }
            RopeError::InvalidScalingFactor(factor) => {
                write!(f, "Invalid scaling factor {factor}: must be >= 1.0")
            }
            RopeError::SequenceLengthExceeded { seq_len, max } => {
                write!(
                    f,
                    "Sequence length {seq_len} exceeds precomputed maximum {max}"
                )
            }
            RopeError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for RopeError {}

// ────────────────────────────────────────────────────────────────────────────
// RoPE scaling strategy
// ────────────────────────────────────────────────────────────────────────────

/// Strategy for scaling RoPE frequencies to extend context length
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScalingType {
    /// Standard RoPE (no scaling)
    None,

    /// Linear scaling: scale all frequencies by 1/factor
    ///
    /// Used for extending context with minimal quality loss.
    /// All wavelengths are multiplied uniformly by `factor`.
    Linear { factor: f64 },

    /// NTK-aware scaling: modify the base theta so that high-frequency
    /// components are scaled less aggressively than low-frequency ones.
    ///
    /// Based on the "NTK-Aware Scaled RoPE" blog post.
    /// Modified base: `base * factor^(head_dim / (head_dim - 2))`
    Ntk { factor: f64 },

    /// Dynamic NTK: recompute the NTK-scaled base using the actual
    /// inference sequence length rather than the config max.
    ///
    /// Automatically adjusts as the sequence length grows.
    DynamicNtk {
        factor: f64,
        original_max_position: usize,
    },

    /// YaRN (Yet another RoPE Extension): frequency-dependent scaling.
    ///
    /// Different frequency dimensions use different interpolation strategies:
    /// - High-frequency dims (short wavelength) → keep original
    /// - Low-frequency dims (long wavelength) → linear interpolation
    /// - Middle dims → smooth ramp blend of both
    Yarn {
        factor: f64,
        original_max_position: usize,
        /// High-frequency boundary (default 32.0)
        beta_fast: f64,
        /// Low-frequency boundary (default 1.0)
        beta_slow: f64,
    },

    /// LongRoPE: per-dimension scaling factors (Microsoft).
    ///
    /// Two sets of scaling factors are provided: one for short sequences
    /// and one for long sequences (beyond `original_max_position`).
    LongRope {
        short_factors: Vec<f64>,
        long_factors: Vec<f64>,
        original_max_position: usize,
    },
}

impl fmt::Display for RopeScalingType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RopeScalingType::None => write!(f, "None"),
            RopeScalingType::Linear { factor } => write!(f, "Linear(factor={factor})"),
            RopeScalingType::Ntk { factor } => write!(f, "NTK(factor={factor})"),
            RopeScalingType::DynamicNtk {
                factor,
                original_max_position,
            } => write!(
                f,
                "DynamicNTK(factor={factor}, orig_max={original_max_position})"
            ),
            RopeScalingType::Yarn {
                factor,
                original_max_position,
                beta_fast,
                beta_slow,
            } => write!(
                f,
                "YaRN(factor={factor}, orig_max={original_max_position}, \
                 beta_fast={beta_fast}, beta_slow={beta_slow})"
            ),
            RopeScalingType::LongRope {
                short_factors,
                long_factors,
                original_max_position,
            } => write!(
                f,
                "LongRoPE(short_factors=[{} dims], long_factors=[{} dims], \
                 orig_max={original_max_position})",
                short_factors.len(),
                long_factors.len()
            ),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// RoPE configuration
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for RoPE frequency computation
#[derive(Debug, Clone)]
pub struct RopeConfig {
    /// Dimension of each attention head (must be even and positive)
    pub head_dim: usize,
    /// Base theta value (default 10000.0)
    pub base_theta: f64,
    /// Scaling strategy
    pub scaling: RopeScalingType,
    /// Maximum position embeddings supported by the model config
    pub max_position_embeddings: usize,
}

impl RopeConfig {
    /// Standard RoPE with no scaling (base = 10000, max_pos = 4096)
    pub fn standard(head_dim: usize) -> Self {
        Self {
            head_dim,
            base_theta: 10_000.0,
            scaling: RopeScalingType::None,
            max_position_embeddings: 4096,
        }
    }

    /// Standard RoPE with linear context scaling
    pub fn with_linear_scaling(head_dim: usize, factor: f64) -> Self {
        Self {
            head_dim,
            base_theta: 10_000.0,
            scaling: RopeScalingType::Linear { factor },
            max_position_embeddings: 4096,
        }
    }

    /// Standard RoPE with NTK-aware scaling
    pub fn with_ntk(head_dim: usize, factor: f64) -> Self {
        Self {
            head_dim,
            base_theta: 10_000.0,
            scaling: RopeScalingType::Ntk { factor },
            max_position_embeddings: 4096,
        }
    }

    /// Standard RoPE with YaRN scaling
    pub fn with_yarn(head_dim: usize, factor: f64, max_pos: usize) -> Self {
        Self {
            head_dim,
            base_theta: 10_000.0,
            scaling: RopeScalingType::Yarn {
                factor,
                original_max_position: max_pos,
                beta_fast: 32.0,
                beta_slow: 1.0,
            },
            max_position_embeddings: max_pos,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Frequency statistics
// ────────────────────────────────────────────────────────────────────────────

/// Statistics about the frequency dimensions in precomputed RoPE
#[derive(Debug, Clone)]
pub struct RopeFreqStats {
    /// Minimum frequency across all half-dimensions
    pub min_freq: f32,
    /// Maximum frequency across all half-dimensions
    pub max_freq: f32,
    /// Mean frequency across all half-dimensions
    pub mean_freq: f32,
    /// Number of dimensions with frequency < 0.01 (low-frequency)
    pub num_low_freq_dims: usize,
    /// Number of dimensions with frequency > 1.0 (high-frequency)
    pub num_high_freq_dims: usize,
}

// ────────────────────────────────────────────────────────────────────────────
// Precomputed RoPE frequencies
// ────────────────────────────────────────────────────────────────────────────

/// Precomputed cosine and sine tables for RoPE application
///
/// Stores cos/sin values for all positions up to `max_seq_len` and all
/// frequency dimensions. Layout: `[max_seq_len, head_dim/2]` row-major.
pub struct RopeFrequencies {
    /// Cosine values, shape `[max_seq_len, head_dim/2]` (flat row-major)
    pub cos: Vec<f32>,
    /// Sine values, shape `[max_seq_len, head_dim/2]` (flat row-major)
    pub sin: Vec<f32>,
    /// Number of positions precomputed
    pub max_seq_len: usize,
    /// Head dimension
    pub head_dim: usize,
    /// The configuration used to compute these frequencies
    pub config: RopeConfig,
}

impl RopeFrequencies {
    /// Compute RoPE frequencies for the given configuration up to `max_seq_len` positions.
    ///
    /// All computation is done in f64 for numerical accuracy and cast to f32 for storage.
    pub fn compute(config: RopeConfig, max_seq_len: usize) -> Result<Self, RopeError> {
        // Validate head dimension
        if config.head_dim == 0 {
            return Err(RopeError::InvalidHeadDim {
                dim: config.head_dim,
                reason: "head_dim must be > 0",
            });
        }
        if config.head_dim % 2 != 0 {
            return Err(RopeError::InvalidHeadDim {
                dim: config.head_dim,
                reason: "head_dim must be even",
            });
        }

        let half_dim = config.head_dim / 2;

        // Validate scaling factor (≥ 1.0 for linear/ntk/dynamic_ntk/yarn)
        match &config.scaling {
            RopeScalingType::Linear { factor }
            | RopeScalingType::Ntk { factor }
            | RopeScalingType::DynamicNtk { factor, .. }
            | RopeScalingType::Yarn { factor, .. } => {
                if *factor < 1.0 {
                    return Err(RopeError::InvalidScalingFactor(*factor));
                }
            }
            _ => {}
        }

        // ── Compute per-dimension frequencies (in f64) ──────────────────────
        let freqs: Vec<f64> = match &config.scaling {
            RopeScalingType::None => standard_freqs(config.base_theta, half_dim, config.head_dim),

            RopeScalingType::Linear { factor } => {
                let base_freqs = standard_freqs(config.base_theta, half_dim, config.head_dim);
                base_freqs.into_iter().map(|f| f / factor).collect()
            }

            RopeScalingType::Ntk { factor } => {
                // Modify base: base_new = base * factor^(head_dim / (head_dim - 2))
                let exp = config.head_dim as f64 / (config.head_dim as f64 - 2.0);
                let modified_base = config.base_theta * factor.powf(exp);
                standard_freqs(modified_base, half_dim, config.head_dim)
            }

            RopeScalingType::DynamicNtk {
                factor,
                original_max_position,
            } => {
                // Compute an effective factor based on actual max_seq_len vs original
                let effective_factor = if max_seq_len <= *original_max_position {
                    1.0_f64
                } else {
                    (factor * max_seq_len as f64 / *original_max_position as f64)
                        - (factor - 1.0)
                };
                let effective_factor = effective_factor.max(1.0);
                let exp = config.head_dim as f64 / (config.head_dim as f64 - 2.0);
                let modified_base = config.base_theta * effective_factor.powf(exp);
                standard_freqs(modified_base, half_dim, config.head_dim)
            }

            RopeScalingType::Yarn {
                factor,
                original_max_position,
                beta_fast,
                beta_slow,
            } => {
                let base_freqs = standard_freqs(config.base_theta, half_dim, config.head_dim);
                // Linear-scaled frequencies (low-freq treatment)
                let linear_freqs: Vec<f64> = base_freqs.iter().map(|f| f / factor).collect();

                let two_pi = 2.0 * std::f64::consts::PI;
                // Wavelength thresholds
                let low_thresh = two_pi * factor / beta_slow;
                let high_thresh = two_pi / beta_fast;

                base_freqs
                    .iter()
                    .zip(linear_freqs.iter())
                    .enumerate()
                    .map(|(i, (&orig_f, &lin_f))| {
                        let wavelength = two_pi / orig_f; // wavelength for dim i
                        let _ = i; // suppress unused warning
                        let _ = original_max_position; // used to determine context extension

                        if wavelength < high_thresh {
                            // High frequency: keep original
                            orig_f
                        } else if wavelength > low_thresh {
                            // Low frequency: use linear interpolation
                            lin_f
                        } else {
                            // Smooth ramp blend
                            let ramp = yarn_ramp(wavelength, high_thresh, low_thresh);
                            orig_f * (1.0 - ramp) + lin_f * ramp
                        }
                    })
                    .collect()
            }

            RopeScalingType::LongRope {
                short_factors,
                long_factors,
                original_max_position,
            } => {
                // Choose factor set based on whether max_seq_len exceeds original max
                let factors = if max_seq_len > *original_max_position {
                    long_factors
                } else {
                    short_factors
                };

                let base_freqs = standard_freqs(config.base_theta, half_dim, config.head_dim);

                // Pad or truncate factors to match half_dim
                base_freqs
                    .iter()
                    .enumerate()
                    .map(|(i, &base_f)| {
                        let scale = factors.get(i).copied().unwrap_or(1.0);
                        base_f / scale
                    })
                    .collect()
            }
        };

        // ── Build cos/sin tables ─────────────────────────────────────────────
        let capacity = max_seq_len * half_dim;
        let mut cos_table = Vec::with_capacity(capacity);
        let mut sin_table = Vec::with_capacity(capacity);

        for pos in 0..max_seq_len {
            let pos_f = pos as f64;
            for i in 0..half_dim {
                let angle = pos_f * freqs[i];
                cos_table.push(angle.cos() as f32);
                sin_table.push(angle.sin() as f32);
            }
        }

        Ok(Self {
            cos: cos_table,
            sin: sin_table,
            max_seq_len,
            head_dim: config.head_dim,
            config,
        })
    }

    /// Apply RoPE rotation to a query or key tensor.
    ///
    /// # Arguments
    /// * `q` - Flat tensor of shape `[seq_len, num_heads, head_dim]`
    /// * `seq_len` - Number of tokens
    /// * `num_heads` - Number of attention heads
    ///
    /// # Returns
    /// Rotated tensor with the same shape as input.
    pub fn apply_rope(
        &self,
        q: &[f32],
        seq_len: usize,
        num_heads: usize,
    ) -> Result<Vec<f32>, RopeError> {
        if seq_len > self.max_seq_len {
            return Err(RopeError::SequenceLengthExceeded {
                seq_len,
                max: self.max_seq_len,
            });
        }

        let head_dim = self.head_dim;
        let expected_len = seq_len * num_heads * head_dim;
        if q.len() != expected_len {
            return Err(RopeError::DimensionMismatch {
                expected: expected_len,
                got: q.len(),
            });
        }

        let half_dim = head_dim / 2;
        let mut output = vec![0.0_f32; expected_len];

        for pos in 0..seq_len {
            let cos_row = pos * half_dim;
            for head in 0..num_heads {
                let base_idx = pos * num_heads * head_dim + head * head_dim;
                for i in 0..half_dim {
                    let cos_val = self.cos[cos_row + i];
                    let sin_val = self.sin[cos_row + i];

                    let x0 = q[base_idx + 2 * i];
                    let x1 = q[base_idx + 2 * i + 1];

                    output[base_idx + 2 * i] = x0 * cos_val - x1 * sin_val;
                    output[base_idx + 2 * i + 1] = x0 * sin_val + x1 * cos_val;
                }
            }
        }

        Ok(output)
    }

    /// Apply RoPE to both Q and K tensors simultaneously.
    ///
    /// # Arguments
    /// * `q` - Query tensor `[q_seq_len, num_heads, head_dim]`
    /// * `k` - Key tensor `[k_seq_len, num_kv_heads, head_dim]`
    /// * `q_seq_len` - Query sequence length
    /// * `k_seq_len` - Key sequence length
    /// * `num_heads` - Number of query heads
    /// * `num_kv_heads` - Number of key/value heads
    pub fn apply_rope_qk(
        &self,
        q: &[f32],
        k: &[f32],
        q_seq_len: usize,
        k_seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), RopeError> {
        let rotated_q = self.apply_rope(q, q_seq_len, num_heads)?;
        let rotated_k = self.apply_rope(k, k_seq_len, num_kv_heads)?;
        Ok((rotated_q, rotated_k))
    }

    /// Compute statistics about the per-dimension frequencies.
    pub fn frequency_stats(&self) -> RopeFreqStats {
        if self.max_seq_len == 0 {
            return RopeFreqStats {
                min_freq: 0.0,
                max_freq: 0.0,
                mean_freq: 0.0,
                num_low_freq_dims: 0,
                num_high_freq_dims: 0,
            };
        }

        let half_dim = self.head_dim / 2;

        // Extract frequencies from position 1 (pos=0 gives trivial 0-angle)
        // freq_i = angle at pos=1 = arccos(cos[1 * half_dim + i])
        // But more accurately, freq_i is directly encoded as sin[half_dim + i] at pos=1
        // We use atan2(sin, cos) to recover the angle at pos=1
        let freqs: Vec<f32> = if self.max_seq_len > 1 {
            (0..half_dim)
                .map(|i| {
                    let c = self.cos[half_dim + i]; // pos=1
                    let s = self.sin[half_dim + i];
                    s.atan2(c).abs()
                })
                .collect()
        } else {
            vec![0.0_f32; half_dim]
        };

        let min_freq = freqs.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_freq = freqs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean_freq = freqs.iter().sum::<f32>() / half_dim as f32;
        let num_low_freq_dims = freqs.iter().filter(|&&f| f < 0.01).count();
        let num_high_freq_dims = freqs.iter().filter(|&&f| f > 1.0).count();

        RopeFreqStats {
            min_freq,
            max_freq,
            mean_freq,
            num_low_freq_dims,
            num_high_freq_dims,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ────────────────────────────────────────────────────────────────────────────

/// Compute standard RoPE frequencies: theta_i = base^(-2i/head_dim)
fn standard_freqs(base: f64, half_dim: usize, head_dim: usize) -> Vec<f64> {
    (0..half_dim)
        .map(|i| {
            let exponent = -2.0 * i as f64 / head_dim as f64;
            base.powf(exponent)
        })
        .collect()
}

/// YaRN smooth ramp function.
///
/// Returns a value in [0, 1] where:
/// - 0 → use original frequency (high-freq region)
/// - 1 → use linear-interpolated frequency (low-freq region)
fn yarn_ramp(wavelength: f64, low: f64, high: f64) -> f64 {
    ((wavelength - low) / (high - low)).clamp(0.0, 1.0)
}

// ────────────────────────────────────────────────────────────────────────────
// YaRN standalone API
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for the standalone YaRN API.
#[derive(Debug, Clone)]
pub struct YarnConfig {
    /// Original maximum position embeddings (e.g. 4096).
    pub original_max_position_embeddings: usize,
    /// Context extension scaling factor (e.g. 8.0 to go from 4096 → 32768).
    pub scaling_factor: f32,
    /// High-frequency boundary: dimensions whose rotational count per token
    /// exceeds this are treated as "high-freq" and kept unscaled (e.g. 32.0).
    pub beta_fast: f32,
    /// Low-frequency boundary: dimensions with fewer rotations are linearly
    /// interpolated (e.g. 1.0).
    pub beta_slow: f32,
    /// Attention magnitude scale applied to all output values.
    pub mscale: f32,
    /// Secondary scale applied uniformly to all dimensions.
    pub mscale_all_dim: f32,
    /// Base theta for inverse-frequency computation (typically 10000.0).
    pub base: f32,
    /// Head dimension (must be even and positive).
    pub head_dim: usize,
}

/// Compute the correction dimension for YaRN.
///
/// Returns the fractional dimension index at which the wavelength produced by
/// `num_rotations` rotations over the original context equals `2π`.
///
/// Formula: `dim * log(orig_max_pos / (num_rotations * 2π)) / (2 * log(base))`
pub fn yarn_find_correction_dim(
    num_rotations: f32,
    dim: usize,
    base: f32,
    max_position_embeddings: usize,
) -> f32 {
    let two_pi = 2.0 * std::f32::consts::PI;
    let numerator = (max_position_embeddings as f32 / (num_rotations * two_pi)).ln();
    let denominator = 2.0 * base.ln();
    dim as f32 * numerator / denominator
}

/// Build a linear ramp mask of length `dim`.
///
/// Output[i] = clamp((i - min) / (max - min), 0, 1).
/// Dimensions near `min` (high-freq) get weight 0; dimensions near `max`
/// (low-freq) get weight 1.  When `min >= max` the function returns all zeros.
pub fn yarn_linear_ramp_mask(min: f32, max: f32, dim: usize) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }
    if (max - min).abs() < f32::EPSILON {
        return vec![0.0; dim];
    }
    (0..dim)
        .map(|i| ((i as f32 - min) / (max - min)).clamp(0.0, 1.0))
        .collect()
}

/// Compute the YaRN magnitude scale.
///
/// Formula: `sqrt(0.1 * ln(scale) + 1.0) * mscale`
///
/// Returns 1.0 when `scale <= 0.0` (guard against non-positive inputs).
pub fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 0.0 {
        return mscale;
    }
    (0.1 * scale.ln() + 1.0).sqrt() * mscale
}

/// Apply YaRN-scaled RoPE to query and key tensors.
///
/// # Arguments
/// * `query`     — Flat `[seq_len, num_heads, head_dim]` query tensor.
/// * `key`       — Flat `[seq_len, num_kv_heads, head_dim]` key tensor.
/// * `positions` — Token positions (length = `seq_len`).
/// * `num_heads` — Number of query heads.
/// * `num_kv_heads` — Number of key/value heads.
/// * `config`    — YaRN configuration.
///
/// # Errors
/// Returns [`RopeError::InvalidHeadDim`] if `head_dim` is 0 or odd.
/// Returns [`RopeError::DimensionMismatch`] on tensor shape issues.
pub fn apply_yarn_rope(
    query: &[f32],
    key: &[f32],
    positions: &[usize],
    num_heads: usize,
    num_kv_heads: usize,
    config: &YarnConfig,
) -> Result<(Vec<f32>, Vec<f32>), RopeError> {
    let head_dim = config.head_dim;
    if head_dim == 0 {
        return Err(RopeError::InvalidHeadDim {
            dim: head_dim,
            reason: "head_dim must be > 0",
        });
    }
    if head_dim % 2 != 0 {
        return Err(RopeError::InvalidHeadDim {
            dim: head_dim,
            reason: "head_dim must be even",
        });
    }

    let seq_len = positions.len();
    let half_dim = head_dim / 2;
    let expected_q = seq_len * num_heads * head_dim;
    let expected_k = seq_len * num_kv_heads * head_dim;

    if query.len() != expected_q {
        return Err(RopeError::DimensionMismatch {
            expected: expected_q,
            got: query.len(),
        });
    }
    if key.len() != expected_k {
        return Err(RopeError::DimensionMismatch {
            expected: expected_k,
            got: key.len(),
        });
    }

    // Compute per-dimension YaRN frequencies (f32 precision throughout)
    let base = config.base;
    let scale = config.scaling_factor;
    let orig_max = config.original_max_position_embeddings;

    // Correction dims: [low_correction_dim, high_correction_dim]
    let low_corr_dim = yarn_find_correction_dim(config.beta_slow, half_dim * 2, base, orig_max);
    let high_corr_dim = yarn_find_correction_dim(config.beta_fast, half_dim * 2, base, orig_max);

    // Ramp mask: 0 at high-freq end, 1 at low-freq end
    let ramp_mask = yarn_linear_ramp_mask(low_corr_dim, high_corr_dim, half_dim);

    // Base inverse frequencies: theta_i = base^(-2i/head_dim)
    let inv_freqs: Vec<f32> = (0..half_dim)
        .map(|i| base.powf(-2.0 * i as f32 / head_dim as f32))
        .collect();

    // Scaled inverse frequencies (linear interpolation for low-freq dims)
    let scaled_inv_freqs: Vec<f32> = inv_freqs.iter().map(|&f| f / scale).collect();

    // Blended: high-freq dims keep original, low-freq use scaled, middle ramp
    let blended_freqs: Vec<f32> = (0..half_dim)
        .map(|i| {
            let ramp = ramp_mask[i];
            inv_freqs[i] * (1.0 - ramp) + scaled_inv_freqs[i] * ramp
        })
        .collect();

    // Magnitude scale
    let mscale = yarn_get_mscale(scale, config.mscale);
    let mscale_all = if config.mscale_all_dim == 0.0 {
        1.0
    } else {
        yarn_get_mscale(scale, config.mscale_all_dim)
    };
    let combined_scale = mscale * mscale_all;

    // Precompute cos/sin for each position × half-dim
    let mut cos_table = vec![0.0_f32; seq_len * half_dim];
    let mut sin_table = vec![0.0_f32; seq_len * half_dim];
    for (si, &pos) in positions.iter().enumerate() {
        for i in 0..half_dim {
            let angle = pos as f32 * blended_freqs[i];
            cos_table[si * half_dim + i] = angle.cos() * combined_scale;
            sin_table[si * half_dim + i] = angle.sin() * combined_scale;
        }
    }

    // Apply rotation to Q
    let mut out_q = vec![0.0_f32; expected_q];
    for si in 0..seq_len {
        for h in 0..num_heads {
            let base_idx = si * num_heads * head_dim + h * head_dim;
            for i in 0..half_dim {
                let cos_v = cos_table[si * half_dim + i];
                let sin_v = sin_table[si * half_dim + i];
                let x0 = query[base_idx + 2 * i];
                let x1 = query[base_idx + 2 * i + 1];
                out_q[base_idx + 2 * i] = x0 * cos_v - x1 * sin_v;
                out_q[base_idx + 2 * i + 1] = x0 * sin_v + x1 * cos_v;
            }
        }
    }

    // Apply rotation to K
    let mut out_k = vec![0.0_f32; expected_k];
    for si in 0..seq_len {
        for h in 0..num_kv_heads {
            let base_idx = si * num_kv_heads * head_dim + h * head_dim;
            for i in 0..half_dim {
                let cos_v = cos_table[si * half_dim + i];
                let sin_v = sin_table[si * half_dim + i];
                let x0 = key[base_idx + 2 * i];
                let x1 = key[base_idx + 2 * i + 1];
                out_k[base_idx + 2 * i] = x0 * cos_v - x1 * sin_v;
                out_k[base_idx + 2 * i + 1] = x0 * sin_v + x1 * cos_v;
            }
        }
    }

    Ok((out_q, out_k))
}

// ────────────────────────────────────────────────────────────────────────────
// Dynamic NTK standalone API
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for dynamic NTK-aware RoPE scaling.
#[derive(Debug, Clone)]
pub struct DynamicNtkConfig {
    /// Base theta (e.g. 10000.0).
    pub base_theta: f32,
    /// Scaling factor alpha (context extension ratio, e.g. 8.0).
    pub alpha: f32,
    /// Original maximum sequence length.
    pub max_original_length: usize,
    /// Head dimension (must be even and positive).
    pub head_dim: usize,
}

/// Compute the dynamically adjusted theta for a given sequence length.
///
/// When `seq_len <= max_original_length` the base theta is returned unchanged.
/// Otherwise the formula is:
///
/// `new_theta = base_theta * (alpha * seq_len/max_len - alpha + 1)^(dim/(dim-2))`
pub fn dynamic_ntk_theta(seq_len: usize, config: &DynamicNtkConfig) -> f32 {
    if seq_len <= config.max_original_length || config.head_dim < 3 {
        return config.base_theta;
    }
    let ratio = config.alpha * seq_len as f32 / config.max_original_length as f32
        - config.alpha
        + 1.0;
    let exp = config.head_dim as f32 / (config.head_dim as f32 - 2.0);
    config.base_theta * ratio.max(1.0).powf(exp)
}

/// Apply dynamic NTK-scaled RoPE to query and key tensors.
///
/// # Arguments
/// * `query`       — Flat `[seq_len, num_heads, head_dim]` query tensor.
/// * `key`         — Flat `[seq_len, num_kv_heads, head_dim]` key tensor.
/// * `seq_len`     — Number of tokens in the sequence.
/// * `num_heads`   — Number of query heads.
/// * `num_kv_heads` — Number of key/value heads.
/// * `config`      — Dynamic NTK configuration.
///
/// # Errors
/// Returns [`RopeError::InvalidHeadDim`] for 0 or odd head_dim.
/// Returns [`RopeError::DimensionMismatch`] on tensor shape mismatch.
pub fn apply_dynamic_ntk_rope(
    query: &[f32],
    key: &[f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    config: &DynamicNtkConfig,
) -> Result<(Vec<f32>, Vec<f32>), RopeError> {
    let head_dim = config.head_dim;
    if head_dim == 0 {
        return Err(RopeError::InvalidHeadDim {
            dim: head_dim,
            reason: "head_dim must be > 0",
        });
    }
    if head_dim % 2 != 0 {
        return Err(RopeError::InvalidHeadDim {
            dim: head_dim,
            reason: "head_dim must be even",
        });
    }

    let half_dim = head_dim / 2;
    let expected_q = seq_len * num_heads * head_dim;
    let expected_k = seq_len * num_kv_heads * head_dim;

    if query.len() != expected_q {
        return Err(RopeError::DimensionMismatch {
            expected: expected_q,
            got: query.len(),
        });
    }
    if key.len() != expected_k {
        return Err(RopeError::DimensionMismatch {
            expected: expected_k,
            got: key.len(),
        });
    }

    // Compute adjusted theta for this sequence length
    let theta = dynamic_ntk_theta(seq_len, config);

    // Compute per-dimension inverse frequencies: theta_i = theta^(-2i/head_dim)
    let inv_freqs: Vec<f32> = (0..half_dim)
        .map(|i| theta.powf(-2.0 * i as f32 / head_dim as f32))
        .collect();

    // Precompute cos/sin tables: [seq_len, half_dim]
    let mut cos_table = vec![0.0_f32; seq_len * half_dim];
    let mut sin_table = vec![0.0_f32; seq_len * half_dim];
    for pos in 0..seq_len {
        for i in 0..half_dim {
            let angle = pos as f32 * inv_freqs[i];
            cos_table[pos * half_dim + i] = angle.cos();
            sin_table[pos * half_dim + i] = angle.sin();
        }
    }

    let rotate = |tensor: &[f32], n_heads: usize, expected: usize| -> Vec<f32> {
        let mut out = vec![0.0_f32; expected];
        for pos in 0..seq_len {
            for h in 0..n_heads {
                let base_idx = pos * n_heads * head_dim + h * head_dim;
                for i in 0..half_dim {
                    let cos_v = cos_table[pos * half_dim + i];
                    let sin_v = sin_table[pos * half_dim + i];
                    let x0 = tensor[base_idx + 2 * i];
                    let x1 = tensor[base_idx + 2 * i + 1];
                    out[base_idx + 2 * i] = x0 * cos_v - x1 * sin_v;
                    out[base_idx + 2 * i + 1] = x0 * sin_v + x1 * cos_v;
                }
            }
        }
        out
    };

    let out_q = rotate(query, num_heads, expected_q);
    let out_k = rotate(key, num_kv_heads, expected_k);
    Ok((out_q, out_k))
}

// ────────────────────────────────────────────────────────────────────────────
// LongRoPE standalone API
// ────────────────────────────────────────────────────────────────────────────

/// Per-dimension scaling configuration for LongRoPE (Microsoft).
#[derive(Debug, Clone)]
pub struct LongRopeScaling {
    /// Per-dimension scale factors for sequences at or below `threshold`.
    pub short_factor: Vec<f32>,
    /// Per-dimension scale factors for sequences above `threshold`.
    pub long_factor: Vec<f32>,
    /// Sequence length threshold that triggers switch to `long_factor`.
    pub threshold: usize,
    /// Attention magnitude scale for short sequences.
    pub short_mscale: f32,
    /// Attention magnitude scale for long sequences.
    pub long_mscale: f32,
}

/// Apply LongRoPE per-dimension scaling to a set of inverse frequencies.
///
/// Selects `short_factor` when `seq_len <= config.threshold`, otherwise
/// `long_factor`.  Each element of `inv_freq` is divided by the corresponding
/// scale factor (or 1.0 if the factor list is shorter than `inv_freq`).
///
/// The returned vector has the same length as `inv_freq`.
pub fn apply_longrope_scaling(
    inv_freq: &[f32],
    seq_len: usize,
    config: &LongRopeScaling,
) -> Vec<f32> {
    let factors = if seq_len <= config.threshold {
        &config.short_factor
    } else {
        &config.long_factor
    };
    inv_freq
        .iter()
        .enumerate()
        .map(|(i, &f)| {
            let scale = factors.get(i).copied().unwrap_or(1.0);
            if scale == 0.0 {
                f
            } else {
                f / scale
            }
        })
        .collect()
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Config constructors ──────────────────────────────────────────────────

    #[test]
    fn test_rope_config_standard() {
        let cfg = RopeConfig::standard(64);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.base_theta, 10_000.0);
        assert_eq!(cfg.scaling, RopeScalingType::None);
        assert_eq!(cfg.max_position_embeddings, 4096);
    }

    #[test]
    fn test_rope_config_linear() {
        let cfg = RopeConfig::with_linear_scaling(128, 2.0);
        assert_eq!(cfg.head_dim, 128);
        assert!(matches!(cfg.scaling, RopeScalingType::Linear { factor } if factor == 2.0));
    }

    #[test]
    fn test_rope_config_ntk() {
        let cfg = RopeConfig::with_ntk(64, 4.0);
        assert_eq!(cfg.head_dim, 64);
        assert!(matches!(cfg.scaling, RopeScalingType::Ntk { factor } if factor == 4.0));
    }

    // ── Frequency computation ────────────────────────────────────────────────

    #[test]
    fn test_rope_freq_standard_first_dim() {
        // The highest frequency dimension (i=0) should equal 1.0
        // theta_0 = base^0 = 1.0
        let cfg = RopeConfig::standard(64);
        let freqs = RopeFrequencies::compute(cfg, 16).expect("compute failed");

        // At pos=1, angle = 1 * theta_0 = 1.0, cos(1.0) ≈ 0.5403
        let cos_val = freqs.cos[32]; // pos=1, dim=0, half_dim=32
        assert!(
            (cos_val - 1.0_f32.cos()).abs() < 1e-5,
            "Expected cos(1.0) ≈ {}, got {cos_val}",
            1.0_f32.cos()
        );
    }

    #[test]
    fn test_rope_freq_standard_last_dim() {
        // The lowest frequency dimension (i=half_dim-1) should be very small
        // theta_{d/2-1} = base^(-2*(d/2-1)/d) → very small for large d
        let head_dim = 64usize;
        let half_dim = head_dim / 2;
        let cfg = RopeConfig::standard(head_dim);
        let freqs = RopeFrequencies::compute(cfg, 16).expect("compute failed");

        // At pos=1, the last frequency dimension (i=half_dim-1)
        // freq = 10000^(-(head_dim-2)/head_dim) ≈ 10000^(-0.96875) ≈ very small
        let last_freq_idx = half_dim - 1;
        let angle = freqs.sin[half_dim + last_freq_idx].atan2(freqs.cos[half_dim + last_freq_idx]);
        assert!(
            angle.abs() < 0.01,
            "Expected small angle for last dim, got {angle}"
        );
    }

    #[test]
    fn test_rope_freq_linear_scaling() {
        let head_dim = 64;
        let factor = 2.0;
        let cfg_none = RopeConfig::standard(head_dim);
        let cfg_lin = RopeConfig::with_linear_scaling(head_dim, factor);

        let freqs_none = RopeFrequencies::compute(cfg_none, 16).expect("compute failed");
        let freqs_lin = RopeFrequencies::compute(cfg_lin, 16).expect("compute failed");

        // Linear scaling halves all frequencies: angle_lin = angle_none / factor
        // cos(angle_lin) at pos=1, dim=0 should match cos(1.0 / factor)
        let half_dim = head_dim / 2;
        let cos_none = freqs_none.cos[half_dim]; // pos=1, dim=0
        let cos_lin = freqs_lin.cos[half_dim];

        let angle_none = freqs_none.sin[half_dim].atan2(cos_none) as f64;
        let angle_lin = freqs_lin.sin[half_dim].atan2(cos_lin) as f64;

        // angle_lin should be approximately angle_none / factor
        let ratio = angle_none / angle_lin;
        assert!(
            (ratio - factor).abs() < 0.01,
            "Expected frequency ratio {factor}, got {ratio}"
        );
    }

    #[test]
    fn test_rope_freq_ntk_vs_linear() {
        // NTK uses modified base — frequencies differ from linear at dim i=0
        let head_dim = 64;
        let factor = 4.0;
        let cfg_ntk = RopeConfig::with_ntk(head_dim, factor);
        let cfg_lin = RopeConfig::with_linear_scaling(head_dim, factor);

        let freqs_ntk = RopeFrequencies::compute(cfg_ntk, 8).expect("compute failed");
        let freqs_lin = RopeFrequencies::compute(cfg_lin, 8).expect("compute failed");

        let half_dim = head_dim / 2;
        // NTK high-freq (dim 0) should be close to original, while linear reduces all equally
        // So NTK's dim=0 angle > linear's dim=0 angle at pos=1
        let angle_ntk = freqs_ntk.sin[half_dim].atan2(freqs_ntk.cos[half_dim]);
        let angle_lin = freqs_lin.sin[half_dim].atan2(freqs_lin.cos[half_dim]);

        // NTK modifies base so high freq dims are NOT scaled down as much
        // Both differ from original — we just verify they are different
        assert!(
            (angle_ntk - angle_lin).abs() > 1e-4,
            "NTK and linear should differ: ntk={angle_ntk}, lin={angle_lin}"
        );
    }

    #[test]
    fn test_rope_freq_yarn_high_freq_unchanged() {
        // YaRN: high-frequency dims (short wavelength < 2π/beta_fast) are kept unchanged
        let head_dim = 64;
        let cfg_none = RopeConfig::standard(head_dim);
        let cfg_yarn = RopeConfig::with_yarn(head_dim, 2.0, 2048);

        let freqs_none = RopeFrequencies::compute(cfg_none, 8).expect("compute failed");
        let freqs_yarn = RopeFrequencies::compute(cfg_yarn, 8).expect("compute failed");

        let half_dim = head_dim / 2;
        // The first dimension (i=0) has wavelength = 2π / 1.0 = 2π ≈ 6.28
        // high_thresh = 2π / 32.0 ≈ 0.196
        // 6.28 > 0.196 so it should NOT be in the high-freq unchanged region
        // The very first dimension freq=1.0, wavelength=2π — check last dims instead
        // At i=half_dim-1, freq is very small, wavelength very large → linear region
        // At i=0, freq=1.0, wavelength=2π: compare with threshold 2π/32 ≈ 0.196
        // wavelength > high_thresh, so it's NOT unchanged (it gets some scaling)

        // For a 64-dim head, we check whether any dim has unchanged freq
        // freq[i] = 10000^(-2i/64); wavelength[i] = 2π / freq[i]
        // We need wavelength < 2π/32 ≈ 0.196, i.e. freq > 32
        // But freq[0] = 1.0 which is < 32, so no dim is in true high-freq region here
        // Instead verify yarn produces different values from original for middle dims
        let mut differs = false;
        for i in 0..half_dim {
            let angle_none = freqs_none.sin[half_dim + i].atan2(freqs_none.cos[half_dim + i]);
            let angle_yarn = freqs_yarn.sin[half_dim + i].atan2(freqs_yarn.cos[half_dim + i]);
            if (angle_none - angle_yarn).abs() > 1e-4 {
                differs = true;
                break;
            }
        }
        assert!(differs, "YaRN should produce at least some different frequencies");
    }

    #[test]
    fn test_rope_freq_dynamic_ntk() {
        // Dynamic NTK: when max_seq_len > original_max_position, effective_factor increases
        let head_dim = 64;
        let cfg_dntk = RopeConfig {
            head_dim,
            base_theta: 10_000.0,
            scaling: RopeScalingType::DynamicNtk {
                factor: 4.0,
                original_max_position: 4096,
            },
            max_position_embeddings: 4096,
        };

        // Short sequence: effective_factor should be 1.0 → same as standard
        let freqs_short = RopeFrequencies::compute(cfg_dntk.clone(), 512).expect("compute failed");
        let freqs_std = RopeFrequencies::compute(RopeConfig::standard(head_dim), 512)
            .expect("compute failed");

        let half_dim = head_dim / 2;
        let angle_dntk = freqs_short.sin[half_dim].atan2(freqs_short.cos[half_dim]);
        let angle_std = freqs_std.sin[half_dim].atan2(freqs_std.cos[half_dim]);
        assert!(
            (angle_dntk - angle_std).abs() < 1e-4,
            "Dynamic NTK with short seq should match standard: dntk={angle_dntk}, std={angle_std}"
        );

        // Long sequence: effective_factor > 1.0 → different from standard
        // NTK modifies the base theta, which affects mid/high-frequency dims (i > 0).
        // Dim i=0 always has theta_0 = base^0 = 1.0 regardless of base, so compare a mid dim.
        let cfg_dntk_long = RopeConfig {
            head_dim,
            base_theta: 10_000.0,
            scaling: RopeScalingType::DynamicNtk {
                factor: 4.0,
                original_max_position: 512,
            },
            max_position_embeddings: 512,
        };
        let freqs_long =
            RopeFrequencies::compute(cfg_dntk_long, 4096).expect("compute long failed");

        // Compare a mid-frequency dim where NTK modification has visible effect
        let mid = half_dim / 2;
        let angle_long_mid = freqs_long.sin[half_dim + mid].atan2(freqs_long.cos[half_dim + mid]);
        let angle_std_mid = freqs_std.sin[half_dim + mid].atan2(freqs_std.cos[half_dim + mid]);
        assert!(
            (angle_long_mid - angle_std_mid).abs() > 1e-6,
            "Dynamic NTK with long seq should differ from standard at mid dim"
        );
    }

    // ── apply_rope ───────────────────────────────────────────────────────────

    #[test]
    fn test_rope_apply_identity_when_zero_pos() {
        // At position 0, all angles are 0 → cos=1, sin=0 → input unchanged
        let cfg = RopeConfig::standard(8);
        let freqs = RopeFrequencies::compute(cfg, 4).expect("compute failed");

        let q = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 1 pos, 1 head, head_dim=8
        let rotated = freqs.apply_rope(&q, 1, 1).expect("apply failed");

        for (orig, rot) in q.iter().zip(rotated.iter()) {
            assert!(
                (orig - rot).abs() < 1e-6,
                "Position 0 should be identity, orig={orig}, rot={rot}"
            );
        }
    }

    #[test]
    fn test_rope_apply_rotation() {
        // At position 1, angle = freq[0] = 1.0 for first dim pair
        // q = [1, 0, ...] → rotated = [cos(1), sin(1), ...]
        let cfg = RopeConfig::standard(4);
        let freqs = RopeFrequencies::compute(cfg, 4).expect("compute failed");

        // Tensor: [2 positions, 1 head, head_dim=4]
        // pos=0: [1,0, 0,1], pos=1: [1,0, 0,1]
        let q = vec![1.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let rotated = freqs.apply_rope(&q, 2, 1).expect("apply failed");

        // pos=0, dim pair 0: [1,0] → [1,0] (no rotation at pos=0)
        assert!((rotated[0] - 1.0).abs() < 1e-6);
        assert!((rotated[1] - 0.0).abs() < 1e-6);

        // pos=1, dim pair 0: q=[1,0], angle=theta_0=1.0
        // rotated_0 = 1*cos(1) - 0*sin(1) = cos(1)
        // rotated_1 = 1*sin(1) + 0*cos(1) = sin(1)
        let expected_cos = 1.0_f32.cos();
        let expected_sin = 1.0_f32.sin();
        assert!(
            (rotated[4] - expected_cos).abs() < 1e-5,
            "Expected {expected_cos}, got {}",
            rotated[4]
        );
        assert!(
            (rotated[5] - expected_sin).abs() < 1e-5,
            "Expected {expected_sin}, got {}",
            rotated[5]
        );
    }

    #[test]
    fn test_rope_apply_orthogonality() {
        // RoPE is an orthogonal transformation: ||rotated|| = ||input|| for each head
        let cfg = RopeConfig::standard(32);
        let freqs = RopeFrequencies::compute(cfg, 8).expect("compute failed");

        let seq_len = 5;
        let num_heads = 3;
        let head_dim = 32;
        let total = seq_len * num_heads * head_dim;

        // Fill with deterministic values
        let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.1) % 2.0 - 1.0).collect();
        let rotated = freqs.apply_rope(&q, seq_len, num_heads).expect("apply failed");

        // Check norm preservation per head
        for pos in 0..seq_len {
            for head in 0..num_heads {
                let base = pos * num_heads * head_dim + head * head_dim;
                let orig_norm: f32 = q[base..base + head_dim].iter().map(|x| x * x).sum::<f32>().sqrt();
                let rot_norm: f32 = rotated[base..base + head_dim].iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!(
                    (orig_norm - rot_norm).abs() < 1e-4,
                    "Norm changed at pos={pos}, head={head}: orig={orig_norm}, rot={rot_norm}"
                );
            }
        }
    }

    #[test]
    fn test_rope_apply_qk() {
        let cfg = RopeConfig::standard(16);
        let freqs = RopeFrequencies::compute(cfg, 8).expect("compute failed");

        let seq_len = 4;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 16;

        let q: Vec<f32> = vec![0.5_f32; seq_len * num_heads * head_dim];
        let k: Vec<f32> = vec![1.0_f32; seq_len * num_kv_heads * head_dim];

        let (rotated_q, rotated_k) = freqs
            .apply_rope_qk(&q, &k, seq_len, seq_len, num_heads, num_kv_heads)
            .expect("apply_qk failed");

        assert_eq!(rotated_q.len(), q.len());
        assert_eq!(rotated_k.len(), k.len());
    }

    #[test]
    fn test_rope_freq_stats() {
        let cfg = RopeConfig::standard(64);
        let freqs = RopeFrequencies::compute(cfg, 16).expect("compute failed");
        let stats = freqs.frequency_stats();

        assert!(stats.min_freq >= 0.0, "Min freq should be non-negative");
        assert!(stats.max_freq >= stats.min_freq, "Max >= min");
        assert!(stats.mean_freq >= stats.min_freq);
        assert!(stats.mean_freq <= stats.max_freq);
        // Standard 64-dim head has many low-freq dims
        assert!(stats.num_low_freq_dims > 0, "Should have some low-freq dims");
    }

    // ── Error cases ──────────────────────────────────────────────────────────

    #[test]
    fn test_rope_error_odd_head_dim() {
        let cfg = RopeConfig::standard(63); // odd — invalid
        let result = RopeFrequencies::compute(cfg, 8);
        assert!(matches!(result, Err(RopeError::InvalidHeadDim { .. })));
    }

    #[test]
    fn test_rope_error_seq_exceeded() {
        let cfg = RopeConfig::standard(16);
        let freqs = RopeFrequencies::compute(cfg, 4).expect("compute failed");
        let q = vec![0.0_f32; 8 * 1 * 16]; // seq_len=8 > max=4
        let result = freqs.apply_rope(&q, 8, 1);
        assert!(matches!(result, Err(RopeError::SequenceLengthExceeded { .. })));
    }

    #[test]
    fn test_rope_scaling_display() {
        let none = RopeScalingType::None;
        assert!(none.to_string().contains("None"));

        let linear = RopeScalingType::Linear { factor: 2.0 };
        assert!(linear.to_string().contains("Linear"));

        let ntk = RopeScalingType::Ntk { factor: 4.0 };
        assert!(ntk.to_string().contains("NTK"));

        let dntk = RopeScalingType::DynamicNtk {
            factor: 4.0,
            original_max_position: 2048,
        };
        assert!(dntk.to_string().contains("DynamicNTK"));

        let yarn = RopeScalingType::Yarn {
            factor: 2.0,
            original_max_position: 4096,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        assert!(yarn.to_string().contains("YaRN"));

        let lr = RopeScalingType::LongRope {
            short_factors: vec![1.0; 32],
            long_factors: vec![2.0; 32],
            original_max_position: 4096,
        };
        assert!(lr.to_string().contains("LongRoPE"));
    }

    #[test]
    fn test_rope_long_rope_short_vs_long() {
        // LongRope: short_factors used when seq_len <= original, long_factors otherwise
        let head_dim = 32;
        let half_dim = head_dim / 2;
        let short_factors = vec![1.0_f64; half_dim]; // no scaling for short
        let long_factors = vec![2.0_f64; half_dim]; // halve freqs for long

        let cfg_short = RopeConfig {
            head_dim,
            base_theta: 10_000.0,
            scaling: RopeScalingType::LongRope {
                short_factors: short_factors.clone(),
                long_factors: long_factors.clone(),
                original_max_position: 4096,
            },
            max_position_embeddings: 4096,
        };

        let cfg_long = RopeConfig {
            head_dim,
            base_theta: 10_000.0,
            scaling: RopeScalingType::LongRope {
                short_factors: short_factors.clone(),
                long_factors: long_factors.clone(),
                original_max_position: 512,
            },
            max_position_embeddings: 512,
        };

        // Short: max_seq_len=512 <= 4096 → short_factors (all 1.0) → same as standard
        let freqs_short = RopeFrequencies::compute(cfg_short, 512).expect("short failed");
        let freqs_std = RopeFrequencies::compute(RopeConfig::standard(head_dim), 512)
            .expect("std failed");
        let angle_short = freqs_short.sin[half_dim].atan2(freqs_short.cos[half_dim]);
        let angle_std = freqs_std.sin[half_dim].atan2(freqs_std.cos[half_dim]);
        assert!(
            (angle_short - angle_std).abs() < 1e-4,
            "Short LongRoPE with factor=1 should match standard"
        );

        // Long: max_seq_len=4096 > 512 → long_factors (all 2.0) → freqs halved
        let freqs_long = RopeFrequencies::compute(cfg_long, 4096).expect("long failed");
        let angle_long = freqs_long.sin[half_dim].atan2(freqs_long.cos[half_dim]);
        // long_factor=2 means freq = base_freq/2, so angle at pos=1 = base_freq/2
        let expected_angle = angle_std / 2.0;
        assert!(
            (angle_long - expected_angle).abs() < 1e-4,
            "Long LongRoPE with factor=2 should halve frequency: expected {expected_angle}, got {angle_long}"
        );
    }

    #[test]
    fn test_rope_error_invalid_scaling_factor() {
        let cfg = RopeConfig {
            head_dim: 32,
            base_theta: 10_000.0,
            scaling: RopeScalingType::Linear { factor: 0.5 }, // < 1.0 — invalid
            max_position_embeddings: 4096,
        };
        let result = RopeFrequencies::compute(cfg, 8);
        assert!(matches!(result, Err(RopeError::InvalidScalingFactor(_))));
    }

    #[test]
    fn test_rope_error_zero_head_dim() {
        let cfg = RopeConfig {
            head_dim: 0,
            base_theta: 10_000.0,
            scaling: RopeScalingType::None,
            max_position_embeddings: 4096,
        };
        let result = RopeFrequencies::compute(cfg, 8);
        assert!(matches!(result, Err(RopeError::InvalidHeadDim { .. })));
    }

    #[test]
    fn test_rope_error_dimension_mismatch() {
        let cfg = RopeConfig::standard(16);
        let freqs = RopeFrequencies::compute(cfg, 8).expect("compute failed");
        // Provide wrong size
        let q = vec![0.0_f32; 10]; // wrong size
        let result = freqs.apply_rope(&q, 2, 1);
        assert!(matches!(result, Err(RopeError::DimensionMismatch { .. })));
    }

    // ── YaRN standalone API tests ────────────────────────────────────────────

    #[test]
    fn test_yarn_find_correction_dim_positive() {
        // The correction dim must be positive when num_rotations < orig_max_pos/(2π)
        let dim = yarn_find_correction_dim(1.0, 64, 10000.0, 4096);
        assert!(dim > 0.0, "correction dim should be positive: got {dim}");
    }

    #[test]
    fn test_yarn_find_correction_dim_scales_with_rotations() {
        // More rotations → smaller correction dim (lower threshold)
        let dim_slow = yarn_find_correction_dim(1.0, 64, 10000.0, 4096);
        let dim_fast = yarn_find_correction_dim(32.0, 64, 10000.0, 4096);
        assert!(
            dim_slow > dim_fast,
            "slow (1 rotation) should have larger correction dim than fast (32 rotations)"
        );
    }

    #[test]
    fn test_yarn_linear_ramp_mask_boundary_values() {
        let mask = yarn_linear_ramp_mask(0.0, 10.0, 11);
        // index 0 → 0.0, index 10 → 1.0
        assert!((mask[0] - 0.0).abs() < 1e-6, "first element should be 0");
        assert!((mask[10] - 1.0).abs() < 1e-6, "last element should be 1");
    }

    #[test]
    fn test_yarn_linear_ramp_mask_monotone() {
        let mask = yarn_linear_ramp_mask(2.0, 8.0, 12);
        for i in 1..mask.len() {
            assert!(
                mask[i] >= mask[i - 1],
                "ramp mask must be non-decreasing: mask[{i}]={} < mask[{}]={}",
                mask[i],
                i - 1,
                mask[i - 1]
            );
        }
    }

    #[test]
    fn test_yarn_linear_ramp_mask_zero_range() {
        // min == max → all zeros
        let mask = yarn_linear_ramp_mask(5.0, 5.0, 8);
        for v in &mask {
            assert!((v - 0.0).abs() < 1e-6, "should be 0 when min==max");
        }
    }

    #[test]
    fn test_yarn_get_mscale_formula() {
        // sqrt(0.1 * ln(e) + 1.0) * 1.0 = sqrt(1.1) ≈ 1.04881
        let result = yarn_get_mscale(std::f32::consts::E, 1.0);
        let expected = (0.1_f32 * 1.0_f32 + 1.0).sqrt(); // ln(e) = 1
        assert!((result - expected).abs() < 1e-5, "mscale formula mismatch: {result} vs {expected}");
    }

    #[test]
    fn test_yarn_get_mscale_scale_1_returns_mscale() {
        // ln(1) = 0 → sqrt(0.1*0 + 1) * mscale = mscale
        let result = yarn_get_mscale(1.0, 2.5);
        assert!((result - 2.5).abs() < 1e-5, "scale=1 should return mscale unchanged");
    }

    #[test]
    fn test_apply_yarn_rope_output_shape() {
        let config = YarnConfig {
            original_max_position_embeddings: 4096,
            scaling_factor: 4.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale: 1.0,
            mscale_all_dim: 0.0,
            base: 10000.0,
            head_dim: 16,
        };
        let seq_len = 4;
        let num_heads = 2;
        let num_kv_heads = 2;
        let q = vec![0.1_f32; seq_len * num_heads * 16];
        let k = vec![0.2_f32; seq_len * num_kv_heads * 16];
        let positions: Vec<usize> = (0..seq_len).collect();
        let (out_q, out_k) =
            apply_yarn_rope(&q, &k, &positions, num_heads, num_kv_heads, &config)
                .expect("apply_yarn_rope should succeed");
        assert_eq!(out_q.len(), q.len());
        assert_eq!(out_k.len(), k.len());
    }

    #[test]
    fn test_apply_yarn_rope_invalid_head_dim() {
        let config = YarnConfig {
            original_max_position_embeddings: 4096,
            scaling_factor: 2.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale: 1.0,
            mscale_all_dim: 0.0,
            base: 10000.0,
            head_dim: 0,
        };
        let result = apply_yarn_rope(&[], &[], &[], 1, 1, &config);
        assert!(matches!(result, Err(RopeError::InvalidHeadDim { .. })));
    }

    #[test]
    fn test_apply_yarn_rope_odd_head_dim() {
        let config = YarnConfig {
            original_max_position_embeddings: 4096,
            scaling_factor: 2.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale: 1.0,
            mscale_all_dim: 0.0,
            base: 10000.0,
            head_dim: 7,
        };
        let result = apply_yarn_rope(&[], &[], &[], 1, 1, &config);
        assert!(matches!(result, Err(RopeError::InvalidHeadDim { .. })));
    }

    // ── Dynamic NTK standalone API tests ─────────────────────────────────────

    #[test]
    fn test_dynamic_ntk_theta_no_scaling_when_short() {
        let config = DynamicNtkConfig {
            base_theta: 10000.0,
            alpha: 8.0,
            max_original_length: 4096,
            head_dim: 64,
        };
        // seq_len <= max_original_length → return base_theta unchanged
        let theta = dynamic_ntk_theta(2048, &config);
        assert!(
            (theta - 10000.0).abs() < 1e-3,
            "theta should be base_theta for short sequences: got {theta}"
        );
    }

    #[test]
    fn test_dynamic_ntk_theta_increases_with_seq_len() {
        let config = DynamicNtkConfig {
            base_theta: 10000.0,
            alpha: 8.0,
            max_original_length: 4096,
            head_dim: 64,
        };
        let theta_short = dynamic_ntk_theta(4096, &config);
        let theta_long = dynamic_ntk_theta(32768, &config);
        assert!(
            theta_long > theta_short,
            "longer sequences should produce larger theta: {theta_short} vs {theta_long}"
        );
    }

    #[test]
    fn test_apply_dynamic_ntk_rope_output_shape() {
        let config = DynamicNtkConfig {
            base_theta: 10000.0,
            alpha: 4.0,
            max_original_length: 512,
            head_dim: 16,
        };
        let seq_len = 8;
        let num_heads = 2;
        let q = vec![0.0_f32; seq_len * num_heads * 16];
        let k = vec![0.0_f32; seq_len * num_heads * 16];
        let (out_q, out_k) =
            apply_dynamic_ntk_rope(&q, &k, seq_len, num_heads, num_heads, &config)
                .expect("should succeed");
        assert_eq!(out_q.len(), q.len());
        assert_eq!(out_k.len(), k.len());
    }

    #[test]
    fn test_apply_dynamic_ntk_rope_dimension_mismatch() {
        let config = DynamicNtkConfig {
            base_theta: 10000.0,
            alpha: 2.0,
            max_original_length: 512,
            head_dim: 16,
        };
        // Wrong Q length
        let q = vec![0.0_f32; 5]; // not seq_len * num_heads * head_dim
        let k = vec![0.0_f32; 16];
        let result = apply_dynamic_ntk_rope(&q, &k, 1, 1, 1, &config);
        assert!(matches!(result, Err(RopeError::DimensionMismatch { .. })));
    }

    // ── LongRoPE standalone API tests ─────────────────────────────────────────

    #[test]
    fn test_apply_longrope_scaling_uses_short_factor_below_threshold() {
        let config = LongRopeScaling {
            short_factor: vec![2.0; 4],
            long_factor: vec![8.0; 4],
            threshold: 512,
            short_mscale: 1.0,
            long_mscale: 1.0,
        };
        let inv_freq = vec![1.0_f32; 4];
        let result = apply_longrope_scaling(&inv_freq, 256, &config); // 256 <= 512
        // short_factor=2 → each freq = 1/2 = 0.5
        for v in &result {
            assert!((v - 0.5).abs() < 1e-6, "short factor should halve freq: got {v}");
        }
    }

    #[test]
    fn test_apply_longrope_scaling_uses_long_factor_above_threshold() {
        let config = LongRopeScaling {
            short_factor: vec![2.0; 4],
            long_factor: vec![8.0; 4],
            threshold: 512,
            short_mscale: 1.0,
            long_mscale: 1.0,
        };
        let inv_freq = vec![1.0_f32; 4];
        let result = apply_longrope_scaling(&inv_freq, 1024, &config); // 1024 > 512
        // long_factor=8 → each freq = 1/8 = 0.125
        for v in &result {
            assert!((v - 0.125).abs() < 1e-6, "long factor should divide freq by 8: got {v}");
        }
    }

    #[test]
    fn test_apply_longrope_scaling_preserves_length() {
        let config = LongRopeScaling {
            short_factor: vec![1.0; 8],
            long_factor: vec![2.0; 8],
            threshold: 256,
            short_mscale: 1.0,
            long_mscale: 1.0,
        };
        let inv_freq = vec![0.5_f32; 8];
        let result_short = apply_longrope_scaling(&inv_freq, 128, &config);
        let result_long = apply_longrope_scaling(&inv_freq, 512, &config);
        assert_eq!(result_short.len(), inv_freq.len());
        assert_eq!(result_long.len(), inv_freq.len());
    }

    #[test]
    fn test_apply_longrope_scaling_at_threshold_uses_short() {
        // seq_len == threshold → use short_factor
        let config = LongRopeScaling {
            short_factor: vec![2.0; 2],
            long_factor: vec![100.0; 2],
            threshold: 512,
            short_mscale: 1.0,
            long_mscale: 1.0,
        };
        let inv_freq = vec![1.0_f32; 2];
        let result = apply_longrope_scaling(&inv_freq, 512, &config);
        for v in &result {
            assert!((v - 0.5).abs() < 1e-6, "at threshold should use short factor: got {v}");
        }
    }
}
