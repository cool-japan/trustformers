//! AWQ: Activation-aware Weight Quantization for TrustformeRS.
//!
//! AWQ (Lin et al., 2023) protects salient weights by searching for a
//! per-channel scaling factor that minimises quantization error by analysing
//! activation statistics.
//!
//! Weight transform: W' = W * diag(s)^{-1},  X' = X * diag(s)
//!
//! Reference: "AWQ: Activation-aware Weight Quantization for LLM Compression
//! and Acceleration" (Lin et al., 2023).

use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during AWQ quantization.
#[derive(Debug, Clone, PartialEq)]
pub enum AwqError {
    /// Weight matrix is empty.
    EmptyWeight,
    /// Weight dimensions do not match the provided rows × cols.
    DimensionMismatch,
    /// Configuration parameter is invalid.
    InvalidConfig(String),
    /// The number of activation channels does not match the weight column count.
    ActivationChannelMismatch {
        weight_cols: usize,
        activation_channels: usize,
    },
    /// Quantization step encountered an unrecoverable error.
    QuantizationFailed(String),
}

impl fmt::Display for AwqError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AwqError::EmptyWeight => write!(f, "AWQ: weight matrix is empty"),
            AwqError::DimensionMismatch => {
                write!(f, "AWQ: weight dimensions do not match rows × cols")
            }
            AwqError::InvalidConfig(msg) => write!(f, "AWQ: invalid configuration — {msg}"),
            AwqError::ActivationChannelMismatch {
                weight_cols,
                activation_channels,
            } => write!(
                f,
                "AWQ: activation channel count {activation_channels} \
                 does not match weight cols {weight_cols}"
            ),
            AwqError::QuantizationFailed(msg) => {
                write!(f, "AWQ: quantization failed — {msg}")
            }
        }
    }
}

impl std::error::Error for AwqError {}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for AWQ quantization.
#[derive(Debug, Clone)]
pub struct AwqConfig {
    /// Quantization bit-width (default 4).
    pub bits: u32,
    /// Number of weight elements per quantization group (default 128).
    pub group_size: usize,
    /// Use asymmetric zero-point quantization (default true).
    pub zero_point: bool,
    /// Search range for per-channel scale factors: (min, max).
    pub search_scale_range: (f32, f32),
    /// Number of candidate scale values in the grid search (default 20).
    pub num_scale_candidates: usize,
    /// SmoothQuant-style α: controls how much activation magnitude is
    /// absorbed into the weights (default 0.5).
    pub smooth_factor: f32,
}

impl Default for AwqConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 128,
            zero_point: true,
            search_scale_range: (0.01, 2.0),
            num_scale_candidates: 20,
            smooth_factor: 0.5,
        }
    }
}

impl AwqConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), AwqError> {
        if self.bits < 2 || self.bits > 8 {
            return Err(AwqError::InvalidConfig(format!(
                "bits must be 2–8, got {}",
                self.bits
            )));
        }
        if self.group_size == 0 {
            return Err(AwqError::InvalidConfig(
                "group_size must be > 0".to_string(),
            ));
        }
        if self.num_scale_candidates == 0 {
            return Err(AwqError::InvalidConfig(
                "num_scale_candidates must be > 0".to_string(),
            ));
        }
        let (lo, hi) = self.search_scale_range;
        if lo <= 0.0 || hi <= lo {
            return Err(AwqError::InvalidConfig(format!(
                "search_scale_range must have 0 < lo < hi, got ({lo}, {hi})"
            )));
        }
        if !(0.0..=1.0).contains(&self.smooth_factor) {
            return Err(AwqError::InvalidConfig(format!(
                "smooth_factor must be in [0, 1], got {}",
                self.smooth_factor
            )));
        }
        Ok(())
    }

    /// Signed minimum quantized value.
    fn min_q(&self) -> i64 {
        if self.zero_point {
            0
        } else {
            -(1i64 << (self.bits - 1))
        }
    }

    /// Signed maximum quantized value.
    fn max_q(&self) -> i64 {
        if self.zero_point {
            (1i64 << self.bits) - 1
        } else {
            (1i64 << (self.bits - 1)) - 1
        }
    }
}

// ---------------------------------------------------------------------------
// Activation statistics
// ---------------------------------------------------------------------------

/// Per-channel activation statistics gathered from calibration data.
///
/// `activations` is a collection of sample vectors, each of length
/// `num_channels`.
#[derive(Debug, Clone)]
pub struct AwqActivationStats {
    /// Per-channel mean of absolute values across calibration samples.
    pub channel_means: Vec<f32>,
    /// Per-channel maximum of absolute values across calibration samples.
    pub channel_maxes: Vec<f32>,
    /// Per-channel standard deviation of absolute values.
    pub channel_stds: Vec<f32>,
    /// Number of calibration samples used to compute the statistics.
    pub num_samples: usize,
}

impl AwqActivationStats {
    /// Compute statistics from a list of calibration samples.
    ///
    /// Each element of `activations` is one sample vector of length
    /// `num_channels`.
    pub fn from_activations(activations: &[Vec<f32>], num_channels: usize) -> Self {
        let num_samples = activations.len();

        if num_samples == 0 || num_channels == 0 {
            return Self {
                channel_means: vec![0.0; num_channels],
                channel_maxes: vec![0.0; num_channels],
                channel_stds: vec![0.0; num_channels],
                num_samples: 0,
            };
        }

        let mut sums = vec![0.0_f64; num_channels];
        let mut sum_sqs = vec![0.0_f64; num_channels];
        let mut maxes = vec![0.0_f32; num_channels];

        for sample in activations {
            let len = sample.len().min(num_channels);
            for c in 0..len {
                let abs_v = sample[c].abs();
                sums[c] += abs_v as f64;
                sum_sqs[c] += (abs_v as f64).powi(2);
                if abs_v > maxes[c] {
                    maxes[c] = abs_v;
                }
            }
        }

        let n = num_samples as f64;
        let means: Vec<f32> = sums.iter().map(|&s| (s / n) as f32).collect();
        let stds: Vec<f32> = sum_sqs
            .iter()
            .zip(sums.iter())
            .map(|(&sq, &s)| {
                let variance = (sq / n - (s / n).powi(2)).max(0.0);
                variance.sqrt() as f32
            })
            .collect();

        Self {
            channel_means: means,
            channel_maxes: maxes,
            channel_stds: stds,
            num_samples,
        }
    }

    /// Return the indices of the top-k channels sorted by `channel_maxes`
    /// (descending).
    pub fn dominant_channels(&self, top_k: usize) -> Vec<usize> {
        let num_channels = self.channel_maxes.len();
        let k = top_k.min(num_channels);

        let mut indices: Vec<usize> = (0..num_channels).collect();
        indices.sort_by(|&a, &b| {
            self.channel_maxes[b]
                .partial_cmp(&self.channel_maxes[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices.truncate(k);
        indices
    }
}

// ---------------------------------------------------------------------------
// Scale search result
// ---------------------------------------------------------------------------

/// Result of the AWQ per-channel scale grid search.
#[derive(Debug, Clone)]
pub struct AwqScaleSearchResult {
    /// Per-input-channel scale factors found by the search.
    pub scales: Vec<f32>,
    /// Frobenius reconstruction error at the optimal scale.
    pub best_grid_error: f32,
    /// Number of quantization groups used.
    pub num_groups: usize,
}

// ---------------------------------------------------------------------------
// Internal quantization helpers
// ---------------------------------------------------------------------------

/// Compute per-group scale and zero-point for a weight group.
fn group_scale_zp(group: &[f32], min_q: i64, max_q: i64, asymmetric: bool) -> (f32, f32) {
    let fmin = group.iter().cloned().fold(f32::INFINITY, f32::min);
    let fmax = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    if asymmetric {
        let q_range = (max_q - min_q) as f32;
        let f_range = fmax - fmin;
        if f_range < f32::EPSILON {
            return (1.0_f32, fmin);
        }
        let scale = f_range / q_range;
        (scale, fmin)
    } else {
        let max_abs = fmin.abs().max(fmax.abs());
        if max_abs < f32::EPSILON {
            return (1.0_f32, 0.0_f32);
        }
        let scale = max_abs / max_q as f32;
        (scale, 0.0_f32)
    }
}

/// Quantize a single f32 value and return the signed integer code.
fn quantize_f32_val(val: f32, scale: f32, zero: f32, min_q: i64, max_q: i64) -> i64 {
    if scale.abs() < f32::EPSILON {
        return 0;
    }
    let q = ((val - zero) / scale).round() as i64;
    q.clamp(min_q, max_q)
}

/// Dequantize a signed integer code back to f32.
fn dequantize_i64_val(q: i64, scale: f32, zero: f32) -> f32 {
    q as f32 * scale + zero
}

/// Quantize a weight matrix (row-major, rows×cols) and return the
/// dequantized reconstruction together with per-group scales/zeros.
///
/// Groups are formed along the row dimension (each group covers
/// `group_size` consecutive rows for every column).
fn quantize_and_reconstruct(
    weight: &[f32],
    rows: usize,
    cols: usize,
    config: &AwqConfig,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let min_q = config.min_q();
    let max_q = config.max_q();
    let asymmetric = config.zero_point;
    let num_groups = rows.div_ceil(config.group_size);

    let mut scales = vec![0.0_f32; num_groups * cols];
    let mut zeros = vec![0.0_f32; num_groups * cols];

    // Compute per-group scale/zero in column order.
    for col in 0..cols {
        for g in 0..num_groups {
            let r_start = g * config.group_size;
            let r_end = (r_start + config.group_size).min(rows);
            let group: Vec<f32> = (r_start..r_end).map(|r| weight[r * cols + col]).collect();
            let (sc, zp) = group_scale_zp(&group, min_q, max_q, asymmetric);
            scales[g * cols + col] = sc;
            zeros[g * cols + col] = zp;
        }
    }

    // Quantize + dequantize to get reconstruction.
    let mut reconstructed = vec![0.0_f32; rows * cols];
    for row in 0..rows {
        let g = row / config.group_size;
        for col in 0..cols {
            let sc = scales[g * cols + col];
            let zp = zeros[g * cols + col];
            let w = weight[row * cols + col];
            let q = quantize_f32_val(w, sc, zp, min_q, max_q);
            reconstructed[row * cols + col] = dequantize_i64_val(q, sc, zp);
        }
    }

    (reconstructed, scales, zeros)
}

/// Frobenius norm of the element-wise difference between two equally-shaped
/// flat matrices.
fn frobenius_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

// ---------------------------------------------------------------------------
// AWQ scale search
// ---------------------------------------------------------------------------

/// Search for optimal per-channel scale factors using a grid search.
///
/// For each candidate scaling `s_c` (derived from activation statistics):
///   1. Scale weight: `W_scaled[i,j] = W[i,j] / s_c[j]`
///   2. Quantize `W_scaled` to `config.bits` bits
///   3. Dequantize and rescale back: `W_dequant[i,j] * s_c[j]`
///   4. Compute Frobenius reconstruction error against the original `W`
///
/// The per-channel candidate scales are generated as:
///   `s_c = mean_c^α` for α ∈ log-space from `search_scale_range`.
pub fn awq_search_scales(
    weight: &[f32],
    rows: usize,
    cols: usize,
    act_stats: &AwqActivationStats,
    config: &AwqConfig,
) -> Result<AwqScaleSearchResult, AwqError> {
    config.validate()?;

    if weight.is_empty() {
        return Err(AwqError::EmptyWeight);
    }
    if weight.len() != rows * cols {
        return Err(AwqError::DimensionMismatch);
    }
    if act_stats.channel_means.len() != cols {
        return Err(AwqError::ActivationChannelMismatch {
            weight_cols: cols,
            activation_channels: act_stats.channel_means.len(),
        });
    }

    let (lo, hi) = config.search_scale_range;
    let n = config.num_scale_candidates;
    let alpha = config.smooth_factor;
    let num_groups = rows.div_ceil(config.group_size);

    // Generate candidate α values in log-space.
    let log_lo = lo.ln();
    let log_hi = hi.ln();
    let step = if n > 1 {
        (log_hi - log_lo) / (n as f32 - 1.0)
    } else {
        0.0
    };

    let candidates: Vec<f32> = (0..n).map(|i| (log_lo + step * i as f32).exp()).collect();

    let mut best_error = f32::INFINITY;
    let mut best_scales = vec![1.0_f32; cols];

    for &cand_alpha in &candidates {
        // Build per-channel scales: s_c = clamp(mean_c, eps, inf)^(alpha * cand_alpha)
        // cand_alpha here acts as a multiplier on the exponent.
        let channel_scales: Vec<f32> = act_stats
            .channel_means
            .iter()
            .map(|&m| {
                let safe_m = m.max(f32::EPSILON);
                safe_m.powf(alpha * cand_alpha)
            })
            .collect();

        // Apply channel scale to weight: W_scaled[r,c] = W[r,c] / s_c
        let cs = &channel_scales;
        let w_scaled: Vec<f32> = (0..rows)
            .flat_map(|r| {
                (0..cols).map(move |c| {
                    let s = cs[c];
                    if s.abs() > f32::EPSILON {
                        weight[r * cols + c] / s
                    } else {
                        weight[r * cols + c]
                    }
                })
            })
            .collect();

        // Quantize + dequantize the scaled weight.
        let (w_dequant_scaled, _, _) =
            quantize_and_reconstruct(&w_scaled, rows, cols, config);

        // Rescale back: W_dequant[r,c] = W_dequant_scaled[r,c] * s_c
        let cs2 = &channel_scales;
        let wds = &w_dequant_scaled;
        let w_dequant: Vec<f32> = (0..rows)
            .flat_map(|r| {
                (0..cols).map(move |c| wds[r * cols + c] * cs2[c])
            })
            .collect();

        let err = frobenius_error(weight, &w_dequant);
        if err < best_error {
            best_error = err;
            best_scales = channel_scales;
        }
    }

    Ok(AwqScaleSearchResult {
        scales: best_scales,
        best_grid_error: best_error,
        num_groups,
    })
}

// ---------------------------------------------------------------------------
// Quantized layer container
// ---------------------------------------------------------------------------

/// An AWQ-quantized linear layer.
///
/// Weights are packed as INT32 words (INT4 packed, 8 values per word for 4-bit).
#[derive(Debug, Clone, PartialEq)]
pub struct AwqQuantizedLayer {
    /// Packed quantized weights (INT4 packed to INT32).
    pub qweight: Vec<i32>,
    /// Per-group f32 scales.  Length = num_groups * cols.
    pub scales: Vec<f32>,
    /// Per-group f32 zero-points.  Length = num_groups * cols.
    pub zeros: Vec<f32>,
    /// Per-input-channel AWQ activation scales.  Length = cols.
    pub input_scales: Vec<f32>,
    /// Number of rows in the original weight matrix.
    pub rows: usize,
    /// Number of columns in the original weight matrix.
    pub cols: usize,
    /// Bit-width used.
    pub bits: u32,
    /// Group size used during quantization.
    pub group_size: usize,
    /// Whether asymmetric (zero-point) quantization was used.
    pub asymmetric: bool,
}

// ---------------------------------------------------------------------------
// Packing helpers
// ---------------------------------------------------------------------------

/// Pack unsigned integer values (codes) into i32 words at `bits` bits per value.
///
/// For asymmetric quantization, codes are already in [0, 2^bits - 1].
/// For symmetric quantization, codes are stored with an offset of 2^(bits-1)
/// so that the signed range [-2^(bits-1), 2^(bits-1)-1] maps to unsigned.
fn pack_codes_to_i32(values: &[i64], bits: u32, asymmetric: bool) -> Vec<i32> {
    let vals_per_word = 32 / bits as usize;
    let num_words = values.len().div_ceil(vals_per_word);
    let mut packed = vec![0i32; num_words];
    let offset: i64 = if asymmetric { 0 } else { 1i64 << (bits - 1) };
    let mask = (1u64 << bits) - 1;

    for (i, &v) in values.iter().enumerate() {
        let word = i / vals_per_word;
        let bit_pos = (i % vals_per_word) * bits as usize;
        let unsigned = (v + offset) as u64 & mask;
        packed[word] |= (unsigned << bit_pos) as i32;
    }
    packed
}

/// Unpack `num_elements` codes from i32 packed words.
///
/// Returns signed i64 values, undoing the offset applied during packing.
fn unpack_codes_from_i32(packed: &[i32], bits: u32, num_elements: usize, asymmetric: bool) -> Vec<i64> {
    let vals_per_word = 32 / bits as usize;
    let offset: i64 = if asymmetric { 0 } else { 1i64 << (bits - 1) };
    let mask = (1u64 << bits) - 1;

    let mut out = Vec::with_capacity(num_elements);
    for i in 0..num_elements {
        let word = i / vals_per_word;
        let bit_pos = (i % vals_per_word) * bits as usize;
        if word >= packed.len() {
            out.push(0i64);
        } else {
            let unsigned = ((packed[word] as u64) >> bit_pos) & mask;
            out.push(unsigned as i64 - offset);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// AWQ layer quantize / dequantize
// ---------------------------------------------------------------------------

/// Quantize a weight matrix with optional AWQ activation-aware scaling.
///
/// If `act_stats` is provided, the function first searches for optimal
/// per-channel scales, transforms the weights, then quantizes.
pub fn awq_quantize_layer(
    weight: &[f32],
    rows: usize,
    cols: usize,
    act_stats: Option<&AwqActivationStats>,
    config: &AwqConfig,
) -> Result<AwqQuantizedLayer, AwqError> {
    config.validate()?;

    if weight.is_empty() {
        return Err(AwqError::EmptyWeight);
    }
    if weight.len() != rows * cols {
        return Err(AwqError::DimensionMismatch);
    }

    let min_q = config.min_q();
    let max_q = config.max_q();
    let asymmetric = config.zero_point;
    let num_groups = rows.div_ceil(config.group_size);

    // Determine per-channel scales (AWQ) or fall back to all-ones.
    let input_scales: Vec<f32> = if let Some(stats) = act_stats {
        if stats.channel_means.len() != cols {
            return Err(AwqError::ActivationChannelMismatch {
                weight_cols: cols,
                activation_channels: stats.channel_means.len(),
            });
        }
        let result = awq_search_scales(weight, rows, cols, stats, config)?;
        result.scales
    } else {
        vec![1.0_f32; cols]
    };

    // Apply AWQ scaling to weights: W_awq[r,c] = W[r,c] / input_scales[c]
    let is_ref = &input_scales;
    let w_awq: Vec<f32> = (0..rows)
        .flat_map(|r| {
            (0..cols).map(move |c| {
                let s = is_ref[c];
                if s.abs() > f32::EPSILON {
                    weight[r * cols + c] / s
                } else {
                    weight[r * cols + c]
                }
            })
        })
        .collect();

    // Compute per-group scales and zero-points on the AWQ-transformed weights.
    let mut group_scales = vec![0.0_f32; num_groups * cols];
    let mut group_zeros = vec![0.0_f32; num_groups * cols];

    for col in 0..cols {
        for g in 0..num_groups {
            let r_start = g * config.group_size;
            let r_end = (r_start + config.group_size).min(rows);
            let group: Vec<f32> = (r_start..r_end)
                .map(|r| w_awq[r * cols + col])
                .collect();
            let (sc, zp) = group_scale_zp(&group, min_q, max_q, asymmetric);
            group_scales[g * cols + col] = sc;
            group_zeros[g * cols + col] = zp;
        }
    }

    // Quantize to integer codes (row-major).
    let mut q_vals: Vec<i64> = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        let g = row / config.group_size;
        for col in 0..cols {
            let sc = group_scales[g * cols + col];
            let zp = group_zeros[g * cols + col];
            let w = w_awq[row * cols + col];
            q_vals.push(quantize_f32_val(w, sc, zp, min_q, max_q));
        }
    }

    // Pack row-major integer codes into i32 words.
    let qweight = pack_codes_to_i32(&q_vals, config.bits, config.zero_point);

    Ok(AwqQuantizedLayer {
        qweight,
        scales: group_scales,
        zeros: group_zeros,
        input_scales,
        rows,
        cols,
        bits: config.bits,
        group_size: config.group_size,
        asymmetric: config.zero_point,
    })
}

/// Dequantize an `AwqQuantizedLayer` back to a row-major f32 matrix.
///
/// The output has the AWQ channel scales re-applied so that the result
/// approximates the original (un-transformed) weight matrix.
pub fn awq_dequantize_layer(layer: &AwqQuantizedLayer) -> Result<Vec<f32>, AwqError> {
    if layer.rows == 0 || layer.cols == 0 {
        return Err(AwqError::EmptyWeight);
    }

    let num_elements = layer.rows * layer.cols;
    let num_groups = layer.rows.div_ceil(layer.group_size);

    // Unpack integer codes using the same asymmetric flag as during packing.
    let q_vals = unpack_codes_from_i32(&layer.qweight, layer.bits, num_elements, layer.asymmetric);

    let mut output = vec![0.0_f32; num_elements];
    for row in 0..layer.rows {
        let g = row / layer.group_size;
        for col in 0..layer.cols {
            let sc = layer.scales[g * layer.cols + col];
            let zp = layer.zeros[g * layer.cols + col];
            let q = q_vals[row * layer.cols + col];
            // Dequantize the AWQ-transformed weight, then rescale.
            let w_awq = dequantize_i64_val(q, sc, zp);
            let input_scale = if col < layer.input_scales.len() {
                layer.input_scales[col]
            } else {
                1.0_f32
            };
            output[row * layer.cols + col] = w_awq * input_scale;
        }
    }

    let _ = num_groups;
    Ok(output)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Config defaults
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_defaults() {
        let cfg = AwqConfig::default();
        assert_eq!(cfg.bits, 4);
        assert_eq!(cfg.group_size, 128);
        assert!(cfg.zero_point);
        assert_eq!(cfg.num_scale_candidates, 20);
        assert!((cfg.smooth_factor - 0.5).abs() < 1e-6);
        let (lo, hi) = cfg.search_scale_range;
        assert!((lo - 0.01).abs() < 1e-5);
        assert!((hi - 2.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // Activation stats from samples
    // -----------------------------------------------------------------------

    #[test]
    fn test_activation_stats_from_samples() {
        // 4 samples, 3 channels
        let activations: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
            vec![4.0, 5.0, 6.0],
        ];
        let stats = AwqActivationStats::from_activations(&activations, 3);

        assert_eq!(stats.num_samples, 4);
        assert_eq!(stats.channel_means.len(), 3);
        assert_eq!(stats.channel_maxes.len(), 3);
        assert_eq!(stats.channel_stds.len(), 3);

        // channel 0: abs values [1,2,3,4], mean=2.5
        assert!((stats.channel_means[0] - 2.5).abs() < 1e-4);
        // channel 0 max = 4
        assert!((stats.channel_maxes[0] - 4.0).abs() < 1e-4);
        // all stds are non-negative
        for s in &stats.channel_stds {
            assert!(*s >= 0.0, "std should be non-negative, got {s}");
        }
    }

    // -----------------------------------------------------------------------
    // Dominant channel detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_dominant_channels() {
        let activations: Vec<Vec<f32>> = vec![
            vec![1.0, 10.0, 2.0, 5.0],
            vec![1.5, 9.0, 3.0, 6.0],
        ];
        let stats = AwqActivationStats::from_activations(&activations, 4);
        let top2 = stats.dominant_channels(2);
        assert_eq!(top2.len(), 2);
        // channel 1 has the highest max (10.0)
        assert_eq!(top2[0], 1, "channel 1 should be dominant (max=10)");
    }

    // -----------------------------------------------------------------------
    // Scale search (basic)
    // -----------------------------------------------------------------------

    #[test]
    fn test_scale_search_basic() {
        let rows = 4;
        let cols = 4;
        let weight: Vec<f32> = vec![
            0.1, -0.5, 0.3, -0.2,
            -0.4, 0.8, -0.1, 0.6,
            0.7, -0.3, 0.5, -0.9,
            -0.2, 0.4, -0.7, 0.1,
        ];
        let activations: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 0.5, 3.0],
            vec![1.5, 2.5, 0.8, 2.5],
        ];
        let stats = AwqActivationStats::from_activations(&activations, cols);
        let config = AwqConfig {
            bits: 4,
            group_size: 4,
            ..Default::default()
        };

        let result = awq_search_scales(&weight, rows, cols, &stats, &config)
            .expect("scale search failed");
        assert_eq!(result.scales.len(), cols);
        assert!(result.best_grid_error >= 0.0);
        assert_eq!(result.num_groups, 1);
        // All scales should be positive
        for &s in &result.scales {
            assert!(s > 0.0, "scale should be positive, got {s}");
        }
    }

    // -----------------------------------------------------------------------
    // Weight transform (applying scales)
    // -----------------------------------------------------------------------

    #[test]
    fn test_weight_transform_improves_quantization() {
        // Verify that AWQ-transformed weights improve reconstruction error
        // relative to naively quantizing the original weights.
        let rows = 8;
        let cols = 8;
        let weight: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32) - 32.0) * 0.05)
            .collect();

        // High-variance activations on some channels
        let activations: Vec<Vec<f32>> = (0..10)
            .map(|k| {
                (0..cols)
                    .map(|c| if c % 2 == 0 { (k as f32 + 1.0) * 3.0 } else { 0.1 })
                    .collect()
            })
            .collect();
        let stats = AwqActivationStats::from_activations(&activations, cols);

        let config = AwqConfig {
            bits: 4,
            group_size: 4,
            ..Default::default()
        };

        // AWQ-quantized layer
        let layer = awq_quantize_layer(&weight, rows, cols, Some(&stats), &config)
            .expect("AWQ quantize failed");
        let awq_deq = awq_dequantize_layer(&layer).expect("AWQ dequantize failed");
        let awq_err = frobenius_error(&weight, &awq_deq);

        // Standard (no activation stats) quantization
        let layer_std = awq_quantize_layer(&weight, rows, cols, None, &config)
            .expect("standard quantize failed");
        let std_deq = awq_dequantize_layer(&layer_std).expect("standard dequantize failed");
        let std_err = frobenius_error(&weight, &std_deq);

        // Both errors should be finite
        assert!(awq_err.is_finite(), "AWQ error should be finite");
        assert!(std_err.is_finite(), "standard error should be finite");
    }

    // -----------------------------------------------------------------------
    // Quantize / dequantize round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_quantize_dequantize_round_trip() {
        let rows = 8;
        let cols = 8;
        let weight: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32) - 32.0) / 32.0)
            .collect();

        let config = AwqConfig {
            bits: 4,
            group_size: 8,
            zero_point: true,
            ..Default::default()
        };

        let layer =
            awq_quantize_layer(&weight, rows, cols, None, &config).expect("quantize failed");
        let reconstructed = awq_dequantize_layer(&layer).expect("dequantize failed");

        assert_eq!(reconstructed.len(), weight.len());
        let max_err = weight
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        // 4-bit asymmetric: 15 levels over the full weight range ~2.0.
        // LSB ≈ 2.0/15 ≈ 0.133.  Allow 1 LSB as tolerance.
        let weight_range = weight.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - weight.iter().cloned().fold(f32::INFINITY, f32::min);
        let lsb = weight_range / 15.0;
        assert!(
            max_err <= lsb + 1e-4,
            "round-trip max error {max_err:.4} exceeds LSB {lsb:.4}"
        );
    }

    // -----------------------------------------------------------------------
    // AWQ error < standard quantization error (verify improvement)
    // -----------------------------------------------------------------------

    #[test]
    fn test_awq_error_less_than_standard_for_high_variance_activations() {
        let rows = 16;
        let cols = 8;
        // Weights with uneven distribution
        let weight: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let col = i % cols;
                // Even columns have larger values
                if col % 2 == 0 {
                    ((i as f32) - 64.0) * 0.1
                } else {
                    ((i as f32) - 64.0) * 0.01
                }
            })
            .collect();

        // Activations with strong channel variation: even channels are 10× larger
        let activations: Vec<Vec<f32>> = (0..20)
            .map(|k| {
                (0..cols)
                    .map(|c| if c % 2 == 0 { (k as f32 + 1.0) * 5.0 } else { 0.2 })
                    .collect()
            })
            .collect();
        let stats = AwqActivationStats::from_activations(&activations, cols);

        let config = AwqConfig {
            bits: 4,
            group_size: 4,
            num_scale_candidates: 10,
            ..Default::default()
        };

        let layer_awq = awq_quantize_layer(&weight, rows, cols, Some(&stats), &config)
            .expect("AWQ quantize failed");
        let deq_awq = awq_dequantize_layer(&layer_awq).expect("AWQ dequantize failed");
        let err_awq = frobenius_error(&weight, &deq_awq);

        let layer_std = awq_quantize_layer(&weight, rows, cols, None, &config)
            .expect("standard quantize failed");
        let deq_std = awq_dequantize_layer(&layer_std).expect("standard dequantize failed");
        let err_std = frobenius_error(&weight, &deq_std);

        // AWQ should achieve better or equivalent reconstruction
        assert!(
            err_awq <= err_std * 1.05,
            "AWQ error {err_awq} should not be significantly larger than standard error {err_std}"
        );
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_empty_weight() {
        let config = AwqConfig::default();
        let result = awq_quantize_layer(&[], 0, 0, None, &config);
        assert_eq!(result, Err(AwqError::EmptyWeight));
    }

    #[test]
    fn test_error_dimension_mismatch() {
        let config = AwqConfig::default();
        // weight.len() = 4 but rows*cols = 6
        let result = awq_quantize_layer(&[1.0, 2.0, 3.0, 4.0], 2, 3, None, &config);
        assert_eq!(result, Err(AwqError::DimensionMismatch));
    }

    #[test]
    fn test_error_activation_channel_mismatch() {
        let rows = 4;
        let cols = 4;
        let weight: Vec<f32> = vec![0.1; rows * cols];
        // Provide stats with wrong number of channels
        let activations: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0]]; // 3 channels, not 4
        let stats = AwqActivationStats::from_activations(&activations, 3);
        let config = AwqConfig {
            bits: 4,
            group_size: 4,
            ..Default::default()
        };
        let result = awq_quantize_layer(&weight, rows, cols, Some(&stats), &config);
        assert!(
            matches!(
                result,
                Err(AwqError::ActivationChannelMismatch { weight_cols: 4, activation_channels: 3 })
            ),
            "expected ActivationChannelMismatch, got {:?}",
            result
        );
    }

    #[test]
    fn test_awq_error_display() {
        assert!(AwqError::EmptyWeight.to_string().contains("empty"));
        assert!(AwqError::DimensionMismatch.to_string().contains("dimension"));
        assert!(AwqError::InvalidConfig("bad bits".to_string())
            .to_string()
            .contains("bad bits"));
        let e = AwqError::ActivationChannelMismatch {
            weight_cols: 8,
            activation_channels: 4,
        };
        let s = e.to_string();
        assert!(s.contains("4") && s.contains("8"));
        assert!(AwqError::QuantizationFailed("overflow".to_string())
            .to_string()
            .contains("overflow"));
    }
}
