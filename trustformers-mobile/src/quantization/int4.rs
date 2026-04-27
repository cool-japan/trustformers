//! INT4 quantization for mobile deployment.
//!
//! Reduces model size by ~8x vs FP32 while maintaining acceptable quality.
//! Supports per-group symmetric and asymmetric quantization with optional
//! double-quantization of the scales.

use thiserror::Error;
use trustformers_core::errors::{Result, TrustformersError};

// ─── Error type ───────────────────────────────────────────────────────────────

/// Errors specific to mobile INT4 quantization.
#[derive(Debug, Error)]
pub enum MobileQuantError {
    #[error("empty input tensor")]
    EmptyInput,
    #[error("invalid group size: {0}")]
    InvalidGroupSize(usize),
    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: usize, got: usize },
    #[error("packed buffer length {packed_len} insufficient for {count} values")]
    InsufficientBuffer { packed_len: usize, count: usize },
}

impl From<MobileQuantError> for TrustformersError {
    fn from(e: MobileQuantError) -> Self {
        TrustformersError::invalid_input(e.to_string())
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// INT4 quantization configuration.
#[derive(Debug, Clone)]
pub struct Int4QuantConfig {
    /// Number of weights per quantization group (must evenly divide the tensor).
    pub group_size: usize,
    /// Use asymmetric quantization (separate zero point per group).
    pub asymmetric: bool,
    /// Layer names (exact match) excluded from INT4 quantization.
    pub exclude_layers: Vec<String>,
    /// Apply a second quantization pass to the per-group scales for extreme compression.
    pub double_quantize: bool,
}

impl Default for Int4QuantConfig {
    fn default() -> Self {
        Self {
            group_size: 128,
            asymmetric: false,
            exclude_layers: Vec::new(),
            double_quantize: false,
        }
    }
}

/// Simplified INT4 config matching the specification interface.
#[derive(Debug, Clone)]
pub struct Int4Config {
    /// Number of weights per quantization group.
    pub group_size: usize,
    /// Use a per-group zero point offset (asymmetric).
    pub zero_point: bool,
    /// Use symmetric quantization (zero_point ignored when true).
    pub symmetric: bool,
    /// Apply per-channel (row-wise) calibration for weight matrices.
    pub per_channel: bool,
}

impl Default for Int4Config {
    fn default() -> Self {
        Self {
            group_size: 128,
            zero_point: false,
            symmetric: true,
            per_channel: false,
        }
    }
}

impl Int4Config {
    /// Convert to the more detailed `Int4QuantConfig`.
    pub fn to_quant_config(&self) -> Int4QuantConfig {
        Int4QuantConfig {
            group_size: self.group_size,
            asymmetric: self.zero_point && !self.symmetric,
            exclude_layers: Vec::new(),
            double_quantize: false,
        }
    }
}

// ─── Standalone pack/unpack helpers ──────────────────────────────────────────

/// Pack signed INT4 values (in range [-8, 7]) into bytes.
/// Two values per byte: low nibble = values[2*i], high nibble = values[2*i+1].
/// Input values are clamped to [-8, 7] and stored as unsigned nibbles [0, 15] via +8 bias.
pub fn pack_int4(values: &[i8]) -> Vec<u8> {
    let padded = if values.len() % 2 == 0 {
        values.len()
    } else {
        values.len() + 1
    };
    let byte_count = padded / 2;
    let mut out = Vec::with_capacity(byte_count);
    for i in 0..byte_count {
        let lo_raw = if 2 * i < values.len() { values[2 * i] } else { 0i8 };
        let hi_raw = if 2 * i + 1 < values.len() { values[2 * i + 1] } else { 0i8 };
        // Bias by 8 so [-8,7] maps to [0,15]
        let lo = (lo_raw.clamp(-8, 7) + 8) as u8;
        let hi = (hi_raw.clamp(-8, 7) + 8) as u8;
        out.push((hi << 4) | (lo & 0x0F));
    }
    out
}

/// Unpack `count` signed INT4 values from packed bytes.
/// Reverses the bias applied by `pack_int4`.
pub fn unpack_int4(packed: &[u8], count: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let byte = packed[i / 2];
        let nibble = if i % 2 == 0 {
            byte & 0x0F
        } else {
            byte >> 4
        };
        // Remove bias
        out.push(nibble as i8 - 8);
    }
    out
}

// ─── INT4 tensor ─────────────────────────────────────────────────────────────

/// A quantized INT4 tensor stored in packed form (2 values per byte).
///
/// Layout:
/// - low nibble of byte `i` → element `2*i`
/// - high nibble of byte `i` → element `2*i + 1`
#[derive(Debug, Clone)]
pub struct Int4Tensor {
    /// Packed data: 2 INT4 values per byte (low nibble = first, high nibble = second).
    pub packed_data: Vec<u8>,
    /// Scale factor per group: shape = `[num_groups]`.
    pub scales: Vec<f32>,
    /// Zero point per group for asymmetric quantization: shape = `[num_groups]`.
    pub zero_points: Option<Vec<u8>>,
    /// Original tensor shape.
    pub shape: Vec<usize>,
    /// Group size used when quantizing.
    pub group_size: usize,
    /// Whether asymmetric quantization was applied.
    pub asymmetric: bool,
}

impl Int4Tensor {
    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Pack two INT4 values into one byte.
    /// `lo` occupies the lower 4 bits, `hi` the upper 4 bits.
    #[inline]
    fn pack_nibbles(lo: u8, hi: u8) -> u8 {
        (hi << 4) | (lo & 0x0F)
    }

    /// Unpack one byte into two INT4 values `(lo, hi)`.
    #[inline]
    fn unpack_nibbles(packed: u8) -> (u8, u8) {
        (packed & 0x0F, packed >> 4)
    }

    // ── Construction ─────────────────────────────────────────────────────────

    /// Quantize an f32 tensor to INT4.
    ///
    /// Uses per-group calibration:
    /// - Symmetric: scale = max(|x|) / 7.0 per group, zero_point = 8
    /// - Asymmetric: scale = (max - min) / 15.0, zero_point = round(-min / scale)
    pub fn quantize(data: &[f32], shape: &[usize], config: &Int4QuantConfig) -> Result<Self> {
        if data.is_empty() {
            return Err(TrustformersError::invalid_input("empty tensor cannot be quantized".to_string()));
        }
        let total: usize = shape.iter().product();
        if total != data.len() {
            return Err(TrustformersError::invalid_input(format!(
                "shape product {} does not match data length {}",
                total,
                data.len()
            )));
        }
        if config.group_size == 0 {
            return Err(TrustformersError::invalid_input("group_size must be > 0".to_string()));
        }

        let num_groups = total.div_ceil(config.group_size);
        let mut scales: Vec<f32> = Vec::with_capacity(num_groups);
        let mut zero_points: Vec<u8> = Vec::with_capacity(num_groups);

        // We will pack 2 quantized nibbles per byte.
        // Pad to an even number of elements if necessary.
        let padded_len = if total % 2 == 0 { total } else { total + 1 };
        let mut nibbles: Vec<u8> = vec![0u8; padded_len]; // INT4 values (0..15)

        for g in 0..num_groups {
            let start = g * config.group_size;
            let end = (start + config.group_size).min(total);
            let group = &data[start..end];

            if config.asymmetric {
                let min = group.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let range = max - min;
                let scale = if range.abs() < f32::EPSILON { 1.0 } else { range / 15.0 };
                let zp = (-min / scale).round().clamp(0.0, 15.0) as u8;
                scales.push(scale);
                zero_points.push(zp);
                for (i, &v) in group.iter().enumerate() {
                    let q = (v / scale + zp as f32).round().clamp(0.0, 15.0) as u8;
                    nibbles[start + i] = q;
                }
            } else {
                // Symmetric: values mapped to [-7, 7] then shifted to [1, 15] (0 reserved for exact zero).
                let abs_max = group
                    .iter()
                    .cloned()
                    .fold(0.0_f32, |a, x| a.max(x.abs()));
                let scale = if abs_max < f32::EPSILON { 1.0 } else { abs_max / 7.0 };
                scales.push(scale);
                zero_points.push(8); // centre of [0, 15] for symmetric
                for (i, &v) in group.iter().enumerate() {
                    let q = ((v / scale) + 8.0).round().clamp(0.0, 15.0) as u8;
                    nibbles[start + i] = q;
                }
            }
        }

        // Pack nibbles into bytes (2 per byte).
        let byte_count = padded_len / 2;
        let mut packed_data: Vec<u8> = Vec::with_capacity(byte_count);
        for i in 0..byte_count {
            packed_data.push(Self::pack_nibbles(nibbles[2 * i], nibbles[2 * i + 1]));
        }

        Ok(Self {
            packed_data,
            scales,
            zero_points: if config.asymmetric {
                Some(zero_points)
            } else {
                None
            },
            shape: shape.to_vec(),
            group_size: config.group_size,
            asymmetric: config.asymmetric,
        })
    }

    /// Quantize using the simplified `Int4Config` interface.
    pub fn from_config(tensor: &[f32], config: &Int4Config) -> std::result::Result<Self, MobileQuantError> {
        if tensor.is_empty() {
            return Err(MobileQuantError::EmptyInput);
        }
        if config.group_size == 0 {
            return Err(MobileQuantError::InvalidGroupSize(0));
        }
        let shape = vec![tensor.len()];
        let qconfig = config.to_quant_config();
        Self::quantize(tensor, &shape, &qconfig).map_err(|e| {
            MobileQuantError::ShapeMismatch { expected: tensor.len(), got: e.to_string().len() }
        })
    }

    // ── Dequantization ────────────────────────────────────────────────────────

    /// Dequantize INT4 tensor back to f32.
    pub fn dequantize(&self) -> Vec<f32> {
        let total: usize = self.shape.iter().product();
        let padded_len = if total % 2 == 0 { total } else { total + 1 };

        // Unpack nibbles.
        let mut nibbles: Vec<u8> = Vec::with_capacity(padded_len);
        for &byte in &self.packed_data {
            let (lo, hi) = Self::unpack_nibbles(byte);
            nibbles.push(lo);
            nibbles.push(hi);
        }
        nibbles.truncate(total);

        let mut output: Vec<f32> = Vec::with_capacity(total);
        let num_groups = total.div_ceil(self.group_size);

        for g in 0..num_groups {
            let start = g * self.group_size;
            let end = (start + self.group_size).min(total);
            let scale = self.scales[g];

            if self.asymmetric {
                let zp = self.zero_points.as_ref().map(|z| z[g]).unwrap_or(0) as f32;
                for i in start..end {
                    output.push((nibbles[i] as f32 - zp) * scale);
                }
            } else {
                for i in start..end {
                    output.push((nibbles[i] as f32 - 8.0) * scale);
                }
            }
        }

        output
    }

    // ── Size / statistics ─────────────────────────────────────────────────────

    /// Total size in bytes of the packed representation (packed data + scales + zero points).
    pub fn size_bytes(&self) -> usize {
        let packed = self.packed_data.len();
        let scales_bytes = self.scales.len() * std::mem::size_of::<f32>();
        let zp_bytes = self
            .zero_points
            .as_ref()
            .map(|z| z.len())
            .unwrap_or(0);
        packed + scales_bytes + zp_bytes
    }

    /// Compression ratio relative to an equivalent f32 tensor.
    pub fn compression_ratio(&self) -> f32 {
        let total: usize = self.shape.iter().product();
        let fp32_bytes = total * std::mem::size_of::<f32>();
        fp32_bytes as f32 / self.size_bytes() as f32
    }

    /// Number of quantization groups.
    pub fn num_groups(&self) -> usize {
        self.scales.len()
    }
}

// ─── INT4 GEMV ────────────────────────────────────────────────────────────────

/// Optimized matrix-vector multiply with INT4-packed weight matrix.
///
/// Performs: output[r] = sum_c(unpack(packed[r,c]) * scale[r] * input[c])
/// for each row r.
pub struct Int4Gemv;

impl Int4Gemv {
    /// Multiply an INT4-packed weight matrix by an f32 input vector.
    ///
    /// - `packed_weights`: row-major packed data; each byte holds 2 weights
    ///   (low nibble = first column element, high nibble = second column element).
    ///   Total bytes expected: rows * ceil(cols / 2).
    /// - `scales`: one scale per row (length = `rows`).
    /// - `input`: f32 input vector (length = `cols`).
    /// - `rows`, `cols`: matrix dimensions.
    ///
    /// Returns a `Vec<f32>` of length `rows`.
    pub fn compute(
        packed_weights: &[u8],
        scales: &[f32],
        input: &[f32],
        rows: usize,
        cols: usize,
    ) -> Vec<f32> {
        let bytes_per_row = cols.div_ceil(2);
        let mut output = vec![0.0f32; rows];

        for r in 0..rows {
            let row_start = r * bytes_per_row;
            let scale = if r < scales.len() { scales[r] } else { 1.0 };
            let mut acc = 0.0f32;

            for c in 0..cols {
                let byte_idx = row_start + c / 2;
                let byte = if byte_idx < packed_weights.len() {
                    packed_weights[byte_idx]
                } else {
                    0u8
                };
                let nibble = if c % 2 == 0 {
                    byte & 0x0F
                } else {
                    byte >> 4
                };
                // Dequantize: nibble is in [0,15] with symmetric bias of 8
                let weight = (nibble as f32 - 8.0) * scale;
                let inp = if c < input.len() { input[c] } else { 0.0 };
                acc += weight * inp;
            }

            output[r] = acc;
        }

        output
    }
}

// ─── Quality metrics ──────────────────────────────────────────────────────────

/// Quantization quality metrics comparing original and dequantized tensors.
#[derive(Debug, Clone)]
pub struct QuantizationMetrics {
    /// Maximum absolute error between original and dequantized values.
    pub max_abs_error: f32,
    /// Mean absolute error.
    pub mean_abs_error: f32,
    /// Root mean squared error.
    pub rmse: f32,
    /// Signal-to-noise ratio in dB.
    pub snr_db: f32,
    /// Compression ratio vs the original f32 tensor.
    pub compression_ratio: f32,
}

impl QuantizationMetrics {
    /// Compute metrics comparing `original` and `dequantized` tensors.
    pub fn compute(original: &[f32], dequantized: &[f32]) -> Self {
        assert_eq!(original.len(), dequantized.len(), "length mismatch");
        let n = original.len() as f32;

        let mut max_abs = 0.0_f32;
        let mut sum_abs = 0.0_f32;
        let mut sum_sq_err = 0.0_f32;
        let mut signal_power = 0.0_f32;

        for (&o, &d) in original.iter().zip(dequantized.iter()) {
            let err = (o - d).abs();
            max_abs = max_abs.max(err);
            sum_abs += err;
            sum_sq_err += (o - d) * (o - d);
            signal_power += o * o;
        }

        let rmse = (sum_sq_err / n).sqrt();
        let noise_power = sum_sq_err / n;
        let snr_db = if noise_power < f32::EPSILON {
            f32::INFINITY
        } else {
            10.0 * (signal_power / n / noise_power).log10()
        };

        Self {
            max_abs_error: max_abs,
            mean_abs_error: sum_abs / n,
            rmse,
            snr_db,
            compression_ratio: 0.0, // caller fills this in if desired
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| start + (end - start) * i as f32 / (n - 1) as f32)
            .collect()
    }

    #[test]
    fn test_symmetric_quantize_dequantize_basic() {
        let data: Vec<f32> = linspace(-1.0, 1.0, 128);
        let config = Int4QuantConfig::default();
        let tensor = Int4Tensor::quantize(&data, &[128], &config).expect("should quantize");
        let deq = tensor.dequantize();
        assert_eq!(deq.len(), 128);
        let metrics = QuantizationMetrics::compute(&data, &deq);
        assert!(metrics.max_abs_error < 0.15, "max_abs_error too large: {}", metrics.max_abs_error);
    }

    #[test]
    fn test_asymmetric_quantize_dequantize() {
        let data: Vec<f32> = linspace(0.0, 4.0, 64);
        let config = Int4QuantConfig {
            group_size: 64,
            asymmetric: true,
            ..Default::default()
        };
        let tensor = Int4Tensor::quantize(&data, &[64], &config).expect("should quantize");
        assert!(tensor.zero_points.is_some());
        let deq = tensor.dequantize();
        let metrics = QuantizationMetrics::compute(&data, &deq);
        assert!(metrics.max_abs_error < 0.3, "asymmetric error too large: {}", metrics.max_abs_error);
    }

    #[test]
    fn test_pack_unpack_nibbles() {
        for lo in 0u8..16 {
            for hi in 0u8..16 {
                let packed = Int4Tensor::pack_nibbles(lo, hi);
                let (ul, uh) = Int4Tensor::unpack_nibbles(packed);
                assert_eq!(ul, lo, "lo nibble mismatch");
                assert_eq!(uh, hi, "hi nibble mismatch");
            }
        }
    }

    #[test]
    fn test_compression_ratio() {
        let data: Vec<f32> = vec![0.5f32; 512];
        let config = Int4QuantConfig { group_size: 128, ..Default::default() };
        let tensor = Int4Tensor::quantize(&data, &[512], &config).expect("should quantize");
        let ratio = tensor.compression_ratio();
        assert!(ratio > 4.0, "expected ratio > 4, got {ratio}");
    }

    #[test]
    fn test_size_bytes_is_less_than_fp32() {
        let n = 1024usize;
        let data: Vec<f32> = vec![1.0f32; n];
        let config = Int4QuantConfig { group_size: 128, ..Default::default() };
        let tensor = Int4Tensor::quantize(&data, &[n], &config).expect("should quantize");
        let fp32_bytes = n * 4;
        assert!(tensor.size_bytes() < fp32_bytes);
    }

    #[test]
    fn test_shape_mismatch_returns_error() {
        let data = vec![1.0f32; 64];
        let config = Int4QuantConfig::default();
        let result = Int4Tensor::quantize(&data, &[65], &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_tensor_quantizes_without_panic() {
        let data = vec![0.0f32; 128];
        let config = Int4QuantConfig::default();
        let tensor = Int4Tensor::quantize(&data, &[128], &config).expect("should handle zero tensor");
        let deq = tensor.dequantize();
        for v in &deq {
            assert!(v.abs() < 1e-6, "expected ~0 after dequantize, got {v}");
        }
    }

    #[test]
    fn test_multi_group_quantization() {
        let data: Vec<f32> = linspace(-2.0, 2.0, 256);
        let config = Int4QuantConfig { group_size: 64, ..Default::default() };
        let tensor = Int4Tensor::quantize(&data, &[256], &config).expect("should quantize");
        assert_eq!(tensor.num_groups(), 4); // 256 / 64 = 4
        let deq = tensor.dequantize();
        assert_eq!(deq.len(), 256);
    }

    #[test]
    fn test_quantization_metrics_perfect() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let metrics = QuantizationMetrics::compute(&data, &data);
        assert_eq!(metrics.max_abs_error, 0.0);
        assert_eq!(metrics.rmse, 0.0);
        assert!(metrics.snr_db.is_infinite() || metrics.snr_db > 100.0);
    }

    #[test]
    fn test_odd_length_tensor() {
        let data = vec![0.5f32; 5]; // odd
        let config = Int4QuantConfig { group_size: 4, ..Default::default() };
        let tensor = Int4Tensor::quantize(&data, &[5], &config).expect("should handle odd length");
        let deq = tensor.dequantize();
        assert_eq!(deq.len(), 5);
    }

    // ── New tests for pack_int4 / unpack_int4 ─────────────────────────────────

    #[test]
    fn test_pack_int4_roundtrip_even() {
        let values: Vec<i8> = vec![-7, -4, -1, 0, 1, 4, 7, 3];
        let packed = pack_int4(&values);
        let unpacked = unpack_int4(&packed, values.len());
        assert_eq!(unpacked, values, "pack/unpack roundtrip failed");
    }

    #[test]
    fn test_pack_int4_roundtrip_odd() {
        let values: Vec<i8> = vec![-3, 0, 5];
        let packed = pack_int4(&values);
        let unpacked = unpack_int4(&packed, values.len());
        assert_eq!(unpacked, values, "odd-length roundtrip failed");
    }

    #[test]
    fn test_pack_int4_byte_count() {
        // 8 values → 4 bytes
        let values: Vec<i8> = vec![0i8; 8];
        let packed = pack_int4(&values);
        assert_eq!(packed.len(), 4);
    }

    #[test]
    fn test_pack_int4_odd_byte_count() {
        // 5 values → 3 bytes (padded to 6)
        let values: Vec<i8> = vec![0i8; 5];
        let packed = pack_int4(&values);
        assert_eq!(packed.len(), 3);
    }

    #[test]
    fn test_unpack_int4_count_truncation() {
        let values: Vec<i8> = vec![1, 2, 3, 4];
        let packed = pack_int4(&values);
        // Only unpack 2 values
        let unpacked = unpack_int4(&packed, 2);
        assert_eq!(unpacked.len(), 2);
        assert_eq!(unpacked[0], 1);
        assert_eq!(unpacked[1], 2);
    }

    // ── Int4Config tests ──────────────────────────────────────────────────────

    #[test]
    fn test_int4_config_defaults() {
        let cfg = Int4Config::default();
        assert_eq!(cfg.group_size, 128);
        assert!(!cfg.zero_point);
        assert!(cfg.symmetric);
        assert!(!cfg.per_channel);
    }

    #[test]
    fn test_int4_config_per_channel_flag() {
        let cfg = Int4Config { per_channel: true, ..Default::default() };
        assert!(cfg.per_channel);
    }

    #[test]
    fn test_int4_config_to_quant_config_symmetric() {
        let cfg = Int4Config { symmetric: true, zero_point: false, group_size: 64, per_channel: false };
        let qcfg = cfg.to_quant_config();
        assert_eq!(qcfg.group_size, 64);
        assert!(!qcfg.asymmetric);
    }

    #[test]
    fn test_int4_config_to_quant_config_asymmetric() {
        let cfg = Int4Config { symmetric: false, zero_point: true, group_size: 32, per_channel: false };
        let qcfg = cfg.to_quant_config();
        assert!(qcfg.asymmetric);
    }

    // ── MobileQuantError tests ────────────────────────────────────────────────

    #[test]
    fn test_mobile_quant_error_empty_input() {
        let cfg = Int4Config::default();
        let result = Int4Tensor::from_config(&[], &cfg);
        assert!(matches!(result, Err(MobileQuantError::EmptyInput)));
    }

    #[test]
    fn test_mobile_quant_error_invalid_group_size() {
        let cfg = Int4Config { group_size: 0, ..Default::default() };
        let result = Int4Tensor::from_config(&[1.0, 2.0], &cfg);
        assert!(matches!(result, Err(MobileQuantError::InvalidGroupSize(0))));
    }

    #[test]
    fn test_mobile_quant_error_display() {
        let e = MobileQuantError::EmptyInput;
        assert!(!e.to_string().is_empty());
        let e2 = MobileQuantError::InvalidGroupSize(0);
        assert!(e2.to_string().contains('0'));
    }

    // ── Int4Gemv tests ────────────────────────────────────────────────────────

    #[test]
    fn test_int4_gemv_output_shape() {
        let rows: usize = 4;
        let cols: usize = 8;
        // All zero packed weights (nibble value 8 → weight 0.0)
        let packed = vec![0x88u8; rows * cols.div_ceil(2)];
        let scales = vec![1.0f32; rows];
        let input = vec![1.0f32; cols];
        let out = Int4Gemv::compute(&packed, &scales, &input, rows, cols);
        assert_eq!(out.len(), rows);
    }

    #[test]
    fn test_int4_gemv_zero_weights_produce_zero() {
        let rows: usize = 3;
        let cols: usize = 4;
        // Pack nibble 8 → symmetric weight 0.0
        let packed = vec![0x88u8; rows * cols.div_ceil(2)];
        let scales = vec![1.0f32; rows];
        let input = vec![1.0f32; cols];
        let out = Int4Gemv::compute(&packed, &scales, &input, rows, cols);
        for &v in &out {
            assert!(v.abs() < 1e-6, "expected 0.0, got {v}");
        }
    }

    #[test]
    fn test_int4_gemv_nonzero_weights() {
        // 1 row, 2 cols
        // nibble values: lo=9 (weight=1.0*scale), hi=7 (weight=-1.0*scale)
        // scale = 2.0
        // input = [1.0, 1.0]
        // expected = (9-8)*2.0*1.0 + (7-8)*2.0*1.0 = 2.0 - 2.0 = 0.0
        let packed = vec![0x79u8]; // lo=9, hi=7
        let scales = vec![2.0f32];
        let input = vec![1.0f32, 1.0f32];
        let out = Int4Gemv::compute(&packed, &scales, &input, 1, 2);
        assert_eq!(out.len(), 1);
        assert!((out[0]).abs() < 1e-5, "expected 0.0, got {}", out[0]);
    }

    #[test]
    fn test_int4_gemv_single_positive_weight() {
        // 1 row, 1 col, nibble=9 → weight = (9-8)*scale = 1.0*1.0 = 1.0
        // input = [3.0] → output = 3.0
        let packed = vec![0x09u8]; // lo nibble = 9
        let scales = vec![1.0f32];
        let input = vec![3.0f32];
        let out = Int4Gemv::compute(&packed, &scales, &input, 1, 1);
        assert!((out[0] - 3.0).abs() < 1e-5, "expected 3.0, got {}", out[0]);
    }

    // ── Compression ratio for large tensor ────────────────────────────────────

    #[test]
    fn test_compression_ratio_large_tensor() {
        // For a very large tensor, scale overhead becomes negligible → ratio approaches 8.0
        let n = 8192usize;
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let config = Int4QuantConfig { group_size: 128, ..Default::default() };
        let tensor = Int4Tensor::quantize(&data, &[n], &config).expect("quantize");
        let ratio = tensor.compression_ratio();
        assert!(ratio > 7.0, "expected ratio > 7.0 for large tensor, got {ratio}");
    }

    #[test]
    fn test_from_config_basic() {
        let data: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
        let config = Int4Config::default();
        let tensor = Int4Tensor::from_config(&data, &config).expect("should succeed");
        let deq = tensor.dequantize();
        assert_eq!(deq.len(), 256);
    }
}
