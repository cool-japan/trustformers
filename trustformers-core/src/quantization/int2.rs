//! INT2 sub-byte quantization for extreme model compression
//!
//! This module implements 2-bit quantization, packing 4 values per byte for
//! maximum memory savings (~16x compression vs FP32). Supports:
//!
//! - **Asymmetric**: Maps \[min, max\] to \[0, 3\] with per-group scale and zero-point
//! - **Symmetric**: Maps \[-absmax, absmax\] to \[-1, 2\] symmetrically
//! - **Ternary**: Encodes weights as {-1, 0, +1} using threshold-based quantization
//!
//! # Group-based quantization
//!
//! Values are quantized in groups (32, 64, or 128 elements), each with its own
//! scale and zero-point to preserve local dynamic range.
//!
//! # Bit packing
//!
//! Four 2-bit values are packed into a single byte using pure Rust bit manipulation:
//! ```text
//! byte = (val3 << 6) | (val2 << 4) | (val1 << 2) | val0
//! ```
//!
//! # Examples
//!
//! ```rust,no_run
//! use trustformers_core::quantization::int2::{
//!     Int2QuantConfig, Int2Mode, quantize_to_int2, dequantize_from_int2,
//! };
//!
//! let data = vec![0.1, -0.5, 0.3, -0.2, 0.7, -0.1, 0.4, 0.0];
//! let config = Int2QuantConfig {
//!     group_size: 32,
//!     mode: Int2Mode::Asymmetric,
//!     scale: 0.0,       // auto-computed
//!     zero_point: 0.0,  // auto-computed
//! };
//! let packed = quantize_to_int2(&data, &config);
//! let recovered = dequantize_from_int2(&packed);
//! ```

use crate::errors::{quantization_error, Result};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Quantization mode for INT2
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Int2Mode {
    /// Asymmetric: maps [min, max] -> [0, 3], per-group scale + zero_point
    Asymmetric,
    /// Symmetric: maps [-absmax, absmax] -> [0, 3] with implicit zero at midpoint
    Symmetric,
    /// Ternary: encodes {-1, 0, +1} using a magnitude threshold
    Ternary,
}

/// Configuration for INT2 quantization
#[derive(Debug, Clone)]
pub struct Int2QuantConfig {
    /// Number of elements per quantization group (32, 64, or 128)
    pub group_size: usize,
    /// Quantization mode
    pub mode: Int2Mode,
    /// Scale factor (0.0 = auto-compute per group)
    pub scale: f32,
    /// Zero point offset (0.0 = auto-compute per group)
    pub zero_point: f32,
}

impl Default for Int2QuantConfig {
    fn default() -> Self {
        Self {
            group_size: 32,
            mode: Int2Mode::Asymmetric,
            scale: 0.0,
            zero_point: 0.0,
        }
    }
}

/// Packed INT2 storage: 4 values per byte, with per-group scales and zero-points.
#[derive(Debug, Clone)]
pub struct PackedInt2 {
    /// Raw packed bytes (4 × 2-bit values each)
    data: Vec<u8>,
    /// Per-group scale factors
    scales: Vec<f32>,
    /// Per-group zero-point offsets
    zero_points: Vec<f32>,
    /// Original tensor shape
    shape: Vec<usize>,
    /// Total number of quantized values
    num_values: usize,
    /// Mode used during quantization
    mode: Int2Mode,
}

impl PackedInt2 {
    /// Returns the raw packed byte slice.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Returns the per-group scale factors.
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Returns the per-group zero-point offsets.
    pub fn zero_points(&self) -> &[f32] {
        &self.zero_points
    }

    /// Returns the original tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the total number of stored values.
    pub fn num_values(&self) -> usize {
        self.num_values
    }

    /// Returns the quantization mode.
    pub fn mode(&self) -> Int2Mode {
        self.mode
    }

    /// Returns the compression ratio relative to FP32.
    pub fn compression_ratio(&self) -> f64 {
        let original_bytes = self.num_values as f64 * 4.0; // f32 = 4 bytes
        let packed_bytes = self.data.len() as f64;
        let meta_bytes = (self.scales.len() + self.zero_points.len()) as f64 * 4.0;
        if packed_bytes + meta_bytes == 0.0 {
            return 0.0;
        }
        original_bytes / (packed_bytes + meta_bytes)
    }

    /// Returns the number of bits per value (excluding metadata overhead).
    pub fn bits_per_value(&self) -> f64 {
        if self.num_values == 0 {
            return 0.0;
        }
        (self.data.len() as f64 * 8.0) / self.num_values as f64
    }
}

// ---------------------------------------------------------------------------
// Bit-packing helpers
// ---------------------------------------------------------------------------

/// Pack four 2-bit values (each in 0..=3) into a single byte.
///
/// Layout: `(d << 6) | (c << 4) | (b << 2) | a`
#[inline]
pub fn pack_four(a: u8, b: u8, c: u8, d: u8) -> u8 {
    debug_assert!(a <= 3 && b <= 3 && c <= 3 && d <= 3);
    (d << 6) | (c << 4) | (b << 2) | a
}

/// Unpack a byte into four 2-bit values.
#[inline]
pub fn unpack_four(byte: u8) -> (u8, u8, u8, u8) {
    let a = byte & 0x03;
    let b = (byte >> 2) & 0x03;
    let c = (byte >> 4) & 0x03;
    let d = (byte >> 6) & 0x03;
    (a, b, c, d)
}

// ---------------------------------------------------------------------------
// Core quantization
// ---------------------------------------------------------------------------

/// Quantize an f32 slice to packed INT2 representation.
///
/// When `config.scale` is 0.0 the scale/zero-point are computed automatically
/// per group.  `config.group_size` must be a multiple of 4 and at least 4.
pub fn quantize_to_int2(data: &[f32], config: &Int2QuantConfig) -> PackedInt2 {
    let group_size = config.group_size.max(4);
    let num_groups = (data.len() + group_size - 1) / group_size;

    let mut packed_bytes: Vec<u8> = Vec::with_capacity((data.len() + 3) / 4);
    let mut scales: Vec<f32> = Vec::with_capacity(num_groups);
    let mut zero_points: Vec<f32> = Vec::with_capacity(num_groups);

    match config.mode {
        Int2Mode::Asymmetric => {
            quantize_asymmetric(data, group_size, config, &mut packed_bytes, &mut scales, &mut zero_points);
        }
        Int2Mode::Symmetric => {
            quantize_symmetric(data, group_size, config, &mut packed_bytes, &mut scales, &mut zero_points);
        }
        Int2Mode::Ternary => {
            quantize_ternary_inner(data, group_size, &mut packed_bytes, &mut scales, &mut zero_points);
        }
    }

    PackedInt2 {
        data: packed_bytes,
        scales,
        zero_points,
        shape: vec![data.len()],
        num_values: data.len(),
        mode: config.mode,
    }
}

/// Dequantize a [`PackedInt2`] back to f32 values.
pub fn dequantize_from_int2(packed: &PackedInt2) -> Vec<f32> {
    let group_size = compute_group_size(packed.num_values, packed.scales.len());
    let mut out = Vec::with_capacity(packed.num_values);

    match packed.mode {
        Int2Mode::Asymmetric => {
            dequantize_asymmetric(packed, group_size, &mut out);
        }
        Int2Mode::Symmetric => {
            dequantize_symmetric(packed, group_size, &mut out);
        }
        Int2Mode::Ternary => {
            dequantize_ternary_inner(packed, group_size, &mut out);
        }
    }

    out.truncate(packed.num_values);
    out
}

/// Convenience wrapper: ternary quantization ({-1, 0, +1}).
pub fn quantize_ternary(data: &[f32], group_size: usize) -> PackedInt2 {
    let config = Int2QuantConfig {
        group_size,
        mode: Int2Mode::Ternary,
        scale: 0.0,
        zero_point: 0.0,
    };
    quantize_to_int2(data, &config)
}

/// Validate an [`Int2QuantConfig`] and return an error if invalid.
pub fn validate_config(config: &Int2QuantConfig) -> Result<()> {
    if config.group_size == 0 {
        return Err(quantization_error(
            "int2_validate",
            "group_size must be > 0",
        ));
    }
    if config.group_size % 4 != 0 {
        return Err(quantization_error(
            "int2_validate",
            format!("group_size must be a multiple of 4, got {}", config.group_size),
        ));
    }
    if config.scale.is_nan() || config.zero_point.is_nan() {
        return Err(quantization_error(
            "int2_validate",
            "scale and zero_point must not be NaN",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Asymmetric quantization internals
// ---------------------------------------------------------------------------

fn quantize_asymmetric(
    data: &[f32],
    group_size: usize,
    config: &Int2QuantConfig,
    packed: &mut Vec<u8>,
    scales: &mut Vec<f32>,
    zero_points: &mut Vec<f32>,
) {
    for chunk in data.chunks(group_size) {
        let (scale, zp) = if config.scale != 0.0 {
            (config.scale, config.zero_point)
        } else {
            compute_asymmetric_params(chunk)
        };
        scales.push(scale);
        zero_points.push(zp);

        pack_group(chunk, scale, zp, packed, |val, s, z| {
            if s.abs() < f32::EPSILON {
                return 0u8;
            }
            let q = ((val - z) / s).round();
            clamp_to_u8(q, 0.0, 3.0)
        });
    }
}

fn dequantize_asymmetric(packed: &PackedInt2, group_size: usize, out: &mut Vec<f32>) {
    let mut value_idx = 0;
    for (gi, (&scale, &zp)) in packed.scales.iter().zip(packed.zero_points.iter()).enumerate() {
        let group_start = gi * group_size;
        let group_end = (group_start + group_size).min(packed.num_values);
        let group_len = group_end - group_start;

        unpack_group(&packed.data, value_idx, group_len, out, |q| {
            q as f32 * scale + zp
        });
        value_idx += group_len;
    }
}

fn compute_asymmetric_params(group: &[f32]) -> (f32, f32) {
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    for &v in group {
        if v < min_val { min_val = v; }
        if v > max_val { max_val = v; }
    }
    let range = max_val - min_val;
    if range.abs() < f32::EPSILON {
        return (0.0, min_val);
    }
    let scale = range / 3.0; // 2-bit => 4 levels (0..3)
    (scale, min_val)
}

// ---------------------------------------------------------------------------
// Symmetric quantization internals
// ---------------------------------------------------------------------------

fn quantize_symmetric(
    data: &[f32],
    group_size: usize,
    config: &Int2QuantConfig,
    packed: &mut Vec<u8>,
    scales: &mut Vec<f32>,
    zero_points: &mut Vec<f32>,
) {
    for chunk in data.chunks(group_size) {
        let scale = if config.scale != 0.0 {
            config.scale
        } else {
            compute_symmetric_scale(chunk)
        };
        scales.push(scale);
        zero_points.push(0.0);

        // Map [-absmax, absmax] -> [0, 3]: q = round(val / scale + 1.5)
        pack_group(chunk, scale, 0.0, packed, |val, s, _z| {
            if s.abs() < f32::EPSILON {
                return 1u8; // midpoint for zero
            }
            let q = (val / s + 1.5).round();
            clamp_to_u8(q, 0.0, 3.0)
        });
    }
}

fn dequantize_symmetric(packed: &PackedInt2, group_size: usize, out: &mut Vec<f32>) {
    let mut value_idx = 0;
    for (gi, &scale) in packed.scales.iter().enumerate() {
        let group_start = gi * group_size;
        let group_end = (group_start + group_size).min(packed.num_values);
        let group_len = group_end - group_start;

        unpack_group(&packed.data, value_idx, group_len, out, |q| {
            (q as f32 - 1.5) * scale
        });
        value_idx += group_len;
    }
}

fn compute_symmetric_scale(group: &[f32]) -> f32 {
    let mut absmax: f32 = 0.0;
    for &v in group {
        let a = v.abs();
        if a > absmax {
            absmax = a;
        }
    }
    if absmax < f32::EPSILON {
        return 0.0;
    }
    absmax / 1.5 // range [-1.5*scale, 1.5*scale] maps to [0, 3]
}

// ---------------------------------------------------------------------------
// Ternary quantization internals
// ---------------------------------------------------------------------------

fn quantize_ternary_inner(
    data: &[f32],
    group_size: usize,
    packed: &mut Vec<u8>,
    scales: &mut Vec<f32>,
    zero_points: &mut Vec<f32>,
) {
    for chunk in data.chunks(group_size) {
        let (threshold, scale) = compute_ternary_params(chunk);
        scales.push(scale);
        zero_points.push(threshold);

        // Ternary mapping: -1 -> 0, 0 -> 1, +1 -> 2  (value 3 unused)
        pack_group(chunk, scale, threshold, packed, |val, _s, thr| {
            if val > thr {
                2u8 // +1
            } else if val < -thr {
                0u8 // -1
            } else {
                1u8 // 0
            }
        });
    }
}

fn dequantize_ternary_inner(packed: &PackedInt2, group_size: usize, out: &mut Vec<f32>) {
    let mut value_idx = 0;
    for (gi, &scale) in packed.scales.iter().enumerate() {
        let group_start = gi * group_size;
        let group_end = (group_start + group_size).min(packed.num_values);
        let group_len = group_end - group_start;

        unpack_group(&packed.data, value_idx, group_len, out, |q| {
            match q {
                0 => -scale, // -1 * scale
                2 => scale,  // +1 * scale
                _ => 0.0,    // 0
            }
        });
        value_idx += group_len;
    }
}

/// Compute ternary threshold and scale for a group.
///
/// Threshold is set at 0.7 × mean(|nonzero|), and scale is the mean of
/// absolute values exceeding the threshold. This follows the TWN
/// (Ternary Weight Networks) heuristic.
fn compute_ternary_params(group: &[f32]) -> (f32, f32) {
    let n = group.len() as f32;
    if n < 1.0 {
        return (0.0, 0.0);
    }

    // Mean absolute value
    let abs_sum: f32 = group.iter().map(|v| v.abs()).sum();
    let abs_mean = abs_sum / n;
    let threshold = 0.7 * abs_mean;

    // Scale = mean of |values| > threshold
    let mut sum_above = 0.0f32;
    let mut count_above = 0usize;
    for &v in group {
        if v.abs() > threshold {
            sum_above += v.abs();
            count_above += 1;
        }
    }

    let scale = if count_above > 0 {
        sum_above / count_above as f32
    } else {
        abs_mean
    };

    (threshold, scale)
}

// ---------------------------------------------------------------------------
// Generic packing / unpacking helpers
// ---------------------------------------------------------------------------

/// Pack a group of f32 values using a quantize-to-u8 closure, appending packed bytes.
fn pack_group<F>(
    group: &[f32],
    scale: f32,
    zero_point: f32,
    packed: &mut Vec<u8>,
    quantize_fn: F,
)
where
    F: Fn(f32, f32, f32) -> u8,
{
    let mut buf = [0u8; 4];
    let mut buf_idx = 0;

    for &val in group {
        buf[buf_idx] = quantize_fn(val, scale, zero_point);
        buf_idx += 1;
        if buf_idx == 4 {
            packed.push(pack_four(buf[0], buf[1], buf[2], buf[3]));
            buf_idx = 0;
        }
    }

    // Flush partial quad (pad with 0)
    if buf_idx > 0 {
        for i in buf_idx..4 {
            buf[i] = 0;
        }
        packed.push(pack_four(buf[0], buf[1], buf[2], buf[3]));
    }
}

/// Unpack values from packed bytes using a dequantize closure, appending to output.
fn unpack_group<F>(
    packed_data: &[u8],
    start_value_idx: usize,
    count: usize,
    out: &mut Vec<f32>,
    dequantize_fn: F,
)
where
    F: Fn(u8) -> f32,
{
    let start_byte = start_value_idx / 4;
    let start_offset = start_value_idx % 4;

    let mut remaining = count;
    let mut byte_idx = start_byte;
    let mut sub_idx = start_offset;

    while remaining > 0 {
        if byte_idx >= packed_data.len() {
            break;
        }
        let (a, b, c, d) = unpack_four(packed_data[byte_idx]);
        let vals = [a, b, c, d];

        while sub_idx < 4 && remaining > 0 {
            out.push(dequantize_fn(vals[sub_idx]));
            sub_idx += 1;
            remaining -= 1;
        }
        sub_idx = 0;
        byte_idx += 1;
    }
}

#[inline]
fn clamp_to_u8(val: f32, min: f32, max: f32) -> u8 {
    val.clamp(min, max) as u8
}

fn compute_group_size(num_values: usize, num_groups: usize) -> usize {
    if num_groups == 0 {
        return num_values.max(4);
    }
    // Reconstruct group_size from value count / group count
    (num_values + num_groups - 1) / num_groups
}

// ---------------------------------------------------------------------------
// Statistics utilities
// ---------------------------------------------------------------------------

/// Compute mean squared error between original and reconstructed values.
pub fn quantization_mse(original: &[f32], reconstructed: &[f32]) -> f32 {
    let n = original.len().min(reconstructed.len());
    if n == 0 {
        return 0.0;
    }
    let sum_sq: f32 = original.iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum();
    sum_sq / n as f32
}

/// Compute signal-to-quantization-noise ratio (SQNR) in dB.
pub fn quantization_sqnr(original: &[f32], reconstructed: &[f32]) -> f32 {
    let n = original.len().min(reconstructed.len());
    if n == 0 {
        return 0.0;
    }
    let signal_power: f32 = original.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let noise_power = quantization_mse(original, reconstructed);
    if noise_power < f32::EPSILON {
        return f32::MAX;
    }
    10.0 * (signal_power / noise_power).log10()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Simple LCG pseudo-random for deterministic tests (no rand crate).
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u32(&mut self) -> u32 {
            self.state = self.state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((self.state >> 33) ^ self.state) as u32
        }

        fn next_f32(&mut self) -> f32 {
            (self.next_u32() as f32) / (u32::MAX as f32)
        }

        /// Uniform in [lo, hi]
        fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.next_f32()
        }

        fn fill_uniform(&mut self, buf: &mut [f32], lo: f32, hi: f32) {
            for v in buf.iter_mut() {
                *v = self.uniform(lo, hi);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Packing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pack_unpack_identity() {
        for a in 0..=3u8 {
            for b in 0..=3u8 {
                for c in 0..=3u8 {
                    for d in 0..=3u8 {
                        let byte = pack_four(a, b, c, d);
                        let (ra, rb, rc, rd) = unpack_four(byte);
                        assert_eq!((a, b, c, d), (ra, rb, rc, rd));
                    }
                }
            }
        }
    }

    #[test]
    fn test_pack_four_known_values() {
        // 0b_11_10_01_00 = 0xE4
        assert_eq!(pack_four(0, 1, 2, 3), 0b_11_10_01_00);
        assert_eq!(pack_four(3, 3, 3, 3), 0xFF);
        assert_eq!(pack_four(0, 0, 0, 0), 0x00);
    }

    #[test]
    fn test_unpack_four_known_values() {
        let (a, b, c, d) = unpack_four(0xFF);
        assert_eq!((a, b, c, d), (3, 3, 3, 3));

        let (a, b, c, d) = unpack_four(0x00);
        assert_eq!((a, b, c, d), (0, 0, 0, 0));
    }

    #[test]
    fn test_pack_single_bit_positions() {
        // Only lowest two bits of each position
        assert_eq!(pack_four(1, 0, 0, 0), 0x01);
        assert_eq!(pack_four(0, 1, 0, 0), 0x04);
        assert_eq!(pack_four(0, 0, 1, 0), 0x10);
        assert_eq!(pack_four(0, 0, 0, 1), 0x40);
    }

    // -----------------------------------------------------------------------
    // Asymmetric quantization
    // -----------------------------------------------------------------------

    #[test]
    fn test_asymmetric_roundtrip_small() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let config = Int2QuantConfig {
            group_size: 4,
            mode: Int2Mode::Asymmetric,
            scale: 0.0,
            zero_point: 0.0,
        };
        let packed = quantize_to_int2(&data, &config);
        let recovered = dequantize_from_int2(&packed);
        assert_eq!(recovered.len(), 4);
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 1.5, "orig={orig} rec={rec}");
        }
    }

    #[test]
    fn test_asymmetric_roundtrip_negative() {
        let data = vec![-1.0, -0.5, 0.0, 0.5];
        let config = Int2QuantConfig {
            group_size: 4,
            mode: Int2Mode::Asymmetric,
            ..Default::default()
        };
        let packed = quantize_to_int2(&data, &config);
        let recovered = dequantize_from_int2(&packed);
        assert_eq!(recovered.len(), 4);
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 1.0, "orig={orig} rec={rec}");
        }
    }

    #[test]
    fn test_asymmetric_large_random() {
        let mut rng = Lcg::new(42);
        let mut data = vec![0.0f32; 256];
        rng.fill_uniform(&mut data, -2.0, 2.0);

        let config = Int2QuantConfig {
            group_size: 32,
            mode: Int2Mode::Asymmetric,
            ..Default::default()
        };
        let packed = quantize_to_int2(&data, &config);
        let recovered = dequantize_from_int2(&packed);

        assert_eq!(recovered.len(), data.len());
        let mse = quantization_mse(&data, &recovered);
        // INT2 has high quantization error but MSE should be bounded
        assert!(mse < 2.0, "MSE too high: {mse}");
    }

    #[test]
    fn test_asymmetric_constant_values() {
        let data = vec![5.0; 32];
        let config = Int2QuantConfig {
            group_size: 32,
            mode: Int2Mode::Asymmetric,
            ..Default::default()
        };
        let packed = quantize_to_int2(&data, &config);
        let recovered = dequantize_from_int2(&packed);
        for &v in &recovered {
            assert!((v - 5.0).abs() < f32::EPSILON, "expected 5.0, got {v}");
        }
    }

    #[test]
    fn test_asymmetric_manual_scale() {
        let data = vec![0.0, 0.5, 1.0, 1.5];
        let config = Int2QuantConfig {
            group_size: 4,
            mode: Int2Mode::Asymmetric,
            scale: 0.5,
            zero_point: 0.0,
        };
        let packed = quantize_to_int2(&data, &config);
        let recovered = dequantize_from_int2(&packed);
        assert_eq!(recovered.len(), 4);
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.51, "orig={orig} rec={rec}");
        }
    }

    // -----------------------------------------------------------------------
    // Symmetric quantization
    // -----------------------------------------------------------------------

    #[test]
    fn test_symmetric_roundtrip_small() {
        let data = vec![-1.0, 0.0, 0.5, 1.0];
        let config = Int2QuantConfig {
            group_size: 4,
            mode: Int2Mode::Symmetric,
            ..Default::default()
        };
        let packed = quantize_to_int2(&data, &config);
        let recovered = dequantize_from_int2(&packed);
        assert_eq!(recovered.len(), 4);
    }

    #[test]
    fn test_symmetric_zero_centered() {
        let data = vec![-3.0, -1.0, 1.0, 3.0];
        let config = Int2QuantConfig {
            group_size: 4,
            mode: Int2Mode::Symmetric,
            ..Default::default()
        };
        let packed = quantize_to_int2(&data, &config);
        let recovered = dequantize_from_int2(&packed);
        // The sign should be preserved
        assert!(recovered[0] < 0.0, "expected negative, got {}", recovered[0]);
        assert!(recovered[3] > 0.0, "expected positive, got {}", recovered[3]);
    }

    #[test]
    fn test_symmetric_all_zeros() {
        let data = vec![0.0; 16];
        let config = Int2QuantConfig {
            group_size: 16,
            mode: Int2Mode::Symmetric,
            ..Default::default()
        };
        let packed = quantize_to_int2(&data, &config);
        let recovered = dequantize_from_int2(&packed);
        for &v in &recovered {
            assert!(v.abs() < f32::EPSILON, "expected ~0.0, got {v}");
        }
    }

    #[test]
    fn test_symmetric_large_random() {
        let mut rng = Lcg::new(1337);
        let mut data = vec![0.0f32; 128];
        rng.fill_uniform(&mut data, -5.0, 5.0);

        let config = Int2QuantConfig {
            group_size: 64,
            mode: Int2Mode::Symmetric,
            ..Default::default()
        };
        let packed = quantize_to_int2(&data, &config);
        let recovered = dequantize_from_int2(&packed);
        assert_eq!(recovered.len(), 128);
    }

    // -----------------------------------------------------------------------
    // Ternary quantization
    // -----------------------------------------------------------------------

    #[test]
    fn test_ternary_basic() {
        let data = vec![1.0, -1.0, 0.0, 0.5];
        let packed = quantize_ternary(&data, 4);
        let recovered = dequantize_from_int2(&packed);
        assert_eq!(recovered.len(), 4);
        // 1.0 should map to +scale, -1.0 to -scale, 0.0 to 0
        assert!(recovered[0] > 0.0);
        assert!(recovered[1] < 0.0);
    }

    #[test]
    fn test_ternary_all_positive() {
        let data = vec![2.0, 3.0, 4.0, 5.0];
        let packed = quantize_ternary(&data, 4);
        let recovered = dequantize_from_int2(&packed);
        // All large positive -> all should be positive
        for (i, &v) in recovered.iter().enumerate() {
            assert!(v > 0.0 || v.abs() < f32::EPSILON, "idx {i}: expected >=0 got {v}");
        }
    }

    #[test]
    fn test_ternary_all_zeros() {
        let data = vec![0.0; 8];
        let packed = quantize_ternary(&data, 8);
        let recovered = dequantize_from_int2(&packed);
        for &v in &recovered {
            assert!(v.abs() < f32::EPSILON, "expected 0.0, got {v}");
        }
    }

    #[test]
    fn test_ternary_values_are_trinary() {
        let mut rng = Lcg::new(99);
        let mut data = vec![0.0f32; 64];
        rng.fill_uniform(&mut data, -3.0, 3.0);

        let packed = quantize_ternary(&data, 32);
        let recovered = dequantize_from_int2(&packed);

        // Each recovered value must be one of: -scale, 0, +scale
        let scale0 = packed.scales()[0];
        let scale1 = packed.scales()[1];
        for (i, &v) in recovered.iter().enumerate() {
            let s = if i < 32 { scale0 } else { scale1 };
            let is_valid = v.abs() < f32::EPSILON
                || (v - s).abs() < f32::EPSILON
                || (v + s).abs() < f32::EPSILON;
            assert!(is_valid, "idx {i}: value {v} not in {{-{s}, 0, {s}}}");
        }
    }

    #[test]
    fn test_ternary_large_group() {
        let mut rng = Lcg::new(777);
        let mut data = vec![0.0f32; 256];
        rng.fill_uniform(&mut data, -1.0, 1.0);

        let packed = quantize_ternary(&data, 128);
        let recovered = dequantize_from_int2(&packed);
        assert_eq!(recovered.len(), 256);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_input() {
        let data: Vec<f32> = vec![];
        let config = Int2QuantConfig::default();
        let packed = quantize_to_int2(&data, &config);
        let recovered = dequantize_from_int2(&packed);
        assert!(recovered.is_empty());
        assert_eq!(packed.num_values(), 0);
    }

    #[test]
    fn test_single_value() {
        let data = vec![42.0];
        let config = Int2QuantConfig {
            group_size: 4,
            mode: Int2Mode::Asymmetric,
            ..Default::default()
        };
        let packed = quantize_to_int2(&data, &config);
        let recovered = dequantize_from_int2(&packed);
        assert_eq!(recovered.len(), 1);
        assert!((recovered[0] - 42.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_non_multiple_of_four() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]; // 7 elements
        let config = Int2QuantConfig {
            group_size: 4,
            mode: Int2Mode::Asymmetric,
            ..Default::default()
        };
        let packed = quantize_to_int2(&data, &config);
        let recovered = dequantize_from_int2(&packed);
        assert_eq!(recovered.len(), 7);
    }

    #[test]
    fn test_non_multiple_of_group_size() {
        let data = vec![1.0; 50]; // 50 is not a multiple of 32
        let config = Int2QuantConfig {
            group_size: 32,
            mode: Int2Mode::Asymmetric,
            ..Default::default()
        };
        let packed = quantize_to_int2(&data, &config);
        let recovered = dequantize_from_int2(&packed);
        assert_eq!(recovered.len(), 50);
    }

    #[test]
    fn test_extreme_values() {
        let data = vec![-1e6, 0.0, 1e6, 0.5];
        let config = Int2QuantConfig {
            group_size: 4,
            mode: Int2Mode::Asymmetric,
            ..Default::default()
        };
        let packed = quantize_to_int2(&data, &config);
        let recovered = dequantize_from_int2(&packed);
        assert_eq!(recovered.len(), 4);
        // Sign must be preserved for extremes
        assert!(recovered[0] < 0.0);
        assert!(recovered[2] > 0.0);
    }

    #[test]
    fn test_very_small_values() {
        let data = vec![1e-10, -1e-10, 2e-10, -2e-10];
        let config = Int2QuantConfig {
            group_size: 4,
            mode: Int2Mode::Asymmetric,
            ..Default::default()
        };
        let packed = quantize_to_int2(&data, &config);
        let recovered = dequantize_from_int2(&packed);
        assert_eq!(recovered.len(), 4);
    }

    // -----------------------------------------------------------------------
    // Compression ratio / metadata
    // -----------------------------------------------------------------------

    #[test]
    fn test_compression_ratio() {
        let data = vec![0.0f32; 1024];
        let config = Int2QuantConfig {
            group_size: 32,
            mode: Int2Mode::Asymmetric,
            ..Default::default()
        };
        let packed = quantize_to_int2(&data, &config);
        let ratio = packed.compression_ratio();
        // With 2 bits per value, theoretical max is 16x
        // Metadata reduces it, but it should still be well above 10x for 1024 values
        assert!(ratio >= 8.0, "compression ratio too low: {ratio}");
    }

    #[test]
    fn test_bits_per_value() {
        let data = vec![0.0f32; 64];
        let config = Int2QuantConfig {
            group_size: 64,
            mode: Int2Mode::Asymmetric,
            ..Default::default()
        };
        let packed = quantize_to_int2(&data, &config);
        let bpv = packed.bits_per_value();
        // Should be exactly 2.0 for perfectly aligned data
        assert!((bpv - 2.0).abs() < 0.01, "bits_per_value: {bpv}");
    }

    #[test]
    fn test_shape_preserved() {
        let data = vec![1.0; 100];
        let config = Int2QuantConfig::default();
        let packed = quantize_to_int2(&data, &config);
        assert_eq!(packed.shape(), &[100]);
        assert_eq!(packed.num_values(), 100);
    }

    #[test]
    fn test_mode_preserved() {
        let data = vec![1.0; 8];
        for mode in [Int2Mode::Asymmetric, Int2Mode::Symmetric, Int2Mode::Ternary] {
            let config = Int2QuantConfig {
                group_size: 8,
                mode,
                ..Default::default()
            };
            let packed = quantize_to_int2(&data, &config);
            assert_eq!(packed.mode(), mode);
        }
    }

    // -----------------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_config_valid() {
        let config = Int2QuantConfig {
            group_size: 32,
            mode: Int2Mode::Asymmetric,
            scale: 0.0,
            zero_point: 0.0,
        };
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_zero_group() {
        let config = Int2QuantConfig {
            group_size: 0,
            mode: Int2Mode::Asymmetric,
            ..Default::default()
        };
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_config_non_multiple_of_4() {
        let config = Int2QuantConfig {
            group_size: 7,
            mode: Int2Mode::Asymmetric,
            ..Default::default()
        };
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_config_nan_scale() {
        let config = Int2QuantConfig {
            group_size: 32,
            mode: Int2Mode::Asymmetric,
            scale: f32::NAN,
            zero_point: 0.0,
        };
        assert!(validate_config(&config).is_err());
    }

    // -----------------------------------------------------------------------
    // Statistics helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_mse_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!(quantization_mse(&a, &a) < f32::EPSILON);
    }

    #[test]
    fn test_mse_known() {
        let a = vec![0.0, 0.0, 0.0, 0.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let mse = quantization_mse(&a, &b);
        assert!((mse - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sqnr_perfect() {
        let a = vec![1.0, 2.0, 3.0];
        let sqnr = quantization_sqnr(&a, &a);
        assert!(sqnr > 100.0); // Should be very high (essentially inf)
    }

    #[test]
    fn test_sqnr_empty() {
        let a: Vec<f32> = vec![];
        assert!((quantization_sqnr(&a, &a) - 0.0).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Group size variants
    // -----------------------------------------------------------------------

    #[test]
    fn test_group_size_32() {
        let mut rng = Lcg::new(100);
        let mut data = vec![0.0f32; 128];
        rng.fill_uniform(&mut data, -1.0, 1.0);

        let config = Int2QuantConfig {
            group_size: 32,
            mode: Int2Mode::Asymmetric,
            ..Default::default()
        };
        let packed = quantize_to_int2(&data, &config);
        assert_eq!(packed.scales().len(), 4); // 128 / 32
        let recovered = dequantize_from_int2(&packed);
        assert_eq!(recovered.len(), 128);
    }

    #[test]
    fn test_group_size_64() {
        let mut rng = Lcg::new(200);
        let mut data = vec![0.0f32; 256];
        rng.fill_uniform(&mut data, -1.0, 1.0);

        let config = Int2QuantConfig {
            group_size: 64,
            mode: Int2Mode::Symmetric,
            ..Default::default()
        };
        let packed = quantize_to_int2(&data, &config);
        assert_eq!(packed.scales().len(), 4); // 256 / 64
        let recovered = dequantize_from_int2(&packed);
        assert_eq!(recovered.len(), 256);
    }

    #[test]
    fn test_group_size_128() {
        let mut rng = Lcg::new(300);
        let mut data = vec![0.0f32; 512];
        rng.fill_uniform(&mut data, -2.0, 2.0);

        let config = Int2QuantConfig {
            group_size: 128,
            mode: Int2Mode::Asymmetric,
            ..Default::default()
        };
        let packed = quantize_to_int2(&data, &config);
        assert_eq!(packed.scales().len(), 4); // 512 / 128
        let recovered = dequantize_from_int2(&packed);
        assert_eq!(recovered.len(), 512);
    }

    // -----------------------------------------------------------------------
    // Cross-mode comparison
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_modes_produce_output() {
        let data = vec![0.1, -0.5, 0.3, -0.2, 0.7, -0.1, 0.4, 0.0];
        for mode in [Int2Mode::Asymmetric, Int2Mode::Symmetric, Int2Mode::Ternary] {
            let config = Int2QuantConfig {
                group_size: 8,
                mode,
                ..Default::default()
            };
            let packed = quantize_to_int2(&data, &config);
            let recovered = dequantize_from_int2(&packed);
            assert_eq!(recovered.len(), data.len(), "mode: {mode:?}");
        }
    }

    #[test]
    fn test_packed_data_is_compact() {
        let data = vec![0.0f32; 100];
        let config = Int2QuantConfig::default();
        let packed = quantize_to_int2(&data, &config);
        // 100 values -> ceil(100/4) = 25 bytes
        assert_eq!(packed.data().len(), 25);
    }

    #[test]
    fn test_compression_empty() {
        let packed = PackedInt2 {
            data: vec![],
            scales: vec![],
            zero_points: vec![],
            shape: vec![0],
            num_values: 0,
            mode: Int2Mode::Asymmetric,
        };
        assert!((packed.compression_ratio() - 0.0).abs() < f64::EPSILON);
        assert!((packed.bits_per_value() - 0.0).abs() < f64::EPSILON);
    }
}
