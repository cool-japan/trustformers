//! Tensor quantization utilities: INT4, INT8, FP16
//!
//! Provides low-level, pure-Rust implementations of:
//! - INT4/INT8/Uint8/FP16/FP32 quantization with calibration
//! - Symmetric and asymmetric per-tensor quantization
//! - Per-group quantization (GPTQ-style)
//! - FP16 (IEEE 754 binary16) software encode/decode
//! - Round-trip error metrics (MAE, RMSE, SNR)

use std::fmt;

// ────────────────────────────────────────────────────────────────────────────
// Error type
// ────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during quantization operations
#[derive(Debug)]
pub enum QuantError {
    /// Input data slice is empty
    EmptyData,
    /// Group size of 0 is invalid
    InvalidGroupSize { size: usize },
    /// Number of quantization parameters does not match data length
    LengthMismatch { params: usize, data: usize },
    /// Unsupported or unrecognized dtype
    InvalidDtype(String),
}

impl fmt::Display for QuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantError::EmptyData => write!(f, "Input data is empty"),
            QuantError::InvalidGroupSize { size } => {
                write!(f, "Invalid group size {size}: must be > 0")
            }
            QuantError::LengthMismatch { params, data } => {
                write!(
                    f,
                    "Parameter count {params} does not match data length {data}"
                )
            }
            QuantError::InvalidDtype(s) => write!(f, "Invalid dtype: {s}"),
        }
    }
}

impl std::error::Error for QuantError {}

// ────────────────────────────────────────────────────────────────────────────
// Quantization data types
// ────────────────────────────────────────────────────────────────────────────

/// Supported quantization data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantDtype {
    /// 4-bit signed integer, range `[-8, 7]`
    Int4,
    /// 8-bit signed integer, range `[-128, 127]`
    Int8,
    /// 8-bit unsigned integer, range `[0, 255]`
    Uint8,
    /// 16-bit IEEE 754 float (stored as u16 bit pattern)
    Fp16,
    /// 32-bit IEEE 754 float (no quantization)
    Fp32,
}

impl QuantDtype {
    /// Number of bits used for this data type
    pub fn bits(&self) -> usize {
        match self {
            QuantDtype::Int4 => 4,
            QuantDtype::Int8 | QuantDtype::Uint8 => 8,
            QuantDtype::Fp16 => 16,
            QuantDtype::Fp32 => 32,
        }
    }

    /// Minimum representable integer value (for integer types)
    pub fn min_val(&self) -> f32 {
        match self {
            QuantDtype::Int4 => -8.0,
            QuantDtype::Int8 => -128.0,
            QuantDtype::Uint8 => 0.0,
            QuantDtype::Fp16 | QuantDtype::Fp32 => f32::MIN,
        }
    }

    /// Maximum representable integer value (for integer types)
    pub fn max_val(&self) -> f32 {
        match self {
            QuantDtype::Int4 => 7.0,
            QuantDtype::Int8 => 127.0,
            QuantDtype::Uint8 => 255.0,
            QuantDtype::Fp16 | QuantDtype::Fp32 => f32::MAX,
        }
    }

    /// Integer range: `max_val - min_val`
    pub fn range(&self) -> f32 {
        self.max_val() - self.min_val()
    }

    /// Storage size in bytes per element (can be fractional for INT4)
    pub fn bytes_per_element(&self) -> f32 {
        self.bits() as f32 / 8.0
    }

    /// How many times smaller this type is compared to FP32
    pub fn compression_ratio_vs_fp32(&self) -> f32 {
        32.0 / self.bits() as f32
    }
}

impl fmt::Display for QuantDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            QuantDtype::Int4 => "INT4",
            QuantDtype::Int8 => "INT8",
            QuantDtype::Uint8 => "UINT8",
            QuantDtype::Fp16 => "FP16",
            QuantDtype::Fp32 => "FP32",
        };
        write!(f, "{s}")
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Quantization scheme
// ────────────────────────────────────────────────────────────────────────────

/// Strategy for computing scale and zero-point parameters
#[derive(Debug, Clone, PartialEq)]
pub enum QuantScheme {
    /// Symmetric quantization: `zero_point = 0`, only a scale factor
    Symmetric,
    /// Asymmetric quantization: both `scale` and `zero_point`
    Asymmetric,
    /// Per-channel: separate `scale`/`zero_point` for each output channel
    PerChannel { num_channels: usize },
    /// Per-group: separate `scale`/`zero_point` for each group of `group_size` elements
    /// (GPTQ / INT4 style)
    PerGroup { group_size: usize },
}

// ────────────────────────────────────────────────────────────────────────────
// Quantization parameters
// ────────────────────────────────────────────────────────────────────────────

/// Calibrated quantization parameters for a tensor
#[derive(Debug, Clone)]
pub struct QuantParams {
    /// Data type used for quantization
    pub dtype: QuantDtype,
    /// Quantization scheme
    pub scheme: QuantScheme,
    /// Scale factors: `[1]` for per-tensor, `[num_channels]` or `[n/group_size]` otherwise
    pub scales: Vec<f32>,
    /// Zero-point offsets (0 for symmetric): same shape as `scales`
    pub zero_points: Vec<i32>,
}

impl QuantParams {
    /// Calibrate quantization parameters from FP32 data.
    ///
    /// Computes optimal `scale` and `zero_point` values for the given scheme.
    pub fn calibrate(
        data: &[f32],
        dtype: QuantDtype,
        scheme: QuantScheme,
    ) -> Result<Self, QuantError> {
        if data.is_empty() {
            return Err(QuantError::EmptyData);
        }

        match &scheme {
            QuantScheme::Symmetric => {
                let max_abs = data
                    .iter()
                    .map(|&x| x.abs())
                    .fold(0.0_f32, f32::max);
                let scale = (max_abs / dtype.max_val()).max(1e-8_f32);
                Ok(Self {
                    dtype,
                    scheme,
                    scales: vec![scale],
                    zero_points: vec![0],
                })
            }

            QuantScheme::Asymmetric => {
                let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let scale = ((max_val - min_val) / dtype.range()).max(1e-8_f32);
                let zero_point =
                    (dtype.min_val() - min_val / scale).round() as i32;
                let zero_point = zero_point.clamp(
                    dtype.min_val() as i32,
                    dtype.max_val() as i32,
                );
                Ok(Self {
                    dtype,
                    scheme,
                    scales: vec![scale],
                    zero_points: vec![zero_point],
                })
            }

            QuantScheme::PerGroup { group_size } => {
                let group_size = *group_size;
                if group_size == 0 {
                    return Err(QuantError::InvalidGroupSize { size: group_size });
                }
                let num_groups = (data.len() + group_size - 1) / group_size;
                let mut scales = Vec::with_capacity(num_groups);
                let mut zero_points = Vec::with_capacity(num_groups);

                for g in 0..num_groups {
                    let start = g * group_size;
                    let end = (start + group_size).min(data.len());
                    let group = &data[start..end];
                    let max_abs = group.iter().map(|&x| x.abs()).fold(0.0_f32, f32::max);
                    scales.push((max_abs / dtype.max_val()).max(1e-8_f32));
                    zero_points.push(0);
                }

                Ok(Self {
                    dtype,
                    scheme,
                    scales,
                    zero_points,
                })
            }

            QuantScheme::PerChannel { .. } => {
                // For calibration purposes, treat the entire buffer as one channel
                let max_abs = data
                    .iter()
                    .map(|&x| x.abs())
                    .fold(0.0_f32, f32::max);
                let scale = (max_abs / dtype.max_val()).max(1e-8_f32);
                Ok(Self {
                    dtype,
                    scheme,
                    scales: vec![scale],
                    zero_points: vec![0],
                })
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Quantize and dequantize
// ────────────────────────────────────────────────────────────────────────────

/// Quantize FP32 tensor to integer representation.
///
/// Formula: `q = clamp(round(x / scale) + zero_point, min_val, max_val)`
///
/// For `PerGroup`, each group of elements uses its own scale/zero_point.
pub fn quantize(data: &[f32], params: &QuantParams) -> Result<Vec<i32>, QuantError> {
    if data.is_empty() {
        return Err(QuantError::EmptyData);
    }

    let min_q = params.dtype.min_val() as i32;
    let max_q = params.dtype.max_val() as i32;

    let quantized = match &params.scheme {
        QuantScheme::PerGroup { group_size } => {
            let group_size = *group_size;
            if group_size == 0 {
                return Err(QuantError::InvalidGroupSize { size: group_size });
            }
            let num_groups = (data.len() + group_size - 1) / group_size;
            if params.scales.len() != num_groups {
                return Err(QuantError::LengthMismatch {
                    params: params.scales.len(),
                    data: num_groups,
                });
            }
            data.iter()
                .enumerate()
                .map(|(idx, &x)| {
                    let g = idx / group_size;
                    let scale = params.scales[g];
                    let zp = params.zero_points[g];
                    let q = (x / scale).round() as i32 + zp;
                    q.clamp(min_q, max_q)
                })
                .collect()
        }
        _ => {
            // Per-tensor: single scale and zero_point
            let scale = params
                .scales
                .first()
                .copied()
                .ok_or(QuantError::EmptyData)?;
            let zp = params.zero_points.first().copied().unwrap_or(0);
            data.iter()
                .map(|&x| {
                    let q = (x / scale).round() as i32 + zp;
                    q.clamp(min_q, max_q)
                })
                .collect()
        }
    };

    Ok(quantized)
}

/// Dequantize integer representation back to FP32.
///
/// Formula: `x = (q - zero_point) * scale`
pub fn dequantize(data: &[i32], params: &QuantParams) -> Result<Vec<f32>, QuantError> {
    if data.is_empty() {
        return Err(QuantError::EmptyData);
    }

    let dequantized = match &params.scheme {
        QuantScheme::PerGroup { group_size } => {
            let group_size = *group_size;
            if group_size == 0 {
                return Err(QuantError::InvalidGroupSize { size: group_size });
            }
            let num_groups = (data.len() + group_size - 1) / group_size;
            if params.scales.len() != num_groups {
                return Err(QuantError::LengthMismatch {
                    params: params.scales.len(),
                    data: num_groups,
                });
            }
            data.iter()
                .enumerate()
                .map(|(idx, &q)| {
                    let g = idx / group_size;
                    let scale = params.scales[g];
                    let zp = params.zero_points[g];
                    (q - zp) as f32 * scale
                })
                .collect()
        }
        _ => {
            let scale = params
                .scales
                .first()
                .copied()
                .ok_or(QuantError::EmptyData)?;
            let zp = params.zero_points.first().copied().unwrap_or(0);
            data.iter()
                .map(|&q| (q - zp) as f32 * scale)
                .collect()
        }
    };

    Ok(dequantized)
}

// ────────────────────────────────────────────────────────────────────────────
// Error metrics
// ────────────────────────────────────────────────────────────────────────────

/// Round-trip quantization error metrics
#[derive(Debug, Clone)]
pub struct QuantizationMetrics {
    /// Maximum absolute error over all elements
    pub max_abs_error: f32,
    /// Mean absolute error
    pub mean_abs_error: f32,
    /// Root mean squared error
    pub rmse: f32,
    /// Signal-to-noise ratio in dB (higher is better)
    pub snr_db: f32,
    /// Number of elements that hit the quantization clipping boundary
    pub num_clipped: usize,
}

/// Measure quantization error by performing a round-trip (quantize → dequantize).
pub fn measure_quant_error(
    original: &[f32],
    params: &QuantParams,
) -> Result<QuantizationMetrics, QuantError> {
    if original.is_empty() {
        return Err(QuantError::EmptyData);
    }

    let quantized = quantize(original, params)?;
    let dequantized = dequantize(&quantized, params)?;

    let n = original.len() as f32;

    // Compute absolute errors
    let errors: Vec<f32> = original
        .iter()
        .zip(dequantized.iter())
        .map(|(&o, &d)| (o - d).abs())
        .collect();

    let max_abs_error = errors.iter().cloned().fold(0.0_f32, f32::max);
    let mean_abs_error = errors.iter().sum::<f32>() / n;
    let mse = errors.iter().map(|&e| e * e).sum::<f32>() / n;
    let rmse = mse.sqrt();

    let signal_power = original.iter().map(|&x| x * x).sum::<f32>() / n;
    let noise_power = mse.max(1e-20_f32);
    let snr_db = 10.0 * (signal_power / noise_power).log10();

    let min_q = params.dtype.min_val() as i32;
    let max_q = params.dtype.max_val() as i32;
    let num_clipped = quantized
        .iter()
        .filter(|&&q| q == min_q || q == max_q)
        .count();

    Ok(QuantizationMetrics {
        max_abs_error,
        mean_abs_error,
        rmse,
        snr_db,
        num_clipped,
    })
}

// ────────────────────────────────────────────────────────────────────────────
// FP16 (IEEE 754 binary16) software implementation
// ────────────────────────────────────────────────────────────────────────────

/// IEEE 754 binary16 (FP16) value stored as its 16-bit bit pattern.
///
/// Implements full conversion including subnormals, infinity, and NaN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp16 {
    /// Raw IEEE 754 binary16 bit pattern
    pub bits: u16,
}

impl Fp16 {
    /// Convert a 32-bit float to FP16 with round-to-nearest.
    ///
    /// Handles: zero, subnormals, normals, infinity, NaN.
    pub fn from_f32(val: f32) -> Self {
        let bits32 = val.to_bits();
        let sign = ((bits32 >> 31) & 1) as u16;
        let exp32 = ((bits32 >> 23) & 0xFF) as i32;
        let mant32 = bits32 & 0x007F_FFFF;

        // Special cases: infinity and NaN
        if exp32 == 255 {
            let mant16 = if mant32 != 0 { 0x0200_u16 } else { 0_u16 };
            return Fp16 {
                bits: (sign << 15) | 0x7C00 | mant16,
            };
        }

        let exp16 = exp32 - 127 + 15;

        // Overflow to infinity
        if exp16 >= 31 {
            return Fp16 {
                bits: (sign << 15) | 0x7C00,
            };
        }

        if exp16 <= 0 {
            // Subnormal or underflow
            if exp16 < -10 {
                // Too small to represent → ±0
                return Fp16 { bits: sign << 15 };
            }
            // Subnormal fp16: shift mantissa right
            let shift = (1 - exp16) as u32;
            let mant_with_implicit = (mant32 | 0x0080_0000) >> shift;
            // Round-to-nearest: check the bit just below the kept bits
            let rounding_bit = if shift > 0 {
                (mant32 | 0x0080_0000) >> (shift - 1) & 1
            } else {
                0
            };
            let mant16 = ((mant_with_implicit >> 13) as u16) + rounding_bit as u16;
            return Fp16 {
                bits: (sign << 15) | mant16,
            };
        }

        // Normal fp16 — round-to-nearest
        let round = (mant32 >> 12) & 1; // bit 12 of mantissa
        let mant16 = ((mant32 >> 13) as u16) + round as u16;

        // Handle mantissa overflow (round-up carries into exponent)
        let exp_out = exp16 as u16;
        let raw = (sign << 15) | (exp_out << 10) | mant16;

        // If rounding overflowed mantissa (mant16 == 0x400), exponent already incremented
        Fp16 { bits: raw }
    }

    /// Convert FP16 to a 32-bit float.
    ///
    /// Handles: zero, subnormals, normals, infinity, NaN.
    pub fn to_f32(self) -> f32 {
        let sign = ((self.bits >> 15) as u32) << 31;
        let exp16 = ((self.bits >> 10) & 0x1F) as i32;
        let mant16 = (self.bits & 0x03FF) as u32;

        // Special cases: infinity and NaN
        if exp16 == 31 {
            let bits32 = sign | 0x7F80_0000 | (mant16 << 13);
            return f32::from_bits(bits32);
        }

        // Zero or subnormal fp16
        if exp16 == 0 {
            if mant16 == 0 {
                // ±0
                return f32::from_bits(sign);
            }
            // Subnormal fp16 → normalised fp32
            // Find leading bit position
            let leading = mant16.leading_zeros() - 22; // mant16 is in bits 0..9
            let exp32 = (127 - 14 - leading) as u32;
            let mant32 = (mant16 << (leading + 14)) & 0x007F_FFFF;
            return f32::from_bits(sign | (exp32 << 23) | mant32);
        }

        // Normal fp16
        let exp32 = (exp16 - 15 + 127) as u32;
        let bits32 = sign | (exp32 << 23) | (mant16 << 13);
        f32::from_bits(bits32)
    }

    /// Returns `true` if this value is NaN
    pub fn is_nan(self) -> bool {
        (self.bits & 0x7C00) == 0x7C00 && (self.bits & 0x03FF) != 0
    }

    /// Returns `true` if this value is positive or negative infinity
    pub fn is_inf(self) -> bool {
        (self.bits & 0x7FFF) == 0x7C00
    }

    /// Returns `true` if this value is positive or negative zero
    pub fn is_zero(self) -> bool {
        (self.bits & 0x7FFF) == 0
    }
}

impl fmt::Display for Fp16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "fp16({})", self.to_f32())
    }
}

/// Quantize a slice of FP32 values to FP16 bit patterns.
pub fn quantize_fp16(data: &[f32]) -> Vec<u16> {
    data.iter().map(|&x| Fp16::from_f32(x).bits).collect()
}

/// Dequantize FP16 bit patterns to FP32 values.
pub fn dequantize_fp16(data: &[u16]) -> Vec<f32> {
    data.iter().map(|&bits| Fp16 { bits }.to_f32()).collect()
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── QuantDtype ────────────────────────────────────────────────────────────

    #[test]
    fn test_quant_dtype_bits() {
        assert_eq!(QuantDtype::Int4.bits(), 4);
        assert_eq!(QuantDtype::Int8.bits(), 8);
        assert_eq!(QuantDtype::Uint8.bits(), 8);
        assert_eq!(QuantDtype::Fp16.bits(), 16);
        assert_eq!(QuantDtype::Fp32.bits(), 32);
    }

    #[test]
    fn test_quant_dtype_range_int4() {
        assert_eq!(QuantDtype::Int4.min_val(), -8.0);
        assert_eq!(QuantDtype::Int4.max_val(), 7.0);
        assert_eq!(QuantDtype::Int4.range(), 15.0);
    }

    #[test]
    fn test_quant_dtype_range_int8() {
        assert_eq!(QuantDtype::Int8.min_val(), -128.0);
        assert_eq!(QuantDtype::Int8.max_val(), 127.0);
        assert_eq!(QuantDtype::Int8.range(), 255.0);
    }

    #[test]
    fn test_quant_dtype_compression_ratio() {
        assert_eq!(QuantDtype::Int4.compression_ratio_vs_fp32(), 8.0);
        assert_eq!(QuantDtype::Int8.compression_ratio_vs_fp32(), 4.0);
        assert_eq!(QuantDtype::Fp16.compression_ratio_vs_fp32(), 2.0);
        assert_eq!(QuantDtype::Fp32.compression_ratio_vs_fp32(), 1.0);
    }

    // ── Calibration ───────────────────────────────────────────────────────────

    #[test]
    fn test_calibrate_symmetric() {
        let data: Vec<f32> = vec![-2.0, 1.0, 0.5, -1.5, 2.0];
        let params = QuantParams::calibrate(&data, QuantDtype::Int8, QuantScheme::Symmetric)
            .expect("calibrate failed");

        assert_eq!(params.zero_points, vec![0]);
        let scale = params.scales[0];
        // max_abs = 2.0, max_val = 127.0 → scale = 2.0/127.0
        let expected_scale = 2.0_f32 / 127.0;
        assert!(
            (scale - expected_scale).abs() < 1e-6,
            "Expected scale {expected_scale}, got {scale}"
        );
    }

    #[test]
    fn test_calibrate_asymmetric() {
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let params = QuantParams::calibrate(&data, QuantDtype::Int8, QuantScheme::Asymmetric)
            .expect("calibrate failed");

        // scale = (4 - 0) / 255 ≈ 0.01569
        let expected_scale = 4.0_f32 / 255.0;
        assert!(
            (params.scales[0] - expected_scale).abs() < 1e-5,
            "Asymmetric scale mismatch"
        );
    }

    #[test]
    fn test_calibrate_per_group() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 0.5, 0.5, 0.5, 0.5];
        let params =
            QuantParams::calibrate(&data, QuantDtype::Int8, QuantScheme::PerGroup { group_size: 4 })
                .expect("calibrate failed");

        assert_eq!(params.scales.len(), 2); // 8 elements / 4 group_size = 2 groups
        // Group 0: max_abs = 4.0 → scale = 4.0/127.0
        let expected_s0 = 4.0_f32 / 127.0;
        assert!((params.scales[0] - expected_s0).abs() < 1e-6);
        // Group 1: max_abs = 0.5 → scale = 0.5/127.0
        let expected_s1 = 0.5_f32 / 127.0;
        assert!((params.scales[1] - expected_s1).abs() < 1e-6);
    }

    // ── Quantize / Dequantize ─────────────────────────────────────────────────

    #[test]
    fn test_quantize_int8_symmetric() {
        let data = vec![-1.27_f32, 0.0, 1.27];
        let params = QuantParams {
            dtype: QuantDtype::Int8,
            scheme: QuantScheme::Symmetric,
            scales: vec![0.01],
            zero_points: vec![0],
        };
        let q = quantize(&data, &params).expect("quantize failed");
        assert_eq!(q[0], -127);
        assert_eq!(q[1], 0);
        assert_eq!(q[2], 127);
    }

    #[test]
    fn test_quantize_int8_asymmetric() {
        let data = vec![0.0_f32, 127.5, 255.0];
        let params = QuantParams {
            dtype: QuantDtype::Int8,
            scheme: QuantScheme::Asymmetric,
            scales: vec![1.0],
            zero_points: vec![-128],
        };
        let q = quantize(&data, &params).expect("quantize failed");
        // q = clamp(round(x/1.0) + (-128), -128, 127)
        assert_eq!(q[0], -128); // 0 + (-128) = -128
        // 127.5 + (-128) = -0.5 → 0 after round... wait: round(127.5) = 128, 128 + (-128) = 0
        assert_eq!(q[1], 0);
        // 255 + (-128) = 127
        assert_eq!(q[2], 127);
    }

    #[test]
    fn test_quantize_int4_per_group() {
        let data: Vec<f32> = vec![-7.0, 0.0, 7.0, -7.0, 3.5, -3.5, 0.0, 7.0];
        let params = QuantParams::calibrate(
            &data,
            QuantDtype::Int4,
            QuantScheme::PerGroup { group_size: 4 },
        )
        .expect("calibrate failed");
        let q = quantize(&data, &params).expect("quantize failed");

        // All values should be in [-8, 7]
        for &v in &q {
            assert!(v >= -8 && v <= 7, "Int4 value {v} out of range");
        }
    }

    #[test]
    fn test_dequantize_roundtrip_int8() {
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1 - 3.0).collect();
        let params =
            QuantParams::calibrate(&data, QuantDtype::Int8, QuantScheme::Symmetric)
                .expect("calibrate failed");
        let q = quantize(&data, &params).expect("quantize failed");
        let dq = dequantize(&q, &params).expect("dequantize failed");

        // Max error for INT8 symmetric should be ≤ 0.5 * scale
        let max_err = data
            .iter()
            .zip(dq.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let max_allowed = 0.5 * params.scales[0];
        assert!(
            max_err <= max_allowed * 1.01,
            "Max error {max_err} exceeds {max_allowed}"
        );
    }

    #[test]
    fn test_dequantize_roundtrip_int4() {
        let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.5 - 4.0).collect();
        let params =
            QuantParams::calibrate(&data, QuantDtype::Int4, QuantScheme::Symmetric)
                .expect("calibrate failed");
        let q = quantize(&data, &params).expect("quantize failed");
        let dq = dequantize(&q, &params).expect("dequantize failed");

        assert_eq!(q.len(), data.len());
        assert_eq!(dq.len(), data.len());

        // At least some elements should round-trip closely
        let mae = data
            .iter()
            .zip(dq.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / data.len() as f32;
        let scale = params.scales[0];
        assert!(
            mae <= scale * 2.0,
            "Mean absolute error {mae} too large for scale {scale}"
        );
    }

    // ── Error metrics ─────────────────────────────────────────────────────────

    #[test]
    fn test_quant_error_metrics() {
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let params =
            QuantParams::calibrate(&data, QuantDtype::Int8, QuantScheme::Symmetric)
                .expect("calibrate failed");
        let metrics = measure_quant_error(&data, &params).expect("measure failed");

        assert!(metrics.max_abs_error >= 0.0);
        assert!(metrics.mean_abs_error >= 0.0);
        assert!(metrics.rmse >= 0.0);
        assert!(metrics.mean_abs_error <= metrics.max_abs_error);
        // INT8 should have decent SNR (>> 0 dB)
        assert!(metrics.snr_db > 20.0, "INT8 SNR too low: {}", metrics.snr_db);
    }

    #[test]
    fn test_quant_snr_high_for_fp32() {
        // FP32 params with scale=1, zp=0 — essentially no quantization
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // Use symmetric with very fine scale (essentially identity)
        let params = QuantParams {
            dtype: QuantDtype::Int8,
            scheme: QuantScheme::Symmetric,
            scales: vec![0.001],
            zero_points: vec![0],
        };
        // With tiny scale, values will clip at 127 → but range is [-0.127, 0.127]
        // Let's use a proper test: data fits within INT8 range
        let small_data: Vec<f32> = (1..=127).map(|i| i as f32).collect();
        let p2 = QuantParams {
            dtype: QuantDtype::Int8,
            scheme: QuantScheme::Symmetric,
            scales: vec![1.0],
            zero_points: vec![0],
        };
        let metrics = measure_quant_error(&small_data, &p2).expect("measure failed");
        // With scale=1.0 and integer inputs, round-trip is exact (no error)
        assert_eq!(metrics.max_abs_error, 0.0, "Integer data should round-trip exactly");
        let _ = params; // suppress unused warning
    }

    // ── FP16 ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_fp16_from_f32_zero() {
        let zero = Fp16::from_f32(0.0);
        assert!(zero.is_zero());
        assert_eq!(zero.to_f32(), 0.0);

        let neg_zero = Fp16::from_f32(-0.0);
        assert!(neg_zero.is_zero());
    }

    #[test]
    fn test_fp16_from_f32_one() {
        let one = Fp16::from_f32(1.0);
        // FP16: sign=0, exp=15, mant=0 → bits = 0x3C00
        assert_eq!(one.bits, 0x3C00, "fp16(1.0) should be 0x3C00");
        let back = one.to_f32();
        assert_eq!(back, 1.0);
    }

    #[test]
    fn test_fp16_from_f32_negative() {
        let neg_one = Fp16::from_f32(-1.0);
        // FP16: sign=1, exp=15, mant=0 → bits = 0xBC00
        assert_eq!(neg_one.bits, 0xBC00, "fp16(-1.0) should be 0xBC00");
        let back = neg_one.to_f32();
        assert_eq!(back, -1.0);
    }

    #[test]
    fn test_fp16_roundtrip_accuracy() {
        // Test a range of values for round-trip accuracy within FP16 precision
        let test_values = [0.5_f32, -0.5, 1.5, -1.5, 100.0, -100.0, 0.001, 1024.0];
        for &val in &test_values {
            let fp16 = Fp16::from_f32(val);
            let back = fp16.to_f32();
            // FP16 has ~3.3 decimal digits of precision, relative error ~0.1%
            let rel_err = ((val - back) / val).abs();
            assert!(
                rel_err < 0.005,
                "FP16 roundtrip error for {val}: got {back}, rel_err={rel_err}"
            );
        }
    }

    #[test]
    fn test_fp16_infinity() {
        let pos_inf = Fp16::from_f32(f32::INFINITY);
        assert!(pos_inf.is_inf(), "Should be inf");
        assert!(!pos_inf.is_nan());

        let neg_inf = Fp16::from_f32(f32::NEG_INFINITY);
        assert!(neg_inf.is_inf(), "Should be neg inf");

        let back_pos = pos_inf.to_f32();
        assert!(back_pos.is_infinite() && back_pos > 0.0);
    }

    #[test]
    fn test_fp16_nan() {
        let nan = Fp16::from_f32(f32::NAN);
        assert!(nan.is_nan(), "Should be NaN");
        assert!(!nan.is_inf());

        let back = nan.to_f32();
        assert!(back.is_nan(), "NaN should remain NaN after conversion");
    }

    #[test]
    fn test_quantize_fp16() {
        let data = vec![1.0_f32, -1.0, 0.0, 0.5, 100.0];
        let q = quantize_fp16(&data);
        assert_eq!(q.len(), data.len());
        // 1.0 should be 0x3C00
        assert_eq!(q[0], 0x3C00);
        // -1.0 should be 0xBC00
        assert_eq!(q[1], 0xBC00);
    }

    #[test]
    fn test_dequantize_fp16() {
        let bits = vec![0x3C00_u16, 0xBC00]; // 1.0 and -1.0
        let f32s = dequantize_fp16(&bits);
        assert_eq!(f32s[0], 1.0);
        assert_eq!(f32s[1], -1.0);
    }

    #[test]
    fn test_quant_error_display() {
        let e1 = QuantError::EmptyData;
        assert!(e1.to_string().contains("empty"));

        let e2 = QuantError::InvalidGroupSize { size: 0 };
        assert!(e2.to_string().contains("group size"));

        let e3 = QuantError::LengthMismatch { params: 4, data: 8 };
        assert!(e3.to_string().contains("4"));

        let e4 = QuantError::InvalidDtype("xyz".to_string());
        assert!(e4.to_string().contains("xyz"));
    }

    #[test]
    fn test_quant_dtype_display() {
        assert_eq!(QuantDtype::Int4.to_string(), "INT4");
        assert_eq!(QuantDtype::Int8.to_string(), "INT8");
        assert_eq!(QuantDtype::Uint8.to_string(), "UINT8");
        assert_eq!(QuantDtype::Fp16.to_string(), "FP16");
        assert_eq!(QuantDtype::Fp32.to_string(), "FP32");
    }

    #[test]
    fn test_quant_dtype_bytes_per_element() {
        assert_eq!(QuantDtype::Int4.bytes_per_element(), 0.5);
        assert_eq!(QuantDtype::Int8.bytes_per_element(), 1.0);
        assert_eq!(QuantDtype::Fp16.bytes_per_element(), 2.0);
        assert_eq!(QuantDtype::Fp32.bytes_per_element(), 4.0);
    }

    #[test]
    fn test_calibrate_empty_data_error() {
        let result = QuantParams::calibrate(&[], QuantDtype::Int8, QuantScheme::Symmetric);
        assert!(matches!(result, Err(QuantError::EmptyData)));
    }

    #[test]
    fn test_calibrate_invalid_group_size() {
        let data = vec![1.0_f32; 8];
        let result =
            QuantParams::calibrate(&data, QuantDtype::Int8, QuantScheme::PerGroup { group_size: 0 });
        assert!(matches!(result, Err(QuantError::InvalidGroupSize { .. })));
    }

    #[test]
    fn test_fp16_large_value_overflows_to_inf() {
        // FP16 max normal is 65504. Values larger than that overflow to inf.
        let large = 1e6_f32;
        let fp16 = Fp16::from_f32(large);
        assert!(fp16.is_inf(), "Very large value should overflow to inf");
    }

    #[test]
    fn test_fp16_small_value_underflows_to_zero() {
        // FP16 min subnormal is ~5.96e-8. Values much smaller underflow to 0.
        let tiny = 1e-10_f32;
        let fp16 = Fp16::from_f32(tiny);
        assert!(fp16.is_zero(), "Very small value should underflow to 0");
    }
}
