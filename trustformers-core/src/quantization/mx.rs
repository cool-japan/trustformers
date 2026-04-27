//! MX (Microscaling) format quantization for future hardware support
//!
//! Microscaling formats use a shared exponent per block with individual mantissa
//! bits per element. This provides an efficient compression scheme that balances
//! dynamic range coverage (via shared exponent) with per-element precision
//! (via individual mantissa bits).
//!
//! Supported formats:
//! - **MXFP8**: 8-bit floating point with shared exponent (1 sign + shared exp + mantissa)
//! - **MXFP6**: 6-bit floating point with shared exponent
//! - **MXFP4**: 4-bit floating point with shared exponent
//! - **MXINT8**: 8-bit integer with shared exponent
//!
//! Block sizes must be powers of 2: 2, 4, 8, 16, or 32.
//!
//! # References
//! - OCP Microscaling Formats (MX) Specification v1.0
//! - "Microscaling Data Formats for Deep Learning" (2023)

use crate::errors::{Result, TrustformersError};

/// MX format variants
///
/// Each variant defines the number of mantissa bits and whether the per-element
/// representation is floating-point or integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MxFormat {
    /// MXFP8: 1 sign bit + 4 exponent bits + 3 mantissa bits per element
    /// Plus 8-bit shared exponent per block
    Mxfp8,

    /// MXFP6: 1 sign bit + 3 exponent bits + 2 mantissa bits per element
    /// Plus 8-bit shared exponent per block
    Mxfp6,

    /// MXFP4: 1 sign bit + 2 exponent bits + 1 mantissa bit per element
    /// Plus 8-bit shared exponent per block
    Mxfp4,

    /// MXINT8: 1 sign bit + 7 magnitude bits per element (integer)
    /// Plus 8-bit shared exponent per block
    Mxint8,
}

impl MxFormat {
    /// Total bits per element (excluding the amortized shared exponent)
    pub fn element_bits(&self) -> u8 {
        match self {
            MxFormat::Mxfp8 => 8,
            MxFormat::Mxfp6 => 6,
            MxFormat::Mxfp4 => 4,
            MxFormat::Mxint8 => 8,
        }
    }

    /// Number of mantissa bits per element
    pub fn mantissa_bits(&self) -> u8 {
        match self {
            MxFormat::Mxfp8 => 3,
            MxFormat::Mxfp6 => 2,
            MxFormat::Mxfp4 => 1,
            MxFormat::Mxint8 => 7, // integer magnitude bits
        }
    }

    /// Number of exponent bits per element (0 for integer format)
    pub fn element_exponent_bits(&self) -> u8 {
        match self {
            MxFormat::Mxfp8 => 4,
            MxFormat::Mxfp6 => 3,
            MxFormat::Mxfp4 => 2,
            MxFormat::Mxint8 => 0,
        }
    }

    /// Exponent bias for per-element exponents
    pub fn element_exponent_bias(&self) -> i32 {
        match self {
            MxFormat::Mxfp8 => 7,  // 2^(4-1) - 1
            MxFormat::Mxfp6 => 3,  // 2^(3-1) - 1
            MxFormat::Mxfp4 => 1,  // 2^(2-1) - 1
            MxFormat::Mxint8 => 0, // no per-element exponent
        }
    }

    /// Whether this format uses floating-point per-element representation
    pub fn is_float_format(&self) -> bool {
        !matches!(self, MxFormat::Mxint8)
    }

    /// Maximum representable per-element value (before shared exponent scaling)
    /// For FP formats: (2 - 2^(-mantissa_bits)) * 2^(max_exp - bias)
    /// For INT format: 2^mantissa_bits - 1 = 127
    pub fn max_element_value(&self) -> f32 {
        match self {
            MxFormat::Mxfp8 => {
                // max exp = 15 (all 1s reserved for special), use 14
                // value = (2 - 2^-3) * 2^(14-7) = 1.875 * 128 = 240
                let max_exp = (1 << self.element_exponent_bits()) - 2; // 14
                let bias = self.element_exponent_bias(); // 7
                let mantissa_frac = 1.0 - 2.0f32.powi(-(self.mantissa_bits() as i32));
                (1.0 + mantissa_frac) * 2.0f32.powi(max_exp - bias)
            },
            MxFormat::Mxfp6 => {
                // max exp = 6, bias = 3
                let max_exp = (1 << self.element_exponent_bits()) - 2; // 6
                let bias = self.element_exponent_bias(); // 3
                let mantissa_frac = 1.0 - 2.0f32.powi(-(self.mantissa_bits() as i32));
                (1.0 + mantissa_frac) * 2.0f32.powi(max_exp - bias)
            },
            MxFormat::Mxfp4 => {
                // max exp = 2, bias = 1
                let max_exp = (1 << self.element_exponent_bits()) - 2; // 2
                let bias = self.element_exponent_bias(); // 1
                let mantissa_frac = 1.0 - 2.0f32.powi(-(self.mantissa_bits() as i32));
                (1.0 + mantissa_frac) * 2.0f32.powi(max_exp - bias)
            },
            MxFormat::Mxint8 => 127.0,
        }
    }
}

impl std::fmt::Display for MxFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MxFormat::Mxfp8 => write!(f, "MXFP8"),
            MxFormat::Mxfp6 => write!(f, "MXFP6"),
            MxFormat::Mxfp4 => write!(f, "MXFP4"),
            MxFormat::Mxint8 => write!(f, "MXINT8"),
        }
    }
}

/// Valid block sizes for MX quantization
const VALID_BLOCK_SIZES: [usize; 5] = [2, 4, 8, 16, 32];

/// Configuration for MX quantization
#[derive(Debug, Clone)]
pub struct MxQuantConfig {
    /// MX format to use
    pub format: MxFormat,

    /// Block size (must be a power of 2: 2, 4, 8, 16, or 32)
    pub block_size: usize,
}

impl MxQuantConfig {
    /// Create a new MX quantization configuration
    ///
    /// # Errors
    /// Returns error if block_size is not a valid power of 2 in {2, 4, 8, 16, 32}
    pub fn new(format: MxFormat, block_size: usize) -> Result<Self> {
        if !VALID_BLOCK_SIZES.contains(&block_size) {
            return Err(TrustformersError::quantization_error(format!(
                "Invalid MX block size: {}. Must be one of: 2, 4, 8, 16, 32",
                block_size
            )));
        }
        Ok(Self { format, block_size })
    }
}

/// MX quantized data representation
#[derive(Debug, Clone)]
pub struct MxQuantized {
    /// Shared exponents: one 8-bit exponent per block
    shared_exponents: Vec<u8>,

    /// Packed mantissa data (bit-packed per-element representations)
    mantissa_data: Vec<u8>,

    /// Quantization configuration
    config: MxQuantConfig,

    /// Original tensor shape
    shape: Vec<usize>,

    /// Total number of elements
    num_elements: usize,
}

impl MxQuantized {
    /// Get the shared exponents
    pub fn shared_exponents(&self) -> &[u8] {
        &self.shared_exponents
    }

    /// Get the packed mantissa data
    pub fn mantissa_data(&self) -> &[u8] {
        &self.mantissa_data
    }

    /// Get the quantization config
    pub fn config(&self) -> &MxQuantConfig {
        &self.config
    }

    /// Get the original shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the number of elements
    pub fn num_elements(&self) -> usize {
        self.num_elements
    }

    /// Compute total size in bytes of the quantized representation
    pub fn size_bytes(&self) -> usize {
        self.shared_exponents.len() + self.mantissa_data.len()
    }
}

/// Compute compression ratio for a given MX format and block size
///
/// Returns the ratio of original size (32-bit float) to quantized size.
/// The quantized size includes the amortized shared exponent overhead.
pub fn compression_ratio(format: MxFormat, block_size: usize) -> f32 {
    let original_bits_per_element = 32.0f32;
    // Each element: element_bits, plus amortized shared exponent: 8 / block_size
    let quantized_bits_per_element = format.element_bits() as f32 + 8.0 / block_size as f32;
    original_bits_per_element / quantized_bits_per_element
}

/// Extract the biased exponent from an f32 value
///
/// Returns the IEEE 754 biased exponent (0-255) from the float bits.
fn extract_f32_exponent(value: f32) -> u8 {
    let bits = value.to_bits();
    ((bits >> 23) & 0xFF) as u8
}

/// Compute the floor-log2 of the absolute value, clamped for shared exponent use.
///
/// For zero values, returns 0.
/// The result is the unbiased exponent of the float representation.
fn floor_log2_abs(value: f32) -> i32 {
    let abs_val = value.abs();
    if abs_val == 0.0 || abs_val.is_subnormal() {
        return -127; // Minimum exponent sentinel for zero/subnormal
    }
    let biased_exp = extract_f32_exponent(abs_val);
    biased_exp as i32 - 127 // Unbias the IEEE 754 exponent
}

/// Find the shared exponent for a block of values.
///
/// The shared exponent is the maximum unbiased exponent among all elements
/// in the block, stored as a biased 8-bit value (bias = 127).
fn compute_shared_exponent(block: &[f32]) -> u8 {
    let mut max_exp = -127i32;
    for &val in block {
        let exp = floor_log2_abs(val);
        if exp > max_exp {
            max_exp = exp;
        }
    }
    // Store as biased exponent (bias = 127), clamped to [0, 254]
    // 255 is reserved for special values (NaN/Inf indication)
    let biased = max_exp + 127;
    if biased < 0 {
        0u8
    } else if biased > 254 {
        254u8
    } else {
        biased as u8
    }
}

/// Quantize a single float value to MX FP format relative to the shared exponent.
///
/// Returns the packed bits for the per-element representation:
/// - sign (1 bit) + element_exponent (N bits) + mantissa (M bits)
///
/// The value is scaled by 2^(-shared_exp_unbiased) before quantization.
fn quantize_element_fp(value: f32, shared_exp_unbiased: i32, format: MxFormat) -> u16 {
    let sign_bit: u16 = if value < 0.0 { 1 } else { 0 };
    let abs_val = value.abs();

    if abs_val == 0.0 {
        return 0;
    }

    let exp_bits = format.element_exponent_bits() as u32;
    let mant_bits = format.mantissa_bits() as u32;
    let elem_bias = format.element_exponent_bias();
    let max_elem_exp = ((1i32 << exp_bits) - 2) - elem_bias;

    // Scale value relative to shared exponent
    let scaled = abs_val * 2.0f32.powi(-shared_exp_unbiased);

    if scaled < f32::MIN_POSITIVE {
        // Value is too small to represent
        return 0;
    }

    // Extract the element-local exponent
    let raw_local_exp = floor_log2_abs(scaled);

    if raw_local_exp > max_elem_exp {
        // Saturate to max representable value
        let biased_exp = (max_elem_exp + elem_bias) as u16;
        let mantissa = (1u32 << mant_bits) - 1;
        return (sign_bit << (exp_bits + mant_bits) as u16)
            | (biased_exp << mant_bits as u16)
            | (mantissa as u16);
    }

    if raw_local_exp >= -elem_bias {
        // Normal number: biased_exp > 0
        let clamped_exp = raw_local_exp;
        let significand = scaled / 2.0f32.powi(clamped_exp);
        let frac = (significand - 1.0).max(0.0);
        let mantissa_max = (1u32 << mant_bits) - 1;
        // Round to nearest
        let mantissa_raw = (frac * (1u32 << mant_bits) as f32 + 0.5) as u32;
        let mantissa = mantissa_raw.min(mantissa_max);

        let biased_exp = (clamped_exp + elem_bias) as u16;
        (sign_bit << (exp_bits + mant_bits) as u16)
            | (biased_exp << mant_bits as u16)
            | (mantissa as u16)
    } else {
        // Subnormal: biased_exp = 0, no implicit leading 1
        // The subnormal value = mantissa * 2^(-elem_bias) / 2^mant_bits
        // So mantissa = scaled * 2^(elem_bias) * 2^mant_bits
        let mantissa_f = scaled * 2.0f32.powi(elem_bias) * (1u32 << mant_bits) as f32;
        let mantissa_max = (1u32 << mant_bits) - 1;
        let mantissa = (mantissa_f + 0.5).min(mantissa_max as f32) as u32;

        if mantissa == 0 {
            return 0;
        }

        // biased_exp = 0 for subnormal
        (sign_bit << (exp_bits + mant_bits) as u16) | (mantissa as u16)
    }
}

/// Quantize a single float value to MXINT8 format relative to the shared exponent.
///
/// Returns packed bits: sign (1 bit) + magnitude (7 bits).
fn quantize_element_int(value: f32, shared_exp_unbiased: i32) -> u8 {
    let sign_bit: u8 = if value < 0.0 { 1 } else { 0 };
    let abs_val = value.abs();

    if abs_val == 0.0 {
        return 0;
    }

    // Scale relative to shared exponent: the shared exponent defines the
    // range as [0, 127 * 2^shared_exp_unbiased].
    // We want: quantized = round(abs_val / 2^shared_exp_unbiased * (127 / max_element_value))
    // Simplification: quantized = round(abs_val * 2^(-shared_exp_unbiased) * 127)
    //
    // Actually for MXINT8, the scale is: value = mantissa * 2^(shared_exp - 127 - 7)
    // So: mantissa = round(value / 2^(shared_exp - 127 - 7))
    let scale = 2.0f32.powi(shared_exp_unbiased - 7);
    let quantized = if scale > 0.0 { (abs_val / scale + 0.5) as u32 } else { 0 };

    let clamped = quantized.min(127) as u8;
    (sign_bit << 7) | clamped
}

/// Dequantize a single MX FP element back to f32.
fn dequantize_element_fp(packed: u16, shared_exp_unbiased: i32, format: MxFormat) -> f32 {
    let exp_bits = format.element_exponent_bits() as u32;
    let mant_bits = format.mantissa_bits() as u32;
    let elem_bias = format.element_exponent_bias();

    let sign_bit = (packed >> (exp_bits + mant_bits)) & 1;
    let biased_exp = ((packed >> mant_bits) & ((1 << exp_bits) - 1)) as i32;
    let mantissa = packed & ((1 << mant_bits) - 1);

    if biased_exp == 0 && mantissa == 0 {
        return if sign_bit != 0 { -0.0 } else { 0.0 };
    }

    let local_exp_unbiased = biased_exp - elem_bias;
    let significand = if biased_exp == 0 {
        // Subnormal: no implicit 1
        mantissa as f32 / (1u32 << mant_bits) as f32
    } else {
        // Normal: implicit 1.mantissa
        1.0 + mantissa as f32 / (1u32 << mant_bits) as f32
    };

    let value = significand * 2.0f32.powi(local_exp_unbiased + shared_exp_unbiased);

    if sign_bit != 0 {
        -value
    } else {
        value
    }
}

/// Dequantize a single MXINT8 element back to f32.
fn dequantize_element_int(packed: u8, shared_exp_unbiased: i32) -> f32 {
    let sign_bit = (packed >> 7) & 1;
    let magnitude = packed & 0x7F;

    let scale = 2.0f32.powi(shared_exp_unbiased - 7);
    let value = magnitude as f32 * scale;

    if sign_bit != 0 {
        -value
    } else {
        value
    }
}

/// Pack a sequence of N-bit values into a byte vector.
///
/// Each value occupies exactly `bits_per_element` bits.
/// Values are packed MSB-first within each byte.
fn pack_bits(values: &[u16], bits_per_element: u8) -> Vec<u8> {
    if values.is_empty() {
        return Vec::new();
    }

    let total_bits = values.len() * bits_per_element as usize;
    let num_bytes = total_bits.div_ceil(8);
    let mut packed = vec![0u8; num_bytes];

    let mut bit_offset = 0usize;
    for &val in values {
        let bpe = bits_per_element as usize;
        for i in 0..bpe {
            let bit = (val >> (bpe - 1 - i)) & 1;
            if bit != 0 {
                let byte_idx = bit_offset / 8;
                let bit_idx = 7 - (bit_offset % 8);
                packed[byte_idx] |= 1 << bit_idx;
            }
            bit_offset += 1;
        }
    }

    packed
}

/// Unpack N-bit values from a byte vector.
///
/// Returns exactly `count` values, each with `bits_per_element` bits.
fn unpack_bits(packed: &[u8], bits_per_element: u8, count: usize) -> Vec<u16> {
    let mut values = Vec::with_capacity(count);
    let bpe = bits_per_element as usize;

    let mut bit_offset = 0usize;
    for _ in 0..count {
        let mut val: u16 = 0;
        for i in 0..bpe {
            let byte_idx = bit_offset / 8;
            let bit_idx = 7 - (bit_offset % 8);
            if byte_idx < packed.len() {
                let bit = (packed[byte_idx] >> bit_idx) & 1;
                val |= (bit as u16) << (bpe - 1 - i);
            }
            bit_offset += 1;
        }
        values.push(val);
    }

    values
}

/// Quantize data to MX format
///
/// # Arguments
/// - `data`: Input f32 data to quantize
/// - `config`: MX quantization configuration
///
/// # Returns
/// An `MxQuantized` struct containing the shared exponents and packed mantissa data.
///
/// # Errors
/// - If `data` is empty
/// - If `config.block_size` is invalid
/// - If data contains NaN values
pub fn quantize_mx(data: &[f32], config: &MxQuantConfig) -> Result<MxQuantized> {
    if data.is_empty() {
        return Err(TrustformersError::quantization_error(
            "MX quantize: input data is empty".to_string(),
        ));
    }

    // Validate no NaN values
    for (i, &val) in data.iter().enumerate() {
        if val.is_nan() {
            return Err(TrustformersError::quantization_error(format!(
                "MX quantize: NaN value at index {}",
                i
            )));
        }
    }

    let block_size = config.block_size;
    let num_elements = data.len();

    // Number of full and partial blocks
    let num_blocks = num_elements.div_ceil(block_size);

    let mut shared_exponents = Vec::with_capacity(num_blocks);
    let mut all_element_values: Vec<u16> = Vec::with_capacity(num_elements);

    for block_idx in 0..num_blocks {
        let start = block_idx * block_size;
        let end = (start + block_size).min(num_elements);
        let block = &data[start..end];

        // Compute shared exponent for this block
        let shared_exp = compute_shared_exponent(block);
        shared_exponents.push(shared_exp);

        let shared_exp_unbiased = shared_exp as i32 - 127;

        // Quantize each element in the block
        for &val in block {
            let quantized = if config.format.is_float_format() {
                quantize_element_fp(val, shared_exp_unbiased, config.format)
            } else {
                quantize_element_int(val, shared_exp_unbiased) as u16
            };
            all_element_values.push(quantized);
        }

        // Pad partial blocks with zeros
        let pad_count = (start + block_size).saturating_sub(end);
        all_element_values.extend(std::iter::repeat_n(0u16, pad_count));
    }

    // Pack the element values into a byte vector
    let bits_per_element = config.format.element_bits();
    let mantissa_data = pack_bits(&all_element_values, bits_per_element);

    // Infer shape as 1-D (caller can reshape if needed)
    let shape = vec![num_elements];

    Ok(MxQuantized {
        shared_exponents,
        mantissa_data,
        config: config.clone(),
        shape,
        num_elements,
    })
}

/// Quantize data with a known shape
///
/// Same as `quantize_mx` but preserves the original tensor shape metadata.
pub fn quantize_mx_with_shape(
    data: &[f32],
    config: &MxQuantConfig,
    shape: &[usize],
) -> Result<MxQuantized> {
    // Validate shape matches data length
    let shape_elements: usize = shape.iter().product();
    if shape_elements != data.len() {
        return Err(TrustformersError::quantization_error(format!(
            "MX quantize: shape {:?} implies {} elements but data has {} elements",
            shape,
            shape_elements,
            data.len()
        )));
    }

    let mut quantized = quantize_mx(data, config)?;
    quantized.shape = shape.to_vec();
    Ok(quantized)
}

/// Dequantize MX data back to f32
///
/// # Arguments
/// - `quantized`: The MX quantized data
///
/// # Returns
/// A vector of f32 values reconstructed from the quantized representation.
pub fn dequantize_mx(quantized: &MxQuantized) -> Vec<f32> {
    let block_size = quantized.config.block_size;
    let bits_per_element = quantized.config.format.element_bits();
    let num_elements = quantized.num_elements;
    let num_blocks = quantized.shared_exponents.len();

    // Total packed elements includes padding for partial blocks
    let total_packed = num_blocks * block_size;

    // Unpack all element values
    let all_values = unpack_bits(&quantized.mantissa_data, bits_per_element, total_packed);

    let mut result = Vec::with_capacity(num_elements);

    for block_idx in 0..num_blocks {
        let shared_exp = quantized.shared_exponents[block_idx];
        let shared_exp_unbiased = shared_exp as i32 - 127;

        let start = block_idx * block_size;
        let end = (start + block_size).min(num_elements);

        for &packed in &all_values[start..end] {
            let value = if quantized.config.format.is_float_format() {
                dequantize_element_fp(packed, shared_exp_unbiased, quantized.config.format)
            } else {
                dequantize_element_int(packed as u8, shared_exp_unbiased)
            };
            result.push(value);
        }
    }

    result
}

/// Compute quantization error statistics between original and dequantized data
pub struct MxErrorStats {
    /// Mean absolute error
    pub mae: f32,
    /// Root mean squared error
    pub rmse: f32,
    /// Maximum absolute error
    pub max_error: f32,
    /// Signal-to-noise ratio in dB
    pub snr_db: f32,
}

/// Compute error statistics for MX quantization
pub fn compute_mx_error(original: &[f32], dequantized: &[f32]) -> Result<MxErrorStats> {
    if original.len() != dequantized.len() {
        return Err(TrustformersError::quantization_error(format!(
            "MX error computation: length mismatch ({} vs {})",
            original.len(),
            dequantized.len()
        )));
    }

    if original.is_empty() {
        return Err(TrustformersError::quantization_error(
            "MX error computation: empty data".to_string(),
        ));
    }

    let n = original.len() as f32;
    let mut sum_abs_error = 0.0f64;
    let mut sum_sq_error = 0.0f64;
    let mut max_error = 0.0f64;
    let mut signal_power = 0.0f64;

    for (o, d) in original.iter().zip(dequantized.iter()) {
        let error = (*o as f64) - (*d as f64);
        let abs_error = error.abs();
        sum_abs_error += abs_error;
        sum_sq_error += error * error;
        if abs_error > max_error {
            max_error = abs_error;
        }
        signal_power += (*o as f64) * (*o as f64);
    }

    let mae = (sum_abs_error / n as f64) as f32;
    let rmse = ((sum_sq_error / n as f64).sqrt()) as f32;
    let max_error = max_error as f32;

    let snr_db = if sum_sq_error > 0.0 {
        (10.0 * (signal_power / sum_sq_error).log10()) as f32
    } else {
        f32::INFINITY
    };

    Ok(MxErrorStats {
        mae,
        rmse,
        max_error,
        snr_db,
    })
}

/// Simple LCG (linear congruential generator) for deterministic test data.
/// State is u64 to avoid overflow. Returns f32 in [-range, range].
#[cfg(test)]
struct LcgRng {
    state: u64,
}

#[cfg(test)]
impl LcgRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f32(&mut self, range: f32) -> f32 {
        // LCG: x_{n+1} = (a * x_n + c) mod m
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Map to [-range, range]
        let normalized = (self.state >> 33) as f32 / (u32::MAX >> 1) as f32;
        (normalized * 2.0 - 1.0) * range
    }

    fn next_f32_positive(&mut self, range: f32) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let normalized = (self.state >> 33) as f32 / (u32::MAX >> 1) as f32;
        normalized * range
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to generate deterministic test data
    fn generate_test_data(seed: u64, count: usize, range: f32) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..count).map(|_| rng.next_f32(range)).collect()
    }

    fn generate_positive_test_data(seed: u64, count: usize, range: f32) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..count).map(|_| rng.next_f32_positive(range)).collect()
    }

    // =====================================================================
    // Config validation tests
    // =====================================================================

    #[test]
    fn test_config_valid_block_sizes() {
        for &bs in &[2, 4, 8, 16, 32] {
            let config = MxQuantConfig::new(MxFormat::Mxfp8, bs);
            assert!(config.is_ok(), "Block size {} should be valid", bs);
        }
    }

    #[test]
    fn test_config_invalid_block_sizes() {
        for &bs in &[0, 1, 3, 5, 6, 7, 9, 15, 17, 31, 33, 64, 128] {
            let config = MxQuantConfig::new(MxFormat::Mxfp8, bs);
            assert!(config.is_err(), "Block size {} should be invalid", bs);
        }
    }

    // =====================================================================
    // Format property tests
    // =====================================================================

    #[test]
    fn test_format_element_bits() {
        assert_eq!(MxFormat::Mxfp8.element_bits(), 8);
        assert_eq!(MxFormat::Mxfp6.element_bits(), 6);
        assert_eq!(MxFormat::Mxfp4.element_bits(), 4);
        assert_eq!(MxFormat::Mxint8.element_bits(), 8);
    }

    #[test]
    fn test_format_mantissa_bits() {
        assert_eq!(MxFormat::Mxfp8.mantissa_bits(), 3);
        assert_eq!(MxFormat::Mxfp6.mantissa_bits(), 2);
        assert_eq!(MxFormat::Mxfp4.mantissa_bits(), 1);
        assert_eq!(MxFormat::Mxint8.mantissa_bits(), 7);
    }

    #[test]
    fn test_format_exponent_bits() {
        assert_eq!(MxFormat::Mxfp8.element_exponent_bits(), 4);
        assert_eq!(MxFormat::Mxfp6.element_exponent_bits(), 3);
        assert_eq!(MxFormat::Mxfp4.element_exponent_bits(), 2);
        assert_eq!(MxFormat::Mxint8.element_exponent_bits(), 0);
    }

    #[test]
    fn test_format_is_float() {
        assert!(MxFormat::Mxfp8.is_float_format());
        assert!(MxFormat::Mxfp6.is_float_format());
        assert!(MxFormat::Mxfp4.is_float_format());
        assert!(!MxFormat::Mxint8.is_float_format());
    }

    #[test]
    fn test_format_display() {
        assert_eq!(format!("{}", MxFormat::Mxfp8), "MXFP8");
        assert_eq!(format!("{}", MxFormat::Mxfp6), "MXFP6");
        assert_eq!(format!("{}", MxFormat::Mxfp4), "MXFP4");
        assert_eq!(format!("{}", MxFormat::Mxint8), "MXINT8");
    }

    #[test]
    fn test_format_max_element_value_positive() {
        for fmt in &[
            MxFormat::Mxfp8,
            MxFormat::Mxfp6,
            MxFormat::Mxfp4,
            MxFormat::Mxint8,
        ] {
            assert!(fmt.max_element_value() > 0.0, "{:?} max must be > 0", fmt);
        }
    }

    // =====================================================================
    // Compression ratio tests
    // =====================================================================

    #[test]
    fn test_compression_ratio_mxfp8_block32() {
        // MXFP8: 8 bits/elem + 8/32 = 8.25 bits/elem => ratio = 32/8.25 ~ 3.88
        let ratio = compression_ratio(MxFormat::Mxfp8, 32);
        assert!(ratio > 3.8 && ratio < 4.0, "MXFP8/32 ratio = {}", ratio);
    }

    #[test]
    fn test_compression_ratio_mxfp4_block32() {
        // MXFP4: 4 bits/elem + 8/32 = 4.25 bits/elem => ratio = 32/4.25 ~ 7.53
        let ratio = compression_ratio(MxFormat::Mxfp4, 32);
        assert!(ratio > 7.0 && ratio < 8.0, "MXFP4/32 ratio = {}", ratio);
    }

    #[test]
    fn test_compression_ratio_mxfp6_block16() {
        // MXFP6: 6 bits/elem + 8/16 = 6.5 bits/elem => ratio = 32/6.5 ~ 4.92
        let ratio = compression_ratio(MxFormat::Mxfp6, 16);
        assert!(ratio > 4.5 && ratio < 5.5, "MXFP6/16 ratio = {}", ratio);
    }

    #[test]
    fn test_compression_ratio_increases_with_block_size() {
        let fmt = MxFormat::Mxfp8;
        let r2 = compression_ratio(fmt, 2);
        let r4 = compression_ratio(fmt, 4);
        let r8 = compression_ratio(fmt, 8);
        let r16 = compression_ratio(fmt, 16);
        let r32 = compression_ratio(fmt, 32);
        assert!(r2 < r4, "ratio should increase: {} < {}", r2, r4);
        assert!(r4 < r8, "ratio should increase: {} < {}", r4, r8);
        assert!(r8 < r16, "ratio should increase: {} < {}", r8, r16);
        assert!(r16 < r32, "ratio should increase: {} < {}", r16, r32);
    }

    // =====================================================================
    // Bit packing tests
    // =====================================================================

    #[test]
    fn test_pack_unpack_8bit_roundtrip() {
        let values: Vec<u16> = vec![0xFF, 0x00, 0xAB, 0x55, 0x01, 0xFE];
        let packed = pack_bits(&values, 8);
        let unpacked = unpack_bits(&packed, 8, values.len());
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_pack_unpack_6bit_roundtrip() {
        let values: Vec<u16> = vec![0x3F, 0x00, 0x15, 0x2A, 0x01, 0x3E];
        let packed = pack_bits(&values, 6);
        let unpacked = unpack_bits(&packed, 6, values.len());
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_pack_unpack_4bit_roundtrip() {
        let values: Vec<u16> = vec![0x0F, 0x00, 0x05, 0x0A, 0x01, 0x0E, 0x07, 0x03];
        let packed = pack_bits(&values, 4);
        let unpacked = unpack_bits(&packed, 4, values.len());
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_pack_empty() {
        let packed = pack_bits(&[], 8);
        assert!(packed.is_empty());
        let unpacked = unpack_bits(&packed, 8, 0);
        assert!(unpacked.is_empty());
    }

    // =====================================================================
    // Quantize/Dequantize roundtrip tests
    // =====================================================================

    #[test]
    fn test_mxfp8_roundtrip_zeros() {
        let data = vec![0.0f32; 16];
        let config = MxQuantConfig::new(MxFormat::Mxfp8, 8).expect("valid config");
        let quantized = quantize_mx(&data, &config).expect("quantize");
        let dequantized = dequantize_mx(&quantized);
        assert_eq!(dequantized.len(), data.len());
        for val in &dequantized {
            assert!(val.abs() < 1e-10, "expected zero, got {}", val);
        }
    }

    #[test]
    fn test_mxfp8_roundtrip_small_block() {
        let data = vec![1.0, -1.0, 0.5, -0.5];
        let config = MxQuantConfig::new(MxFormat::Mxfp8, 2).expect("valid config");
        let quantized = quantize_mx(&data, &config).expect("quantize");
        let dequantized = dequantize_mx(&quantized);
        assert_eq!(dequantized.len(), 4);
        // MXFP8 has reasonable precision; check error is small
        for (o, d) in data.iter().zip(dequantized.iter()) {
            let error = (o - d).abs();
            assert!(error < 0.5, "error too large: orig={}, deq={}", o, d);
        }
    }

    #[test]
    fn test_mxfp8_roundtrip_random_data() {
        let data = generate_test_data(42, 128, 10.0);
        let config = MxQuantConfig::new(MxFormat::Mxfp8, 16).expect("valid config");
        let quantized = quantize_mx(&data, &config).expect("quantize");
        let dequantized = dequantize_mx(&quantized);
        assert_eq!(dequantized.len(), data.len());

        let stats = compute_mx_error(&data, &dequantized).expect("error stats");
        // MXFP8 should have reasonable accuracy
        assert!(stats.rmse < 5.0, "MXFP8 RMSE too high: {}", stats.rmse);
    }

    #[test]
    fn test_mxfp6_roundtrip_random_data() {
        let data = generate_test_data(123, 64, 5.0);
        let config = MxQuantConfig::new(MxFormat::Mxfp6, 8).expect("valid config");
        let quantized = quantize_mx(&data, &config).expect("quantize");
        let dequantized = dequantize_mx(&quantized);
        assert_eq!(dequantized.len(), data.len());
    }

    #[test]
    fn test_mxfp4_roundtrip_random_data() {
        let data = generate_test_data(456, 32, 2.0);
        let config = MxQuantConfig::new(MxFormat::Mxfp4, 4).expect("valid config");
        let quantized = quantize_mx(&data, &config).expect("quantize");
        let dequantized = dequantize_mx(&quantized);
        assert_eq!(dequantized.len(), data.len());
    }

    #[test]
    fn test_mxint8_roundtrip_random_data() {
        let data = generate_test_data(789, 64, 100.0);
        let config = MxQuantConfig::new(MxFormat::Mxint8, 16).expect("valid config");
        let quantized = quantize_mx(&data, &config).expect("quantize");
        let dequantized = dequantize_mx(&quantized);
        assert_eq!(dequantized.len(), data.len());
    }

    #[test]
    fn test_mxint8_roundtrip_zeros() {
        let data = vec![0.0f32; 32];
        let config = MxQuantConfig::new(MxFormat::Mxint8, 8).expect("valid config");
        let quantized = quantize_mx(&data, &config).expect("quantize");
        let dequantized = dequantize_mx(&quantized);
        for val in &dequantized {
            assert!(val.abs() < 1e-10, "expected zero, got {}", val);
        }
    }

    // =====================================================================
    // Edge case tests
    // =====================================================================

    #[test]
    fn test_quantize_empty_data() {
        let config = MxQuantConfig::new(MxFormat::Mxfp8, 8).expect("valid config");
        let result = quantize_mx(&[], &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantize_nan_data() {
        let data = vec![1.0, f32::NAN, 3.0];
        let config = MxQuantConfig::new(MxFormat::Mxfp8, 4).expect("valid config");
        let result = quantize_mx(&data, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantize_partial_block() {
        // 5 elements with block_size=4 => 1 full block + 1 partial block
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = MxQuantConfig::new(MxFormat::Mxfp8, 4).expect("valid config");
        let quantized = quantize_mx(&data, &config).expect("quantize");
        assert_eq!(quantized.num_elements(), 5);
        assert_eq!(quantized.shared_exponents().len(), 2); // 2 blocks

        let dequantized = dequantize_mx(&quantized);
        assert_eq!(dequantized.len(), 5);
    }

    #[test]
    fn test_quantize_single_element() {
        let data = vec![42.0];
        let config = MxQuantConfig::new(MxFormat::Mxfp8, 2).expect("valid config");
        let quantized = quantize_mx(&data, &config).expect("quantize");
        assert_eq!(quantized.num_elements(), 1);
        let dequantized = dequantize_mx(&quantized);
        assert_eq!(dequantized.len(), 1);
    }

    #[test]
    fn test_quantize_exactly_one_block() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let config = MxQuantConfig::new(MxFormat::Mxfp8, 4).expect("valid config");
        let quantized = quantize_mx(&data, &config).expect("quantize");
        assert_eq!(quantized.shared_exponents().len(), 1);
    }

    #[test]
    fn test_quantize_large_values() {
        let data = vec![1e6, -1e6, 5e5, -5e5, 1e3, -1e3, 1.0, -1.0];
        let config = MxQuantConfig::new(MxFormat::Mxfp8, 8).expect("valid config");
        let quantized = quantize_mx(&data, &config).expect("quantize");
        let dequantized = dequantize_mx(&quantized);
        assert_eq!(dequantized.len(), 8);
        // For large-magnitude values, sign should be preserved.
        // Small values relative to the block max may be quantized to zero,
        // so we only check sign for values that remain non-zero after dequantization.
        for i in 0..data.len() {
            if data[i] != 0.0 && dequantized[i] != 0.0 {
                assert_eq!(
                    data[i].is_sign_positive(),
                    dequantized[i].is_sign_positive(),
                    "Sign mismatch at index {}: orig={}, deq={}",
                    i,
                    data[i],
                    dequantized[i]
                );
            }
        }
    }

    #[test]
    fn test_quantize_very_small_values() {
        let data = vec![1e-20, -1e-20, 1e-30, -1e-30];
        let config = MxQuantConfig::new(MxFormat::Mxfp8, 4).expect("valid config");
        let quantized = quantize_mx(&data, &config).expect("quantize");
        let dequantized = dequantize_mx(&quantized);
        assert_eq!(dequantized.len(), 4);
    }

    #[test]
    fn test_quantize_mixed_magnitudes() {
        // Elements with very different magnitudes in the same block
        let data = vec![1000.0, 0.001, -500.0, 0.0005];
        let config = MxQuantConfig::new(MxFormat::Mxfp8, 4).expect("valid config");
        let quantized = quantize_mx(&data, &config).expect("quantize");
        let dequantized = dequantize_mx(&quantized);
        assert_eq!(dequantized.len(), 4);
        // Large values should be somewhat preserved
        assert!((dequantized[0] - 1000.0).abs() < 200.0);
    }

    // =====================================================================
    // Shape handling tests
    // =====================================================================

    #[test]
    fn test_quantize_with_shape() {
        let data = generate_test_data(100, 24, 1.0);
        let config = MxQuantConfig::new(MxFormat::Mxfp8, 8).expect("valid config");
        let quantized =
            quantize_mx_with_shape(&data, &config, &[2, 3, 4]).expect("quantize with shape");
        assert_eq!(quantized.shape(), &[2, 3, 4]);
        assert_eq!(quantized.num_elements(), 24);
    }

    #[test]
    fn test_quantize_with_shape_mismatch() {
        let data = vec![1.0; 24];
        let config = MxQuantConfig::new(MxFormat::Mxfp8, 8).expect("valid config");
        let result = quantize_mx_with_shape(&data, &config, &[2, 3, 5]);
        assert!(result.is_err());
    }

    // =====================================================================
    // Error statistics tests
    // =====================================================================

    #[test]
    fn test_error_stats_identical() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let stats = compute_mx_error(&data, &data).expect("error stats");
        assert!(stats.mae < 1e-10);
        assert!(stats.rmse < 1e-10);
        assert!(stats.max_error < 1e-10);
        assert!(stats.snr_db.is_infinite());
    }

    #[test]
    fn test_error_stats_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        let result = compute_mx_error(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_stats_empty() {
        let result = compute_mx_error(&[], &[]);
        assert!(result.is_err());
    }

    // =====================================================================
    // MxQuantized accessors tests
    // =====================================================================

    #[test]
    fn test_quantized_accessors() {
        let data = generate_test_data(999, 32, 5.0);
        let config = MxQuantConfig::new(MxFormat::Mxfp8, 8).expect("valid config");
        let quantized = quantize_mx(&data, &config).expect("quantize");

        assert_eq!(quantized.num_elements(), 32);
        assert_eq!(quantized.shape(), &[32]);
        assert_eq!(quantized.shared_exponents().len(), 4); // 32 / 8 = 4 blocks
        assert!(!quantized.mantissa_data().is_empty());
        assert!(quantized.size_bytes() > 0);
    }

    #[test]
    fn test_quantized_size_bytes() {
        let data = vec![1.0f32; 32];
        let config_fp8 = MxQuantConfig::new(MxFormat::Mxfp8, 8).expect("valid config");
        let q_fp8 = quantize_mx(&data, &config_fp8).expect("quantize");

        let config_fp4 = MxQuantConfig::new(MxFormat::Mxfp4, 8).expect("valid config");
        let q_fp4 = quantize_mx(&data, &config_fp4).expect("quantize");

        // FP4 should use less space than FP8 for mantissa data
        assert!(
            q_fp4.mantissa_data().len() < q_fp8.mantissa_data().len(),
            "FP4 mantissa ({}) should be smaller than FP8 ({})",
            q_fp4.mantissa_data().len(),
            q_fp8.mantissa_data().len()
        );
    }

    // =====================================================================
    // Quality ordering: MXFP8 > MXFP6 > MXFP4
    // =====================================================================

    #[test]
    fn test_quality_ordering_fp4_worst() {
        // Use data within a single block's dynamic range to isolate
        // the effect of mantissa precision differences.
        // All values in a similar magnitude range so the shared exponent
        // doesn't dominate the error.
        let mut data = Vec::with_capacity(128);
        let mut rng = LcgRng::new(555);
        for _ in 0..128 {
            // Values in [0.5, 2.0] -- all same order of magnitude
            data.push(0.5 + rng.next_f32_positive(1.5));
        }

        let config_fp8 = MxQuantConfig::new(MxFormat::Mxfp8, 32).expect("valid config");
        let config_fp4 = MxQuantConfig::new(MxFormat::Mxfp4, 32).expect("valid config");

        let d_fp8 = dequantize_mx(&quantize_mx(&data, &config_fp8).expect("q"));
        let d_fp4 = dequantize_mx(&quantize_mx(&data, &config_fp4).expect("q"));

        let e_fp8 = compute_mx_error(&data, &d_fp8).expect("err");
        let e_fp4 = compute_mx_error(&data, &d_fp4).expect("err");

        // FP8 (3 mantissa bits) should have lower or equal error to FP4 (1 mantissa bit)
        assert!(
            e_fp8.rmse <= e_fp4.rmse,
            "FP8 RMSE ({}) should be <= FP4 ({})",
            e_fp8.rmse,
            e_fp4.rmse
        );
    }

    // =====================================================================
    // Shared exponent computation tests
    // =====================================================================

    #[test]
    fn test_shared_exponent_all_zeros() {
        let exp = compute_shared_exponent(&[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(exp, 0); // -127 + 127 = 0
    }

    #[test]
    fn test_shared_exponent_powers_of_two() {
        // 1.0 has exponent 0, 2.0 has exponent 1, 4.0 has exponent 2
        let exp = compute_shared_exponent(&[1.0, 2.0, 4.0]);
        let unbiased = exp as i32 - 127;
        assert_eq!(unbiased, 2, "shared exp should be 2 for max=4.0");
    }

    #[test]
    fn test_shared_exponent_negative_values() {
        let exp = compute_shared_exponent(&[-8.0, 1.0, -0.5]);
        let unbiased = exp as i32 - 127;
        assert_eq!(unbiased, 3, "shared exp should be 3 for |max|=8.0");
    }

    // =====================================================================
    // All formats x all block sizes combinatorial test
    // =====================================================================

    #[test]
    fn test_all_format_block_size_combinations() {
        let formats = [
            MxFormat::Mxfp8,
            MxFormat::Mxfp6,
            MxFormat::Mxfp4,
            MxFormat::Mxint8,
        ];
        let block_sizes = [2, 4, 8, 16, 32];
        let data = generate_test_data(777, 64, 10.0);

        for fmt in &formats {
            for &bs in &block_sizes {
                let config = MxQuantConfig::new(*fmt, bs).expect("valid config");
                let quantized = quantize_mx(&data, &config);
                assert!(
                    quantized.is_ok(),
                    "Failed for format {:?} block_size {}",
                    fmt,
                    bs
                );
                let deq = dequantize_mx(&quantized.expect("already checked"));
                assert_eq!(deq.len(), data.len());
            }
        }
    }

    // =====================================================================
    // Positive data test (exercises generate_positive_test_data)
    // =====================================================================

    #[test]
    fn test_mxfp8_positive_only_data() {
        let data = generate_positive_test_data(333, 64, 20.0);
        let config = MxQuantConfig::new(MxFormat::Mxfp8, 8).expect("valid config");
        let quantized = quantize_mx(&data, &config).expect("quantize");
        let dequantized = dequantize_mx(&quantized);
        assert_eq!(dequantized.len(), data.len());
        // All dequantized values should be non-negative
        for (i, val) in dequantized.iter().enumerate() {
            assert!(
                *val >= 0.0,
                "Expected non-negative at index {}, got {}",
                i,
                val
            );
        }
    }

    // =====================================================================
    // Inf handling test
    // =====================================================================

    #[test]
    fn test_quantize_inf_values() {
        let data = vec![f32::INFINITY, f32::NEG_INFINITY, 1.0, -1.0];
        let config = MxQuantConfig::new(MxFormat::Mxfp8, 4).expect("valid config");
        // Inf should not crash; the shared exponent will be clamped
        let quantized = quantize_mx(&data, &config).expect("quantize");
        let dequantized = dequantize_mx(&quantized);
        assert_eq!(dequantized.len(), 4);
    }

    // =====================================================================
    // Determinism test
    // =====================================================================

    #[test]
    fn test_quantization_deterministic() {
        let data = generate_test_data(12345, 64, 8.0);
        let config = MxQuantConfig::new(MxFormat::Mxfp8, 8).expect("valid config");

        let q1 = quantize_mx(&data, &config).expect("q1");
        let q2 = quantize_mx(&data, &config).expect("q2");

        assert_eq!(q1.shared_exponents(), q2.shared_exponents());
        assert_eq!(q1.mantissa_data(), q2.mantissa_data());
    }
}
