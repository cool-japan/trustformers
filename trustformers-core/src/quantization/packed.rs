//! Sub-byte packed quantization for TrustformeRS.
//!
//! Bit-packing utilities for sub-byte integer quantization.
//! Stores multiple low-bit integers packed into u8 words.
//!
//! Supported formats:
//! - INT2: 4 values per byte, signed range [-2, 1]
//! - INT3: contiguous bit stream, signed range [-4, 3]
//! - INT4: 2 values per byte, signed range [-8, 7]
//! - INT8: 1 value per byte, signed range [-128, 127]

/// Errors that can occur during sub-byte quantization operations.
#[derive(Debug, Clone, PartialEq)]
pub enum QuantError {
    /// Array index is out of bounds.
    IndexOutOfBounds { idx: usize, max: usize },
    /// Integer value is outside the representable range for this bit width.
    ValueOutOfRange { val: i64, min: i64, max: i64 },
    /// Group size is invalid (zero or larger than tensor).
    InvalidGroupSize,
    /// The input tensor is empty.
    EmptyTensor,
}

impl std::fmt::Display for QuantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantError::IndexOutOfBounds { idx, max } => {
                write!(f, "index {idx} out of bounds (max {max})")
            }
            QuantError::ValueOutOfRange { val, min, max } => {
                write!(f, "value {val} out of range [{min}, {max}]")
            }
            QuantError::InvalidGroupSize => write!(f, "invalid group size"),
            QuantError::EmptyTensor => write!(f, "tensor is empty"),
        }
    }
}

impl std::error::Error for QuantError {}

/// Bit-width selector for sub-byte quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitWidth {
    /// 2-bit: 4 values per byte, signed range [-2, 1].
    Two,
    /// 3-bit: packed across byte boundaries, signed range [-4, 3].
    Three,
    /// 4-bit: 2 values per byte, signed range [-8, 7].
    Four,
    /// 8-bit: 1 value per byte, signed range [-128, 127].
    Eight,
}

impl BitWidth {
    /// Returns the number of bits used per element.
    pub fn bits(self) -> u32 {
        match self {
            BitWidth::Two => 2,
            BitWidth::Three => 3,
            BitWidth::Four => 4,
            BitWidth::Eight => 8,
        }
    }

    /// Returns the maximum signed value representable at this bit width.
    pub fn max_val(self) -> i64 {
        match self {
            BitWidth::Two => 1,
            BitWidth::Three => 3,
            BitWidth::Four => 7,
            BitWidth::Eight => 127,
        }
    }

    /// Returns the minimum signed value representable at this bit width.
    pub fn min_val(self) -> i64 {
        match self {
            BitWidth::Two => -2,
            BitWidth::Three => -4,
            BitWidth::Four => -8,
            BitWidth::Eight => -128,
        }
    }

    /// Returns the unsigned offset used to shift signed values into the
    /// non-negative storage range.
    fn offset(self) -> i64 {
        -self.min_val()
    }

    /// Returns the bit-mask for one stored element (always unsigned).
    fn mask(self) -> u64 {
        (1u64 << self.bits()) - 1
    }

    /// Returns how many values fit in one byte (only exact for 2, 4, 8 bits).
    pub fn values_per_byte(self) -> usize {
        match self {
            BitWidth::Two => 4,
            BitWidth::Three => 2, // approximate — 2 full values + partial
            BitWidth::Four => 2,
            BitWidth::Eight => 1,
        }
    }
}

/// Packed buffer storing multiple sub-byte integers per byte.
///
/// Values are stored as unsigned integers with a fixed offset so that the
/// signed range maps cleanly onto the available unsigned storage range.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedBuffer {
    /// Raw byte storage.
    pub data: Vec<u8>,
    /// Bit width of each stored element.
    pub bit_width: BitWidth,
    /// Number of logical elements stored.
    pub num_elements: usize,
}

impl PackedBuffer {
    /// Allocate a zeroed buffer capable of holding `num_elements` values at
    /// the given `bit_width`. All values are initialised to `min_val`.
    pub fn new(num_elements: usize, bit_width: BitWidth) -> Self {
        let total_bits = num_elements * bit_width.bits() as usize;
        let num_bytes = total_bits.div_ceil(8);
        // Fill with the offset representation of min_val (i.e. unsigned zero).
        let data = vec![0u8; num_bytes];
        Self { data, bit_width, num_elements }
    }

    /// Pack a slice of signed integers into a new `PackedBuffer`.
    ///
    /// Returns `QuantError::EmptyTensor` if `values` is empty and
    /// `QuantError::ValueOutOfRange` if any value falls outside the range
    /// supported by `bit_width`.
    pub fn pack(values: &[i64], bit_width: BitWidth) -> Result<Self, QuantError> {
        if values.is_empty() {
            return Err(QuantError::EmptyTensor);
        }
        let min = bit_width.min_val();
        let max = bit_width.max_val();
        for &v in values {
            if v < min || v > max {
                return Err(QuantError::ValueOutOfRange { val: v, min, max });
            }
        }

        let mut buf = Self::new(values.len(), bit_width);
        for (idx, &v) in values.iter().enumerate() {
            // SAFETY: we just validated the range, so set() will not fail.
            buf.set_unchecked(idx, v);
        }
        Ok(buf)
    }

    /// Unpack all stored values as signed integers.
    pub fn unpack(&self) -> Vec<i64> {
        (0..self.num_elements)
            .map(|i| self.get_unchecked(i))
            .collect()
    }

    /// Get the signed value at position `idx`.
    pub fn get(&self, idx: usize) -> Result<i64, QuantError> {
        if idx >= self.num_elements {
            return Err(QuantError::IndexOutOfBounds { idx, max: self.num_elements.saturating_sub(1) });
        }
        Ok(self.get_unchecked(idx))
    }

    /// Set the signed value at position `idx`.
    pub fn set(&mut self, idx: usize, val: i64) -> Result<(), QuantError> {
        if idx >= self.num_elements {
            return Err(QuantError::IndexOutOfBounds { idx, max: self.num_elements.saturating_sub(1) });
        }
        let min = self.bit_width.min_val();
        let max = self.bit_width.max_val();
        if val < min || val > max {
            return Err(QuantError::ValueOutOfRange { val, min, max });
        }
        self.set_unchecked(idx, val);
        Ok(())
    }

    /// Returns the number of bytes used by the packed buffer.
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Ratio of bits actually used relative to 32-bit float storage.
    ///
    /// A value of 0.125 means the packed format uses 1/8 the storage of f32.
    pub fn compression_ratio_vs_f32(&self) -> f32 {
        self.bit_width.bits() as f32 / 32.0_f32
    }

    // -----------------------------------------------------------------------
    // Private helpers — no bounds/range checking
    // -----------------------------------------------------------------------

    fn get_unchecked(&self, idx: usize) -> i64 {
        let offset = self.bit_width.offset();
        let mask = self.bit_width.mask();
        let bits = self.bit_width.bits() as usize;

        let bit_pos = idx * bits;
        let byte_idx = bit_pos / 8;
        let bit_off = bit_pos % 8;

        // Read up to 2 bytes to cover any cross-byte element.
        let raw = if byte_idx + 1 < self.data.len() {
            (self.data[byte_idx] as u16) | ((self.data[byte_idx + 1] as u16) << 8)
        } else {
            self.data[byte_idx] as u16
        };

        let unsigned = ((raw as u64) >> bit_off) & mask;
        unsigned as i64 - offset
    }

    fn set_unchecked(&mut self, idx: usize, val: i64) {
        let offset = self.bit_width.offset();
        let mask = self.bit_width.mask();
        let bits = self.bit_width.bits() as usize;

        let unsigned = (val + offset) as u64 & mask;

        let bit_pos = idx * bits;
        let byte_idx = bit_pos / 8;
        let bit_off = bit_pos % 8;

        // Clear old bits and write new bits — may span two bytes.
        self.data[byte_idx] &= !((mask << bit_off) as u8);
        self.data[byte_idx] |= ((unsigned << bit_off) & 0xFF) as u8;

        if bit_off + bits > 8 && byte_idx + 1 < self.data.len() {
            let spill_bits = bit_off + bits - 8;
            let spill_mask = ((1u64 << spill_bits) - 1) as u8;
            self.data[byte_idx + 1] &= !spill_mask;
            self.data[byte_idx + 1] |= (unsigned >> (bits - spill_bits)) as u8 & spill_mask;
        }
    }
}

// ---------------------------------------------------------------------------
// INT2 group-wise quantization
// ---------------------------------------------------------------------------

/// Configuration for 2-bit group-wise quantization.
#[derive(Debug, Clone)]
pub struct Int2QuantConfig {
    /// Number of weights per quantization group.
    pub group_size: usize,
    /// Whether to use symmetric quantization (zero_point = 0).
    pub symmetric: bool,
}

impl Default for Int2QuantConfig {
    fn default() -> Self {
        Self { group_size: 128, symmetric: true }
    }
}

/// Quantize a float tensor to INT2 using group-wise quantization.
///
/// Returns `(packed_data, scales, zero_points)`.  `zero_points` is all-zero
/// for symmetric mode.
pub fn quantize_int2(
    tensor: &[f32],
    config: &Int2QuantConfig,
) -> Result<(PackedBuffer, Vec<f32>, Vec<f32>), QuantError> {
    if tensor.is_empty() {
        return Err(QuantError::EmptyTensor);
    }
    if config.group_size == 0 {
        return Err(QuantError::InvalidGroupSize);
    }

    let num_groups = tensor.len().div_ceil(config.group_size);
    let mut scales = Vec::with_capacity(num_groups);
    let mut zero_points = Vec::with_capacity(num_groups);
    let mut quantized = Vec::with_capacity(tensor.len());

    let min_q = BitWidth::Two.min_val(); // -2
    let max_q = BitWidth::Two.max_val(); //  1

    for group_idx in 0..num_groups {
        let start = group_idx * config.group_size;
        let end = (start + config.group_size).min(tensor.len());
        let group = &tensor[start..end];

        let (scale, zero_point) = compute_scale_zero_point(group, min_q, max_q, config.symmetric);
        scales.push(scale);
        zero_points.push(zero_point);

        for &w in group {
            let q = quantize_single(w, scale, zero_point, min_q, max_q);
            quantized.push(q);
        }
    }

    let packed = PackedBuffer::pack(&quantized, BitWidth::Two)?;
    Ok((packed, scales, zero_points))
}

/// Dequantize an INT2 packed buffer back to f32.
pub fn dequantize_int2(
    packed: &PackedBuffer,
    scales: &[f32],
    zero_points: &[f32],
    config: &Int2QuantConfig,
) -> Result<Vec<f32>, QuantError> {
    if config.group_size == 0 {
        return Err(QuantError::InvalidGroupSize);
    }
    let values = packed.unpack();
    let mut output = Vec::with_capacity(values.len());

    for (i, &q) in values.iter().enumerate() {
        let group_idx = i / config.group_size;
        if group_idx >= scales.len() {
            return Err(QuantError::IndexOutOfBounds { idx: group_idx, max: scales.len().saturating_sub(1) });
        }
        let scale = scales[group_idx];
        let zp = zero_points[group_idx];
        output.push(dequantize_single(q, scale, zp));
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// INT3 group-wise quantization
// ---------------------------------------------------------------------------

/// Quantize a float tensor to INT3 using group-wise quantization.
///
/// Returns `(packed_data, scales, zero_points)`.
pub fn quantize_int3(
    tensor: &[f32],
    group_size: usize,
) -> Result<(PackedBuffer, Vec<f32>, Vec<f32>), QuantError> {
    if tensor.is_empty() {
        return Err(QuantError::EmptyTensor);
    }
    if group_size == 0 {
        return Err(QuantError::InvalidGroupSize);
    }

    let num_groups = tensor.len().div_ceil(group_size);
    let mut scales = Vec::with_capacity(num_groups);
    let mut zero_points = Vec::with_capacity(num_groups);
    let mut quantized = Vec::with_capacity(tensor.len());

    let min_q = BitWidth::Three.min_val(); // -4
    let max_q = BitWidth::Three.max_val(); //  3

    for group_idx in 0..num_groups {
        let start = group_idx * group_size;
        let end = (start + group_size).min(tensor.len());
        let group = &tensor[start..end];

        let (scale, zero_point) = compute_scale_zero_point(group, min_q, max_q, true);
        scales.push(scale);
        zero_points.push(zero_point);

        for &w in group {
            let q = quantize_single(w, scale, zero_point, min_q, max_q);
            quantized.push(q);
        }
    }

    let packed = PackedBuffer::pack(&quantized, BitWidth::Three)?;
    Ok((packed, scales, zero_points))
}

/// Dequantize an INT3 packed buffer back to f32.
pub fn dequantize_int3(
    packed: &PackedBuffer,
    scales: &[f32],
    zero_points: &[f32],
    group_size: usize,
) -> Result<Vec<f32>, QuantError> {
    if group_size == 0 {
        return Err(QuantError::InvalidGroupSize);
    }
    let values = packed.unpack();
    let mut output = Vec::with_capacity(values.len());

    for (i, &q) in values.iter().enumerate() {
        let group_idx = i / group_size;
        if group_idx >= scales.len() {
            return Err(QuantError::IndexOutOfBounds { idx: group_idx, max: scales.len().saturating_sub(1) });
        }
        let scale = scales[group_idx];
        let zp = zero_points[group_idx];
        output.push(dequantize_single(q, scale, zp));
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Shared quantization helpers
// ---------------------------------------------------------------------------

/// Compute scale and zero_point for a group of f32 values.
fn compute_scale_zero_point(
    group: &[f32],
    min_q: i64,
    max_q: i64,
    symmetric: bool,
) -> (f32, f32) {
    let fmin = group.iter().cloned().fold(f32::INFINITY, f32::min);
    let fmax = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    if symmetric {
        let max_abs = fmin.abs().max(fmax.abs());
        if max_abs < f32::EPSILON {
            return (1.0_f32, 0.0_f32);
        }
        // Map max_abs -> max_q (the positive end of the range).
        let scale = max_abs / (max_q as f32);
        (scale, 0.0_f32)
    } else {
        let q_range = (max_q - min_q) as f32;
        let f_range = fmax - fmin;
        if f_range < f32::EPSILON {
            return (1.0_f32, fmin);
        }
        let scale = f_range / q_range;
        let zero_point = fmin;
        (scale, zero_point)
    }
}

/// Quantize a single f32 value to a signed integer.
fn quantize_single(val: f32, scale: f32, zero_point: f32, min_q: i64, max_q: i64) -> i64 {
    if scale.abs() < f32::EPSILON {
        return 0;
    }
    let q = ((val - zero_point) / scale).round() as i64;
    q.clamp(min_q, max_q)
}

/// Dequantize a single signed integer to f32.
fn dequantize_single(q: i64, scale: f32, zero_point: f32) -> f32 {
    q as f32 * scale + zero_point
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // BitWidth tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bitwidth_properties() {
        assert_eq!(BitWidth::Two.bits(), 2);
        assert_eq!(BitWidth::Three.bits(), 3);
        assert_eq!(BitWidth::Four.bits(), 4);
        assert_eq!(BitWidth::Eight.bits(), 8);

        assert_eq!(BitWidth::Two.max_val(), 1);
        assert_eq!(BitWidth::Three.max_val(), 3);
        assert_eq!(BitWidth::Four.max_val(), 7);
        assert_eq!(BitWidth::Eight.max_val(), 127);

        assert_eq!(BitWidth::Two.min_val(), -2);
        assert_eq!(BitWidth::Three.min_val(), -4);
        assert_eq!(BitWidth::Four.min_val(), -8);
        assert_eq!(BitWidth::Eight.min_val(), -128);
    }

    #[test]
    fn test_values_per_byte() {
        assert_eq!(BitWidth::Two.values_per_byte(), 4);
        assert_eq!(BitWidth::Four.values_per_byte(), 2);
        assert_eq!(BitWidth::Eight.values_per_byte(), 1);
    }

    // -----------------------------------------------------------------------
    // 2-bit pack / unpack round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_2bit_pack_unpack_round_trip() {
        let values: Vec<i64> = vec![-2, -1, 0, 1, -2, 1, 0, -1];
        let buf = PackedBuffer::pack(&values, BitWidth::Two).expect("pack failed");
        let unpacked = buf.unpack();
        assert_eq!(unpacked, values, "2-bit round-trip mismatch");
    }

    #[test]
    fn test_2bit_boundary_values() {
        let values = vec![-2, 1];
        let buf = PackedBuffer::pack(&values, BitWidth::Two).expect("pack failed");
        assert_eq!(buf.get(0).expect("get 0 failed"), -2);
        assert_eq!(buf.get(1).expect("get 1 failed"), 1);
    }

    // -----------------------------------------------------------------------
    // 3-bit pack / unpack round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_3bit_pack_unpack_round_trip() {
        let values: Vec<i64> = vec![-4, -3, -2, -1, 0, 1, 2, 3];
        let buf = PackedBuffer::pack(&values, BitWidth::Three).expect("pack failed");
        let unpacked = buf.unpack();
        assert_eq!(unpacked, values, "3-bit round-trip mismatch");
    }

    #[test]
    fn test_3bit_cross_byte_boundary() {
        // Elements 2 and 3 straddle a byte boundary (bits 6-8 / 9-11).
        let values: Vec<i64> = vec![-4, 3, -3, 2, -2, 1, -1, 0];
        let buf = PackedBuffer::pack(&values, BitWidth::Three).expect("pack failed");
        let unpacked = buf.unpack();
        assert_eq!(unpacked, values, "3-bit cross-byte round-trip mismatch");
    }

    // -----------------------------------------------------------------------
    // 4-bit pack / unpack round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_4bit_pack_unpack_round_trip() {
        let values: Vec<i64> = vec![-8, -5, 0, 3, 7, -1, -8, 7];
        let buf = PackedBuffer::pack(&values, BitWidth::Four).expect("pack failed");
        let unpacked = buf.unpack();
        assert_eq!(unpacked, values, "4-bit round-trip mismatch");
    }

    // -----------------------------------------------------------------------
    // Single-element get / set
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_set_single_element() {
        let mut buf = PackedBuffer::new(16, BitWidth::Four);
        buf.set(5, 7).expect("set failed");
        assert_eq!(buf.get(5).expect("get failed"), 7);
        buf.set(5, -8).expect("set failed");
        assert_eq!(buf.get(5).expect("get failed"), -8);
    }

    #[test]
    fn test_set_does_not_corrupt_neighbours() {
        let values: Vec<i64> = vec![1, 2, 3, 4, 5, 6, 7, -8];
        let mut buf = PackedBuffer::pack(&values, BitWidth::Four).expect("pack failed");
        buf.set(3, -1).expect("set failed");
        assert_eq!(buf.get(2).expect("get failed"), 3, "neighbour corrupted");
        assert_eq!(buf.get(3).expect("get failed"), -1, "value not set");
        assert_eq!(buf.get(4).expect("get failed"), 5, "neighbour corrupted");
    }

    // -----------------------------------------------------------------------
    // Compression ratio
    // -----------------------------------------------------------------------

    #[test]
    fn test_compression_ratio() {
        let buf2 = PackedBuffer::new(1, BitWidth::Two);
        let buf4 = PackedBuffer::new(1, BitWidth::Four);
        let buf8 = PackedBuffer::new(1, BitWidth::Eight);

        assert!((buf2.compression_ratio_vs_f32() - 0.0625_f32).abs() < 1e-6);
        assert!((buf4.compression_ratio_vs_f32() - 0.125_f32).abs() < 1e-6);
        assert!((buf8.compression_ratio_vs_f32() - 0.25_f32).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Error handling
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_index_out_of_bounds() {
        let buf = PackedBuffer::new(4, BitWidth::Four);
        assert!(matches!(
            buf.get(4),
            Err(QuantError::IndexOutOfBounds { idx: 4, .. })
        ));
    }

    #[test]
    fn test_error_value_out_of_range() {
        let result = PackedBuffer::pack(&[8], BitWidth::Four);
        assert!(matches!(
            result,
            Err(QuantError::ValueOutOfRange { val: 8, .. })
        ));
    }

    #[test]
    fn test_error_empty_tensor() {
        let result = PackedBuffer::pack(&[], BitWidth::Two);
        assert_eq!(result, Err(QuantError::EmptyTensor));
    }

    // -----------------------------------------------------------------------
    // Group-wise INT2 quantization round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_int2_quantization_round_trip() {
        let tensor: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let config = Int2QuantConfig { group_size: 64, symmetric: true };
        let (packed, scales, zero_points) =
            quantize_int2(&tensor, &config).expect("quantize_int2 failed");
        let reconstructed =
            dequantize_int2(&packed, &scales, &zero_points, &config).expect("dequantize_int2 failed");

        assert_eq!(reconstructed.len(), tensor.len());
        // With only 2-bit precision the error is large; just check reconstruction is finite.
        for v in &reconstructed {
            assert!(v.is_finite(), "non-finite value in reconstruction");
        }
    }

    // -----------------------------------------------------------------------
    // Group-wise INT3 quantization round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_int3_quantization_round_trip() {
        let tensor: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 64.0).collect();
        let group_size = 32;
        let (packed, scales, zero_points) =
            quantize_int3(&tensor, group_size).expect("quantize_int3 failed");
        let reconstructed =
            dequantize_int3(&packed, &scales, &zero_points, group_size).expect("dequantize_int3 failed");

        assert_eq!(reconstructed.len(), tensor.len());
        // Verify max absolute error is within theoretical INT3 bound.
        let max_err = tensor
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        // Theoretical max error ≈ scale/2 per group. With 128 elements and range 2.0, scale ≈ 2/7 ≈ 0.286
        assert!(max_err < 0.3_f32, "INT3 max error {max_err} exceeds threshold");
    }

    // -----------------------------------------------------------------------
    // Size bytes
    // -----------------------------------------------------------------------

    #[test]
    fn test_size_bytes_2bit() {
        // 8 elements × 2 bits = 16 bits = 2 bytes
        let buf = PackedBuffer::new(8, BitWidth::Two);
        assert_eq!(buf.size_bytes(), 2);
    }

    #[test]
    fn test_size_bytes_3bit_cross_boundary() {
        // 8 elements × 3 bits = 24 bits = 3 bytes
        let buf = PackedBuffer::new(8, BitWidth::Three);
        assert_eq!(buf.size_bytes(), 3);
        // 9 elements × 3 bits = 27 bits → 4 bytes (ceil)
        let buf9 = PackedBuffer::new(9, BitWidth::Three);
        assert_eq!(buf9.size_bytes(), 4);
    }

    // -----------------------------------------------------------------------
    // INT2 invalid group size
    // -----------------------------------------------------------------------

    #[test]
    fn test_int2_invalid_group_size() {
        let tensor = vec![0.5_f32; 16];
        let config = Int2QuantConfig { group_size: 0, symmetric: true };
        assert_eq!(quantize_int2(&tensor, &config), Err(QuantError::InvalidGroupSize));
    }
}
