//! GPTQ quantization algorithm for TrustformeRS.
//!
//! GPTQ (Generative Pre-trained Transformer Quantization) uses second-order
//! information (Hessian inverse) to minimise quantization error column by
//! column.
//!
//! Reference: "GPTQ: Accurate Post-Training Quantization for Generative
//! Pre-trained Transformers" (Frantar et al., 2022).

use crate::quantization::packed::QuantError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for GPTQ quantization.
#[derive(Debug, Clone)]
pub struct GptqConfig {
    /// Quantization bit-width (2, 3, 4, or 8).
    pub bits: u32,
    /// Number of weight columns per quantization group.
    pub group_size: usize,
    /// Diagonal dampening fraction applied to the Hessian.
    pub damp_percent: f32,
    /// Activation ordering: process columns by decreasing Hessian diagonal.
    pub desc_act: bool,
    /// Use static (pre-computed) groups instead of dynamic assignment.
    pub static_groups: bool,
    /// Symmetric quantization (zero_point = 0).
    pub sym: bool,
}

impl Default for GptqConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 128,
            damp_percent: 0.01,
            desc_act: false,
            static_groups: false,
            sym: true,
        }
    }
}

impl GptqConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), QuantError> {
        match self.bits {
            2 | 3 | 4 | 8 => {}
            _ => return Err(QuantError::ValueOutOfRange {
                val: self.bits as i64,
                min: 2,
                max: 8,
            }),
        }
        if self.group_size == 0 {
            return Err(QuantError::InvalidGroupSize);
        }
        Ok(())
    }

    /// Returns the signed minimum quantized value for the configured bit-width.
    fn min_q(&self) -> i64 {
        if self.sym {
            -(1i64 << (self.bits - 1))
        } else {
            0
        }
    }

    /// Returns the signed maximum quantized value for the configured bit-width.
    fn max_q(&self) -> i64 {
        if self.sym {
            (1i64 << (self.bits - 1)) - 1
        } else {
            (1i64 << self.bits) - 1
        }
    }

    /// Values packed per i32 word.
    fn vals_per_i32(&self) -> usize {
        32 / self.bits as usize
    }
}

// ---------------------------------------------------------------------------
// Quantized weight container
// ---------------------------------------------------------------------------

/// GPTQ quantized weight matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct GptqQuantizedWeight {
    /// Quantized weights packed as i32 (column-major, `vals_per_i32` weights per word).
    pub qweight: Vec<i32>,
    /// Quantized zero-points packed as i32.
    pub qzeros: Vec<i32>,
    /// Per-group f32 scales.  Shape: `[num_groups, cols]`.
    pub scales: Vec<f32>,
    /// Group index for each column (used when `desc_act = true`).
    pub g_idx: Vec<i32>,
    /// Bit-width used.
    pub bits: u32,
    /// Group size used during quantization.
    pub group_size: usize,
    /// Number of rows in the original weight matrix.
    pub rows: usize,
    /// Number of columns in the original weight matrix.
    pub cols: usize,
}

// ---------------------------------------------------------------------------
// Packing / unpacking helpers
// ---------------------------------------------------------------------------

/// Pack `32/bits` signed integer values per i32 word.
///
/// Values are stored as unsigned integers with an offset so that the signed
/// range maps to non-negative storage (same convention as `PackedBuffer`).
pub fn pack_weights_to_i32(weights: &[i64], bits: u32) -> Vec<i32> {
    let vals_per_word = 32 / bits as usize;
    let num_words = weights.len().div_ceil(vals_per_word);
    let mut packed = vec![0i32; num_words];

    let offset = 1i64 << (bits - 1); // e.g. 8 for 4-bit
    let mask = (1u64 << bits) - 1;

    for (i, &w) in weights.iter().enumerate() {
        let word_idx = i / vals_per_word;
        let bit_pos = (i % vals_per_word) * bits as usize;
        let unsigned = (w + offset) as u64 & mask;
        packed[word_idx] |= (unsigned << bit_pos) as i32;
    }
    packed
}

/// Unpack `num_elements` signed integers from i32 packed words.
pub fn unpack_weights_from_i32(packed: &[i32], bits: u32, num_elements: usize) -> Vec<i64> {
    let vals_per_word = 32 / bits as usize;
    let offset = 1i64 << (bits - 1);
    let mask = (1u64 << bits) - 1;

    let mut out = Vec::with_capacity(num_elements);
    for i in 0..num_elements {
        let word_idx = i / vals_per_word;
        let bit_pos = (i % vals_per_word) * bits as usize;
        if word_idx >= packed.len() {
            out.push(0i64);
        } else {
            let unsigned = ((packed[word_idx] as u64) >> bit_pos) & mask;
            out.push(unsigned as i64 - offset);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Core GPTQ quantization
// ---------------------------------------------------------------------------

/// Quantize a weight matrix using the GPTQ algorithm.
///
/// `weight` is stored in row-major order with shape `[rows, cols]`.
/// `hessian` (optional) is the diagonal of the Hessian in the same column
/// ordering.  When `None` an identity diagonal is used, which degrades to
/// round-to-nearest quantization with GPTQ-style error propagation.
pub fn gptq_quantize_layer(
    weight: &[f32],
    rows: usize,
    cols: usize,
    hessian: Option<&[f32]>,
    config: &GptqConfig,
) -> Result<GptqQuantizedWeight, QuantError> {
    config.validate()?;

    if weight.is_empty() {
        return Err(QuantError::EmptyTensor);
    }
    if weight.len() != rows * cols {
        return Err(QuantError::InvalidGroupSize);
    }

    let min_q = config.min_q();
    let max_q = config.max_q();
    let num_groups = rows.div_ceil(config.group_size);

    // Track whether a real Hessian was provided (controls error propagation).
    let has_real_hessian = hessian.map(|h| h.len() == cols).unwrap_or(false);

    // Build or validate the Hessian diagonal (one value per column).
    let hess_diag: Vec<f32> = match hessian {
        Some(h) if h.len() == cols => {
            // Apply dampening: H_ii += damp * mean(diag(H))
            let mean_diag = h.iter().sum::<f32>() / h.len() as f32;
            let damp = config.damp_percent * mean_diag;
            h.iter().map(|&v| v + damp).collect()
        }
        Some(_) => {
            // Wrong length — fall back to identity.
            vec![1.0_f32; cols]
        }
        None => {
            // No Hessian provided — use identity diagonal.
            vec![1.0_f32; cols]
        }
    };

    // Column ordering (desc_act: sort by decreasing H diagonal).
    let col_order: Vec<usize> = if config.desc_act {
        let mut order: Vec<usize> = (0..cols).collect();
        order.sort_by(|&a, &b| {
            hess_diag[b]
                .partial_cmp(&hess_diag[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        order
    } else {
        (0..cols).collect()
    };

    // Working copy of weights (row-major [rows, cols]).
    let mut w_work: Vec<f32> = weight.to_vec();

    // Per-group scales and zero-points, indexed [group_row, col].
    let mut scales = vec![0.0_f32; num_groups * cols];
    let mut zero_points = vec![0.0_f32; num_groups * cols];

    // Compute per-group scales / zero-points from the original weights.
    for col in 0..cols {
        for group in 0..num_groups {
            let row_start = group * config.group_size;
            let row_end = (row_start + config.group_size).min(rows);
            let group_vals: Vec<f32> = (row_start..row_end)
                .map(|r| w_work[r * cols + col])
                .collect();

            let (scale, zp) =
                group_scale_zero_point(&group_vals, min_q, max_q, config.sym);
            scales[group * cols + col] = scale;
            zero_points[group * cols + col] = zp;
        }
    }

    // GPTQ error propagation — process columns in the chosen order.
    let mut quantized_weights: Vec<i64> = vec![0i64; rows * cols];

    for &col in &col_order {
        let h_inv = if hess_diag[col].abs() > f32::EPSILON {
            1.0_f32 / hess_diag[col]
        } else {
            1.0_f32
        };

        for row in 0..rows {
            let group = row / config.group_size;
            let scale = scales[group * cols + col];
            let zp = zero_points[group * cols + col];
            let w = w_work[row * cols + col];

            let q = quantize_f32(w, scale, zp, min_q, max_q);
            quantized_weights[row * cols + col] = q;

            // Error between original and quantized value.
            let w_q = dequantize_i64(q, scale, zp);
            let err = (w - w_q) * h_inv;

            // Propagate quantization error to remaining columns in the same row.
            // In the full GPTQ algorithm the update is:
            //   W[:, j+1:] -= (w_err / H_inv[j,j]) * H_inv[j, j+1:]
            // With only the diagonal available (and no cross-column Hessian
            // terms), we only propagate when a real Hessian was supplied.
            // When using the identity fallback, there is no second-order
            // information so the update would be arbitrary noise.
            if has_real_hessian {
                for remaining_col in (col + 1)..cols {
                    let ratio = if hess_diag[col].abs() > f32::EPSILON {
                        hess_diag[remaining_col] / hess_diag[col]
                    } else {
                        0.0_f32
                    };
                    w_work[row * cols + remaining_col] -= err * ratio;
                }
            }
        }
    }

    // Build g_idx: group index for each column.
    let g_idx: Vec<i32> = if config.desc_act {
        // Map original column -> position in col_order, then derive group.
        let mut position = vec![0usize; cols];
        for (pos, &col) in col_order.iter().enumerate() {
            position[col] = pos;
        }
        position.iter().map(|&p| (p / config.group_size) as i32).collect()
    } else {
        (0..cols).map(|c| (c / config.group_size) as i32).collect()
    };

    // Pack quantized weights (column-major for GPTQ convention):
    // Reorder to column-major [cols, rows] before packing.
    let mut col_major: Vec<i64> = Vec::with_capacity(cols * rows);
    for c in 0..cols {
        for r in 0..rows {
            col_major.push(quantized_weights[r * cols + c]);
        }
    }
    let vals_per_word = config.vals_per_i32();
    let words_per_col = rows.div_ceil(vals_per_word);
    let qweight = pack_weights_to_i32(&col_major, config.bits);

    // Pack zero-points (column-major: [cols, num_groups]).
    // zero_points layout: [num_groups, cols] → transpose to [cols, num_groups].
    // Store the raw quantized zp (signed) — pack_weights_to_i32 handles the
    // unsigned offset internally, so we must NOT pre-apply the offset here.
    let mut zp_col_major: Vec<i64> = Vec::with_capacity(cols * num_groups);
    for c in 0..cols {
        for g in 0..num_groups {
            let zp_f = zero_points[g * cols + c];
            // Round the f32 zero-point to the nearest representable integer.
            let zp_q = (zp_f.round() as i64).clamp(min_q, max_q);
            zp_col_major.push(zp_q);
        }
    }
    let qzeros = pack_weights_to_i32(&zp_col_major, config.bits);

    // Flatten scales to [cols, num_groups] order.
    let mut scales_out: Vec<f32> = Vec::with_capacity(cols * num_groups);
    for c in 0..cols {
        for g in 0..num_groups {
            scales_out.push(scales[g * cols + c]);
        }
    }

    let _ = words_per_col; // used for documentation purposes
    Ok(GptqQuantizedWeight {
        qweight,
        qzeros,
        scales: scales_out,
        g_idx,
        bits: config.bits,
        group_size: config.group_size,
        rows,
        cols,
    })
}

/// Dequantize a `GptqQuantizedWeight` back to a row-major f32 matrix.
pub fn gptq_dequantize(qw: &GptqQuantizedWeight) -> Result<Vec<f32>, QuantError> {
    if qw.rows == 0 || qw.cols == 0 {
        return Err(QuantError::EmptyTensor);
    }

    let num_groups = qw.rows.div_ceil(qw.group_size);
    let num_elements = qw.rows * qw.cols;

    // Unpack column-major weights.
    let col_major = unpack_weights_from_i32(&qw.qweight, qw.bits, num_elements);

    // Unpack zero-points (column-major: [cols, num_groups]).
    let num_zp = qw.cols * num_groups;
    let zp_col_major = unpack_weights_from_i32(&qw.qzeros, qw.bits, num_zp);

    let mut output = vec![0.0_f32; num_elements];

    for col in 0..qw.cols {
        for row in 0..qw.rows {
            let group = row / qw.group_size;
            // scales layout: [cols, num_groups]
            let scale = qw.scales[col * num_groups + group];
            // zero-points are stored as signed integers via pack_weights_to_i32,
            // which applies its own unsigned offset.  unpack_weights_from_i32
            // already undoes that offset, so the returned value is the raw
            // signed zp_q — convert directly to f32.
            let zp_q = zp_col_major[col * num_groups + group];
            let zp = zp_q as f32;

            let q = col_major[col * qw.rows + row];
            // Convert back to row-major output.
            output[row * qw.cols + col] = dequantize_i64(q, scale, zp);
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Quantization helpers
// ---------------------------------------------------------------------------

fn group_scale_zero_point(
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
        let scale = max_abs / max_q as f32;
        (scale, 0.0_f32)
    } else {
        let q_range = (max_q - min_q) as f32;
        let f_range = fmax - fmin;
        if f_range < f32::EPSILON {
            return (1.0_f32, fmin);
        }
        let scale = f_range / q_range;
        (scale, fmin)
    }
}

fn quantize_f32(val: f32, scale: f32, zero_point: f32, min_q: i64, max_q: i64) -> i64 {
    if scale.abs() < f32::EPSILON {
        return 0;
    }
    let q = ((val - zero_point) / scale).round() as i64;
    q.clamp(min_q, max_q)
}

fn dequantize_i64(q: i64, scale: f32, zero_point: f32) -> f32 {
    q as f32 * scale + zero_point
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
        let cfg = GptqConfig::default();
        assert_eq!(cfg.bits, 4);
        assert_eq!(cfg.group_size, 128);
        assert!((cfg.damp_percent - 0.01).abs() < 1e-6);
        assert!(!cfg.desc_act);
        assert!(!cfg.static_groups);
        assert!(cfg.sym);
    }

    #[test]
    fn test_config_validation_valid() {
        for &bits in &[2u32, 3, 4, 8] {
            let cfg = GptqConfig { bits, ..Default::default() };
            assert!(cfg.validate().is_ok(), "bits={bits} should be valid");
        }
    }

    #[test]
    fn test_config_validation_invalid_bits() {
        let cfg = GptqConfig { bits: 5, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validation_zero_group_size() {
        let cfg = GptqConfig { group_size: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    // -----------------------------------------------------------------------
    // Weight packing / unpacking
    // -----------------------------------------------------------------------

    #[test]
    fn test_pack_unpack_4bit_round_trip() {
        let values: Vec<i64> = vec![-8, -4, 0, 3, 7, -1, -8, 7];
        let packed = pack_weights_to_i32(&values, 4);
        let unpacked = unpack_weights_from_i32(&packed, 4, values.len());
        assert_eq!(unpacked, values, "4-bit pack/unpack round-trip failed");
    }

    #[test]
    fn test_pack_unpack_2bit_round_trip() {
        let values: Vec<i64> = vec![-2, -1, 0, 1, -2, 1, 0, -1];
        let packed = pack_weights_to_i32(&values, 2);
        let unpacked = unpack_weights_from_i32(&packed, 2, values.len());
        assert_eq!(unpacked, values, "2-bit pack/unpack round-trip failed");
    }

    #[test]
    fn test_pack_unpack_8bit_round_trip() {
        let values: Vec<i64> = vec![-128, -64, 0, 63, 127, -1, 42, -100];
        let packed = pack_weights_to_i32(&values, 8);
        let unpacked = unpack_weights_from_i32(&packed, 8, values.len());
        assert_eq!(unpacked, values, "8-bit pack/unpack round-trip failed");
    }

    // -----------------------------------------------------------------------
    // Basic quantization round-trip (small matrix)
    // -----------------------------------------------------------------------

    #[test]
    fn test_quantize_small_matrix_no_hessian() {
        let rows = 4;
        let cols = 4;
        // Simple weight matrix with values in [-1, 1].
        let weight: Vec<f32> = vec![
            0.1, -0.2, 0.3, -0.4,
            0.5, -0.6, 0.7, -0.8,
            0.9, -1.0, 0.0,  0.5,
            -0.5, 0.25, -0.75, 1.0,
        ];
        let config = GptqConfig {
            bits: 4,
            group_size: 4, // one group per column
            ..Default::default()
        };
        let qw = gptq_quantize_layer(&weight, rows, cols, None, &config)
            .expect("quantize failed");
        assert_eq!(qw.rows, rows);
        assert_eq!(qw.cols, cols);
        assert_eq!(qw.bits, 4);
    }

    // -----------------------------------------------------------------------
    // Dequantization accuracy
    // -----------------------------------------------------------------------

    #[test]
    fn test_dequantization_reconstruction_error() {
        let rows = 8;
        let cols = 8;
        let weight: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32 - 32.0) / 32.0))
            .collect();
        let config = GptqConfig {
            bits: 4,
            group_size: 8,
            sym: true,
            ..Default::default()
        };
        let qw = gptq_quantize_layer(&weight, rows, cols, None, &config)
            .expect("quantize failed");
        let reconstructed = gptq_dequantize(&qw).expect("dequantize failed");

        assert_eq!(reconstructed.len(), weight.len());

        let max_err = weight
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        // With 4-bit symmetric quantization, max error should be < 0.15 for this range.
        assert!(max_err < 0.15_f32, "reconstruction error {max_err} exceeds threshold");
    }

    // -----------------------------------------------------------------------
    // Group size handling
    // -----------------------------------------------------------------------

    #[test]
    fn test_group_size_larger_than_rows() {
        // group_size > rows → single group for all rows.
        let rows = 4;
        let cols = 2;
        let weight: Vec<f32> = vec![0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4];
        let config = GptqConfig {
            bits: 4,
            group_size: 128, // much larger than rows=4
            ..Default::default()
        };
        let qw = gptq_quantize_layer(&weight, rows, cols, None, &config)
            .expect("quantize failed");
        assert_eq!(qw.rows, rows);
        assert_eq!(qw.cols, cols);
    }

    // -----------------------------------------------------------------------
    // With Hessian diagonal
    // -----------------------------------------------------------------------

    #[test]
    fn test_quantize_with_hessian() {
        let rows = 4;
        let cols = 4;
        let weight: Vec<f32> = vec![
            0.1, 0.2, -0.3, 0.4,
            -0.1, 0.5, 0.0, -0.2,
            0.3, -0.4, 0.2, 0.1,
            -0.5, 0.3, -0.1, 0.4,
        ];
        // Hessian diagonal — one value per column.
        let hessian: Vec<f32> = vec![1.0, 2.0, 0.5, 3.0];
        let config = GptqConfig {
            bits: 4,
            group_size: 4,
            damp_percent: 0.01,
            ..Default::default()
        };
        let qw = gptq_quantize_layer(&weight, rows, cols, Some(&hessian), &config)
            .expect("quantize with hessian failed");
        let reconstructed = gptq_dequantize(&qw).expect("dequantize failed");
        assert_eq!(reconstructed.len(), weight.len());
        for v in &reconstructed {
            assert!(v.is_finite(), "non-finite dequantized value");
        }
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_weight_error() {
        let config = GptqConfig::default();
        let result = gptq_quantize_layer(&[], 0, 0, None, &config);
        assert_eq!(result, Err(QuantError::EmptyTensor));
    }

    #[test]
    fn test_dequantize_empty_error() {
        let qw = GptqQuantizedWeight {
            qweight: vec![],
            qzeros: vec![],
            scales: vec![],
            g_idx: vec![],
            bits: 4,
            group_size: 128,
            rows: 0,
            cols: 0,
        };
        assert_eq!(gptq_dequantize(&qw), Err(QuantError::EmptyTensor));
    }

    // -----------------------------------------------------------------------
    // desc_act ordering
    // -----------------------------------------------------------------------

    #[test]
    fn test_desc_act_ordering() {
        let rows = 4;
        let cols = 4;
        let weight: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        // High hessian on last column → it should be processed first.
        let hessian: Vec<f32> = vec![1.0, 1.0, 1.0, 10.0];
        let config = GptqConfig {
            bits: 4,
            group_size: 4,
            desc_act: true,
            ..Default::default()
        };
        let qw = gptq_quantize_layer(&weight, rows, cols, Some(&hessian), &config)
            .expect("desc_act quantize failed");
        // Column with highest hessian (col 3) should appear at position 0 in g_idx.
        assert_eq!(qw.g_idx[3], 0, "highest hessian column should be in group 0");
    }
}
