//! Basic quantization algorithms (Dynamic, Static, Post-training)

use crate::optimization::quantization::config::*;
use std::vec::Vec;
use wasm_bindgen::prelude::*;

/// Quantize-then-dequantize `data` using real affine (min/max, scale/zero-point) quantization
/// sized to `precision.bits()`. This simulates the accuracy impact of quantizing to the given
/// bit-width while keeping the existing `Vec<f32>` API shape of this "basic algorithms" module —
/// actual bit-packed storage size is computed separately via `QuantizationPrecision::bytes_per_element`
/// (see `WebQuantizer::get_stats`).
fn affine_quantize_dequantize(data: &[f32], precision: QuantizationPrecision) -> Vec<f32> {
    if data.is_empty() {
        return Vec::new();
    }

    let bits = precision.bits();
    let levels: u64 = 1u64 << bits; // e.g. 16 for INT4
    let qmax = (levels - 1) as f32;

    let (min_val, max_val) =
        data.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &x| {
            (lo.min(x), hi.max(x))
        });

    if max_val <= min_val {
        // Constant (or degenerate) input: nothing to quantize, round-trip is exact.
        return vec![min_val; data.len()];
    }

    let scale = (max_val - min_val) / qmax;
    let zero_point = (-min_val / scale).round().clamp(0.0, qmax);

    data.iter()
        .map(|&x| {
            let q = (x / scale + zero_point).round().clamp(0.0, qmax);
            (q - zero_point) * scale
        })
        .collect()
}

/// Apply dynamic quantization: ranges are derived from `data` itself at call time.
pub fn apply_dynamic_quantization(
    data: &[f32],
    precision: QuantizationPrecision,
) -> Result<Vec<f32>, JsValue> {
    Ok(affine_quantize_dequantize(data, precision))
}

/// Apply static quantization: conceptually uses pre-computed calibration ranges; this
/// calibration-free module has no separate calibration dataset to draw from, so — like
/// `apply_dynamic_quantization` — it derives its range from `data` itself.
pub fn apply_static_quantization(
    data: &[f32],
    precision: QuantizationPrecision,
) -> Result<Vec<f32>, JsValue> {
    Ok(affine_quantize_dequantize(data, precision))
}

/// Apply post-training quantization: same affine quantize/dequantize round-trip — this
/// strategy differs from the others only in *when* calibration happens (offline, after
/// training), not in the elementwise math applied here.
pub fn apply_post_training_quantization(
    data: &[f32],
    precision: QuantizationPrecision,
) -> Result<Vec<f32>, JsValue> {
    Ok(affine_quantize_dequantize(data, precision))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn distinct_sorted(mut v: Vec<f32>) -> Vec<f32> {
        v.sort_by(|a, b| a.partial_cmp(b).expect("no NaNs in test data"));
        v.dedup();
        v
    }

    #[test]
    fn test_dynamic_quantization_int4_collapses_to_at_most_16_levels() {
        let data: Vec<f32> = (0..100).map(|i| i as f32 / 10.0).collect();
        let result = apply_dynamic_quantization(&data, QuantizationPrecision::INT4)
            .expect("quantization should not fail");
        assert_eq!(result.len(), data.len());
        let levels = distinct_sorted(result);
        assert!(
            levels.len() <= 16,
            "expected <= 16 distinct INT4 levels, got {}",
            levels.len()
        );
    }

    #[test]
    fn test_static_quantization_int8_collapses_to_at_most_256_levels() {
        let data: Vec<f32> = (0..1000).map(|i| (i as f32).sin() * 50.0).collect();
        let result = apply_static_quantization(&data, QuantizationPrecision::INT8)
            .expect("quantization should not fail");
        let levels = distinct_sorted(result);
        assert!(levels.len() <= 256);
    }

    #[test]
    fn test_post_training_quantization_int1_collapses_to_at_most_2_levels() {
        let data = vec![-3.0, -1.0, 0.0, 2.0, 5.0, 10.0];
        let result = apply_post_training_quantization(&data, QuantizationPrecision::INT1)
            .expect("quantization should not fail");
        let levels = distinct_sorted(result);
        assert!(
            levels.len() <= 2,
            "INT1 must collapse to at most 2 levels, got {}",
            levels.len()
        );
    }

    #[test]
    fn test_constant_input_round_trips_exactly() {
        let data = vec![7.0f32; 10];
        let result = apply_dynamic_quantization(&data, QuantizationPrecision::INT4)
            .expect("quantization should not fail");
        assert_eq!(result.len(), 10);
        assert!(result.iter().all(|&v| v == 7.0));
    }

    #[test]
    fn test_output_is_not_the_old_fixed_constant_multiplier() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 100.0];
        let result = apply_dynamic_quantization(&data, QuantizationPrecision::INT8)
            .expect("quantization should not fail");
        let old_placeholder: Vec<f32> = data.iter().map(|&x| x * 0.5).collect();
        assert_ne!(result, old_placeholder);
    }

    #[test]
    fn test_empty_input_returns_empty_output() {
        let data: Vec<f32> = Vec::new();
        let result = apply_dynamic_quantization(&data, QuantizationPrecision::INT8)
            .expect("quantization should not fail");
        assert!(result.is_empty());
    }

    #[test]
    fn test_output_length_matches_input_length_for_all_public_functions() {
        let data = vec![-10.0, -5.0, 0.0, 5.0, 10.0, 42.0, -42.0];
        let dynamic = apply_dynamic_quantization(&data, QuantizationPrecision::INT4)
            .expect("quantization should not fail");
        let static_q = apply_static_quantization(&data, QuantizationPrecision::INT4)
            .expect("quantization should not fail");
        let post_training = apply_post_training_quantization(&data, QuantizationPrecision::INT4)
            .expect("quantization should not fail");
        assert_eq!(dynamic.len(), data.len());
        assert_eq!(static_q.len(), data.len());
        assert_eq!(post_training.len(), data.len());
    }
}
