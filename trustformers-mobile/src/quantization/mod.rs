//! Quantization modules for mobile deployment.
//!
//! - `int4`: INT4 per-group quantization (packed nibbles).
//! - `gguf_mobile`: GGUF file reader and quantization type mapping.

pub mod gguf_mobile;
pub mod int4;

pub use gguf_mobile::{
    GgufHeader, GgufLayerInfo, GgufMobileConfig, GgufMobileLoader, GgufQuantType, GgufReader,
};
pub use int4::{
    pack_int4, unpack_int4, Int4Config, Int4Gemv, Int4QuantConfig, Int4Tensor, MobileQuantError,
    QuantizationMetrics,
};

// ─── Module-level tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── pack_int4 / unpack_int4 roundtrip ─────────────────────────────────────

    #[test]
    fn test_mod_pack_unpack_roundtrip_full_range() {
        let values: Vec<i8> = (-7..=7).collect();
        let packed = pack_int4(&values);
        let unpacked = unpack_int4(&packed, values.len());
        assert_eq!(unpacked, values, "full [-7,7] roundtrip failed");
    }

    #[test]
    fn test_mod_pack_int4_zero_values() {
        let values = vec![0i8, 0, 0, 0];
        let packed = pack_int4(&values);
        let unpacked = unpack_int4(&packed, 4);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_mod_pack_int4_single_value() {
        let values = vec![3i8];
        let packed = pack_int4(&values);
        let unpacked = unpack_int4(&packed, 1);
        assert_eq!(unpacked[0], 3i8);
    }

    #[test]
    fn test_mod_unpack_count_shorter_than_packed() {
        let values: Vec<i8> = vec![-1, 2, -3, 4];
        let packed = pack_int4(&values);
        // Only retrieve 2 values
        let short = unpack_int4(&packed, 2);
        assert_eq!(short.len(), 2);
        assert_eq!(short[0], -1);
        assert_eq!(short[1], 2);
    }

    // ── Int4Config ────────────────────────────────────────────────────────────

    #[test]
    fn test_mod_int4_config_default_group_size() {
        let cfg = Int4Config::default();
        assert_eq!(cfg.group_size, 128);
    }

    #[test]
    fn test_mod_int4_config_symmetric_default() {
        let cfg = Int4Config::default();
        assert!(cfg.symmetric);
        assert!(!cfg.zero_point);
    }

    #[test]
    fn test_mod_int4_config_per_channel_off_default() {
        let cfg = Int4Config::default();
        assert!(!cfg.per_channel);
    }

    #[test]
    fn test_mod_int4_config_custom() {
        let cfg = Int4Config {
            group_size: 64,
            zero_point: true,
            symmetric: false,
            per_channel: true,
        };
        assert_eq!(cfg.group_size, 64);
        assert!(cfg.zero_point);
        assert!(!cfg.symmetric);
        assert!(cfg.per_channel);
    }

    // ── MobileQuantError ─────────────────────────────────────────────────────

    #[test]
    fn test_mod_mobile_quant_error_empty_input() {
        let result = Int4Tensor::from_config(&[], &Int4Config::default());
        assert!(matches!(result, Err(MobileQuantError::EmptyInput)));
    }

    #[test]
    fn test_mod_mobile_quant_error_group_size_zero() {
        let cfg = Int4Config { group_size: 0, ..Default::default() };
        let result = Int4Tensor::from_config(&[1.0, 2.0], &cfg);
        assert!(matches!(result, Err(MobileQuantError::InvalidGroupSize(0))));
    }

    #[test]
    fn test_mod_mobile_quant_error_shape_mismatch_message() {
        let e = MobileQuantError::ShapeMismatch { expected: 10, got: 5 };
        let msg = e.to_string();
        assert!(msg.contains("10") && msg.contains('5'));
    }

    // ── Int4Gemv ─────────────────────────────────────────────────────────────

    #[test]
    fn test_mod_gemv_output_shape_2x4() {
        let rows: usize = 2;
        let cols: usize = 4;
        let packed = vec![0x88u8; rows * cols.div_ceil(2)]; // all zero weights
        let scales = vec![1.0f32; rows];
        let input = vec![1.0f32; cols];
        let out = Int4Gemv::compute(&packed, &scales, &input, rows, cols);
        assert_eq!(out.len(), rows);
    }

    #[test]
    fn test_mod_gemv_zero_input_produces_zero() {
        let rows: usize = 3;
        let cols: usize = 6;
        let packed = vec![0xAAu8; rows * cols.div_ceil(2)]; // non-zero weights
        let scales = vec![1.0f32; rows];
        let input = vec![0.0f32; cols];
        let out = Int4Gemv::compute(&packed, &scales, &input, rows, cols);
        for &v in &out {
            assert!(v.abs() < 1e-6, "with zero input, output should be zero, got {v}");
        }
    }

    #[test]
    fn test_mod_gemv_scale_zero_produces_zero() {
        let rows: usize = 2;
        let cols: usize = 4;
        let packed = vec![0xAAu8; rows * cols.div_ceil(2)];
        let scales = vec![0.0f32; rows];
        let input = vec![1.0f32; cols];
        let out = Int4Gemv::compute(&packed, &scales, &input, rows, cols);
        for &v in &out {
            assert!(v.abs() < 1e-6, "with scale=0, output should be zero, got {v}");
        }
    }

    // ── GgufMobileConfig ─────────────────────────────────────────────────────

    #[test]
    fn test_mod_gguf_mobile_config_default() {
        let cfg = GgufMobileConfig::default();
        assert!(cfg.max_model_size_mb > 0.0);
        assert!(cfg.mmap);
        assert_eq!(cfg.offload_layers, 0);
    }

    #[test]
    fn test_mod_gguf_mobile_config_custom() {
        let cfg = GgufMobileConfig {
            max_model_size_mb: 512.0,
            offload_layers: 4,
            mmap: false,
        };
        assert_eq!(cfg.max_model_size_mb, 512.0);
        assert_eq!(cfg.offload_layers, 4);
        assert!(!cfg.mmap);
    }

    // ── GgufLayerInfo ────────────────────────────────────────────────────────

    #[test]
    fn test_mod_gguf_layer_info_construction() {
        let info = GgufLayerInfo::new("layer.weight", GgufQuantType::Q4_0, 1024, vec![32, 32]);
        assert_eq!(info.name, "layer.weight");
        assert_eq!(info.quant_type, GgufQuantType::Q4_0);
        assert_eq!(info.size_bytes, 1024);
        assert_eq!(info.tensor_shape, vec![32, 32]);
    }

    // ── GgufMobileLoader::estimate_memory_requirement ─────────────────────────

    #[test]
    fn test_mod_estimate_memory_requirement_empty() {
        assert_eq!(GgufMobileLoader::estimate_memory_requirement(&[]), 0);
    }

    #[test]
    fn test_mod_estimate_memory_requirement_sum() {
        let layers = vec![
            GgufLayerInfo::new("a", GgufQuantType::Q4_0, 1000, vec![]),
            GgufLayerInfo::new("b", GgufQuantType::Q8_0, 2000, vec![]),
            GgufLayerInfo::new("c", GgufQuantType::F16, 3000, vec![]),
        ];
        assert_eq!(GgufMobileLoader::estimate_memory_requirement(&layers), 6000);
    }

    // ── GgufMobileLoader::layers_that_fit ─────────────────────────────────────

    #[test]
    fn test_mod_layers_that_fit_all_fit() {
        let layers = vec![
            GgufLayerInfo::new("a", GgufQuantType::Q4_0, 100, vec![]),
            GgufLayerInfo::new("b", GgufQuantType::Q4_0, 200, vec![]),
        ];
        let budget_mb = 10.0; // plenty of space
        let indices = GgufMobileLoader::layers_that_fit(&layers, budget_mb);
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn test_mod_layers_that_fit_tight_budget() {
        let mb = 1.0 / (1024.0 * 1024.0); // 1 byte in MB units
        let layers = vec![
            GgufLayerInfo::new("tiny", GgufQuantType::Q4_0, 1, vec![]),
            GgufLayerInfo::new("big", GgufQuantType::F32, 10 * 1024 * 1024, vec![]),
        ];
        let indices = GgufMobileLoader::layers_that_fit(&layers, mb * 1.0);
        // Only first layer (1 byte) fits in 1-byte budget
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn test_mod_layers_that_fit_none_fit() {
        let layers = vec![
            GgufLayerInfo::new("huge", GgufQuantType::F32, 100 * 1024 * 1024, vec![]),
        ];
        let indices = GgufMobileLoader::layers_that_fit(&layers, 0.001); // 1 KB budget
        assert!(indices.is_empty());
    }

    // ── GgufMobileLoader::effective_bits_per_weight ───────────────────────────

    #[test]
    fn test_mod_effective_bits_q4_0() {
        assert_eq!(GgufMobileLoader::effective_bits_per_weight(GgufQuantType::Q4_0), 4.5);
    }

    #[test]
    fn test_mod_effective_bits_q4_1() {
        assert_eq!(GgufMobileLoader::effective_bits_per_weight(GgufQuantType::Q4_1), 5.0);
    }

    #[test]
    fn test_mod_effective_bits_q5_0() {
        assert_eq!(GgufMobileLoader::effective_bits_per_weight(GgufQuantType::Q5_0), 5.5);
    }

    #[test]
    fn test_mod_effective_bits_q5_1() {
        assert_eq!(GgufMobileLoader::effective_bits_per_weight(GgufQuantType::Q5_1), 6.0);
    }

    #[test]
    fn test_mod_effective_bits_q8_0() {
        assert_eq!(GgufMobileLoader::effective_bits_per_weight(GgufQuantType::Q8_0), 8.5);
    }

    #[test]
    fn test_mod_effective_bits_f16() {
        assert_eq!(GgufMobileLoader::effective_bits_per_weight(GgufQuantType::F16), 16.0);
    }

    #[test]
    fn test_mod_effective_bits_f32() {
        assert_eq!(GgufMobileLoader::effective_bits_per_weight(GgufQuantType::F32), 32.0);
    }

    // ── GgufMobileLoader::compression_ratio_vs_f32 ────────────────────────────

    #[test]
    fn test_mod_compression_ratio_f32() {
        // F32 vs F32 → ratio = 1.0
        let ratio = GgufMobileLoader::compression_ratio_vs_f32(GgufQuantType::F32);
        assert!((ratio - 1.0).abs() < 1e-5, "expected 1.0, got {ratio}");
    }

    #[test]
    fn test_mod_compression_ratio_q8_0() {
        let ratio = GgufMobileLoader::compression_ratio_vs_f32(GgufQuantType::Q8_0);
        // 32 / 8.5 ≈ 3.76
        assert!(ratio > 3.5 && ratio < 4.0, "Q8_0 ratio unexpected: {ratio}");
    }

    #[test]
    fn test_mod_compression_ratio_q4_0() {
        let ratio = GgufMobileLoader::compression_ratio_vs_f32(GgufQuantType::Q4_0);
        // 32 / 4.5 ≈ 7.11
        assert!(ratio > 7.0 && ratio < 7.5, "Q4_0 ratio unexpected: {ratio}");
    }

    // ── Re-export checks ──────────────────────────────────────────────────────

    #[test]
    fn test_mod_reexports_work() {
        // Verify that the top-level re-exports compile and are usable
        let _cfg = Int4QuantConfig::default();
        let _tensor_result = Int4Tensor::quantize(&[1.0f32, 2.0, 3.0, 4.0], &[4], &Int4QuantConfig { group_size: 4, ..Default::default() });
        let _qt = GgufQuantType::Q4_0;
    }

    #[test]
    fn test_mod_gguf_reader_reexport() {
        use gguf_mobile::make_minimal_gguf;
        let data = make_minimal_gguf("test");
        let reader = GgufReader::from_bytes(&data).expect("should parse");
        assert_eq!(reader.architecture(), Some("test"));
    }

    // ── Int4Tensor compression ratio ≈ 8.0 for very large tensors ─────────────

    #[test]
    fn test_mod_int4_compression_ratio_large() {
        let n = 8192usize;
        let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.0001 - 0.4096).collect();
        let config = Int4QuantConfig { group_size: 128, ..Default::default() };
        let tensor = Int4Tensor::quantize(&data, &[n], &config).expect("quantize");
        let ratio = tensor.compression_ratio();
        // With 8192 values: packed=4096 bytes, scales=8192/128*4=256 bytes → total=4352
        // fp32=32768 → ratio = 32768/4352 ≈ 7.53
        assert!(ratio > 7.0, "expected ratio > 7.0 for large tensor, got {ratio}");
    }
}
