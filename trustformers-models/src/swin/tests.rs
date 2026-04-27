#[cfg(test)]
mod tests {
    use crate::swin::config::SwinConfig;
    use crate::swin::model::{
        window_partition, window_reverse, cyclic_shift,
        PatchMerging, SwinModel, SwinStage,
    };
    use crate::swin::tasks::SwinForImageClassification;
    use scirs2_core::ndarray::Array4;
    use trustformers_core::device::Device;
    use trustformers_core::traits::Config;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Very small Swin config for fast unit tests.
    ///
    /// image=32, patch=4 → initial_resolution=8
    /// stage 0: dim=16, depth=1, heads=2, window=4, downsample=True  → output 4×4 (dim=32)
    /// stage 1: dim=32, depth=1, heads=4, window=4, no downsample    → output 4×4 (dim=32)
    fn mini_config() -> SwinConfig {
        SwinConfig {
            image_size: 32, // 32/4 = 8 — 8x8 initial feature map
            patch_size: 4,
            num_channels: 3,
            embed_dim: 16,
            depths: vec![1, 1],
            num_heads: vec![2, 4],
            window_size: 4,   // 8 and 4 are both divisible by 4
            mlp_ratio: 2.0,
            qkv_bias: true,
            drop_rate: 0.0,
            attn_drop_rate: 0.0,
            drop_path_rate: 0.0,
            num_labels: 5,
            layer_norm_eps: 1e-5,
        }
    }

    // ── config tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_swin_config_tiny() {
        let config = SwinConfig::swin_tiny_patch4_window7_224();
        assert_eq!(config.image_size, 224);
        assert_eq!(config.patch_size, 4);
        assert_eq!(config.embed_dim, 96);
        assert_eq!(config.depths, vec![2, 2, 6, 2]);
        assert_eq!(config.num_heads, vec![3, 6, 12, 24]);
        assert_eq!(config.window_size, 7);
        assert_eq!(config.num_stages(), 4);
        assert_eq!(config.initial_resolution(), 56); // 224 / 4
        assert_eq!(config.stage_dim(0), 96);
        assert_eq!(config.stage_dim(1), 192);
        assert_eq!(config.stage_dim(2), 384);
        assert_eq!(config.stage_dim(3), 768);
        assert_eq!(config.final_dim(), 768);
        config.validate().expect("tiny config should be valid");
    }

    #[test]
    fn test_swin_config_small() {
        let config = SwinConfig::swin_small_patch4_window7_224();
        assert_eq!(config.embed_dim, 96);
        assert_eq!(config.depths, vec![2, 2, 18, 2]);
        config.validate().expect("small config should be valid");
    }

    #[test]
    fn test_swin_config_base() {
        let config = SwinConfig::swin_base_patch4_window7_224();
        assert_eq!(config.embed_dim, 128);
        assert_eq!(config.final_dim(), 1024); // 128 * 8
        config.validate().expect("base config should be valid");
    }

    #[test]
    fn test_swin_config_base_384() {
        let config = SwinConfig::swin_base_patch4_window12_384();
        assert_eq!(config.image_size, 384);
        assert_eq!(config.window_size, 12);
        assert_eq!(config.initial_resolution(), 96);
        config.validate().expect("base-384 config should be valid");
    }

    #[test]
    fn test_swin_config_invalid_depths_heads_mismatch() {
        let config = SwinConfig {
            depths: vec![2, 2, 6],
            num_heads: vec![3, 6], // length mismatch
            ..SwinConfig::swin_tiny_patch4_window7_224()
        };
        assert!(config.validate().is_err());
    }

    // ── window partition / reverse ────────────────────────────────────────────

    #[test]
    fn test_window_partition() {
        // 4D tensor (1, 14, 14, 16), window_size=7 → 4 windows of (7,7,16)
        let x = Array4::<f32>::ones((1, 14, 14, 16));
        let windows = window_partition(&x, 7).expect("window_partition should succeed");
        assert_eq!(windows.shape(), &[4, 7, 7, 16]);
    }

    #[test]
    fn test_window_partition_multiple_batches() {
        let x = Array4::<f32>::ones((2, 14, 14, 8));
        let windows = window_partition(&x, 7).expect("window_partition should succeed");
        // 2 batches × (14/7)^2 = 2 × 4 = 8 windows
        assert_eq!(windows.shape(), &[8, 7, 7, 8]);
    }

    #[test]
    fn test_window_reverse_roundtrip() {
        let x = Array4::from_shape_fn((1, 14, 14, 4), |(_, i, j, k)| {
            (i * 14 * 4 + j * 4 + k) as f32
        });
        let windows = window_partition(&x, 7).expect("partition should succeed");
        let recovered = window_reverse(&windows, 7, 14, 14).expect("reverse should succeed");
        assert_eq!(recovered.shape(), x.shape());
        // Check round-trip fidelity
        for bi in 0..1 {
            for i in 0..14 {
                for j in 0..14 {
                    for k in 0..4 {
                        assert!((recovered[[bi, i, j, k]] - x[[bi, i, j, k]]).abs() < 1e-5);
                    }
                }
            }
        }
    }

    #[test]
    fn test_cyclic_shift_identity() {
        let x = Array4::<f32>::ones((1, 7, 7, 4));
        let shifted = cyclic_shift(&x, 0);
        assert_eq!(shifted.shape(), x.shape());
        // Shifting by 0 returns the original
        for bi in 0..1 {
            for i in 0..7 {
                for j in 0..7 {
                    for k in 0..4 {
                        assert!((shifted[[bi, i, j, k]] - x[[bi, i, j, k]]).abs() < 1e-5);
                    }
                }
            }
        }
    }

    // ── patch merging ─────────────────────────────────────────────────────────

    #[test]
    fn test_patch_merging_shapes() {
        let pm = PatchMerging::new(16, 1e-5, Device::CPU).expect("PatchMerging::new should succeed");
        let x = Array4::<f32>::zeros((2, 14, 14, 16));
        let out = pm.forward(&x).expect("PatchMerging forward should succeed");
        // (B, H/2, W/2, 2C)
        assert_eq!(out.shape(), &[2, 7, 7, 32]);
    }

    #[test]
    fn test_patch_merging_wrong_channels() {
        let pm = PatchMerging::new(16, 1e-5, Device::CPU).expect("PatchMerging::new should succeed");
        let x = Array4::<f32>::zeros((1, 14, 14, 8)); // wrong channels
        assert!(pm.forward(&x).is_err());
    }

    // ── stage forward ─────────────────────────────────────────────────────────

    #[test]
    fn test_swin_stage_forward_no_downsample() {
        // Stage without downsampling (last stage)
        let stage = SwinStage::new(
            16,       // dim
            2,        // depth
            2,        // num_heads
            7,        // window_size
            2.0,      // mlp_ratio
            true,     // qkv_bias
            0.0,      // drop_rate
            0.0,      // attn_drop
            1e-5,     // layer_norm_eps
            false,    // no downsample
            Device::CPU,
        )
        .expect("SwinStage::new should succeed");

        let x = Array4::<f32>::zeros((1, 7, 7, 16));
        let out = stage.forward(&x).expect("stage forward should succeed");
        assert_eq!(out.shape(), &[1, 7, 7, 16]);
    }

    #[test]
    fn test_swin_stage_forward_with_downsample() {
        let stage = SwinStage::new(
            16,    // dim
            2,     // depth
            2,     // num_heads
            7,     // window_size
            2.0,   // mlp_ratio
            true,  // qkv_bias
            0.0,   // drop_rate
            0.0,   // attn_drop
            1e-5,  // layer_norm_eps
            true,  // with downsample
            Device::CPU,
        )
        .expect("SwinStage::new should succeed");

        let x = Array4::<f32>::zeros((1, 14, 14, 16));
        let out = stage.forward(&x).expect("stage with downsample should succeed");
        // After PatchMerging: (B, H/2, W/2, 2C)
        assert_eq!(out.shape(), &[1, 7, 7, 32]);
    }

    // ── model shapes ─────────────────────────────────────────────────────────

    #[test]
    fn test_swin_model_shapes() {
        let config = mini_config();
        let model = SwinModel::new(config.clone()).expect("SwinModel::new should succeed");

        let images = Array4::<f32>::zeros((1, 32, 32, 3));
        let features = model.forward(&images).expect("SwinModel forward should succeed");

        // final_dim = embed_dim * 2^(num_stages-1) = 16 * 2 = 32
        let expected_dim = config.final_dim();
        assert_eq!(features.shape(), &[1, expected_dim]);
    }

    #[test]
    fn test_swin_model_batch() {
        let config = mini_config();
        let model = SwinModel::new(config.clone()).expect("SwinModel::new should succeed");

        let images = Array4::<f32>::zeros((3, 32, 32, 3));
        let features = model.forward(&images).expect("SwinModel forward should succeed");
        assert_eq!(features.shape(), &[3, config.final_dim()]);
    }

    // ── classification head ───────────────────────────────────────────────────

    #[test]
    fn test_swin_classification_head() {
        let config = mini_config();
        let model = SwinForImageClassification::new(config.clone())
            .expect("SwinForImageClassification::new should succeed");

        let images = Array4::<f32>::zeros((2, 32, 32, 3));
        let logits = model.forward(&images).expect("classification forward should succeed");
        assert_eq!(logits.shape(), &[2, config.num_labels]);
    }

    #[test]
    fn test_swin_weight_map_structure() {
        let config = mini_config();
        let model = SwinForImageClassification::new(config.clone())
            .expect("SwinForImageClassification::new should succeed");
        let wmap = model.weight_map();

        // Check classifier head entries are present
        assert!(wmap.contains_key("classifier.weight"));
        assert!(wmap.contains_key("classifier.bias"));
        assert!(wmap.contains_key("swin.layernorm.weight"));

        // Each stage produces block entries
        for s_idx in 0..config.num_stages() {
            for b_idx in 0..config.depths[s_idx] {
                let key = format!(
                    "swin.encoder.layers.{}.blocks.{}.layernorm_before.weight",
                    s_idx, b_idx
                );
                assert!(wmap.contains_key(&key), "Missing key: {}", key);
            }
        }
    }
}
