#[cfg(test)]
mod tests {
    use crate::deit::config::DeiTConfig;
    use crate::deit::model::{DeiTModel, DeiTPatchEmbedding};
    use crate::deit::tasks::DeiTForImageClassification;
    use scirs2_core::ndarray::Array4;
    use trustformers_core::traits::Config;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Return a minimal DeiT config that compiles quickly in tests.
    fn mini_config() -> DeiTConfig {
        DeiTConfig {
            image_size: 32,
            patch_size: 8,
            num_channels: 3,
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            intermediate_size: 256,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            num_labels: 10,
            qkv_bias: true,
            layer_norm_eps: 1e-6,
            use_distillation_token: true,
            model_type: "deit".to_string(),
        }
    }

    // ── config tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_deit_config_tiny() {
        let config = DeiTConfig::deit_tiny_patch16_224();
        assert_eq!(config.image_size, 224);
        assert_eq!(config.patch_size, 16);
        assert_eq!(config.hidden_size, 192);
        assert_eq!(config.num_attention_heads, 3);
        assert_eq!(config.num_hidden_layers, 12);
        assert_eq!(config.intermediate_size, 768);
        assert!(config.use_distillation_token);
        assert_eq!(config.num_patches(), 196); // (224/16)^2
        assert_eq!(config.seq_length(), 198); // 196 + CLS + DIST
        config.validate().expect("tiny config should be valid");
    }

    #[test]
    fn test_deit_config_base() {
        let config = DeiTConfig::deit_base_patch16_224();
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_attention_heads, 12);
        assert_eq!(config.intermediate_size, 3072);
        assert_eq!(config.num_patches_per_side(), 14);
        assert_eq!(config.num_patches(), 196);
        assert_eq!(config.seq_length(), 198);
        config.validate().expect("base config should be valid");
    }

    #[test]
    fn test_deit_config_base_384() {
        let config = DeiTConfig::deit_base_patch16_384();
        assert_eq!(config.image_size, 384);
        assert_eq!(config.num_patches(), 576); // (384/16)^2
        assert_eq!(config.seq_length(), 578); // 576 + CLS + DIST
        config.validate().expect("base-384 config should be valid");
    }

    #[test]
    fn test_deit_config_small() {
        let config = DeiTConfig::deit_small_patch16_224();
        assert_eq!(config.hidden_size, 384);
        assert_eq!(config.num_attention_heads, 6);
        config.validate().expect("small config should be valid");
    }

    #[test]
    fn test_deit_config_without_distillation_token() {
        let config = DeiTConfig {
            use_distillation_token: false,
            ..DeiTConfig::deit_tiny_patch16_224()
        };
        assert_eq!(config.seq_length(), 197); // 196 + CLS only
        config.validate().expect("config without distillation should be valid");
    }

    #[test]
    fn test_deit_config_invalid_hidden_size() {
        let config = DeiTConfig {
            hidden_size: 100, // not divisible by 3 heads (100 % 3 != 0)
            num_attention_heads: 3,
            ..DeiTConfig::deit_tiny_patch16_224()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_deit_config_invalid_image_size() {
        let config = DeiTConfig {
            image_size: 225, // not divisible by 16
            ..DeiTConfig::deit_tiny_patch16_224()
        };
        assert!(config.validate().is_err());
    }

    // ── patch embedding ───────────────────────────────────────────────────────

    #[test]
    fn test_deit_patch_embedding() {
        let config = mini_config();
        let patch_emb = DeiTPatchEmbedding::new(&config);

        // 32×32 image with 8×8 patches → (32/8)^2 = 16 patches
        let image = Array4::<f32>::zeros((1, 32, 32, 3));
        let result = patch_emb.forward(&image).expect("patch embedding should succeed");
        assert_eq!(result.shape(), &[1, 16, 64]);
    }

    #[test]
    fn test_deit_patch_embedding_batch() {
        let config = mini_config();
        let patch_emb = DeiTPatchEmbedding::new(&config);

        let image = Array4::<f32>::zeros((3, 32, 32, 3));
        let result = patch_emb.forward(&image).expect("batch patch embedding should succeed");
        assert_eq!(result.shape(), &[3, 16, 64]);
    }

    #[test]
    fn test_deit_patch_embedding_wrong_channels() {
        let config = mini_config();
        let patch_emb = DeiTPatchEmbedding::new(&config);

        let image = Array4::<f32>::zeros((1, 32, 32, 1)); // wrong channel count
        assert!(patch_emb.forward(&image).is_err());
    }

    // ── model shapes ─────────────────────────────────────────────────────────

    #[test]
    fn test_deit_model_shapes() {
        let config = mini_config();
        let model = DeiTModel::new(config).expect("model construction should succeed");

        let images = Array4::<f32>::zeros((1, 32, 32, 3));
        let output = model.forward(&images).expect("forward should succeed");

        // 16 patches + CLS + DIST = 18 tokens
        assert_eq!(output.shape(), &[1, 18, 64]);
    }

    #[test]
    fn test_deit_model_cls_output() {
        let config = mini_config();
        let model = DeiTModel::new(config).expect("model construction should succeed");

        let images = Array4::<f32>::zeros((2, 32, 32, 3));
        let cls = model.get_cls_output(&images).expect("cls output should succeed");
        assert_eq!(cls.shape(), &[2, 64]);
    }

    #[test]
    fn test_deit_distillation_token() {
        let config = mini_config();
        let model = DeiTModel::new(config).expect("model construction should succeed");

        let images = Array4::<f32>::zeros((2, 32, 32, 3));
        let dist = model
            .get_distillation_output(&images)
            .expect("distillation output should succeed");
        assert_eq!(dist.shape(), &[2, 64]);
    }

    #[test]
    fn test_deit_distillation_token_absent() {
        let config = DeiTConfig {
            use_distillation_token: false,
            ..mini_config()
        };
        let model = DeiTModel::new(config).expect("model construction should succeed");

        let images = Array4::<f32>::zeros((1, 32, 32, 3));
        // Should error when distillation token is absent
        assert!(model.get_distillation_output(&images).is_err());
    }

    // ── classification head ───────────────────────────────────────────────────

    #[test]
    fn test_deit_classification_head() {
        let config = mini_config();
        let model =
            DeiTForImageClassification::new(config).expect("model construction should succeed");

        let images = Array4::<f32>::zeros((2, 32, 32, 3));
        let logits = model.forward(&images).expect("classification forward should succeed");
        assert_eq!(logits.shape(), &[2, 10]);
    }

    #[test]
    fn test_deit_cls_only_head() {
        let config = mini_config();
        let model =
            DeiTForImageClassification::new(config).expect("model construction should succeed");

        let images = Array4::<f32>::zeros((1, 32, 32, 3));
        let logits = model.forward_cls_only(&images).expect("cls-only forward should succeed");
        assert_eq!(logits.shape(), &[1, 10]);
    }

    #[test]
    fn test_deit_weight_loading_structure() {
        let config = mini_config();
        let model =
            DeiTForImageClassification::new(config).expect("model construction should succeed");
        let weight_map = model.weight_map();

        // Verify that key HF weight names are present
        assert!(weight_map
            .contains_key("deit.embeddings.patch_embeddings.projection.weight"));
        assert!(weight_map.contains_key("deit.embeddings.cls_token"));
        assert!(weight_map.contains_key("deit.embeddings.distillation_token"));
        assert!(weight_map.contains_key("deit.embeddings.position_embeddings"));
        assert!(weight_map.contains_key("deit.layernorm.weight"));
        assert!(weight_map.contains_key("classifier.weight"));
        assert!(weight_map.contains_key("dist_head.weight"));

        // One entry per layer × 7 keys (q/k/v weight, ln_before, ln_after, intermediate, output)
        let layer_keys: Vec<_> = weight_map
            .keys()
            .filter(|k| k.starts_with("deit.encoder.layer."))
            .collect();
        assert_eq!(layer_keys.len(), 2 * 7); // 2 layers × 7 named tensors
    }
}
