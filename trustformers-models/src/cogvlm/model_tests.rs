#[cfg(test)]
mod tests {
    use crate::cogvlm::config::*;
    use crate::cogvlm::model::*;
    use trustformers_core::device::Device;

    // --- CogVlmConfig tests ---

    #[test]
    fn test_cogvlm_config_default() {
        let config = CogVlmConfig::default();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.num_attention_heads, 32);
    }

    #[test]
    fn test_cogvlm_config_vision_defaults() {
        let config = CogVlmConfig::default();
        assert_eq!(config.vision_config.hidden_size, 1792);
        assert_eq!(config.vision_config.patch_size, 14);
        assert_eq!(config.vision_config.image_size, 490);
    }

    #[test]
    fn test_cogvlm_config_cross_modal() {
        let config = CogVlmConfig::default();
        assert_eq!(config.cross_hidden_size, 4096);
        assert_eq!(config.cogvlm_stage, 2);
    }

    #[test]
    fn test_cogvlm_config_clone() {
        let config = CogVlmConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.vocab_size, config.vocab_size);
        assert_eq!(cloned.hidden_size, config.hidden_size);
    }

    #[test]
    fn test_cogvlm_config_rope_theta() {
        let config = CogVlmConfig::default();
        assert!((config.rope_theta - 10000.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cogvlm_config_no_lora() {
        let config = CogVlmConfig::default();
        assert!(!config.use_lora);
        assert!(config.lora_rank.is_none());
    }

    // --- CogVlmVisionConfig tests ---

    #[test]
    fn test_vision_config_default() {
        let config = CogVlmVisionConfig::default();
        assert_eq!(config.num_channels, 3);
        assert_eq!(config.hidden_act, "gelu");
        assert!(config.use_flash_attn);
    }

    #[test]
    fn test_vision_config_clone() {
        let config = CogVlmVisionConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.hidden_size, config.hidden_size);
        assert_eq!(cloned.patch_size, config.patch_size);
    }

    #[test]
    fn test_vision_config_custom() {
        let config = CogVlmVisionConfig {
            hidden_size: 1024,
            intermediate_size: 4096,
            num_hidden_layers: 12,
            num_attention_heads: 16,
            num_channels: 3,
            patch_size: 16,
            image_size: 336,
            initializer_range: 0.02,
            layer_norm_eps: 1e-6,
            hidden_act: "gelu".to_string(),
            model_type: "clip_vision_model".to_string(),
            attention_dropout: 0.0,
            dropout: 0.0,
            use_flash_attn: false,
        };
        assert_eq!(config.image_size, 336);
        assert_eq!(config.patch_size, 16);
    }

    // --- CogVlmVisionTransformer tests ---

    #[test]
    fn test_vision_transformer_creation() {
        let config = CogVlmVisionConfig {
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_channels: 3,
            patch_size: 4,
            image_size: 16,
            initializer_range: 0.02,
            layer_norm_eps: 1e-6,
            hidden_act: "gelu".to_string(),
            model_type: "test".to_string(),
            attention_dropout: 0.0,
            dropout: 0.0,
            use_flash_attn: false,
        };
        let vit = CogVlmVisionTransformer::new(config);
        assert!(vit.is_ok());
    }

    #[test]
    fn test_vision_transformer_device() {
        let config = CogVlmVisionConfig {
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_channels: 3,
            patch_size: 4,
            image_size: 16,
            initializer_range: 0.02,
            layer_norm_eps: 1e-6,
            hidden_act: "gelu".to_string(),
            model_type: "test".to_string(),
            attention_dropout: 0.0,
            dropout: 0.0,
            use_flash_attn: false,
        };
        if let Ok(vit) = CogVlmVisionTransformer::new(config) {
            assert!(matches!(vit.device(), Device::CPU));
        }
    }

    #[test]
    fn test_vision_transformer_parameter_count() {
        let config = CogVlmVisionConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_channels: 3,
            patch_size: 4,
            image_size: 8,
            initializer_range: 0.02,
            layer_norm_eps: 1e-6,
            hidden_act: "gelu".to_string(),
            model_type: "test".to_string(),
            attention_dropout: 0.0,
            dropout: 0.0,
            use_flash_attn: false,
        };
        if let Ok(vit) = CogVlmVisionTransformer::new(config) {
            assert!(vit.parameter_count() > 0);
        }
    }

    // --- CogVlmVisionEmbeddings tests ---

    #[test]
    fn test_vision_embeddings_creation() {
        let config = CogVlmVisionConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_channels: 3,
            patch_size: 4,
            image_size: 8,
            initializer_range: 0.02,
            layer_norm_eps: 1e-6,
            hidden_act: "gelu".to_string(),
            model_type: "test".to_string(),
            attention_dropout: 0.0,
            dropout: 0.0,
            use_flash_attn: false,
        };
        let embeddings = CogVlmVisionEmbeddings::new(config);
        assert!(embeddings.is_ok());
    }

    // --- RopeScaling tests ---

    #[test]
    fn test_rope_scaling_creation() {
        let scaling = RopeScaling {
            type_: "linear".to_string(),
            factor: 2.0,
        };
        assert_eq!(scaling.type_, "linear");
        assert!((scaling.factor - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_rope_scaling_clone() {
        let scaling = RopeScaling {
            type_: "dynamic".to_string(),
            factor: 4.0,
        };
        let cloned = scaling.clone();
        assert_eq!(cloned.type_, "dynamic");
    }

    // --- CogVideoConfig tests ---

    #[test]
    fn test_cogvideo_config_default() {
        let config = CogVideoConfig::default();
        assert!(config.video_frames > 0);
        assert!(config.frame_stride > 0);
    }

    #[test]
    fn test_cogvideo_config_clone() {
        let config = CogVideoConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.video_frames, config.video_frames);
    }
}
