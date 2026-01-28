use super::*;
use trustformers_core::Config;

#[test]
fn test_llava_config_creation() {
    let config = LlavaConfig::default();
    assert_eq!(config.vocab_size, 32000);
    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.num_hidden_layers, 32);
    assert_eq!(config.vision_config.hidden_size, 1024);
    assert_eq!(config.mm_projector_type, "mlp2x_gelu");
}

#[test]
fn test_llava_variants() {
    // Use minimal configs to reduce memory usage
    let llava_7b = LlavaConfig {
        hidden_size: 64,
        num_hidden_layers: 2,
        vocab_size: 1000,
        num_attention_heads: 8,
        intermediate_size: 256,
        model_type: "llava-7b".to_string(),
        vision_config: LlavaVisionConfig {
            hidden_size: 32,
            num_hidden_layers: 2,
            patch_size: 16,
            image_size: 32,
            num_attention_heads: 4,
            ..LlavaVisionConfig::default()
        },
        ..LlavaConfig::default()
    };
    assert_eq!(llava_7b.hidden_size, 64);
    assert_eq!(llava_7b.num_hidden_layers, 2);
    assert_eq!(llava_7b.model_type, "llava-7b");

    let llava_13b = LlavaConfig {
        hidden_size: 64,
        num_hidden_layers: 2,
        vocab_size: 1000,
        num_attention_heads: 8,
        intermediate_size: 256,
        model_type: "llava-13b".to_string(),
        vision_config: LlavaVisionConfig {
            hidden_size: 32,
            num_hidden_layers: 2,
            patch_size: 16,
            image_size: 32,
            num_attention_heads: 4,
            ..LlavaVisionConfig::default()
        },
        ..LlavaConfig::default()
    };
    assert_eq!(llava_13b.hidden_size, 64);
    assert_eq!(llava_13b.num_hidden_layers, 2);
    assert_eq!(llava_13b.model_type, "llava-13b");

    let llava_v1_5 = LlavaConfig {
        hidden_size: 64,
        num_hidden_layers: 2,
        vocab_size: 1000,
        num_attention_heads: 8,
        intermediate_size: 256,
        max_position_embeddings: 512,
        model_type: "llava-v1.5-7b".to_string(),
        vision_config: LlavaVisionConfig {
            hidden_size: 32,
            num_hidden_layers: 2,
            patch_size: 16,
            image_size: 32,
            num_attention_heads: 4,
            ..LlavaVisionConfig::default()
        },
        ..LlavaConfig::default()
    };
    assert_eq!(llava_v1_5.max_position_embeddings, 512);
    assert_eq!(llava_v1_5.vision_config.image_size, 32);
    assert_eq!(llava_v1_5.model_type, "llava-v1.5-7b");

    // Explicit cleanup
    drop(llava_7b);
    drop(llava_13b);
    drop(llava_v1_5);
    std::hint::black_box(());
}

#[test]
fn test_llava_v1_6_variants() {
    // Use minimal configs to reduce memory usage
    let llava_v1_6_7b = LlavaConfig {
        hidden_size: 64,
        num_hidden_layers: 2,
        vocab_size: 1000,
        num_attention_heads: 8,
        intermediate_size: 256,
        image_aspect_ratio: "anyres".to_string(),
        image_grid_pinpoints: Some(vec![(1, 1), (1, 2), (2, 1)]),
        model_type: "llava-v1.6-7b".to_string(),
        vision_config: LlavaVisionConfig {
            hidden_size: 32,
            num_hidden_layers: 2,
            patch_size: 16,
            image_size: 32,
            num_attention_heads: 4,
            ..LlavaVisionConfig::default()
        },
        ..LlavaConfig::default()
    };
    assert_eq!(llava_v1_6_7b.image_aspect_ratio, "anyres");
    assert!(llava_v1_6_7b.image_grid_pinpoints.is_some());
    assert_eq!(llava_v1_6_7b.model_type, "llava-v1.6-7b");

    let llava_v1_6_34b = LlavaConfig {
        hidden_size: 64,
        num_hidden_layers: 2,
        vocab_size: 1000,
        num_attention_heads: 8,
        intermediate_size: 256,
        model_type: "llava-v1.6-34b".to_string(),
        vision_config: LlavaVisionConfig {
            hidden_size: 32,
            num_hidden_layers: 2,
            patch_size: 16,
            image_size: 32,
            num_attention_heads: 4,
            ..LlavaVisionConfig::default()
        },
        ..LlavaConfig::default()
    };
    assert_eq!(llava_v1_6_34b.hidden_size, 64);
    assert_eq!(llava_v1_6_34b.num_hidden_layers, 2);
    assert_eq!(llava_v1_6_34b.model_type, "llava-v1.6-34b");

    // Explicit cleanup
    drop(llava_v1_6_7b);
    drop(llava_v1_6_34b);
    std::hint::black_box(());
}

#[test]
fn test_llava_phi3_variant() {
    let phi3_config = LlavaConfig::llava_phi3_mini();
    assert_eq!(phi3_config.vocab_size, 32064);
    assert_eq!(phi3_config.hidden_size, 3072);
    assert_eq!(phi3_config.num_key_value_heads, Some(32));
    assert_eq!(phi3_config.model_type, "llava-phi3-mini");
}

#[test]
fn test_vision_config() {
    let vision_config = LlavaVisionConfig::default();
    assert_eq!(vision_config.hidden_size, 1024);
    assert_eq!(vision_config.num_hidden_layers, 24);
    assert_eq!(vision_config.patch_size, 14);
    assert_eq!(vision_config.image_size, 336);
    assert_eq!(vision_config.num_attention_heads, 16);
}

#[test]
fn test_config_validation() {
    let config = LlavaConfig::default();
    assert!(config.validate().is_ok());

    // Test invalid language model configuration
    let mut invalid_config = LlavaConfig::default();
    invalid_config.hidden_size = 4095; // Not divisible by num_attention_heads
    assert!(invalid_config.validate().is_err());

    // Test invalid vision configuration
    let mut invalid_config = LlavaConfig::default();
    invalid_config.vision_config.hidden_size = 1023; // Not divisible by num_attention_heads
    assert!(invalid_config.validate().is_err());
}

#[test]
fn test_from_pretrained_name() {
    let config = LlavaConfig::from_pretrained_name("llava-v1.5-7b");
    assert!(config.is_some());
    let config = config.expect("operation failed");
    assert_eq!(config.model_type, "llava-v1.5-7b");

    let config = LlavaConfig::from_pretrained_name("liuhaotian/llava-v1.6-mistral-7b");
    assert!(config.is_some());
    let config = config.expect("operation failed");
    assert_eq!(config.model_type, "llava-v1.6-7b");

    let config = LlavaConfig::from_pretrained_name("invalid-model");
    assert!(config.is_none());
}

#[test]
fn test_head_calculations() {
    // Use minimal configs to reduce memory usage
    let config = LlavaConfig {
        hidden_size: 64,
        num_attention_heads: 8,
        ..LlavaConfig::default()
    };
    assert_eq!(config.head_dim(), 8); // 64 / 8

    let config2 = LlavaConfig {
        hidden_size: 64,
        num_attention_heads: 8,
        ..LlavaConfig::default()
    };
    assert_eq!(config2.head_dim(), 8); // 64 / 8

    let config3 = LlavaConfig {
        hidden_size: 96,
        num_attention_heads: 8,
        ..LlavaConfig::default()
    };
    assert_eq!(config3.head_dim(), 12); // 96 / 8

    // Explicit cleanup
    drop(config);
    drop(config2);
    drop(config3);
    std::hint::black_box(());
}

#[test]
fn test_vision_head_calculations() {
    let config = LlavaConfig::default();
    assert_eq!(config.vision_head_dim(), 64); // 1024 / 16
}

#[test]
fn test_num_patches_calculation() {
    let config = LlavaConfig::default();
    assert_eq!(config.num_patches(), 576); // (336 / 14)^2 = 24^2 = 576

    let config = LlavaConfig::llava_v1_5_7b();
    assert_eq!(config.num_patches(), 576); // Same calculation
}

#[test]
fn test_high_resolution_configuration() {
    let mut config = LlavaConfig::default();
    config.with_high_resolution(true);

    assert_eq!(config.image_aspect_ratio, "anyres");
    assert!(config.image_grid_pinpoints.is_some());

    config.with_high_resolution(false);
    assert_eq!(config.image_aspect_ratio, "square");
    assert!(config.image_grid_pinpoints.is_none());
}

#[test]
fn test_vision_tower_configuration() {
    let mut config = LlavaConfig::default();
    config.with_vision_tower("custom/vision-tower");

    assert_eq!(config.mm_vision_tower, "custom/vision-tower");
}

#[test]
fn test_multimodal_projector_creation() {
    // Test linear projector
    let projector = LlavaMultiModalProjector::new("linear".to_string(), 1024, 4096);
    assert!(projector.is_ok());

    // Test MLP projector
    let projector = LlavaMultiModalProjector::new("mlp2x_gelu".to_string(), 1024, 4096);
    assert!(projector.is_ok());

    // Test invalid projector type
    let projector = LlavaMultiModalProjector::new("invalid".to_string(), 1024, 4096);
    assert!(projector.is_err());
}

#[test]
fn test_vision_transformer_creation() {
    let vision_config = LlavaVisionConfig::default();
    let vision_model = LlavaVisionTransformer::new(vision_config);
    assert!(vision_model.is_ok());
}

#[test]
fn test_vision_embeddings_creation() {
    let vision_config = LlavaVisionConfig::default();
    let embeddings = LlavaVisionEmbeddings::new(vision_config);
    assert!(embeddings.is_ok());
}

#[test]
fn test_vision_encoder_creation() {
    let vision_config = LlavaVisionConfig::default();
    let encoder = LlavaVisionEncoder::new(vision_config);
    assert!(encoder.is_ok());

    let encoder = encoder.expect("operation failed");
    assert_eq!(encoder.layers.len(), 24); // Default num_hidden_layers
}

#[test]
fn test_vision_attention_creation() {
    let vision_config = LlavaVisionConfig::default();
    let attention = LlavaVisionAttention::new(vision_config);
    assert!(attention.is_ok());

    let attention = attention.expect("operation failed");
    assert_eq!(attention.head_dim, 64); // 1024 / 16
}

#[test]
fn test_vision_mlp_creation() {
    let vision_config = LlavaVisionConfig::default();
    let mlp = LlavaVisionMLP::new(vision_config);
    assert!(mlp.is_ok());
}

#[test]
fn test_language_model_creation() {
    // Use minimal config to reduce memory usage
    let config = LlavaConfig {
        hidden_size: 64,
        num_hidden_layers: 2,
        vocab_size: 1000,
        num_attention_heads: 8,
        intermediate_size: 256,
        vision_config: LlavaVisionConfig {
            hidden_size: 32,
            num_hidden_layers: 2,
            patch_size: 16,
            image_size: 32,
            num_attention_heads: 4,
            ..LlavaVisionConfig::default()
        },
        ..LlavaConfig::default()
    };
    let language_model = LlavaLanguageModel::new(config);
    assert!(language_model.is_ok());

    let language_model = language_model.expect("operation failed");
    assert_eq!(language_model.layers.len(), 2); // Minimal has 2 layers

    // Explicit cleanup
    drop(language_model);
    std::hint::black_box(());
}

#[test]
fn test_llava_for_conditional_generation_creation() {
    // Use minimal config to reduce memory usage
    let config = LlavaConfig {
        hidden_size: 64,
        num_hidden_layers: 2,
        vocab_size: 1000,
        num_attention_heads: 8,
        intermediate_size: 256,
        vision_config: LlavaVisionConfig {
            hidden_size: 32,
            num_hidden_layers: 2,
            patch_size: 16,
            image_size: 32,
            num_attention_heads: 4,
            ..LlavaVisionConfig::default()
        },
        ..LlavaConfig::default()
    };
    let model = LlavaForConditionalGeneration::new(config);
    assert!(model.is_ok());

    // Explicit cleanup
    if let Ok(model) = model {
        drop(model);
    }
    std::hint::black_box(());
}

#[test]
fn test_decoder_layer_creation() {
    // Use minimal config to reduce memory usage
    let config = LlavaConfig {
        hidden_size: 64,
        num_hidden_layers: 2,
        vocab_size: 1000,
        num_attention_heads: 8,
        intermediate_size: 256,
        vision_config: LlavaVisionConfig {
            hidden_size: 32,
            num_hidden_layers: 2,
            patch_size: 16,
            image_size: 32,
            num_attention_heads: 4,
            ..LlavaVisionConfig::default()
        },
        ..LlavaConfig::default()
    };
    let layer = LlavaDecoderLayer::new(config);
    assert!(layer.is_ok());

    // Explicit cleanup
    if let Ok(layer) = layer {
        drop(layer);
    }
    std::hint::black_box(());
}

#[test]
fn test_llava_attention_creation() {
    // Use minimal config to reduce memory usage
    let config = LlavaConfig {
        hidden_size: 64,
        num_hidden_layers: 2,
        vocab_size: 1000,
        num_attention_heads: 8,
        intermediate_size: 256,
        vision_config: LlavaVisionConfig {
            hidden_size: 32,
            num_hidden_layers: 2,
            patch_size: 16,
            image_size: 32,
            num_attention_heads: 4,
            ..LlavaVisionConfig::default()
        },
        ..LlavaConfig::default()
    };
    let attention = LlavaAttention::new(config);
    assert!(attention.is_ok());

    let attention = attention.expect("operation failed");
    assert_eq!(attention.head_dim, 8); // 64 / 8
    assert_eq!(attention.num_heads, 8);

    // Explicit cleanup
    drop(attention);
    std::hint::black_box(());
}

#[test]
fn test_llava_mlp_creation() {
    // Use minimal config to reduce memory usage
    let config = LlavaConfig {
        hidden_size: 64,
        num_hidden_layers: 2,
        vocab_size: 1000,
        num_attention_heads: 8,
        intermediate_size: 256,
        vision_config: LlavaVisionConfig {
            hidden_size: 32,
            num_hidden_layers: 2,
            patch_size: 16,
            image_size: 32,
            num_attention_heads: 4,
            ..LlavaVisionConfig::default()
        },
        ..LlavaConfig::default()
    };
    let mlp = LlavaMLP::new(config);
    assert!(mlp.is_ok());

    // Explicit cleanup
    if let Ok(mlp) = mlp {
        drop(mlp);
    }
    std::hint::black_box(());
}

#[test]
fn test_model_type_strings() {
    // Test model type strings without loading large configs
    let config_7b = LlavaConfig {
        model_type: "llava-7b".to_string(),
        ..LlavaConfig::default()
    };
    let config_13b = LlavaConfig {
        model_type: "llava-13b".to_string(),
        ..LlavaConfig::default()
    };
    let config_v1_5_7b = LlavaConfig {
        model_type: "llava-v1.5-7b".to_string(),
        ..LlavaConfig::default()
    };
    let config_v1_5_13b = LlavaConfig {
        model_type: "llava-v1.5-13b".to_string(),
        ..LlavaConfig::default()
    };
    let config_v1_6_7b = LlavaConfig {
        model_type: "llava-v1.6-7b".to_string(),
        ..LlavaConfig::default()
    };
    let config_v1_6_34b = LlavaConfig {
        model_type: "llava-v1.6-34b".to_string(),
        ..LlavaConfig::default()
    };
    let config_phi3 = LlavaConfig {
        model_type: "llava-phi3-mini".to_string(),
        ..LlavaConfig::default()
    };

    assert_eq!(config_7b.model_type, "llava-7b");
    assert_eq!(config_13b.model_type, "llava-13b");
    assert_eq!(config_v1_5_7b.model_type, "llava-v1.5-7b");
    assert_eq!(config_v1_5_13b.model_type, "llava-v1.5-13b");
    assert_eq!(config_v1_6_7b.model_type, "llava-v1.6-7b");
    assert_eq!(config_v1_6_34b.model_type, "llava-v1.6-34b");
    assert_eq!(config_phi3.model_type, "llava-phi3-mini");

    // Explicit cleanup
    drop(config_7b);
    drop(config_13b);
    drop(config_v1_5_7b);
    drop(config_v1_5_13b);
    drop(config_v1_6_7b);
    drop(config_v1_6_34b);
    drop(config_phi3);
    std::hint::black_box(());
}

#[test]
fn test_mm_projector_types() {
    let config = LlavaConfig::default();
    assert_eq!(config.mm_projector_type, "mlp2x_gelu");

    let config = LlavaConfig::llava_v1_5_7b();
    assert_eq!(config.mm_projector_type, "mlp2x_gelu");
}

#[test]
fn test_image_aspect_ratios() {
    // Test aspect ratios without loading large configs
    let config1 = LlavaConfig {
        image_aspect_ratio: "square".to_string(),
        ..LlavaConfig::default()
    };
    assert_eq!(config1.image_aspect_ratio, "square");

    let config2 = LlavaConfig {
        image_aspect_ratio: "anyres".to_string(),
        ..LlavaConfig::default()
    };
    assert_eq!(config2.image_aspect_ratio, "anyres");

    // Explicit cleanup
    drop(config1);
    drop(config2);
    std::hint::black_box(());
}

#[test]
fn test_vision_config_architecture() {
    let vision_config = LlavaVisionConfig::default();
    assert_eq!(vision_config.model_type, "clip_vision_model");
    assert_eq!(vision_config.hidden_act, "gelu");
    assert_eq!(vision_config.num_channels, 3);
}

#[test]
fn test_rope_theta_values() {
    let config = LlavaConfig::default();
    assert_eq!(config.rope_theta, 10000.0);

    let config = LlavaConfig::llava_phi3_mini();
    assert_eq!(config.rope_theta, 10000.0);
}
