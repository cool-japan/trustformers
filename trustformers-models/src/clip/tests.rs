//! Tests for CLIP model implementation and weight loading

use super::config::{CLIPConfig, CLIPEncoderConfig, CLIPTextConfig, CLIPVisionConfig};
use super::model::{
    CLIPEncoder, CLIPEncoderLayer, CLIPEncoderLayerConfig, CLIPModel, CLIPTextEmbeddings,
    CLIPVisionEmbeddings,
};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Config, Layer, Model};

#[test]
fn test_clip_config_validation() {
    let text_config = CLIPTextConfig {
        hidden_act: "quick_gelu".to_string(),
        initializer_range: 0.02,
        initializer_factor: 1.0,
        pad_token_id: 1,
        bos_token_id: 49406,
        eos_token_id: 49407,
        vocab_size: 49408,
        hidden_size: 512,
        intermediate_size: 2048,
        num_hidden_layers: 12,
        num_attention_heads: 8,
        max_position_embeddings: 77,
        layer_norm_eps: 1e-5,
        dropout: 0.0,
        attention_dropout: 0.0,
    };

    let vision_config = CLIPVisionConfig {
        hidden_act: "quick_gelu".to_string(),
        initializer_range: 0.02,
        initializer_factor: 1.0,
        hidden_size: 768,
        intermediate_size: 3072,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        num_channels: 3,
        image_size: 224,
        patch_size: 32,
        layer_norm_eps: 1e-5,
        dropout: 0.0,
        attention_dropout: 0.0,
    };

    let config = CLIPConfig {
        initializer_range: 0.02,
        initializer_factor: 1.0,
        text_config,
        vision_config,
        projection_dim: 512,
        logit_scale_init_value: 2.6592,
    };

    assert!(config.validate().is_ok());
}

#[test]
fn test_clip_model_creation() {
    let text_config = CLIPTextConfig {
        hidden_act: "quick_gelu".to_string(),
        initializer_range: 0.02,
        initializer_factor: 1.0,
        pad_token_id: 1,
        bos_token_id: 49406,
        eos_token_id: 49407,
        vocab_size: 1000, // Smaller for testing
        hidden_size: 128,
        intermediate_size: 256,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        max_position_embeddings: 77,
        layer_norm_eps: 1e-5,
        dropout: 0.0,
        attention_dropout: 0.0,
    };

    let vision_config = CLIPVisionConfig {
        hidden_act: "quick_gelu".to_string(),
        initializer_range: 0.02,
        initializer_factor: 1.0,
        hidden_size: 128,
        intermediate_size: 256,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_channels: 3,
        image_size: 32, // Small for testing
        patch_size: 16,
        layer_norm_eps: 1e-5,
        dropout: 0.0,
        attention_dropout: 0.0,
    };

    let config = CLIPConfig {
        initializer_range: 0.02,
        initializer_factor: 1.0,
        text_config,
        vision_config,
        projection_dim: 64,
        logit_scale_init_value: 2.6592,
    };

    let model = CLIPModel::new(config);
    assert!(model.is_ok());
}

#[test]
fn test_clip_text_embeddings() {
    let config = CLIPTextConfig {
        vocab_size: 1000,
        hidden_size: 128,
        intermediate_size: 256,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        max_position_embeddings: 77,
        hidden_act: "quick_gelu".to_string(),
        layer_norm_eps: 1e-5,
        dropout: 0.0,
        attention_dropout: 0.0,
        initializer_range: 0.02,
        initializer_factor: 1.0,
        pad_token_id: 1,
        bos_token_id: 49406,
        eos_token_id: 49407,
    };

    let embeddings = CLIPTextEmbeddings::new(&config);
    assert!(embeddings.is_ok());

    let embeddings = embeddings.expect("operation failed");

    // Test forward pass with token IDs
    let input_ids = vec![1, 2, 3, 4, 5];
    let output = embeddings.forward(input_ids);
    assert!(output.is_ok());

    let output_tensor = output.expect("operation failed");
    // Expected shape: [seq_len, hidden_size] = [5, 128]
    assert_eq!(output_tensor.shape(), &[5, 128]);
}

#[test]
fn test_clip_vision_embeddings() {
    let config = CLIPVisionConfig {
        hidden_size: 128,
        intermediate_size: 256,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_channels: 3,
        image_size: 32,
        patch_size: 16,
        hidden_act: "quick_gelu".to_string(),
        layer_norm_eps: 1e-5,
        dropout: 0.0,
        attention_dropout: 0.0,
        initializer_range: 0.02,
        initializer_factor: 1.0,
    };

    let embeddings = CLIPVisionEmbeddings::new(&config);
    assert!(embeddings.is_ok());
}

#[test]
fn test_clip_encoder_layer() {
    let layer_config = CLIPEncoderLayerConfig {
        hidden_size: 128,
        num_attention_heads: 4,
        intermediate_size: 256,
        hidden_act: "gelu".to_string(),
        layer_norm_eps: 1e-5,
        attention_dropout: 0.0,
        dropout: 0.0,
    };

    let layer = CLIPEncoderLayer::new(&layer_config);
    assert!(layer.is_ok());

    let layer = layer.expect("operation failed");

    // Test forward pass
    let input = Tensor::randn(&[2, 10, 128]).expect("operation failed"); // [batch, seq_len, hidden_size]
    let output = layer.forward(input);
    assert!(output.is_ok());

    let output_tensor = output.expect("operation failed");
    assert_eq!(output_tensor.shape(), &[2, 10, 128]);
}

#[test]
fn test_clip_parameter_count() {
    let text_config = CLIPTextConfig {
        hidden_act: "quick_gelu".to_string(),
        initializer_range: 0.02,
        initializer_factor: 1.0,
        pad_token_id: 1,
        bos_token_id: 49406,
        eos_token_id: 49407,
        vocab_size: 1000,
        hidden_size: 128,
        intermediate_size: 256,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        max_position_embeddings: 77,
        layer_norm_eps: 1e-5,
        dropout: 0.0,
        attention_dropout: 0.0,
    };

    let vision_config = CLIPVisionConfig {
        hidden_act: "quick_gelu".to_string(),
        initializer_range: 0.02,
        initializer_factor: 1.0,
        hidden_size: 128,
        intermediate_size: 256,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_channels: 3,
        image_size: 32,
        patch_size: 16,
        layer_norm_eps: 1e-5,
        dropout: 0.0,
        attention_dropout: 0.0,
    };

    let config = CLIPConfig {
        initializer_range: 0.02,
        initializer_factor: 1.0,
        text_config,
        vision_config,
        projection_dim: 64,
        logit_scale_init_value: 2.6592,
    };

    let model = CLIPModel::new(config).expect("operation failed");
    let param_count = model.num_parameters();

    // Should have non-zero parameters
    assert!(param_count > 0);

    // Parameter count should be reasonable for this small test model
    // (much smaller than full CLIP which has hundreds of millions)
    assert!(param_count < 10_000_000);
}

#[test]
fn test_clip_weight_loading_methods_exist() {
    // Test that weight loading methods compile and are accessible
    let text_config = CLIPTextConfig {
        hidden_act: "quick_gelu".to_string(),
        initializer_range: 0.02,
        initializer_factor: 1.0,
        pad_token_id: 1,
        bos_token_id: 49406,
        eos_token_id: 49407,
        vocab_size: 1000,
        hidden_size: 128,
        intermediate_size: 256,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        max_position_embeddings: 77,
        layer_norm_eps: 1e-5,
        dropout: 0.0,
        attention_dropout: 0.0,
    };

    let vision_config = CLIPVisionConfig {
        hidden_act: "quick_gelu".to_string(),
        initializer_range: 0.02,
        initializer_factor: 1.0,
        hidden_size: 128,
        intermediate_size: 256,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_channels: 3,
        image_size: 32,
        patch_size: 16,
        layer_norm_eps: 1e-5,
        dropout: 0.0,
        attention_dropout: 0.0,
    };

    let config = CLIPConfig {
        initializer_range: 0.02,
        initializer_factor: 1.0,
        text_config,
        vision_config,
        projection_dim: 64,
        logit_scale_init_value: 2.6592,
    };

    let mut model = CLIPModel::new(config).expect("operation failed");

    // Test that load_from_path method exists (it will fail without actual weights, but should compile)
    let result = model.load_from_path("/nonexistent/path");
    assert!(result.is_err()); // Expected to fail with nonexistent path
}

#[test]
fn test_clip_logit_scale() {
    let text_config = CLIPTextConfig {
        hidden_act: "quick_gelu".to_string(),
        initializer_range: 0.02,
        initializer_factor: 1.0,
        pad_token_id: 1,
        bos_token_id: 49406,
        eos_token_id: 49407,
        vocab_size: 1000,
        hidden_size: 128,
        intermediate_size: 256,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        max_position_embeddings: 77,
        layer_norm_eps: 1e-5,
        dropout: 0.0,
        attention_dropout: 0.0,
    };

    let vision_config = CLIPVisionConfig {
        hidden_act: "quick_gelu".to_string(),
        initializer_range: 0.02,
        initializer_factor: 1.0,
        hidden_size: 128,
        intermediate_size: 256,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_channels: 3,
        image_size: 32,
        patch_size: 16,
        layer_norm_eps: 1e-5,
        dropout: 0.0,
        attention_dropout: 0.0,
    };

    let config = CLIPConfig {
        initializer_range: 0.02,
        initializer_factor: 1.0,
        text_config,
        vision_config,
        projection_dim: 64,
        logit_scale_init_value: 2.6592,
    };

    let model = CLIPModel::new(config).expect("operation failed");

    // Verify logit_scale is initialized correctly
    match &model.logit_scale {
        Tensor::F32(arr) => {
            assert_eq!(arr.len(), 1);
            let value = arr.iter().next().expect("operation failed");
            assert!((value - 2.6592).abs() < 1e-6);
        },
        _ => panic!("Expected F32 tensor for logit_scale"),
    }
}

#[test]
fn test_clip_encoder_config_trait_text() {
    let config = CLIPTextConfig {
        vocab_size: 1000,
        hidden_size: 256,
        intermediate_size: 512,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        max_position_embeddings: 77,
        hidden_act: "quick_gelu".to_string(),
        layer_norm_eps: 1e-5,
        dropout: 0.1,
        attention_dropout: 0.05,
        initializer_range: 0.02,
        initializer_factor: 1.0,
        pad_token_id: 1,
        bos_token_id: 49406,
        eos_token_id: 49407,
    };

    assert_eq!(config.hidden_size(), 256);
    assert_eq!(config.num_attention_heads(), 4);
    assert_eq!(config.intermediate_size(), 512);
    assert_eq!(config.num_hidden_layers(), 4);
    assert_eq!(config.hidden_act(), "quick_gelu");
    assert!((config.layer_norm_eps() - 1e-5).abs() < 1e-10);
    assert!((config.attention_dropout() - 0.05).abs() < 1e-10);
    assert!((config.dropout() - 0.1).abs() < 1e-10);
}

#[test]
fn test_clip_encoder_config_trait_vision() {
    let config = CLIPVisionConfig {
        hidden_size: 768,
        intermediate_size: 3072,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        num_channels: 3,
        image_size: 224,
        patch_size: 32,
        hidden_act: "quick_gelu".to_string(),
        layer_norm_eps: 1e-5,
        dropout: 0.0,
        attention_dropout: 0.0,
        initializer_range: 0.02,
        initializer_factor: 1.0,
    };

    assert_eq!(config.hidden_size(), 768);
    assert_eq!(config.num_hidden_layers(), 12);
    assert_eq!(config.num_attention_heads(), 12);
    assert_eq!(config.intermediate_size(), 3072);
}

#[test]
fn test_clip_encoder_uses_config_values() {
    // Create a text config with non-default values to prove they are read
    let text_config = CLIPTextConfig {
        vocab_size: 500,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 3,
        num_attention_heads: 2,
        max_position_embeddings: 16,
        hidden_act: "gelu".to_string(),
        layer_norm_eps: 1e-6,
        dropout: 0.0,
        attention_dropout: 0.0,
        initializer_range: 0.02,
        initializer_factor: 1.0,
        pad_token_id: 1,
        bos_token_id: 49406,
        eos_token_id: 49407,
    };

    let encoder = CLIPEncoder::<CLIPTextConfig>::new(&text_config);
    assert!(encoder.is_ok());
    let encoder = encoder.expect("operation failed");

    // The encoder must have exactly num_hidden_layers layers
    assert_eq!(encoder.layers.len(), 3);

    // Forward pass with matching hidden_size should succeed
    let input = Tensor::randn(&[1, 5, 64]).expect("operation failed");
    let output = encoder.forward(input);
    assert!(output.is_ok());
    let output = output.expect("operation failed");
    assert_eq!(output.shape(), &[1, 5, 64]);
}

#[test]
fn test_clip_encoder_vision_config_values() {
    let vision_config = CLIPVisionConfig {
        hidden_size: 96,
        intermediate_size: 192,
        num_hidden_layers: 2,
        num_attention_heads: 3,
        num_channels: 3,
        image_size: 32,
        patch_size: 16,
        hidden_act: "quick_gelu".to_string(),
        layer_norm_eps: 1e-5,
        dropout: 0.0,
        attention_dropout: 0.0,
        initializer_range: 0.02,
        initializer_factor: 1.0,
    };

    let encoder = CLIPEncoder::<CLIPVisionConfig>::new(&vision_config);
    assert!(encoder.is_ok());
    let encoder = encoder.expect("operation failed");
    assert_eq!(encoder.layers.len(), 2);

    let input = Tensor::randn(&[1, 5, 96]).expect("operation failed");
    let output = encoder.forward(input);
    assert!(output.is_ok());
    assert_eq!(output.expect("operation failed").shape(), &[1, 5, 96]);
}

#[test]
fn test_clip_load_weights_chunked_nonexistent_path() {
    let text_config = CLIPTextConfig {
        hidden_act: "quick_gelu".to_string(),
        initializer_range: 0.02,
        initializer_factor: 1.0,
        pad_token_id: 1,
        bos_token_id: 49406,
        eos_token_id: 49407,
        vocab_size: 1000,
        hidden_size: 128,
        intermediate_size: 256,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        max_position_embeddings: 77,
        layer_norm_eps: 1e-5,
        dropout: 0.0,
        attention_dropout: 0.0,
    };

    let vision_config = CLIPVisionConfig {
        hidden_act: "quick_gelu".to_string(),
        initializer_range: 0.02,
        initializer_factor: 1.0,
        hidden_size: 128,
        intermediate_size: 256,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_channels: 3,
        image_size: 32,
        patch_size: 16,
        layer_norm_eps: 1e-5,
        dropout: 0.0,
        attention_dropout: 0.0,
    };

    let config = CLIPConfig {
        initializer_range: 0.02,
        initializer_factor: 1.0,
        text_config,
        vision_config,
        projection_dim: 64,
        logit_scale_init_value: 2.6592,
    };

    let mut model = CLIPModel::new(config).expect("operation failed");

    let mut progress_log: Vec<(usize, usize, String)> = Vec::new();
    let result = model.load_weights_chunked("/nonexistent/path", |idx, total, desc| {
        progress_log.push((idx, total, desc.to_string()));
    });

    // Should fail because path does not exist
    assert!(result.is_err());
}

#[test]
fn test_clip_load_weights_chunked_empty_dir() {
    let tmp_dir = std::env::temp_dir().join("trustformers_clip_chunked_test");
    let _ = std::fs::create_dir_all(&tmp_dir);

    let text_config = CLIPTextConfig {
        hidden_act: "quick_gelu".to_string(),
        initializer_range: 0.02,
        initializer_factor: 1.0,
        pad_token_id: 1,
        bos_token_id: 49406,
        eos_token_id: 49407,
        vocab_size: 500,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        max_position_embeddings: 16,
        layer_norm_eps: 1e-5,
        dropout: 0.0,
        attention_dropout: 0.0,
    };

    let vision_config = CLIPVisionConfig {
        hidden_act: "quick_gelu".to_string(),
        initializer_range: 0.02,
        initializer_factor: 1.0,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_channels: 3,
        image_size: 32,
        patch_size: 16,
        layer_norm_eps: 1e-5,
        dropout: 0.0,
        attention_dropout: 0.0,
    };

    let config = CLIPConfig {
        initializer_range: 0.02,
        initializer_factor: 1.0,
        text_config,
        vision_config,
        projection_dim: 32,
        logit_scale_init_value: 2.6592,
    };

    let mut model = CLIPModel::new(config).expect("operation failed");

    let mut progress_calls = 0usize;
    let result = model.load_weights_chunked(&tmp_dir, |_idx, _total, _desc| {
        progress_calls += 1;
    });

    // auto_create_loader on an empty dir will likely fail; that is fine.
    // The important thing is no panic.
    assert!(result.is_err());

    // Clean up
    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn test_clip_model_respects_custom_layer_count() {
    // Ensure the full CLIPModel respects num_hidden_layers from config
    let text_config = CLIPTextConfig {
        hidden_act: "quick_gelu".to_string(),
        initializer_range: 0.02,
        initializer_factor: 1.0,
        pad_token_id: 1,
        bos_token_id: 49406,
        eos_token_id: 49407,
        vocab_size: 500,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 3,
        num_attention_heads: 2,
        max_position_embeddings: 16,
        layer_norm_eps: 1e-5,
        dropout: 0.0,
        attention_dropout: 0.0,
    };

    let vision_config = CLIPVisionConfig {
        hidden_act: "quick_gelu".to_string(),
        initializer_range: 0.02,
        initializer_factor: 1.0,
        hidden_size: 96,
        intermediate_size: 192,
        num_hidden_layers: 5,
        num_attention_heads: 3,
        num_channels: 3,
        image_size: 48,
        patch_size: 16,
        layer_norm_eps: 1e-5,
        dropout: 0.0,
        attention_dropout: 0.0,
    };

    let config = CLIPConfig {
        initializer_range: 0.02,
        initializer_factor: 1.0,
        text_config,
        vision_config,
        projection_dim: 32,
        logit_scale_init_value: 2.6592,
    };

    let model = CLIPModel::new(config).expect("operation failed");

    // text encoder should have 3 layers, vision encoder should have 5
    assert_eq!(model.text_model.encoder.layers.len(), 3);
    assert_eq!(model.vision_model.encoder.layers.len(), 5);
}
