//! Tests for CLIP model implementation and weight loading

use super::config::{CLIPConfig, CLIPTextConfig, CLIPVisionConfig};
use super::model::{
    CLIPEncoderLayer, CLIPEncoderLayerConfig, CLIPModel, CLIPTextEmbeddings, CLIPVisionEmbeddings,
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
