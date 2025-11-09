/// Flamingo model configurations
pub mod config;

/// Flamingo model implementations
pub mod model;

// Re-export main types for convenience
pub use config::{
    FlamingoConfig, FlamingoLanguageConfig, FlamingoPerceiverConfig, FlamingoVisionConfig,
    FlamingoXAttentionConfig,
};

pub use model::{
    FlamingoLanguageLayer, FlamingoLanguageLayerOutput, FlamingoLanguageModel,
    FlamingoLanguageOutput, FlamingoMLP, FlamingoModel, FlamingoOutput, FlamingoVisionEncoder,
    FlamingoVisionLayer, GatedCrossAttention, GatedCrossAttentionOutput, PerceiverLayer,
    PerceiverResampler,
};

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::{Tensor, TensorType};

    #[test]
    fn test_flamingo_module_imports() {
        // Test that all main types can be imported
        let _config = FlamingoConfig::default();
        let _vision_config = FlamingoVisionConfig::default();
        let _language_config = FlamingoLanguageConfig::default();
        let _perceiver_config = FlamingoPerceiverConfig::default();
        let _x_attention_config = FlamingoXAttentionConfig::default();

        // Test model creation
        let config = FlamingoConfig::flamingo_3b();
        let model = FlamingoModel::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_flamingo_3b_end_to_end() {
        let config = FlamingoConfig::flamingo_3b();
        let model = FlamingoModel::new(config.clone()).unwrap();

        // Create sample inputs for few-shot learning scenario
        let batch_size = 1;
        let seq_len = 50; // Longer sequence for few-shot examples
        let input_ids = Tensor::randint(
            0,
            config.language_config.vocab_size as i64,
            &[batch_size, seq_len],
            TensorType::I64,
        )
        .unwrap();
        let attention_mask = Tensor::ones(&[batch_size, seq_len]).unwrap();
        let pixel_values = Tensor::randn(&[batch_size, 3, 224, 224]).unwrap();

        // Create media locations mask (simulate interleaved text-image input)
        let mut media_locations = Tensor::zeros(&[batch_size, seq_len]).unwrap();
        // Mark positions 5-9, 20-24, 35-39 as media locations
        for &start in &[5, 20, 35] {
            for i in 0..5 {
                if start + i < seq_len {
                    media_locations = media_locations.set_scalar(&[0, start + i], 1.0).unwrap();
                }
            }
        }

        // Test training forward pass
        let train_output = model.forward_train(
            &input_ids,
            &attention_mask,
            Some(&pixel_values),
            Some(&media_locations),
            None,
        );
        assert!(train_output.is_ok());

        let train_output = train_output.unwrap();

        // Verify outputs have correct shapes
        assert_eq!(
            train_output.logits.shape(),
            &[batch_size, seq_len, config.language_config.vocab_size]
        );
        assert!(train_output.vision_features.is_some());
        assert!(!train_output.cross_attention_weights.is_empty());

        let vision_features = train_output.vision_features.unwrap();
        assert_eq!(vision_features.shape()[0], batch_size);
        assert_eq!(vision_features.shape()[1], config.media_token_length);
        assert_eq!(vision_features.shape()[2], config.vision_language_dim);

        // Test generation
        let generated = model.generate_with_shots(
            &input_ids,
            &attention_mask,
            Some(&pixel_values),
            Some(&media_locations),
            10,    // max_new_tokens
            1.0,   // temperature
            false, // do_sample (greedy)
        );

        assert!(generated.is_ok());
        let generated = generated.unwrap();
        assert_eq!(generated.shape()[0], batch_size);
        assert!(generated.shape()[1] > seq_len); // Should have generated new tokens
        assert!(generated.shape()[1] <= seq_len + 10); // Should not exceed max_new_tokens
    }

    #[test]
    fn test_flamingo_9b_configuration() {
        let config = FlamingoConfig::flamingo_9b();
        let model = FlamingoModel::new(config.clone()).unwrap();

        // Verify configuration parameters for 9B model
        assert_eq!(config.language_config.vocab_size, 32000);
        assert_eq!(config.language_config.hidden_size, 4096);
        assert_eq!(config.language_config.num_hidden_layers, 32);
        assert_eq!(config.vision_language_dim, 4096);
        assert_eq!(config.perceiver_config.latent_dim, 4096);
        assert_eq!(config.cross_attention_config.cross_attention_dim, 4096);
        assert_eq!(config.num_shots, 8);
        assert_eq!(config.max_seq_length, 4096);

        // Test with larger inputs
        let batch_size = 1;
        let seq_len = 32;
        let _input_ids = Tensor::randint(
            0,
            config.language_config.vocab_size as i64,
            &[batch_size, seq_len],
            TensorType::I64,
        )
        .unwrap();
        let _attention_mask = Tensor::ones(&[batch_size, seq_len]).unwrap();
        let pixel_values = Tensor::randn(&[batch_size, 3, 224, 224]).unwrap();

        // Test vision encoding
        let vision_features = model.encode_vision(&pixel_values);
        assert!(vision_features.is_ok());

        let vision_features = vision_features.unwrap();
        assert_eq!(vision_features.shape()[0], batch_size);
        assert_eq!(vision_features.shape()[1], config.media_token_length);
        assert_eq!(vision_features.shape()[2], config.vision_language_dim);
    }

    #[test]
    fn test_flamingo_open_source_variant() {
        let config = FlamingoConfig::open_flamingo();

        // Verify OpenFlamingo specific configurations
        assert_eq!(config.language_config.vocab_size, 50432); // MPT vocab
        assert_eq!(config.language_config.hidden_size, 4096);
        assert_eq!(config.language_config.hidden_act, "gelu");
        assert_eq!(config.vision_language_dim, 4096);
        assert_eq!(config.num_shots, 8);

        // Test model creation
        let model = FlamingoModel::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_flamingo_components_separately() {
        let config = FlamingoConfig::flamingo_3b();

        // Test vision encoder
        let vision_encoder = FlamingoVisionEncoder::new(config.vision_config.clone());
        assert!(vision_encoder.is_ok());

        // Test language model
        let language_model = FlamingoLanguageModel::new(
            config.language_config.clone(),
            config.cross_attention_config.clone(),
            config.cross_attention_layers.clone(),
        );
        assert!(language_model.is_ok());

        // Test perceiver resampler
        let perceiver = PerceiverResampler::new(
            config.perceiver_config.clone(),
            config.vision_config.hidden_size,
        );
        assert!(perceiver.is_ok());

        // Test gated cross-attention
        let cross_attn = GatedCrossAttention::new(
            config.language_config.hidden_size,
            config.cross_attention_config.clone(),
        );
        assert!(cross_attn.is_ok());
    }

    #[test]
    fn test_flamingo_cross_attention_layers() {
        let config = FlamingoConfig::flamingo_9b();
        let model = FlamingoModel::new(config.clone()).unwrap();

        // Verify cross-attention layers are correctly configured
        assert_eq!(
            model.language_model.cross_attention_layers,
            config.cross_attention_layers
        );

        // Verify that all cross-attention layer indices are valid
        for &layer_idx in &config.cross_attention_layers {
            assert!(layer_idx < config.language_config.num_hidden_layers);
        }

        // Check that layers have cross-attention based on the configuration
        for (i, layer) in model.language_model.layers.iter().enumerate() {
            if config.cross_attention_layers.contains(&i) {
                assert!(layer.cross_attention.is_some());
                assert!(layer.layer_norm3.is_some());
            } else {
                assert!(layer.cross_attention.is_none());
                assert!(layer.layer_norm3.is_none());
            }
        }
    }

    #[test]
    fn test_flamingo_perceiver_functionality() {
        let config = FlamingoPerceiverConfig::large();
        let input_dim = 1024;
        let perceiver = PerceiverResampler::new(config.clone(), input_dim).unwrap();

        let batch_size = 2;
        let input_seq_len = 257; // ViT sequence length
        let vision_features = Tensor::randn(&[batch_size, input_seq_len, input_dim]).unwrap();

        let output = perceiver.forward(&vision_features).unwrap();

        // Verify perceiver reduces sequence length to fixed number of latents
        assert_eq!(
            output.shape(),
            &[batch_size, config.num_latents, config.latent_dim]
        );
        assert!(config.num_latents < input_seq_len); // Should compress the sequence
    }

    #[test]
    fn test_flamingo_config_serialization() {
        let configs = vec![
            FlamingoConfig::flamingo_3b(),
            FlamingoConfig::flamingo_9b(),
            FlamingoConfig::open_flamingo(),
        ];

        for config in configs {
            // Test JSON serialization
            let json = serde_json::to_string(&config);
            assert!(json.is_ok());

            let json_str = json.unwrap();
            let deserialized: Result<FlamingoConfig, _> = serde_json::from_str(&json_str);
            assert!(deserialized.is_ok());

            let deserialized = deserialized.unwrap();

            // Verify key fields are preserved
            assert_eq!(config.media_token_length, deserialized.media_token_length);
            assert_eq!(config.vision_language_dim, deserialized.vision_language_dim);
            assert_eq!(
                config.use_gated_cross_attention,
                deserialized.use_gated_cross_attention
            );
            assert_eq!(
                config.language_config.vocab_size,
                deserialized.language_config.vocab_size
            );
            assert_eq!(
                config.vision_config.hidden_size,
                deserialized.vision_config.hidden_size
            );
            assert_eq!(
                config.perceiver_config.num_latents,
                deserialized.perceiver_config.num_latents
            );
            assert_eq!(
                config.cross_attention_config.cross_attention_dim,
                deserialized.cross_attention_config.cross_attention_dim
            );
            assert_eq!(
                config.cross_attention_layers,
                deserialized.cross_attention_layers
            );
        }
    }

    #[test]
    fn test_flamingo_few_shot_simulation() {
        let config = FlamingoConfig::flamingo_3b();
        let model = FlamingoModel::new(config.clone()).unwrap();

        // Simulate a few-shot learning scenario with multiple examples
        let batch_size = 1;
        let num_examples = 3;
        let text_per_example = 15;
        let media_tokens_per_example = 5;
        let seq_len = num_examples * (text_per_example + media_tokens_per_example) + 10; // + query

        let input_ids = Tensor::randint(
            0,
            config.language_config.vocab_size as i64,
            &[batch_size, seq_len],
            TensorType::I64,
        )
        .unwrap();
        let attention_mask = Tensor::ones(&[batch_size, seq_len]).unwrap();
        let pixel_values = Tensor::randn(&[batch_size, 3, 224, 224]).unwrap();

        // Create media locations for few-shot examples
        let mut media_locations = Tensor::zeros(&[batch_size, seq_len]).unwrap();
        for example in 0..num_examples {
            let start_pos = example * (text_per_example + media_tokens_per_example);
            for i in 0..media_tokens_per_example {
                if start_pos + i < seq_len {
                    media_locations = media_locations.set_scalar(&[0, start_pos + i], 1.0).unwrap();
                }
            }
        }

        // Test forward pass with few-shot setup
        let output = model.forward_train(
            &input_ids,
            &attention_mask,
            Some(&pixel_values),
            Some(&media_locations),
            None,
        );
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(
            output.logits.shape(),
            &[batch_size, seq_len, config.language_config.vocab_size]
        );
        assert!(output.vision_features.is_some());

        // Verify cross-attention is working (should have attention weights)
        assert!(!output.cross_attention_weights.is_empty());

        // Test generation for the query
        let generated = model.generate_with_shots(
            &input_ids,
            &attention_mask,
            Some(&pixel_values),
            Some(&media_locations),
            15,   // Generate answer
            0.8,  // Lower temperature for more focused generation
            true, // Sample for diversity
        );

        assert!(generated.is_ok());
        let generated = generated.unwrap();
        assert!(generated.shape()[1] > seq_len);
    }

    #[test]
    fn test_flamingo_gating_mechanisms() {
        let mut config = FlamingoXAttentionConfig::default();
        let hidden_size = 2048;

        // Test different gating types
        let gating_types = vec!["tanh", "sigmoid", "relu"];

        for gating_type in gating_types {
            config.gating_type = gating_type.to_string();
            let cross_attn = GatedCrossAttention::new(hidden_size, config.clone()).unwrap();

            let batch_size = 1;
            let seq_len = 10;
            let vision_seq_len = 64;

            let hidden_states = Tensor::randn(&[batch_size, seq_len, hidden_size]).unwrap();
            let vision_features =
                Tensor::randn(&[batch_size, vision_seq_len, config.cross_attention_dim]).unwrap();

            let output = cross_attn.forward(&hidden_states, &vision_features, None);
            assert!(output.is_ok(), "Gating type {} failed", gating_type);

            let output = output.unwrap();
            assert_eq!(
                output.hidden_states.shape(),
                &[batch_size, seq_len, config.cross_attention_dim]
            );
        }
    }
}
