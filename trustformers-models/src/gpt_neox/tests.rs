#[cfg(test)]
mod tests {
    use crate::gpt_neox::config::GPTNeoXConfig;
    use crate::gpt_neox::model::{GPTNeoXForCausalLM, GPTNeoXModel};
    use trustformers_core::traits::{Config, Model};

    #[test]
    fn test_gpt_neox_config_validation() {
        // Use minimal config to reduce memory usage
        let config = GPTNeoXConfig {
            vocab_size: 1000,
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            intermediate_size: 256,
            max_position_embeddings: 512,
            layer_norm_eps: 1e-5,
            hidden_act: "gelu".to_string(),
            rotary_emb_base: 10000.0,
            rotary_pct: 1.0,
            use_parallel_residual: false,
            tie_word_embeddings: false,
            initializer_range: 0.02,
            bos_token_id: Some(0),
            eos_token_id: Some(2),
        };
        assert!(config.validate().is_ok());

        // Test head dimension calculation
        assert_eq!(config.hidden_size / config.num_attention_heads, 8); // 64 / 8

        // Explicit cleanup
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neox_config_with_parallel_residual() {
        // Test parallel residual configuration
        let config = GPTNeoXConfig {
            vocab_size: 1000,
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            intermediate_size: 256,
            max_position_embeddings: 512,
            use_parallel_residual: true, // Parallel attention + MLP
            ..GPTNeoXConfig::default()
        };
        assert!(config.validate().is_ok());
        assert!(config.use_parallel_residual);

        // Explicit cleanup
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_rinna_japanese_config() {
        // Test the pre-configured Rinna Japanese 3.6B config
        let config = GPTNeoXConfig::rinna_japanese_3_6b();
        assert!(config.validate().is_ok());
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 2816);
        assert_eq!(config.num_hidden_layers, 36);
        assert_eq!(config.num_attention_heads, 22);
        assert_eq!(config.intermediate_size, 11264);

        // Explicit cleanup
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neox_architecture() {
        let config = GPTNeoXConfig::default();
        assert_eq!(config.architecture(), "gpt_neox");
    }

    #[test]
    fn test_invalid_gpt_neox_config() {
        // Test invalid config: hidden_size not divisible by num_attention_heads
        let config = GPTNeoXConfig {
            hidden_size: 65, // Not divisible by 8
            num_attention_heads: 8,
            ..GPTNeoXConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_rotary_pct() {
        // Test invalid rotary percentage
        let config = GPTNeoXConfig {
            rotary_pct: 1.5, // Should be between 0.0 and 1.0
            ..GPTNeoXConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_gpt_neox_model_creation() {
        // Minimal config for fast test
        let config = GPTNeoXConfig {
            vocab_size: 100,
            hidden_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            intermediate_size: 128,
            max_position_embeddings: 64,
            ..GPTNeoXConfig::default()
        };

        let model = GPTNeoXModel::new(config.clone()).unwrap();
        assert_eq!(model.get_config().num_hidden_layers, 2);
        assert_eq!(model.get_config().hidden_size, 32);
        assert_eq!(model.get_config().num_attention_heads, 4);

        // Explicit cleanup
        drop(model);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neox_for_causal_lm_creation() {
        // Minimal config for fast test
        let config = GPTNeoXConfig {
            vocab_size: 100,
            hidden_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            intermediate_size: 128,
            max_position_embeddings: 64,
            ..GPTNeoXConfig::default()
        };

        let model = GPTNeoXForCausalLM::new(config.clone()).unwrap();
        assert_eq!(model.get_config().num_hidden_layers, 2);
        assert_eq!(model.get_config().vocab_size, 100);

        // Explicit cleanup
        drop(model);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neox_forward_pass() {
        use trustformers_core::tensor::Tensor;
        use trustformers_core::traits::Model;

        // Minimal config for fast test
        let config = GPTNeoXConfig {
            vocab_size: 100,
            hidden_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            intermediate_size: 128,
            max_position_embeddings: 64,
            ..GPTNeoXConfig::default()
        };

        let model = GPTNeoXModel::new(config).unwrap();
        let input_ids = vec![1, 2, 3, 4, 5];

        let output = model.forward(input_ids.clone()).unwrap();
        match &output {
            Tensor::F32(arr) => {
                // Should be [seq_len, hidden_size]
                assert_eq!(arr.shape().len(), 2);
                assert_eq!(arr.shape()[0], input_ids.len());
                assert_eq!(arr.shape()[1], 32);
            },
            _ => panic!("Expected F32 tensor"),
        }

        // Explicit cleanup
        drop(model);
        drop(output);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neox_parameter_count() {
        use trustformers_core::traits::Model;

        // Minimal config for fast test
        let config = GPTNeoXConfig {
            vocab_size: 100,
            hidden_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            intermediate_size: 128,
            max_position_embeddings: 64,
            ..GPTNeoXConfig::default()
        };

        let model = GPTNeoXModel::new(config).unwrap();
        let param_count = model.num_parameters();

        // Basic sanity check: should have more than 0 parameters
        assert!(param_count > 0);
        println!("GPT-NeoX model parameter count: {}", param_count);

        // Explicit cleanup
        drop(model);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neox_config_default_values() {
        let config = GPTNeoXConfig::default();
        assert_eq!(config.vocab_size, 50432);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_hidden_layers, 16);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.intermediate_size, 8192);
        assert_eq!(config.max_position_embeddings, 2048);
        assert_eq!(config.layer_norm_eps, 1e-5);
        assert_eq!(config.hidden_act, "gelu");
        assert_eq!(config.rotary_emb_base, 10000.0);
        assert_eq!(config.rotary_pct, 1.0);
        assert!(!config.use_parallel_residual);
        assert!(!config.tie_word_embeddings);
        assert_eq!(config.bos_token_id, Some(0));
        assert_eq!(config.eos_token_id, Some(2));

        // Explicit cleanup
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neox_variable_sequence_lengths() {
        use trustformers_core::tensor::Tensor;
        use trustformers_core::traits::Model;

        // Minimal config for fast test
        let config = GPTNeoXConfig {
            vocab_size: 100,
            hidden_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            intermediate_size: 128,
            max_position_embeddings: 64,
            ..GPTNeoXConfig::default()
        };

        let model = GPTNeoXModel::new(config).unwrap();

        // Test with different sequence lengths
        for seq_len in [1, 5, 10, 20] {
            let input_ids: Vec<u32> = (0..seq_len).map(|i| (i % 100) as u32).collect();
            let output = model.forward(input_ids.clone()).unwrap();

            match &output {
                Tensor::F32(arr) => {
                    assert_eq!(arr.shape()[0], seq_len);
                    assert_eq!(arr.shape()[1], 32);
                },
                _ => panic!("Expected F32 tensor"),
            }
        }

        // Explicit cleanup
        drop(model);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neox_rotary_embedding_coverage() {
        // Test partial rotary embedding (rotary_pct < 1.0)
        let config = GPTNeoXConfig {
            vocab_size: 100,
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            intermediate_size: 256,
            max_position_embeddings: 128,
            rotary_pct: 0.5, // Only apply RoPE to 50% of dimensions
            ..GPTNeoXConfig::default()
        };

        assert!(config.validate().is_ok());

        let model = GPTNeoXModel::new(config).unwrap();
        // Just verify it can be created without errors

        // Explicit cleanup
        drop(model);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neox_zero_config_fails() {
        // Test that zero values are properly rejected
        let config = GPTNeoXConfig {
            hidden_size: 0, // Should fail
            ..GPTNeoXConfig::default()
        };
        assert!(config.validate().is_err());

        let config = GPTNeoXConfig {
            num_hidden_layers: 0, // Should fail
            ..GPTNeoXConfig::default()
        };
        assert!(config.validate().is_err());

        let config = GPTNeoXConfig {
            num_attention_heads: 0, // Should fail
            ..GPTNeoXConfig::default()
        };
        assert!(config.validate().is_err());
    }
}
