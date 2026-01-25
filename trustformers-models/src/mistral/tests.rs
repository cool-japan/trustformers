#[cfg(test)]
mod tests {
    use crate::mistral::config::MistralConfig;
    use crate::mistral::model::{
        MistralForCausalLM, MistralModel, MixtralExpert, MixtralSparseMoE,
    };
    use trustformers_core::traits::Config;

    #[test]
    fn test_mistral_config_validation() {
        // Use minimal config to reduce memory usage
        let config = MistralConfig {
            num_hidden_layers: 2,
            vocab_size: 1000,
            hidden_size: 64,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            intermediate_size: 256,
            ..MistralConfig::default()
        };
        assert!(config.validate().is_ok());

        // Test head dimension calculation
        assert_eq!(config.head_dim(), 8); // 64 / 8

        // Test grouped-query attention
        assert_eq!(config.num_query_groups(), 4); // 8 / 2

        // Explicit cleanup
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_mistral_sliding_window() {
        // Use minimal config with sliding window to reduce memory usage
        let config = MistralConfig {
            num_hidden_layers: 2,
            vocab_size: 1000,
            hidden_size: 64,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            intermediate_size: 256,
            sliding_window: Some(512), // Reduced from 4096
            ..MistralConfig::default()
        };
        assert!(config.uses_sliding_window());
        assert_eq!(config.sliding_window_size(), 512);

        // Explicit cleanup
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_mixtral_config() {
        // Use minimal Mixtral config to reduce memory usage
        let config = MistralConfig {
            num_hidden_layers: 2,
            vocab_size: 1000,
            hidden_size: 64,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            intermediate_size: 256,
            model_type: "mixtral".to_string(),
            sliding_window: None, // Mixtral doesn't use sliding window
            ..MistralConfig::default()
        };
        assert!(config.validate().is_ok());
        assert!(!config.uses_sliding_window()); // Mixtral doesn't use sliding window
        assert_eq!(config.model_type, "mixtral");

        // Explicit cleanup
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_mistral_architecture() {
        let config = MistralConfig::default();
        assert_eq!(config.architecture(), "Mistral");
    }

    #[test]
    fn test_invalid_mistral_config() {
        let mut config = MistralConfig::default();
        config.num_attention_heads = 31; // Not divisible by num_key_value_heads (8)
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_mistral_model_creation() {
        let config = MistralConfig {
            num_hidden_layers: 2, // Smaller for testing
            vocab_size: 1000,
            hidden_size: 64,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            intermediate_size: 256,
            ..MistralConfig::default()
        };

        let model = MistralModel::new(config);
        assert!(model.is_ok());

        // Explicit cleanup
        if let Ok(model) = model {
            drop(model);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_mistral_for_causal_lm_creation() {
        let config = MistralConfig {
            num_hidden_layers: 1, // Minimal for testing
            vocab_size: 100,
            hidden_size: 32,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 128,
            ..MistralConfig::default()
        };

        let model = MistralForCausalLM::new(config);
        assert!(model.is_ok());

        // Explicit cleanup
        if let Ok(model) = model {
            drop(model);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_mixtral_sparse_moe_creation() {
        let config = MistralConfig {
            hidden_size: 64,
            intermediate_size: 256,
            ..MistralConfig::default()
        };

        // Create experts
        let mut experts = Vec::new();
        for i in 0..4 {
            experts.push(MixtralExpert::new(i, &config).expect("operation failed"));
        }

        // Create MoE config
        let moe_config = crate::moe::MoEConfig {
            num_experts: 4,
            num_experts_per_token: 2,
            hidden_size: config.hidden_size,
            expert_capacity: None,
            ..Default::default()
        };

        let moe = MixtralSparseMoE::new(experts, moe_config);
        assert!(moe.is_ok());

        // Explicit cleanup
        if let Ok(moe) = moe {
            drop(moe);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_mixtral_expert_creation() {
        let config = MistralConfig {
            hidden_size: 64,
            intermediate_size: 256,
            ..MistralConfig::default()
        };

        let expert = MixtralExpert::new(0, &config);
        assert!(expert.is_ok());

        // Explicit cleanup
        if let Ok(expert) = expert {
            drop(expert);
        }
        std::hint::black_box(());
    }
}
