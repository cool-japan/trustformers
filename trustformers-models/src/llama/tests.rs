#[cfg(test)]
mod tests {
    use crate::llama::config::LlamaConfig;
    use crate::llama::model::{LlamaForCausalLM, LlamaModel, RMSNorm, RotaryEmbedding};
    use trustformers_core::traits::Config;

    #[test]
    fn test_llama_config_validation() {
        // Use minimal config to reduce memory usage
        let config = LlamaConfig {
            num_hidden_layers: 2,
            vocab_size: 1000,
            hidden_size: 64,
            num_attention_heads: 8,
            intermediate_size: 256,
            ..LlamaConfig::default()
        };
        assert!(config.validate().is_ok());

        // Test head dimension calculation
        assert_eq!(config.head_dim(), 8); // 64 / 8

        // Test KV heads
        assert_eq!(config.num_kv_heads(), 8); // No grouped-query attention by default
        assert_eq!(config.num_query_groups(), 1);

        // Explicit cleanup
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_llama2_config_with_gqa() {
        // Use minimal config with GQA to reduce memory usage
        let config = LlamaConfig {
            num_hidden_layers: 2,
            vocab_size: 1000,
            hidden_size: 64,
            num_attention_heads: 8,
            num_key_value_heads: Some(2), // Grouped-query attention
            intermediate_size: 256,
            ..LlamaConfig::default()
        };
        assert!(config.validate().is_ok());

        // Test grouped-query attention
        assert_eq!(config.num_kv_heads(), 2);
        assert_eq!(config.num_query_groups(), 4); // 8 / 2
        assert_eq!(config.head_dim(), 8); // 64 / 8

        // Explicit cleanup
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_code_llama_config() {
        // Use minimal config inspired by Code Llama to reduce memory usage
        let config = LlamaConfig {
            num_hidden_layers: 2,
            vocab_size: 1000,
            hidden_size: 64,
            num_attention_heads: 8,
            intermediate_size: 256,
            max_position_embeddings: 512, // Reduced from 16384
            ..LlamaConfig::default()
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.max_position_embeddings, 512); // Reduced context for testing
        assert_eq!(config.vocab_size, 1000); // Smaller vocab

        // Explicit cleanup
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_llama_architecture() {
        let config = LlamaConfig::default();
        assert_eq!(config.architecture(), "LLaMA");
    }

    #[test]
    fn test_invalid_llama_config() {
        let mut config = LlamaConfig::default();
        config.hidden_size = 4095; // Not divisible by num_attention_heads (32)
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_rmsnorm_creation() {
        let rmsnorm = RMSNorm::new(4096, 1e-6);
        assert!(rmsnorm.is_ok());
    }

    #[test]
    fn test_rotary_embedding_creation() {
        let rope = RotaryEmbedding::new(128, 2048, 10000.0);
        // Just test that it doesn't panic
        assert_eq!(rope.dim, 128);
        assert_eq!(rope.max_seq_len, 2048);
        assert_eq!(rope.base, 10000.0);
    }

    #[test]
    fn test_llama_model_creation() {
        let config = LlamaConfig {
            num_hidden_layers: 2, // Smaller for testing
            vocab_size: 1000,
            hidden_size: 64,
            num_attention_heads: 8,
            intermediate_size: 256,
            ..LlamaConfig::default()
        };

        let model = LlamaModel::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_llama_for_causal_lm_creation() {
        let config = LlamaConfig {
            num_hidden_layers: 1, // Minimal for testing
            vocab_size: 100,
            hidden_size: 32,
            num_attention_heads: 4,
            intermediate_size: 128,
            ..LlamaConfig::default()
        };

        let model = LlamaForCausalLM::new(config);
        assert!(model.is_ok());
    }
}
