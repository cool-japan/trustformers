#[cfg(test)]
mod tests {
    use crate::recursive::config::RecursiveConfig;
    use crate::recursive::model::RecursiveTransformer;

    fn small_recursive_config() -> RecursiveConfig {
        RecursiveConfig {
            vocab_size: 100,
            hidden_size: 32,
            intermediate_size: 64,
            num_attention_heads: 4,
            max_position_embeddings: 64,
            num_recursive_layers: 2,
            recursion_depth: 2,
            chunk_size: 16,
            overlap_size: 4,
            use_adaptive_depth: false,
            use_hierarchical_attention: false,
            use_universal_transformer: false,
            memory_size: 32,
            use_memory_compression: false,
            hierarchy_levels: 2,
            level_compression_ratios: vec![1.0, 0.5],
            ..RecursiveConfig::default()
        }
    }

    // --- RecursiveConfig tests ---

    #[test]
    fn test_recursive_config_default() {
        let config = RecursiveConfig::default();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_recursive_layers, 6);
        assert_eq!(config.recursion_depth, 3);
    }

    #[test]
    fn test_recursive_config_chunk_settings() {
        let config = RecursiveConfig::default();
        assert_eq!(config.chunk_size, 512);
        assert_eq!(config.overlap_size, 64);
    }

    #[test]
    fn test_recursive_config_memory_settings() {
        let config = RecursiveConfig::default();
        assert_eq!(config.memory_size, 1024);
        assert!(config.use_memory_compression);
        assert!((config.compression_ratio - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_recursive_config_hierarchical_settings() {
        let config = RecursiveConfig::default();
        assert!(config.use_hierarchical_attention);
        assert_eq!(config.hierarchy_levels, 3);
        assert_eq!(config.level_compression_ratios.len(), 3);
        assert!(config.cross_level_attention);
    }

    #[test]
    fn test_recursive_config_universal_settings() {
        let config = RecursiveConfig::default();
        // Default may or may not enable universal transformer
        // Just verify the fields exist and are consistent
        if config.use_universal_transformer {
            assert!(config.max_steps > 0);
        }
    }

    #[test]
    fn test_recursive_config_clone() {
        let config = RecursiveConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.vocab_size, config.vocab_size);
        assert_eq!(cloned.hidden_size, config.hidden_size);
        assert_eq!(
            cloned.level_compression_ratios.len(),
            config.level_compression_ratios.len()
        );
    }

    #[test]
    fn test_recursive_config_adaptive_depth() {
        let config = RecursiveConfig::default();
        assert!(config.use_adaptive_depth);
        assert_eq!(config.min_depth, 1);
        assert_eq!(config.max_depth, 5);
        assert!((config.depth_threshold - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_recursive_config_optimization_flags() {
        let config = RecursiveConfig::default();
        assert!(config.use_gradient_checkpointing);
        assert!(config.use_flash_attention);
    }

    // --- RecursiveTransformer tests ---

    #[test]
    fn test_recursive_transformer_creation() {
        let config = small_recursive_config();
        let model = RecursiveTransformer::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_recursive_transformer_with_adaptive_depth() {
        let mut config = small_recursive_config();
        config.use_adaptive_depth = true;
        let model = RecursiveTransformer::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_recursive_transformer_with_hierarchical() {
        let mut config = small_recursive_config();
        config.use_hierarchical_attention = true;
        let model = RecursiveTransformer::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_recursive_transformer_with_universal() {
        let mut config = small_recursive_config();
        config.use_universal_transformer = true;
        let model = RecursiveTransformer::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_recursive_transformer_all_features() {
        let mut config = small_recursive_config();
        config.use_adaptive_depth = true;
        config.use_hierarchical_attention = true;
        config.use_universal_transformer = true;
        let model = RecursiveTransformer::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_recursive_transformer_no_features() {
        let config = small_recursive_config();
        let model = RecursiveTransformer::new(config);
        assert!(model.is_ok());
    }

    // --- Small config specific tests ---

    #[test]
    fn test_small_config_values() {
        let config = small_recursive_config();
        assert_eq!(config.vocab_size, 100);
        assert_eq!(config.hidden_size, 32);
        assert_eq!(config.num_recursive_layers, 2);
    }

    #[test]
    fn test_config_model_type() {
        let config = RecursiveConfig::default();
        assert_eq!(config.model_type, "recursive");
    }

    #[test]
    fn test_config_dropout_values() {
        let config = RecursiveConfig::default();
        assert!((config.dropout - 0.1).abs() < f32::EPSILON);
        assert!((config.attention_dropout - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_layer_norm_eps() {
        let config = RecursiveConfig::default();
        assert!(config.layer_norm_eps > 0.0);
        assert!(config.layer_norm_eps < 1e-6);
    }

    #[test]
    fn test_config_max_position_embeddings() {
        let config = RecursiveConfig::default();
        assert_eq!(config.max_position_embeddings, 16384);
    }

    #[test]
    fn test_config_hidden_act() {
        let config = RecursiveConfig::default();
        assert_eq!(config.hidden_act, "gelu");
    }

    #[test]
    fn test_config_token_ids() {
        let config = RecursiveConfig::default();
        assert_eq!(config.bos_token_id, 1);
        assert_eq!(config.eos_token_id, 2);
        assert_eq!(config.pad_token_id, Some(0));
    }

    #[test]
    fn test_config_memory_update_strategy() {
        let config = RecursiveConfig::default();
        assert_eq!(config.memory_update_strategy, "gated");
    }
}
