#[cfg(test)]
mod tests {
    use crate::cross_attention::config::*;
    use crate::cross_attention::layers::*;
    use crate::cross_attention::utils::{create_attention_mask, MaskType};

    // --- CrossAttentionConfig tests ---

    #[test]
    fn test_config_default() {
        let config = CrossAttentionConfig::default();
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.num_heads, 8);
        assert!(config.bias);
    }

    #[test]
    fn test_config_standard() {
        let config = CrossAttentionConfig::standard(256, 4);
        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.num_heads, 4);
    }

    #[test]
    fn test_config_sparse() {
        let config = CrossAttentionConfig::sparse(256, 4, 0.5);
        assert!(config.sparse_config.is_some());
        if let Some(sc) = &config.sparse_config {
            assert!((sc.sparsity_ratio - 0.5).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_config_get_head_dim() {
        let config = CrossAttentionConfig::default();
        assert_eq!(config.get_head_dim(), 64); // 512 / 8
    }

    #[test]
    fn test_config_get_head_dim_custom() {
        let config = CrossAttentionConfig {
            head_dim: Some(32),
            ..CrossAttentionConfig::default()
        };
        assert_eq!(config.get_head_dim(), 32);
    }

    #[test]
    fn test_config_get_scale() {
        let config = CrossAttentionConfig::default();
        let expected = 1.0 / (64.0_f32).sqrt();
        assert!((config.get_scale() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_config_validate_valid() {
        let config = CrossAttentionConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_hidden_not_divisible() {
        let config = CrossAttentionConfig {
            hidden_size: 100,
            num_heads: 3,
            ..CrossAttentionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_bad_dropout() {
        let config = CrossAttentionConfig {
            attention_dropout: 1.5,
            ..CrossAttentionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    // --- CrossAttention creation tests ---

    #[test]
    fn test_cross_attention_creation() {
        let config = CrossAttentionConfig::standard(64, 4);
        let attn = CrossAttention::new(config);
        assert!(attn.is_ok());
    }

    #[test]
    fn test_multi_head_cross_attention_creation() {
        let config = CrossAttentionConfig::standard(64, 4);
        let attn = MultiHeadCrossAttention::new(config);
        assert!(attn.is_ok());
    }

    #[test]
    fn test_sparse_cross_attention_creation() {
        let config = CrossAttentionConfig::sparse(64, 4, 0.3);
        let attn = SparseCrossAttention::new(config);
        assert!(attn.is_ok());
    }

    #[test]
    fn test_hierarchical_cross_attention_creation() {
        let config = CrossAttentionConfig {
            hidden_size: 64,
            num_heads: 4,
            hierarchical_config: Some(HierarchicalAttentionConfig::default()),
            ..CrossAttentionConfig::default()
        };
        let attn = HierarchicalCrossAttention::new(config);
        assert!(attn.is_ok());
    }

    // --- Attention mask tests ---

    #[test]
    fn test_none_mask() {
        let mask = create_attention_mask(4, 4, MaskType::None);
        assert!(mask.is_ok());
        if let Ok(m) = mask {
            assert_eq!(m.shape(), &[4, 4]);
        }
    }

    #[test]
    fn test_causal_mask() {
        let mask = create_attention_mask(4, 4, MaskType::Causal);
        assert!(mask.is_ok());
        if let Ok(m) = mask {
            assert_eq!(m.shape(), &[4, 4]);
        }
    }

    #[test]
    fn test_local_mask() {
        let mask = create_attention_mask(8, 8, MaskType::Local(3));
        assert!(mask.is_ok());
        if let Ok(m) = mask {
            assert_eq!(m.shape(), &[8, 8]);
        }
    }

    // --- SparseAttentionConfig defaults ---

    #[test]
    fn test_sparse_config_default() {
        let sc = SparseAttentionConfig::default();
        assert!((sc.sparsity_ratio - 0.1).abs() < f32::EPSILON);
        assert_eq!(sc.block_size, Some(64));
    }

    #[test]
    fn test_hierarchical_config_default() {
        let hc = HierarchicalAttentionConfig::default();
        assert_eq!(hc.num_levels, 3);
        assert_eq!(hc.pooling_factor, 2);
        assert!(hc.learnable_pooling);
    }

    #[test]
    fn test_adaptive_config_default() {
        let ac = AdaptiveAttentionConfig::default();
        assert_eq!(ac.num_patterns, 4);
        assert!((ac.temperature - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gated_config_default() {
        let gc = GatedAttentionConfig::default();
        assert!(gc.gate_bias);
        assert!(!gc.separate_gates);
    }

    // --- Config validation edge cases ---

    #[test]
    fn test_config_validate_bad_sparsity() {
        let config = CrossAttentionConfig {
            sparse_config: Some(SparseAttentionConfig {
                sparsity_ratio: -0.5,
                ..SparseAttentionConfig::default()
            }),
            ..CrossAttentionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_bad_hierarchical() {
        let config = CrossAttentionConfig {
            hierarchical_config: Some(HierarchicalAttentionConfig {
                num_levels: 0,
                ..HierarchicalAttentionConfig::default()
            }),
            ..CrossAttentionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_bad_adaptive() {
        let config = CrossAttentionConfig {
            adaptive_config: Some(AdaptiveAttentionConfig {
                temperature: 0.0,
                ..AdaptiveAttentionConfig::default()
            }),
            ..CrossAttentionConfig::default()
        };
        assert!(config.validate().is_err());
    }
}
