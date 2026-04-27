#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::hierarchical::config::{
        AggregationMethod, HierarchicalConfig, HierarchicalType, NestedConfig, PyramidConfig,
        ReductionMethod, TreeConfig, UpsamplingMethod,
    };
    use crate::hierarchical::models::HierarchicalTransformer;
    use crate::hierarchical::utils::{
        compute_hierarchical_positions, create_tree_mask, HierarchicalOutput,
    };
    use trustformers_core::tensor::Tensor;
    use trustformers_core::traits::Config;

    fn tiny_config() -> HierarchicalConfig {
        HierarchicalConfig {
            hidden_size: 32,
            num_levels: 2,
            num_heads: 4,
            reduction_factor: 2,
            num_layers_per_level: 1,
            intermediate_size: 64,
            dropout: 0.0,
            attention_dropout: 0.0,
            layer_norm_eps: 1e-5,
            hierarchical_type: HierarchicalType::Hierarchical,
            reduction_method: ReductionMethod::AveragePooling,
            aggregation_method: AggregationMethod::Sum,
            max_seq_lengths: vec![16, 8],
            cross_level_residual: true,
            use_position_embeddings: true,
            tree_config: None,
            pyramid_config: None,
            nested_config: None,
        }
    }

    /// Helper to validate using the string-returning validate method
    fn validate_config(config: &HierarchicalConfig) -> Result<(), String> {
        // Calls the inherent validate method (returns Result<(), String>)
        config.validate()
    }

    // --- Config Tests ---

    #[test]
    fn test_hierarchical_config_default_validates() {
        let config = HierarchicalConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_hierarchical_config_trait_validates() {
        let config = HierarchicalConfig::default();
        // This calls the Config trait method
        assert!(Config::validate(&config).is_ok());
    }

    #[test]
    fn test_hierarchical_config_invalid_num_levels_zero() {
        let mut config = tiny_config();
        config.num_levels = 0;
        config.max_seq_lengths = vec![];
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_hierarchical_config_invalid_reduction_factor_zero() {
        let mut config = tiny_config();
        config.reduction_factor = 0;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_hierarchical_config_invalid_hidden_heads() {
        let mut config = tiny_config();
        config.hidden_size = 33;
        config.num_heads = 4;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_hierarchical_config_invalid_dropout() {
        let mut config = tiny_config();
        config.dropout = 1.5;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_hierarchical_config_invalid_max_seq_lengths_mismatch() {
        let mut config = tiny_config();
        config.num_levels = 3;
        config.max_seq_lengths = vec![16, 8]; // Should be 3 elements
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_hierarchical_config_hierarchical_preset() {
        // hidden_size must be divisible by num_heads (default 12), use 768
        let config = HierarchicalConfig::hierarchical(768, 3);
        assert!(validate_config(&config).is_ok());
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_levels, 3);
        assert_eq!(config.max_seq_lengths.len(), 3);
    }

    #[test]
    fn test_hierarchical_config_pyramid_preset() {
        let config = HierarchicalConfig::pyramid(768, 3);
        assert!(validate_config(&config).is_ok());
        assert!(config.pyramid_config.is_some());
    }

    #[test]
    fn test_hierarchical_config_tree_preset() {
        let config = HierarchicalConfig::tree(768, 2, 4);
        assert!(config.tree_config.is_some());
        let tree = config.tree_config.as_ref().expect("expected tree config");
        assert_eq!(tree.branching_factor, 2);
        assert_eq!(tree.max_depth, 4);
    }

    #[test]
    fn test_hierarchical_config_nested_preset() {
        let config = HierarchicalConfig::nested(768, 3);
        assert!(config.nested_config.is_some());
        let nested = config.nested_config.as_ref().expect("expected nested config");
        assert_eq!(nested.num_nested_levels, 3);
    }

    #[test]
    fn test_hierarchical_config_get_hidden_size() {
        let config = tiny_config();
        assert_eq!(config.get_hidden_size(0), 32);
        assert_eq!(config.get_hidden_size(1), 32);
    }

    #[test]
    fn test_hierarchical_config_get_hidden_size_pyramid() {
        let mut config = tiny_config();
        config.hierarchical_type = HierarchicalType::Pyramid;
        config.pyramid_config = Some(PyramidConfig {
            scaling_factors: vec![1.0, 0.5],
            skip_connections: true,
            upsampling_method: UpsamplingMethod::Linear,
            use_fpn: false,
        });
        assert_eq!(config.get_hidden_size(0), 32);
        assert_eq!(config.get_hidden_size(1), 16);
    }

    #[test]
    fn test_hierarchical_config_get_seq_length() {
        let config = tiny_config();
        assert_eq!(config.get_seq_length(0), 16);
        assert_eq!(config.get_seq_length(1), 8);
        let fallback = config.get_seq_length(5);
        assert!(fallback > 0);
    }

    #[test]
    fn test_hierarchical_config_get_reduction_factor() {
        let config = tiny_config();
        assert_eq!(config.get_reduction_factor(0), 1);
        assert_eq!(config.get_reduction_factor(1), 2);
        assert_eq!(config.get_reduction_factor(2), 4);
    }

    #[test]
    fn test_hierarchical_config_estimate_parameters() {
        let config = tiny_config();
        let params = config.estimate_parameters();
        assert!(params > 0);
    }

    #[test]
    fn test_hierarchical_config_architecture() {
        let config = HierarchicalConfig::default();
        assert_eq!(config.architecture(), "hierarchical");
    }

    #[test]
    fn test_tree_config_default() {
        let config = TreeConfig::default();
        assert_eq!(config.branching_factor, 2);
        assert_eq!(config.max_depth, 8);
        assert!(!config.learnable_structure);
    }

    #[test]
    fn test_pyramid_config_default() {
        let config = PyramidConfig::default();
        assert_eq!(config.scaling_factors.len(), 4);
        assert!(config.skip_connections);
        assert!(!config.use_fpn);
    }

    #[test]
    fn test_nested_config_default() {
        let config = NestedConfig::default();
        assert_eq!(config.num_nested_levels, 3);
        assert!(!config.share_parameters);
        assert!(!config.progressive_training);
    }

    #[test]
    fn test_invalid_tree_config_zero_branching() {
        let mut config = tiny_config();
        config.tree_config = Some(TreeConfig {
            branching_factor: 0,
            ..TreeConfig::default()
        });
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_invalid_pyramid_config_empty_scaling() {
        let mut config = tiny_config();
        config.pyramid_config = Some(PyramidConfig {
            scaling_factors: vec![],
            ..PyramidConfig::default()
        });
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_invalid_nested_config_zero_levels() {
        let mut config = tiny_config();
        config.nested_config = Some(NestedConfig {
            num_nested_levels: 0,
            ..NestedConfig::default()
        });
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_hierarchical_transformer_creation() {
        let config = tiny_config();
        let model = HierarchicalTransformer::new(config, 128);
        assert!(model.is_ok());
    }

    #[test]
    fn test_hierarchical_output_creation() {
        let hidden = Tensor::zeros(&[1, 4, 32]).expect("Failed to create tensor");
        let output = HierarchicalOutput {
            output: hidden,
            level_outputs: vec![],
            attention_weights: None,
            hierarchical_positions: None,
        };
        assert_eq!(output.output.shape(), &[1, 4, 32]);
    }

    #[test]
    fn test_compute_hierarchical_positions() {
        let positions = compute_hierarchical_positions(8, 2, 2);
        assert!(positions.is_ok());
        let positions = positions.expect("Failed to compute positions");
        assert!(!positions.is_empty());
    }

    #[test]
    fn test_create_tree_mask() {
        use crate::hierarchical::config::TreeConstruction;
        let mask = create_tree_mask(8, 2, &TreeConstruction::Binary);
        assert!(mask.is_ok());
    }
}
