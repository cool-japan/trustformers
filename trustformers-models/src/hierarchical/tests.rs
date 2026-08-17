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

    // --- Layer-level regression tests (real pooling, not zeros) ---

    /// `HierarchicalAttention::forward` must emit a real, input-dependent signal.
    ///
    /// Before the pooling fix every hierarchy level above the first was an all-zero
    /// tensor and every aggregation path routed through a zero-filled
    /// `upsample_to_shape`, so this output was identically zero.
    #[test]
    fn test_hierarchical_attention_forward_is_nonzero_and_input_dependent() {
        use crate::hierarchical::layers::HierarchicalAttention;
        use trustformers_core::traits::Layer;

        let layer =
            HierarchicalAttention::new(tiny_config()).expect("hierarchical attention creation");

        let input_a = Tensor::from_vec(
            (0..(8 * 32)).map(|i| (i as f32 * 0.01).sin()).collect(),
            &[1, 8, 32],
        )
        .expect("input a");
        let input_b = Tensor::from_vec(
            (0..(8 * 32)).map(|i| (i as f32 * 0.03).cos()).collect(),
            &[1, 8, 32],
        )
        .expect("input b");

        let out_a = layer.forward(input_a).expect("forward a");
        let out_b = layer.forward(input_b).expect("forward b");

        let data_a = out_a.output.data().expect("data a");
        let data_b = out_b.output.data().expect("data b");

        assert_eq!(out_a.output.shape(), vec![1, 8, 32]);
        assert!(
            data_a.iter().any(|v| v.abs() > 1e-6),
            "hierarchical attention returned an all-zero tensor"
        );
        assert!(
            data_a.iter().zip(data_b.iter()).any(|(x, y)| (x - y).abs() > 1e-6),
            "hierarchical attention output does not depend on its input"
        );
        assert!(data_a.iter().all(|v| v.is_finite()));

        // Every hierarchy level must carry real values too.
        assert_eq!(out_a.level_outputs.len(), 2);
        for (level, output) in out_a.level_outputs.iter().enumerate() {
            assert!(
                output.data().expect("level data").iter().any(|v| v.abs() > 1e-6),
                "level {level} output is all zeros"
            );
        }
    }

    // --- Weight-loading honesty tests ---

    /// Every hierarchical model must report that it cannot parse a checkpoint.
    ///
    /// The previous implementations spooled the reader into a temporary file,
    /// printed `Weight loading fallback - weights successfully processed`, deleted
    /// the file and returned `Ok(())`. A caller therefore ran a randomly
    /// initialised model believing it held pretrained weights. A 4096-byte buffer
    /// cleared the old `buffer.len() < 1024` guard, so this test would have failed
    /// against every one of them.
    #[test]
    fn test_load_pretrained_never_fakes_success() {
        use crate::hierarchical::models::{
            HierarchicalForLanguageModeling, HierarchicalForSequenceClassification,
            NestedTransformer, PyramidTransformer, TreeTransformer,
        };
        use std::io::Cursor;
        use trustformers_core::traits::Model;

        let config = tiny_config();
        let vocab_size = 64usize;
        let payload = vec![0u8; 4096];

        let mut hierarchical = HierarchicalTransformer::new(config.clone(), vocab_size)
            .expect("hierarchical transformer");
        assert!(hierarchical.load_pretrained(&mut Cursor::new(payload.clone())).is_err());

        let mut pyramid =
            PyramidTransformer::new(config.clone(), vocab_size).expect("pyramid transformer");
        assert!(pyramid.load_pretrained(&mut Cursor::new(payload.clone())).is_err());

        let mut tree = TreeTransformer::new(config.clone(), vocab_size).expect("tree transformer");
        assert!(tree.load_pretrained(&mut Cursor::new(payload.clone())).is_err());

        let mut nested =
            NestedTransformer::new(config.clone(), vocab_size).expect("nested transformer");
        assert!(nested.load_pretrained(&mut Cursor::new(payload.clone())).is_err());

        let mut classifier =
            HierarchicalForSequenceClassification::new(config.clone(), vocab_size, 3)
                .expect("classification model");
        assert!(classifier.load_pretrained(&mut Cursor::new(payload.clone())).is_err());

        let mut lm = HierarchicalForLanguageModeling::new(config, vocab_size)
            .expect("language modelling head");
        assert!(lm.load_pretrained(&mut Cursor::new(payload)).is_err());
    }

    /// The path-based loaders must fail too, instead of printing "Loaded <tensor>"
    /// for tensors they never assign.
    #[test]
    fn test_path_loaders_never_fake_success() {
        use crate::hierarchical::models::PyramidTransformer;

        let config = tiny_config();
        let mut hierarchical =
            HierarchicalTransformer::new(config.clone(), 64).expect("hierarchical transformer");
        let checkpoint_dir = std::env::temp_dir().join("trustformers_hierarchical_absent");
        assert!(hierarchical.load_from_path(&checkpoint_dir).is_err());
        assert!(hierarchical.load_from_huggingface("some-org/some-model").is_err());

        let mut pyramid = PyramidTransformer::new(config, 64).expect("pyramid transformer");
        assert!(pyramid.load_from_path(&checkpoint_dir.to_string_lossy()).is_err());
    }

    // --- Model-level end-to-end forward tests ---

    /// Every hierarchical model family must run end to end from token ids and
    /// return a rank-3 `[1, seq_len, hidden]` activation.
    ///
    /// This is the regression test for the missing batch axis: `Embedding` returns
    /// `[seq_len, hidden]`, and the model `forward`s fed that straight into layers
    /// defined on `[batch, seq, hidden]`. `HierarchicalTransformer` failed with
    /// "expected a 3-D [batch, seq, hidden] tensor", `TreeTransformer` and
    /// `NestedTransformer` with "split_heads … got 2", and `PyramidTransformer`
    /// returned a rank-2 tensor that silently broke the downstream CLS selection.
    #[test]
    fn test_every_model_family_runs_end_to_end() {
        use crate::hierarchical::models::{NestedTransformer, PyramidTransformer, TreeTransformer};
        use trustformers_core::traits::Model;

        let config = tiny_config();
        let vocab_size = 64usize;
        let ids: Vec<u32> = vec![1, 2, 3, 4];
        let other: Vec<u32> = vec![9, 8, 7, 6];
        let expected_shape = vec![1, ids.len(), config.hidden_size];

        type BoxedFamily = Box<
            dyn Model<Config = HierarchicalConfig, Input = Vec<u32>, Output = HierarchicalOutput>,
        >;

        let families: Vec<(&str, BoxedFamily)> = vec![
            (
                "hierarchical",
                Box::new(
                    HierarchicalTransformer::new(config.clone(), vocab_size)
                        .expect("hierarchical transformer"),
                ),
            ),
            (
                "pyramid",
                Box::new(
                    PyramidTransformer::new(config.clone(), vocab_size)
                        .expect("pyramid transformer"),
                ),
            ),
            (
                "tree",
                Box::new(
                    TreeTransformer::new(config.clone(), vocab_size).expect("tree transformer"),
                ),
            ),
            (
                "nested",
                Box::new(NestedTransformer::new(config, vocab_size).expect("nested transformer")),
            ),
        ];

        for (name, model) in families {
            let out = model
                .forward(ids.clone())
                .unwrap_or_else(|error| panic!("{name} forward failed: {error}"));
            assert_eq!(out.output.shape(), expected_shape, "{name} output shape");

            let data = out.output.data().expect("output data");
            assert!(
                data.iter().all(|v| v.is_finite()),
                "{name} produced non-finite values"
            );
            assert!(
                data.iter().any(|v| v.abs() > 1e-6),
                "{name} returned an all-zero tensor"
            );

            let other_data = model
                .forward(other.clone())
                .expect("second forward")
                .output
                .data()
                .expect("data");
            assert!(
                data.iter().zip(other_data.iter()).any(|(x, y)| (x - y).abs() > 1e-6),
                "{name} output does not depend on the input tokens"
            );

            assert!(
                model.forward(Vec::new()).is_err(),
                "{name} accepted an empty token sequence"
            );
        }
    }

    /// The classification head must pool the first *token*, giving `[1, num_labels]`.
    ///
    /// With the rank-2 embedding output the base model either errored outright or
    /// (pyramid) handed `select(1, 0)` a `[seq, hidden]` tensor, which selects
    /// hidden channel 0 across the sequence rather than the CLS token's vector.
    #[test]
    fn test_classification_head_selects_the_cls_token() {
        use crate::hierarchical::models::HierarchicalForSequenceClassification;
        use trustformers_core::traits::Model;

        let num_labels = 3usize;
        let model = HierarchicalForSequenceClassification::new(tiny_config(), 64, num_labels)
            .expect("classification model");
        assert_eq!(model.num_labels(), num_labels);

        let logits = model.forward(vec![1, 2, 3, 4]).expect("classification forward");
        assert_eq!(logits.shape(), vec![1, num_labels]);

        let data = logits.data().expect("logits data");
        assert!(data.iter().all(|v| v.is_finite()));

        let other = model.forward(vec![7, 7, 1, 0]).expect("second forward").data().expect("data");
        assert!(
            data.iter().zip(other.iter()).any(|(x, y)| (x - y).abs() > 1e-6),
            "classification logits ignore the input"
        );
    }

    /// The language-modelling head must project every position onto the vocabulary.
    #[test]
    fn test_language_modeling_head_shape_and_signal() {
        use crate::hierarchical::models::HierarchicalForLanguageModeling;
        use trustformers_core::traits::Model;

        let vocab_size = 64usize;
        let model = HierarchicalForLanguageModeling::new(tiny_config(), vocab_size)
            .expect("language modelling head");
        assert_eq!(model.vocab_size(), vocab_size);

        let logits = model.forward(vec![1, 2, 3]).expect("lm forward");
        assert_eq!(logits.shape(), vec![1, 3, vocab_size]);
        let data = logits.data().expect("logits data");
        assert!(data.iter().all(|v| v.is_finite()));
        assert!(data.iter().any(|v| v.abs() > 1e-6));
    }

    /// The factory must not substitute a different architecture for `Hybrid`.
    #[test]
    fn test_factory_reports_hybrid_as_unimplemented() {
        use crate::hierarchical::models::create_hierarchical_transformer;

        for hierarchical_type in [
            HierarchicalType::Hierarchical,
            HierarchicalType::Pyramid,
            HierarchicalType::Tree,
            HierarchicalType::Nested,
        ] {
            let config = HierarchicalConfig {
                hierarchical_type,
                ..tiny_config()
            };
            assert!(create_hierarchical_transformer(config, 64).is_ok());
        }

        let hybrid = HierarchicalConfig {
            hierarchical_type: HierarchicalType::Hybrid,
            ..tiny_config()
        };
        assert!(
            create_hierarchical_transformer(hybrid, 64).is_err(),
            "Hybrid must not silently build a plain hierarchical transformer"
        );
    }

    /// The tree mask must follow the sequence length, not the configured maximum.
    ///
    /// The mask used to be built once for `max_seq_lengths[0]` and handed to
    /// attention verbatim, so any sequence of a different length failed with
    /// "Attention mask shape [16, 16] is not broadcastable to …".
    #[test]
    fn test_tree_attention_mask_follows_the_sequence_length() {
        use crate::hierarchical::config::{TreeConfig, TreeConstruction};
        use crate::hierarchical::layers::TreeAttention;
        use trustformers_core::traits::Layer;

        let config = HierarchicalConfig {
            hierarchical_type: HierarchicalType::Tree,
            tree_config: Some(TreeConfig {
                branching_factor: 2,
                tree_construction: TreeConstruction::Binary,
                ..TreeConfig::default()
            }),
            ..tiny_config()
        };
        let layer = TreeAttention::new(config.clone()).expect("tree attention");
        assert_eq!(
            layer.tree_mask().shape(),
            vec![config.max_seq_lengths[0], config.max_seq_lengths[0]]
        );

        // Shorter than the cached mask, exactly the cached length, and longer.
        for seq_len in [4usize, config.max_seq_lengths[0], 20] {
            let input = Tensor::from_vec(
                (0..seq_len * config.hidden_size).map(|i| (i as f32 * 0.011).sin()).collect(),
                &[1, seq_len, config.hidden_size],
            )
            .expect("input");
            let out = layer
                .forward(input)
                .unwrap_or_else(|error| panic!("seq_len {seq_len} failed: {error}"));
            assert_eq!(out.output.shape(), vec![1, seq_len, config.hidden_size]);
            assert!(out.output.data().expect("data").iter().all(|v| v.is_finite()));
        }

        // A rank-2 activation is reported rather than indexed blindly.
        let flat = Tensor::zeros(&[4, config.hidden_size]).expect("flat");
        assert!(layer.forward(flat).is_err());
    }

    /// The corner of a large tree mask must allow and deny exactly the same pairs
    /// as a mask built for that shorter length outright.
    ///
    /// This is the invariant `TreeAttention::mask_for` relies on when it slices
    /// instead of rebuilding. An additive tree mask only ever holds `0.0` (allowed)
    /// or `-inf` (forbidden), so agreeing on finiteness is agreeing on the mask.
    #[test]
    fn test_tree_mask_corner_equals_a_freshly_built_mask() {
        use crate::hierarchical::config::TreeConstruction;

        for construction in [TreeConstruction::Binary, TreeConstruction::Balanced] {
            let branching_factor = 3usize;
            let large = create_tree_mask(16, branching_factor, &construction)
                .expect("large mask")
                .data()
                .expect("large data");
            let small = create_tree_mask(5, branching_factor, &construction)
                .expect("small mask")
                .data()
                .expect("small data");

            for row in 0..5usize {
                for column in 0..5usize {
                    let corner = large[row * 16 + column];
                    let direct = small[row * 5 + column];
                    assert_eq!(
                        corner.is_finite(),
                        direct.is_finite(),
                        "{construction:?} disagrees at ({row}, {column})"
                    );
                }
            }
        }
    }

    /// `HierarchicalFeedForward` must likewise produce a real signal per level.
    #[test]
    fn test_hierarchical_feed_forward_is_nonzero() {
        use crate::hierarchical::layers::HierarchicalFeedForward;
        use trustformers_core::traits::Layer;

        let layer = HierarchicalFeedForward::new(tiny_config()).expect("feed forward creation");
        let input = Tensor::from_vec(
            (0..(8 * 32)).map(|i| (i as f32 * 0.02).sin()).collect(),
            &[1, 8, 32],
        )
        .expect("input");

        let out = layer.forward(input).expect("forward");
        assert_eq!(out.output.shape(), vec![1, 8, 32]);
        assert!(
            out.output.data().expect("data").iter().any(|v| v.abs() > 1e-6),
            "hierarchical feed-forward returned an all-zero tensor"
        );
    }
}
