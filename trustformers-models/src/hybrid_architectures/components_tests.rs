#[cfg(test)]
mod tests {
    use crate::hybrid_architectures::*;
    use trustformers_core::tensor::Tensor;

    #[test]
    fn test_hybrid_config_builder() {
        let config = HybridConfig::builder()
            .add_component(ArchitecturalComponent::Transformer {
                layers: 6,
                hidden_size: 512,
                num_heads: 8,
                variant: TransformerVariant::BERT,
            })
            .add_component(ArchitecturalComponent::CNN {
                layers: 3,
                channels: 64,
                kernel_size: 3,
                architecture: CNNArchitecture::ResNet,
            })
            .fusion_strategy(FusionStrategy::Parallel {
                fusion_method: ParallelFusionMethod::Concatenation,
            })
            .build()
            .expect("operation failed");

        assert_eq!(config.components.len(), 2);
        assert!(matches!(
            config.fusion_strategy,
            FusionStrategy::Parallel { .. }
        ));
    }

    #[test]
    fn test_hybrid_architecture_creation() {
        let config = HybridConfig::builder()
            .add_component(ArchitecturalComponent::Transformer {
                layers: 6,
                hidden_size: 512,
                num_heads: 8,
                variant: TransformerVariant::Standard,
            })
            .build()
            .expect("operation failed");

        let hybrid_arch = HybridArchitecture::new(config).expect("operation failed");
        assert_eq!(hybrid_arch.num_components(), 1);
    }

    #[test]
    fn test_component_activation() {
        let config = HybridConfig::builder()
            .add_component(ArchitecturalComponent::CNN {
                layers: 3,
                channels: 32,
                kernel_size: 3,
                architecture: CNNArchitecture::ResNet,
            })
            .build()
            .expect("operation failed");

        let mut hybrid_arch = HybridArchitecture::new(config).expect("operation failed");

        // Test component activation/deactivation
        assert!(hybrid_arch.set_component_active(0, false).is_ok());
        assert!(hybrid_arch.set_component_active(1, false).is_err()); // Invalid index
    }

    #[test]
    fn test_adaptive_config() {
        let adaptive_config = AdaptiveConfig {
            input_routing: true,
            performance_threshold: 0.8,
            confidence_threshold: 0.9,
            resource_budget: ResourceBudget {
                max_compute_time: 100.0,
                max_memory_mb: 1024.0,
                max_energy: 50.0,
            },
            adaptation_rate: 0.01,
        };

        assert_eq!(adaptive_config.performance_threshold, 0.8);
        assert!(adaptive_config.input_routing);
    }

    #[test]
    fn test_cross_modal_config() {
        let cross_modal_config = CrossModalConfig {
            modalities: vec![Modality::Text, Modality::Vision],
            fusion_points: vec![FusionPoint {
                component_indices: vec![0, 1],
                fusion_method: ParallelFusionMethod::CrossAttention,
                fusion_depth: 6,
            }],
            alignment_strategy: AlignmentStrategy::Contrastive,
            shared_repr_size: 512,
        };

        assert_eq!(cross_modal_config.modalities.len(), 2);
        assert_eq!(cross_modal_config.shared_repr_size, 512);
    }

    #[test]
    fn test_architecture_summary() {
        let config = HybridConfig::builder()
            .add_component(ArchitecturalComponent::Transformer {
                layers: 12,
                hidden_size: 768,
                num_heads: 12,
                variant: TransformerVariant::GPT,
            })
            .add_component(ArchitecturalComponent::CNN {
                layers: 5,
                channels: 128,
                kernel_size: 3,
                architecture: CNNArchitecture::EfficientNet,
            })
            .build()
            .expect("operation failed");

        let hybrid_arch = HybridArchitecture::new(config).expect("operation failed");
        let summary = hybrid_arch.get_architecture_summary();

        assert_eq!(summary.num_components, 2);
        assert_eq!(summary.component_types.len(), 2);
        // Components are instantiated lazily against the first input's width,
        // so before any forward pass there are genuinely no parameters. The
        // previous `components.len() * 1_000_000` estimate reported a figure
        // for a network that had not been built yet.
        assert_eq!(summary.total_parameters, 0);
        assert_eq!(summary.memory_usage, 0.0);
    }

    #[test]
    fn test_fusion_strategies() {
        // Test different fusion strategies
        let strategies = vec![
            FusionStrategy::Sequential,
            FusionStrategy::Parallel {
                fusion_method: ParallelFusionMethod::Addition,
            },
            FusionStrategy::Hierarchical {
                hierarchy_type: HierarchyType::BottomUp,
            },
            FusionStrategy::Ensemble {
                combination_method: EnsembleMethod::WeightedAveraging,
            },
        ];

        for strategy in strategies {
            let config = HybridConfig::builder()
                .add_component(ArchitecturalComponent::RNN {
                    layers: 2,
                    hidden_size: 256,
                    cell_type: RNNCellType::LSTM,
                    bidirectional: true,
                })
                .fusion_strategy(strategy)
                .build();

            assert!(config.is_ok());
        }
    }

    #[test]
    fn test_component_types() {
        let components = vec![
            ArchitecturalComponent::Transformer {
                layers: 6,
                hidden_size: 512,
                num_heads: 8,
                variant: TransformerVariant::BERT,
            },
            ArchitecturalComponent::CNN {
                layers: 4,
                channels: 96,
                kernel_size: 5,
                architecture: CNNArchitecture::MobileNet,
            },
            ArchitecturalComponent::RNN {
                layers: 3,
                hidden_size: 384,
                cell_type: RNNCellType::GRU,
                bidirectional: false,
            },
            ArchitecturalComponent::StateSpace {
                layers: 8,
                state_size: 256,
                model_type: StateSpaceType::Mamba,
            },
            ArchitecturalComponent::Attention {
                attention_type: AttentionType::MultiHead,
                num_heads: 16,
                key_dim: 64,
            },
        ];

        for component in components {
            let config = HybridConfig::builder()
                .add_component(component)
                .build()
                .expect("operation failed");

            let hybrid_arch = HybridArchitecture::new(config).expect("operation failed");
            assert_eq!(hybrid_arch.num_components(), 1);
        }
    }

    // ------------------------------------------------------------------
    // Regression tests: every component forward and every fusion / ensemble
    // combinator used to return its input (or `outputs[0]`) unchanged.
    // ------------------------------------------------------------------

    use crate::hybrid_architectures::components::{
        activation_energy, prediction_confidence, ComponentModule, FusionOperator,
    };

    fn sequence(batch: usize, seq: usize, width: usize, seed: f32) -> Tensor {
        let values: Vec<f32> =
            (0..batch * seq * width).map(|i| ((i as f32) * 0.37 + seed).sin()).collect();
        Tensor::from_vec(values, &[batch, seq, width]).expect("input")
    }

    fn build(component: ArchitecturalComponent, width: usize) -> ComponentModule {
        ComponentModule::build(&component, width).expect("module")
    }

    #[test]
    fn test_transformer_component_is_not_the_identity() {
        let module = build(
            ArchitecturalComponent::Transformer {
                layers: 2,
                hidden_size: 8,
                num_heads: 2,
                variant: TransformerVariant::Standard,
            },
            8,
        );
        let input = sequence(1, 4, 8, 0.1);
        let output = module.forward(&input).expect("forward");

        assert_eq!(output.shape(), input.shape());
        let a = input.to_vec_f32().expect("a");
        let b = output.to_vec_f32().expect("b");
        assert!(
            a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-5),
            "the transformer component must transform its input"
        );
        assert!(b.iter().all(|v| v.is_finite()));
        assert!(module.parameter_count() > 0);
    }

    #[test]
    fn test_attention_component_mixes_positions() {
        let module = build(
            ArchitecturalComponent::Attention {
                attention_type: AttentionType::SelfAttention,
                num_heads: 2,
                key_dim: 8,
            },
            8,
        );
        let input = sequence(1, 3, 8, 0.5);
        let output = module.forward(&input).expect("forward");
        assert_eq!(output.shape(), input.shape());
        let a = input.to_vec_f32().expect("a");
        let b = output.to_vec_f32().expect("b");
        assert!(a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-5));
    }

    #[test]
    fn test_recurrent_component_is_order_dependent() {
        for cell_type in [RNNCellType::RNN, RNNCellType::GRU, RNNCellType::LSTM] {
            let module = build(
                ArchitecturalComponent::RNN {
                    layers: 1,
                    hidden_size: 6,
                    cell_type: cell_type.clone(),
                    bidirectional: false,
                },
                6,
            );

            let forward_input = sequence(1, 4, 6, 0.2);
            let mut reversed = forward_input.to_vec_f32().expect("v");
            reversed.chunks_mut(6).collect::<Vec<_>>().reverse();
            let reversed_input =
                Tensor::from_vec(reverse_steps(&forward_input, 4, 6), &[1, 4, 6]).expect("rev");

            let a = module.forward(&forward_input).expect("forward");
            let b = module.forward(&reversed_input).expect("forward");
            assert_eq!(a.shape(), vec![1, 4, 6]);
            assert!(
                a.to_vec_f32()
                    .expect("a")
                    .iter()
                    .zip(b.to_vec_f32().expect("b").iter())
                    .any(|(x, y)| (x - y).abs() > 1e-5),
                "{cell_type:?} recurrence must depend on the order of the sequence"
            );
        }
    }

    fn reverse_steps(tensor: &Tensor, seq: usize, width: usize) -> Vec<f32> {
        let values = tensor.to_vec_f32().expect("v");
        let mut out = vec![0.0f32; values.len()];
        for step in 0..seq {
            let source = (seq - 1 - step) * width;
            out[step * width..(step + 1) * width].copy_from_slice(&values[source..source + width]);
        }
        out
    }

    #[test]
    fn test_convolutional_component_mixes_neighbouring_steps() {
        let module = build(
            ArchitecturalComponent::CNN {
                layers: 1,
                channels: 4,
                kernel_size: 3,
                architecture: CNNArchitecture::ResNet,
            },
            4,
        );
        // Only position 1 is non-zero; a real convolution spreads it to 0 and 2.
        let mut values = vec![0.0f32; 3 * 4];
        for feature in 0..4 {
            values[4 + feature] = 1.0;
        }
        let input = Tensor::from_vec(values, &[1, 3, 4]).expect("input");
        let output = module.forward(&input).expect("forward").to_vec_f32().expect("o");

        let energy_at_0: f32 = output[0..4].iter().map(|v| v.abs()).sum();
        let energy_at_2: f32 = output[8..12].iter().map(|v| v.abs()).sum();
        assert!(
            energy_at_0 > 0.0 && energy_at_2 > 0.0,
            "a kernel of width 3 must leak into both neighbours: {output:?}"
        );
    }

    #[test]
    fn test_state_space_component_accumulates_over_time() {
        let module = build(
            ArchitecturalComponent::StateSpace {
                layers: 1,
                state_size: 4,
                model_type: StateSpaceType::Diagonal,
            },
            4,
        );
        // A single impulse at t = 0 must still be visible at later steps.
        let mut values = vec![0.0f32; 4 * 4];
        for feature in 0..4 {
            values[feature] = 1.0;
        }
        let input = Tensor::from_vec(values, &[1, 4, 4]).expect("input");
        let output = module.forward(&input).expect("forward").to_vec_f32().expect("o");

        let tail: f32 = output[12..16].iter().map(|v| v.abs()).sum();
        assert!(
            tail > 0.0,
            "the state must decay, not vanish instantly: {output:?}"
        );
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_memory_component_reads_from_its_bank() {
        let module = build(
            ArchitecturalComponent::Memory {
                memory_type: MemoryType::ExternalMemory,
                memory_size: 8,
                addressing: AddressingMode::ContentBased,
            },
            4,
        );
        let input = sequence(1, 2, 4, 0.9);
        let output = module.forward(&input).expect("forward");
        assert_eq!(output.shape(), input.shape());
        assert!(output
            .to_vec_f32()
            .expect("o")
            .iter()
            .zip(input.to_vec_f32().expect("i").iter())
            .any(|(a, b)| (a - b).abs() > 1e-9));
    }

    #[test]
    fn test_unsupported_components_report_instead_of_passing_through() {
        assert!(ComponentModule::build(
            &ArchitecturalComponent::GNN {
                layers: 1,
                hidden_size: 4,
                graph_type: GraphType::GCN,
            },
            4
        )
        .is_err());

        assert!(ComponentModule::build(
            &ArchitecturalComponent::Custom {
                name: "mystery".to_string(),
                parameters: HashMap::new(),
                config: HashMap::new(),
            },
            4
        )
        .is_err());
    }

    // --- fusion ---

    fn fusion_inputs() -> Vec<Tensor> {
        vec![
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2]).expect("a"),
            Tensor::from_vec(vec![-1.0, 0.5, 2.0, -3.0], &[1, 2, 2]).expect("b"),
        ]
    }

    #[test]
    fn test_every_fusion_method_uses_all_of_its_inputs() {
        // Regression: all six methods used to return `outputs[0].clone()`.
        let inputs = fusion_inputs();
        let first = inputs[0].to_vec_f32().expect("a");

        for method in [
            ParallelFusionMethod::Concatenation,
            ParallelFusionMethod::Addition,
            ParallelFusionMethod::Multiplication,
            ParallelFusionMethod::Gating,
            ParallelFusionMethod::CrossAttention,
            ParallelFusionMethod::MultiModal,
        ] {
            let fused = FusionOperator::new().fuse(&inputs, &method).expect("fuse");
            assert_eq!(
                fused.shape(),
                inputs[0].shape(),
                "{method:?} must keep the shape"
            );
            let values = fused.to_vec_f32().expect("f");
            assert!(
                values.iter().zip(first.iter()).any(|(x, y)| (x - y).abs() > 1e-6),
                "{method:?} must not return its first input unchanged: {values:?}"
            );
            assert!(values.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn test_additive_and_multiplicative_fusion_match_their_definitions() {
        let inputs = fusion_inputs();

        let added = FusionOperator::new()
            .fuse(&inputs, &ParallelFusionMethod::Addition)
            .expect("add")
            .to_vec_f32()
            .expect("v");
        assert!((added[0] - 0.0).abs() < 1e-6, "mean of 1.0 and -1.0 is 0");
        assert!(
            (added[1] - 1.25).abs() < 1e-6,
            "mean of 2.0 and 0.5 is 1.25"
        );

        let multiplied = FusionOperator::new()
            .fuse(&inputs, &ParallelFusionMethod::Multiplication)
            .expect("mul")
            .to_vec_f32()
            .expect("v");
        assert!((multiplied[0] + 1.0).abs() < 1e-6);
        assert!((multiplied[3] + 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_single_input_fusion_is_the_identity() {
        let inputs = vec![fusion_inputs()[0].clone()];
        let fused = FusionOperator::new()
            .fuse(&inputs, &ParallelFusionMethod::Gating)
            .expect("fuse");
        assert_eq!(
            fused.to_vec_f32().expect("f"),
            inputs[0].to_vec_f32().expect("i")
        );
        assert!(FusionOperator::new().fuse(&[], &ParallelFusionMethod::Addition).is_err());
    }

    // --- ensembles ---

    #[test]
    fn test_ensemble_combinators_use_every_member() {
        let outputs = fusion_inputs();
        let weights = [0.2f32, 0.8];
        let first = outputs[0].to_vec_f32().expect("a");

        for method in [
            EnsembleMethod::WeightedAveraging,
            EnsembleMethod::Bagging,
            EnsembleMethod::Stacking,
            EnsembleMethod::Boosting,
        ] {
            let combined = FusionOperator::new()
                .combine_ensemble(&outputs, &weights, &method)
                .expect("combine");
            let values = combined.to_vec_f32().expect("v");
            assert!(
                values.iter().zip(first.iter()).any(|(x, y)| (x - y).abs() > 1e-6),
                "{method:?} must combine the members: {values:?}"
            );
        }
    }

    #[test]
    fn test_weighted_averaging_matches_its_weights() {
        let outputs = fusion_inputs();
        let combined = FusionOperator::new()
            .combine_ensemble(&outputs, &[0.25, 0.75], &EnsembleMethod::WeightedAveraging)
            .expect("combine")
            .to_vec_f32()
            .expect("v");
        // 0.25 * 1.0 + 0.75 * -1.0 = -0.5
        assert!((combined[0] + 0.5).abs() < 1e-6, "{combined:?}");
    }

    #[test]
    fn test_dynamic_selection_picks_the_best_member() {
        let outputs = fusion_inputs();
        let selected = FusionOperator::new()
            .combine_ensemble(&outputs, &[0.1, 0.9], &EnsembleMethod::DynamicSelection)
            .expect("combine");
        assert_eq!(
            selected.to_vec_f32().expect("v"),
            outputs[1].to_vec_f32().expect("v")
        );
    }

    #[test]
    fn test_majority_voting_returns_a_one_hot_consensus() {
        let members = vec![
            Tensor::from_vec(vec![3.0, 0.0, 0.0], &[1, 3]).expect("a"),
            Tensor::from_vec(vec![2.0, 1.0, 0.0], &[1, 3]).expect("b"),
            Tensor::from_vec(vec![0.0, 5.0, 0.0], &[1, 3]).expect("c"),
        ];
        let combined = FusionOperator::new()
            .combine_ensemble(&members, &[1.0, 1.0, 1.0], &EnsembleMethod::MajorityVoting)
            .expect("vote")
            .to_vec_f32()
            .expect("v");
        assert_eq!(
            combined,
            vec![1.0, 0.0, 0.0],
            "two of three members chose class 0"
        );
    }

    // --- routing and metrics ---

    #[test]
    fn test_confidence_and_energy_are_measured() {
        // Regression: performance was a hardcoded 0.85 and every routing
        // decision returned component 0.
        let confident = Tensor::from_vec(vec![10.0, -10.0, -10.0], &[1, 3]).expect("c");
        let ambiguous = Tensor::from_vec(vec![0.0, 0.0, 0.0], &[1, 3]).expect("a");

        let high = prediction_confidence(&confident).expect("high");
        let low = prediction_confidence(&ambiguous).expect("low");
        assert!(
            high > low,
            "a peaked output must be more confident ({high} vs {low})"
        );
        assert!((0.0..=1.0).contains(&high));
        assert!(low.abs() < 1e-6);

        assert!(activation_energy(&confident).expect("e") > 0.0);
        assert!(activation_energy(&ambiguous).expect("e").abs() < 1e-9);
    }

    #[test]
    fn test_architecture_summary_counts_real_parameters() {
        // Regression: the summary reported `components.len() * 1_000_000`.
        let config = HybridConfig::builder()
            .add_component(ArchitecturalComponent::Transformer {
                layers: 1,
                hidden_size: 8,
                num_heads: 2,
                variant: TransformerVariant::Standard,
            })
            .build()
            .expect("config");
        let mut architecture = HybridArchitecture::new(config).expect("architecture");

        let before = architecture.get_architecture_summary();
        assert_eq!(
            before.total_parameters, 0,
            "a component that has not run yet has no instantiated parameters"
        );

        let input = sequence(1, 3, 8, 0.4);
        architecture.forward(&[input]).expect("forward");

        let after = architecture.get_architecture_summary();
        assert!(after.total_parameters > 0);
        assert_ne!(
            after.total_parameters, 1_000_000,
            "the parameter count must be measured, not a round placeholder"
        );
        assert!(after.memory_usage > 0.0);
        assert!(after.computational_complexity > 0.0);
    }

    #[test]
    fn test_sequential_forward_transforms_the_input() {
        let config = HybridConfig::builder()
            .add_component(ArchitecturalComponent::Attention {
                attention_type: AttentionType::SelfAttention,
                num_heads: 2,
                key_dim: 8,
            })
            .add_component(ArchitecturalComponent::RNN {
                layers: 1,
                hidden_size: 8,
                cell_type: RNNCellType::GRU,
                bidirectional: false,
            })
            .fusion_strategy(FusionStrategy::Sequential)
            .build()
            .expect("config");
        let mut architecture = HybridArchitecture::new(config).expect("architecture");

        let input = sequence(1, 3, 8, 0.6);
        let output = architecture.forward(std::slice::from_ref(&input)).expect("forward");
        assert_eq!(output.shape(), input.shape());
        assert!(output
            .to_vec_f32()
            .expect("o")
            .iter()
            .zip(input.to_vec_f32().expect("i").iter())
            .any(|(a, b)| (a - b).abs() > 1e-5));
    }

    #[test]
    fn test_custom_fusion_strategy_reports_instead_of_aliasing_sequential() {
        let config = HybridConfig::builder()
            .add_component(ArchitecturalComponent::Attention {
                attention_type: AttentionType::SelfAttention,
                num_heads: 1,
                key_dim: 4,
            })
            .fusion_strategy(FusionStrategy::Custom {
                name: "mystery".to_string(),
                parameters: HashMap::new(),
            })
            .build()
            .expect("config");
        let mut architecture = HybridArchitecture::new(config).expect("architecture");
        assert!(architecture.forward(&[sequence(1, 2, 4, 0.1)]).is_err());
    }

    #[test]
    fn test_fusion_parameters_persist_across_calls() {
        // Regression: building the projection inside `fuse` re-randomized the
        // weights on every call, so the same inputs gave different outputs.
        let inputs = fusion_inputs();
        let mut operator = FusionOperator::new();

        for method in [
            ParallelFusionMethod::Concatenation,
            ParallelFusionMethod::Gating,
            ParallelFusionMethod::CrossAttention,
        ] {
            let first = operator.fuse(&inputs, &method).expect("first");
            let second = operator.fuse(&inputs, &method).expect("second");
            assert_eq!(
                first.to_vec_f32().expect("a"),
                second.to_vec_f32().expect("b"),
                "{method:?} must be deterministic across calls"
            );
        }
        assert!(
            operator.parameter_count() > 0,
            "the operator must own the projections it uses"
        );
    }

    #[test]
    fn test_stacking_ensemble_is_deterministic() {
        let outputs = fusion_inputs();
        let mut operator = FusionOperator::new();
        let first = operator
            .combine_ensemble(&outputs, &[0.5, 0.5], &EnsembleMethod::Stacking)
            .expect("first");
        let second = operator
            .combine_ensemble(&outputs, &[0.5, 0.5], &EnsembleMethod::Stacking)
            .expect("second");
        assert_eq!(
            first.to_vec_f32().expect("a"),
            second.to_vec_f32().expect("b")
        );
    }

    #[test]
    fn test_architecture_forward_is_deterministic() {
        let config = HybridConfig::builder()
            .add_component(ArchitecturalComponent::Attention {
                attention_type: AttentionType::SelfAttention,
                num_heads: 2,
                key_dim: 8,
            })
            .add_component(ArchitecturalComponent::RNN {
                layers: 1,
                hidden_size: 8,
                cell_type: RNNCellType::GRU,
                bidirectional: false,
            })
            .fusion_strategy(FusionStrategy::Parallel {
                fusion_method: ParallelFusionMethod::Concatenation,
            })
            .build()
            .expect("config");
        let mut architecture = HybridArchitecture::new(config).expect("architecture");
        let input = sequence(1, 3, 8, 0.3);

        let first = architecture.forward(std::slice::from_ref(&input)).expect("first");
        let second = architecture.forward(std::slice::from_ref(&input)).expect("second");
        assert_eq!(
            first.to_vec_f32().expect("a"),
            second.to_vec_f32().expect("b"),
            "a forward pass must not re-randomize its own parameters"
        );
    }
}
