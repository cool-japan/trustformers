#[cfg(test)]
mod tests {
    use crate::continual_learning::*;
    use std::collections::HashMap;
    use trustformers_core::tensor::Tensor;

    #[test]
    fn test_continual_learning_config_default() {
        let config = ContinualLearningConfig::default();
        assert_eq!(config.memory_size, 1000);
        assert!(config.task_specific_heads);
        assert!(!config.automatic_task_detection);

        if let ContinualStrategy::ElasticWeightConsolidation {
            lambda,
            fisher_samples,
        } = config.strategy
        {
            assert_eq!(lambda, 0.4);
            assert_eq!(fisher_samples, 1000);
        } else {
            panic!("Expected EWC strategy");
        }
    }

    #[test]
    fn test_memory_buffer() {
        let mut buffer = MemoryBuffer::new(3, MemorySelectionStrategy::Random);
        assert!(buffer.is_empty());
        assert_eq!(buffer.size(), 0);

        // Add examples
        let input1 = Tensor::zeros(&[1, 10]).expect("operation failed");
        let target1 = Tensor::zeros(&[1]).expect("operation failed");
        buffer.add_example(input1, target1, 0, 1.0);
        assert_eq!(buffer.size(), 1);

        let input2 = Tensor::ones(&[1, 10]).expect("operation failed");
        let target2 = Tensor::ones(&[1]).expect("operation failed");
        buffer.add_example(input2, target2, 1, 2.0);
        assert_eq!(buffer.size(), 2);

        // Sample batch
        let (inputs, targets, task_ids) = buffer.sample_batch(2).expect("operation failed");
        assert_eq!(inputs.len(), 2);
        assert_eq!(targets.len(), 2);
        assert_eq!(task_ids.len(), 2);
    }

    #[test]
    fn test_ewc_config() {
        let config = utils::ewc_config(0.5, 2000, 500);
        assert_eq!(config.memory_size, 500);

        if let ContinualStrategy::ElasticWeightConsolidation {
            lambda,
            fisher_samples,
        } = config.strategy
        {
            assert_eq!(lambda, 0.5);
            assert_eq!(fisher_samples, 2000);
        } else {
            panic!("Expected EWC strategy");
        }
    }

    #[test]
    fn test_experience_replay_config() {
        let config = utils::experience_replay_config(1000, 64);
        assert_eq!(config.memory_size, 1000);

        if let ContinualStrategy::ExperienceReplay {
            memory_strength,
            replay_batch_size,
        } = config.strategy
        {
            assert_eq!(memory_strength, 1.0);
            assert_eq!(replay_batch_size, 64);
        } else {
            panic!("Expected ExperienceReplay strategy");
        }
    }

    #[test]
    fn test_l2_regularization_config() {
        let config = utils::l2_regularization_config(0.01);
        assert_eq!(config.memory_size, 0);

        if let ContinualStrategy::L2Regularization { lambda } = config.strategy {
            assert_eq!(lambda, 0.01);
        } else {
            panic!("Expected L2Regularization strategy");
        }
    }

    #[test]
    fn test_task_info() {
        let mut info = TaskInfo::new(5);
        assert_eq!(info.task_id, 5);
        assert_eq!(info.num_examples_seen, 0);

        info.update_statistics(0.5);
        assert_eq!(info.num_examples_seen, 1);
        assert_eq!(info.average_loss, 0.5);

        info.update_statistics(1.0);
        assert_eq!(info.num_examples_seen, 2);
        assert_eq!(info.average_loss, 0.75);
    }

    #[test]
    fn test_backward_transfer_computation() {
        let mut before = HashMap::new();
        before.insert(
            0,
            TaskEvaluation {
                task_id: 0,
                average_loss: 0.5,
                accuracy: 0.8,
                num_examples: 100,
            },
        );
        before.insert(
            1,
            TaskEvaluation {
                task_id: 1,
                average_loss: 0.6,
                accuracy: 0.7,
                num_examples: 100,
            },
        );

        let mut after = HashMap::new();
        after.insert(
            0,
            TaskEvaluation {
                task_id: 0,
                average_loss: 0.4,
                accuracy: 0.85,
                num_examples: 100,
            },
        );
        after.insert(
            1,
            TaskEvaluation {
                task_id: 1,
                average_loss: 0.55,
                accuracy: 0.72,
                num_examples: 100,
            },
        );

        let backward_transfer = utils::compute_backward_transfer(&before, &after);
        assert!((backward_transfer - 0.035).abs() < 1e-6); // (0.05 + 0.02) / 2
    }

    #[test]
    fn test_forgetting_computation() {
        let mut max_accuracies = HashMap::new();
        max_accuracies.insert(0, 0.9);
        max_accuracies.insert(1, 0.85);

        let mut final_accuracies = HashMap::new();
        final_accuracies.insert(0, 0.8);
        final_accuracies.insert(1, 0.75);

        let forgetting = utils::compute_forgetting(&max_accuracies, &final_accuracies);
        assert!((forgetting - 0.1).abs() < 1e-6); // (0.1 + 0.1) / 2
    }

    // ------------------------------------------------------------------
    // Regression tests for the previously fabricated continual-learning
    // machinery (Fisher = ones([10]), accuracy always 1.0, zero penalties).
    // ------------------------------------------------------------------

    use serde::{Deserialize, Serialize};
    use trustformers_core::traits::{Config, Model};
    use trustformers_core::Result;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct TinyConfig;

    impl Config for TinyConfig {
        fn architecture(&self) -> &'static str {
            "tiny"
        }
    }

    /// A two-class linear model: `logits = input * weight`, with the weight
    /// exposed through [`NamedParameters`] so the trainer can regularize it.
    struct TinyModel {
        weight: Tensor,
        config: TinyConfig,
    }

    impl TinyModel {
        fn new(values: &[f32]) -> Self {
            Self {
                weight: Tensor::from_vec(values.to_vec(), &[1, values.len()])
                    .expect("weight tensor"),
                config: TinyConfig,
            }
        }
    }

    impl Model for TinyModel {
        type Config = TinyConfig;
        type Input = Tensor;
        type Output = Tensor;

        fn forward(&self, input: Tensor) -> Result<Tensor> {
            input.mul(&self.weight)
        }

        fn load_pretrained(&mut self, _reader: &mut dyn std::io::Read) -> Result<()> {
            Ok(())
        }

        fn get_config(&self) -> &TinyConfig {
            &self.config
        }

        fn num_parameters(&self) -> usize {
            self.weight.to_vec_f32().map(|v| v.len()).unwrap_or(0)
        }
    }

    impl NamedParameters for TinyModel {
        fn named_parameters(&self) -> Vec<(String, Tensor)> {
            vec![("weight".to_string(), self.weight.clone())]
        }

        fn set_named_parameter(&mut self, name: &str, value: Tensor) -> Result<()> {
            if name != "weight" {
                return Err(trustformers_core::errors::invalid_input(format!(
                    "unknown parameter {}",
                    name
                )));
            }
            self.weight = value;
            Ok(())
        }
    }

    fn trainer_with(
        strategy: ContinualStrategy,
        weights: &[f32],
    ) -> ContinualLearningTrainer<TinyModel> {
        let config = ContinualLearningConfig {
            strategy,
            memory_size: 16,
            ..Default::default()
        };
        ContinualLearningTrainer::new(TinyModel::new(weights), config).expect("trainer")
    }

    fn one_hot(index: usize, classes: usize) -> Tensor {
        let mut data = vec![0.0f32; classes];
        data[index] = 1.0;
        Tensor::from_vec(data, &[1, classes]).expect("one hot")
    }

    #[test]
    fn test_evaluate_task_is_not_always_perfect() {
        // Regression: accuracy used to compare two freshly created zero
        // tensors, so it was exactly 1.0 for every model and every input.
        let trainer = trainer_with(
            ContinualStrategy::L2Regularization { lambda: 0.0 },
            &[2.0, 1.0],
        );

        let input = Tensor::from_vec(vec![1.0, 1.0], &[1, 2]).expect("input");
        // The model predicts class 0 (weight 2.0 > 1.0); the label says class 1.
        let wrong_target = one_hot(1, 2);
        let evaluation = trainer
            .evaluate_task(std::slice::from_ref(&input), &[wrong_target], 0)
            .expect("evaluate");
        assert_eq!(
            evaluation.accuracy, 0.0,
            "a wrong prediction must not score 1.0"
        );

        let right_target = one_hot(0, 2);
        let evaluation = trainer.evaluate_task(&[input], &[right_target], 0).expect("evaluate");
        assert_eq!(evaluation.accuracy, 1.0);
    }

    #[test]
    fn test_evaluate_task_accepts_class_index_targets() {
        let trainer = trainer_with(
            ContinualStrategy::L2Regularization { lambda: 0.0 },
            &[2.0, 1.0],
        );
        let input = Tensor::from_vec(vec![1.0, 1.0], &[1, 2]).expect("input");
        let index_target = Tensor::from_vec(vec![0.0], &[1]).expect("target");
        let evaluation = trainer.evaluate_task(&[input], &[index_target], 3).expect("evaluate");
        assert_eq!(evaluation.accuracy, 1.0);
        assert_eq!(evaluation.task_id, 3);
        assert_eq!(evaluation.num_examples, 1);
    }

    #[test]
    fn test_fisher_information_is_measured_not_a_constant() {
        // Regression: Fisher used to be `Tensor::ones(&[10])` keyed by
        // `param_{i}` regardless of the model.
        let mut trainer = trainer_with(
            ContinualStrategy::ElasticWeightConsolidation {
                lambda: 1.0,
                fisher_samples: 4,
            },
            &[0.5, -0.25],
        );

        // Two examples whose features differ a lot between the coordinates, so
        // the two parameters cannot receive the same Fisher value.
        trainer.memory.add_example(
            Tensor::from_vec(vec![3.0, 0.1], &[1, 2]).expect("input"),
            one_hot(0, 2),
            0,
            1.0,
        );
        trainer.memory.add_example(
            Tensor::from_vec(vec![2.5, 0.2], &[1, 2]).expect("input"),
            one_hot(0, 2),
            0,
            1.0,
        );

        trainer.compute_fisher_information(0, 2).expect("fisher");

        assert!(
            trainer.fisher_matrices.contains_key("weight"),
            "Fisher must be keyed by real parameter names, got {:?}",
            trainer.fisher_matrices.keys().collect::<Vec<_>>()
        );
        let fisher = trainer.fisher_matrices["weight"].to_vec_f32().expect("data");
        assert_eq!(fisher.len(), 2);
        assert!(fisher.iter().all(|f| *f >= 0.0));
        assert!(
            (fisher[0] - fisher[1]).abs() > 1e-6,
            "Fisher must reflect the data, not be a constant: {fisher:?}"
        );
        assert!(
            fisher.iter().any(|f| (*f - 1.0).abs() > 1e-6),
            "Fisher must not be all ones: {fisher:?}"
        );
    }

    #[test]
    fn test_ewc_penalty_is_zero_at_the_anchor_and_grows_away_from_it() {
        // Regression: the penalty compared two zero tensors and was constant.
        let mut trainer = trainer_with(
            ContinualStrategy::ElasticWeightConsolidation {
                lambda: 1.0,
                fisher_samples: 2,
            },
            &[0.5, -0.25],
        );
        trainer.memory.add_example(
            Tensor::from_vec(vec![1.5, 0.5], &[1, 2]).expect("input"),
            one_hot(0, 2),
            0,
            1.0,
        );
        trainer.compute_fisher_information(0, 1).expect("fisher");
        trainer.save_optimal_parameters().expect("anchor");

        let at_anchor = trainer.compute_ewc_loss(1.0).expect("ewc");
        assert!(
            at_anchor.to_vec_f32().expect("data")[0].abs() < 1e-9,
            "the EWC penalty must vanish at the saved optimum"
        );

        trainer
            .model
            .set_named_parameter(
                "weight",
                Tensor::from_vec(vec![5.0, -3.0], &[1, 2]).expect("moved"),
            )
            .expect("move");
        let moved = trainer.compute_ewc_loss(1.0).expect("ewc");
        assert!(
            moved.to_vec_f32().expect("data")[0] > 1e-6,
            "the EWC penalty must grow once the parameters drift"
        );
    }

    #[test]
    fn test_l2_penalty_anchors_to_the_snapshot() {
        let mut trainer = trainer_with(
            ContinualStrategy::L2Regularization { lambda: 2.0 },
            &[1.0, -1.0],
        );

        // Without a snapshot this is plain weight decay: 2 * (1 + 1) = 4.
        let decay = trainer.compute_l2_regularization(2.0).expect("l2");
        assert!((decay.to_vec_f32().expect("d")[0] - 4.0).abs() < 1e-5);

        trainer.save_optimal_parameters().expect("anchor");
        let at_anchor = trainer.compute_l2_regularization(2.0).expect("l2");
        assert!(at_anchor.to_vec_f32().expect("d")[0].abs() < 1e-9);

        trainer
            .model
            .set_named_parameter(
                "weight",
                Tensor::from_vec(vec![2.0, -1.0], &[1, 2]).expect("moved"),
            )
            .expect("move");
        let moved = trainer.compute_l2_regularization(2.0).expect("l2");
        assert!((moved.to_vec_f32().expect("d")[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_lwf_distillation_is_zero_without_a_snapshot_and_positive_after_drift() {
        let mut trainer = trainer_with(
            ContinualStrategy::LearningWithoutForgetting {
                lambda: 1.0,
                temperature: 2.0,
            },
            &[1.0, 0.0],
        );
        let inputs = [Tensor::from_vec(vec![1.0, 1.0], &[1, 2]).expect("input")];

        let before = trainer.compute_lwf_loss(&inputs, 1.0, 2.0).expect("lwf");
        assert!(before.to_vec_f32().expect("d")[0].abs() < 1e-12);

        trainer.save_optimal_parameters().expect("anchor");
        let at_anchor = trainer.compute_lwf_loss(&inputs, 1.0, 2.0).expect("lwf");
        assert!(
            at_anchor.to_vec_f32().expect("d")[0].abs() < 1e-5,
            "distillation against an identical teacher must vanish"
        );

        trainer
            .model
            .set_named_parameter(
                "weight",
                Tensor::from_vec(vec![0.0, 4.0], &[1, 2]).expect("moved"),
            )
            .expect("move");
        let drifted = trainer.compute_lwf_loss(&inputs, 1.0, 2.0).expect("lwf");
        assert!(
            drifted.to_vec_f32().expect("d")[0] > 1e-4,
            "distillation must penalise divergence from the old task"
        );

        // The temporary parameter swap must leave the live model untouched.
        assert_eq!(
            trainer.model.named_parameters()[0].1.to_vec_f32().expect("d"),
            vec![0.0, 4.0]
        );
    }

    #[test]
    fn test_gem_penalty_only_fires_on_a_real_constraint_violation() {
        let mut trainer = trainer_with(
            ContinualStrategy::GradientEpisodicMemory {
                memory_strength: 1.0,
                constraint_violation_threshold: 0.0,
            },
            &[0.5, 0.5],
        );
        let input = Tensor::from_vec(vec![1.0, 0.5], &[1, 2]).expect("input");
        let target = one_hot(0, 2);
        let current = Tensor::from_vec(vec![1.0], &[1]).expect("loss");

        // No memory: the loss passes through unchanged.
        let passthrough =
            trainer.compute_gem_loss(&input, &target, &current, 1.0, 0.0).expect("gem");
        assert!((passthrough.to_vec_f32().expect("d")[0] - 1.0).abs() < 1e-9);

        // An identical memory example has a positive gradient inner product,
        // so the constraint is satisfied and nothing is added.
        trainer.memory.add_example(input.clone(), target.clone(), 0, 1.0);
        let satisfied = trainer.compute_gem_loss(&input, &target, &current, 1.0, 0.0).expect("gem");
        assert!((satisfied.to_vec_f32().expect("d")[0] - 1.0).abs() < 1e-4);

        // A memory example pushing the opposite way violates the constraint.
        trainer.memory.clear();
        trainer.memory.add_example(input.clone(), one_hot(1, 2), 0, 1.0);
        let violated = trainer.compute_gem_loss(&input, &target, &current, 1.0, 0.0).expect("gem");
        assert!(
            violated.to_vec_f32().expect("d")[0] >= 1.0,
            "a violated GEM constraint must not reduce the loss"
        );
    }

    #[test]
    fn test_packnet_pruning_zeroes_the_smallest_weights() {
        let mut trainer = trainer_with(
            ContinualStrategy::PackNet {
                prune_ratio: 0.5,
                retrain_epochs: 0,
            },
            &[0.1, 5.0, -0.2, -4.0],
        );

        trainer.finalize_task(0).expect("finalize");
        let weights = trainer.model.named_parameters()[0].1.to_vec_f32().expect("w");
        assert_eq!(weights[0], 0.0, "the smallest weight must be pruned");
        assert_eq!(weights[2], 0.0);
        assert_eq!(weights[1], 5.0, "large weights survive");
        assert_eq!(weights[3], -4.0);

        let mask = &trainer.packnet_masks["weight"];
        assert_eq!(mask, &vec![false, true, false, true]);
    }

    #[test]
    fn test_progressive_networks_reports_the_missing_capability() {
        // Honest failure beats a silent no-op: growing a new network column
        // cannot be expressed through the Model trait.
        let mut trainer = trainer_with(
            ContinualStrategy::ProgressiveNeuralNetworks {
                lateral_connections: true,
                adapter_layers: true,
            },
            &[1.0, 1.0],
        );
        assert!(trainer.start_task(0).is_err());
    }

    #[test]
    fn test_task_detector_reacts_to_input_drift() {
        let mut detector = TaskDetector::new(0.5);
        let a = Tensor::from_vec(vec![1.0, 0.0], &[1, 2]).expect("a");
        let b = Tensor::from_vec(vec![0.0, 1.0], &[1, 2]).expect("b");

        assert_eq!(
            detector.detect_task_change(std::slice::from_ref(&a), &[]).expect("d"),
            None
        );
        assert_eq!(
            detector.detect_task_change(std::slice::from_ref(&a), &[]).expect("d"),
            None
        );
        assert_eq!(
            detector.detect_task_change(&[b], &[]).expect("d"),
            Some(1),
            "an orthogonal input distribution must be reported as a new task"
        );
        assert_eq!(detector.boundaries(), 1);
    }

    #[test]
    fn test_metrics_report_absent_accuracy_instead_of_zero() {
        let trainer = trainer_with(
            ContinualStrategy::L2Regularization { lambda: 0.0 },
            &[1.0, 1.0],
        );
        let metrics = trainer.get_metrics().expect("metrics");
        assert_eq!(metrics.average_accuracy, None);
        assert_eq!(metrics.num_tasks_learned, 0);
    }

    #[test]
    fn test_learn_batch_produces_a_real_regularization_term() {
        let mut trainer = trainer_with(
            ContinualStrategy::L2Regularization { lambda: 0.5 },
            &[1.0, -1.0],
        );
        let inputs = [Tensor::from_vec(vec![1.0, 1.0], &[1, 2]).expect("input")];
        let targets = [one_hot(0, 2)];

        let output = trainer.learn_batch(&inputs, &targets, Some(0)).expect("learn");
        let regularization = output.regularization_loss.to_vec_f32().expect("d")[0];
        assert!(
            (regularization - 1.0).abs() < 1e-5,
            "L2 penalty 0.5 * (1 + 1) must appear in the output, got {regularization}"
        );
    }
}
