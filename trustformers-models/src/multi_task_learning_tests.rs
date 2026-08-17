#[cfg(test)]
mod tests {
    use crate::multi_task_learning::*;
    use std::collections::HashMap;
    use trustformers_core::tensor::Tensor;

    #[test]
    fn test_mtl_config_default() {
        let config = MTLConfig::default();
        assert_eq!(config.tasks.len(), 0);
        assert!(!config.use_task_embeddings);
        assert!(!config.use_auxiliary_tasks);

        if let MTLArchitecture::HardParameterSharing {
            shared_layers,
            task_specific_layers,
        } = config.architecture
        {
            assert_eq!(shared_layers, 8);
            assert_eq!(task_specific_layers, 2);
        } else {
            panic!("Expected HardParameterSharing architecture");
        }
    }

    #[test]
    fn test_task_config() {
        let task = TaskConfig::new(
            "test",
            TaskType::Classification {
                num_classes: 10,
                use_class_weights: false,
            },
        );

        assert_eq!(task.name, "test");
        assert_eq!(task.weight, 1.0);
        assert!(!task.is_main_task);

        let weighted_task = task.with_weight(2.0);
        assert_eq!(weighted_task.weight, 2.0);
    }

    #[test]
    fn test_classification_task_util() {
        let task = utils::classification_task("sentiment", 3);
        assert_eq!(task.name, "sentiment");

        if let TaskType::Classification { num_classes, .. } = task.task_type {
            assert_eq!(num_classes, 3);
        } else {
            panic!("Expected Classification task type");
        }
    }

    #[test]
    fn test_regression_task_util() {
        let task = utils::regression_task("score", 1);
        assert_eq!(task.name, "score");

        if let TaskType::Regression { output_dim, .. } = task.task_type {
            assert_eq!(output_dim, 1);
        } else {
            panic!("Expected Regression task type");
        }
    }

    #[test]
    fn test_hard_parameter_sharing_config() {
        let tasks = vec![
            utils::classification_task("task1", 5),
            utils::regression_task("task2", 1),
        ];

        let config = utils::hard_parameter_sharing_config(tasks, 6, 2);
        assert_eq!(config.tasks.len(), 2);

        if let MTLArchitecture::HardParameterSharing {
            shared_layers,
            task_specific_layers,
        } = config.architecture
        {
            assert_eq!(shared_layers, 6);
            assert_eq!(task_specific_layers, 2);
        } else {
            panic!("Expected HardParameterSharing architecture");
        }
    }

    #[test]
    fn test_soft_parameter_sharing_config() {
        let tasks = vec![utils::classification_task("task1", 5)];
        let config = utils::soft_parameter_sharing_config(tasks, 0.01);

        if let MTLArchitecture::SoftParameterSharing {
            regularization_weight,
            ..
        } = config.architecture
        {
            assert_eq!(regularization_weight, 0.01);
        } else {
            panic!("Expected SoftParameterSharing architecture");
        }
    }

    #[test]
    fn test_mmoe_config() {
        let tasks = vec![
            utils::classification_task("task1", 5),
            utils::classification_task("task2", 3),
        ];

        let config = utils::mmoe_config(tasks, 4, 128);

        if let MTLArchitecture::MultiGateMixtureOfExperts {
            num_experts,
            expert_dim,
            num_gates,
        } = config.architecture
        {
            assert_eq!(num_experts, 4);
            assert_eq!(expert_dim, 128);
            assert_eq!(num_gates, 2);
        } else {
            panic!("Expected MultiGateMixtureOfExperts architecture");
        }
    }

    #[test]
    fn test_mlm_auxiliary_task() {
        let aux_task = utils::mlm_auxiliary_task(0.1);
        assert_eq!(aux_task.name, "mlm");
        assert_eq!(aux_task.weight, 0.1);

        if let AuxiliaryType::MaskedLanguageModeling = aux_task.auxiliary_type {
            // Expected
        } else {
            panic!("Expected MaskedLanguageModeling auxiliary type");
        }
    }

    #[test]
    fn test_compute_correlation() {
        let seq1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let seq2 = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect positive correlation

        let correlation = utils::compute_correlation(&seq1, &seq2);
        assert!((correlation - 1.0).abs() < 1e-6);

        let seq3 = vec![5.0, 4.0, 3.0, 2.0, 1.0]; // Perfect negative correlation
        let correlation_neg = utils::compute_correlation(&seq1, &seq3);
        assert!((correlation_neg + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mtl_analysis() {
        let mut single_task = HashMap::new();
        single_task.insert("task1".to_string(), 0.8);
        single_task.insert("task2".to_string(), 0.7);
        single_task.insert("task3".to_string(), 0.6);

        let mut multi_task = HashMap::new();
        multi_task.insert("task1".to_string(), 0.85); // Positive transfer
        multi_task.insert("task2".to_string(), 0.65); // Negative transfer
        multi_task.insert("task3".to_string(), 0.65); // Positive transfer

        let analysis = utils::analyze_mtl_effectiveness(&single_task, &multi_task);
        assert_eq!(analysis.num_tasks, 3);
        assert_eq!(analysis.positive_transfer_tasks.len(), 2);
        assert_eq!(analysis.negative_transfer_tasks.len(), 1);
        assert!(analysis.positive_transfer_tasks.contains(&"task1".to_string()));
        assert!(analysis.negative_transfer_tasks.contains(&"task2".to_string()));
    }

    // ------------------------------------------------------------------
    // Regression tests for the previously fabricated MTL machinery:
    // softmax-instead-of-log-softmax cross-entropy, Huber == MSE, DWA with a
    // hardcoded previous loss, pass-through GradNorm / UncertaintyWeighting.
    // ------------------------------------------------------------------

    use crate::continual_learning::NamedParameters;
    use serde::{Deserialize, Serialize};
    use trustformers_core::traits::{Config, Model};
    use trustformers_core::Result;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct TinyConfig;

    impl Config for TinyConfig {
        fn architecture(&self) -> &'static str {
            "tiny-mtl"
        }
    }

    /// Identity trunk with a single learnable scale, so the shared-parameter
    /// gradient GradNorm needs is well defined.
    struct TinyTrunk {
        scale: Tensor,
        config: TinyConfig,
    }

    impl TinyTrunk {
        fn new(width: usize) -> Self {
            Self {
                scale: Tensor::ones(&[1, width]).expect("scale"),
                config: TinyConfig,
            }
        }
    }

    impl Model for TinyTrunk {
        type Config = TinyConfig;
        type Input = Tensor;
        type Output = Tensor;

        fn forward(&self, input: Tensor) -> Result<Tensor> {
            input.mul(&self.scale)
        }

        fn load_pretrained(&mut self, _reader: &mut dyn std::io::Read) -> Result<()> {
            Ok(())
        }

        fn get_config(&self) -> &TinyConfig {
            &self.config
        }

        fn num_parameters(&self) -> usize {
            self.scale.to_vec_f32().map(|v| v.len()).unwrap_or(0)
        }
    }

    impl NamedParameters for TinyTrunk {
        fn named_parameters(&self) -> Vec<(String, Tensor)> {
            vec![("scale".to_string(), self.scale.clone())]
        }

        fn set_named_parameter(&mut self, name: &str, value: Tensor) -> Result<()> {
            if name != "scale" {
                return Err(trustformers_core::errors::invalid_input(format!(
                    "unknown parameter {}",
                    name
                )));
            }
            self.scale = value;
            Ok(())
        }
    }

    fn trainer(
        balancing: LossBalancingStrategy,
        tasks: Vec<TaskConfig>,
    ) -> MultiTaskLearningTrainer<TinyTrunk> {
        let config = MTLConfig {
            loss_balancing: balancing,
            tasks,
            ..Default::default()
        };
        MultiTaskLearningTrainer::new(TinyTrunk::new(768), config).expect("trainer")
    }

    fn classification_task(name: &str) -> TaskConfig {
        TaskConfig::new(
            name,
            TaskType::Classification {
                num_classes: 3,
                use_class_weights: false,
            },
        )
    }

    fn regression_task(name: &str, loss_type: RegressionLossType) -> TaskConfig {
        TaskConfig::new(
            name,
            TaskType::Regression {
                output_dim: 1,
                loss_type,
            },
        )
    }

    #[test]
    fn test_classification_loss_is_real_cross_entropy() {
        // Regression: the loss used softmax where log-softmax is required, so
        // it was bounded in [0, 1] and a confidently-wrong prediction could
        // never be penalised.
        let mtl = trainer(
            LossBalancingStrategy::EqualWeighting,
            vec![classification_task("cls")],
        );

        let logits = Tensor::from_vec(vec![10.0, -10.0, -10.0], &[1, 3]).expect("logits");
        let wrong = Tensor::from_vec(vec![0.0, 1.0, 0.0], &[1, 3]).expect("target");
        let right = Tensor::from_vec(vec![1.0, 0.0, 0.0], &[1, 3]).expect("target");

        let wrong_loss = mtl
            .compute_task_loss("cls", &logits, &wrong)
            .expect("loss")
            .to_vec_f32()
            .expect("d")[0];
        let right_loss = mtl
            .compute_task_loss("cls", &logits, &right)
            .expect("loss")
            .to_vec_f32()
            .expect("d")[0];

        assert!(
            wrong_loss > 15.0,
            "a confidently wrong prediction must cost far more than 1.0, got {wrong_loss}"
        );
        assert!(
            right_loss < 1e-4,
            "a confidently correct prediction must cost ~0, got {right_loss}"
        );
    }

    #[test]
    fn test_huber_loss_is_not_plain_mse() {
        // Regression: the Huber branch computed the linear region and threw it
        // away, so `delta` had no effect at all.
        let huber = trainer(
            LossBalancingStrategy::EqualWeighting,
            vec![regression_task(
                "reg",
                RegressionLossType::Huber { delta: 0.1 },
            )],
        );
        let mse = trainer(
            LossBalancingStrategy::EqualWeighting,
            vec![regression_task("reg", RegressionLossType::MSE)],
        );

        let outputs = Tensor::from_vec(vec![0.0], &[1, 1]).expect("outputs");
        let targets = Tensor::from_vec(vec![5.0], &[1, 1]).expect("targets");

        let huber_loss = huber
            .compute_task_loss("reg", &outputs, &targets)
            .expect("loss")
            .to_vec_f32()
            .expect("d")[0];
        let mse_loss = mse
            .compute_task_loss("reg", &outputs, &targets)
            .expect("loss")
            .to_vec_f32()
            .expect("d")[0];

        // delta * |d| - 0.5 * delta^2 = 0.1 * 5 - 0.005 = 0.495
        assert!(
            (huber_loss - 0.495).abs() < 1e-4,
            "Huber must use its linear branch for |d| > delta, got {huber_loss}"
        );
        assert!((mse_loss - 25.0).abs() < 1e-3, "MSE reference: {mse_loss}");
        assert!(huber_loss < mse_loss);
    }

    #[test]
    fn test_huber_matches_its_definition_on_the_quadratic_branch() {
        let mtl = trainer(
            LossBalancingStrategy::EqualWeighting,
            vec![regression_task(
                "reg",
                RegressionLossType::Huber { delta: 10.0 },
            )],
        );
        let outputs = Tensor::from_vec(vec![0.0], &[1, 1]).expect("outputs");
        let targets = Tensor::from_vec(vec![1.0], &[1, 1]).expect("targets");
        let loss = mtl
            .compute_task_loss("reg", &outputs, &targets)
            .expect("loss")
            .to_vec_f32()
            .expect("d")[0];
        // |d| = 1 <= delta = 10, so the quadratic branch applies: 0.5 * 1^2.
        assert!((loss - 0.5).abs() < 1e-4, "quadratic branch: {loss}");
    }

    #[test]
    fn test_dynamic_weight_average_downweights_fast_improving_tasks() {
        // Regression: `get_previous_task_loss` returned a hardcoded 1.0, which
        // inverted DWA into a weight that grows with the loss.
        let mut mtl = trainer(
            LossBalancingStrategy::DynamicWeightAverage,
            vec![classification_task("fast"), classification_task("slow")],
        );

        // Task "fast" halves its loss each step; "slow" is flat.
        for (step, (fast, slow)) in [(1.0f32, 1.0f32), (0.5, 1.0)].into_iter().enumerate() {
            let mut losses = HashMap::new();
            losses.insert(
                "fast".to_string(),
                Tensor::from_vec(vec![fast], &[1]).expect("loss"),
            );
            losses.insert(
                "slow".to_string(),
                Tensor::from_vec(vec![slow], &[1]).expect("loss"),
            );
            mtl.record_task_losses(&losses).expect("record");
            let _ = step;
        }

        let history = mtl.task_loss_history.get("fast").expect("history");
        assert_eq!(history.len(), 2);

        let mut losses = HashMap::new();
        losses.insert(
            "fast".to_string(),
            Tensor::from_vec(vec![1.0], &[1]).expect("loss"),
        );
        losses.insert(
            "slow".to_string(),
            Tensor::from_vec(vec![1.0], &[1]).expect("loss"),
        );
        let balanced = mtl.balance_losses(&losses).expect("balance");

        let fast = balanced["fast"].to_vec_f32().expect("d")[0];
        let slow = balanced["slow"].to_vec_f32().expect("d")[0];
        assert!(
            fast < slow,
            "a task whose loss halved must be down-weighted relative to a flat one \
             ({fast} vs {slow})"
        );
        // DWA weights sum to the number of tasks.
        assert!((fast + slow - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_uncertainty_weighting_is_not_a_pass_through() {
        let mut mtl = trainer(
            LossBalancingStrategy::UncertaintyWeighting,
            vec![classification_task("a"), classification_task("b")],
        );

        let mut losses = HashMap::new();
        losses.insert(
            "a".to_string(),
            Tensor::from_vec(vec![4.0], &[1]).expect("l"),
        );
        losses.insert(
            "b".to_string(),
            Tensor::from_vec(vec![0.1], &[1]).expect("l"),
        );

        // Drive the learned log-variances apart.
        for _ in 0..200 {
            mtl.record_task_losses(&losses).expect("record");
        }

        let variance_a = mtl.task_log_variances["a"];
        let variance_b = mtl.task_log_variances["b"];
        assert!(
            variance_a > variance_b,
            "the noisier task must learn a larger log-variance ({variance_a} vs {variance_b})"
        );

        let balanced = mtl.balance_losses(&losses).expect("balance");
        let scaled_a = balanced["a"].to_vec_f32().expect("d")[0];
        assert!(
            (scaled_a - 4.0).abs() > 1e-6,
            "uncertainty weighting must actually rescale the loss"
        );
    }

    #[test]
    fn test_gradnorm_uses_measured_gradient_norms() {
        let mut mtl = trainer(
            LossBalancingStrategy::GradNorm { alpha: 1.0 },
            vec![regression_task("r1", RegressionLossType::MSE)],
        );

        let mut data = HashMap::new();
        data.insert(
            "r1".to_string(),
            TaskBatch {
                inputs: Tensor::from_vec(vec![0.5; 768], &[1, 768]).expect("inputs"),
                targets: Tensor::from_vec(vec![1.0], &[1, 1]).expect("targets"),
                task_name: "r1".to_string(),
            },
        );

        assert!(mtl.gradient_stats.is_empty());
        mtl.update_gradient_stats(&data, &["r1".to_string()]).expect("stats");
        let stats = mtl.gradient_stats.get("r1").expect("stats");
        assert!(
            stats.gradient_norm > 0.0,
            "GradNorm must measure a non-zero gradient norm, got {}",
            stats.gradient_norm
        );
        assert_eq!(stats.update_count, 1);
    }

    #[test]
    fn test_ranking_tasks_report_the_missing_objective() {
        let mtl = trainer(
            LossBalancingStrategy::EqualWeighting,
            vec![TaskConfig::new(
                "rank",
                TaskType::Ranking {
                    ranking_type: RankingType::Pairwise,
                },
            )],
        );
        let outputs = Tensor::zeros(&[1, 4]).expect("outputs");
        let targets = Tensor::zeros(&[1, 4]).expect("targets");
        assert!(mtl.compute_task_loss("rank", &outputs, &targets).is_err());
    }

    #[test]
    fn test_classification_accuracy_is_batch_wide() {
        let mtl = trainer(
            LossBalancingStrategy::EqualWeighting,
            vec![classification_task("cls")],
        );
        // Two rows: one correct, one wrong -> 0.5, not 0.0 or 1.0.
        let outputs =
            Tensor::from_vec(vec![3.0, 0.0, 0.0, 0.0, 3.0, 0.0], &[2, 3]).expect("outputs");
        let targets =
            Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0], &[2, 3]).expect("targets");
        let accuracy = mtl.compute_task_accuracy("cls", &outputs, &targets).expect("acc");
        assert!(
            (accuracy - 0.5).abs() < 1e-6,
            "expected 0.5, got {accuracy}"
        );
    }
}
