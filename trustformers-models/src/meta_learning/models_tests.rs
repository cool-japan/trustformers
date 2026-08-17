#[cfg(test)]
mod tests {
    use crate::meta_learning::{
        Example, ExampleSet, MAMLModel, MetaAlgorithm, MetaLearner, MetaLearningConfig,
        MetaLearningModel, MetaSGDModel, MlpLearner, PrototypicalModel, RelationNetModel,
        ReptileModel, Task, TaskBatch, TaskType,
    };
    use trustformers_core::tensor::Tensor;

    fn small_config(algorithm: MetaAlgorithm) -> MetaLearningConfig {
        MetaLearningConfig {
            algorithm,
            inner_lr: 0.05,
            meta_lr: 0.01,
            inner_steps: 3,
            support_size: 8,
            query_size: 8,
            num_ways: 2,
            embedding_dim: 4,
            meta_batch_size: 2,
            ..Default::default()
        }
    }

    /// A one-dimensional sine-wave regression task: `y = amplitude · sin(x + phase)`.
    ///
    /// The learner sees the scalar `x` broadcast over its input width, so
    /// adapting on the support set has to reduce the query loss for a model
    /// that genuinely learns.
    fn sine_examples(amplitude: f32, phase: f32, xs: &[f32], width: usize) -> ExampleSet {
        let examples = xs
            .iter()
            .map(|x| {
                let input = Tensor::from_vec(vec![*x; width], &[width]).expect("input");
                let target =
                    Tensor::from_vec(vec![amplitude * (x + phase).sin()], &[1]).expect("target");
                Example::regression(input, target)
            })
            .collect();
        ExampleSet {
            examples,
            num_classes: 1,
        }
    }

    fn separable_examples(width: usize, count: usize) -> ExampleSet {
        let examples = (0..count)
            .map(|index| {
                let label = index % 2;
                let base = if label == 0 { 1.0f32 } else { -1.0f32 };
                let values: Vec<f32> = (0..width)
                    .map(|feature| base + 0.01 * (feature as f32) * (index as f32 % 3.0))
                    .collect();
                Example::classification(Tensor::from_vec(values, &[width]).expect("input"), label)
            })
            .collect();
        ExampleSet {
            examples,
            num_classes: 2,
        }
    }

    // --- the MLP learner itself ---

    #[test]
    fn test_learner_gradients_match_finite_differences() {
        let mut learner = MlpLearner::new(4, 5, 2).expect("learner");
        let examples = separable_examples(4, 6);

        let gradients = learner.gradients(&examples).expect("gradients");
        let parameters = learner.parameters().expect("parameters");

        let epsilon = 1e-3f32;
        for (name, tensor) in &parameters.parameters {
            let baseline = tensor.to_vec_f32().expect("values");
            let shape = tensor.shape();
            let analytic =
                gradients.gradients.get(name).expect("gradient").to_vec_f32().expect("g");

            // Check the first few coordinates; a full sweep is unnecessary and
            // slow, but every layer is covered.
            for index in 0..baseline.len().min(4) {
                let mut plus = baseline.clone();
                plus[index] += epsilon;
                let mut updated = parameters.clone();
                updated
                    .parameters
                    .insert(name.clone(), Tensor::from_vec(plus, &shape).expect("t"));
                learner.set_parameters(&updated).expect("set");
                let loss_plus = learner.loss(&examples).expect("loss");

                let mut minus = baseline.clone();
                minus[index] -= epsilon;
                let mut updated = parameters.clone();
                updated
                    .parameters
                    .insert(name.clone(), Tensor::from_vec(minus, &shape).expect("t"));
                learner.set_parameters(&updated).expect("set");
                let loss_minus = learner.loss(&examples).expect("loss");

                learner.set_parameters(&parameters).expect("restore");

                let numeric = ((loss_plus - loss_minus) / (2.0 * epsilon as f64)) as f32;
                assert!(
                    (analytic[index] - numeric).abs() < 5e-2,
                    "{name}[{index}]: analytic {} vs numeric {numeric}",
                    analytic[index]
                );
            }
        }
    }

    #[test]
    fn test_learner_gradient_descent_reduces_the_loss() {
        let mut learner = MlpLearner::new(4, 8, 2).expect("learner");
        let examples = separable_examples(4, 8);

        let before = learner.loss(&examples).expect("loss");
        for _ in 0..50 {
            let gradients = learner.gradients(&examples).expect("gradients");
            learner.apply_gradients(&gradients, 0.2).expect("apply");
        }
        let after = learner.loss(&examples).expect("loss");

        assert!(
            after < before,
            "gradient descent must reduce the loss ({before} -> {after})"
        );
    }

    #[test]
    fn test_learner_rejects_dimension_mismatch() {
        assert!(MlpLearner::new(0, 4, 2).is_err());
        let learner = MlpLearner::new(4, 4, 2).expect("learner");
        let wrong = ExampleSet {
            examples: vec![Example::classification(
                Tensor::zeros(&[3]).expect("input"),
                0,
            )],
            num_classes: 2,
        };
        assert!(learner.loss(&wrong).is_err());
    }

    // --- the regression the audit reported ---

    #[test]
    fn test_maml_forward_is_not_a_constant() {
        // Regression: every model returned `Ok(0.5)` for the loss and
        // `Ok(0.8)` for the accuracy regardless of the data.
        let config = small_config(MetaAlgorithm::MAML);
        let mut model = MAMLModel::new(&config).expect("model");

        let easy = separable_examples(4, 6);
        let hard = ExampleSet {
            // The same inputs but with the labels swapped: the loss must differ.
            examples: easy
                .examples
                .iter()
                .map(|e| Example::classification(e.input.clone(), 1 - e.label))
                .collect(),
            num_classes: 2,
        };

        let easy_loss = model.forward(&easy).expect("loss");
        let hard_loss = model.forward(&hard).expect("loss");
        assert!(
            (easy_loss - hard_loss).abs() > 1e-6,
            "the loss must depend on the data ({easy_loss} vs {hard_loss})"
        );
        assert!(
            (easy_loss - 0.5).abs() > 1e-9,
            "the placeholder 0.5 must be gone"
        );

        let accuracy = model.compute_accuracy(&easy).expect("accuracy");
        assert!((0.0..=1.0).contains(&accuracy));
    }

    #[test]
    fn test_maml_gradients_are_non_empty_and_data_dependent() {
        let config = small_config(MetaAlgorithm::MAML);
        let mut model = MAMLModel::new(&config).expect("model");
        let examples = separable_examples(4, 6);

        // Regression: compute_gradients used to return an empty ModelGradients.
        assert!(
            model.compute_gradients(0.0).is_err(),
            "forward must run first"
        );

        let loss = model.forward(&examples).expect("loss");
        let gradients = model.compute_gradients(loss).expect("gradients");
        assert!(!gradients.gradients.is_empty());

        let magnitude: f32 = gradients
            .gradients
            .values()
            .flat_map(|t| t.to_vec_f32().expect("g"))
            .map(|g| g.abs())
            .sum();
        assert!(magnitude > 0.0, "gradients must be non-zero");
    }

    #[test]
    fn test_apply_gradients_moves_the_parameters() {
        let config = small_config(MetaAlgorithm::MAML);
        let mut model = MAMLModel::new(&config).expect("model");
        let examples = separable_examples(4, 6);

        let before = model.get_parameters().expect("params");
        let loss = model.forward(&examples).expect("loss");
        let gradients = model.compute_gradients(loss).expect("gradients");
        model.apply_gradients(&gradients, 0.1).expect("apply");
        let after = model.get_parameters().expect("params");

        let changed = before.parameters.iter().any(|(name, tensor)| {
            let a = tensor.to_vec_f32().expect("a");
            let b = after.parameters[name].to_vec_f32().expect("b");
            a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-9)
        });
        assert!(changed, "applying gradients must change the parameters");
    }

    #[test]
    fn test_set_parameters_restores_the_model_exactly() {
        let config = small_config(MetaAlgorithm::MAML);
        let mut model = MAMLModel::new(&config).expect("model");
        let examples = separable_examples(4, 6);

        let snapshot = model.get_parameters().expect("params");
        let loss = model.forward(&examples).expect("loss");
        let gradients = model.compute_gradients(loss).expect("gradients");
        model.apply_gradients(&gradients, 0.5).expect("apply");
        model.set_parameters(snapshot.clone()).expect("restore");

        let restored = model.get_parameters().expect("params");
        for (name, tensor) in &snapshot.parameters {
            assert_eq!(
                tensor.to_vec_f32().expect("a"),
                restored.parameters[name].to_vec_f32().expect("b"),
                "{name} must be restored exactly"
            );
        }
    }

    #[test]
    fn test_inner_loop_adaptation_reduces_the_sine_regression_loss() {
        // The headline claim of MAML: a few gradient steps on the support set
        // must measurably improve the loss on the held-out query set of the
        // same sine wave.
        let config = MetaLearningConfig {
            algorithm: MetaAlgorithm::MAML,
            inner_lr: 0.02,
            inner_steps: 40,
            embedding_dim: 4,
            num_ways: 1,
            ..Default::default()
        };
        let mut model = MAMLModel::new(&config).expect("model");

        let support = sine_examples(2.0, 0.3, &[-1.5, -0.75, 0.0, 0.75, 1.5], 4);
        let query = sine_examples(2.0, 0.3, &[-1.2, -0.3, 0.4, 1.1], 4);

        let before = model.forward(&query).expect("loss");
        for _ in 0..config.inner_steps {
            let support_loss = model.forward(&support).expect("loss");
            let gradients = model.compute_gradients(support_loss).expect("gradients");
            model.apply_gradients(&gradients, config.inner_lr).expect("apply");
        }
        let after = model.forward(&query).expect("loss");

        assert!(
            after < before,
            "adaptation on the support set must reduce the query loss ({before} -> {after})"
        );
    }

    #[test]
    fn test_second_order_meta_gradient_differs_from_first_order() {
        let config = small_config(MetaAlgorithm::MAML);
        let mut model = MAMLModel::new(&config).expect("model");
        let support = separable_examples(4, 6);
        let query = separable_examples(4, 4);

        // Populate both the support and the query cache, as MAML does.
        let support_loss = model.forward(&support).expect("loss");
        let _ = support_loss;
        let query_loss = model.forward(&query).expect("loss");

        let first_order = model.compute_first_order_gradients(query_loss).expect("fomaml");
        let initial = model.get_parameters().expect("params");
        let second_order =
            model.compute_second_order_gradients(&initial, query_loss).expect("maml");

        assert_eq!(first_order.gradients.len(), second_order.gradients.len());
        let differs = first_order.gradients.iter().any(|(name, tensor)| {
            let a = tensor.to_vec_f32().expect("a");
            let b = second_order.gradients[name].to_vec_f32().expect("b");
            a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6)
        });
        assert!(
            differs,
            "the Hessian-vector correction must change the meta-gradient"
        );
    }

    #[test]
    fn test_second_order_requires_two_forward_passes() {
        let config = small_config(MetaAlgorithm::MAML);
        let mut model = MAMLModel::new(&config).expect("model");
        let query = separable_examples(4, 4);
        let loss = model.forward(&query).expect("loss");
        let initial = model.get_parameters().expect("params");
        assert!(model.compute_second_order_gradients(&initial, loss).is_err());
    }

    #[test]
    fn test_meta_sgd_learning_rates_are_per_parameter_and_used() {
        let config = small_config(MetaAlgorithm::MetaSGD);
        let mut model = MetaSGDModel::new(&config).expect("model");
        let support = separable_examples(4, 6);
        let query = separable_examples(4, 4);

        let rates = model.get_learning_rates().expect("rates");
        assert!(!rates.is_empty());
        assert!(rates.iter().all(|r| (*r - config.inner_lr).abs() < 1e-12));

        let support_loss = model.forward(&support).expect("loss");
        let gradients = model.compute_gradients(support_loss).expect("gradients");
        let before = model.get_parameters().expect("params");
        model.apply_gradients_with_lr(&gradients, &rates).expect("apply");
        let after = model.get_parameters().expect("params");
        let changed = before.parameters.iter().any(|(name, tensor)| {
            let a = tensor.to_vec_f32().expect("a");
            let b = after.parameters[name].to_vec_f32().expect("b");
            a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-9)
        });
        assert!(changed);

        let query_loss = model.forward(&query).expect("loss");
        let lr_gradients = model.compute_lr_gradients(query_loss).expect("lr grads");
        assert_eq!(lr_gradients.len(), rates.len());
        assert!(lr_gradients.iter().any(|g| g.abs() > 0.0));
    }

    #[test]
    fn test_relation_scores_are_symmetric_and_bounded() {
        let config = small_config(MetaAlgorithm::RelationNet);
        let model = RelationNetModel::new(&config).expect("model");
        let a = Tensor::from_vec(vec![0.1, -0.2, 0.3], &[3]).expect("a");
        let b = Tensor::from_vec(vec![0.1, -0.2, 0.3], &[3]).expect("b");
        let c = Tensor::from_vec(vec![-0.9, 0.8, -0.7], &[3]).expect("c");

        let same = model.compute_relation(&a, &b).expect("same");
        let different = model.compute_relation(&a, &c).expect("different");
        assert!((same - 1.0).abs() < 1e-9, "identical embeddings score 1");
        assert!(different < same);
        assert!((0.0..=1.0).contains(&different));
        assert!(model.compute_relation(&a, &Tensor::zeros(&[2]).expect("z")).is_err());
    }

    #[test]
    fn test_embeddings_are_learned_not_the_raw_input() {
        let config = small_config(MetaAlgorithm::ProtoNet);
        let model = PrototypicalModel::new(&config).expect("model");
        let example = Example::classification(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).expect("i"),
            0,
        );
        let embedding = model.embed(&example).expect("embed").to_vec_f32().expect("e");
        assert_eq!(
            embedding.len(),
            4,
            "hidden width is clamp(embedding_dim, 4, 64)"
        );
        assert!(
            embedding.iter().all(|v| v.abs() <= 1.0),
            "tanh embeddings are bounded, unlike the raw input"
        );
        assert_ne!(embedding, vec![1.0, 2.0, 3.0, 4.0]);
    }

    // --- MetaLearner orchestration ---

    fn synthetic_task(id: &str) -> Task {
        Task {
            task_id: id.to_string(),
            support_set: separable_examples(4, 6),
            query_set: separable_examples(4, 4),
            task_type: TaskType::Classification,
        }
    }

    #[test]
    fn test_maml_episode_reports_measured_metrics() {
        let config = small_config(MetaAlgorithm::MAML);
        let mut learner = MetaLearner::new(config).expect("learner");
        let batch = TaskBatch {
            tasks: vec![synthetic_task("a"), synthetic_task("b")],
            batch_id: "batch".to_string(),
        };

        let result = learner.train_episode(batch).expect("episode");
        assert_eq!(result.num_tasks, 2);
        assert!(
            (result.meta_loss - 0.5).abs() > 1e-9,
            "the constant 0.5 placeholder loss must be gone"
        );
        assert!(
            (result.meta_accuracy - 0.8).abs() > 1e-9 || result.meta_accuracy == 1.0,
            "accuracy must be measured, not the constant 0.8"
        );
        assert!((0.0..=1.0).contains(&result.meta_accuracy));
    }

    #[test]
    fn test_reptile_episode_runs_end_to_end() {
        let config = small_config(MetaAlgorithm::Reptile);
        let mut learner = MetaLearner::new(config).expect("learner");
        let batch = TaskBatch {
            tasks: vec![synthetic_task("a")],
            batch_id: "batch".to_string(),
        };
        let result = learner.train_episode(batch).expect("episode");
        assert!(result.meta_loss.is_finite());
    }

    #[test]
    fn test_prototypical_episode_uses_real_prototypes() {
        let config = small_config(MetaAlgorithm::ProtoNet);
        let mut learner = MetaLearner::new(config).expect("learner");
        let batch = TaskBatch {
            tasks: vec![synthetic_task("a")],
            batch_id: "batch".to_string(),
        };
        let result = learner.train_episode(batch).expect("episode");
        assert!(result.meta_loss.is_finite());
        assert!((0.0..=1.0).contains(&result.meta_accuracy));
    }

    #[test]
    fn test_matching_network_attention_is_a_distribution() {
        let config = small_config(MetaAlgorithm::MatchingNet);
        let mut learner = MetaLearner::new(config).expect("learner");
        let batch = TaskBatch {
            tasks: vec![synthetic_task("a")],
            batch_id: "batch".to_string(),
        };
        // Regression: attention used to be `vec![vec![1.0]]` and the accuracy a
        // constant 0.8.
        let result = learner.train_episode(batch).expect("episode");
        assert!(result.meta_loss.is_finite());
        assert!(
            (result.meta_accuracy - 0.8).abs() > 1e-9 || result.meta_accuracy == 1.0,
            "matching accuracy must be measured"
        );
    }

    #[test]
    fn test_unimplemented_algorithms_report_instead_of_faking() {
        for algorithm in [MetaAlgorithm::MANN, MetaAlgorithm::GBML, MetaAlgorithm::L2L] {
            let config = small_config(algorithm);
            let mut learner = MetaLearner::new(config).expect("learner");
            let batch = TaskBatch {
                tasks: vec![synthetic_task("a")],
                batch_id: "batch".to_string(),
            };
            assert!(
                learner.train_episode(batch).is_err(),
                "{algorithm:?} must report its missing pieces rather than invent metrics"
            );
        }
    }

    #[test]
    fn test_task_sampler_produces_learnable_tasks() {
        // Regression: the sampler used to attach labels independent of the
        // randomly generated inputs, so no model could ever learn the task.
        let config = small_config(MetaAlgorithm::MAML);
        let mut learner = MetaLearner::new(config).expect("learner");
        let batch = learner.sample_task_batch().expect("batch");
        assert_eq!(batch.tasks.len(), 2);

        let task = &batch.tasks[0];
        // Examples of the same class must be closer to each other than to the
        // other class, which is exactly what makes the task learnable.
        let class_zero: Vec<Vec<f32>> = task
            .support_set
            .examples
            .iter()
            .filter(|e| e.label == 0)
            .map(|e| e.input.to_vec_f32().expect("v"))
            .collect();
        let class_one: Vec<Vec<f32>> = task
            .support_set
            .examples
            .iter()
            .filter(|e| e.label == 1)
            .map(|e| e.input.to_vec_f32().expect("v"))
            .collect();
        assert!(!class_zero.is_empty() && !class_one.is_empty());

        let within = squared_distance(&class_zero[0], &class_zero[1]);
        let between = squared_distance(&class_zero[0], &class_one[0]);
        assert!(
            within < between,
            "within-class distance {within} must be below the between-class distance {between}"
        );
    }

    fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    #[test]
    fn test_reptile_parameter_difference_is_the_meta_gradient() {
        let config = small_config(MetaAlgorithm::Reptile);
        let mut model = ReptileModel::new(&config).expect("model");
        let support = separable_examples(4, 6);

        let initial = model.get_parameters().expect("params");
        for _ in 0..3 {
            let loss = model.forward(&support).expect("loss");
            let gradients = model.compute_gradients(loss).expect("gradients");
            model.apply_gradients(&gradients, 0.1).expect("apply");
        }
        let adapted = model.get_parameters().expect("params");

        let moved = initial.parameters.iter().any(|(name, tensor)| {
            let a = tensor.to_vec_f32().expect("a");
            let b = adapted.parameters[name].to_vec_f32().expect("b");
            a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-9)
        });
        assert!(moved, "the Reptile inner loop must move the parameters");
    }
}
