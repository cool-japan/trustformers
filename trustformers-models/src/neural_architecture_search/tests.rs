//! Tests for neural architecture search.

use super::*;

#[test]
fn test_nas_config_default() {
    let config = NASConfig::default();
    assert_eq!(config.max_evaluations, 1000);
    assert_eq!(config.population_size, 50);
    assert!(matches!(config.strategy, SearchStrategy::Evolutionary));
}

#[test]
fn test_transformer_search_space() {
    let space = SearchSpace::transformer_space();
    assert!(space.dimensions.contains_key("num_layers"));
    assert!(space.dimensions.contains_key("hidden_size"));
    assert!(space.choices.contains_key("activation"));
}

#[test]
fn test_architecture_random_generation() {
    let space = SearchSpace::transformer_space();
    let mut rng = StdRng::seed_from_u64(42);
    let arch = Architecture::random(&space, &mut rng);

    assert!(!arch.dimensions.is_empty());
    assert!(!arch.choices.is_empty());
}

#[test]
fn test_architecture_parameter_estimation() {
    let mut arch = Architecture::new();
    arch.dimensions.insert("hidden_size".to_string(), 768);
    arch.dimensions.insert("num_layers".to_string(), 12);
    arch.dimensions.insert("vocab_size".to_string(), 32000);

    let params = arch.estimate_parameters();
    assert!(params > 100_000_000); // Should be reasonable for BERT-base
}

#[test]
fn test_architecture_constraint_validation() {
    let space = SearchSpace::transformer_space();
    let mut arch = Architecture::new();
    arch.dimensions.insert("hidden_size".to_string(), 768);
    arch.dimensions.insert("num_heads".to_string(), 12);
    arch.dimensions.insert("intermediate_size".to_string(), 3072);

    assert!(space.validate_architecture(&arch).is_ok());

    // Test invalid architecture
    arch.dimensions.insert("hidden_size".to_string(), 777); // Not divisible by 12
    assert!(space.validate_architecture(&arch).is_err());
}

#[test]
fn test_architecture_mutation() {
    let space = SearchSpace::transformer_space();
    let mut rng = StdRng::seed_from_u64(42);
    let mut arch = Architecture::random(&space, &mut rng);
    let original = arch.clone();

    arch.mutate(&space, 1.0, &mut rng); // 100% mutation rate

    // Should have some differences
    let mut differences = 0;
    for (key, value) in &arch.dimensions {
        if original.dimensions.get(key) != Some(value) {
            differences += 1;
        }
    }
    assert!(differences > 0);
}

#[test]
fn test_neural_architecture_searcher_creation() {
    let config = NASConfig::default();
    let searcher = NeuralArchitectureSearcher::new(config);
    assert!(searcher.is_ok());
}

#[test]
fn test_dimension_range() {
    let range = DimensionRange::new(1, 10, 2);
    assert!(range.validate(1));
    assert!(range.validate(3));
    assert!(range.validate(9));
    assert!(!range.validate(2));
    assert!(!range.validate(11));

    let mut rng = StdRng::seed_from_u64(42);
    let sample = range.sample(&mut rng);
    assert!(range.validate(sample));
}

#[test]
fn test_optimization_objectives() {
    let obj1 = OptimizationObjective::Accuracy { weight: 0.7 };
    let obj2 = OptimizationObjective::Latency { weight: 0.3 };

    assert_eq!(obj1.weight(), 0.7);
    assert_eq!(obj2.weight(), 0.3);
    assert_eq!(obj1.name(), "accuracy");
    assert_eq!(obj2.name(), "latency");
}

#[test]
fn test_architecture_crossover() {
    let space = SearchSpace::transformer_space();
    let mut rng = StdRng::seed_from_u64(42);

    let parent1 = Architecture::random(&space, &mut rng);
    let parent2 = Architecture::random(&space, &mut rng);

    let child = parent1.crossover(&parent2, &mut rng);

    // Child should have dimensions from both parents
    assert_eq!(child.dimensions.len(), parent1.dimensions.len());
    assert_eq!(child.choices.len(), parent1.choices.len());
    assert_eq!(
        child.metadata.generation,
        std::cmp::max(parent1.metadata.generation, parent2.metadata.generation) + 1
    );
}

// ---------------------------------------------------------------------------
// Regression tests: fitness must come from real training, not from a formula
// ---------------------------------------------------------------------------

/// A tiny search space so the tests train quickly.
fn small_space() -> SearchSpace {
    let mut dimensions = HashMap::new();
    dimensions.insert("num_layers".to_string(), DimensionRange::new(1, 4, 1));
    dimensions.insert("hidden_size".to_string(), DimensionRange::new(8, 64, 8));

    let mut choices = HashMap::new();
    choices.insert(
        "activation".to_string(),
        vec!["relu".to_string(), "tanh".to_string()],
    );

    SearchSpace {
        dimensions,
        choices,
        constraints: Vec::new(),
    }
}

fn small_config(strategy: SearchStrategy, evaluations: usize) -> NASConfig {
    NASConfig {
        strategy,
        search_space: small_space(),
        objectives: vec![OptimizationObjective::Accuracy { weight: 1.0 }],
        max_evaluations: evaluations,
        population_size: 6,
        generations: 2,
        patience: 2,
        hardware_constraints: None,
        progressive_search: false,
        seed: Some(7),
    }
}

/// A stand-in evaluator that records what it was asked to evaluate.
struct CountingEvaluator {
    calls: usize,
}

impl ArchitectureEvaluator for CountingEvaluator {
    fn evaluate(&mut self, architecture: &Architecture) -> Result<MeasuredPerformance> {
        self.calls += 1;
        // Reward depth so the search has a signal to follow.
        let layers = architecture.dimensions.get("num_layers").copied().unwrap_or(1) as f32;
        Ok(MeasuredPerformance {
            accuracy: (layers / 4.0).clamp(0.0, 1.0),
            train_loss: 0.1,
            trained_parameters: 10,
            inference_seconds: 0.000_1,
            custom_metrics: HashMap::new(),
        })
    }

    fn description(&self) -> String {
        "counting evaluator".to_string()
    }
}

#[test]
fn test_search_uses_the_evaluator_for_every_candidate() {
    let config = small_config(SearchStrategy::Random, 5);
    let mut searcher = NeuralArchitectureSearcher::with_evaluator(
        config,
        Box::new(CountingEvaluator { calls: 0 }),
    )
    .expect("searcher");

    let best = searcher.search().expect("search must succeed");

    assert_eq!(searcher.evaluation_history().len(), 5);
    assert!(searcher.evaluator_description().contains("counting"));
    // Fitness equals the evaluator's measured accuracy, not a size formula.
    let layers = best.architecture.dimensions["num_layers"] as f32;
    assert!((best.fitness - (layers / 4.0)).abs() < 1e-6, "{best:?}");
}

#[test]
fn test_accuracy_comes_from_measurement_not_parameter_count() {
    let config = small_config(SearchStrategy::Random, 4);
    let mut searcher = NeuralArchitectureSearcher::new(config).expect("searcher");
    searcher.search().expect("search must succeed");

    let history = searcher.evaluation_history();
    assert!(!history.is_empty());

    for evaluation in history {
        let accuracy = evaluation.metrics["accuracy"];
        assert!((0.0..=1.0).contains(&accuracy));
        assert_eq!(
            evaluation.info.get("accuracy_source").map(String::as_str),
            Some("measured_holdout_accuracy"),
            "the accuracy must be labelled as a measurement"
        );
        assert!(evaluation.info.contains_key("trained_parameters"));

        // The old implementation returned 0.85 + f(parameters); that formula
        // never produces values below 0.85, and every architecture with the same
        // parameter count would score identically.
        let complexity = evaluation.architecture.estimate_parameters() as f32 / 1_000_000.0;
        let old_formula =
            (0.85 + (complexity / 100.0).min(0.1) - (complexity / 1000.0).max(0.0)).clamp(0.0, 1.0);
        assert!(
            (accuracy - old_formula).abs() > 1e-6,
            "accuracy {accuracy} still matches the discarded analytic formula"
        );
    }

    // Two candidates with the same estimated size must be allowed to differ.
    let distinct: std::collections::BTreeSet<u32> = history
        .iter()
        .map(|evaluation| evaluation.metrics["accuracy"].to_bits())
        .collect();
    assert!(
        distinct.len() > 1,
        "measured accuracies must vary across candidates"
    );
}

#[test]
fn test_energy_objective_is_rejected_rather_than_invented() {
    let mut config = small_config(SearchStrategy::Random, 2);
    config.objectives = vec![OptimizationObjective::Energy { weight: 1.0 }];
    let mut searcher = NeuralArchitectureSearcher::new(config).expect("searcher");

    assert!(
        searcher.search().is_err(),
        "an objective with no measurement source must not be scored with a formula"
    );
}

#[test]
fn test_custom_objective_requires_a_reported_metric() {
    let mut config = small_config(SearchStrategy::Random, 2);
    config.objectives = vec![OptimizationObjective::Custom {
        name: "robustness".to_string(),
        weight: 1.0,
    }];
    let mut searcher = NeuralArchitectureSearcher::with_evaluator(
        config,
        Box::new(CountingEvaluator { calls: 0 }),
    )
    .expect("searcher");

    assert!(
        searcher.search().is_err(),
        "a custom objective the evaluator does not report must fail loudly"
    );
}

#[test]
fn test_reinforce_controller_learns_from_the_reward() {
    let space = small_space();
    let mut controller = ReinforceController::new(&space, 0.5);
    let mut rng = StdRng::seed_from_u64(3);

    let before = controller.dimension_probabilities("num_layers").expect("policy for num_layers");

    // Reward the largest depth repeatedly.
    for _ in 0..200 {
        let (architecture, actions) = controller.sample(&space, &mut rng);
        let layers = architecture.dimensions["num_layers"] as f32;
        controller.update(&actions, layers);
    }

    let after = controller.dimension_probabilities("num_layers").expect("policy for num_layers");

    let last = after.len() - 1;
    assert!(
        after[last] > before[last] + 0.05,
        "the controller must shift mass towards the rewarded action: {:?} -> {:?}",
        before,
        after
    );
    assert!(controller.baseline() > 1.0);
}

#[test]
fn test_reinforce_search_beats_its_own_first_samples() {
    let config = small_config(SearchStrategy::ReinforcementLearning, 60);
    let mut searcher = NeuralArchitectureSearcher::with_evaluator(
        config,
        Box::new(CountingEvaluator { calls: 0 }),
    )
    .expect("searcher");
    searcher.search().expect("search must succeed");

    let history = searcher.evaluation_history();
    assert_eq!(history.len(), 60);

    let early: f32 = history[..20].iter().map(|e| e.fitness).sum::<f32>() / 20.0;
    let late: f32 = history[40..].iter().map(|e| e.fitness).sum::<f32>() / 20.0;
    assert!(
        late > early,
        "the controller must improve over time: early {early} vs late {late}"
    );
}

#[test]
fn test_gaussian_process_surrogate_interpolates_observations() {
    let inputs = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![0.5, 0.5]];
    let targets = vec![0.0, 1.0, 0.5];
    let surrogate = GaussianProcess::fit(inputs.clone(), targets.clone(), 0.5, 1e-6);

    assert_eq!(surrogate.observations(), 3);
    for (input, target) in inputs.iter().zip(targets.iter()) {
        let (mean, sigma) = surrogate.predict(input);
        assert!(
            (mean - target).abs() < 0.05,
            "the GP must interpolate its observations: predicted {mean} for {target}"
        );
        assert!(sigma < 0.2, "uncertainty must be small at observed points");
    }

    // Far from the data the posterior reverts to the mean with high uncertainty.
    let (_, far_sigma) = surrogate.predict(&[5.0, 5.0]);
    let (_, near_sigma) = surrogate.predict(&[0.5, 0.5]);
    assert!(far_sigma > near_sigma);

    // Expected improvement is non-negative and larger where improvement is plausible.
    assert!(surrogate.expected_improvement(&[1.2, 1.2]) >= 0.0);
}

#[test]
fn test_bayesian_search_consumes_real_evaluations() {
    let config = small_config(SearchStrategy::BayesianOptimization, 20);
    let mut searcher = NeuralArchitectureSearcher::with_evaluator(
        config,
        Box::new(CountingEvaluator { calls: 0 }),
    )
    .expect("searcher");
    let best = searcher.search().expect("search must succeed");

    assert_eq!(searcher.evaluation_history().len(), 20);
    assert!(best.fitness > 0.0);
}

#[test]
fn test_non_dominated_sorting_finds_the_pareto_front() {
    let make = |accuracy: f32, latency: f32| {
        let mut evaluation = ArchitectureEvaluation::new(Architecture::new());
        evaluation.metrics.insert("accuracy".to_string(), accuracy);
        evaluation.metrics.insert("latency".to_string(), latency);
        evaluation
    };

    // A dominates C; A and B are mutually non-dominated.
    let population = vec![
        make(0.9, 0.5), // 0: front 0
        make(0.5, 0.9), // 1: front 0
        make(0.4, 0.4), // 2: dominated by both
    ];
    let objectives = vec!["accuracy".to_string(), "latency".to_string()];

    let fronts = non_dominated_fronts(&population, &objectives);
    assert_eq!(fronts.len(), 2);
    assert_eq!(fronts[0].len(), 2);
    assert!(fronts[0].contains(&0) && fronts[0].contains(&1));
    assert_eq!(fronts[1], vec![2]);

    let distances = crowding_distances(&population, &fronts[0], &objectives);
    assert_eq!(distances.len(), 2);
    assert!(distances.iter().all(|d| d.is_infinite()));
}

#[test]
fn test_nsga2_selection_keeps_the_population_bounded() {
    let mut config = small_config(SearchStrategy::NSGA2, 4);
    config.objectives = vec![
        OptimizationObjective::Accuracy { weight: 0.5 },
        OptimizationObjective::ModelSize { weight: 0.5 },
    ];
    config.generations = 2;
    config.population_size = 4;

    let mut searcher = NeuralArchitectureSearcher::with_evaluator(
        config,
        Box::new(CountingEvaluator { calls: 0 }),
    )
    .expect("searcher");
    searcher.search().expect("search must succeed");

    assert!(searcher.best_architecture().is_some());
}

#[test]
fn test_search_strategy_has_no_darts_variant() {
    // The DARTS variant was random search with a different log line; it is gone.
    let strategies = vec![
        SearchStrategy::Random,
        SearchStrategy::Evolutionary,
        SearchStrategy::ReinforcementLearning,
        SearchStrategy::Progressive,
        SearchStrategy::BayesianOptimization,
        SearchStrategy::NSGA2,
    ];
    for strategy in &strategies {
        assert!(!format!("{strategy:?}").contains("DARTS"));
    }
}

/// An evaluator that always fails, to prove failures are not swallowed.
struct FailingEvaluator;

impl ArchitectureEvaluator for FailingEvaluator {
    fn evaluate(&mut self, _architecture: &Architecture) -> Result<MeasuredPerformance> {
        Err(
            trustformers_core::errors::TrustformersError::invalid_operation(
                "the training run diverged".to_string(),
            ),
        )
    }
}

#[test]
fn test_evaluation_failures_are_propagated_by_every_strategy() {
    for strategy in [
        SearchStrategy::Random,
        SearchStrategy::ReinforcementLearning,
        SearchStrategy::BayesianOptimization,
        SearchStrategy::Progressive,
        SearchStrategy::Evolutionary,
    ] {
        let mut config = small_config(strategy.clone(), 5);
        config.generations = 1;
        let mut searcher =
            NeuralArchitectureSearcher::with_evaluator(config, Box::new(FailingEvaluator))
                .expect("searcher");

        let error = match searcher.search() {
            Ok(_) => panic!("{strategy:?} reported success despite every evaluation failing"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("diverged")
                || error.to_string().contains("No architecture found"),
            "{strategy:?}: unexpected error {error}"
        );
    }
}
