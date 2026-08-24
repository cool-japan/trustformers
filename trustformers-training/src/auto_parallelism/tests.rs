//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use std::time::Duration;

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;

    // ── Evaluation modes and strategy generation ─────────────────────────────

    #[test]
    fn test_profiling_based_evaluation_reports_that_it_is_unavailable() {
        // Regression: this mode silently returned analytical estimates while the caller had
        // explicitly asked for measurements.
        let config = AutoParallelismConfig {
            evaluation_method: EvaluationMethod::ProfilingBased,
            ..Default::default()
        };
        let selector = AutoParallelismSelector::new(config);
        let strategies =
            selector.generate_cost_based_strategies().expect("candidate generation failed");

        let err = selector
            .evaluate_profiling_based(strategies)
            .expect_err("profiling must not fabricate measurements");
        let message = err.to_string();
        assert!(
            message.contains("ProfilingBased"),
            "the error must name the mode: {message}"
        );
    }

    #[test]
    fn test_simulation_based_evaluation_differs_from_the_model() {
        // Regression: `evaluate_simulation_based` was `self.evaluate_model_based(..)`.
        let config = AutoParallelismConfig::default();
        let selector = AutoParallelismSelector::new(config);
        let strategy = selector.create_3d_strategy_with_config(1, 1, 4).expect("3d strategy");

        let analytic = selector.evaluate_model_based(vec![strategy.clone()]).expect("model based");
        let simulated =
            selector.evaluate_simulation_based(vec![strategy]).expect("simulation based");

        assert!(
            simulated[0].expected_performance.time_per_step
                > analytic[0].expected_performance.time_per_step,
            "a 4-stage pipeline must pay a bubble in simulation: {:?} vs {:?}",
            simulated[0].expected_performance.time_per_step,
            analytic[0].expected_performance.time_per_step
        );
        assert!(
            simulated[0].expected_performance.throughput
                < analytic[0].expected_performance.throughput,
            "the extra time must lower the throughput"
        );
        assert!(simulated[0].rationale.contains("simulated pipeline schedule"));
    }

    #[test]
    fn test_simulation_matches_the_model_without_a_pipeline_dimension() {
        let selector = AutoParallelismSelector::new(AutoParallelismConfig::default());
        let strategy = selector.create_3d_strategy_with_config(4, 1, 1).expect("3d strategy");

        let analytic = selector.evaluate_model_based(vec![strategy.clone()]).expect("model based");
        let simulated =
            selector.evaluate_simulation_based(vec![strategy]).expect("simulation based");
        assert_eq!(
            simulated[0].expected_performance.time_per_step,
            analytic[0].expected_performance.time_per_step,
            "with pp = 1 there is no schedule to simulate"
        );
    }

    #[test]
    fn test_hybrid_evaluation_is_conservative_and_lowers_confidence_on_disagreement() {
        let selector = AutoParallelismSelector::new(AutoParallelismConfig::default());
        let strategy = selector.create_3d_strategy_with_config(1, 1, 4).expect("3d strategy");

        let analytic = selector.evaluate_model_based(vec![strategy.clone()]).expect("model based");
        let hybrid = selector.evaluate_hybrid(vec![strategy]).expect("hybrid");

        assert!(
            hybrid[0].expected_performance.time_per_step
                >= analytic[0].expected_performance.time_per_step,
            "the hybrid estimate must take the slower of the two"
        );
        assert!(
            hybrid[0].confidence < analytic[0].confidence,
            "disagreement between the two models must reduce the confidence ({} vs {})",
            hybrid[0].confidence,
            analytic[0].confidence
        );
        assert!(hybrid[0].rationale.contains("hybrid"));
    }

    #[test]
    fn test_annealing_search_explores_more_than_one_configuration() {
        // Regression: this delegated verbatim to `generate_cost_based_strategies`.
        let selector = AutoParallelismSelector::new(AutoParallelismConfig::default());
        let annealed = selector.generate_annealing_strategies().expect("annealing failed");
        assert!(
            annealed.len() > 1,
            "the annealing walk must visit more than the initial configuration"
        );

        // Deterministic for a fixed problem definition.
        let again = selector.generate_annealing_strategies().expect("annealing failed");
        let ids: Vec<&str> = annealed.iter().map(|s| s.strategy_id.as_str()).collect();
        let ids_again: Vec<&str> = again.iter().map(|s| s.strategy_id.as_str()).collect();
        assert_eq!(ids, ids_again, "the seeded search must be reproducible");

        // Every configuration must respect the device budget.
        let max_devices = selector.config.hardware_constraints.num_devices;
        for strategy in &annealed {
            if let Some(cfg) = &strategy.parallelism_3d {
                assert!(
                    cfg.dp_size * cfg.mp_size * cfg.pp_size <= max_devices,
                    "annealing produced an infeasible configuration"
                );
            }
        }
    }

    #[test]
    fn test_multi_objective_returns_a_pareto_subset() {
        // Regression: this delegated verbatim to `generate_cost_based_strategies`.
        let selector = AutoParallelismSelector::new(AutoParallelismConfig::default());
        let all = selector.generate_cost_based_strategies().expect("cost based");
        let front = selector.generate_multi_objective_strategies().expect("multi objective");

        assert!(!front.is_empty());
        assert!(
            front.len() <= all.len(),
            "a Pareto front cannot be larger than the candidate set"
        );

        // No member of the front may be dominated by another candidate.
        let key = |s: &ParallelismStrategy| {
            [
                -s.expected_performance.throughput,
                s.expected_performance.memory_per_device as f64,
                s.expected_performance.communication_overhead as f64,
            ]
        };
        for member in &front {
            let m = key(member);
            for other in &all {
                let o = key(other);
                let dominates = o.iter().zip(m.iter()).all(|(x, y)| x <= y)
                    && o.iter().zip(m.iter()).any(|(x, y)| x < y);
                assert!(!dominates, "front member is dominated by another candidate");
            }
        }
    }

    #[test]
    fn test_auto_parallelism_config() {
        let config = AutoParallelismConfig::default();
        assert!(config.enabled);
        assert_eq!(config.hardware_constraints.num_devices, 8);
    }

    #[test]
    fn test_auto_parallelism_selector_creation() {
        let config = AutoParallelismConfig::default();
        let selector = AutoParallelismSelector::new(config);
        assert!(selector.current_strategy.is_none());
    }

    #[test]
    fn test_strategy_selection() {
        let config = AutoParallelismConfig::default();
        let mut selector = AutoParallelismSelector::new(config);

        let strategy = selector.select_strategy();
        assert!(strategy.is_ok());
        assert!(selector.current_strategy.is_some());
    }

    #[test]
    fn test_rule_based_strategy_generation() {
        let config = AutoParallelismConfig {
            selection_algorithm: SelectionAlgorithm::RuleBased,
            ..Default::default()
        };
        let selector = AutoParallelismSelector::new(config);

        let strategies = selector.generate_rule_based_strategies();
        assert!(strategies.is_ok());
        assert!(!strategies.expect("operation failed in test").is_empty());
    }

    #[test]
    fn test_performance_estimation() {
        let config = AutoParallelismConfig::default();
        let selector = AutoParallelismSelector::new(config);

        let metrics = selector.estimate_performance_data_parallel();
        assert!(metrics.is_ok());

        let metrics = metrics.expect("operation failed in test");
        assert!(metrics.time_per_step.as_secs_f64() > 0.0);
        assert!(metrics.memory_per_device > 0);
    }

    #[test]
    fn test_strategy_comparison() {
        let config = AutoParallelismConfig {
            optimization_objective: OptimizationObjective::MinimizeTime,
            ..Default::default()
        };
        let selector = AutoParallelismSelector::new(config);

        let strategy1 = ParallelismStrategy {
            strategy_id: "test1".to_string(),
            data_parallel: None,
            parallelism_3d: None,
            expert_parallel: None,
            sequence_parallel: None,
            tensor_parallel: None,
            expected_performance: PerformanceMetrics {
                time_per_step: Duration::from_secs(1),
                memory_per_device: 1000,
                communication_overhead: 0.1,
                throughput: 1.0,
                efficiency: 0.8,
                scalability: 0.9,
            },
            confidence: 0.8,
            rationale: "Test strategy 1".to_string(),
        };

        let strategy2 = ParallelismStrategy {
            strategy_id: "test2".to_string(),
            data_parallel: None,
            parallelism_3d: None,
            expert_parallel: None,
            sequence_parallel: None,
            tensor_parallel: None,
            expected_performance: PerformanceMetrics {
                time_per_step: Duration::from_secs(2),
                memory_per_device: 800,
                communication_overhead: 0.05,
                throughput: 0.5,
                efficiency: 0.9,
                scalability: 0.85,
            },
            confidence: 0.9,
            rationale: "Test strategy 2".to_string(),
        };

        let comparison = selector.compare_strategies(&strategy1, &strategy2);
        assert!(comparison.is_ok());
        assert_eq!(
            comparison.expect("operation failed in test"),
            std::cmp::Ordering::Less
        ); // strategy1 has less time
    }

    #[test]
    fn test_memory_estimation() {
        let constraints = ModelConstraints {
            num_parameters: 1_000_000,
            ..Default::default()
        };

        let memory = utils::estimate_model_memory(&constraints);
        assert_eq!(memory, 16_000_000); // 4 * 4 * 1M = 16MB
    }

    #[test]
    fn test_requirements_checking() {
        let strategy = ParallelismStrategy {
            strategy_id: "test".to_string(),
            data_parallel: None,
            parallelism_3d: None,
            expert_parallel: None,
            sequence_parallel: None,
            tensor_parallel: None,
            expected_performance: PerformanceMetrics {
                time_per_step: Duration::from_secs(1),
                memory_per_device: 1000,
                communication_overhead: 0.2,
                throughput: 2.0,
                efficiency: 0.8,
                scalability: 0.9,
            },
            confidence: 0.8,
            rationale: "Test strategy".to_string(),
        };

        let requirements = PerformanceRequirements {
            max_training_time: Some(Duration::from_secs(2)),
            min_throughput: Some(1.0),
            max_memory_per_device: Some(2000),
            max_communication_overhead: Some(0.3),
            min_efficiency: Some(0.7),
        };

        assert!(utils::meets_requirements(&strategy, &requirements));

        let strict_requirements = PerformanceRequirements {
            max_training_time: Some(Duration::from_millis(500)),
            ..requirements
        };

        assert!(!utils::meets_requirements(&strategy, &strict_requirements));
    }
}
