//! Automated Hyperparameter Tuning Framework
//!
//! This module provides a comprehensive hyperparameter optimization framework
//! for TrustformeRS models, supporting multiple search strategies and automated
//! experiment tracking.

pub mod auto_tuner;
pub mod efficiency;
pub mod examples;
pub mod sampler;
pub mod search_space;
pub mod strategies;
pub mod surrogate_models;
pub mod trial;
pub mod tuner;

use serde::{Deserialize, Serialize};

pub use auto_tuner::{
    AcquisitionFunction as AutoTunerAcquisitionFunction, AutomatedHyperparameterTuner,
    BayesianOptimizationTuner, GaussianProcess, HyperparameterConfig, HyperparameterSpace,
    HyperparameterTuner as AutoTunerHyperparameterTuner, Kernel,
    OptimizationDirection as AutoTunerOptimizationDirection, ParameterConstraint, ParameterScale,
    ParameterSpec, ParameterValue as AutoTunerParameterValue, RandomSearchTuner,
    ResourceAllocation as AutoTunerResourceAllocation, ResourceSharingStrategy, SearchAlgorithm,
    TuningConfig, TuningResult,
};
pub use efficiency::{
    AcquisitionFunction, AcquisitionFunctionType, AdvancedEarlyStoppingConfig,
    ArmGenerationStrategy, ArmStatistics, BanditAlgorithm, BanditConfig, BanditOptimizer,
    EarlyStoppingStrategy, EvaluationJob, EvaluationResult, ExplorationStrategy,
    FaultToleranceConfig, GPUAllocation, JobStatus, KernelType, LoadBalancer,
    ParallelEvaluationConfig, ParallelEvaluator, ParallelStrategy, PriorityLevel,
    ResourceAllocation, ResourceUsage, RewardFunction, SurrogateConfig, SurrogateModel,
    SurrogateModelType, SurrogateOptimizer, WarmStartConfig, WarmStartDataSource,
    WarmStartStrategy,
};
pub use examples::{
    computer_vision_objective, language_modeling_objective, params_to_training_args,
    HyperparameterOptimizer, HyperparameterStudy, MultiStrategyOptimizer,
};
pub use sampler::{GPSampler, RandomSampler, Sampler, SamplerConfig, TPESampler};
pub use search_space::{
    CategoricalParameter, ContinuousParameter, DiscreteParameter, HyperParameter, LogParameter,
    ParameterValue, SearchSpace,
};
pub use strategies::{
    BayesianOptimization, GridSearch, HalvingStrategy, Hyperband, PBTConfig, PBTMember, PBTStats,
    PopulationBasedTraining, RandomSearch, SearchStrategy, SuccessiveHalving,
};
pub use surrogate_models::{
    create_acquisition_function, create_surrogate_model, ExpectedImprovement,
    SimpleGaussianProcess, UpperConfidenceBound,
};
pub use trial::{Trial, TrialHistory, TrialMetrics, TrialResult, TrialState};
pub use tuner::{HyperparameterTuner, OptimizationDirection, StudyStatistics, TunerConfig};

/// Direction for optimization (minimize or maximize the objective)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Direction {
    /// Minimize the objective value (e.g., loss)
    Minimize,
    /// Maximize the objective value (e.g., accuracy)
    Maximize,
}

/// Result of a hyperparameter optimization study
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Best trial found
    pub best_trial: Trial,
    /// All trials run during the study
    pub trials: Vec<Trial>,
    /// Number of trials that completed successfully
    pub completed_trials: usize,
    /// Number of trials that failed
    pub failed_trials: usize,
    /// Total time spent on optimization
    pub total_duration: std::time::Duration,
    /// Statistics about the study
    pub statistics: StudyStatistics,
}

/// Configuration for early stopping of trials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Patience: number of evaluation steps to wait before stopping
    pub patience: usize,
    /// Minimum improvement threshold
    pub min_delta: f64,
    /// Whether to restore best weights when stopping
    pub restore_best_weights: bool,
}

/// Configuration for pruning unpromising trials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    /// Strategy to use for pruning
    pub strategy: PruningStrategy,
    /// Minimum number of steps before pruning can occur
    pub min_steps: usize,
    /// Percentile threshold for pruning (e.g., 0.5 = median)
    pub percentile: f64,
}

/// Strategy for pruning trials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PruningStrategy {
    /// No pruning
    None,
    /// Median pruning: stop if performance is below median
    Median,
    /// Percentile pruning: stop if performance is below specified percentile
    Percentile(f64),
    /// Successive halving: eliminate worst performing trials at each stage
    SuccessiveHalving,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direction() {
        assert_eq!(Direction::Minimize, Direction::Minimize);
        assert_ne!(Direction::Minimize, Direction::Maximize);
    }

    #[test]
    fn test_pruning_strategy() {
        let strategy = PruningStrategy::Percentile(0.25);
        match strategy {
            PruningStrategy::Percentile(p) => assert_eq!(p, 0.25),
            _ => panic!("Expected Percentile strategy"),
        }
    }

    // ── Direction serialization round-trip ──────────────────────────────────

    #[test]
    fn test_direction_serialize_roundtrip() {
        let minimize = Direction::Minimize;
        let json = serde_json::to_string(&minimize).expect("serialize Direction::Minimize");
        let back: Direction = serde_json::from_str(&json).expect("deserialize Direction::Minimize");
        assert_eq!(back, Direction::Minimize);

        let maximize = Direction::Maximize;
        let json2 = serde_json::to_string(&maximize).expect("serialize Direction::Maximize");
        let back2: Direction =
            serde_json::from_str(&json2).expect("deserialize Direction::Maximize");
        assert_eq!(back2, Direction::Maximize);
    }

    #[test]
    fn test_direction_clone() {
        let d = Direction::Maximize;
        let d2 = d.clone();
        assert_eq!(d, d2);
    }

    // ── PruningConfig construction ──────────────────────────────────────────

    #[test]
    fn test_pruning_config_construction() {
        let config = PruningConfig {
            strategy: PruningStrategy::Median,
            min_steps: 10,
            percentile: 0.5,
        };
        assert_eq!(config.min_steps, 10);
        assert!((config.percentile - 0.5).abs() < 1e-10);
        assert!(matches!(config.strategy, PruningStrategy::Median));
    }

    #[test]
    fn test_pruning_config_none_strategy() {
        let config = PruningConfig {
            strategy: PruningStrategy::None,
            min_steps: 0,
            percentile: 0.0,
        };
        assert!(matches!(config.strategy, PruningStrategy::None));
    }

    #[test]
    fn test_pruning_config_successive_halving() {
        let config = PruningConfig {
            strategy: PruningStrategy::SuccessiveHalving,
            min_steps: 5,
            percentile: 0.3,
        };
        assert!(matches!(
            config.strategy,
            PruningStrategy::SuccessiveHalving
        ));
    }

    // ── EarlyStoppingConfig ─────────────────────────────────────────────────

    #[test]
    fn test_early_stopping_config_construction() {
        let config = EarlyStoppingConfig {
            patience: 5,
            min_delta: 1e-4,
            restore_best_weights: true,
        };
        assert_eq!(config.patience, 5);
        assert!((config.min_delta - 1e-4).abs() < 1e-12);
        assert!(config.restore_best_weights);
    }

    #[test]
    fn test_early_stopping_config_no_restore() {
        let config = EarlyStoppingConfig {
            patience: 3,
            min_delta: 0.01,
            restore_best_weights: false,
        };
        assert!(!config.restore_best_weights);
    }

    // ── PruningStrategy clone and debug ────────────────────────────────────

    #[test]
    fn test_pruning_strategy_debug_output() {
        let strat = PruningStrategy::SuccessiveHalving;
        let debug_str = format!("{strat:?}");
        assert!(debug_str.contains("SuccessiveHalving"), "got: {debug_str}");
    }

    #[test]
    fn test_pruning_strategy_percentile_clone() {
        let s = PruningStrategy::Percentile(0.75);
        let s2 = s.clone();
        if let PruningStrategy::Percentile(p) = s2 {
            assert!((p - 0.75).abs() < 1e-10);
        } else {
            panic!("Expected Percentile after clone");
        }
    }

    #[test]
    fn test_direction_debug_output() {
        let d = Direction::Minimize;
        let debug_str = format!("{d:?}");
        assert!(debug_str.contains("Minimize"), "got: {debug_str}");
    }

    // ── StudyStatistics structure ───────────────────────────────────────────

    #[test]
    fn test_study_statistics_fields_accessible() {
        let stats = StudyStatistics {
            total_trials: 5,
            completed_trials: 4,
            failed_trials: 1,
            pruned_trials: 0,
            best_value: Some(0.9),
            best_trial_number: Some(2),
            total_duration: std::time::Duration::from_secs(10),
            average_trial_duration: std::time::Duration::from_secs(2),
            success_rate: 80.0,
            pruning_rate: 0.0,
        };
        assert_eq!(stats.total_trials, 5);
        assert_eq!(stats.completed_trials, 4);
        assert_eq!(stats.failed_trials, 1);
        assert!(stats.best_value.is_some());
        assert_eq!(stats.best_trial_number, Some(2));
    }

    // ── EarlyStoppingConfig serialization ──────────────────────────────────

    #[test]
    fn test_early_stopping_config_serialize_roundtrip() {
        let config = EarlyStoppingConfig {
            patience: 7,
            min_delta: 0.001,
            restore_best_weights: true,
        };
        let json = serde_json::to_string(&config).expect("serialize EarlyStoppingConfig");
        let back: EarlyStoppingConfig =
            serde_json::from_str(&json).expect("deserialize EarlyStoppingConfig");
        assert_eq!(back.patience, 7);
        assert!((back.min_delta - 0.001).abs() < 1e-10);
        assert!(back.restore_best_weights);
    }

    // ── PruningConfig serialization ─────────────────────────────────────────

    #[test]
    fn test_pruning_config_serialize_roundtrip() {
        let config = PruningConfig {
            strategy: PruningStrategy::Percentile(0.4),
            min_steps: 8,
            percentile: 0.4,
        };
        let json = serde_json::to_string(&config).expect("serialize PruningConfig");
        let back: PruningConfig = serde_json::from_str(&json).expect("deserialize PruningConfig");
        assert_eq!(back.min_steps, 8);
        if let PruningStrategy::Percentile(p) = back.strategy {
            assert!((p - 0.4).abs() < 1e-10);
        } else {
            panic!("Expected Percentile strategy after roundtrip");
        }
    }
}
