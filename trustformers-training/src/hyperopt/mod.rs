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
}
