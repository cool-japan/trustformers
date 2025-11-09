//! Main hyperparameter tuner implementation

use super::{
    BayesianOptimization, Direction, EarlyStoppingConfig, ParameterValue, PruningConfig,
    PruningStrategy, RandomSearch, SearchSpace, SearchStrategy, Trial, TrialHistory, TrialResult,
};
use crate::TrainingArguments;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use trustformers_core::errors::{file_not_found, invalid_format, Result};

/// Direction for optimization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationDirection {
    /// Minimize the objective (e.g., loss)
    Minimize,
    /// Maximize the objective (e.g., accuracy)
    Maximize,
}

impl From<OptimizationDirection> for Direction {
    fn from(dir: OptimizationDirection) -> Self {
        match dir {
            OptimizationDirection::Minimize => Direction::Minimize,
            OptimizationDirection::Maximize => Direction::Maximize,
        }
    }
}

/// Configuration for the hyperparameter tuner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunerConfig {
    /// Name of the study
    pub study_name: String,
    /// Direction of optimization
    pub direction: OptimizationDirection,
    /// Name of the metric to optimize
    pub objective_metric: String,
    /// Maximum number of trials
    pub max_trials: Option<usize>,
    /// Maximum time to spend on optimization
    pub max_duration: Option<Duration>,
    /// Early stopping configuration
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// Pruning configuration
    pub pruning: Option<PruningConfig>,
    /// Directory to save study results
    pub output_dir: PathBuf,
    /// Whether to save intermediate checkpoints
    pub save_checkpoints: bool,
    /// Minimum number of trials before considering pruning
    pub min_trials_for_pruning: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for TunerConfig {
    fn default() -> Self {
        Self {
            study_name: "hyperparameter_study".to_string(),
            direction: OptimizationDirection::Maximize,
            objective_metric: "eval_accuracy".to_string(),
            max_trials: Some(100),
            max_duration: None,
            early_stopping: None,
            pruning: None,
            output_dir: PathBuf::from("./hyperopt_results"),
            save_checkpoints: true,
            min_trials_for_pruning: 10,
            seed: None,
        }
    }
}

impl TunerConfig {
    /// Create a new tuner configuration
    pub fn new(study_name: impl Into<String>) -> Self {
        Self {
            study_name: study_name.into(),
            ..Default::default()
        }
    }

    /// Set the optimization direction
    pub fn direction(mut self, direction: OptimizationDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Set the objective metric name
    pub fn objective_metric(mut self, metric: impl Into<String>) -> Self {
        self.objective_metric = metric.into();
        self
    }

    /// Set the maximum number of trials
    pub fn max_trials(mut self, max_trials: usize) -> Self {
        self.max_trials = Some(max_trials);
        self
    }

    /// Set the maximum duration
    pub fn max_duration(mut self, duration: Duration) -> Self {
        self.max_duration = Some(duration);
        self
    }

    /// Set early stopping configuration
    pub fn early_stopping(mut self, config: EarlyStoppingConfig) -> Self {
        self.early_stopping = Some(config);
        self
    }

    /// Set pruning configuration
    pub fn pruning(mut self, config: PruningConfig) -> Self {
        self.pruning = Some(config);
        self
    }

    /// Set the output directory
    pub fn output_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.output_dir = dir.into();
        self
    }

    /// Set random seed
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Statistics about an optimization study
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudyStatistics {
    /// Total number of trials
    pub total_trials: usize,
    /// Number of completed trials
    pub completed_trials: usize,
    /// Number of failed trials
    pub failed_trials: usize,
    /// Number of pruned trials
    pub pruned_trials: usize,
    /// Best objective value found
    pub best_value: Option<f64>,
    /// Best trial
    pub best_trial_number: Option<usize>,
    /// Total time spent
    pub total_duration: Duration,
    /// Average trial duration
    pub average_trial_duration: Duration,
    /// Success rate (percentage)
    pub success_rate: f64,
    /// Pruning rate (percentage)
    pub pruning_rate: f64,
}

/// Callback for hyperparameter tuning events
pub trait HyperparameterCallback: Send + Sync {
    /// Called when a study starts
    fn on_study_start(&mut self, _config: &TunerConfig) {}

    /// Called when a study ends
    fn on_study_end(&mut self, _config: &TunerConfig, _statistics: &StudyStatistics) {}

    /// Called when a trial starts
    fn on_trial_start(&mut self, _trial: &Trial) {}

    /// Called when a trial completes
    fn on_trial_complete(&mut self, _trial: &Trial) {}

    /// Called when a trial is pruned
    fn on_trial_pruned(&mut self, _trial: &Trial, _reason: &str) {}

    /// Called when a new best trial is found
    fn on_new_best(&mut self, _trial: &Trial, _improvement: f64) {}
}

/// Default callback that logs events
pub struct LoggingCallback;

impl HyperparameterCallback for LoggingCallback {
    fn on_study_start(&mut self, config: &TunerConfig) {
        println!("Starting hyperparameter study: {}", config.study_name);
        println!("Direction: {:?}", config.direction);
        println!("Objective metric: {}", config.objective_metric);
        if let Some(max_trials) = config.max_trials {
            println!("Max trials: {}", max_trials);
        }
    }

    fn on_study_end(&mut self, _config: &TunerConfig, statistics: &StudyStatistics) {
        println!("\nHyperparameter study completed!");
        println!("Total trials: {}", statistics.total_trials);
        println!("Completed trials: {}", statistics.completed_trials);
        println!("Success rate: {:.2}%", statistics.success_rate);
        if let Some(best_value) = statistics.best_value {
            println!("Best value: {:.6}", best_value);
        }
        println!("Total duration: {:?}", statistics.total_duration);
    }

    fn on_trial_start(&mut self, trial: &Trial) {
        println!("Starting trial {}: {}", trial.number, trial.summary());
    }

    fn on_trial_complete(&mut self, trial: &Trial) {
        println!("Completed trial {}: {}", trial.number, trial.summary());
    }

    fn on_trial_pruned(&mut self, trial: &Trial, reason: &str) {
        println!(
            "Pruned trial {} ({}): {}",
            trial.number,
            reason,
            trial.summary()
        );
    }

    fn on_new_best(&mut self, trial: &Trial, improvement: f64) {
        println!(
            "New best trial {}: improvement={:.6}, {}",
            trial.number,
            improvement,
            trial.summary()
        );
    }
}

/// Main hyperparameter tuner
pub struct HyperparameterTuner {
    /// Configuration
    config: TunerConfig,
    /// Search space
    search_space: SearchSpace,
    /// Search strategy
    strategy: Box<dyn SearchStrategy>,
    /// Trial history
    history: TrialHistory,
    /// Start time of the study
    start_time: Option<Instant>,
    /// Callbacks
    callbacks: Vec<Box<dyn HyperparameterCallback>>,
    /// Current trial number
    current_trial_number: usize,
}

impl HyperparameterTuner {
    /// Create a new hyperparameter tuner
    pub fn new(
        config: TunerConfig,
        search_space: SearchSpace,
        strategy: Box<dyn SearchStrategy>,
    ) -> Self {
        let direction = config.direction.clone().into();

        Self {
            config,
            search_space,
            strategy,
            history: TrialHistory::new(direction),
            start_time: None,
            callbacks: vec![Box::new(LoggingCallback)],
            current_trial_number: 0,
        }
    }

    /// Create a tuner with random search strategy
    pub fn with_random_search(config: TunerConfig, search_space: SearchSpace) -> Self {
        let max_trials = config.max_trials.unwrap_or(100);
        let strategy = if let Some(seed) = config.seed {
            Box::new(RandomSearch::with_seed(max_trials, seed))
        } else {
            Box::new(RandomSearch::new(max_trials))
        };

        Self::new(config, search_space, strategy)
    }

    /// Create a tuner with Bayesian optimization strategy
    pub fn with_bayesian_optimization(config: TunerConfig, search_space: SearchSpace) -> Self {
        let max_trials = config.max_trials.unwrap_or(100);
        let strategy = Box::new(BayesianOptimization::new(max_trials));

        Self::new(config, search_space, strategy)
    }

    /// Add a callback
    pub fn add_callback(mut self, callback: Box<dyn HyperparameterCallback>) -> Self {
        self.callbacks.push(callback);
        self
    }

    /// Get the current best trial
    pub fn best_trial(&self) -> Option<&Trial> {
        self.history.best_trial()
    }

    /// Get the best value found so far
    pub fn best_value(&self) -> Option<f64> {
        self.history.best_value()
    }

    /// Get all trials
    pub fn trials(&self) -> &[Trial] {
        &self.history.trials
    }

    /// Get study statistics
    pub fn statistics(&self) -> StudyStatistics {
        let trial_stats = self.history.statistics();
        let total_duration =
            self.start_time.map(|start| start.elapsed()).unwrap_or(Duration::from_secs(0));

        StudyStatistics {
            total_trials: trial_stats.total_trials,
            completed_trials: trial_stats.completed_trials,
            failed_trials: trial_stats.failed_trials,
            pruned_trials: trial_stats.pruned_trials,
            best_value: trial_stats.best_value,
            best_trial_number: self.best_trial().map(|t| t.number),
            total_duration,
            average_trial_duration: trial_stats.average_trial_duration,
            success_rate: trial_stats.success_rate(),
            pruning_rate: trial_stats.pruning_rate(),
        }
    }

    /// Run the hyperparameter optimization study
    pub fn optimize<F>(&mut self, mut objective_fn: F) -> Result<super::OptimizationResult>
    where
        F: FnMut(HashMap<String, ParameterValue>) -> Result<TrialResult>,
    {
        self.start_time = Some(Instant::now());

        // Create output directory
        std::fs::create_dir_all(&self.config.output_dir)
            .map_err(|e| file_not_found(e.to_string()))?;

        // Notify callbacks
        for callback in &mut self.callbacks {
            callback.on_study_start(&self.config);
        }

        let mut last_best_value = None;

        // Main optimization loop
        while !self.should_terminate() {
            // Get next suggestion
            if let Some(params) = self.strategy.suggest(&self.search_space, &self.history) {
                // Validate parameters
                if let Err(e) = self.search_space.validate(&params) {
                    eprintln!("Warning: Invalid parameters suggested: {}", e);
                    continue;
                }

                // Create new trial
                let mut trial = Trial::new(self.current_trial_number, params);
                self.current_trial_number += 1;

                // Notify callbacks
                for callback in &mut self.callbacks {
                    callback.on_trial_start(&trial);
                }

                // Start the trial
                trial.start();

                // Run the objective function
                match objective_fn(trial.params.clone()) {
                    Ok(result) => {
                        // Check if we should prune this trial
                        if self.should_prune_trial(&trial, &result) {
                            trial.prune("Poor performance");
                            for callback in &mut self.callbacks {
                                callback.on_trial_pruned(&trial, "Poor performance");
                            }
                        } else {
                            // Complete the trial
                            trial.complete(result);

                            // Check for new best
                            if let Some(objective_value) = trial.objective_value() {
                                let is_new_best = match last_best_value {
                                    None => true,
                                    Some(prev_best) => match self.config.direction {
                                        OptimizationDirection::Maximize => {
                                            objective_value > prev_best
                                        },
                                        OptimizationDirection::Minimize => {
                                            objective_value < prev_best
                                        },
                                    },
                                };

                                if is_new_best {
                                    let improvement = match last_best_value {
                                        None => 0.0,
                                        Some(prev) => (objective_value - prev).abs(),
                                    };
                                    last_best_value = Some(objective_value);

                                    for callback in &mut self.callbacks {
                                        callback.on_new_best(&trial, improvement);
                                    }
                                }
                            }

                            for callback in &mut self.callbacks {
                                callback.on_trial_complete(&trial);
                            }
                        }
                    },
                    Err(e) => {
                        // Trial failed
                        let result = TrialResult::failure(e.to_string());
                        trial.complete(result);

                        for callback in &mut self.callbacks {
                            callback.on_trial_complete(&trial);
                        }
                    },
                }

                // Update strategy with completed trial
                self.strategy.update(&trial);

                // Add trial to history
                self.history.add_trial(trial);

                // Save checkpoint if enabled
                if self.config.save_checkpoints {
                    if let Err(e) = self.save_checkpoint() {
                        eprintln!("Warning: Failed to save checkpoint: {}", e);
                    }
                }
            } else {
                // No more suggestions from strategy
                break;
            }
        }

        // Get final statistics
        let statistics = self.statistics();

        // Notify callbacks
        for callback in &mut self.callbacks {
            callback.on_study_end(&self.config, &statistics);
        }

        // Save final results
        self.save_results()?;

        // Create optimization result
        Ok(super::OptimizationResult {
            best_trial: self.best_trial().unwrap_or(&Trial::new(0, HashMap::new())).clone(),
            trials: self.history.trials.clone(),
            completed_trials: statistics.completed_trials,
            failed_trials: statistics.failed_trials,
            total_duration: statistics.total_duration,
            statistics,
        })
    }

    fn should_terminate(&self) -> bool {
        // Check if strategy wants to terminate
        if self.strategy.should_terminate(&self.history) {
            return true;
        }

        // Check max trials
        if let Some(max_trials) = self.config.max_trials {
            if self.history.trials.len() >= max_trials {
                return true;
            }
        }

        // Check max duration
        if let Some(max_duration) = self.config.max_duration {
            if let Some(start_time) = self.start_time {
                if start_time.elapsed() >= max_duration {
                    return true;
                }
            }
        }

        false
    }

    fn should_prune_trial(&self, trial: &Trial, result: &TrialResult) -> bool {
        if let Some(pruning_config) = &self.config.pruning {
            // Only prune if we have enough trials for comparison
            if self.history.completed_trials().len() < self.config.min_trials_for_pruning {
                return false;
            }

            // Check if we have intermediate values to evaluate
            if result.metrics.intermediate_values.is_empty() {
                return false;
            }

            match &pruning_config.strategy {
                PruningStrategy::None => false,
                PruningStrategy::Median => self.is_below_median(trial, result, pruning_config),
                PruningStrategy::Percentile(percentile) => {
                    self.is_below_percentile(trial, result, *percentile, pruning_config)
                },
                PruningStrategy::SuccessiveHalving => {
                    // Implement successive halving pruning logic
                    false // Simplified for now
                },
            }
        } else {
            false
        }
    }

    fn is_below_median(
        &self,
        _trial: &Trial,
        result: &TrialResult,
        config: &PruningConfig,
    ) -> bool {
        self.is_below_percentile(_trial, result, 0.5, config)
    }

    fn is_below_percentile(
        &self,
        _trial: &Trial,
        result: &TrialResult,
        percentile: f64,
        config: &PruningConfig,
    ) -> bool {
        if let Some((latest_step, latest_value)) = result.metrics.intermediate_values.last() {
            if *latest_step < config.min_steps {
                return false;
            }

            // Get intermediate values from other trials at the same step
            let mut values_at_step = Vec::new();
            for historical_trial in self.history.completed_trials() {
                if let Some(trial_result) = &historical_trial.result {
                    if let Some(value) =
                        trial_result.metrics.intermediate_value_at_step(*latest_step)
                    {
                        values_at_step.push(value);
                    }
                }
            }

            if values_at_step.is_empty() {
                return false;
            }

            // Sort values and find percentile
            values_at_step.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let percentile_index = (percentile * (values_at_step.len() - 1) as f64) as usize;
            let percentile_value = values_at_step[percentile_index];

            // Prune if current value is significantly below percentile
            match self.config.direction {
                OptimizationDirection::Maximize => *latest_value < percentile_value,
                OptimizationDirection::Minimize => *latest_value > percentile_value,
            }
        } else {
            false
        }
    }

    fn save_checkpoint(&self) -> Result<()> {
        let checkpoint_path = self.config.output_dir.join("checkpoint.json");
        let checkpoint_data = serde_json::to_string_pretty(&self.history)
            .map_err(|e| invalid_format("json", e.to_string()))?;
        std::fs::write(checkpoint_path, checkpoint_data)
            .map_err(|e| file_not_found(e.to_string()))?;
        Ok(())
    }

    fn save_results(&self) -> Result<()> {
        // Save trial history
        let history_path = self.config.output_dir.join("trial_history.json");
        let history_data = serde_json::to_string_pretty(&self.history)
            .map_err(|e| invalid_format("json", e.to_string()))?;
        std::fs::write(history_path, history_data).map_err(|e| file_not_found(e.to_string()))?;

        // Save statistics
        let stats_path = self.config.output_dir.join("statistics.json");
        let statistics = self.statistics();
        let stats_data = serde_json::to_string_pretty(&statistics)
            .map_err(|e| invalid_format("json", e.to_string()))?;
        std::fs::write(stats_path, stats_data).map_err(|e| file_not_found(e.to_string()))?;

        // Save best parameters
        if let Some(best_trial) = self.best_trial() {
            let best_params_path = self.config.output_dir.join("best_parameters.json");
            let params_data = serde_json::to_string_pretty(&best_trial.params)
                .map_err(|e| invalid_format("json", e.to_string()))?;
            std::fs::write(best_params_path, params_data)
                .map_err(|e| file_not_found(e.to_string()))?;
        }

        Ok(())
    }

    /// Load a previous study from checkpoint
    pub fn load_checkpoint(&mut self, checkpoint_path: &Path) -> Result<()> {
        let checkpoint_data =
            std::fs::read_to_string(checkpoint_path).map_err(|e| file_not_found(e.to_string()))?;
        self.history = serde_json::from_str(&checkpoint_data)
            .map_err(|e| invalid_format("json", e.to_string()))?;

        // Update trial counter
        self.current_trial_number = self.history.trials.len();

        Ok(())
    }
}

/// Helper function to create training arguments from hyperparameters
pub fn hyperparams_to_training_args(
    base_args: &TrainingArguments,
    hyperparams: &HashMap<String, ParameterValue>,
) -> TrainingArguments {
    let mut args = base_args.clone();

    // Update training arguments based on hyperparameters
    for (name, value) in hyperparams {
        match name.as_str() {
            "learning_rate" => {
                if let Some(lr) = value.as_float() {
                    args.learning_rate = lr as f32;
                }
            },
            "weight_decay" => {
                if let Some(wd) = value.as_float() {
                    args.weight_decay = wd as f32;
                }
            },
            "per_device_train_batch_size" | "batch_size" => {
                if let Some(bs) = value.as_int() {
                    args.per_device_train_batch_size = bs as usize;
                }
            },
            "num_train_epochs" => {
                if let Some(epochs) = value.as_float() {
                    args.num_train_epochs = epochs as f32;
                }
            },
            "warmup_ratio" => {
                if let Some(ratio) = value.as_float() {
                    args.warmup_ratio = ratio as f32;
                }
            },
            "adam_beta1" => {
                if let Some(beta1) = value.as_float() {
                    args.adam_beta1 = beta1 as f32;
                }
            },
            "adam_beta2" => {
                if let Some(beta2) = value.as_float() {
                    args.adam_beta2 = beta2 as f32;
                }
            },
            "max_grad_norm" => {
                if let Some(norm) = value.as_float() {
                    args.max_grad_norm = norm as f32;
                }
            },
            "gradient_accumulation_steps" => {
                if let Some(steps) = value.as_int() {
                    args.gradient_accumulation_steps = steps as usize;
                }
            },
            _ => {
                // Unknown hyperparameter, ignore or log warning
                eprintln!("Warning: Unknown hyperparameter: {}", name);
            },
        }
    }

    args
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyperopt::search_space::SearchSpaceBuilder;
    use std::time::Duration;

    #[test]
    fn test_tuner_config() {
        let config = TunerConfig::new("test_study")
            .direction(OptimizationDirection::Minimize)
            .objective_metric("loss")
            .max_trials(50)
            .max_duration(Duration::from_secs(3600))
            .seed(42);

        assert_eq!(config.study_name, "test_study");
        assert_eq!(config.direction, OptimizationDirection::Minimize);
        assert_eq!(config.objective_metric, "loss");
        assert_eq!(config.max_trials, Some(50));
        assert_eq!(config.max_duration, Some(Duration::from_secs(3600)));
        assert_eq!(config.seed, Some(42));
    }

    #[test]
    fn test_hyperparameter_tuner_creation() {
        let config = TunerConfig::new("test");
        let search_space = SearchSpaceBuilder::new()
            .continuous("learning_rate", 1e-5, 1e-1)
            .discrete("batch_size", 8, 64, 8)
            .build();

        let tuner = HyperparameterTuner::with_random_search(config, search_space);

        assert_eq!(tuner.config.study_name, "test");
        assert_eq!(tuner.current_trial_number, 0);
        assert!(tuner.history.trials.is_empty());
    }

    #[test]
    fn test_hyperparams_to_training_args() {
        let base_args = TrainingArguments::default();
        let mut hyperparams = HashMap::new();
        hyperparams.insert("learning_rate".to_string(), ParameterValue::Float(0.001));
        hyperparams.insert("batch_size".to_string(), ParameterValue::Int(32));
        hyperparams.insert("num_train_epochs".to_string(), ParameterValue::Float(5.0));

        let updated_args = hyperparams_to_training_args(&base_args, &hyperparams);

        assert_eq!(updated_args.learning_rate, 0.001);
        assert_eq!(updated_args.per_device_train_batch_size, 32);
        assert_eq!(updated_args.num_train_epochs, 5.0);
    }

    #[test]
    fn test_optimization_direction_conversion() {
        let max_dir: Direction = OptimizationDirection::Maximize.into();
        let min_dir: Direction = OptimizationDirection::Minimize.into();

        assert_eq!(max_dir, Direction::Maximize);
        assert_eq!(min_dir, Direction::Minimize);
    }
}
