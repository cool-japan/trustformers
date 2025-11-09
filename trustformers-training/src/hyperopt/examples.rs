//! Comprehensive examples and utilities for hyperparameter optimization
//!
//! This module provides practical examples showing how to use the hyperparameter
//! optimization framework with TrustformeRS models.

use super::{
    HyperparameterTuner, OptimizationDirection, ParameterValue, PopulationBasedTraining,
    TunerConfig,
};
use crate::hyperopt::search_space::{SearchSpace, SearchSpaceBuilder};
use crate::hyperopt::{BanditAlgorithm, BanditConfig, BanditOptimizer, PBTConfig};
use std::collections::HashMap;
use std::time::Duration;
use trustformers_core::errors::Result;

/// Comprehensive hyperparameter optimization utility
pub struct HyperparameterOptimizer {
    search_space: SearchSpace,
    config: TunerConfig,
}

impl HyperparameterOptimizer {
    /// Create a new hyperparameter optimizer for transformer models
    pub fn for_transformer_model(study_name: &str) -> Self {
        let search_space = SearchSpaceBuilder::new()
            .continuous("learning_rate", 1e-5, 1e-2)
            .log_uniform("weight_decay", 1e-6, 1e-2)
            .discrete("per_device_train_batch_size", 4, 32, 4)
            .discrete("gradient_accumulation_steps", 1, 8, 1)
            .continuous("warmup_ratio", 0.0, 0.2)
            .continuous("adam_beta1", 0.8, 0.99)
            .continuous("adam_beta2", 0.9, 0.999)
            .continuous("max_grad_norm", 0.5, 2.0)
            .categorical(
                "lr_scheduler_type",
                vec![
                    "linear".to_string(),
                    "cosine".to_string(),
                    "cosine_with_restarts".to_string(),
                    "polynomial".to_string(),
                ],
            )
            .categorical(
                "optimizer",
                vec![
                    "adamw".to_string(),
                    "adafactor".to_string(),
                    "sgd".to_string(),
                ],
            )
            .build();

        let config = TunerConfig::new(study_name)
            .direction(OptimizationDirection::Maximize)
            .objective_metric("eval_accuracy")
            .max_trials(100)
            .max_duration(Duration::from_secs(24 * 3600)) // 24 hours
            .seed(42);

        Self {
            search_space,
            config,
        }
    }

    /// Create optimizer for computer vision models
    pub fn for_vision_model(study_name: &str) -> Self {
        let search_space = SearchSpaceBuilder::new()
            .continuous("learning_rate", 1e-5, 1e-1)
            .log_uniform("weight_decay", 1e-6, 1e-2)
            .discrete("batch_size", 8, 128, 8)
            .continuous("dropout_rate", 0.0, 0.5)
            .categorical(
                "augmentation_strategy",
                vec![
                    "basic".to_string(),
                    "autoaugment".to_string(),
                    "randaugment".to_string(),
                    "trivialaugment".to_string(),
                ],
            )
            .continuous("mixup_alpha", 0.0, 1.0)
            .continuous("cutmix_alpha", 0.0, 1.0)
            .discrete("image_size", 224, 512, 32)
            .build();

        let config = TunerConfig::new(study_name)
            .direction(OptimizationDirection::Maximize)
            .objective_metric("eval_top1_accuracy")
            .max_trials(50);

        Self {
            search_space,
            config,
        }
    }

    /// Run optimization with random search
    pub fn optimize_with_random_search<F>(
        &self,
        objective_fn: F,
    ) -> Result<super::OptimizationResult>
    where
        F: FnMut(HashMap<String, ParameterValue>) -> Result<super::TrialResult>,
    {
        let mut tuner =
            HyperparameterTuner::with_random_search(self.config.clone(), self.search_space.clone());

        tuner.optimize(objective_fn)
    }

    /// Run optimization with Bayesian optimization
    pub fn optimize_with_bayesian_optimization<F>(
        &self,
        objective_fn: F,
    ) -> Result<super::OptimizationResult>
    where
        F: FnMut(HashMap<String, ParameterValue>) -> Result<super::TrialResult>,
    {
        let mut tuner = HyperparameterTuner::with_bayesian_optimization(
            self.config.clone(),
            self.search_space.clone(),
        );

        tuner.optimize(objective_fn)
    }

    /// Run optimization with population-based training
    pub fn optimize_with_pbt<F>(&self, objective_fn: F) -> Result<super::OptimizationResult>
    where
        F: FnMut(HashMap<String, ParameterValue>) -> Result<super::TrialResult>,
    {
        let pbt_config = PBTConfig {
            population_size: 20,
            exploit_interval: 1000,
            exploit_fraction: 0.25,
            perturbation_std: 0.15,
            seed: Some(42),
            ..Default::default()
        };

        let strategy = Box::new(PopulationBasedTraining::new(pbt_config, &self.search_space));
        let mut tuner =
            HyperparameterTuner::new(self.config.clone(), self.search_space.clone(), strategy);

        tuner.optimize(objective_fn)
    }

    /// Run optimization with multi-armed bandit
    pub fn optimize_with_bandit<F>(&self, objective_fn: F) -> Result<super::OptimizationResult>
    where
        F: FnMut(HashMap<String, ParameterValue>) -> Result<super::TrialResult>,
    {
        let bandit_config = BanditConfig {
            algorithm: BanditAlgorithm::UCB {
                confidence_parameter: 1.0,
            },
            num_arms: 50,
            ..Default::default()
        };

        let strategy = Box::new(BanditOptimizer::new(bandit_config, &self.search_space)?);
        let mut tuner =
            HyperparameterTuner::new(self.config.clone(), self.search_space.clone(), strategy);

        tuner.optimize(objective_fn)
    }

    /// Get the search space
    pub fn search_space(&self) -> &SearchSpace {
        &self.search_space
    }

    /// Get the tuner configuration
    pub fn config(&self) -> &TunerConfig {
        &self.config
    }
}

/// Example objective function for language modeling
pub fn language_modeling_objective(
    params: HashMap<String, ParameterValue>,
) -> Result<super::TrialResult> {
    // Extract hyperparameters
    let learning_rate = params.get("learning_rate").and_then(|v| v.as_float()).unwrap_or(1e-4);

    let batch_size =
        params.get("per_device_train_batch_size").and_then(|v| v.as_int()).unwrap_or(16) as usize;

    let weight_decay = params.get("weight_decay").and_then(|v| v.as_float()).unwrap_or(1e-2);

    // Simulate training (in practice, this would call actual training)
    let epochs = 3;
    let mut metrics = HashMap::new();

    // Simulate realistic performance based on hyperparameters
    let base_accuracy = 0.75;
    let lr_penalty = if learning_rate > 1e-3 { 0.1 } else { 0.0 };
    let batch_penalty = if batch_size < 8 { 0.05 } else { 0.0 };
    let wd_bonus = if weight_decay > 1e-3 && weight_decay < 1e-2 { 0.02 } else { 0.0 };

    let accuracy = base_accuracy - lr_penalty - batch_penalty + wd_bonus + 0.05 * fastrand::f64(); // Add some randomness

    let loss = 2.0 - accuracy; // Inverse relationship

    metrics.insert("eval_accuracy".to_string(), accuracy);
    metrics.insert("eval_loss".to_string(), loss);
    metrics.insert("train_loss".to_string(), loss + 0.1);

    // Simulate some training dynamics
    let mut intermediate_values = Vec::new();
    for epoch in 1..=epochs {
        let intermediate_acc = accuracy * (epoch as f64 / epochs as f64) * 0.9;
        intermediate_values.push((epoch * 1000, intermediate_acc));
    }

    let trial_metrics = super::TrialMetrics {
        objective_value: accuracy,
        metrics,
        intermediate_values,
    };

    Ok(super::TrialResult::success(trial_metrics))
}

/// Example objective function for computer vision
pub fn computer_vision_objective(
    params: HashMap<String, ParameterValue>,
) -> Result<super::TrialResult> {
    let learning_rate = params.get("learning_rate").and_then(|v| v.as_float()).unwrap_or(1e-3);

    let batch_size = params.get("batch_size").and_then(|v| v.as_int()).unwrap_or(32) as usize;

    let dropout_rate = params.get("dropout_rate").and_then(|v| v.as_float()).unwrap_or(0.1);

    // Simulate training
    let mut metrics = HashMap::new();

    let base_accuracy = 0.85;
    let lr_penalty = if learning_rate > 1e-2 { 0.15 } else { 0.0 };
    let batch_penalty = if batch_size < 16 { 0.03 } else { 0.0 };
    let dropout_bonus = if dropout_rate > 0.0 && dropout_rate < 0.3 { 0.02 } else { 0.0 };

    let top1_accuracy =
        base_accuracy - lr_penalty - batch_penalty + dropout_bonus + 0.03 * fastrand::f64();

    let top5_accuracy = (top1_accuracy + 0.1).min(1.0);

    metrics.insert("eval_top1_accuracy".to_string(), top1_accuracy);
    metrics.insert("eval_top5_accuracy".to_string(), top5_accuracy);
    metrics.insert("eval_loss".to_string(), 1.0 - top1_accuracy);

    let trial_metrics = super::TrialMetrics {
        objective_value: top1_accuracy,
        metrics,
        intermediate_values: vec![(100, top1_accuracy * 0.8), (200, top1_accuracy)],
    };

    Ok(super::TrialResult::success(trial_metrics))
}

/// Utility function to convert hyperparameters to training arguments
pub fn params_to_training_args(
    params: &HashMap<String, ParameterValue>,
) -> crate::TrainingArguments {
    let mut args = crate::TrainingArguments::default();

    // Apply hyperparameters
    for (name, value) in params {
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
            "gradient_accumulation_steps" => {
                if let Some(steps) = value.as_int() {
                    args.gradient_accumulation_steps = steps as usize;
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
            "lr_scheduler_type" => {
                if let Some(_scheduler) = value.as_string() {
                    // Note: TrainingArguments doesn't currently have lr_scheduler_type field
                    // This would need to be handled differently in actual implementation
                }
            },
            _ => {
                // Log unknown parameters
                eprintln!("Unknown hyperparameter: {}", name);
            },
        }
    }

    args
}

/// Create a complete hyperparameter optimization study
pub struct HyperparameterStudy {
    name: String,
    search_space: SearchSpace,
    strategy: String,
}

impl HyperparameterStudy {
    pub fn new(name: String, search_space: SearchSpace, strategy: String) -> Self {
        Self {
            name,
            search_space,
            strategy,
        }
    }

    /// Run a complete hyperparameter optimization study
    pub fn run<F>(&self, objective_fn: F) -> Result<super::OptimizationResult>
    where
        F: FnMut(HashMap<String, ParameterValue>) -> Result<super::TrialResult>,
    {
        let config = TunerConfig::new(&self.name)
            .direction(OptimizationDirection::Maximize)
            .max_trials(50)
            .seed(42);

        let mut tuner = match self.strategy.as_str() {
            "random" => HyperparameterTuner::with_random_search(config, self.search_space.clone()),
            "bayesian" => {
                HyperparameterTuner::with_bayesian_optimization(config, self.search_space.clone())
            },
            _ => HyperparameterTuner::with_random_search(config, self.search_space.clone()),
        };

        tuner.optimize(objective_fn)
    }
}

/// Advanced hyperparameter optimization with multiple strategies
pub struct MultiStrategyOptimizer {
    search_space: SearchSpace,
    config: TunerConfig,
}

impl MultiStrategyOptimizer {
    pub fn new(search_space: SearchSpace, config: TunerConfig) -> Self {
        Self {
            search_space,
            config,
        }
    }

    /// Run optimization with multiple strategies and return the best result
    pub fn optimize_multi_strategy<F>(
        &self,
        objective_fn: F,
    ) -> Result<Vec<super::OptimizationResult>>
    where
        F: FnMut(HashMap<String, ParameterValue>) -> Result<super::TrialResult> + Clone,
    {
        let mut results = Vec::new();

        // Strategy 1: Random Search
        {
            let mut tuner = HyperparameterTuner::with_random_search(
                self.config.clone(),
                self.search_space.clone(),
            );
            let result = tuner.optimize(objective_fn.clone())?;
            results.push(result);
        }

        // Strategy 2: Bayesian Optimization
        {
            let mut tuner = HyperparameterTuner::with_bayesian_optimization(
                self.config.clone(),
                self.search_space.clone(),
            );
            let result = tuner.optimize(objective_fn.clone())?;
            results.push(result);
        }

        // Strategy 3: Multi-armed Bandit
        {
            let bandit_config = BanditConfig::default();
            if let Ok(strategy) = BanditOptimizer::new(bandit_config, &self.search_space) {
                let mut tuner = HyperparameterTuner::new(
                    self.config.clone(),
                    self.search_space.clone(),
                    Box::new(strategy),
                );
                if let Ok(result) = tuner.optimize(objective_fn.clone()) {
                    results.push(result);
                }
            }
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_optimizer() {
        let optimizer = HyperparameterOptimizer::for_transformer_model("test_study");
        assert_eq!(optimizer.config().study_name, "test_study");
        assert!(optimizer.search_space().parameters.len() > 5);
    }

    #[test]
    fn test_vision_optimizer() {
        let optimizer = HyperparameterOptimizer::for_vision_model("vision_study");
        assert_eq!(optimizer.config().study_name, "vision_study");
        assert!(optimizer.search_space().parameters.len() > 5);
    }

    #[test]
    fn test_language_modeling_objective() {
        let mut params = HashMap::new();
        params.insert("learning_rate".to_string(), ParameterValue::Float(1e-4));
        params.insert(
            "per_device_train_batch_size".to_string(),
            ParameterValue::Int(16),
        );
        params.insert("weight_decay".to_string(), ParameterValue::Float(1e-3));

        let result = language_modeling_objective(params).unwrap();
        assert!(result.metrics.objective_value > 0.0);
        assert!(result.metrics.objective_value <= 1.0);
        assert!(result.metrics.metrics.contains_key("eval_accuracy"));
    }

    #[test]
    fn test_computer_vision_objective() {
        let mut params = HashMap::new();
        params.insert("learning_rate".to_string(), ParameterValue::Float(1e-3));
        params.insert("batch_size".to_string(), ParameterValue::Int(32));
        params.insert("dropout_rate".to_string(), ParameterValue::Float(0.1));

        let result = computer_vision_objective(params).unwrap();
        assert!(result.metrics.objective_value > 0.0);
        assert!(result.metrics.objective_value <= 1.0);
        assert!(result.metrics.metrics.contains_key("eval_top1_accuracy"));
    }

    #[test]
    fn test_params_to_training_args() {
        let mut params = HashMap::new();
        params.insert("learning_rate".to_string(), ParameterValue::Float(1e-4));
        params.insert("batch_size".to_string(), ParameterValue::Int(32));

        let args = params_to_training_args(&params);
        assert_eq!(args.learning_rate, 1e-4);
        assert_eq!(args.per_device_train_batch_size, 32);
    }
}
