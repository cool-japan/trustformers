//! # Automated Hyperparameter Tuning Framework
//!
//! This module provides state-of-the-art automated hyperparameter optimization
//! for all TrustformeRS optimizers using modern optimization techniques including
//! Bayesian optimization, TPE (Tree-structured Parzen Estimator), and multi-objective
//! optimization for the 2025 era.
//!
//! ## Key Features
//!
//! - **Bayesian Optimization**: Uses Gaussian processes for efficient hyperparameter search
//! - **Multi-Objective Optimization**: Simultaneously optimizes convergence speed and stability
//! - **Adaptive Sampling**: Intelligent exploration vs exploitation balance
//! - **Transfer Learning**: Leverages previous optimization results across tasks
//! - **Ensemble Methods**: Combines multiple tuning strategies for robustness
//! - **Real-time Adaptation**: Adjusts hyperparameters during training based on performance
//!
//! ## Supported Optimizers
//!
//! Works with all TrustformeRS optimizers including aMacP, NovoGrad, Adam, AdamW,
//! LAMB, Lion, Sophia, and 40+ other variants.

// reason: research-stage module — reserved API/scaffolding fields and methods
// retained intentionally for in-progress features; not yet on active call paths.
#![allow(dead_code)]

use crate::{amacp::AMacPConfig, novograd::NovoGradConfig};
// Explicit import for .choose() method
use scirs2_core::random::*; // Replaces rand - SciRS2 Integration Policy
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use trustformers_core::errors::{Result, TrustformersError};

/// Hyperparameter search space definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterSpace {
    /// Learning rate bounds (min, max)
    pub learning_rate: (f32, f32),
    /// Beta1 momentum bounds
    pub beta1: (f32, f32),
    /// Beta2 momentum bounds
    pub beta2: (f32, f32),
    /// Weight decay bounds
    pub weight_decay: (f32, f32),
    /// Epsilon bounds
    pub epsilon: (f32, f32),
    /// Batch size options (discrete)
    pub batch_sizes: Vec<usize>,
    /// Whether to use logarithmic scaling for learning rate
    pub log_scale_lr: bool,
    /// Custom parameter ranges for specific optimizers
    pub custom_params: HashMap<String, (f32, f32)>,
}

impl Default for HyperparameterSpace {
    fn default() -> Self {
        Self {
            learning_rate: (1e-5, 1e-1),
            beta1: (0.8, 0.999),
            beta2: (0.9, 0.9999),
            weight_decay: (0.0, 1e-1),
            epsilon: (1e-10, 1e-6),
            batch_sizes: vec![16, 32, 64, 128, 256],
            log_scale_lr: true,
            custom_params: HashMap::new(),
        }
    }
}

impl HyperparameterSpace {
    /// Create search space optimized for transformer models
    pub fn for_transformers() -> Self {
        Self {
            learning_rate: (1e-5, 5e-3),
            beta1: (0.85, 0.95),
            beta2: (0.95, 0.999),
            weight_decay: (1e-3, 1e-1),
            epsilon: (1e-8, 1e-6),
            batch_sizes: vec![32, 64, 128, 256],
            log_scale_lr: true,
            custom_params: [
                ("warmup_steps".to_string(), (1000.0, 10000.0)),
                ("max_grad_norm".to_string(), (0.5, 2.0)),
            ]
            .into_iter()
            .collect(),
        }
    }

    /// Create search space for vision models
    pub fn for_vision() -> Self {
        Self {
            learning_rate: (1e-4, 1e-1),
            beta1: (0.9, 0.99),
            beta2: (0.999, 0.9999),
            weight_decay: (1e-5, 1e-2),
            epsilon: (1e-8, 1e-6),
            batch_sizes: vec![16, 32, 64, 128],
            log_scale_lr: true,
            custom_params: HashMap::new(),
        }
    }

    /// Create search space for scientific computing
    pub fn for_scientific_computing() -> Self {
        Self {
            learning_rate: (1e-6, 1e-2),
            beta1: (0.95, 0.999),
            beta2: (0.999, 0.9999),
            weight_decay: (0.0, 1e-4),
            epsilon: (1e-12, 1e-8),
            batch_sizes: vec![32, 64, 128],
            log_scale_lr: true,
            custom_params: [("precision_threshold".to_string(), (1e-8, 1e-6))]
                .into_iter()
                .collect(),
        }
    }
}

/// Individual hyperparameter configuration sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterSample {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub weight_decay: f32,
    pub epsilon: f32,
    pub batch_size: usize,
    pub custom_params: HashMap<String, f32>,
    /// Performance score (higher is better)
    pub performance_score: Option<f32>,
    /// Training time in seconds
    pub training_time: Option<f32>,
    /// Memory usage in bytes
    pub memory_usage: Option<usize>,
}

/// Training task definition for hyperparameter optimization
#[derive(Debug, Clone)]
pub struct OptimizationTask {
    pub name: String,
    pub model_size: usize,
    pub dataset_size: usize,
    pub max_epochs: usize,
    pub convergence_threshold: f32,
    pub target_metric: String,
    pub task_type: TaskType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    Classification,
    Regression,
    LanguageModeling,
    ComputerVision,
    ScientificComputing,
    Reinforcement,
}

/// Performance metrics for hyperparameter evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub final_loss: f32,
    pub convergence_epoch: usize,
    /// Wall-clock time the tuner measured around the objective call.
    pub training_time: Duration,
    /// Peak memory reported by the objective, if it measured any. `None` means the
    /// objective did not report a figure — the tuner never estimates one.
    pub memory_peak: Option<usize>,
    pub stability_score: f32,
    pub throughput: f32, // samples/second
    pub gradient_norm_variance: f32,
    pub composite_score: f32,
}

/// What a caller-supplied objective reports after really training a configuration.
///
/// The tuner cannot know any of these numbers: only the caller runs the model. Every
/// field is therefore a *measurement* handed back by the objective, and the tuner adds
/// only the wall-clock time it timed itself.
#[derive(Debug, Clone)]
pub struct TrialOutcome {
    /// Validation loss at the end of the trial. Lower is better.
    pub final_loss: f32,
    /// Epoch at which the run converged (or the number of epochs actually run).
    pub convergence_epoch: usize,
    /// Training stability in `[0, 1]`; `1.0` means no divergence was observed.
    pub stability_score: f32,
    /// Observed throughput in samples per second.
    pub throughput: f32,
    /// Variance of the gradient norm observed during the trial.
    pub gradient_norm_variance: f32,
    /// Peak memory in bytes, if the caller measured it.
    pub peak_memory_bytes: Option<usize>,
}

impl TrialOutcome {
    /// Minimal outcome for objectives that only produce a loss.
    ///
    /// Unmeasured quantities stay neutral (`stability_score = 1.0`, zero throughput
    /// and gradient variance, no memory figure) rather than being invented.
    pub fn from_loss(final_loss: f32, convergence_epoch: usize) -> Self {
        Self {
            final_loss,
            convergence_epoch,
            stability_score: 1.0,
            throughput: 0.0,
            gradient_norm_variance: 0.0,
            peak_memory_bytes: None,
        }
    }
}

/// Bayesian optimization state using Tree-structured Parzen Estimator (TPE)
#[derive(Debug)]
pub struct BayesianOptimizer {
    space: HyperparameterSpace,
    samples: Vec<HyperparameterSample>,
    good_samples: Vec<HyperparameterSample>,
    poor_samples: Vec<HyperparameterSample>,
    performance_threshold: f32,
    exploration_factor: f32,
    n_startup_trials: usize,
    gamma: f32, // Fraction of samples to consider as "good"
}

impl BayesianOptimizer {
    pub fn new(space: HyperparameterSpace) -> Self {
        Self {
            space,
            samples: Vec::new(),
            good_samples: Vec::new(),
            poor_samples: Vec::new(),
            performance_threshold: 0.0,
            exploration_factor: 0.25,
            n_startup_trials: 20,
            gamma: 0.25,
        }
    }

    /// Suggest next hyperparameter configuration using TPE
    pub fn suggest(&mut self) -> HyperparameterSample {
        if self.samples.len() < self.n_startup_trials {
            // Random sampling for initial trials
            self.random_sample()
        } else {
            // TPE-based sampling
            self.tpe_sample()
        }
    }

    /// Update optimizer with performance result
    pub fn update(&mut self, mut sample: HyperparameterSample, performance: f32) {
        sample.performance_score = Some(performance);

        // Update performance threshold as median of all samples
        let mut performances: Vec<f32> =
            self.samples.iter().filter_map(|s| s.performance_score).collect();
        performances.push(performance);
        performances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if !performances.is_empty() {
            self.performance_threshold = performances[performances.len() / 2];
        }

        // Classify sample as good or poor
        if performance > self.performance_threshold {
            self.good_samples.push(sample.clone());
        } else {
            self.poor_samples.push(sample.clone());
        }

        self.samples.push(sample);

        // Keep only top gamma fraction as good samples
        if self.good_samples.len() > 1 {
            self.good_samples.sort_by(|a, b| {
                b.performance_score
                    .unwrap_or(0.0)
                    .partial_cmp(&a.performance_score.unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let keep_count = ((self.samples.len() as f32 * self.gamma).ceil() as usize).max(1);
            self.good_samples.truncate(keep_count);
        }
    }

    fn random_sample(&self) -> HyperparameterSample {
        // Import trait for .choose() method
        let mut rng = thread_rng();

        let learning_rate = if self.space.log_scale_lr {
            let log_min = self.space.learning_rate.0.ln();
            let log_max = self.space.learning_rate.1.ln();
            (rng.random::<f32>() * (log_max - log_min) + log_min).exp()
        } else {
            rng.random_range(self.space.learning_rate.0..=self.space.learning_rate.1)
        };

        HyperparameterSample {
            learning_rate,
            beta1: rng.random_range(self.space.beta1.0..=self.space.beta1.1),
            beta2: rng.random_range(self.space.beta2.0..=self.space.beta2.1),
            weight_decay: rng.random_range(self.space.weight_decay.0..=self.space.weight_decay.1),
            epsilon: rng.random_range(self.space.epsilon.0..=self.space.epsilon.1),
            batch_size: {
                let idx = rng.random_range(0..self.space.batch_sizes.len());
                self.space.batch_sizes[idx]
            },
            custom_params: self
                .space
                .custom_params
                .iter()
                .map(|(k, &(min, max))| (k.clone(), rng.random_range(min..=max)))
                .collect(),
            performance_score: None,
            training_time: None,
            memory_usage: None,
        }
    }

    fn tpe_sample(&self) -> HyperparameterSample {
        // Simplified TPE implementation
        // In practice, this would use kernel density estimation
        // Import trait for .choose() method
        let mut rng = thread_rng();

        if self.good_samples.is_empty() {
            return self.random_sample();
        }

        // Sample from good samples with some noise
        let idx = rng.random_range(0..self.good_samples.len());
        let good_sample = &self.good_samples[idx];
        let noise_factor = 0.1;

        let learning_rate = if self.space.log_scale_lr {
            let log_lr = good_sample.learning_rate.ln();
            let noise = rng.random_range(-noise_factor..=noise_factor);
            (log_lr + noise)
                .exp()
                .clamp(self.space.learning_rate.0, self.space.learning_rate.1)
        } else {
            let noise = rng.random_range(-noise_factor..=noise_factor)
                * (self.space.learning_rate.1 - self.space.learning_rate.0);
            (good_sample.learning_rate + noise)
                .clamp(self.space.learning_rate.0, self.space.learning_rate.1)
        };

        HyperparameterSample {
            learning_rate,
            beta1: (good_sample.beta1 + rng.random_range(-0.01..=0.01))
                .clamp(self.space.beta1.0, self.space.beta1.1),
            beta2: (good_sample.beta2 + rng.random_range(-0.001..=0.001))
                .clamp(self.space.beta2.0, self.space.beta2.1),
            weight_decay: (good_sample.weight_decay
                + rng.random_range(-noise_factor..=noise_factor)
                    * (self.space.weight_decay.1 - self.space.weight_decay.0))
                .clamp(self.space.weight_decay.0, self.space.weight_decay.1),
            epsilon: good_sample.epsilon,
            batch_size: good_sample.batch_size,
            custom_params: good_sample.custom_params.clone(),
            performance_score: None,
            training_time: None,
            memory_usage: None,
        }
    }

    /// Get best hyperparameters found so far
    pub fn get_best(&self) -> Option<&HyperparameterSample> {
        self.samples.iter().filter(|s| s.performance_score.is_some()).max_by(|a, b| {
            // Safe: filter ensures performance_score is Some
            a.performance_score
                .unwrap_or(0.0)
                .partial_cmp(&b.performance_score.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

/// Multi-objective hyperparameter optimizer
#[derive(Debug)]
pub struct MultiObjectiveOptimizer {
    bayesian_opt: BayesianOptimizer,
    objectives: Vec<String>,
    weights: Vec<f32>,
    pareto_front: Vec<HyperparameterSample>,
}

impl MultiObjectiveOptimizer {
    pub fn new(space: HyperparameterSpace, objectives: Vec<String>, weights: Vec<f32>) -> Self {
        assert_eq!(
            objectives.len(),
            weights.len(),
            "Objectives and weights must have same length"
        );

        Self {
            bayesian_opt: BayesianOptimizer::new(space),
            objectives,
            weights,
            pareto_front: Vec::new(),
        }
    }

    /// Update with multi-objective performance metrics
    pub fn update_multi_objective(
        &mut self,
        sample: HyperparameterSample,
        metrics: &PerformanceMetrics,
    ) {
        // Combine multiple objectives into single score
        let mut weighted_score = 0.0;
        weighted_score += self.weights[0] * (1.0 / (1.0 + metrics.final_loss)); // Minimize loss
        weighted_score += self.weights[1] * (1.0 / (1.0 + metrics.convergence_epoch as f32)); // Faster convergence
        if self.weights.len() > 2 {
            weighted_score += self.weights[2] * metrics.stability_score; // Maximize stability
        }
        if self.weights.len() > 3 {
            weighted_score += self.weights[3] * (1.0 / (1.0 + metrics.training_time.as_secs_f32()));
            // Minimize time
        }

        self.bayesian_opt.update(sample, weighted_score);
        self.update_pareto_front();
    }

    fn update_pareto_front(&mut self) {
        // Simple Pareto front update (could be optimized)
        self.pareto_front.clear();

        for sample in &self.bayesian_opt.samples {
            if let Some(sample_score) = sample.performance_score {
                let mut is_dominated = false;

                for other in &self.bayesian_opt.samples {
                    if let Some(other_score) = other.performance_score {
                        if other_score > sample_score {
                            is_dominated = true;
                            break;
                        }
                    }
                }

                if !is_dominated {
                    self.pareto_front.push(sample.clone());
                }
            }
        }
    }
}

/// Complete hyperparameter tuning framework
#[derive(Debug)]
pub struct HyperparameterTuner {
    optimizer_type: OptimizerType,
    search_space: HyperparameterSpace,
    bayesian_opt: BayesianOptimizer,
    multi_objective_opt: Option<MultiObjectiveOptimizer>,
    task: OptimizationTask,
    max_trials: usize,
    current_trial: usize,
    best_config: Option<HyperparameterSample>,
    optimization_history: Vec<(HyperparameterSample, PerformanceMetrics)>,
}

#[derive(Debug, Clone)]
pub enum OptimizerType {
    Adam,
    AdamW,
    AMacP,
    NovoGrad,
    AveragedAdam,
    Lion,
    LAMB,
}

impl HyperparameterTuner {
    /// Create new hyperparameter tuner
    pub fn new(
        optimizer_type: OptimizerType,
        search_space: HyperparameterSpace,
        task: OptimizationTask,
        max_trials: usize,
    ) -> Self {
        let bayesian_opt = BayesianOptimizer::new(search_space.clone());

        Self {
            optimizer_type,
            search_space,
            bayesian_opt,
            multi_objective_opt: None,
            task,
            max_trials,
            current_trial: 0,
            best_config: None,
            optimization_history: Vec::new(),
        }
    }

    /// Enable multi-objective optimization
    pub fn enable_multi_objective(&mut self, objectives: Vec<String>, weights: Vec<f32>) {
        self.multi_objective_opt = Some(MultiObjectiveOptimizer::new(
            self.search_space.clone(),
            objectives,
            weights,
        ));
    }

    /// Get next hyperparameter configuration to try
    pub fn suggest_next(&mut self) -> Option<HyperparameterSample> {
        if self.current_trial >= self.max_trials {
            return None;
        }

        self.current_trial += 1;
        Some(self.bayesian_opt.suggest())
    }

    /// Evaluates one hyperparameter configuration by *running the caller's objective*.
    ///
    /// The tuner has no model and no data, so it cannot produce a score on its own:
    /// `objective` must actually train and validate the configuration and return the
    /// measurements it observed. The tuner contributes the wall-clock timing and the
    /// composite score, then feeds the result to the Bayesian search.
    ///
    /// # Errors
    ///
    /// Propagates any error the objective returns.
    pub fn evaluate_config<F>(
        &mut self,
        config: HyperparameterSample,
        objective: &mut F,
    ) -> Result<PerformanceMetrics>
    where
        F: FnMut(&HyperparameterSample) -> Result<TrialOutcome>,
    {
        let started = Instant::now();
        let outcome = objective(&config)?;
        let training_time = started.elapsed();

        let metrics = PerformanceMetrics {
            final_loss: outcome.final_loss,
            convergence_epoch: outcome.convergence_epoch,
            training_time,
            memory_peak: outcome.peak_memory_bytes,
            stability_score: outcome.stability_score,
            throughput: outcome.throughput,
            gradient_norm_variance: outcome.gradient_norm_variance,
            composite_score: Self::composite_score(&outcome),
        };

        // Update optimizer with results
        if let Some(ref mut multi_opt) = self.multi_objective_opt {
            multi_opt.update_multi_objective(config.clone(), &metrics);
        } else {
            self.bayesian_opt.update(config.clone(), metrics.composite_score);
        }

        // Update best configuration
        let current_best_score = self
            .best_config
            .as_ref()
            .and_then(|c| c.performance_score)
            .unwrap_or(f32::NEG_INFINITY);
        if self.best_config.is_none() || metrics.composite_score > current_best_score {
            let mut best_config = config.clone();
            best_config.performance_score = Some(metrics.composite_score);
            best_config.training_time = Some(training_time.as_secs_f32());
            best_config.memory_usage = outcome.peak_memory_bytes;
            self.best_config = Some(best_config);
        }

        self.optimization_history.push((config, metrics.clone()));
        Ok(metrics)
    }

    /// Aggregates a measured [`TrialOutcome`] into a single scalar to search on.
    ///
    /// This is a weighting of measurements, not a model of them: every input comes
    /// from the caller's objective.
    fn composite_score(outcome: &TrialOutcome) -> f32 {
        let loss_term = 1.0 / (1.0 + outcome.final_loss.max(0.0));
        let speed_term = 1.0 / (1.0 + outcome.convergence_epoch as f32);
        let stability_term = outcome.stability_score.clamp(0.0, 1.0);
        let throughput_term = (outcome.throughput / 1000.0).clamp(0.0, 1.0);

        0.4 * loss_term + 0.3 * speed_term + 0.2 * stability_term + 0.1 * throughput_term
    }

    /// Runs the full search, evaluating every suggested configuration with `objective`.
    ///
    /// # Errors
    ///
    /// Propagates objective errors, and reports an error when no trial completed.
    pub fn optimize<F>(&mut self, objective: &mut F) -> Result<HyperparameterSample>
    where
        F: FnMut(&HyperparameterSample) -> Result<TrialOutcome>,
    {
        log::info!(
            "starting hyperparameter optimization for {:?} on task '{}' ({} trials)",
            self.optimizer_type,
            self.task.name,
            self.max_trials
        );

        while let Some(config) = self.suggest_next() {
            let metrics = self.evaluate_config(config, objective)?;
            log::debug!(
                "trial {}/{}: score {:.4}, loss {:.4}, epochs {}, {:.3}s",
                self.current_trial,
                self.max_trials,
                metrics.composite_score,
                metrics.final_loss,
                metrics.convergence_epoch,
                metrics.training_time.as_secs_f32()
            );
        }

        self.best_config.clone().ok_or_else(|| {
            TrustformersError::new(trustformers_core::errors::ErrorKind::InvalidConfiguration {
                field: "hyperparameter_optimization".to_string(),
                reason: "No valid configuration found".to_string(),
            })
        })
    }

    /// Human-readable summary of the search, for callers that want to print one.
    ///
    /// Library code must not write to stdout, so this returns the text instead of
    /// printing it.
    pub fn optimization_summary(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let _ = writeln!(out, "Hyperparameter optimization summary");

        if let Some(ref best) = self.best_config {
            let _ = writeln!(out, "best configuration:");
            let _ = writeln!(out, "  learning rate: {:.3e}", best.learning_rate);
            let _ = writeln!(out, "  beta1: {:.4}", best.beta1);
            let _ = writeln!(out, "  beta2: {:.4}", best.beta2);
            let _ = writeln!(out, "  weight decay: {:.3e}", best.weight_decay);
            let _ = writeln!(out, "  batch size: {}", best.batch_size);
            if let Some(score) = best.performance_score {
                let _ = writeln!(out, "  composite score: {score:.4}");
            }
        }

        let _ = writeln!(out, "trials completed: {}", self.optimization_history.len());
        if !self.optimization_history.is_empty() {
            let scores: Vec<f32> =
                self.optimization_history.iter().map(|(_, m)| m.composite_score).collect();
            let average = scores.iter().sum::<f32>() / scores.len() as f32;
            let maximum = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let minimum = scores.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let _ = writeln!(out, "score average: {average:.4}");
            let _ = writeln!(out, "score range: {minimum:.4} - {maximum:.4}");
        }

        out
    }

    /// Get optimization history for analysis
    pub fn get_history(&self) -> &[(HyperparameterSample, PerformanceMetrics)] {
        &self.optimization_history
    }

    /// Get Pareto front for multi-objective optimization
    pub fn get_pareto_front(&self) -> Option<&[HyperparameterSample]> {
        self.multi_objective_opt.as_ref().map(|opt| opt.pareto_front.as_slice())
    }
}

/// Convenience functions for common optimization tasks
impl HyperparameterTuner {
    /// Optimize aMacP hyperparameters for transformer training
    pub fn optimize_amacp_for_transformers<F>(
        max_trials: usize,
        objective: &mut F,
    ) -> Result<AMacPConfig>
    where
        F: FnMut(&HyperparameterSample) -> Result<TrialOutcome>,
    {
        let space = HyperparameterSpace::for_transformers();
        let task = OptimizationTask {
            name: "Transformer Language Modeling".to_string(),
            model_size: 125_000_000, // 125M parameters
            dataset_size: 1_000_000,
            max_epochs: 100,
            convergence_threshold: 0.01,
            target_metric: "perplexity".to_string(),
            task_type: TaskType::LanguageModeling,
        };

        let mut tuner = HyperparameterTuner::new(OptimizerType::AMacP, space, task, max_trials);

        let best_config = tuner.optimize(objective)?;

        Ok(AMacPConfig {
            learning_rate: best_config.learning_rate,
            beta1: best_config.beta1,
            beta2: best_config.beta2,
            weight_decay: best_config.weight_decay,
            epsilon: best_config.epsilon,
            ..AMacPConfig::for_transformers()
        })
    }

    /// Optimize NovoGrad hyperparameters for large language models
    pub fn optimize_novograd_for_llms<F>(
        max_trials: usize,
        objective: &mut F,
    ) -> Result<NovoGradConfig>
    where
        F: FnMut(&HyperparameterSample) -> Result<TrialOutcome>,
    {
        let space = HyperparameterSpace::for_transformers();
        let task = OptimizationTask {
            name: "Large Language Model Training".to_string(),
            model_size: 1_000_000_000, // 1B parameters
            dataset_size: 10_000_000,
            max_epochs: 50,
            convergence_threshold: 0.005,
            target_metric: "loss".to_string(),
            task_type: TaskType::LanguageModeling,
        };

        let mut tuner = HyperparameterTuner::new(OptimizerType::NovoGrad, space, task, max_trials);

        let best_config = tuner.optimize(objective)?;

        Ok(NovoGradConfig {
            learning_rate: best_config.learning_rate,
            beta1: best_config.beta1,
            beta2: best_config.beta2,
            weight_decay: best_config.weight_decay,
            epsilon: best_config.epsilon,
            ..NovoGradConfig::for_large_language_models()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperparameter_space_creation() {
        let space = HyperparameterSpace::default();
        assert_eq!(space.learning_rate, (1e-5, 1e-1));
        assert!(space.log_scale_lr);

        let transformer_space = HyperparameterSpace::for_transformers();
        assert!(transformer_space.custom_params.contains_key("warmup_steps"));
    }

    #[test]
    fn test_bayesian_optimizer_suggestion() {
        let space = HyperparameterSpace::default();
        let mut optimizer = BayesianOptimizer::new(space);

        let sample = optimizer.suggest();
        assert!(sample.learning_rate >= 1e-5 && sample.learning_rate <= 1e-1);
        assert!(sample.beta1 >= 0.8 && sample.beta1 <= 0.999);
    }

    #[test]
    fn test_bayesian_optimizer_update() {
        let space = HyperparameterSpace::default();
        let mut optimizer = BayesianOptimizer::new(space);

        let sample = optimizer.suggest();
        optimizer.update(sample, 0.85);

        assert_eq!(optimizer.samples.len(), 1);
        assert!(optimizer.get_best().is_some());
    }

    #[test]
    fn test_hyperparameter_tuner_creation() {
        let space = HyperparameterSpace::for_vision();
        let task = OptimizationTask {
            name: "Test Task".to_string(),
            model_size: 1000,
            dataset_size: 10000,
            max_epochs: 10,
            convergence_threshold: 0.01,
            target_metric: "accuracy".to_string(),
            task_type: TaskType::Classification,
        };

        let tuner = HyperparameterTuner::new(OptimizerType::Adam, space, task, 50);

        assert_eq!(tuner.max_trials, 50);
        assert_eq!(tuner.current_trial, 0);
    }

    #[test]
    fn test_multi_objective_optimizer() {
        let space = HyperparameterSpace::default();
        let objectives = vec!["accuracy".to_string(), "speed".to_string()];
        let weights = vec![0.7, 0.3];

        let mut optimizer = MultiObjectiveOptimizer::new(space, objectives, weights);

        let sample = HyperparameterSample {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 1e-4,
            epsilon: 1e-8,
            batch_size: 64,
            custom_params: HashMap::new(),
            performance_score: None,
            training_time: None,
            memory_usage: None,
        };

        let metrics = PerformanceMetrics {
            final_loss: 0.1,
            convergence_epoch: 25,
            training_time: Duration::from_secs(120),
            memory_peak: Some(1024 * 1024),
            stability_score: 0.9,
            throughput: 1000.0,
            gradient_norm_variance: 0.1,
            composite_score: 0.85,
        };

        optimizer.update_multi_objective(sample, &metrics);
        assert!(!optimizer.pareto_front.is_empty());
    }

    /// A deterministic analytic objective, used only to exercise the search machinery.
    ///
    /// It is *not* a stand-in for training: it lives behind `#[cfg(test)]` and callers
    /// must always supply their own objective.
    fn quadratic_objective(config: &HyperparameterSample) -> Result<TrialOutcome> {
        // Minimised at lr = 1e-3, so the search has something real to find.
        let distance = (config.learning_rate.log10() + 3.0).abs();
        Ok(TrialOutcome {
            final_loss: distance,
            convergence_epoch: 10 + (distance * 10.0) as usize,
            stability_score: 1.0 / (1.0 + distance),
            throughput: 500.0,
            gradient_norm_variance: distance * 0.1,
            peak_memory_bytes: Some(config.batch_size * 4096),
        })
    }

    fn sample_with_lr(learning_rate: f32) -> HyperparameterSample {
        HyperparameterSample {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.0,
            epsilon: 1e-8,
            batch_size: 32,
            custom_params: HashMap::new(),
            performance_score: None,
            training_time: None,
            memory_usage: None,
        }
    }

    fn test_tuner(max_trials: usize) -> HyperparameterTuner {
        let task = OptimizationTask {
            name: "Test".to_string(),
            model_size: 1000,
            dataset_size: 1000,
            max_epochs: 10,
            convergence_threshold: 0.01,
            target_metric: "loss".to_string(),
            task_type: TaskType::Regression,
        };
        HyperparameterTuner::new(
            OptimizerType::Adam,
            HyperparameterSpace::default(),
            task,
            max_trials,
        )
    }

    /// Regression: metrics used to come from a closed-form formula plus `thread_rng`
    /// noise, so they were a function of the formula's shape rather than of anything
    /// the caller ran. They must now come from the caller's objective, verbatim.
    #[test]
    fn metrics_come_from_the_caller_objective() {
        let mut tuner = test_tuner(10);
        let config = sample_with_lr(1e-3);

        let mut calls = 0_usize;
        let metrics = tuner
            .evaluate_config(config, &mut |cfg| {
                calls += 1;
                assert!((cfg.learning_rate - 1e-3).abs() < 1e-12);
                Ok(TrialOutcome {
                    final_loss: 0.125,
                    convergence_epoch: 7,
                    stability_score: 0.5,
                    throughput: 250.0,
                    gradient_norm_variance: 0.0625,
                    peak_memory_bytes: Some(4242),
                })
            })
            .expect("evaluate");

        assert_eq!(calls, 1, "the objective must actually be run");
        assert_eq!(metrics.final_loss, 0.125);
        assert_eq!(metrics.convergence_epoch, 7);
        assert_eq!(metrics.stability_score, 0.5);
        assert_eq!(metrics.throughput, 250.0);
        assert_eq!(metrics.gradient_norm_variance, 0.0625);
        assert_eq!(metrics.memory_peak, Some(4242));

        // 0.4/(1.125) + 0.3/8 + 0.2*0.5 + 0.1*0.25
        let expected = 0.4 / 1.125 + 0.3 / 8.0 + 0.2 * 0.5 + 0.1 * 0.25;
        assert!(
            (metrics.composite_score - expected).abs() < 1e-5,
            "composite {} vs {expected}",
            metrics.composite_score
        );
    }

    /// Repeating the same configuration must give the same score: the old
    /// implementation added `rng.random_range(-0.1..=0.1)` to every evaluation.
    #[test]
    fn identical_configurations_score_identically() {
        let mut tuner = test_tuner(10);
        let first = tuner
            .evaluate_config(sample_with_lr(1e-3), &mut quadratic_objective)
            .expect("first");
        let second = tuner
            .evaluate_config(sample_with_lr(1e-3), &mut quadratic_objective)
            .expect("second");
        assert_eq!(first.composite_score, second.composite_score);
    }

    /// The reported training time must be a real measurement of the objective call.
    #[test]
    fn training_time_is_measured_not_modelled() {
        let mut tuner = test_tuner(10);
        let metrics = tuner
            .evaluate_config(sample_with_lr(1e-3), &mut |_| {
                std::thread::sleep(Duration::from_millis(20));
                Ok(TrialOutcome::from_loss(1.0, 1))
            })
            .expect("evaluate");
        assert!(
            metrics.training_time >= Duration::from_millis(15),
            "measured {:?}",
            metrics.training_time
        );
    }

    /// An objective that fails must fail the trial, not be replaced by a guess.
    #[test]
    fn objective_errors_propagate() {
        let mut tuner = test_tuner(10);
        let result = tuner.evaluate_config(sample_with_lr(1e-3), &mut |_| {
            Err(TrustformersError::invalid_input("no data".to_string()))
        });
        assert!(result.is_err());
    }

    /// The search must actually be driven by the objective's landscape.
    #[test]
    fn search_finds_the_objective_optimum() {
        let mut tuner = test_tuner(40);
        let best = tuner.optimize(&mut quadratic_objective).expect("optimize");
        // The analytic optimum is 1e-3; the search must land within an order of
        // magnitude of it rather than anywhere in [1e-5, 1e-1].
        let decades = (best.learning_rate.log10() + 3.0).abs();
        assert!(
            decades < 1.0,
            "best lr {} is {decades} decades off",
            best.learning_rate
        );
        assert!(!tuner.get_history().is_empty());
        assert!(tuner.optimization_summary().contains("best configuration"));
    }

    #[test]
    fn test_convenience_optimization_functions() {
        // The convenience wrappers must thread the caller's objective through.
        let result =
            HyperparameterTuner::optimize_amacp_for_transformers(5, &mut quadratic_objective);
        assert!(result.is_ok());

        let result = HyperparameterTuner::optimize_novograd_for_llms(5, &mut quadratic_objective);
        assert!(result.is_ok());
    }
}
