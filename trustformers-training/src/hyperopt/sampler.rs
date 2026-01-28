//! Sampling strategies for hyperparameter optimization

use super::search_space::HyperParameter;
use super::{Direction, ParameterValue, SearchSpace, Trial, TrialHistory};
use scirs2_core::random::*; // SciRS2 Integration Policy (was: use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for samplers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplerConfig {
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Number of random trials before using more sophisticated strategies
    pub n_startup_trials: usize,
    /// Number of random trials between each optimization step
    pub n_ei_candidates: usize,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            seed: None,
            n_startup_trials: 10,
            n_ei_candidates: 24,
        }
    }
}

/// Trait for hyperparameter samplers
pub trait Sampler: Send + Sync {
    /// Sample a new configuration given the search space and trial history
    fn sample(
        &mut self,
        search_space: &SearchSpace,
        trial_history: &TrialHistory,
    ) -> HashMap<String, ParameterValue>;

    /// Update the sampler with information from a completed trial
    fn update(&mut self, _trial: &Trial) {}

    /// Get the name of this sampler
    fn name(&self) -> &str;
}

/// Simple random sampler
pub struct RandomSampler {
    rng: StdRng,
    name: String,
}

impl RandomSampler {
    /// Create a new random sampler
    pub fn new() -> Self {
        Self {
            rng: StdRng::seed_from_u64(thread_rng().random()),
            name: "RandomSampler".to_string(),
        }
    }

    /// Create a random sampler with a specific seed
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            name: format!("RandomSampler(seed={})", seed),
        }
    }
}

impl Sampler for RandomSampler {
    fn sample(
        &mut self,
        search_space: &SearchSpace,
        _trial_history: &TrialHistory,
    ) -> HashMap<String, ParameterValue> {
        search_space.sample(&mut self.rng).unwrap_or_else(|e| {
            log::warn!(
                "Failed to sample from search space: {}. Using empty configuration.",
                e
            );
            HashMap::new()
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl Default for RandomSampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Tree-structured Parzen Estimator (TPE) sampler
/// This is a simplified version of the TPE algorithm used in Optuna
pub struct TPESampler {
    config: SamplerConfig,
    rng: StdRng,
    name: String,
    /// Parameters from good trials (top percentile)
    good_trials: Vec<HashMap<String, ParameterValue>>,
    /// Parameters from bad trials (bottom percentile)
    bad_trials: Vec<HashMap<String, ParameterValue>>,
    /// Percentile threshold for splitting good/bad trials
    percentile: f64,
}

impl TPESampler {
    /// Create a new TPE sampler
    pub fn new() -> Self {
        Self::with_config(SamplerConfig::default())
    }

    /// Create a TPE sampler with configuration
    pub fn with_config(config: SamplerConfig) -> Self {
        let rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().random())
        };

        let name = format!(
            "TPESampler(startup={}, candidates={})",
            config.n_startup_trials, config.n_ei_candidates
        );

        Self {
            config,
            rng,
            name,
            good_trials: Vec::new(),
            bad_trials: Vec::new(),
            percentile: 0.1, // Top 10% are "good"
        }
    }
}

impl Sampler for TPESampler {
    fn sample(
        &mut self,
        search_space: &SearchSpace,
        trial_history: &TrialHistory,
    ) -> HashMap<String, ParameterValue> {
        let completed_trials = trial_history.completed_trials();

        // Use random sampling for initial trials
        if completed_trials.len() < self.config.n_startup_trials {
            return search_space.sample(&mut self.rng).unwrap_or_else(|e| {
                log::warn!(
                    "Failed to sample from search space: {}. Using empty configuration.",
                    e
                );
                HashMap::new()
            });
        }

        // Split trials into good and bad based on objective values
        self.update_trial_groups(&completed_trials, trial_history.direction.clone());

        // If we don't have enough trials for TPE, fall back to random
        if self.good_trials.is_empty() {
            return search_space.sample(&mut self.rng).unwrap_or_else(|e| {
                log::warn!(
                    "Failed to sample from search space: {}. Using empty configuration.",
                    e
                );
                HashMap::new()
            });
        }

        // Generate candidates and select the best one based on Expected Improvement
        let mut best_candidate = None;
        let mut best_score = f64::NEG_INFINITY;

        for _ in 0..self.config.n_ei_candidates {
            let candidate = self.sample_from_good_trials(search_space);
            let score = self.compute_expected_improvement(&candidate, search_space);

            if score > best_score {
                best_score = score;
                best_candidate = Some(candidate);
            }
        }

        best_candidate.unwrap_or_else(|| {
            search_space.sample(&mut self.rng).unwrap_or_else(|e| {
                log::warn!(
                    "Failed to sample from search space: {}. Using empty configuration.",
                    e
                );
                HashMap::new()
            })
        })
    }

    fn update(&mut self, _trial: &Trial) {
        // TPE updates happen in the sample method when we have trial history
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl TPESampler {
    fn update_trial_groups(&mut self, trials: &[&Trial], direction: Direction) {
        if trials.is_empty() {
            return;
        }

        // Sort trials by objective value
        let mut sorted_trials = trials.to_vec();
        sorted_trials.sort_by(|a, b| {
            let a_val = a.objective_value().unwrap_or(f64::NEG_INFINITY);
            let b_val = b.objective_value().unwrap_or(f64::NEG_INFINITY);

            match direction {
                Direction::Maximize => {
                    b_val.partial_cmp(&a_val).unwrap_or(std::cmp::Ordering::Equal)
                },
                Direction::Minimize => {
                    a_val.partial_cmp(&b_val).unwrap_or(std::cmp::Ordering::Equal)
                },
            }
        });

        // Split into good and bad trials
        let split_idx = ((trials.len() as f64 * self.percentile).ceil() as usize).max(1);

        self.good_trials = sorted_trials[..split_idx].iter().map(|t| t.params.clone()).collect();

        self.bad_trials = sorted_trials[split_idx..].iter().map(|t| t.params.clone()).collect();
    }

    fn sample_from_good_trials(
        &mut self,
        search_space: &SearchSpace,
    ) -> HashMap<String, ParameterValue> {
        if self.good_trials.is_empty() {
            return search_space.sample(&mut self.rng).unwrap_or_else(|e| {
                log::warn!(
                    "Failed to sample from search space: {}. Using empty configuration.",
                    e
                );
                HashMap::new()
            });
        }

        // Sample a random good trial and add noise
        let base_trial_index = self.rng.gen_range(0..self.good_trials.len());
        let base_trial = self.good_trials[base_trial_index].clone();
        let mut result = HashMap::new();

        for param in &search_space.parameters {
            let param_name = param.name();

            if let Some(base_value) = base_trial.get(param_name) {
                // Add noise to the base value
                let new_value = self.add_noise_to_parameter(param, base_value);
                result.insert(param_name.to_string(), new_value);
            } else {
                // If parameter not in base trial, sample randomly
                result.insert(
                    param_name.to_string(),
                    param.sample(&mut self.rng).unwrap_or_else(|e| {
                        log::warn!(
                            "Failed to sample parameter '{}': {}. Using default value.",
                            param.name(),
                            e
                        );
                        // Provide a sensible default based on parameter type
                        match param {
                            HyperParameter::Categorical(_) => {
                                ParameterValue::String("default".to_string())
                            },
                            HyperParameter::Continuous(_) => ParameterValue::Float(0.0),
                            HyperParameter::Discrete(_) => ParameterValue::Int(0),
                            HyperParameter::Log(_) => ParameterValue::Float(1e-3),
                        }
                    }),
                );
            }
        }

        result
    }

    fn add_noise_to_parameter(
        &mut self,
        param: &super::search_space::HyperParameter,
        base_value: &ParameterValue,
    ) -> ParameterValue {
        use super::search_space::HyperParameter;

        match param {
            HyperParameter::Categorical(_) => {
                // For categorical, sample randomly from the parameter space
                param.sample(&mut self.rng).unwrap_or_else(|e| {
                    log::warn!(
                        "Failed to sample parameter '{}': {}. Using default value.",
                        param.name(),
                        e
                    );
                    // Provide a sensible default based on parameter type
                    match param {
                        HyperParameter::Categorical(_) => {
                            ParameterValue::String("default".to_string())
                        },
                        HyperParameter::Continuous(_) => ParameterValue::Float(0.0),
                        HyperParameter::Discrete(_) => ParameterValue::Int(0),
                        HyperParameter::Log(_) => ParameterValue::Float(1e-3),
                    }
                })
            },
            HyperParameter::Continuous(p) => {
                if let Some(base_float) = base_value.as_float() {
                    // Add Gaussian noise
                    let noise_std = (p.high - p.low) * 0.1; // 10% of range
                                                            // Distribution and Normal already available via scirs2_core::random::*
                    let normal = Normal::new(0.0, noise_std)
                        .unwrap_or_else(|_| Normal::new(0.0, 1.0).unwrap());
                    let noisy_value = base_float + normal.sample(&mut self.rng);
                    let clamped_value = noisy_value.clamp(p.low, p.high);
                    ParameterValue::Float(clamped_value)
                } else {
                    param.sample(&mut self.rng).unwrap_or_else(|e| {
                        log::warn!(
                            "Failed to sample parameter '{}': {}. Using default value.",
                            param.name(),
                            e
                        );
                        // Provide a sensible default based on parameter type
                        match param {
                            HyperParameter::Categorical(_) => {
                                ParameterValue::String("default".to_string())
                            },
                            HyperParameter::Continuous(_) => ParameterValue::Float(0.0),
                            HyperParameter::Discrete(_) => ParameterValue::Int(0),
                            HyperParameter::Log(_) => ParameterValue::Float(1e-3),
                        }
                    })
                }
            },
            HyperParameter::Discrete(p) => {
                if let Some(base_int) = base_value.as_int() {
                    // Add discrete noise
                    let noise_range = ((p.high - p.low) / 10).max(p.step); // 10% of range
                    let noise = self.rng.gen_range(-noise_range..=noise_range);
                    let noisy_value = base_int + noise;
                    let clamped_value = noisy_value.clamp(p.low, p.high);
                    // Round to nearest valid step
                    let stepped_value = p.low + ((clamped_value - p.low) / p.step) * p.step;
                    ParameterValue::Int(stepped_value)
                } else {
                    param.sample(&mut self.rng).unwrap_or_else(|e| {
                        log::warn!(
                            "Failed to sample parameter '{}': {}. Using default value.",
                            param.name(),
                            e
                        );
                        // Provide a sensible default based on parameter type
                        match param {
                            HyperParameter::Categorical(_) => {
                                ParameterValue::String("default".to_string())
                            },
                            HyperParameter::Continuous(_) => ParameterValue::Float(0.0),
                            HyperParameter::Discrete(_) => ParameterValue::Int(0),
                            HyperParameter::Log(_) => ParameterValue::Float(1e-3),
                        }
                    })
                }
            },
            HyperParameter::Log(p) => {
                if let Some(base_float) = base_value.as_float() {
                    // Add noise in log space
                    let log_base = base_float.log(p.base);
                    let log_low = p.low.log(p.base);
                    let log_high = p.high.log(p.base);
                    let noise_std = (log_high - log_low) * 0.1;

                    // Distribution and Normal already available via scirs2_core::random::*
                    let normal = Normal::new(0.0, noise_std)
                        .unwrap_or_else(|_| Normal::new(0.0, 1.0).unwrap());
                    let noisy_log = log_base + normal.sample(&mut self.rng);
                    let clamped_log = noisy_log.clamp(log_low, log_high);
                    let new_value = p.base.powf(clamped_log);
                    ParameterValue::Float(new_value)
                } else {
                    param.sample(&mut self.rng).unwrap_or_else(|e| {
                        log::warn!(
                            "Failed to sample parameter '{}': {}. Using default value.",
                            param.name(),
                            e
                        );
                        // Provide a sensible default based on parameter type
                        match param {
                            HyperParameter::Categorical(_) => {
                                ParameterValue::String("default".to_string())
                            },
                            HyperParameter::Continuous(_) => ParameterValue::Float(0.0),
                            HyperParameter::Discrete(_) => ParameterValue::Int(0),
                            HyperParameter::Log(_) => ParameterValue::Float(1e-3),
                        }
                    })
                }
            },
        }
    }

    fn compute_expected_improvement(
        &mut self,
        candidate: &HashMap<String, ParameterValue>,
        _search_space: &SearchSpace,
    ) -> f64 {
        // Simplified EI calculation
        // In a full implementation, this would use kernel density estimation
        // to model the probability distributions of good vs bad trials

        if self.good_trials.is_empty() || self.bad_trials.is_empty() {
            return self.rng.random::<f64>(); // Random score if no historical data
        }

        // Simple scoring: count how similar candidate is to good vs bad trials
        let good_similarity = self.compute_similarity(candidate, &self.good_trials);
        let bad_similarity = self.compute_similarity(candidate, &self.bad_trials);

        // Higher score for more similarity to good trials, less to bad trials
        good_similarity - bad_similarity
    }

    fn compute_similarity(
        &self,
        candidate: &HashMap<String, ParameterValue>,
        trials: &[HashMap<String, ParameterValue>],
    ) -> f64 {
        if trials.is_empty() {
            return 0.0;
        }

        let total_similarity: f64 =
            trials.iter().map(|trial| self.parameter_similarity(candidate, trial)).sum();

        total_similarity / trials.len() as f64
    }

    fn parameter_similarity(
        &self,
        a: &HashMap<String, ParameterValue>,
        b: &HashMap<String, ParameterValue>,
    ) -> f64 {
        let mut total_similarity = 0.0;
        let mut count = 0;

        for (name, value_a) in a {
            if let Some(value_b) = b.get(name) {
                let similarity = match (value_a, value_b) {
                    (ParameterValue::Float(a), ParameterValue::Float(b)) => {
                        let diff = (a - b).abs();
                        1.0 / (1.0 + diff) // Exponential decay similarity
                    },
                    (ParameterValue::Int(a), ParameterValue::Int(b)) => {
                        let diff = (a - b).abs() as f64;
                        1.0 / (1.0 + diff)
                    },
                    (ParameterValue::String(a), ParameterValue::String(b)) => {
                        if a == b {
                            1.0
                        } else {
                            0.0
                        }
                    },
                    (ParameterValue::Bool(a), ParameterValue::Bool(b)) => {
                        if a == b {
                            1.0
                        } else {
                            0.0
                        }
                    },
                    _ => 0.0, // Type mismatch
                };

                total_similarity += similarity;
                count += 1;
            }
        }

        if count > 0 {
            total_similarity / count as f64
        } else {
            0.0
        }
    }
}

impl Default for TPESampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Gaussian Process (GP) sampler for Bayesian optimization
/// This is a simplified implementation
pub struct GPSampler {
    config: SamplerConfig,
    rng: StdRng,
    name: String,
    trials: Vec<(HashMap<String, ParameterValue>, f64)>, // (params, objective)
}

impl GPSampler {
    /// Create a new GP sampler
    pub fn new() -> Self {
        Self::with_config(SamplerConfig::default())
    }

    /// Create a GP sampler with configuration
    pub fn with_config(config: SamplerConfig) -> Self {
        let rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().random())
        };

        let name = format!("GPSampler(startup={})", config.n_startup_trials);

        Self {
            config,
            rng,
            name,
            trials: Vec::new(),
        }
    }
}

impl Sampler for GPSampler {
    fn sample(
        &mut self,
        search_space: &SearchSpace,
        trial_history: &TrialHistory,
    ) -> HashMap<String, ParameterValue> {
        let completed_trials = trial_history.completed_trials();

        // Use random sampling for initial trials
        if completed_trials.len() < self.config.n_startup_trials {
            return search_space.sample(&mut self.rng).unwrap_or_else(|e| {
                log::warn!(
                    "Failed to sample from search space: {}. Using empty configuration.",
                    e
                );
                HashMap::new()
            });
        }

        // Update internal state with completed trials
        self.trials.clear();
        for trial in completed_trials {
            if let Some(objective) = trial.objective_value() {
                self.trials.push((trial.params.clone(), objective));
            }
        }

        // For now, use a simplified acquisition function
        // In a full implementation, this would fit a GP and use acquisition functions
        // like Expected Improvement or Upper Confidence Bound

        let mut best_candidate = None;
        let mut best_score = f64::NEG_INFINITY;

        // Generate candidates and score them
        for _ in 0..self.config.n_ei_candidates {
            let candidate = match search_space.sample(&mut self.rng) {
                Ok(c) => c,
                Err(e) => {
                    log::warn!("Failed to sample candidate: {}. Skipping.", e);
                    continue;
                },
            };
            let score = self.acquisition_function(&candidate);

            if score > best_score {
                best_score = score;
                best_candidate = Some(candidate);
            }
        }

        best_candidate.unwrap_or_else(|| {
            search_space.sample(&mut self.rng).unwrap_or_else(|e| {
                log::warn!(
                    "Failed to sample from search space: {}. Using empty configuration.",
                    e
                );
                HashMap::new()
            })
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl GPSampler {
    fn acquisition_function(&mut self, candidate: &HashMap<String, ParameterValue>) -> f64 {
        // Simplified acquisition function
        // Real GP would compute mean and variance from the posterior

        if self.trials.is_empty() {
            return self.rng.random::<f64>();
        }

        // Find the most similar historical trial
        let mut best_similarity = 0.0;
        let mut corresponding_objective = 0.0;

        for (trial_params, objective) in &self.trials {
            let similarity = self.compute_similarity(candidate, trial_params);
            if similarity > best_similarity {
                best_similarity = similarity;
                corresponding_objective = *objective;
            }
        }

        // Add exploration bonus (uncertainty)
        let exploration = 1.0 - best_similarity; // Higher for dissimilar candidates
        let exploitation = corresponding_objective;

        exploitation + 0.1 * exploration // Balance exploration vs exploitation
    }

    fn compute_similarity(
        &self,
        a: &HashMap<String, ParameterValue>,
        b: &HashMap<String, ParameterValue>,
    ) -> f64 {
        let mut total_similarity = 0.0;
        let mut count = 0;

        for (name, value_a) in a {
            if let Some(value_b) = b.get(name) {
                let similarity = match (value_a, value_b) {
                    (ParameterValue::Float(a), ParameterValue::Float(b)) => {
                        let diff = (a - b).abs();
                        (-diff).exp() // Gaussian-like similarity
                    },
                    (ParameterValue::Int(a), ParameterValue::Int(b)) => {
                        let diff = (a - b).abs() as f64;
                        (-diff).exp()
                    },
                    (ParameterValue::String(a), ParameterValue::String(b)) => {
                        if a == b {
                            1.0
                        } else {
                            0.0
                        }
                    },
                    (ParameterValue::Bool(a), ParameterValue::Bool(b)) => {
                        if a == b {
                            1.0
                        } else {
                            0.0
                        }
                    },
                    _ => 0.0,
                };

                total_similarity += similarity;
                count += 1;
            }
        }

        if count > 0 {
            total_similarity / count as f64
        } else {
            0.0
        }
    }
}

impl Default for GPSampler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyperopt::{search_space::SearchSpaceBuilder, Trial, TrialMetrics, TrialResult};

    fn create_test_search_space() -> SearchSpace {
        SearchSpaceBuilder::new()
            .continuous("learning_rate", 1e-5, 1e-1)
            .discrete("batch_size", 8, 128, 8)
            .categorical("optimizer", vec!["adam", "sgd", "adamw"])
            .build()
    }

    fn create_test_trial_history() -> TrialHistory {
        let mut history = TrialHistory::new(Direction::Maximize);

        // Add some completed trials
        for i in 0..5 {
            let mut params = HashMap::new();
            params.insert(
                "learning_rate".to_string(),
                ParameterValue::Float(0.01 * (i + 1) as f64),
            );
            params.insert("batch_size".to_string(), ParameterValue::Int(32));
            params.insert(
                "optimizer".to_string(),
                ParameterValue::String("adam".to_string()),
            );

            let mut trial = Trial::new(i, params);
            trial.complete(TrialResult::success(TrialMetrics::new(
                0.8 + i as f64 * 0.02,
            )));
            history.add_trial(trial);
        }

        history
    }

    #[test]
    fn test_random_sampler() {
        let mut sampler = RandomSampler::with_seed(42);
        let search_space = create_test_search_space();
        let history = TrialHistory::new(Direction::Maximize);

        let config = sampler.sample(&search_space, &history);

        assert_eq!(config.len(), 3);
        assert!(config.contains_key("learning_rate"));
        assert!(config.contains_key("batch_size"));
        assert!(config.contains_key("optimizer"));

        // Validate the sampled values
        assert!(search_space.validate(&config).is_ok());
    }

    #[test]
    fn test_tpe_sampler() {
        let mut sampler = TPESampler::with_config(SamplerConfig {
            seed: Some(42),
            n_startup_trials: 3,
            n_ei_candidates: 5,
        });

        let search_space = create_test_search_space();
        let history = create_test_trial_history();

        let config = sampler.sample(&search_space, &history);

        assert_eq!(config.len(), 3);
        assert!(search_space.validate(&config).is_ok());
        assert_eq!(sampler.name(), "TPESampler(startup=3, candidates=5)");
    }

    #[test]
    fn test_gp_sampler() {
        let mut sampler = GPSampler::with_config(SamplerConfig {
            seed: Some(42),
            n_startup_trials: 3,
            n_ei_candidates: 5,
        });

        let search_space = create_test_search_space();
        let history = create_test_trial_history();

        let config = sampler.sample(&search_space, &history);

        assert_eq!(config.len(), 3);
        assert!(search_space.validate(&config).is_ok());
    }

    #[test]
    fn test_sampler_with_insufficient_trials() {
        let mut sampler = TPESampler::with_config(SamplerConfig {
            seed: Some(42),
            n_startup_trials: 10, // More than available trials
            n_ei_candidates: 5,
        });

        let search_space = create_test_search_space();
        let history = create_test_trial_history(); // Only has 5 trials

        // Should fall back to random sampling
        let config = sampler.sample(&search_space, &history);
        assert!(search_space.validate(&config).is_ok());
    }
}
