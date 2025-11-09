use scirs2_core::random::*; // SciRS2 Integration Policy
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterSpace {
    pub parameters: HashMap<String, ParameterSpec>,
    pub constraints: Vec<ParameterConstraint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterSpec {
    Float {
        min: f64,
        max: f64,
        scale: ParameterScale,
    },
    Int {
        min: i64,
        max: i64,
    },
    Categorical {
        choices: Vec<String>,
    },
    Boolean,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterScale {
    Linear,
    Logarithmic,
    Exponential,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterConstraint {
    pub constraint_type: ConstraintType,
    pub parameters: Vec<String>,
    pub condition: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    Sum,
    Product,
    Conditional,
    Ordering,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterConfig {
    pub values: HashMap<String, ParameterValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterValue {
    Float(f64),
    Int(i64),
    String(String),
    Bool(bool),
}

impl ParameterValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ParameterValue::Float(v) => Some(*v),
            ParameterValue::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            ParameterValue::Int(v) => Some(*v),
            ParameterValue::Float(v) => Some(*v as i64),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<String> {
        match self {
            ParameterValue::String(v) => Some(v.clone()),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ParameterValue::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningResult {
    pub config: HyperparameterConfig,
    pub metrics: HashMap<String, f64>,
    pub primary_metric: f64,
    pub training_time: Duration,
    pub trial_id: String,
    pub iteration: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningConfig {
    pub max_trials: usize,
    pub max_duration: Option<Duration>,
    pub early_stopping_patience: Option<usize>,
    pub early_stopping_threshold: Option<f64>,
    pub primary_metric: String,
    pub optimization_direction: OptimizationDirection,
    pub search_algorithm: SearchAlgorithm,
    pub parallel_trials: usize,
    pub resource_allocation: ResourceAllocation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationDirection {
    Maximize,
    Minimize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchAlgorithm {
    Random,
    GridSearch,
    BayesianOptimization,
    TPE,        // Tree-structured Parzen Estimator
    BOHB,       // Bayesian Optimization and HyperBand
    Population, // Population Based Training
    Hyperband,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub max_gpu_memory_per_trial: Option<u64>,
    pub max_cpu_cores_per_trial: Option<u32>,
    pub max_training_time_per_trial: Option<Duration>,
    pub resource_sharing_strategy: ResourceSharingStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceSharingStrategy {
    Exclusive,
    Shared,
    Adaptive,
}

pub trait HyperparameterTuner: Send + Sync {
    fn suggest_configuration(
        &self,
        space: &HyperparameterSpace,
        history: &[TuningResult],
    ) -> Result<HyperparameterConfig, Box<dyn std::error::Error>>;

    fn update_with_result(
        &mut self,
        result: &TuningResult,
    ) -> Result<(), Box<dyn std::error::Error>>;

    fn get_best_configuration(&self) -> Option<&HyperparameterConfig>;

    fn should_stop(&self, history: &[TuningResult], config: &TuningConfig) -> bool;
}

pub struct RandomSearchTuner {
    best_config: Option<HyperparameterConfig>,
    best_score: Option<f64>,
}

impl Default for RandomSearchTuner {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomSearchTuner {
    pub fn new() -> Self {
        Self {
            best_config: None,
            best_score: None,
        }
    }
}

impl HyperparameterTuner for RandomSearchTuner {
    fn suggest_configuration(
        &self,
        space: &HyperparameterSpace,
        _history: &[TuningResult],
    ) -> Result<HyperparameterConfig, Box<dyn std::error::Error>> {
        let mut rng = thread_rng();
        let mut values = HashMap::new();

        for (param_name, param_spec) in &space.parameters {
            let value = match param_spec {
                ParameterSpec::Float { min, max, scale } => {
                    let val = match scale {
                        ParameterScale::Linear => rng.random_range(*min..=*max),
                        ParameterScale::Logarithmic => {
                            let log_min = min.ln();
                            let log_max = max.ln();
                            (rng.random_range(log_min..=log_max)).exp()
                        },
                        ParameterScale::Exponential => {
                            let exp_val: f64 = rng.random_range(0.0..=1.0);
                            min + (max - min) * exp_val.powi(2)
                        },
                    };
                    ParameterValue::Float(val)
                },
                ParameterSpec::Int { min, max } => {
                    ParameterValue::Int(rng.random_range(*min..=*max))
                },
                ParameterSpec::Categorical { choices } => {
                    let choice = choices[rng.random_range(0..choices.len())].clone();
                    ParameterValue::String(choice)
                },
                ParameterSpec::Boolean => ParameterValue::Bool(rng.gen_bool(0.5)),
            };
            values.insert(param_name.clone(), value);
        }

        Ok(HyperparameterConfig { values })
    }

    fn update_with_result(
        &mut self,
        result: &TuningResult,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match &self.best_score {
            None => {
                self.best_score = Some(result.primary_metric);
                self.best_config = Some(result.config.clone());
            },
            Some(current_best) => {
                if result.primary_metric > *current_best {
                    self.best_score = Some(result.primary_metric);
                    self.best_config = Some(result.config.clone());
                }
            },
        }
        Ok(())
    }

    fn get_best_configuration(&self) -> Option<&HyperparameterConfig> {
        self.best_config.as_ref()
    }

    fn should_stop(&self, history: &[TuningResult], config: &TuningConfig) -> bool {
        if let Some(patience) = config.early_stopping_patience {
            if history.len() >= patience {
                let recent_scores: Vec<f64> =
                    history.iter().rev().take(patience).map(|r| r.primary_metric).collect();

                let max_recent = recent_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let improvement = match &self.best_score {
                    Some(best) => max_recent - best,
                    None => 0.0,
                };

                if let Some(threshold) = config.early_stopping_threshold {
                    return improvement < threshold;
                }
            }
        }

        false
    }
}

pub struct BayesianOptimizationTuner {
    gaussian_process: Arc<RwLock<GaussianProcess>>,
    acquisition_function: AcquisitionFunction,
    best_config: Option<HyperparameterConfig>,
    best_score: Option<f64>,
}

impl BayesianOptimizationTuner {
    pub fn new(acquisition_function: AcquisitionFunction) -> Self {
        Self {
            gaussian_process: Arc::new(RwLock::new(GaussianProcess::new())),
            acquisition_function,
            best_config: None,
            best_score: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GaussianProcess {
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    kernel: Kernel,
}

impl Default for GaussianProcess {
    fn default() -> Self {
        Self::new()
    }
}

impl GaussianProcess {
    pub fn new() -> Self {
        Self {
            x_train: Vec::new(),
            y_train: Vec::new(),
            kernel: Kernel::RBF { length_scale: 1.0 },
        }
    }

    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) {
        self.x_train = x;
        self.y_train = y;
    }

    pub fn predict(&self, x: &[f64]) -> (f64, f64) {
        if self.x_train.is_empty() {
            return (0.0, 1.0); // mean=0, std=1
        }

        // Simplified GP prediction
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for (i, x_train) in self.x_train.iter().enumerate() {
            let similarity = self.kernel.compute(x, x_train);
            weighted_sum += similarity * self.y_train[i];
            weight_sum += similarity;
        }

        let mean = if weight_sum > 0.0 { weighted_sum / weight_sum } else { 0.0 };

        let std = 1.0 / (1.0 + weight_sum); // Simplified uncertainty

        (mean, std)
    }
}

#[derive(Debug, Clone)]
pub enum Kernel {
    RBF { length_scale: f64 },
    Matern { length_scale: f64, nu: f64 },
}

impl Kernel {
    fn compute(&self, x1: &[f64], x2: &[f64]) -> f64 {
        match self {
            Kernel::RBF { length_scale } => {
                let distance_sq: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                (-distance_sq / (2.0 * length_scale.powi(2))).exp()
            },
            Kernel::Matern { length_scale, nu } => {
                // Simplified Matern kernel (nu=1.5 case)
                let distance: f64 =
                    x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();

                let scaled_distance = distance / length_scale;
                if *nu == 1.5 {
                    (1.0 + scaled_distance * 3_f64.sqrt()) * (-scaled_distance * 3_f64.sqrt()).exp()
                } else {
                    // Fallback to RBF
                    (-distance / (2.0 * length_scale.powi(2))).exp()
                }
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum AcquisitionFunction {
    ExpectedImprovement { xi: f64 },
    UpperConfidenceBound { beta: f64 },
    ProbabilityOfImprovement { xi: f64 },
}

impl AcquisitionFunction {
    fn evaluate(&self, mean: f64, std: f64, best_so_far: f64) -> f64 {
        match self {
            AcquisitionFunction::ExpectedImprovement { xi } => {
                if std == 0.0 {
                    return 0.0;
                }
                let improvement = mean - best_so_far - xi;
                let z = improvement / std;
                improvement * normal_cdf(z) + std * normal_pdf(z)
            },
            AcquisitionFunction::UpperConfidenceBound { beta } => mean + beta * std,
            AcquisitionFunction::ProbabilityOfImprovement { xi } => {
                if std == 0.0 {
                    return if mean > best_so_far + xi { 1.0 } else { 0.0 };
                }
                let z = (mean - best_so_far - xi) / std;
                normal_cdf(z)
            },
        }
    }
}

// Simplified normal distribution functions
fn normal_pdf(x: f64) -> f64 {
    (1.0 / (2.0 * std::f64::consts::PI).sqrt()) * (-0.5 * x.powi(2)).exp()
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2_f64.sqrt()))
}

fn erf(x: f64) -> f64 {
    // Approximation of error function
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

impl HyperparameterTuner for BayesianOptimizationTuner {
    fn suggest_configuration(
        &self,
        space: &HyperparameterSpace,
        history: &[TuningResult],
    ) -> Result<HyperparameterConfig, Box<dyn std::error::Error>> {
        if history.is_empty() {
            // Use random search for first configuration
            return RandomSearchTuner::new().suggest_configuration(space, history);
        }

        // Update GP with history
        let x_train: Vec<Vec<f64>> = history
            .iter()
            .map(|r| config_to_vector(&r.config, space))
            .collect::<Result<Vec<_>, _>>()?;

        let y_train: Vec<f64> = history.iter().map(|r| r.primary_metric).collect();

        {
            let mut gp = self.gaussian_process.write().unwrap();
            gp.fit(x_train, y_train);
        }

        // Find best configuration so far
        let best_so_far =
            history.iter().map(|r| r.primary_metric).fold(f64::NEG_INFINITY, f64::max);

        // Optimize acquisition function
        let best_config = self.optimize_acquisition(space, best_so_far)?;
        Ok(best_config)
    }

    fn update_with_result(
        &mut self,
        result: &TuningResult,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match &self.best_score {
            None => {
                self.best_score = Some(result.primary_metric);
                self.best_config = Some(result.config.clone());
            },
            Some(current_best) => {
                if result.primary_metric > *current_best {
                    self.best_score = Some(result.primary_metric);
                    self.best_config = Some(result.config.clone());
                }
            },
        }
        Ok(())
    }

    fn get_best_configuration(&self) -> Option<&HyperparameterConfig> {
        self.best_config.as_ref()
    }

    fn should_stop(&self, history: &[TuningResult], config: &TuningConfig) -> bool {
        if let Some(patience) = config.early_stopping_patience {
            if history.len() >= patience {
                let recent_scores: Vec<f64> =
                    history.iter().rev().take(patience).map(|r| r.primary_metric).collect();

                let variance = recent_scores.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>()
                    / (recent_scores.len() - 1) as f64;

                if let Some(threshold) = config.early_stopping_threshold {
                    return variance < threshold;
                }
            }
        }

        false
    }
}

impl BayesianOptimizationTuner {
    fn optimize_acquisition(
        &self,
        space: &HyperparameterSpace,
        best_so_far: f64,
    ) -> Result<HyperparameterConfig, Box<dyn std::error::Error>> {
        let mut best_acquisition = f64::NEG_INFINITY;
        let mut best_config = None;

        // Simple grid search over acquisition function
        let num_candidates = 1000;
        let random_tuner = RandomSearchTuner::new();

        for _ in 0..num_candidates {
            let candidate = random_tuner.suggest_configuration(space, &[])?;
            let x = config_to_vector(&candidate, space)?;

            let (mean, std) = {
                let gp = self.gaussian_process.read().unwrap();
                gp.predict(&x)
            };

            let acquisition_value = self.acquisition_function.evaluate(mean, std, best_so_far);

            if acquisition_value > best_acquisition {
                best_acquisition = acquisition_value;
                best_config = Some(candidate);
            }
        }

        best_config.ok_or_else(|| "Failed to find candidate configuration".into())
    }
}

fn config_to_vector(
    config: &HyperparameterConfig,
    space: &HyperparameterSpace,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let mut vector = Vec::new();

    for (param_name, param_spec) in &space.parameters {
        let value = config
            .values
            .get(param_name)
            .ok_or_else(|| format!("Missing parameter: {}", param_name))?;

        let normalized_value = match (param_spec, value) {
            (ParameterSpec::Float { min, max, .. }, ParameterValue::Float(v)) => {
                (v - min) / (max - min)
            },
            (ParameterSpec::Int { min, max }, ParameterValue::Int(v)) => {
                (*v as f64 - *min as f64) / (*max as f64 - *min as f64)
            },
            (ParameterSpec::Categorical { choices }, ParameterValue::String(v)) => {
                choices
                    .iter()
                    .position(|c| c == v)
                    .ok_or_else(|| format!("Invalid categorical value: {}", v))?
                    as f64
                    / (choices.len() - 1) as f64
            },
            (ParameterSpec::Boolean, ParameterValue::Bool(v)) => {
                if *v {
                    1.0
                } else {
                    0.0
                }
            },
            _ => return Err("Parameter type mismatch".into()),
        };

        vector.push(normalized_value);
    }

    Ok(vector)
}

pub struct AutomatedHyperparameterTuner {
    tuner: Box<dyn HyperparameterTuner>,
    config: TuningConfig,
    results: Arc<RwLock<Vec<TuningResult>>>,
    start_time: Instant,
}

impl AutomatedHyperparameterTuner {
    pub fn new(tuner: Box<dyn HyperparameterTuner>, config: TuningConfig) -> Self {
        Self {
            tuner,
            config,
            results: Arc::new(RwLock::new(Vec::new())),
            start_time: Instant::now(),
        }
    }

    pub async fn optimize<F>(
        &mut self,
        space: &HyperparameterSpace,
        objective_fn: F,
    ) -> Result<TuningResult, Box<dyn std::error::Error>>
    where
        F: Fn(&HyperparameterConfig) -> Result<HashMap<String, f64>, Box<dyn std::error::Error>>
            + Send
            + Sync
            + Clone,
    {
        println!(
            "Starting hyperparameter optimization with {} max trials",
            self.config.max_trials
        );

        let mut iteration = 0;
        #[allow(unused_variables)]
        let mut trials_without_improvement = 0;

        while iteration < self.config.max_trials {
            // Check time limit
            if let Some(max_duration) = self.config.max_duration {
                if self.start_time.elapsed() > max_duration {
                    println!("Reached maximum duration, stopping optimization");
                    break;
                }
            }

            let results_snapshot = {
                let results = self.results.read().unwrap();
                results.clone()
            };

            // Check early stopping
            if self.tuner.should_stop(&results_snapshot, &self.config) {
                println!("Early stopping triggered after {} trials", iteration);
                break;
            }

            // Determine number of parallel trials
            let remaining_trials = self.config.max_trials - iteration;
            let parallel_trials = self.config.parallel_trials.min(remaining_trials);

            println!(
                "Starting batch of {} parallel trials (iteration {})",
                parallel_trials,
                iteration + 1
            );

            // Generate configurations for parallel trials
            let mut configs = Vec::new();
            for _ in 0..parallel_trials {
                let config = self.tuner.suggest_configuration(space, &results_snapshot)?;
                configs.push(config);
            }

            // Execute trials in parallel
            let objective_fn_clone = objective_fn.clone();
            let trial_results: Vec<Result<TuningResult, Box<dyn std::error::Error>>> = configs
                .into_iter()
                .enumerate()
                .map(|(i, config)| {
                    let trial_start = Instant::now();
                    let trial_id = format!("trial_{}", iteration + i);

                    println!("Executing {}", trial_id);

                    match objective_fn_clone(&config) {
                        Ok(metrics) => {
                            let primary_metric =
                                metrics.get(&self.config.primary_metric).copied().unwrap_or(0.0);

                            Ok(TuningResult {
                                config,
                                metrics,
                                primary_metric,
                                training_time: trial_start.elapsed(),
                                trial_id,
                                iteration: iteration + i,
                            })
                        },
                        Err(e) => Err(e),
                    }
                })
                .collect();

            // Process results
            let mut batch_improved = false;
            for result in trial_results {
                match result {
                    Ok(trial_result) => {
                        println!(
                            "Trial {} completed: {} = {:.4} (took {:?})",
                            trial_result.trial_id,
                            self.config.primary_metric,
                            trial_result.primary_metric,
                            trial_result.training_time
                        );

                        // Check if this result improves on the best so far
                        let is_improvement = match self.config.optimization_direction {
                            OptimizationDirection::Maximize => {
                                self.tuner.get_best_configuration().map_or(true, |_| {
                                    trial_result.primary_metric
                                        > self
                                            .results
                                            .read()
                                            .unwrap()
                                            .iter()
                                            .map(|r| r.primary_metric)
                                            .fold(f64::NEG_INFINITY, f64::max)
                                })
                            },
                            OptimizationDirection::Minimize => {
                                self.tuner.get_best_configuration().map_or(true, |_| {
                                    trial_result.primary_metric
                                        < self
                                            .results
                                            .read()
                                            .unwrap()
                                            .iter()
                                            .map(|r| r.primary_metric)
                                            .fold(f64::INFINITY, f64::min)
                                })
                            },
                        };

                        if is_improvement {
                            batch_improved = true;
                            trials_without_improvement = 0;
                            println!("🎉 New best result found!");
                        }

                        self.tuner.update_with_result(&trial_result)?;
                        {
                            let mut results = self.results.write().unwrap();
                            results.push(trial_result);
                        }
                    },
                    Err(e) => {
                        println!("Trial failed: {}", e);
                    },
                }
            }

            if !batch_improved {
                trials_without_improvement += parallel_trials;
            }

            iteration += parallel_trials;

            // Print progress
            self.print_progress(iteration);
        }

        // Return best result
        let results = self.results.read().unwrap();
        let best_result = match self.config.optimization_direction {
            OptimizationDirection::Maximize => results
                .iter()
                .max_by(|a, b| a.primary_metric.partial_cmp(&b.primary_metric).unwrap()),
            OptimizationDirection::Minimize => results
                .iter()
                .min_by(|a, b| a.primary_metric.partial_cmp(&b.primary_metric).unwrap()),
        };

        best_result.cloned().ok_or_else(|| "No successful trials completed".into())
    }

    fn print_progress(&self, completed_trials: usize) {
        let results = self.results.read().unwrap();
        if results.is_empty() {
            return;
        }

        let best_score = match self.config.optimization_direction {
            OptimizationDirection::Maximize => {
                results.iter().map(|r| r.primary_metric).fold(f64::NEG_INFINITY, f64::max)
            },
            OptimizationDirection::Minimize => {
                results.iter().map(|r| r.primary_metric).fold(f64::INFINITY, f64::min)
            },
        };

        let avg_score: f64 =
            results.iter().map(|r| r.primary_metric).sum::<f64>() / results.len() as f64;
        let elapsed = self.start_time.elapsed();

        println!(
            "Progress: {}/{} trials completed ({:.1}%) | Best {}: {:.4} | Avg: {:.4} | Elapsed: {:?}",
            completed_trials,
            self.config.max_trials,
            (completed_trials as f64 / self.config.max_trials as f64) * 100.0,
            self.config.primary_metric,
            best_score,
            avg_score,
            elapsed
        );
    }

    pub fn get_optimization_history(&self) -> Vec<TuningResult> {
        let results = self.results.read().unwrap();
        results.clone()
    }

    pub fn export_results(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let results = self.results.read().unwrap();
        let json = serde_json::to_string_pretty(&*results)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_space() -> HyperparameterSpace {
        let mut parameters = HashMap::new();

        parameters.insert(
            "learning_rate".to_string(),
            ParameterSpec::Float {
                min: 1e-5,
                max: 1e-1,
                scale: ParameterScale::Logarithmic,
            },
        );

        parameters.insert(
            "batch_size".to_string(),
            ParameterSpec::Int { min: 8, max: 128 },
        );

        parameters.insert(
            "optimizer".to_string(),
            ParameterSpec::Categorical {
                choices: vec!["adam".to_string(), "sgd".to_string(), "adamw".to_string()],
            },
        );

        HyperparameterSpace {
            parameters,
            constraints: Vec::new(),
        }
    }

    #[test]
    fn test_random_search_tuner() {
        let mut tuner = RandomSearchTuner::new();
        let space = create_test_space();

        let config = tuner.suggest_configuration(&space, &[]).unwrap();

        assert!(config.values.contains_key("learning_rate"));
        assert!(config.values.contains_key("batch_size"));
        assert!(config.values.contains_key("optimizer"));

        // Test update
        let result = TuningResult {
            config: config.clone(),
            metrics: {
                let mut m = HashMap::new();
                m.insert("accuracy".to_string(), 0.85);
                m
            },
            primary_metric: 0.85,
            training_time: Duration::from_secs(60),
            trial_id: "test".to_string(),
            iteration: 0,
        };

        tuner.update_with_result(&result).unwrap();
        assert!(tuner.get_best_configuration().is_some());
    }

    #[test]
    fn test_bayesian_optimization_tuner() {
        let tuner =
            BayesianOptimizationTuner::new(AcquisitionFunction::ExpectedImprovement { xi: 0.01 });
        let space = create_test_space();

        // First suggestion should be random (no history)
        let config = tuner.suggest_configuration(&space, &[]).unwrap();
        assert!(config.values.contains_key("learning_rate"));
    }

    #[test]
    fn test_config_to_vector() {
        let space = create_test_space();
        let mut values = HashMap::new();
        values.insert("learning_rate".to_string(), ParameterValue::Float(0.001));
        values.insert("batch_size".to_string(), ParameterValue::Int(32));
        values.insert(
            "optimizer".to_string(),
            ParameterValue::String("adam".to_string()),
        );

        let config = HyperparameterConfig { values };
        let vector = config_to_vector(&config, &space).unwrap();

        assert_eq!(vector.len(), 3);
        assert!(vector.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }

    #[tokio::test]
    async fn test_automated_tuner() {
        let tuner = Box::new(RandomSearchTuner::new());
        let config = TuningConfig {
            max_trials: 5,
            max_duration: Some(Duration::from_secs(30)),
            early_stopping_patience: None,
            early_stopping_threshold: None,
            primary_metric: "accuracy".to_string(),
            optimization_direction: OptimizationDirection::Maximize,
            search_algorithm: SearchAlgorithm::Random,
            parallel_trials: 2,
            resource_allocation: ResourceAllocation {
                max_gpu_memory_per_trial: None,
                max_cpu_cores_per_trial: None,
                max_training_time_per_trial: None,
                resource_sharing_strategy: ResourceSharingStrategy::Shared,
            },
        };

        let mut automated_tuner = AutomatedHyperparameterTuner::new(tuner, config);
        let space = create_test_space();

        // Mock objective function
        let objective_fn = |_config: &HyperparameterConfig| {
            let mut metrics = HashMap::new();
            metrics.insert("accuracy".to_string(), random::<f64>());
            Ok(metrics)
        };

        let result = automated_tuner.optimize(&space, objective_fn).await.unwrap();
        assert!(result.primary_metric >= 0.0 && result.primary_metric <= 1.0);

        let history = automated_tuner.get_optimization_history();
        assert!(history.len() <= 5);
    }
}
