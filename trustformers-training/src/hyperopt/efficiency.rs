//! Efficiency features for hyperparameter optimization
//!
//! This module provides advanced efficiency features to accelerate hyperparameter
//! optimization including warm starting, bandit algorithms, surrogate models,
//! and parallel evaluation strategies.

use super::{ParameterValue, SearchSpace, SearchStrategy, Trial, TrialHistory, TrialState};
use anyhow::Result;
use scirs2_core::random::*; // SciRS2 Integration Policy
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

/// Advanced early stopping configuration with multiple strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedEarlyStoppingConfig {
    /// Basic patience configuration
    pub patience: usize,
    /// Minimum improvement threshold
    pub min_delta: f64,
    /// Early stopping strategy
    pub strategy: EarlyStoppingStrategy,
    /// Adaptive patience adjustment
    pub adaptive_patience: bool,
    /// Minimum evaluation steps before early stopping
    pub min_evaluation_steps: usize,
    /// Grace period for initial convergence
    pub grace_period: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EarlyStoppingStrategy {
    /// Standard early stopping based on validation loss
    Standard,
    /// Early stopping based on training dynamics
    TrainingDynamics {
        /// Maximum gradient norm threshold
        max_gradient_norm: f64,
        /// Loss oscillation threshold
        loss_oscillation_threshold: f64,
    },
    /// Multi-objective early stopping
    MultiObjective {
        /// Primary metric for early stopping
        primary_metric: String,
        /// Secondary metrics to consider
        secondary_metrics: Vec<String>,
        /// Weights for each metric
        metric_weights: HashMap<String, f64>,
    },
    /// Bayesian early stopping using posterior predictions
    Bayesian {
        /// Confidence threshold for stopping
        confidence_threshold: f64,
        /// Number of posterior samples
        num_samples: usize,
    },
}

/// Warm starting strategies for hyperparameter optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmStartConfig {
    /// Strategy for warm starting
    pub strategy: WarmStartStrategy,
    /// Source of historical data
    pub data_source: WarmStartDataSource,
    /// Number of warm start trials to use
    pub num_warm_start_trials: usize,
    /// Weight decay for historical data importance
    pub historical_weight_decay: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WarmStartStrategy {
    /// Use best trials from previous studies
    BestTrials,
    /// Use diverse set of good trials
    DiverseBest {
        /// Diversity threshold
        diversity_threshold: f64,
    },
    /// Transfer learning from similar models
    TransferLearning {
        /// Similarity threshold
        similarity_threshold: f64,
        /// Feature mapping function
        feature_mapping: String,
    },
    /// Meta-learning based warm start
    MetaLearning {
        /// Meta-features to use
        meta_features: Vec<String>,
        /// Number of meta-learning epochs
        meta_epochs: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WarmStartDataSource {
    /// Local database of previous runs
    LocalDatabase { path: String },
    /// Remote database or API
    RemoteDatabase { url: String, auth_token: String },
    /// File-based storage
    FileStorage { directory: String },
    /// In-memory cache
    InMemory,
}

/// Multi-armed bandit algorithms for hyperparameter optimization
#[derive(Debug, Clone)]
pub struct BanditOptimizer {
    /// Bandit algorithm configuration
    config: BanditConfig,
    /// Arms (hyperparameter configurations)
    arms: Vec<HashMap<String, ParameterValue>>,
    /// Arm statistics
    arm_stats: Vec<ArmStatistics>,
    /// Current exploration factor
    #[allow(dead_code)]
    exploration_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditConfig {
    /// Bandit algorithm type
    pub algorithm: BanditAlgorithm,
    /// Exploration strategy
    pub exploration: ExplorationStrategy,
    /// Reward function configuration
    pub reward_function: RewardFunction,
    /// Number of arms to maintain
    pub num_arms: usize,
    /// Arm generation strategy
    pub arm_generation: ArmGenerationStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BanditAlgorithm {
    /// Upper Confidence Bound
    UCB {
        /// Confidence parameter
        confidence_parameter: f64,
    },
    /// Thompson Sampling
    ThompsonSampling {
        /// Prior parameters for Beta distribution
        alpha_prior: f64,
        beta_prior: f64,
    },
    /// Epsilon-Greedy
    EpsilonGreedy {
        /// Exploration probability
        epsilon: f64,
        /// Epsilon decay rate
        decay_rate: f64,
    },
    /// EXP3 (Exponential-weight algorithm for Exploration and Exploitation)
    EXP3 {
        /// Learning rate
        gamma: f64,
    },
    /// LinUCB for contextual bandits
    LinUCB {
        /// Regularization parameter
        alpha: f64,
        /// Context dimensionality
        context_dim: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExplorationStrategy {
    /// Fixed exploration rate
    Fixed { rate: f64 },
    /// Decaying exploration rate
    Decaying {
        initial_rate: f64,
        decay_factor: f64,
        min_rate: f64,
    },
    /// Adaptive exploration based on uncertainty
    Adaptive { uncertainty_threshold: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RewardFunction {
    /// Direct performance metric
    Direct { metric_name: String },
    /// Normalized performance
    Normalized {
        metric_name: String,
        min_value: f64,
        max_value: f64,
    },
    /// Time-weighted performance
    TimeWeighted {
        metric_name: String,
        time_weight: f64,
    },
    /// Multi-objective reward
    MultiObjective {
        metrics: HashMap<String, f64>, // metric_name -> weight
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArmGenerationStrategy {
    /// Random sampling from search space
    Random,
    /// Latin Hypercube Sampling
    LatinHypercube,
    /// Sobol sequences
    Sobol,
    /// Evolutionary generation
    Evolutionary {
        population_size: usize,
        mutation_rate: f64,
        crossover_rate: f64,
    },
}

#[derive(Debug, Clone)]
pub struct ArmStatistics {
    /// Number of times this arm was pulled
    pub pulls: usize,
    /// Total reward accumulated
    pub total_reward: f64,
    /// Average reward
    pub average_reward: f64,
    /// Confidence bounds
    pub confidence_bounds: (f64, f64),
    /// Last update timestamp
    pub last_update: SystemTime,
}

/// Surrogate model optimization for expensive hyperparameter evaluations
#[allow(dead_code)]
pub struct SurrogateOptimizer {
    /// Surrogate model configuration
    config: SurrogateConfig,
    /// Observed data points
    observations: Vec<(HashMap<String, ParameterValue>, f64)>,
    /// Surrogate model
    #[allow(dead_code)]
    model: Box<dyn SurrogateModel>,
    /// Acquisition function
    acquisition: Box<dyn AcquisitionFunction>,
}

impl std::fmt::Debug for SurrogateOptimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SurrogateOptimizer")
            .field("config", &self.config)
            .field("observations", &self.observations)
            .field("model", &"<dyn SurrogateModel>")
            .field("acquisition", &"<dyn AcquisitionFunction>")
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrogateConfig {
    /// Surrogate model type
    pub model_type: SurrogateModelType,
    /// Acquisition function type
    pub acquisition_function: AcquisitionFunctionType,
    /// Number of initial random samples
    pub initial_samples: usize,
    /// Model update frequency
    pub update_frequency: usize,
    /// Optimization budget per iteration
    pub optimization_budget: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SurrogateModelType {
    /// Gaussian Process
    GaussianProcess {
        /// Kernel type
        kernel: KernelType,
        /// Noise level
        noise_level: f64,
        /// Length scales
        length_scales: Vec<f64>,
    },
    /// Random Forest
    RandomForest {
        /// Number of trees
        num_trees: usize,
        /// Maximum depth
        max_depth: usize,
        /// Minimum samples per leaf
        min_samples_leaf: usize,
    },
    /// Neural Network
    NeuralNetwork {
        /// Hidden layer sizes
        hidden_sizes: Vec<usize>,
        /// Learning rate
        learning_rate: f64,
        /// Number of epochs
        epochs: usize,
    },
    /// Tree-structured Parzen Estimator
    TPE {
        /// Number of good/bad samples split
        n_startup_trials: usize,
        /// Gamma parameter
        gamma: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KernelType {
    /// Radial Basis Function kernel
    RBF,
    /// Matern kernel
    Matern { nu: f64 },
    /// Linear kernel
    Linear,
    /// Polynomial kernel
    Polynomial { degree: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AcquisitionFunctionType {
    /// Expected Improvement
    ExpectedImprovement { xi: f64 },
    /// Probability of Improvement
    ProbabilityOfImprovement { xi: f64 },
    /// Upper Confidence Bound
    UpperConfidenceBound { beta: f64 },
    /// Entropy Search
    EntropySearch,
    /// Knowledge Gradient
    KnowledgeGradient,
}

/// Parallel evaluation strategies for hyperparameter optimization
#[allow(dead_code)]
pub struct ParallelEvaluator {
    /// Configuration for parallel evaluation
    config: ParallelEvaluationConfig,
    /// Active evaluation jobs
    #[allow(dead_code)]
    active_jobs: Arc<Mutex<HashMap<String, EvaluationJob>>>,
    /// Completed jobs queue
    completed_jobs: Arc<Mutex<VecDeque<EvaluationResult>>>,
    /// Load balancer
    load_balancer: Box<dyn LoadBalancer>,
}

impl std::fmt::Debug for ParallelEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelEvaluator")
            .field("config", &self.config)
            .field(
                "active_jobs",
                &"<Arc<Mutex<HashMap<String, EvaluationJob>>>>",
            )
            .field(
                "completed_jobs",
                &"<Arc<Mutex<VecDeque<EvaluationResult>>>>",
            )
            .field("load_balancer", &"<dyn LoadBalancer>")
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelEvaluationConfig {
    /// Maximum number of parallel evaluations
    pub max_parallel: usize,
    /// Evaluation strategy
    pub strategy: ParallelStrategy,
    /// Resource allocation
    pub resource_allocation: ResourceAllocation,
    /// Fault tolerance settings
    pub fault_tolerance: FaultToleranceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParallelStrategy {
    /// Independent parallel evaluations
    Independent,
    /// Synchronized batch evaluations
    Batch { batch_size: usize },
    /// Asynchronous with speculation
    Asynchronous {
        /// Maximum speculation depth
        speculation_depth: usize,
    },
    /// Hierarchical evaluation
    Hierarchical {
        /// Levels in hierarchy
        levels: Vec<usize>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// CPU cores per evaluation
    pub cpu_cores: usize,
    /// Memory per evaluation (GB)
    pub memory_gb: f64,
    /// GPU allocation
    pub gpu_allocation: GPUAllocation,
    /// Priority levels
    pub priority_levels: Vec<PriorityLevel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GPUAllocation {
    /// No GPU
    None,
    /// Shared GPU
    Shared { memory_fraction: f64 },
    /// Dedicated GPU
    Dedicated { gpu_count: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityLevel {
    /// Priority value
    pub priority: i32,
    /// Resource multiplier
    pub resource_multiplier: f64,
    /// Maximum evaluations at this priority
    pub max_evaluations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceConfig {
    /// Maximum retries per evaluation
    pub max_retries: usize,
    /// Timeout per evaluation
    pub evaluation_timeout: Duration,
    /// Checkpoint frequency
    pub checkpoint_frequency: Duration,
}

#[derive(Debug, Clone)]
pub struct EvaluationJob {
    /// Job identifier
    pub job_id: String,
    /// Hyperparameters being evaluated
    pub parameters: HashMap<String, ParameterValue>,
    /// Start time
    pub start_time: SystemTime,
    /// Resource allocation
    pub resources: ResourceAllocation,
    /// Current status
    pub status: JobStatus,
}

#[derive(Debug, Clone)]
pub enum JobStatus {
    Queued,
    Running,
    Completed,
    Failed { error: String },
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// Job identifier
    pub job_id: String,
    /// Hyperparameters that were evaluated
    pub parameters: HashMap<String, ParameterValue>,
    /// Evaluation metrics
    pub metrics: HashMap<String, f64>,
    /// Evaluation time
    pub evaluation_time: Duration,
    /// Resource usage
    pub resource_usage: ResourceUsage,
}

#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Memory usage
    pub memory_usage: f64,
    /// GPU utilization
    pub gpu_utilization: f64,
    /// Network I/O
    pub network_io: f64,
}

/// Traits for extensibility

pub trait SurrogateModel: Send + Sync {
    /// Fit the model to observed data
    fn fit(&mut self, observations: &[(HashMap<String, ParameterValue>, f64)]) -> Result<()>;

    /// Predict mean and variance for given parameters
    fn predict(&self, parameters: &HashMap<String, ParameterValue>) -> Result<(f64, f64)>;

    /// Update model with new observation
    fn update(&mut self, parameters: HashMap<String, ParameterValue>, value: f64) -> Result<()>;
}

pub trait AcquisitionFunction: Send + Sync {
    /// Compute acquisition value for given parameters
    fn compute(
        &self,
        parameters: &HashMap<String, ParameterValue>,
        model: &dyn SurrogateModel,
        best_value: f64,
    ) -> Result<f64>;

    /// Optimize acquisition function to find next candidate
    fn optimize(
        &self,
        model: &dyn SurrogateModel,
        search_space: &SearchSpace,
        best_value: f64,
    ) -> Result<HashMap<String, ParameterValue>>;
}

pub trait LoadBalancer: Send + Sync {
    /// Assign job to best available resource
    fn assign_job(&mut self, job: &EvaluationJob) -> Result<String>;

    /// Update resource status
    fn update_resource_status(&mut self, resource_id: &str, usage: &ResourceUsage) -> Result<()>;

    /// Get available resources
    fn get_available_resources(&self) -> Vec<String>;
}

// Implementations

impl BanditOptimizer {
    pub fn new(config: BanditConfig, search_space: &SearchSpace) -> Result<Self> {
        let arms = Self::generate_arms(&config, search_space)?;
        let arm_stats = vec![ArmStatistics::new(); arms.len()];

        Ok(Self {
            config,
            arms,
            arm_stats,
            exploration_factor: 1.0,
        })
    }

    pub fn select_arm(&mut self) -> Result<usize> {
        match &self.config.algorithm {
            BanditAlgorithm::UCB {
                confidence_parameter,
            } => self.ucb_select(*confidence_parameter),
            BanditAlgorithm::ThompsonSampling {
                alpha_prior,
                beta_prior,
            } => self.thompson_sampling_select(*alpha_prior, *beta_prior),
            BanditAlgorithm::EpsilonGreedy {
                epsilon,
                decay_rate: _,
            } => self.epsilon_greedy_select(*epsilon),
            BanditAlgorithm::EXP3 { gamma } => self.exp3_select(*gamma),
            BanditAlgorithm::LinUCB {
                alpha,
                context_dim: _,
            } => self.linucb_select(*alpha),
        }
    }

    pub fn update_arm(&mut self, arm_index: usize, reward: f64) -> Result<()> {
        if arm_index >= self.arm_stats.len() {
            return Err(anyhow::anyhow!("Invalid arm index"));
        }

        let stats = &mut self.arm_stats[arm_index];
        stats.pulls += 1;
        stats.total_reward += reward;
        stats.average_reward = stats.total_reward / stats.pulls as f64;
        stats.last_update = SystemTime::now();

        // Update confidence bounds
        let confidence_radius = (2.0 * (stats.pulls as f64).ln() / stats.pulls as f64).sqrt();
        stats.confidence_bounds = (
            stats.average_reward - confidence_radius,
            stats.average_reward + confidence_radius,
        );

        Ok(())
    }

    fn generate_arms(
        config: &BanditConfig,
        search_space: &SearchSpace,
    ) -> Result<Vec<HashMap<String, ParameterValue>>> {
        let mut arms = Vec::new();

        match &config.arm_generation {
            ArmGenerationStrategy::Random => {
                for _ in 0..config.num_arms {
                    arms.push(search_space.sample_random()?);
                }
            },
            ArmGenerationStrategy::LatinHypercube => {
                arms = search_space.latin_hypercube_sample(config.num_arms)?;
            },
            ArmGenerationStrategy::Sobol => {
                arms = search_space.sobol_sample(config.num_arms)?;
            },
            ArmGenerationStrategy::Evolutionary { .. } => {
                // Implement evolutionary arm generation
                arms = search_space.evolutionary_sample(config.num_arms)?;
            },
        }

        Ok(arms)
    }

    fn ucb_select(&self, confidence_parameter: f64) -> Result<usize> {
        let total_pulls: usize = self.arm_stats.iter().map(|s| s.pulls).sum();

        if total_pulls == 0 {
            return Ok(0);
        }

        let mut best_arm = 0;
        let mut best_value = f64::NEG_INFINITY;

        for (i, stats) in self.arm_stats.iter().enumerate() {
            if stats.pulls == 0 {
                return Ok(i); // Explore unplayed arms first
            }

            let confidence_bound = confidence_parameter
                * (2.0 * (total_pulls as f64).ln() / stats.pulls as f64).sqrt();
            let ucb_value = stats.average_reward + confidence_bound;

            if ucb_value > best_value {
                best_value = ucb_value;
                best_arm = i;
            }
        }

        Ok(best_arm)
    }

    fn thompson_sampling_select(&self, alpha_prior: f64, beta_prior: f64) -> Result<usize> {
        let mut rng = thread_rng();

        let mut best_arm = 0;
        let mut best_sample = f64::NEG_INFINITY;

        for (i, stats) in self.arm_stats.iter().enumerate() {
            // Beta distribution parameters
            let _alpha = alpha_prior + stats.total_reward;
            let _beta = beta_prior + stats.pulls as f64 - stats.total_reward;

            // Sample from Beta distribution (simplified)
            let sample = rng.random::<f64>(); // In practice, use proper Beta sampling

            if sample > best_sample {
                best_sample = sample;
                best_arm = i;
            }
        }

        Ok(best_arm)
    }

    fn epsilon_greedy_select(&self, epsilon: f64) -> Result<usize> {
        let mut rng = thread_rng();

        if rng.random::<f64>() < epsilon {
            // Explore: select random arm
            Ok(rng.gen_range(0..self.arms.len()))
        } else {
            // Exploit: select best arm
            let best_arm = self
                .arm_stats
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.average_reward.partial_cmp(&b.average_reward).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            Ok(best_arm)
        }
    }

    fn exp3_select(&self, gamma: f64) -> Result<usize> {
        let mut rng = thread_rng();

        let num_arms = self.arms.len();
        if num_arms == 0 {
            return Err(anyhow::anyhow!("No arms available"));
        }

        // Compute weights based on cumulative rewards
        let mut weights = vec![1.0; num_arms];
        for (i, stats) in self.arm_stats.iter().enumerate() {
            if stats.pulls > 0 {
                // EXP3 weight update: w_i = exp(gamma * average_reward / num_arms)
                weights[i] = (gamma * stats.average_reward / num_arms as f64).exp();
            }
        }

        // Compute probabilities
        let weight_sum: f64 = weights.iter().sum();
        let mut probabilities = vec![0.0; num_arms];

        for i in 0..num_arms {
            probabilities[i] = (1.0 - gamma) * weights[i] / weight_sum + gamma / num_arms as f64;
        }

        // Sample according to probabilities
        let mut cumulative_prob = 0.0;
        let random_value = rng.random::<f64>();

        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative_prob += prob;
            if random_value <= cumulative_prob {
                return Ok(i);
            }
        }

        // Fallback to last arm
        Ok(num_arms - 1)
    }

    fn linucb_select(&self, alpha: f64) -> Result<usize> {
        // LinUCB requires contextual information which isn't available in the current design
        // For now, implement a simplified version that falls back to UCB
        // In a full implementation, this would use feature vectors for each arm

        let total_pulls: usize = self.arm_stats.iter().map(|s| s.pulls).sum();

        if total_pulls == 0 {
            return Ok(0);
        }

        let mut best_arm = 0;
        let mut best_value = f64::NEG_INFINITY;

        for (i, stats) in self.arm_stats.iter().enumerate() {
            if stats.pulls == 0 {
                return Ok(i); // Explore unplayed arms first
            }

            // Simplified LinUCB using parameter uncertainty as context
            let confidence_width = alpha * (total_pulls as f64 / stats.pulls as f64).ln().sqrt();
            let upper_bound = stats.average_reward + confidence_width;

            if upper_bound > best_value {
                best_value = upper_bound;
                best_arm = i;
            }
        }

        Ok(best_arm)
    }
}

impl SearchStrategy for BanditOptimizer {
    fn suggest(
        &mut self,
        _search_space: &SearchSpace,
        _history: &TrialHistory,
    ) -> Option<HashMap<String, ParameterValue>> {
        match self.select_arm() {
            Ok(arm_index) => Some(self.arms[arm_index].clone()),
            Err(_) => None,
        }
    }

    fn should_terminate(&self, _history: &TrialHistory) -> bool {
        false // Bandit algorithms typically don't self-terminate
    }

    fn name(&self) -> &str {
        "BanditOptimizer"
    }

    fn update(&mut self, trial: &Trial) {
        if let TrialState::Complete = trial.state {
            if let Some(value) =
                trial.result.as_ref().and_then(|r| r.metrics.metrics.get("objective"))
            {
                // Find which arm this trial corresponds to
                for (i, arm) in self.arms.iter().enumerate() {
                    if arm == &trial.params {
                        let _ = self.update_arm(i, *value);
                        break;
                    }
                }
            }
        }
    }
}

impl ArmStatistics {
    fn new() -> Self {
        Self {
            pulls: 0,
            total_reward: 0.0,
            average_reward: 0.0,
            confidence_bounds: (0.0, 0.0),
            last_update: SystemTime::now(),
        }
    }
}

impl Default for AdvancedEarlyStoppingConfig {
    fn default() -> Self {
        Self {
            patience: 10,
            min_delta: 0.001,
            strategy: EarlyStoppingStrategy::Standard,
            adaptive_patience: false,
            min_evaluation_steps: 100,
            grace_period: 5,
        }
    }
}

impl Default for WarmStartConfig {
    fn default() -> Self {
        Self {
            strategy: WarmStartStrategy::BestTrials,
            data_source: WarmStartDataSource::InMemory,
            num_warm_start_trials: 10,
            historical_weight_decay: 0.9,
        }
    }
}

impl Default for BanditConfig {
    fn default() -> Self {
        Self {
            algorithm: BanditAlgorithm::UCB {
                confidence_parameter: 1.0,
            },
            exploration: ExplorationStrategy::Fixed { rate: 0.1 },
            reward_function: RewardFunction::Direct {
                metric_name: "objective".to_string(),
            },
            num_arms: 10,
            arm_generation: ArmGenerationStrategy::Random,
        }
    }
}

impl Default for SurrogateConfig {
    fn default() -> Self {
        Self {
            model_type: SurrogateModelType::GaussianProcess {
                kernel: KernelType::RBF,
                noise_level: 0.01,
                length_scales: vec![1.0],
            },
            acquisition_function: AcquisitionFunctionType::ExpectedImprovement { xi: 0.01 },
            initial_samples: 20,
            update_frequency: 5,
            optimization_budget: 1000,
        }
    }
}

impl Default for ParallelEvaluationConfig {
    fn default() -> Self {
        Self {
            max_parallel: 4,
            strategy: ParallelStrategy::Independent,
            resource_allocation: ResourceAllocation {
                cpu_cores: 2,
                memory_gb: 4.0,
                gpu_allocation: GPUAllocation::None,
                priority_levels: vec![],
            },
            fault_tolerance: FaultToleranceConfig {
                max_retries: 3,
                evaluation_timeout: Duration::from_secs(3600),
                checkpoint_frequency: Duration::from_secs(300),
            },
        }
    }
}

// Extension methods for SearchSpace
impl SearchSpace {
    pub fn sample_random(&self) -> Result<HashMap<String, ParameterValue>> {
        let mut rng = thread_rng();
        let mut params = HashMap::new();

        for param in &self.parameters {
            let value = match param {
                super::search_space::HyperParameter::Continuous(p) => {
                    let val = rng.gen_range(p.low..=p.high);
                    ParameterValue::Float(val)
                },
                super::search_space::HyperParameter::Log(p) => {
                    let log_low = p.low.ln();
                    let log_high = p.high.ln();
                    let log_val = rng.gen_range(log_low..=log_high);
                    ParameterValue::Float(log_val.exp())
                },
                super::search_space::HyperParameter::Discrete(p) => {
                    let val = rng.gen_range(p.low..=p.high);
                    ParameterValue::Int(val)
                },
                super::search_space::HyperParameter::Categorical(p) => {
                    let choice = &p.choices[rng.gen_range(0..p.choices.len())];
                    ParameterValue::String(choice.clone())
                },
            };
            params.insert(param.name().to_string(), value);
        }

        Ok(params)
    }

    pub fn latin_hypercube_sample(
        &self,
        n_samples: usize,
    ) -> Result<Vec<HashMap<String, ParameterValue>>> {
        let mut rng = thread_rng();
        let mut samples = Vec::new();

        if n_samples == 0 {
            return Ok(samples);
        }

        // Get only continuous/log parameters for LHS
        let continuous_params: Vec<_> = self
            .parameters
            .iter()
            .filter(|p| {
                matches!(
                    p,
                    super::search_space::HyperParameter::Continuous(_)
                        | super::search_space::HyperParameter::Log(_)
                )
            })
            .collect();

        let n_dims = continuous_params.len();

        if n_dims == 0 {
            // Fall back to random sampling for discrete/categorical only
            for _ in 0..n_samples {
                samples.push(self.sample_random()?);
            }
            return Ok(samples);
        }

        // Generate LHS matrix
        let mut lhs_matrix = vec![vec![0.0; n_dims]; n_samples];

        for dim in 0..n_dims {
            let mut indices: Vec<usize> = (0..n_samples).collect();

            // Shuffle indices
            for i in (1..indices.len()).rev() {
                let j = rng.gen_range(0..=i);
                indices.swap(i, j);
            }

            for (i, &idx) in indices.iter().enumerate() {
                let lower = idx as f64 / n_samples as f64;
                let upper = (idx + 1) as f64 / n_samples as f64;
                lhs_matrix[i][dim] = rng.gen_range(lower..upper);
            }
        }

        // Convert LHS matrix to parameter samples
        for i in 0..n_samples {
            let mut params = HashMap::new();

            // Handle continuous/log parameters with LHS
            for (dim, param) in continuous_params.iter().enumerate() {
                let unit_value = lhs_matrix[i][dim];
                let value = match param {
                    super::search_space::HyperParameter::Continuous(p) => {
                        let val = p.low + unit_value * (p.high - p.low);
                        ParameterValue::Float(val)
                    },
                    super::search_space::HyperParameter::Log(p) => {
                        let log_low = p.low.ln();
                        let log_high = p.high.ln();
                        let log_val = log_low + unit_value * (log_high - log_low);
                        ParameterValue::Float(log_val.exp())
                    },
                    _ => unreachable!(),
                };
                params.insert(param.name().to_string(), value);
            }

            // Handle discrete/categorical parameters randomly
            for param in &self.parameters {
                if !matches!(
                    param,
                    super::search_space::HyperParameter::Continuous(_)
                        | super::search_space::HyperParameter::Log(_)
                ) {
                    let value = match param {
                        super::search_space::HyperParameter::Discrete(p) => {
                            let val = rng.gen_range(p.low..=p.high);
                            ParameterValue::Int(val)
                        },
                        super::search_space::HyperParameter::Categorical(p) => {
                            let choice = &p.choices[rng.gen_range(0..p.choices.len())];
                            ParameterValue::String(choice.clone())
                        },
                        _ => unreachable!(),
                    };
                    params.insert(param.name().to_string(), value);
                }
            }

            samples.push(params);
        }

        Ok(samples)
    }

    pub fn sobol_sample(&self, n_samples: usize) -> Result<Vec<HashMap<String, ParameterValue>>> {
        // Simplified Sobol sequence implementation
        // For production use, consider using a proper Sobol sequence library
        let mut rng = thread_rng();
        let mut samples = Vec::new();

        let continuous_params: Vec<_> = self
            .parameters
            .iter()
            .filter(|p| {
                matches!(
                    p,
                    super::search_space::HyperParameter::Continuous(_)
                        | super::search_space::HyperParameter::Log(_)
                )
            })
            .collect();

        let n_dims = continuous_params.len();

        if n_dims == 0 {
            // Fall back to random sampling
            for _ in 0..n_samples {
                samples.push(self.sample_random()?);
            }
            return Ok(samples);
        }

        // Generate quasi-random Sobol-like sequence
        for i in 0..n_samples {
            let mut params = HashMap::new();

            for (dim, param) in continuous_params.iter().enumerate() {
                // Simple Van der Corput sequence for each dimension
                let unit_value = self.van_der_corput(i + 1, 2 + dim);

                let value = match param {
                    super::search_space::HyperParameter::Continuous(p) => {
                        let val = p.low + unit_value * (p.high - p.low);
                        ParameterValue::Float(val)
                    },
                    super::search_space::HyperParameter::Log(p) => {
                        let log_low = p.low.ln();
                        let log_high = p.high.ln();
                        let log_val = log_low + unit_value * (log_high - log_low);
                        ParameterValue::Float(log_val.exp())
                    },
                    _ => unreachable!(),
                };
                params.insert(param.name().to_string(), value);
            }

            // Handle discrete/categorical parameters randomly
            for param in &self.parameters {
                if !matches!(
                    param,
                    super::search_space::HyperParameter::Continuous(_)
                        | super::search_space::HyperParameter::Log(_)
                ) {
                    let value = match param {
                        super::search_space::HyperParameter::Discrete(p) => {
                            let val = rng.gen_range(p.low..=p.high);
                            ParameterValue::Int(val)
                        },
                        super::search_space::HyperParameter::Categorical(p) => {
                            let choice = &p.choices[rng.gen_range(0..p.choices.len())];
                            ParameterValue::String(choice.clone())
                        },
                        _ => unreachable!(),
                    };
                    params.insert(param.name().to_string(), value);
                }
            }

            samples.push(params);
        }

        Ok(samples)
    }

    pub fn evolutionary_sample(
        &self,
        n_samples: usize,
    ) -> Result<Vec<HashMap<String, ParameterValue>>> {
        let mut rng = thread_rng();
        let mut samples = Vec::new();

        if n_samples == 0 {
            return Ok(samples);
        }

        // Initialize population with random samples
        let population_size = (n_samples / 4).max(10);
        let mut population = Vec::new();

        for _ in 0..population_size {
            population.push(self.sample_random()?);
        }

        // Evolve population to generate samples
        let generations = (n_samples / population_size).max(1);
        let mutation_rate = 0.1;
        let crossover_rate = 0.7;

        for _gen in 0..generations {
            let mut new_population = Vec::new();

            // Selection and reproduction
            for _ in 0..population_size {
                if rng.random::<f64>() < crossover_rate && population.len() >= 2 {
                    // Crossover
                    let parent1_idx = rng.gen_range(0..population.len());
                    let parent2_idx = rng.gen_range(0..population.len());
                    let offspring =
                        self.crossover(&population[parent1_idx], &population[parent2_idx])?;
                    new_population.push(offspring);
                } else {
                    // Mutation
                    let parent_idx = rng.gen_range(0..population.len());
                    let mutated = self.mutate(&population[parent_idx], mutation_rate)?;
                    new_population.push(mutated);
                }
            }

            // Replace population
            population = new_population;

            // Add best individuals to samples
            for individual in &population {
                if samples.len() < n_samples {
                    samples.push(individual.clone());
                }
            }
        }

        // Fill remaining samples with random if needed
        while samples.len() < n_samples {
            samples.push(self.sample_random()?);
        }

        samples.truncate(n_samples);
        Ok(samples)
    }

    // Helper function for Van der Corput sequence
    fn van_der_corput(&self, n: usize, base: usize) -> f64 {
        let mut result = 0.0;
        let mut denominator = 1.0;
        let mut num = n;

        while num > 0 {
            denominator *= base as f64;
            result += (num % base) as f64 / denominator;
            num /= base;
        }

        result
    }

    // Helper function for crossover in evolutionary sampling
    fn crossover(
        &self,
        parent1: &HashMap<String, ParameterValue>,
        parent2: &HashMap<String, ParameterValue>,
    ) -> Result<HashMap<String, ParameterValue>> {
        let mut rng = thread_rng();
        let mut offspring = HashMap::new();

        for param in &self.parameters {
            let param_name = param.name();
            let value = if rng.random::<f64>() < 0.5 {
                parent1.get(param_name).cloned()
            } else {
                parent2.get(param_name).cloned()
            };

            if let Some(v) = value {
                offspring.insert(param_name.to_string(), v);
            } else {
                // Fallback to random value if parent doesn't have this parameter
                let random_value = match param {
                    super::search_space::HyperParameter::Continuous(p) => {
                        ParameterValue::Float(rng.gen_range(p.low..=p.high))
                    },
                    super::search_space::HyperParameter::Log(p) => {
                        let log_val = rng.gen_range(p.low.ln()..=p.high.ln());
                        ParameterValue::Float(log_val.exp())
                    },
                    super::search_space::HyperParameter::Discrete(p) => {
                        ParameterValue::Int(rng.gen_range(p.low..=p.high))
                    },
                    super::search_space::HyperParameter::Categorical(p) => {
                        let choice = &p.choices[rng.gen_range(0..p.choices.len())];
                        ParameterValue::String(choice.clone())
                    },
                };
                offspring.insert(param_name.to_string(), random_value);
            }
        }

        Ok(offspring)
    }

    // Helper function for mutation in evolutionary sampling
    fn mutate(
        &self,
        individual: &HashMap<String, ParameterValue>,
        mutation_rate: f64,
    ) -> Result<HashMap<String, ParameterValue>> {
        let mut rng = thread_rng();
        let mut mutated = individual.clone();

        for param in &self.parameters {
            if rng.random::<f64>() < mutation_rate {
                let param_name = param.name();
                let new_value = match param {
                    super::search_space::HyperParameter::Continuous(p) => {
                        if let Some(ParameterValue::Float(current)) = individual.get(param_name) {
                            // Gaussian mutation
                            let std_dev = (p.high - p.low) * 0.1;
                            let noise = rng.random::<f64>() * 2.0 - 1.0; // Simple noise
                            let new_val = (current + noise * std_dev).clamp(p.low, p.high);
                            ParameterValue::Float(new_val)
                        } else {
                            ParameterValue::Float(rng.gen_range(p.low..=p.high))
                        }
                    },
                    super::search_space::HyperParameter::Log(p) => {
                        if let Some(ParameterValue::Float(current)) = individual.get(param_name) {
                            let log_current = current.ln();
                            let log_std = (p.high.ln() - p.low.ln()) * 0.1;
                            let noise = rng.random::<f64>() * 2.0 - 1.0;
                            let new_log =
                                (log_current + noise * log_std).clamp(p.low.ln(), p.high.ln());
                            ParameterValue::Float(new_log.exp())
                        } else {
                            let log_val = rng.gen_range(p.low.ln()..=p.high.ln());
                            ParameterValue::Float(log_val.exp())
                        }
                    },
                    super::search_space::HyperParameter::Discrete(p) => {
                        ParameterValue::Int(rng.gen_range(p.low..=p.high))
                    },
                    super::search_space::HyperParameter::Categorical(p) => {
                        let choice = &p.choices[rng.gen_range(0..p.choices.len())];
                        ParameterValue::String(choice.clone())
                    },
                };
                mutated.insert(param_name.to_string(), new_value);
            }
        }

        Ok(mutated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_early_stopping_config() {
        let config = AdvancedEarlyStoppingConfig::default();
        assert_eq!(config.patience, 10);
        assert!(matches!(config.strategy, EarlyStoppingStrategy::Standard));
    }

    #[test]
    fn test_warm_start_config() {
        let config = WarmStartConfig::default();
        assert!(matches!(config.strategy, WarmStartStrategy::BestTrials));
        assert_eq!(config.num_warm_start_trials, 10);
    }

    #[test]
    fn test_bandit_config() {
        let config = BanditConfig::default();
        assert!(matches!(config.algorithm, BanditAlgorithm::UCB { .. }));
        assert_eq!(config.num_arms, 10);
    }

    #[test]
    fn test_surrogate_config() {
        let config = SurrogateConfig::default();
        assert!(matches!(
            config.model_type,
            SurrogateModelType::GaussianProcess { .. }
        ));
        assert_eq!(config.initial_samples, 20);
    }

    #[test]
    fn test_parallel_evaluation_config() {
        let config = ParallelEvaluationConfig::default();
        assert_eq!(config.max_parallel, 4);
        assert!(matches!(config.strategy, ParallelStrategy::Independent));
    }

    #[test]
    fn test_arm_statistics() {
        let stats = ArmStatistics::new();
        assert_eq!(stats.pulls, 0);
        assert_eq!(stats.total_reward, 0.0);
        assert_eq!(stats.average_reward, 0.0);
    }
}
