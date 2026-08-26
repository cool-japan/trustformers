//! # Multi-Task Learning Framework
//!
//! This module provides a comprehensive framework for multi-task learning,
//! enabling models to learn multiple related tasks simultaneously to improve
//! generalization and efficiency.
//!
//! ## Features
//!
//! - **Multiple MTL Architectures**: Hard parameter sharing, soft parameter sharing, task-specific layers
//! - **Loss Balancing**: Various strategies for balancing losses across tasks
//! - **Task Weighting**: Dynamic and static task weight adjustment
//! - **Auxiliary Tasks**: Support for auxiliary tasks to improve main task performance
//! - **Task Clustering**: Grouping related tasks for better sharing
//! - **Evaluation Metrics**: Specialized metrics for multi-task scenarios
//!
//! ## Usage
//!
//! ```rust,no_run
//! use trustformers_models::multi_task_learning::{
//!     MultiTaskLearningTrainer, MTLConfig, MTLArchitecture,
//!     LossBalancingStrategy, TaskConfig, TaskType, RegressionLossType,
//! };
//! use trustformers_models::continual_learning::NamedParameters;
//! use trustformers_core::{traits::{Config, Model}, tensor::Tensor, Result};
//! use serde::{Deserialize, Serialize};
//! use std::collections::HashMap;
//!
//! # #[derive(Debug, Clone, Serialize, Deserialize)]
//! # struct DocConfig;
//! # impl Config for DocConfig {
//! #     fn architecture(&self) -> &'static str { "doc" }
//! # }
//! # struct DocModel { scale: Tensor }
//! # impl Model for DocModel {
//! #     type Config = DocConfig;
//! #     type Input = Tensor;
//! #     type Output = Tensor;
//! #     fn forward(&self, input: Tensor) -> Result<Tensor> { input.mul(&self.scale) }
//! #     fn load_pretrained(&mut self, _r: &mut dyn std::io::Read) -> Result<()> { Ok(()) }
//! #     fn get_config(&self) -> &DocConfig { &DocConfig }
//! #     fn num_parameters(&self) -> usize { 1 }
//! # }
//! # impl NamedParameters for DocModel {
//! #     fn named_parameters(&self) -> Vec<(String, Tensor)> {
//! #         vec![("scale".to_string(), self.scale.clone())]
//! #     }
//! #     fn set_named_parameter(&mut self, name: &str, value: Tensor) -> Result<()> {
//! #         if name == "scale" { self.scale = value; }
//! #         Ok(())
//! #     }
//! # }
//!
//! # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
//! let config = MTLConfig {
//!     architecture: MTLArchitecture::HardParameterSharing {
//!         shared_layers: 8,
//!         task_specific_layers: 2,
//!     },
//!     loss_balancing: LossBalancingStrategy::DynamicWeightAverage,
//!     tasks: vec![
//!         TaskConfig::new("classification", TaskType::Classification { num_classes: 10, use_class_weights: false }),
//!         TaskConfig::new("regression", TaskType::Regression { output_dim: 1, loss_type: RegressionLossType::MSE }),
//!     ],
//!     ..Default::default()
//! };
//!
//! # let base_model = DocModel { scale: Tensor::ones(&[1, 768])? };
//! let mut trainer = MultiTaskLearningTrainer::new(base_model, config)?;
//! # let task_data = HashMap::new();
//! trainer.train_multi_task_step(&task_data)?;
//! # Ok(())
//! # }
//! ```

use crate::continual_learning::NamedParameters;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use trustformers_core::{
    errors::invalid_input,
    layers::Linear,
    tensor::Tensor,
    traits::{Layer, Model},
    Result,
};

/// Configuration for multi-task learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MTLConfig {
    /// Multi-task learning architecture
    pub architecture: MTLArchitecture,
    /// Strategy for balancing losses across tasks
    pub loss_balancing: LossBalancingStrategy,
    /// Task configurations
    pub tasks: Vec<TaskConfig>,
    /// Whether to use task embeddings
    pub use_task_embeddings: bool,
    /// Task embedding dimension
    pub task_embedding_dim: usize,
    /// Whether to use auxiliary tasks
    pub use_auxiliary_tasks: bool,
    /// Auxiliary task configurations
    pub auxiliary_tasks: Vec<AuxiliaryTaskConfig>,
    /// Task clustering configuration
    pub task_clustering: Option<TaskClusteringConfig>,
    /// Evaluation frequency for each task
    pub evaluation_frequency: usize,
    /// Whether to use task scheduling
    pub use_task_scheduling: bool,
    /// Task scheduling strategy
    pub task_scheduling: TaskSchedulingStrategy,
}

impl Default for MTLConfig {
    fn default() -> Self {
        Self {
            architecture: MTLArchitecture::HardParameterSharing {
                shared_layers: 8,
                task_specific_layers: 2,
            },
            loss_balancing: LossBalancingStrategy::EqualWeighting,
            tasks: Vec::new(),
            use_task_embeddings: false,
            task_embedding_dim: 64,
            use_auxiliary_tasks: false,
            auxiliary_tasks: Vec::new(),
            task_clustering: None,
            evaluation_frequency: 1000,
            use_task_scheduling: false,
            task_scheduling: TaskSchedulingStrategy::RoundRobin,
        }
    }
}

/// Multi-task learning architectures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MTLArchitecture {
    /// Hard parameter sharing - shared bottom layers, task-specific top layers
    HardParameterSharing {
        shared_layers: usize,
        task_specific_layers: usize,
    },
    /// Soft parameter sharing - each task has its own parameters with regularization
    SoftParameterSharing {
        regularization_weight: f32,
        regularization_type: RegularizationType,
    },
    /// Multi-gate mixture of experts
    MultiGateMixtureOfExperts {
        num_experts: usize,
        expert_dim: usize,
        num_gates: usize,
    },
    /// Cross-stitch networks
    CrossStitchNetworks {
        num_tasks: usize,
        cross_stitch_layers: Vec<usize>,
    },
    /// Task routing networks
    TaskRoutingNetworks {
        num_routers: usize,
        routing_dim: usize,
    },
    /// Progressive Neural Networks for MTL
    ProgressiveNetworks {
        lateral_connections: bool,
        adapter_layers: bool,
    },
    /// Attention-based task sharing
    AttentionBasedSharing {
        attention_dim: usize,
        num_attention_heads: usize,
    },
}

/// Regularization types for soft parameter sharing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegularizationType {
    /// L2 regularization between task parameters
    L2Regularization,
    /// Trace norm regularization
    TraceNorm,
    /// Group LASSO
    GroupLasso,
    /// Elastic net
    ElasticNet { l1_weight: f32, l2_weight: f32 },
}

/// Strategies for balancing losses across tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LossBalancingStrategy {
    /// Equal weighting for all tasks
    EqualWeighting,
    /// Manual task weights
    ManualWeighting { weights: Vec<f32> },
    /// Uncertainty-based weighting
    UncertaintyWeighting,
    /// Dynamic weight average
    DynamicWeightAverage,
    /// GradNorm - gradient magnitude balancing
    GradNorm { alpha: f32 },
    /// Task-balanced sampling
    TaskBalancedSampling,
    /// Focal loss for hard tasks
    FocalLoss { gamma: f32 },
    /// Meta-learning based weighting
    MetaLearning { meta_lr: f32 },
}

/// Task configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskConfig {
    /// Task name/identifier
    pub name: String,
    /// Task type and parameters
    pub task_type: TaskType,
    /// Task weight (if using manual weighting)
    pub weight: f32,
    /// Task priority
    pub priority: TaskPriority,
    /// Whether this is the main task
    pub is_main_task: bool,
    /// Task-specific learning rate
    pub learning_rate: Option<f32>,
    /// Task-specific batch size
    pub batch_size: Option<usize>,
}

impl TaskConfig {
    pub fn new(name: &str, task_type: TaskType) -> Self {
        Self {
            name: name.to_string(),
            task_type,
            weight: 1.0,
            priority: TaskPriority::Normal,
            is_main_task: false,
            learning_rate: None,
            batch_size: None,
        }
    }

    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    pub fn with_priority(mut self, priority: TaskPriority) -> Self {
        self.priority = priority;
        self
    }

    pub fn as_main_task(mut self) -> Self {
        self.is_main_task = true;
        self
    }
}

/// Task types and their specific parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    /// Classification task
    Classification {
        num_classes: usize,
        use_class_weights: bool,
    },
    /// Regression task
    Regression {
        output_dim: usize,
        loss_type: RegressionLossType,
    },
    /// Sequence labeling task
    SequenceLabeling { num_labels: usize, use_crf: bool },
    /// Generation task
    Generation {
        vocab_size: usize,
        max_length: usize,
    },
    /// Ranking task
    Ranking { ranking_type: RankingType },
    /// Auxiliary task
    Auxiliary { auxiliary_type: AuxiliaryType },
}

/// Regression loss types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegressionLossType {
    MSE,
    MAE,
    Huber { delta: f32 },
    LogCosh,
}

/// Ranking task types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RankingType {
    Pairwise,
    Listwise,
    Pointwise,
}

/// Auxiliary task types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuxiliaryType {
    LanguageModeling,
    MaskedLanguageModeling,
    NextSentencePrediction,
    SentenceOrderPrediction,
    WordOrderPrediction,
    Custom { name: String },
}

/// Task priorities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Auxiliary task configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuxiliaryTaskConfig {
    pub name: String,
    pub auxiliary_type: AuxiliaryType,
    pub weight: f32,
    pub frequency: AuxiliaryTaskFrequency,
}

/// Frequency of auxiliary task training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuxiliaryTaskFrequency {
    /// Train every N main task steps
    EveryNSteps(usize),
    /// Train with probability P
    WithProbability(f32),
    /// Train continuously
    Continuous,
    /// Train only in certain epochs
    EpochRange { start: usize, end: usize },
}

/// Task clustering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskClusteringConfig {
    pub clustering_method: ClusteringMethod,
    pub num_clusters: usize,
    pub update_frequency: usize,
}

/// Clustering methods for tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClusteringMethod {
    /// Cluster by gradient similarity
    GradientSimilarity,
    /// Cluster by task performance correlation
    PerformanceCorrelation,
    /// Cluster by data similarity
    DataSimilarity,
    /// Manual clustering
    Manual { clusters: Vec<Vec<String>> },
}

/// Task scheduling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskSchedulingStrategy {
    /// Round-robin scheduling
    RoundRobin,
    /// Weighted sampling by task priority
    WeightedSampling,
    /// Performance-based scheduling
    PerformanceBased,
    /// Curriculum-based scheduling
    CurriculumBased { difficulty_order: Vec<String> },
    /// Random scheduling
    Random,
}

/// Multi-task learning trainer
pub struct MultiTaskLearningTrainer<M: Model> {
    /// Base model (shared layers)
    pub base_model: M,
    /// Task-specific heads
    pub task_heads: HashMap<String, TaskHead>,
    /// Configuration
    pub config: MTLConfig,
    /// Task losses and weights
    pub task_weights: HashMap<String, f32>,
    /// Task performance history
    pub task_performance: HashMap<String, Vec<f32>>,
    /// Current training step
    pub step_counter: usize,
    /// Task scheduling state
    pub scheduler_state: TaskSchedulerState,
    /// Gradient statistics for balancing
    pub gradient_stats: HashMap<String, GradientStats>,
    /// Per-task loss history (most recent last), used by Dynamic Weight Average
    pub task_loss_history: HashMap<String, VecDeque<f32>>,
    /// First observed loss per task, used as the GradNorm reference `L_k(0)`
    pub initial_task_losses: HashMap<String, f32>,
    /// Learned homoscedastic log-variances, used by Uncertainty Weighting
    pub task_log_variances: HashMap<String, f32>,
}

/// Number of past losses retained per task for Dynamic Weight Average.
const LOSS_HISTORY_CAPACITY: usize = 8;

/// Read a one-element loss tensor as an `f32`.
fn loss_scalar(tensor: &Tensor) -> Result<f32> {
    tensor
        .to_vec_f32()?
        .first()
        .copied()
        .ok_or_else(|| invalid_input("expected a non-empty loss tensor"))
}

/// Index of the largest element of a slice.
fn argmax_index(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

impl<M: Model<Input = Tensor, Output = Tensor> + NamedParameters> MultiTaskLearningTrainer<M> {
    /// Create a new multi-task learning trainer
    pub fn new(base_model: M, config: MTLConfig) -> Result<Self> {
        let mut task_heads = HashMap::new();
        let mut task_weights = HashMap::new();
        let mut task_log_variances = HashMap::new();

        // Initialize task heads
        for task_config in &config.tasks {
            let task_head = TaskHead::new(&task_config.task_type)?;
            task_heads.insert(task_config.name.clone(), task_head);
            task_weights.insert(task_config.name.clone(), task_config.weight);
            // log sigma^2 = 0 => sigma = 1 => the task starts unweighted.
            task_log_variances.insert(task_config.name.clone(), 0.0);
        }

        let scheduler_state = TaskSchedulerState::new(&config.task_scheduling);

        Ok(Self {
            base_model,
            task_heads,
            config,
            task_weights,
            task_performance: HashMap::new(),
            step_counter: 0,
            scheduler_state,
            gradient_stats: HashMap::new(),
            task_loss_history: HashMap::new(),
            initial_task_losses: HashMap::new(),
            task_log_variances,
        })
    }

    /// Train on multiple tasks for one step
    pub fn train_multi_task_step(
        &mut self,
        task_data: &HashMap<String, TaskBatch>,
    ) -> Result<MultiTaskOutput> {
        let mut task_losses = HashMap::new();
        let mut task_accuracies = HashMap::new();
        let mut total_loss = Tensor::zeros(&[1])?;

        // Determine which tasks to train on this step
        let active_tasks = self.get_active_tasks(task_data)?;

        for task_name in &active_tasks {
            if let Some(batch) = task_data.get(task_name) {
                // Forward pass through shared layers
                let shared_features = self.base_model.forward(batch.inputs.clone())?;

                // Task-specific forward pass
                let task_head = self
                    .task_heads
                    .get(task_name)
                    .ok_or_else(|| anyhow::anyhow!("Task head not found: {}", task_name))?;

                let task_outputs = task_head.forward(&shared_features)?;
                let task_loss = self.compute_task_loss(task_name, &task_outputs, &batch.targets)?;
                let task_accuracy =
                    self.compute_task_accuracy(task_name, &task_outputs, &batch.targets)?;

                task_losses.insert(task_name.clone(), task_loss.clone());
                task_accuracies.insert(task_name.clone(), task_accuracy);

                // Update task performance history
                self.task_performance.entry(task_name.clone()).or_default().push(task_accuracy);
            }
        }

        // GradNorm needs real gradient magnitudes against the shared trunk.
        if matches!(
            self.config.loss_balancing,
            LossBalancingStrategy::GradNorm { .. }
        ) {
            self.update_gradient_stats(task_data, &active_tasks)?;
        }

        // Balance losses across tasks
        let balanced_losses = self.balance_losses(&task_losses)?;

        // Compute total loss
        for (task_name, loss) in &balanced_losses {
            let weight = self.task_weights.get(task_name).copied().unwrap_or(1.0);
            total_loss = total_loss.add(&loss.scalar_mul(weight)?)?;
        }

        // Update task weights if using dynamic balancing
        self.record_task_losses(&task_losses)?;

        // Update auxiliary tasks if enabled
        if self.config.use_auxiliary_tasks {
            let aux_loss = self.compute_auxiliary_losses(task_data)?;
            total_loss = total_loss.add(&aux_loss)?;
        }

        self.step_counter += 1;

        let mut reported_losses = HashMap::new();
        for (task_name, loss) in task_losses {
            reported_losses.insert(task_name, loss_scalar(&loss)?);
        }

        Ok(MultiTaskOutput {
            total_loss,
            task_losses: reported_losses,
            task_accuracies,
            active_tasks,
            task_weights: self.task_weights.clone(),
        })
    }

    /// Get active tasks for current training step
    fn get_active_tasks(&mut self, task_data: &HashMap<String, TaskBatch>) -> Result<Vec<String>> {
        match &self.config.task_scheduling {
            TaskSchedulingStrategy::RoundRobin => {
                let task_names: Vec<String> = task_data.keys().cloned().collect();
                if task_names.is_empty() {
                    return Ok(Vec::new());
                }
                let current_task = &task_names[self.step_counter % task_names.len()];
                Ok(vec![current_task.clone()])
            },
            TaskSchedulingStrategy::WeightedSampling => {
                // Sample tasks based on their weights/priorities
                let mut weighted_tasks = Vec::new();
                for task_config in &self.config.tasks {
                    if task_data.contains_key(&task_config.name) {
                        let weight = match task_config.priority {
                            TaskPriority::Low => 0.5,
                            TaskPriority::Normal => 1.0,
                            TaskPriority::High => 2.0,
                            TaskPriority::Critical => 3.0,
                        };
                        for _ in 0..(weight * 10.0) as usize {
                            weighted_tasks.push(task_config.name.clone());
                        }
                    }
                }
                if weighted_tasks.is_empty() {
                    return Ok(Vec::new());
                }
                let selected_task = &weighted_tasks[self.step_counter % weighted_tasks.len()];
                Ok(vec![selected_task.clone()])
            },
            TaskSchedulingStrategy::Random => {
                let task_names: Vec<String> = task_data.keys().cloned().collect();
                if task_names.is_empty() {
                    return Ok(Vec::new());
                }
                let random_idx = fastrand::usize(..task_names.len());
                Ok(vec![task_names[random_idx].clone()])
            },
            _ => {
                // For other strategies, train on all available tasks
                Ok(task_data.keys().cloned().collect())
            },
        }
    }

    /// Balance losses across tasks
    pub fn balance_losses(
        &self,
        task_losses: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        match &self.config.loss_balancing {
            LossBalancingStrategy::EqualWeighting => Ok(task_losses.clone()),
            LossBalancingStrategy::ManualWeighting { weights } => {
                let mut balanced = HashMap::new();
                // Iterate in the configured task order so index-based weights
                // map to a stable task, not to HashMap iteration order.
                let mut names: Vec<&String> = task_losses.keys().collect();
                names.sort();
                for (i, task_name) in names.into_iter().enumerate() {
                    let Some(loss) = task_losses.get(task_name) else {
                        continue;
                    };
                    let weight = weights.get(i).copied().unwrap_or(1.0);
                    balanced.insert(task_name.clone(), loss.scalar_mul(weight)?);
                }
                Ok(balanced)
            },
            LossBalancingStrategy::UncertaintyWeighting => {
                self.apply_uncertainty_weighting(task_losses)
            },
            LossBalancingStrategy::DynamicWeightAverage => {
                self.apply_dynamic_weight_average(task_losses)
            },
            LossBalancingStrategy::GradNorm { alpha } => self.apply_gradnorm(task_losses, *alpha),
            _ => Ok(task_losses.clone()),
        }
    }

    /// Kendall-style homoscedastic uncertainty weighting.
    ///
    /// Each task contributes `L_k / (2 σ_k²) + log σ_k`, where the learned
    /// log-variance `s_k = log σ_k²` is stored on the trainer and updated by
    /// [`Self::record_task_losses`]. Tasks with a large learned variance are
    /// down-weighted, and the `log σ_k` term prevents the trivial solution of
    /// pushing every variance to infinity.
    fn apply_uncertainty_weighting(
        &self,
        task_losses: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut balanced = HashMap::new();

        for (task_name, loss) in task_losses {
            let log_variance = self.task_log_variances.get(task_name).copied().unwrap_or(0.0);
            let precision = 0.5 * (-log_variance).exp();
            let scaled = loss.scalar_mul(precision)?;
            // + log sigma = 0.5 * log sigma^2
            balanced.insert(task_name.clone(), scaled.add_scalar(0.5 * log_variance)?);
        }

        Ok(balanced)
    }

    /// Dynamic Weight Average (Liu et al., 2019).
    ///
    /// `w_k(t) = L_k(t-1) / L_k(t-2)` measures how fast task `k` is still
    /// improving; the weights are `λ_k = K · softmax(w / T)_k`, so they sum to
    /// the number of tasks and a *slowly* improving task gets a larger weight.
    /// Before two steps of history exist the losses pass through unchanged.
    fn apply_dynamic_weight_average(
        &self,
        task_losses: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let temperature = 2.0f32;

        let mut names: Vec<String> = task_losses.keys().cloned().collect();
        names.sort();

        let mut ratios = Vec::with_capacity(names.len());
        for name in &names {
            let Some((previous, before)) = self.previous_two_task_losses(name) else {
                // Not enough history yet for any task: leave the losses alone.
                return Ok(task_losses.clone());
            };
            if before.abs() <= f32::EPSILON {
                return Ok(task_losses.clone());
            }
            ratios.push(previous / before);
        }

        let max_ratio = ratios.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exponentials: Vec<f32> =
            ratios.iter().map(|r| ((r - max_ratio) / temperature).exp()).collect();
        let total: f32 = exponentials.iter().sum();
        if total <= 0.0 {
            return Ok(task_losses.clone());
        }

        let count = names.len() as f32;
        let mut balanced = HashMap::new();
        for (name, exponential) in names.iter().zip(exponentials.iter()) {
            let Some(loss) = task_losses.get(name) else {
                continue;
            };
            let weight = count * exponential / total;
            balanced.insert(name.clone(), loss.mul_scalar(weight)?);
        }

        Ok(balanced)
    }

    /// GradNorm (Chen et al., 2018).
    ///
    /// Uses the per-task gradient norms measured against the shared trunk (see
    /// [`Self::update_gradient_stats`]) to rescale each task loss towards the
    /// target `Ḡ · r_k^α`, where `r_k` is the task's relative inverse training
    /// rate. When no gradient statistics have been measured yet the losses
    /// pass through unchanged.
    fn apply_gradnorm(
        &self,
        task_losses: &HashMap<String, Tensor>,
        alpha: f32,
    ) -> Result<HashMap<String, Tensor>> {
        let mut names: Vec<String> = task_losses.keys().cloned().collect();
        names.sort();

        let mut norms = Vec::with_capacity(names.len());
        let mut relative_rates = Vec::with_capacity(names.len());
        for name in &names {
            let Some(stats) = self.gradient_stats.get(name) else {
                return Ok(task_losses.clone());
            };
            norms.push(stats.gradient_norm.max(0.0));

            let Some(loss) = task_losses.get(name) else {
                return Ok(task_losses.clone());
            };
            let current = loss_scalar(loss)?;
            let initial = self.initial_task_losses.get(name).copied().unwrap_or(current);
            relative_rates.push(if initial.abs() > f32::EPSILON { current / initial } else { 1.0 });
        }

        let mean_norm = norms.iter().sum::<f32>() / norms.len().max(1) as f32;
        let mean_rate = relative_rates.iter().sum::<f32>() / relative_rates.len().max(1) as f32;
        if mean_norm <= f32::EPSILON || mean_rate <= f32::EPSILON {
            return Ok(task_losses.clone());
        }

        let mut balanced = HashMap::new();
        for ((name, norm), rate) in names.iter().zip(norms.iter()).zip(relative_rates.iter()) {
            let Some(loss) = task_losses.get(name) else {
                continue;
            };
            let target = mean_norm * (rate / mean_rate).powf(alpha);
            let scale = if *norm > f32::EPSILON { (target / norm).clamp(0.1, 10.0) } else { 1.0 };
            balanced.insert(name.clone(), loss.mul_scalar(scale)?);
        }

        Ok(balanced)
    }

    /// Measure the gradient norm of each task loss with respect to the shared
    /// trunk parameters.
    ///
    /// Analytic gradients are used when the base model provides them through
    /// [`NamedParameters::parameter_gradients`]; otherwise the norm is
    /// estimated with central finite differences, which costs `2 · P` forward
    /// passes per task for `P` shared scalar parameters.
    pub fn update_gradient_stats(
        &mut self,
        task_data: &HashMap<String, TaskBatch>,
        active_tasks: &[String],
    ) -> Result<()> {
        let epsilon = 1e-3f32;

        for task_name in active_tasks {
            let Some(batch) = task_data.get(task_name) else {
                continue;
            };

            let mut squared_norm = 0.0f32;
            let parameters = self.base_model.named_parameters();

            if let Some(gradients) =
                self.base_model.parameter_gradients(&batch.inputs, &batch.targets)?
            {
                for gradient in gradients.values() {
                    squared_norm += gradient.to_vec_f32()?.iter().map(|g| g * g).sum::<f32>();
                }
            } else {
                for (name, tensor) in parameters {
                    let shape = tensor.shape();
                    let baseline = tensor.to_vec_f32()?;

                    for index in 0..baseline.len() {
                        let mut plus = baseline.clone();
                        plus[index] += epsilon;
                        self.base_model
                            .set_named_parameter(&name, Tensor::from_vec(plus, &shape)?)?;
                        let loss_plus = self.task_loss_value(task_name, batch)?;

                        let mut minus = baseline.clone();
                        minus[index] -= epsilon;
                        self.base_model
                            .set_named_parameter(&name, Tensor::from_vec(minus, &shape)?)?;
                        let loss_minus = self.task_loss_value(task_name, batch)?;

                        let gradient = (loss_plus - loss_minus) / (2.0 * epsilon);
                        squared_norm += gradient * gradient;
                    }

                    self.base_model
                        .set_named_parameter(&name, Tensor::from_vec(baseline, &shape)?)?;
                }
            }

            let weight = self.task_weights.get(task_name).copied().unwrap_or(1.0);
            let norm = weight * squared_norm.sqrt();
            let entry = self.gradient_stats.entry(task_name.clone()).or_insert(GradientStats {
                gradient_norm: norm,
                gradient_variance: 0.0,
                update_count: 0,
            });
            let previous = entry.gradient_norm;
            entry.update_count += 1;
            entry.gradient_norm = norm;
            entry.gradient_variance = (norm - previous).powi(2);
        }

        Ok(())
    }

    /// Recompute one task's loss through the current shared trunk and head.
    fn task_loss_value(&self, task_name: &str, batch: &TaskBatch) -> Result<f32> {
        let shared_features = self.base_model.forward(batch.inputs.clone())?;
        let task_head = self
            .task_heads
            .get(task_name)
            .ok_or_else(|| invalid_input(format!("Task head not found: {}", task_name)))?;
        let outputs = task_head.forward(&shared_features)?;
        let loss = self.compute_task_loss(task_name, &outputs, &batch.targets)?;
        loss_scalar(&loss)
    }

    /// Record this step's losses and advance any learned balancing state.
    pub fn record_task_losses(&mut self, task_losses: &HashMap<String, Tensor>) -> Result<()> {
        // Every strategy needs the loss history; record it first.
        for (task_name, loss) in task_losses {
            let value = loss_scalar(loss)?;
            self.initial_task_losses.entry(task_name.clone()).or_insert(value);

            let history = self.task_loss_history.entry(task_name.clone()).or_default();
            history.push_back(value);
            while history.len() > LOSS_HISTORY_CAPACITY {
                history.pop_front();
            }
        }

        if let LossBalancingStrategy::UncertaintyWeighting = &self.config.loss_balancing {
            // Gradient-descent step on Kendall's objective with respect to the
            // log-variance: d/ds [ L e^{-s} / 2 + s / 2 ] = (1 - L e^{-s}) / 2.
            let learning_rate = 0.01f32;
            for (task_name, loss) in task_losses {
                let value = loss_scalar(loss)?;
                let entry = self.task_log_variances.entry(task_name.clone()).or_insert(0.0);
                let gradient = 0.5 * (1.0 - value * (-*entry).exp());
                *entry = (*entry - learning_rate * gradient).clamp(-5.0, 5.0);
            }
        }

        Ok(())
    }

    /// The two most recent recorded losses for a task, `(t-1, t-2)`.
    fn previous_two_task_losses(&self, task_name: &str) -> Option<(f32, f32)> {
        let history = self.task_loss_history.get(task_name)?;
        let length = history.len();
        if length < 2 {
            return None;
        }
        Some((history[length - 1], history[length - 2]))
    }

    /// Compute auxiliary task losses
    fn compute_auxiliary_losses(&self, task_data: &HashMap<String, TaskBatch>) -> Result<Tensor> {
        let mut aux_loss: Tensor = Tensor::zeros(&[1])?;

        for aux_config in &self.config.auxiliary_tasks {
            if self.should_train_auxiliary_task(aux_config) {
                if let Some(aux_data) = task_data.get(&aux_config.name) {
                    let aux_task_loss: Tensor =
                        self.compute_auxiliary_task_loss(aux_config, aux_data)?;
                    let weighted_loss: Tensor = aux_task_loss.mul_scalar(aux_config.weight)?;
                    aux_loss = aux_loss.add(&weighted_loss)?;
                }
            }
        }

        Ok(aux_loss)
    }

    /// Check if auxiliary task should be trained this step
    fn should_train_auxiliary_task(&self, aux_config: &AuxiliaryTaskConfig) -> bool {
        match &aux_config.frequency {
            AuxiliaryTaskFrequency::EveryNSteps(n) => self.step_counter.is_multiple_of(*n),
            AuxiliaryTaskFrequency::WithProbability(p) => fastrand::f32() < *p,
            AuxiliaryTaskFrequency::Continuous => true,
            AuxiliaryTaskFrequency::EpochRange { start, end } => {
                let current_epoch = self.step_counter / 1000; // Simplified epoch calculation
                current_epoch >= *start && current_epoch <= *end
            },
        }
    }

    /// Compute auxiliary task loss
    fn compute_auxiliary_task_loss(
        &self,
        aux_config: &AuxiliaryTaskConfig,
        data: &TaskBatch,
    ) -> Result<Tensor> {
        // Compute loss for auxiliary task
        let shared_features: Tensor = self.base_model.forward(data.inputs.clone())?;

        match &aux_config.auxiliary_type {
            AuxiliaryType::LanguageModeling => {
                // Causal LM: predict position t+1 from the features at t.
                self.compute_lm_loss(&shared_features, &data.targets)
            },
            AuxiliaryType::MaskedLanguageModeling => {
                // MLM: score only the positions the mask marks as predicted.
                self.compute_mlm_loss(&shared_features, &data.targets)
            },
            other => Err(invalid_input(format!(
                "auxiliary task type {:?} has no implemented loss; remove it from \
                 config.auxiliary_tasks or implement its objective",
                other
            ))),
        }
    }

    /// Causal language-modeling cross-entropy.
    ///
    /// `features` are `[batch, seq_len, vocab]` logits and `targets` the
    /// matching one-hot distribution. Positions are shifted by one so that
    /// step `t` predicts the token at `t + 1`; a sequence shorter than two
    /// steps has nothing to predict and scores zero.
    fn compute_lm_loss(&self, features: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let shape = features.shape();
        if shape.len() != 3 {
            return Err(invalid_input(format!(
                "language-modeling loss expects [batch, seq_len, vocab] logits, got {:?}",
                shape
            )));
        }
        let seq_len = shape[1];
        if seq_len < 2 {
            return Tensor::from_vec(vec![0.0], &[1]);
        }

        let predictions = features.slice(1, 0, seq_len - 1)?.contiguous()?;
        let labels = targets.slice(1, 1, seq_len)?.contiguous()?;
        if predictions.shape() != labels.shape() {
            return Err(invalid_input(format!(
                "language-modeling targets {:?} do not match logits {:?}",
                labels.shape(),
                predictions.shape()
            )));
        }

        let log_probs = predictions.log_softmax(-1)?;
        let nll = labels.mul(&log_probs)?.sum(Some(vec![2]), false)?;
        nll.mean()?.mul_scalar(-1.0)?.reshape(&[1])
    }

    /// Masked language-modeling cross-entropy.
    ///
    /// Only positions whose one-hot target row is non-empty count towards the
    /// loss, which is exactly the set of masked positions.
    fn compute_mlm_loss(&self, features: &Tensor, targets: &Tensor) -> Result<Tensor> {
        if features.shape() != targets.shape() {
            return Err(invalid_input(format!(
                "masked language-modeling targets {:?} do not match logits {:?}",
                targets.shape(),
                features.shape()
            )));
        }

        let shape = features.shape();
        let vocab = *shape.last().ok_or_else(|| {
            invalid_input("masked language-modeling logits need at least one dimension")
        })?;
        if vocab == 0 {
            return Err(invalid_input(
                "masked language-modeling logits have no vocabulary",
            ));
        }

        let log_probs = features.log_softmax(-1)?.to_vec_f32()?;
        let target_values = targets.to_vec_f32()?;
        let positions = log_probs.len() / vocab;

        let mut total = 0.0f32;
        let mut counted = 0usize;
        for position in 0..positions {
            let range = position * vocab..(position + 1) * vocab;
            let mass: f32 = target_values[range.clone()].iter().sum();
            if mass <= f32::EPSILON {
                continue; // unmasked position
            }
            let contribution: f32 = target_values[range.clone()]
                .iter()
                .zip(log_probs[range].iter())
                .map(|(t, l)| t * l)
                .sum();
            total -= contribution / mass;
            counted += 1;
        }

        let value = if counted == 0 { 0.0 } else { total / counted as f32 };
        Tensor::from_vec(vec![value], &[1])
    }

    /// Compute task-specific loss
    pub fn compute_task_loss(
        &self,
        task_name: &str,
        outputs: &Tensor,
        targets: &Tensor,
    ) -> Result<Tensor> {
        let task_config = self
            .config
            .tasks
            .iter()
            .find(|t| t.name == task_name)
            .ok_or_else(|| invalid_input(format!("Task not found: {}", task_name)))?;

        match &task_config.task_type {
            TaskType::Classification { .. } => {
                // Cross-entropy: -sum(target * log softmax(logits)). The fused
                // log-softmax is used both for numerical stability and because
                // plain softmax here would compute -sum(t * p), which is
                // bounded in [-1, 0] and has the wrong gradient.
                let log_probs = outputs.log_softmax(-1)?;
                let nll_loss = targets.mul(&log_probs)?.sum(Some(vec![1]), false)?;
                nll_loss.mean()?.mul_scalar(-1.0)?.reshape(&[1])
            },
            TaskType::Regression { loss_type, .. } => {
                let diff = outputs.sub(targets)?;
                match loss_type {
                    RegressionLossType::MSE => diff.mul(&diff)?.mean()?.reshape(&[1]),
                    RegressionLossType::MAE => diff.abs()?.mean()?.reshape(&[1]),
                    RegressionLossType::Huber { delta } => {
                        // where(|d| <= delta, 0.5 d^2, delta |d| - 0.5 delta^2)
                        let abs_diff = diff.abs()?;
                        let delta_tensor = Tensor::full(*delta, abs_diff.shape())?;
                        // 1 where |d| > delta, 0 otherwise.
                        let large = abs_diff.greater(&delta_tensor)?;
                        let small = Tensor::ones_like(&large)?.sub(&large)?;

                        let quadratic = diff.mul(&diff)?.mul_scalar(0.5)?;
                        let linear =
                            abs_diff.mul_scalar(*delta)?.sub_scalar(0.5 * *delta * *delta)?;

                        quadratic.mul(&small)?.add(&linear.mul(&large)?)?.mean()?.reshape(&[1])
                    },
                    RegressionLossType::LogCosh => {
                        // log(cosh(d)) computed as |d| + log1p(exp(-2|d|)) - ln 2
                        // for numerical stability at large residuals.
                        let values: Vec<f32> = diff
                            .to_vec_f32()?
                            .into_iter()
                            .map(|d| {
                                let a = d.abs();
                                a + (-2.0 * a).exp().ln_1p() - std::f32::consts::LN_2
                            })
                            .collect();
                        let count = values.len().max(1) as f32;
                        Tensor::from_vec(vec![values.iter().sum::<f32>() / count], &[1])
                    },
                }
            },
            // Token-level cross-entropy over the last dimension.
            TaskType::SequenceLabeling { .. } | TaskType::Generation { .. } => {
                let log_probs = outputs.log_softmax(-1)?;
                let last_axis = outputs.shape().len().saturating_sub(1);
                let nll = targets.mul(&log_probs)?.sum(Some(vec![last_axis]), false)?;
                nll.mean()?.mul_scalar(-1.0)?.reshape(&[1])
            },
            TaskType::Ranking { ranking_type } => Err(invalid_input(format!(
                "ranking loss ({:?}) needs pairwise/listwise structure that TaskBatch does not \
                 carry; supply a ranking-aware batch type before selecting this task type",
                ranking_type
            ))),
            TaskType::Auxiliary { .. } => Err(invalid_input(
                "auxiliary tasks are scored through compute_auxiliary_losses, not \
                 compute_task_loss",
            )),
        }
    }

    /// Compute task-specific accuracy
    pub fn compute_task_accuracy(
        &self,
        task_name: &str,
        outputs: &Tensor,
        targets: &Tensor,
    ) -> Result<f32> {
        let task_config = self
            .config
            .tasks
            .iter()
            .find(|t| t.name == task_name)
            .ok_or_else(|| invalid_input(format!("Task not found: {}", task_name)))?;

        match &task_config.task_type {
            TaskType::Classification { .. } => {
                // Batch-wise accuracy: compare argmax per row, not only the
                // first element of the batch.
                let classes = *outputs.shape().last().ok_or_else(|| {
                    invalid_input("classification outputs need at least one dimension")
                })?;
                if classes == 0 {
                    return Err(invalid_input("classification outputs have zero classes"));
                }
                let output_values = outputs.to_vec_f32()?;
                let target_values = targets.to_vec_f32()?;
                if target_values.len() != output_values.len() {
                    return Err(invalid_input(
                        "classification accuracy needs one-hot targets matching the logits",
                    ));
                }

                let rows = output_values.len() / classes;
                let mut correct = 0usize;
                for row in 0..rows {
                    let range = row * classes..(row + 1) * classes;
                    let predicted = argmax_index(&output_values[range.clone()]);
                    let expected = argmax_index(&target_values[range]);
                    if predicted == expected {
                        correct += 1;
                    }
                }
                Ok(correct as f32 / rows.max(1) as f32)
            },
            TaskType::Regression { .. } => {
                // For regression, compute R² or similar metric
                let diff = outputs.sub(targets)?;
                let mse = diff.mul(&diff)?.mean()?;
                let mean_targets = targets.mean()?;
                let diff_from_mean = targets.sub(&mean_targets)?;
                let variance = diff_from_mean.pow_scalar(2.0)?.mean()?;
                let mse_value = loss_scalar(&mse.reshape(&[1])?)?;
                let variance_value = loss_scalar(&variance.reshape(&[1])?)?;
                if variance_value.abs() <= f32::EPSILON {
                    // A constant target has no variance to explain; R^2 is
                    // undefined, so report perfect fit only for an exact match.
                    return Ok(if mse_value <= f32::EPSILON { 1.0 } else { 0.0 });
                }
                Ok((1.0 - mse_value / variance_value).max(0.0))
            },
            _ => Ok(0.0),
        }
    }

    /// Evaluate all tasks
    pub fn evaluate_all_tasks(
        &self,
        test_data: &HashMap<String, TaskBatch>,
    ) -> Result<MultiTaskEvaluation> {
        let mut task_evaluations = HashMap::new();

        for (task_name, batch) in test_data {
            if let Some(task_head) = self.task_heads.get(task_name) {
                let shared_features = self.base_model.forward(batch.inputs.clone())?;
                let task_outputs = task_head.forward(&shared_features)?;
                let loss = self.compute_task_loss(task_name, &task_outputs, &batch.targets)?;
                let accuracy =
                    self.compute_task_accuracy(task_name, &task_outputs, &batch.targets)?;

                task_evaluations.insert(
                    task_name.clone(),
                    TaskEvaluation {
                        task_name: task_name.clone(),
                        loss: loss_scalar(&loss)?,
                        accuracy,
                        num_examples: batch.inputs.shape()[0],
                    },
                );
            }
        }

        let overall_accuracy = if !task_evaluations.is_empty() {
            task_evaluations.values().map(|e| e.accuracy).sum::<f32>()
                / task_evaluations.len() as f32
        } else {
            0.0
        };

        Ok(MultiTaskEvaluation {
            task_evaluations,
            overall_accuracy,
            step: self.step_counter,
        })
    }

    /// Get multi-task learning statistics
    pub fn get_mtl_stats(&self) -> MTLStats {
        MTLStats {
            num_tasks: self.config.tasks.len(),
            task_weights: self.task_weights.clone(),
            step_counter: self.step_counter,
            architecture: self.config.architecture.clone(),
            loss_balancing: self.config.loss_balancing.clone(),
        }
    }
}

/// Task-specific neural network head
pub struct TaskHead {
    layers: Vec<Linear>,
    #[allow(dead_code)]
    task_type: TaskType,
}

impl TaskHead {
    pub fn new(task_type: &TaskType) -> Result<Self> {
        let mut layers = Vec::new();

        match task_type {
            TaskType::Classification { num_classes, .. } => {
                // Simple classification head
                layers.push(Linear::new(768, *num_classes, true)); // Assuming 768 hidden size
            },
            TaskType::Regression { output_dim, .. } => {
                layers.push(Linear::new(768, *output_dim, true));
            },
            _ => {
                // Default head
                layers.push(Linear::new(768, 768, true));
            },
        }

        Ok(Self {
            layers,
            task_type: task_type.clone(),
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(output)?;
        }
        Ok(output)
    }
}

/// Training data batch for a specific task
#[derive(Debug, Clone)]
pub struct TaskBatch {
    pub inputs: Tensor,
    pub targets: Tensor,
    pub task_name: String,
}

/// Task scheduler state
pub struct TaskSchedulerState {
    pub current_task_index: usize,
    pub task_counters: HashMap<String, usize>,
}

impl TaskSchedulerState {
    pub fn new(_strategy: &TaskSchedulingStrategy) -> Self {
        Self {
            current_task_index: 0,
            task_counters: HashMap::new(),
        }
    }
}

/// Gradient statistics for task balancing
#[derive(Debug, Clone)]
pub struct GradientStats {
    pub gradient_norm: f32,
    pub gradient_variance: f32,
    pub update_count: usize,
}

/// Output from multi-task training step
#[derive(Debug, Clone)]
pub struct MultiTaskOutput {
    pub total_loss: Tensor,
    pub task_losses: HashMap<String, f32>,
    pub task_accuracies: HashMap<String, f32>,
    pub active_tasks: Vec<String>,
    pub task_weights: HashMap<String, f32>,
}

/// Task evaluation results
#[derive(Debug, Clone)]
pub struct TaskEvaluation {
    pub task_name: String,
    pub loss: f32,
    pub accuracy: f32,
    pub num_examples: usize,
}

/// Multi-task evaluation results
#[derive(Debug, Clone)]
pub struct MultiTaskEvaluation {
    pub task_evaluations: HashMap<String, TaskEvaluation>,
    pub overall_accuracy: f32,
    pub step: usize,
}

/// Multi-task learning statistics
#[derive(Debug, Clone)]
pub struct MTLStats {
    pub num_tasks: usize,
    pub task_weights: HashMap<String, f32>,
    pub step_counter: usize,
    pub architecture: MTLArchitecture,
    pub loss_balancing: LossBalancingStrategy,
}

/// Utilities for multi-task learning
pub mod utils {
    use super::*;

    /// Create a simple hard parameter sharing configuration
    pub fn hard_parameter_sharing_config(
        tasks: Vec<TaskConfig>,
        shared_layers: usize,
        task_specific_layers: usize,
    ) -> MTLConfig {
        MTLConfig {
            architecture: MTLArchitecture::HardParameterSharing {
                shared_layers,
                task_specific_layers,
            },
            tasks,
            ..Default::default()
        }
    }

    /// Create a soft parameter sharing configuration
    pub fn soft_parameter_sharing_config(
        tasks: Vec<TaskConfig>,
        regularization_weight: f32,
    ) -> MTLConfig {
        MTLConfig {
            architecture: MTLArchitecture::SoftParameterSharing {
                regularization_weight,
                regularization_type: RegularizationType::L2Regularization,
            },
            tasks,
            ..Default::default()
        }
    }

    /// Create a multi-gate mixture of experts configuration
    pub fn mmoe_config(tasks: Vec<TaskConfig>, num_experts: usize, expert_dim: usize) -> MTLConfig {
        MTLConfig {
            architecture: MTLArchitecture::MultiGateMixtureOfExperts {
                num_experts,
                expert_dim,
                num_gates: tasks.len(),
            },
            tasks,
            ..Default::default()
        }
    }

    /// Create task configuration for classification
    pub fn classification_task(name: &str, num_classes: usize) -> TaskConfig {
        TaskConfig::new(
            name,
            TaskType::Classification {
                num_classes,
                use_class_weights: false,
            },
        )
    }

    /// Create task configuration for regression
    pub fn regression_task(name: &str, output_dim: usize) -> TaskConfig {
        TaskConfig::new(
            name,
            TaskType::Regression {
                output_dim,
                loss_type: RegressionLossType::MSE,
            },
        )
    }

    /// Create auxiliary task configuration for MLM
    pub fn mlm_auxiliary_task(weight: f32) -> AuxiliaryTaskConfig {
        AuxiliaryTaskConfig {
            name: "mlm".to_string(),
            auxiliary_type: AuxiliaryType::MaskedLanguageModeling,
            weight,
            frequency: AuxiliaryTaskFrequency::EveryNSteps(10),
        }
    }

    /// Compute task similarity matrix
    pub fn compute_task_similarity(
        task_performances: &HashMap<String, Vec<f32>>,
    ) -> HashMap<(String, String), f32> {
        let mut similarities = HashMap::new();
        let tasks: Vec<String> = task_performances.keys().cloned().collect();

        for i in 0..tasks.len() {
            for j in i + 1..tasks.len() {
                let task1 = &tasks[i];
                let task2 = &tasks[j];

                if let (Some(perf1), Some(perf2)) =
                    (task_performances.get(task1), task_performances.get(task2))
                {
                    let similarity = compute_correlation(perf1, perf2);
                    similarities.insert((task1.clone(), task2.clone()), similarity);
                    similarities.insert((task2.clone(), task1.clone()), similarity);
                }
            }
        }

        similarities
    }

    /// Compute correlation between two performance sequences
    pub fn compute_correlation(seq1: &[f32], seq2: &[f32]) -> f32 {
        if seq1.len() != seq2.len() || seq1.is_empty() {
            return 0.0;
        }

        let n = seq1.len() as f32;
        let mean1 = seq1.iter().sum::<f32>() / n;
        let mean2 = seq2.iter().sum::<f32>() / n;

        let mut numerator = 0.0;
        let mut denom1 = 0.0;
        let mut denom2 = 0.0;

        for i in 0..seq1.len() {
            let diff1 = seq1[i] - mean1;
            let diff2 = seq2[i] - mean2;
            numerator += diff1 * diff2;
            denom1 += diff1 * diff1;
            denom2 += diff2 * diff2;
        }

        if denom1 * denom2 > 0.0 {
            numerator / (denom1 * denom2).sqrt()
        } else {
            0.0
        }
    }

    /// Analyze multi-task learning effectiveness
    pub fn analyze_mtl_effectiveness(
        single_task_performances: &HashMap<String, f32>,
        multi_task_performances: &HashMap<String, f32>,
    ) -> MTLAnalysis {
        let mut positive_transfer_tasks = Vec::new();
        let mut negative_transfer_tasks = Vec::new();
        let mut total_improvement = 0.0;
        let mut num_tasks = 0;

        for (task_name, &mtl_perf) in multi_task_performances {
            if let Some(&single_perf) = single_task_performances.get(task_name) {
                let improvement = mtl_perf - single_perf;
                total_improvement += improvement;
                num_tasks += 1;

                if improvement > 0.0 {
                    positive_transfer_tasks.push(task_name.clone());
                } else if improvement < 0.0 {
                    negative_transfer_tasks.push(task_name.clone());
                }
            }
        }

        let average_improvement =
            if num_tasks > 0 { total_improvement / num_tasks as f32 } else { 0.0 };

        MTLAnalysis {
            average_improvement,
            positive_transfer_tasks,
            negative_transfer_tasks,
            num_tasks,
        }
    }
}

/// Analysis of multi-task learning effectiveness
#[derive(Debug, Clone)]
pub struct MTLAnalysis {
    pub average_improvement: f32,
    pub positive_transfer_tasks: Vec<String>,
    pub negative_transfer_tasks: Vec<String>,
    pub num_tasks: usize,
}
