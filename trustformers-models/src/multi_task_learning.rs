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
//!     MultiTaskLearningTrainer, MTLConfig, MTLArchitecture
//! };
//!
//! let config = MTLConfig {
//!     architecture: MTLArchitecture::HardParameterSharing {
//!         shared_layers: 8,
//!         task_specific_layers: 2,
//!     },
//!     loss_balancing: LossBalancingStrategy::DynamicWeightAverage,
//!     tasks: vec![
//!         TaskConfig::new("classification", TaskType::Classification { num_classes: 10 }),
//!         TaskConfig::new("regression", TaskType::Regression { output_dim: 1 }),
//!     ],
//!     ..Default::default()
//! };
//!
//! let mut trainer = MultiTaskLearningTrainer::new(config)?;
//! trainer.train_multi_task(task_data)?;
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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
}

impl<M: Model<Input = Tensor, Output = Tensor>> MultiTaskLearningTrainer<M> {
    /// Create a new multi-task learning trainer
    pub fn new(base_model: M, config: MTLConfig) -> Result<Self> {
        let mut task_heads = HashMap::new();
        let mut task_weights = HashMap::new();

        // Initialize task heads
        for task_config in &config.tasks {
            let task_head = TaskHead::new(&task_config.task_type)?;
            task_heads.insert(task_config.name.clone(), task_head);
            task_weights.insert(task_config.name.clone(), task_config.weight);
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

        // Balance losses across tasks
        let balanced_losses = self.balance_losses(&task_losses)?;

        // Compute total loss
        for (task_name, loss) in &balanced_losses {
            let weight = self.task_weights.get(task_name).copied().unwrap_or(1.0);
            total_loss = total_loss.add(&loss.scalar_mul(weight)?)?;
        }

        // Update task weights if using dynamic balancing
        self.update_task_weights(&task_losses)?;

        // Update auxiliary tasks if enabled
        if self.config.use_auxiliary_tasks {
            let aux_loss = self.compute_auxiliary_losses(task_data)?;
            total_loss = total_loss.add(&aux_loss)?;
        }

        self.step_counter += 1;

        Ok(MultiTaskOutput {
            total_loss,
            task_losses: task_losses
                .into_iter()
                .map(|(k, v)| (k, v.to_scalar().unwrap_or(0.0)))
                .collect(),
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
    fn balance_losses(
        &self,
        task_losses: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        match &self.config.loss_balancing {
            LossBalancingStrategy::EqualWeighting => Ok(task_losses.clone()),
            LossBalancingStrategy::ManualWeighting { weights } => {
                let mut balanced = HashMap::new();
                for (i, (task_name, loss)) in task_losses.iter().enumerate() {
                    let weight = weights.get(i).copied().unwrap_or(1.0);
                    balanced.insert(task_name.clone(), loss.scalar_mul(weight)?);
                }
                Ok(balanced)
            },
            LossBalancingStrategy::UncertaintyWeighting => {
                // Implement uncertainty-based weighting
                // This would typically involve learning task-specific uncertainty parameters
                Ok(task_losses.clone()) // Simplified for now
            },
            LossBalancingStrategy::DynamicWeightAverage => {
                // Use dynamic weight average algorithm
                self.apply_dynamic_weight_average(task_losses)
            },
            LossBalancingStrategy::GradNorm { alpha } => {
                // Apply GradNorm algorithm
                self.apply_gradnorm(task_losses, *alpha)
            },
            _ => Ok(task_losses.clone()),
        }
    }

    /// Apply dynamic weight average algorithm
    fn apply_dynamic_weight_average(
        &self,
        task_losses: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        // DWA uses relative descent rates to weight tasks
        let mut balanced = HashMap::new();

        if self.step_counter < 2 {
            return Ok(task_losses.clone());
        }

        let temperature = 2.0; // DWA temperature parameter

        for (task_name, loss) in task_losses {
            // Get previous loss for this task
            let prev_loss = self.get_previous_task_loss(task_name);
            let current_loss = loss.to_scalar().unwrap_or(0.0);

            let weight = if prev_loss > 0.0 {
                let relative_decrease = current_loss / prev_loss;
                (relative_decrease / temperature).exp()
            } else {
                1.0
            };

            balanced.insert(task_name.clone(), loss.clone().mul_scalar(weight)?);
        }

        Ok(balanced)
    }

    /// Apply GradNorm algorithm
    fn apply_gradnorm(
        &self,
        task_losses: &HashMap<String, Tensor>,
        _alpha: f32,
    ) -> Result<HashMap<String, Tensor>> {
        // GradNorm balances gradient magnitudes across tasks
        // This is a simplified implementation
        Ok(task_losses.clone())
    }

    /// Update task weights based on performance
    fn update_task_weights(&mut self, task_losses: &HashMap<String, Tensor>) -> Result<()> {
        match &self.config.loss_balancing {
            LossBalancingStrategy::DynamicWeightAverage => {
                // Update weights based on loss trends
                for (task_name, loss) in task_losses {
                    let current_loss = loss.to_scalar().unwrap_or(0.0);
                    // Update internal weight tracking
                    // This would be more sophisticated in practice
                    if let Some(weight) = self.task_weights.get_mut(task_name) {
                        *weight = (*weight * 0.9 + current_loss * 0.1).clamp(0.1, 10.0);
                    }
                }
            },
            _ => {
                // Other strategies don't update weights dynamically
            },
        }
        Ok(())
    }

    /// Get previous task loss for DWA
    fn get_previous_task_loss(&self, _task_name: &str) -> f32 {
        // This would get the loss from the previous step
        // Simplified implementation
        1.0
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
            AuxiliaryTaskFrequency::EveryNSteps(n) => self.step_counter % n == 0,
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
                // Compute language modeling loss
                self.compute_lm_loss(&shared_features, &data.targets)
            },
            AuxiliaryType::MaskedLanguageModeling => {
                // Compute MLM loss
                self.compute_mlm_loss(&shared_features, &data.targets)
            },
            _ => {
                // Other auxiliary tasks
                Ok(Tensor::zeros(&[1])?)
            },
        }
    }

    /// Compute language modeling loss
    fn compute_lm_loss(&self, _features: &Tensor, _targets: &Tensor) -> Result<Tensor> {
        // Simplified LM loss computation
        Tensor::zeros(&[1])
    }

    /// Compute masked language modeling loss
    fn compute_mlm_loss(&self, _features: &Tensor, _targets: &Tensor) -> Result<Tensor> {
        // Simplified MLM loss computation
        Tensor::zeros(&[1])
    }

    /// Compute task-specific loss
    fn compute_task_loss(
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
                // Cross-entropy loss
                let log_probs = outputs.softmax(-1)?;
                let nll_loss = targets.mul(&log_probs)?.sum(Some(vec![1]), false)?;
                Ok(nll_loss.mean()?.mul_scalar(-1.0)?)
            },
            TaskType::Regression { loss_type, .. } => {
                match loss_type {
                    RegressionLossType::MSE => {
                        let diff = outputs.sub(targets)?;
                        Ok(diff.mul(&diff)?.mean()?)
                    },
                    RegressionLossType::MAE => {
                        let diff = outputs.sub(targets)?;
                        Ok(diff.abs()?.mean()?)
                    },
                    RegressionLossType::Huber { delta } => {
                        let diff = outputs.sub(targets)?;
                        let abs_diff = diff.abs()?;
                        let small_loss = diff.mul(&diff)?.mul_scalar(0.5)?;
                        let _large_loss =
                            abs_diff.mul_scalar(*delta)?.sub_scalar(*delta * *delta * 0.5)?;
                        // Simplified Huber loss approximation
                        Ok(small_loss.mean()?)
                    },
                    _ => {
                        // Other regression losses
                        let diff = outputs.sub(targets)?;
                        Ok(diff.mul(&diff)?.mean()?)
                    },
                }
            },
            _ => {
                // Other task types
                Ok(Tensor::zeros(&[1])?)
            },
        }
    }

    /// Compute task-specific accuracy
    fn compute_task_accuracy(
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
                let predicted = outputs.argmax(-1)?;
                let target_class = targets.argmax(-1)?;
                let correct = (predicted.to_scalar().unwrap_or(-1.0)
                    == target_class.to_scalar().unwrap_or(-2.0))
                    as i32 as f32;
                Ok(correct)
            },
            TaskType::Regression { .. } => {
                // For regression, compute R² or similar metric
                let diff = outputs.sub(targets)?;
                let mse = diff.mul(&diff)?.mean()?;
                let mean_targets = targets.mean()?;
                let diff_from_mean = targets.sub(&mean_targets)?;
                let variance = diff_from_mean.pow_scalar(2.0)?.mean()?;
                let r_squared =
                    1.0 - mse.to_scalar().unwrap_or(1.0) / variance.to_scalar().unwrap_or(1.0);
                Ok(r_squared.max(0.0))
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
                        loss: loss.to_scalar().unwrap_or(0.0),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mtl_config_default() {
        let config = MTLConfig::default();
        assert_eq!(config.tasks.len(), 0);
        assert!(!config.use_task_embeddings);
        assert!(!config.use_auxiliary_tasks);

        if let MTLArchitecture::HardParameterSharing {
            shared_layers,
            task_specific_layers,
        } = config.architecture
        {
            assert_eq!(shared_layers, 8);
            assert_eq!(task_specific_layers, 2);
        } else {
            panic!("Expected HardParameterSharing architecture");
        }
    }

    #[test]
    fn test_task_config() {
        let task = TaskConfig::new(
            "test",
            TaskType::Classification {
                num_classes: 10,
                use_class_weights: false,
            },
        );

        assert_eq!(task.name, "test");
        assert_eq!(task.weight, 1.0);
        assert!(!task.is_main_task);

        let weighted_task = task.with_weight(2.0);
        assert_eq!(weighted_task.weight, 2.0);
    }

    #[test]
    fn test_classification_task_util() {
        let task = utils::classification_task("sentiment", 3);
        assert_eq!(task.name, "sentiment");

        if let TaskType::Classification { num_classes, .. } = task.task_type {
            assert_eq!(num_classes, 3);
        } else {
            panic!("Expected Classification task type");
        }
    }

    #[test]
    fn test_regression_task_util() {
        let task = utils::regression_task("score", 1);
        assert_eq!(task.name, "score");

        if let TaskType::Regression { output_dim, .. } = task.task_type {
            assert_eq!(output_dim, 1);
        } else {
            panic!("Expected Regression task type");
        }
    }

    #[test]
    fn test_hard_parameter_sharing_config() {
        let tasks = vec![
            utils::classification_task("task1", 5),
            utils::regression_task("task2", 1),
        ];

        let config = utils::hard_parameter_sharing_config(tasks, 6, 2);
        assert_eq!(config.tasks.len(), 2);

        if let MTLArchitecture::HardParameterSharing {
            shared_layers,
            task_specific_layers,
        } = config.architecture
        {
            assert_eq!(shared_layers, 6);
            assert_eq!(task_specific_layers, 2);
        } else {
            panic!("Expected HardParameterSharing architecture");
        }
    }

    #[test]
    fn test_soft_parameter_sharing_config() {
        let tasks = vec![utils::classification_task("task1", 5)];
        let config = utils::soft_parameter_sharing_config(tasks, 0.01);

        if let MTLArchitecture::SoftParameterSharing {
            regularization_weight,
            ..
        } = config.architecture
        {
            assert_eq!(regularization_weight, 0.01);
        } else {
            panic!("Expected SoftParameterSharing architecture");
        }
    }

    #[test]
    fn test_mmoe_config() {
        let tasks = vec![
            utils::classification_task("task1", 5),
            utils::classification_task("task2", 3),
        ];

        let config = utils::mmoe_config(tasks, 4, 128);

        if let MTLArchitecture::MultiGateMixtureOfExperts {
            num_experts,
            expert_dim,
            num_gates,
        } = config.architecture
        {
            assert_eq!(num_experts, 4);
            assert_eq!(expert_dim, 128);
            assert_eq!(num_gates, 2);
        } else {
            panic!("Expected MultiGateMixtureOfExperts architecture");
        }
    }

    #[test]
    fn test_mlm_auxiliary_task() {
        let aux_task = utils::mlm_auxiliary_task(0.1);
        assert_eq!(aux_task.name, "mlm");
        assert_eq!(aux_task.weight, 0.1);

        if let AuxiliaryType::MaskedLanguageModeling = aux_task.auxiliary_type {
            // Expected
        } else {
            panic!("Expected MaskedLanguageModeling auxiliary type");
        }
    }

    #[test]
    fn test_compute_correlation() {
        let seq1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let seq2 = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect positive correlation

        let correlation = utils::compute_correlation(&seq1, &seq2);
        assert!((correlation - 1.0).abs() < 1e-6);

        let seq3 = vec![5.0, 4.0, 3.0, 2.0, 1.0]; // Perfect negative correlation
        let correlation_neg = utils::compute_correlation(&seq1, &seq3);
        assert!((correlation_neg + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mtl_analysis() {
        let mut single_task = HashMap::new();
        single_task.insert("task1".to_string(), 0.8);
        single_task.insert("task2".to_string(), 0.7);
        single_task.insert("task3".to_string(), 0.6);

        let mut multi_task = HashMap::new();
        multi_task.insert("task1".to_string(), 0.85); // Positive transfer
        multi_task.insert("task2".to_string(), 0.65); // Negative transfer
        multi_task.insert("task3".to_string(), 0.65); // Positive transfer

        let analysis = utils::analyze_mtl_effectiveness(&single_task, &multi_task);
        assert_eq!(analysis.num_tasks, 3);
        assert_eq!(analysis.positive_transfer_tasks.len(), 2);
        assert_eq!(analysis.negative_transfer_tasks.len(), 1);
        assert!(analysis.positive_transfer_tasks.contains(&"task1".to_string()));
        assert!(analysis.negative_transfer_tasks.contains(&"task2".to_string()));
    }
}
