//! # Continual Learning Framework
//!
//! This module provides a comprehensive framework for continual learning,
//! enabling models to learn new tasks while retaining knowledge from previous tasks.
//!
//! ## Features
//!
//! - **Multiple Continual Learning Strategies**: EWC, PackNet, Progressive Networks, etc.
//! - **Catastrophic Forgetting Prevention**: Various regularization techniques
//! - **Memory Management**: Experience replay and memory-based approaches
//! - **Task Detection**: Automatic task boundary detection
//! - **Evaluation Metrics**: Specialized metrics for continual learning scenarios
//! - **Multi-task Support**: Learning multiple tasks simultaneously
//!
//! ## Usage
//!
//! Parameter-space strategies (EWC, L2, LwF, GEM, PackNet) additionally
//! require the model to implement [`NamedParameters`], which is how the
//! trainer reads and writes the parameters it regularizes.
//!
//! ```rust,no_run
//! use trustformers_models::continual_learning::{
//!     ContinualLearningTrainer, ContinualLearningConfig, ContinualStrategy, NamedParameters
//! };
//! use trustformers_core::{traits::{Config, Model}, tensor::Tensor, Result};
//! use serde::{Deserialize, Serialize};
//!
//! # #[derive(Debug, Clone, Serialize, Deserialize)]
//! # struct DocConfig;
//! # impl Config for DocConfig {
//! #     fn architecture(&self) -> &'static str { "doc" }
//! # }
//! # struct DocModel { weight: Tensor }
//! # impl Model for DocModel {
//! #     type Config = DocConfig;
//! #     type Input = Tensor;
//! #     type Output = Tensor;
//! #     fn forward(&self, input: Tensor) -> Result<Tensor> { input.mul(&self.weight) }
//! #     fn load_pretrained(&mut self, _r: &mut dyn std::io::Read) -> Result<()> { Ok(()) }
//! #     fn get_config(&self) -> &DocConfig { &DocConfig }
//! #     fn num_parameters(&self) -> usize { 4 }
//! # }
//! # impl NamedParameters for DocModel {
//! #     fn named_parameters(&self) -> Vec<(String, Tensor)> {
//! #         vec![("weight".to_string(), self.weight.clone())]
//! #     }
//! #     fn set_named_parameter(&mut self, name: &str, value: Tensor) -> Result<()> {
//! #         if name == "weight" { self.weight = value; }
//! #         Ok(())
//! #     }
//! # }
//!
//! # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
//! let config = ContinualLearningConfig {
//!     strategy: ContinualStrategy::ElasticWeightConsolidation {
//!         lambda: 0.4,
//!         fisher_samples: 1000,
//!     },
//!     memory_size: 1000,
//!     ..Default::default()
//! };
//!
//! # let model = DocModel { weight: Tensor::ones(&[1, 4])? };
//! let mut trainer = ContinualLearningTrainer::new(model, config)?;
//!
//! // Learn task 1 (inputs/targets must share shape and be non-empty)
//! # let inputs = [Tensor::zeros(&[1, 4])?];
//! # let targets = [Tensor::zeros(&[1, 4])?];
//! trainer.learn_batch(&inputs, &targets, Some(0))?;
//! // Learn task 2 without forgetting task 1
//! trainer.learn_batch(&inputs, &targets, Some(1))?;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::{
    errors::{invalid_input, not_implemented},
    tensor::Tensor,
    traits::Model,
    Result,
};

/// Step size used when a model does not supply analytic gradients and the
/// trainer has to estimate them with central finite differences.
pub const FINITE_DIFFERENCE_EPSILON: f32 = 1e-3;

/// Access to a model's named parameter tensors.
///
/// Parameter-space continual-learning strategies (EWC, L2, LwF, GEM, PackNet)
/// cannot work through the opaque [`Model`] trait alone: they need to read the
/// current parameter values, to write perturbed values, and — for Fisher
/// information — to differentiate the loss with respect to them. Implementing
/// this trait is what makes those strategies available.
///
/// A model that can differentiate itself should override
/// [`NamedParameters::parameter_gradients`]; otherwise the trainer falls back
/// to central finite differences, which costs `2 · P` forward passes per
/// example for `P` scalar parameters.
pub trait NamedParameters {
    /// Every trainable parameter tensor together with a stable name.
    fn named_parameters(&self) -> Vec<(String, Tensor)>;

    /// Overwrite the parameter tensor called `name`.
    ///
    /// Returns an error for an unknown name or a shape mismatch.
    fn set_named_parameter(&mut self, name: &str, value: Tensor) -> Result<()>;

    /// Analytic gradients of the task loss with respect to every named
    /// parameter, when the model can compute them.
    ///
    /// Returning `Ok(None)` (the default) tells the trainer to fall back to
    /// finite differences.
    fn parameter_gradients(
        &self,
        _input: &Tensor,
        _target: &Tensor,
    ) -> Result<Option<HashMap<String, Tensor>>> {
        Ok(None)
    }
}

/// Wrap a scalar as a `[1]`-shaped tensor so every loss term has one shape.
fn scalar_tensor(value: f32) -> Result<Tensor> {
    Tensor::from_vec(vec![value], &[1])
}

/// Read a one-element tensor as an `f32`.
fn scalar_value(tensor: &Tensor) -> Result<f32> {
    let data = tensor.to_vec_f32()?;
    data.first()
        .copied()
        .ok_or_else(|| invalid_input("expected a non-empty loss tensor"))
}

/// Index of the largest element of a slice.
fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

/// Configuration for continual learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinualLearningConfig {
    /// Continual learning strategy to use
    pub strategy: ContinualStrategy,
    /// Size of memory buffer for experience replay
    pub memory_size: usize,
    /// Memory selection strategy
    pub memory_selection: MemorySelectionStrategy,
    /// Whether to use task-specific heads
    pub task_specific_heads: bool,
    /// Number of tasks to prepare for
    pub max_tasks: usize,
    /// Learning rate schedule for continual learning
    pub learning_rate_schedule: LearningRateSchedule,
    /// Evaluation frequency (in training steps)
    pub evaluation_frequency: usize,
    /// Whether to use task detection
    pub automatic_task_detection: bool,
    /// Task detection threshold
    pub task_detection_threshold: f32,
}

impl Default for ContinualLearningConfig {
    fn default() -> Self {
        Self {
            strategy: ContinualStrategy::ElasticWeightConsolidation {
                lambda: 0.4,
                fisher_samples: 1000,
            },
            memory_size: 1000,
            memory_selection: MemorySelectionStrategy::Random,
            task_specific_heads: true,
            max_tasks: 10,
            learning_rate_schedule: LearningRateSchedule::Constant { lr: 1e-4 },
            evaluation_frequency: 1000,
            automatic_task_detection: false,
            task_detection_threshold: 0.8,
        }
    }
}

/// Different continual learning strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContinualStrategy {
    /// Elastic Weight Consolidation (EWC)
    ElasticWeightConsolidation { lambda: f32, fisher_samples: usize },
    /// Online EWC
    OnlineElasticWeightConsolidation {
        lambda: f32,
        gamma: f32,
        fisher_samples: usize,
    },
    /// Synaptic Intelligence (SI)
    SynapticIntelligence { c: f32, xi: f32 },
    /// Learning without Forgetting (LwF)
    LearningWithoutForgetting { lambda: f32, temperature: f32 },
    /// Progressive Neural Networks
    ProgressiveNeuralNetworks {
        lateral_connections: bool,
        adapter_layers: bool,
    },
    /// PackNet
    PackNet {
        prune_ratio: f32,
        retrain_epochs: usize,
    },
    /// Experience Replay
    ExperienceReplay {
        memory_strength: f32,
        replay_batch_size: usize,
    },
    /// Gradient Episodic Memory (GEM)
    GradientEpisodicMemory {
        memory_strength: f32,
        constraint_violation_threshold: f32,
    },
    /// Averaged Gradient Episodic Memory (A-GEM)
    AveragedGradientEpisodicMemory {
        memory_strength: f32,
        replay_batch_size: usize,
    },
    /// Meta-Experience Replay (MER)
    MetaExperienceReplay {
        beta: f32,
        gamma: f32,
        replay_steps: usize,
    },
    /// L2 Regularization (simple baseline)
    L2Regularization { lambda: f32 },
    /// Dropout-based approaches
    VariationalContinualLearning {
        kl_weight: f32,
        prior_precision: f32,
    },
}

/// Memory selection strategies for experience replay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemorySelectionStrategy {
    /// Random selection
    Random,
    /// Select most uncertain examples
    Uncertainty,
    /// Select most diverse examples
    Diversity,
    /// Gradient-based selection
    Gradient,
    /// Select examples with highest loss
    HighestLoss,
    /// Cluster-based selection
    ClusterBased,
    /// FIFO (First In, First Out)
    FIFO,
    /// Ring buffer
    RingBuffer,
}

/// Learning rate scheduling for continual learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningRateSchedule {
    /// Constant learning rate
    Constant { lr: f32 },
    /// Exponential decay
    ExponentialDecay { initial_lr: f32, decay_rate: f32 },
    /// Step decay
    StepDecay {
        initial_lr: f32,
        step_size: usize,
        gamma: f32,
    },
    /// Cosine annealing
    CosineAnnealing { initial_lr: f32, t_max: usize },
    /// Warm restart
    WarmRestart {
        initial_lr: f32,
        t_0: usize,
        t_mult: usize,
    },
}

/// Memory buffer for storing past experiences
#[derive(Debug, Clone)]
pub struct MemoryBuffer {
    /// Stored examples (inputs)
    pub inputs: Vec<Tensor>,
    /// Stored targets
    pub targets: Vec<Tensor>,
    /// Task IDs for each example
    pub task_ids: Vec<usize>,
    /// Example priorities/weights
    pub priorities: Vec<f32>,
    /// Maximum buffer size
    pub max_size: usize,
    /// Current insertion pointer
    pub insertion_ptr: usize,
    /// Selection strategy
    pub selection_strategy: MemorySelectionStrategy,
}

impl MemoryBuffer {
    /// Create a new memory buffer
    pub fn new(max_size: usize, selection_strategy: MemorySelectionStrategy) -> Self {
        Self {
            inputs: Vec::new(),
            targets: Vec::new(),
            task_ids: Vec::new(),
            priorities: Vec::new(),
            max_size,
            insertion_ptr: 0,
            selection_strategy,
        }
    }

    /// Add a new example to the buffer
    pub fn add_example(&mut self, input: Tensor, target: Tensor, task_id: usize, priority: f32) {
        if self.inputs.len() < self.max_size {
            // Buffer not full, just append
            self.inputs.push(input);
            self.targets.push(target);
            self.task_ids.push(task_id);
            self.priorities.push(priority);
        } else {
            // Buffer full, need to replace
            match self.selection_strategy {
                MemorySelectionStrategy::Random => {
                    let idx = fastrand::usize(..self.max_size);
                    self.inputs[idx] = input;
                    self.targets[idx] = target;
                    self.task_ids[idx] = task_id;
                    self.priorities[idx] = priority;
                },
                MemorySelectionStrategy::FIFO | MemorySelectionStrategy::RingBuffer => {
                    self.inputs[self.insertion_ptr] = input;
                    self.targets[self.insertion_ptr] = target;
                    self.task_ids[self.insertion_ptr] = task_id;
                    self.priorities[self.insertion_ptr] = priority;
                    self.insertion_ptr = (self.insertion_ptr + 1) % self.max_size;
                },
                _ => {
                    // For other strategies, replace the least important example
                    let min_idx = self
                        .priorities
                        .iter()
                        .enumerate()
                        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);

                    if priority > self.priorities[min_idx] {
                        self.inputs[min_idx] = input;
                        self.targets[min_idx] = target;
                        self.task_ids[min_idx] = task_id;
                        self.priorities[min_idx] = priority;
                    }
                },
            }
        }
    }

    /// Sample a batch from the buffer
    pub fn sample_batch(
        &self,
        batch_size: usize,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>, Vec<usize>)> {
        if self.inputs.is_empty() {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        let sample_size = batch_size.min(self.inputs.len());
        let mut indices = Vec::new();

        match self.selection_strategy {
            MemorySelectionStrategy::Random => {
                for _ in 0..sample_size {
                    indices.push(fastrand::usize(..self.inputs.len()));
                }
            },
            _ => {
                // For other strategies, sample proportional to priority
                let total_priority: f32 = self.priorities.iter().sum();
                for _ in 0..sample_size {
                    let mut cumsum = 0.0;
                    let threshold = fastrand::f32() * total_priority;
                    for (i, &priority) in self.priorities.iter().enumerate() {
                        cumsum += priority;
                        if cumsum >= threshold {
                            indices.push(i);
                            break;
                        }
                    }
                }
            },
        }

        let inputs: Vec<Tensor> = indices.iter().map(|&i| self.inputs[i].clone()).collect();
        let targets: Vec<Tensor> = indices.iter().map(|&i| self.targets[i].clone()).collect();
        let task_ids: Vec<usize> = indices.iter().map(|&i| self.task_ids[i]).collect();

        Ok((inputs, targets, task_ids))
    }

    /// Get examples from a specific task
    pub fn get_task_examples(&self, task_id: usize) -> (Vec<Tensor>, Vec<Tensor>) {
        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        for (i, &tid) in self.task_ids.iter().enumerate() {
            if tid == task_id {
                inputs.push(self.inputs[i].clone());
                targets.push(self.targets[i].clone());
            }
        }

        (inputs, targets)
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.inputs.clear();
        self.targets.clear();
        self.task_ids.clear();
        self.priorities.clear();
        self.insertion_ptr = 0;
    }

    /// Get buffer size
    pub fn size(&self) -> usize {
        self.inputs.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }
}

/// Continual learning trainer
pub struct ContinualLearningTrainer<M: Model> {
    /// The model being trained
    pub model: M,
    /// Configuration
    pub config: ContinualLearningConfig,
    /// Memory buffer for experience replay
    pub memory: MemoryBuffer,
    /// Task-specific information
    pub task_info: HashMap<usize, TaskInfo>,
    /// Current task ID
    pub current_task: Option<usize>,
    /// Fisher information matrices (for EWC)
    pub fisher_matrices: HashMap<String, Tensor>,
    /// Optimal parameters (for EWC)
    pub optimal_parameters: HashMap<String, Tensor>,
    /// Training step counter
    pub step_counter: usize,
    /// Task detection state
    pub task_detector: Option<TaskDetector>,
    /// PackNet capacity masks: `true` marks a weight still owned by this task
    pub packnet_masks: HashMap<String, Vec<bool>>,
}

impl<M: Model<Input = Tensor, Output = Tensor> + NamedParameters> ContinualLearningTrainer<M> {
    /// Create a new continual learning trainer
    pub fn new(model: M, config: ContinualLearningConfig) -> Result<Self> {
        let memory = MemoryBuffer::new(config.memory_size, config.memory_selection.clone());

        let task_detector = if config.automatic_task_detection {
            Some(TaskDetector::new(config.task_detection_threshold))
        } else {
            None
        };

        Ok(Self {
            model,
            config,
            memory,
            task_info: HashMap::new(),
            current_task: None,
            fisher_matrices: HashMap::new(),
            optimal_parameters: HashMap::new(),
            step_counter: 0,
            task_detector,
            packnet_masks: HashMap::new(),
        })
    }

    /// Task loss for one example, as a plain scalar.
    fn example_loss(&self, input: &Tensor, target: &Tensor) -> Result<f32> {
        let outputs = self.model.forward(input.clone())?;
        scalar_value(&self.compute_task_loss(&outputs, target)?)
    }

    /// Gradients of the task loss with respect to every named parameter.
    ///
    /// Uses [`NamedParameters::parameter_gradients`] when the model provides
    /// analytic gradients, and otherwise estimates them with central finite
    /// differences. The model's parameters are restored exactly on return.
    pub fn loss_gradients(
        &mut self,
        input: &Tensor,
        target: &Tensor,
    ) -> Result<HashMap<String, Tensor>> {
        if let Some(gradients) = self.model.parameter_gradients(input, target)? {
            return Ok(gradients);
        }

        let epsilon = FINITE_DIFFERENCE_EPSILON;
        let parameters = self.model.named_parameters();
        let mut gradients = HashMap::new();

        for (name, tensor) in parameters {
            let shape = tensor.shape();
            let baseline = tensor.to_vec_f32()?;
            let mut gradient = vec![0.0f32; baseline.len()];

            for index in 0..baseline.len() {
                let mut plus = baseline.clone();
                plus[index] += epsilon;
                self.model.set_named_parameter(&name, Tensor::from_vec(plus, &shape)?)?;
                let loss_plus = self.example_loss(input, target)?;

                let mut minus = baseline.clone();
                minus[index] -= epsilon;
                self.model.set_named_parameter(&name, Tensor::from_vec(minus, &shape)?)?;
                let loss_minus = self.example_loss(input, target)?;

                gradient[index] = (loss_plus - loss_minus) / (2.0 * epsilon);
            }

            // Restore the original values before moving on.
            self.model.set_named_parameter(&name, Tensor::from_vec(baseline, &shape)?)?;
            gradients.insert(name, Tensor::from_vec(gradient, &shape)?);
        }

        Ok(gradients)
    }

    /// Start learning a new task
    pub fn start_task(&mut self, task_id: usize) -> Result<()> {
        // Save current task information if this is a task switch
        if let Some(current_id) = self.current_task {
            if current_id != task_id {
                self.finalize_task(current_id)?;
            }
        }

        self.current_task = Some(task_id);

        // Initialize task info if new
        self.task_info.entry(task_id).or_insert_with(|| TaskInfo::new(task_id));

        // Apply strategy-specific initialization
        match &self.config.strategy {
            ContinualStrategy::ProgressiveNeuralNetworks { .. } => {
                // Add new columns for progressive networks
                self.add_progressive_columns(task_id)?;
            },
            ContinualStrategy::PackNet { .. } => {
                // Prepare for pruning-based learning
                self.prepare_packnet(task_id)?;
            },
            _ => {
                // Most strategies don't require special initialization
            },
        }

        Ok(())
    }

    /// Learn from a batch of data
    pub fn learn_batch(
        &mut self,
        inputs: &[Tensor],
        targets: &[Tensor],
        task_id: Option<usize>,
    ) -> Result<ContinualLearningOutput> {
        let task_id = task_id
            .or(self.current_task)
            .ok_or_else(|| invalid_input("No task ID specified"))?;

        // Detect task boundaries if enabled
        if let Some(detector) = &mut self.task_detector {
            if let Some(detected_task) = detector.detect_task_change(inputs, targets)? {
                if detected_task != task_id {
                    self.start_task(detected_task)?;
                }
            }
        }

        if inputs.is_empty() || targets.is_empty() {
            return Err(invalid_input("learn_batch requires at least one example"));
        }

        // Compute forward pass and loss
        let outputs = self.model.forward(inputs[0].clone())?; // Simplified single input
        let current_loss = self.compute_task_loss(&outputs, &targets[0])?;
        let current_loss_for_output = current_loss.clone();

        // Apply continual learning strategy
        let strategy = self.config.strategy.clone();
        let total_loss = match &strategy {
            ContinualStrategy::ElasticWeightConsolidation { lambda, .. }
            | ContinualStrategy::OnlineElasticWeightConsolidation { lambda, .. } => {
                let ewc_loss = self.compute_ewc_loss(*lambda)?;
                current_loss.add(&ewc_loss)?
            },
            ContinualStrategy::LearningWithoutForgetting {
                lambda,
                temperature,
            } => {
                let distillation_loss = self.compute_lwf_loss(inputs, *lambda, *temperature)?;
                current_loss.add(&distillation_loss)?
            },
            ContinualStrategy::ExperienceReplay {
                memory_strength,
                replay_batch_size,
            } => {
                let replay_loss = self.compute_replay_loss(*memory_strength, *replay_batch_size)?;
                current_loss.add(&replay_loss)?
            },
            ContinualStrategy::GradientEpisodicMemory {
                memory_strength,
                constraint_violation_threshold,
            } => self.compute_gem_loss(
                &inputs[0],
                &targets[0],
                &current_loss,
                *memory_strength,
                *constraint_violation_threshold,
            )?,
            ContinualStrategy::L2Regularization { lambda } => {
                let l2_loss = self.compute_l2_regularization(*lambda)?;
                current_loss.add(&l2_loss)?
            },
            _ => current_loss,
        };

        // Store examples in memory if needed
        if !matches!(
            self.config.strategy,
            ContinualStrategy::L2Regularization { .. }
        ) {
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let priority = self.compute_example_priority(input, target)?;
                self.memory.add_example(input.clone(), target.clone(), task_id, priority);
            }
        }

        // Update training step counter
        self.step_counter += 1;

        // Update task statistics
        let total_loss_value = scalar_value(&total_loss)?;
        if let Some(task_info) = self.task_info.get_mut(&task_id) {
            task_info.update_statistics(total_loss_value);
        }

        let total_loss_clone = total_loss.clone();

        Ok(ContinualLearningOutput {
            total_loss: total_loss_clone.clone(),
            task_loss: current_loss_for_output.clone(),
            regularization_loss: total_loss_clone.sub(&current_loss_for_output)?,
            task_id,
            memory_usage: self.memory.size(),
        })
    }

    /// Finalize learning for a task
    pub fn finalize_task(&mut self, task_id: usize) -> Result<()> {
        match self.config.strategy.clone() {
            ContinualStrategy::ElasticWeightConsolidation { fisher_samples, .. } => {
                self.compute_fisher_information(task_id, fisher_samples)?;
                self.save_optimal_parameters()?;
            },
            ContinualStrategy::OnlineElasticWeightConsolidation {
                gamma,
                fisher_samples,
                ..
            } => {
                // Online EWC keeps a single running Fisher estimate that is
                // decayed by `gamma` before each task's contribution is added.
                self.accumulate_fisher_information(task_id, fisher_samples, gamma)?;
                self.save_optimal_parameters()?;
            },
            ContinualStrategy::PackNet { prune_ratio, .. } => {
                // Pruning and mask bookkeeping happen here; retraining the
                // surviving weights belongs to the caller's optimizer loop,
                // which this trainer does not own.
                self.apply_packnet_pruning(prune_ratio)?;
            },
            _ => {
                // Most strategies don't require finalization
            },
        }

        Ok(())
    }

    /// Compute task-specific loss
    fn compute_task_loss(&self, outputs: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // Implement cross-entropy loss for classification tasks
        let log_probs = outputs.softmax(-1)?.log()?;

        // Check if targets are one-hot encoded or class indices
        let targets_shape = targets.shape();
        let outputs_shape = outputs.shape();

        if targets_shape == outputs_shape {
            // Targets are one-hot encoded
            let element_wise = log_probs.mul(targets)?;
            let sum_per_sample = element_wise.sum(Some(vec![outputs_shape.len() - 1]), false)?; // Sum across the last dimension
            sum_per_sample.neg()?.mean()?.reshape(&[1])
        } else {
            // Targets are class indices - use simplified approach
            // In a full implementation, we'd use proper gather operation
            // For now, compute difference between predictions and one-hot targets
            let batch_size = outputs_shape[0];
            let num_classes = outputs_shape[outputs_shape.len() - 1];

            // Create one-hot encoding manually (simplified)
            let mut one_hot_data = vec![0.0f32; batch_size * num_classes];
            let targets_data = targets.data()?;

            for (i, &target_idx) in targets_data.iter().enumerate() {
                if target_idx >= 0.0 && (target_idx as usize) < num_classes {
                    one_hot_data[i * num_classes + target_idx as usize] = 1.0;
                }
            }

            let one_hot_targets = Tensor::new(one_hot_data)?.reshape(&outputs_shape)?;
            let element_wise = log_probs.mul(&one_hot_targets)?;
            let sum_per_sample = element_wise.sum(Some(vec![outputs_shape.len() - 1]), false)?; // Sum across the last dimension
            sum_per_sample.neg()?.mean()?.reshape(&[1])
        }
    }

    /// Elastic Weight Consolidation penalty `λ/2 · Σ_i F_i (θ_i − θ*_i)²`.
    ///
    /// `F` is the diagonal Fisher information accumulated by
    /// [`Self::compute_fisher_information`] and `θ*` the parameter snapshot
    /// saved by [`Self::save_optimal_parameters`]; both are read from the live
    /// model, so the penalty is zero at the snapshot and grows as the
    /// parameters drift away from it.
    pub fn compute_ewc_loss(&self, lambda: f32) -> Result<Tensor> {
        let mut total = 0.0f32;

        for (name, current) in self.model.named_parameters() {
            let (Some(fisher), Some(optimal)) = (
                self.fisher_matrices.get(&name),
                self.optimal_parameters.get(&name),
            ) else {
                continue;
            };

            let current_values = current.to_vec_f32()?;
            let fisher_values = fisher.to_vec_f32()?;
            let optimal_values = optimal.to_vec_f32()?;
            if current_values.len() != fisher_values.len()
                || current_values.len() != optimal_values.len()
            {
                return Err(invalid_input(format!(
                    "EWC state for parameter '{}' has a mismatched length",
                    name
                )));
            }

            for ((value, fisher), optimal) in
                current_values.iter().zip(fisher_values.iter()).zip(optimal_values.iter())
            {
                let delta = value - optimal;
                total += fisher * delta * delta;
            }
        }

        scalar_tensor(0.5 * lambda * total)
    }

    /// Learning-without-Forgetting distillation loss.
    ///
    /// The previous task's parameter snapshot is temporarily swapped into the
    /// model to obtain the old logits, the current logits are recomputed, and
    /// the returned term is
    /// `λ · T² · KL(softmax(z_old/T) ‖ softmax(z_new/T))`. Without a snapshot
    /// (i.e. before the first task has been finalized) there is nothing to
    /// distill and the term is zero.
    pub fn compute_lwf_loss(
        &mut self,
        inputs: &[Tensor],
        lambda: f32,
        temperature: f32,
    ) -> Result<Tensor> {
        if inputs.is_empty() || self.optimal_parameters.is_empty() {
            return scalar_tensor(0.0);
        }

        let temperature = temperature.abs().max(1e-6);
        let current_parameters = self.model.named_parameters();

        // Swap in the previous task's parameters to obtain the teacher logits.
        for (name, value) in &self.optimal_parameters {
            self.model.set_named_parameter(name, value.clone())?;
        }
        let teacher_logits = self.model.forward(inputs[0].clone());

        // Restore the live parameters before propagating any error.
        for (name, value) in &current_parameters {
            self.model.set_named_parameter(name, value.clone())?;
        }
        let teacher_logits = teacher_logits?;
        let student_logits = self.model.forward(inputs[0].clone())?;

        let teacher_log_probs = teacher_logits.div_scalar(temperature)?.log_softmax(-1)?;
        let student_log_probs = student_logits.div_scalar(temperature)?.log_softmax(-1)?;

        let teacher = teacher_log_probs.to_vec_f32()?;
        let student = student_log_probs.to_vec_f32()?;
        if teacher.len() != student.len() {
            return Err(invalid_input(
                "LwF teacher and student logits have different sizes",
            ));
        }

        let classes = *teacher_logits.shape().last().unwrap_or(&teacher.len().max(1));
        let rows = if classes == 0 { 1 } else { teacher.len() / classes.max(1) };

        let mut divergence = 0.0f32;
        for (log_t, log_s) in teacher.iter().zip(student.iter()) {
            divergence += log_t.exp() * (log_t - log_s);
        }
        if rows > 0 {
            divergence /= rows as f32;
        }

        scalar_tensor(lambda * temperature * temperature * divergence)
    }

    /// Compute experience replay loss
    fn compute_replay_loss(
        &mut self,
        memory_strength: f32,
        replay_batch_size: usize,
    ) -> Result<Tensor> {
        if self.memory.is_empty() {
            return scalar_tensor(0.0);
        }

        let (replay_inputs, replay_targets, _) = self.memory.sample_batch(replay_batch_size)?;

        if replay_inputs.is_empty() || replay_targets.is_empty() {
            return scalar_tensor(0.0);
        }

        // Compute loss on replay data
        let replay_outputs = self.model.forward(replay_inputs[0].clone())?; // Simplified
        let replay_loss = self.compute_task_loss(&replay_outputs, &replay_targets[0])?;

        scalar_tensor(memory_strength * scalar_value(&replay_loss)?)
    }

    /// Gradient Episodic Memory constraint term.
    ///
    /// GEM requires the update on the current batch not to increase the loss
    /// on stored episodes, i.e. `⟨g, g_mem⟩ ≥ 0`. This trainer computes losses
    /// rather than applying updates, so the inequality is enforced in its
    /// penalty (A-GEM-style) relaxation: the real inner product between the
    /// current-batch gradient and the memory gradient is measured, and a
    /// violation beyond `constraint_violation_threshold` is added to the loss
    /// scaled by `memory_strength`. A satisfied constraint adds nothing.
    pub fn compute_gem_loss(
        &mut self,
        input: &Tensor,
        target: &Tensor,
        current_loss: &Tensor,
        memory_strength: f32,
        constraint_violation_threshold: f32,
    ) -> Result<Tensor> {
        let base = scalar_value(current_loss)?;
        if self.memory.is_empty() {
            return scalar_tensor(base);
        }

        let (memory_inputs, memory_targets, _) = self.memory.sample_batch(1)?;
        let (Some(memory_input), Some(memory_target)) =
            (memory_inputs.first(), memory_targets.first())
        else {
            return scalar_tensor(base);
        };
        let memory_input = memory_input.clone();
        let memory_target = memory_target.clone();

        let current_gradients = self.loss_gradients(input, target)?;
        let memory_gradients = self.loss_gradients(&memory_input, &memory_target)?;

        let mut inner_product = 0.0f32;
        for (name, gradient) in &current_gradients {
            let Some(memory_gradient) = memory_gradients.get(name) else {
                continue;
            };
            let a = gradient.to_vec_f32()?;
            let b = memory_gradient.to_vec_f32()?;
            inner_product += a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
        }

        let violation = (-inner_product) - constraint_violation_threshold;
        let penalty = if violation > 0.0 { memory_strength * violation } else { 0.0 };

        scalar_tensor(base + penalty)
    }

    /// L2 continual-learning penalty.
    ///
    /// After a task has been finalized this is `λ · Σ_i (θ_i − θ*_i)²`, the
    /// standard "L2 baseline" that anchors the parameters to the previous
    /// task's solution. Before any snapshot exists it degenerates to plain
    /// weight decay `λ · Σ_i θ_i²`.
    pub fn compute_l2_regularization(&self, lambda: f32) -> Result<Tensor> {
        let mut total = 0.0f32;

        for (name, current) in self.model.named_parameters() {
            let values = current.to_vec_f32()?;
            match self.optimal_parameters.get(&name) {
                Some(optimal) => {
                    let anchor = optimal.to_vec_f32()?;
                    if anchor.len() != values.len() {
                        return Err(invalid_input(format!(
                            "L2 anchor for parameter '{}' has a mismatched length",
                            name
                        )));
                    }
                    for (value, target) in values.iter().zip(anchor.iter()) {
                        let delta = value - target;
                        total += delta * delta;
                    }
                },
                None => {
                    total += values.iter().map(|v| v * v).sum::<f32>();
                },
            }
        }

        scalar_tensor(lambda * total)
    }

    /// Compute example priority for memory storage
    fn compute_example_priority(&self, input: &Tensor, target: &Tensor) -> Result<f32> {
        match self.config.memory_selection {
            MemorySelectionStrategy::Random => Ok(1.0),
            MemorySelectionStrategy::Uncertainty => {
                // Predictive entropy of the model's own distribution.
                let outputs = self.model.forward(input.clone())?;
                let log_probs = outputs.log_softmax(-1)?.to_vec_f32()?;
                let entropy: f32 = -log_probs.iter().map(|l| l.exp() * l).sum::<f32>();
                Ok(entropy)
            },
            MemorySelectionStrategy::HighestLoss => {
                let outputs = self.model.forward(input.clone())?;
                let loss = self.compute_task_loss(&outputs, target)?;
                scalar_value(&loss)
            },
            _ => Ok(1.0), // Default priority
        }
    }

    /// Accumulate the diagonal Fisher information for EWC.
    ///
    /// For every sampled example the gradient of the task loss with respect to
    /// each named parameter is computed (analytically when the model provides
    /// it, otherwise by central finite differences) and its square is averaged
    /// into `fisher_matrices`. The result is the empirical diagonal Fisher
    /// `F_i = E[(∂L/∂θ_i)²]`, keyed by real parameter names.
    pub fn compute_fisher_information(&mut self, task_id: usize, num_samples: usize) -> Result<()> {
        // `gamma = 0` discards any previous estimate, which is exactly the
        // per-task (non-online) EWC behaviour.
        self.accumulate_fisher_information(task_id, num_samples, 0.0)
    }

    /// Accumulate the diagonal Fisher information, decaying whatever was
    /// already stored by `gamma` first.
    ///
    /// `gamma = 0` replaces the estimate (standard EWC); `0 < gamma <= 1`
    /// retains a fraction of the previous tasks' Fisher (Online EWC).
    pub fn accumulate_fisher_information(
        &mut self,
        task_id: usize,
        num_samples: usize,
        gamma: f32,
    ) -> Result<()> {
        let (task_inputs, task_targets) = self.memory.get_task_examples(task_id);
        if task_inputs.is_empty() || task_targets.is_empty() {
            return Ok(());
        }

        let sample_size = num_samples.min(task_inputs.len()).max(1);
        let mut accumulator: HashMap<String, Vec<f32>> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

        for index in 0..sample_size {
            let input = task_inputs[index % task_inputs.len()].clone();
            let target = task_targets[index % task_targets.len()].clone();
            let gradients = self.loss_gradients(&input, &target)?;

            for (name, gradient) in gradients {
                let shape = gradient.shape();
                let squared: Vec<f32> = gradient.to_vec_f32()?.into_iter().map(|g| g * g).collect();
                let entry =
                    accumulator.entry(name.clone()).or_insert_with(|| vec![0.0; squared.len()]);
                if entry.len() != squared.len() {
                    return Err(invalid_input(format!(
                        "parameter '{}' changed size during Fisher estimation",
                        name
                    )));
                }
                for (slot, value) in entry.iter_mut().zip(squared) {
                    *slot += value;
                }
                shapes.insert(name, shape);
            }
        }

        let decay = gamma.clamp(0.0, 1.0);
        let mut updated = HashMap::new();
        for (name, mut values) in accumulator {
            for value in values.iter_mut() {
                *value /= sample_size as f32;
            }

            if decay > 0.0 {
                if let Some(previous) = self.fisher_matrices.get(&name) {
                    let previous_values = previous.to_vec_f32()?;
                    if previous_values.len() == values.len() {
                        for (value, old) in values.iter_mut().zip(previous_values) {
                            *value += decay * old;
                        }
                    }
                }
            }

            let shape = shapes.remove(&name).unwrap_or_else(|| vec![values.len()]);
            updated.insert(name, Tensor::from_vec(values, &shape)?);
        }

        self.fisher_matrices = updated;
        Ok(())
    }

    /// Snapshot the current parameters as the EWC/L2 anchor `θ*`.
    pub fn save_optimal_parameters(&mut self) -> Result<()> {
        self.optimal_parameters.clear();
        for (name, tensor) in self.model.named_parameters() {
            self.optimal_parameters.insert(name, tensor);
        }
        Ok(())
    }

    /// Progressive Neural Networks require adding a new network column per
    /// task, which cannot be expressed through the [`Model`] trait: the
    /// trainer can read and write existing parameters but cannot change the
    /// architecture. This therefore reports the missing capability instead of
    /// silently doing nothing.
    fn add_progressive_columns(&mut self, _task_id: usize) -> Result<()> {
        Err(not_implemented(
            "ContinualStrategy::ProgressiveNeuralNetworks requires architecture growth, which \
             the Model trait does not expose",
        ))
    }

    /// Record which weights are still free for the incoming task.
    ///
    /// A weight is free when it is not already pinned by an earlier task's
    /// PackNet mask.
    fn prepare_packnet(&mut self, _task_id: usize) -> Result<()> {
        for (name, tensor) in self.model.named_parameters() {
            let length = tensor.to_vec_f32()?.len();
            self.packnet_masks.entry(name).or_insert_with(|| vec![false; length]);
        }
        Ok(())
    }

    /// Apply PackNet magnitude pruning.
    ///
    /// The smallest-magnitude `prune_ratio` fraction of every free (not yet
    /// pinned) weight is zeroed, and the surviving weights are pinned to the
    /// current task so later tasks cannot claim them. Retraining the surviving
    /// weights is the caller's responsibility.
    fn apply_packnet_pruning(&mut self, prune_ratio: f32) -> Result<()> {
        if !(0.0..1.0).contains(&prune_ratio) {
            return Err(invalid_input(format!(
                "PackNet prune_ratio must be in [0, 1), got {}",
                prune_ratio
            )));
        }

        for (name, tensor) in self.model.named_parameters() {
            let shape = tensor.shape();
            let mut values = tensor.to_vec_f32()?;
            let pinned = self
                .packnet_masks
                .get(&name)
                .cloned()
                .unwrap_or_else(|| vec![false; values.len()]);

            let free_indices: Vec<usize> =
                (0..values.len()).filter(|index| !pinned[*index]).collect();
            if free_indices.is_empty() {
                continue;
            }

            let mut magnitudes: Vec<f32> =
                free_indices.iter().map(|index| values[*index].abs()).collect();
            magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let cutoff = ((free_indices.len() as f32) * prune_ratio).floor() as usize;
            let mut mask = pinned;
            if cutoff > 0 {
                let threshold = magnitudes[(cutoff - 1).min(magnitudes.len() - 1)];
                for index in &free_indices {
                    if values[*index].abs() <= threshold {
                        values[*index] = 0.0;
                    } else {
                        mask[*index] = true;
                    }
                }
            } else {
                for index in &free_indices {
                    mask[*index] = true;
                }
            }

            self.model.set_named_parameter(&name, Tensor::from_vec(values, &shape)?)?;
            self.packnet_masks.insert(name, mask);
        }

        Ok(())
    }

    /// Evaluate on all tasks
    pub fn evaluate_all_tasks(&self) -> Result<HashMap<usize, TaskEvaluation>> {
        let mut evaluations = HashMap::new();

        for &task_id in self.task_info.keys() {
            let (task_inputs, task_targets) = self.memory.get_task_examples(task_id);

            if !task_inputs.is_empty() {
                let evaluation = self.evaluate_task(&task_inputs, &task_targets, task_id)?;
                evaluations.insert(task_id, evaluation);
            }
        }

        Ok(evaluations)
    }

    /// Evaluate the live model on a specific task.
    ///
    /// Accuracy is the fraction of rows whose predicted class (`argmax` over
    /// the last output dimension) matches the reference class, taken either
    /// from one-hot targets (`argmax`) or from class-index targets.
    pub fn evaluate_task(
        &self,
        inputs: &[Tensor],
        targets: &[Tensor],
        task_id: usize,
    ) -> Result<TaskEvaluation> {
        if inputs.is_empty() {
            return Err(invalid_input("evaluate_task requires at least one example"));
        }

        let mut total_loss = 0.0f32;
        let mut correct_predictions = 0usize;
        let mut total_rows = 0usize;

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let outputs = self.model.forward(input.clone())?;
            let loss = self.compute_task_loss(&outputs, target)?;
            total_loss += scalar_value(&loss)?;

            let output_shape = outputs.shape();
            let classes = *output_shape
                .last()
                .ok_or_else(|| invalid_input("model output must have at least one dimension"))?;
            if classes == 0 {
                return Err(invalid_input("model output has zero classes"));
            }

            let output_values = outputs.to_vec_f32()?;
            let target_values = target.to_vec_f32()?;
            let rows = output_values.len() / classes;
            let targets_are_one_hot = target_values.len() == output_values.len();

            for row in 0..rows {
                let predicted = argmax(&output_values[row * classes..(row + 1) * classes]);
                let reference = if targets_are_one_hot {
                    argmax(&target_values[row * classes..(row + 1) * classes])
                } else {
                    match target_values.get(row) {
                        Some(index) if *index >= 0.0 => *index as usize,
                        _ => usize::MAX,
                    }
                };

                if predicted == reference {
                    correct_predictions += 1;
                }
                total_rows += 1;
            }
        }

        let denominator = total_rows.max(1) as f32;
        Ok(TaskEvaluation {
            task_id,
            average_loss: total_loss / inputs.len() as f32,
            accuracy: correct_predictions as f32 / denominator,
            num_examples: total_rows,
        })
    }

    /// Continual-learning metrics measured on the live model.
    ///
    /// Returns an error when the underlying evaluation fails, rather than
    /// reporting a fabricated `average_accuracy` of `0.0`. When no task has
    /// stored examples yet the accuracy is reported as `None`.
    pub fn get_metrics(&self) -> Result<ContinualLearningMetrics> {
        let all_evaluations = self.evaluate_all_tasks()?;

        let average_accuracy = if all_evaluations.is_empty() {
            None
        } else {
            Some(
                all_evaluations.values().map(|e| e.accuracy).sum::<f32>()
                    / all_evaluations.len() as f32,
            )
        };

        let memory_efficiency = if self.config.memory_size == 0 {
            0.0
        } else {
            self.memory.size() as f32 / self.config.memory_size as f32
        };

        Ok(ContinualLearningMetrics {
            average_accuracy,
            task_evaluations: all_evaluations,
            memory_efficiency,
            num_tasks_learned: self.task_info.len(),
            current_task: self.current_task,
        })
    }
}

/// Information about a specific task
#[derive(Debug, Clone)]
pub struct TaskInfo {
    pub task_id: usize,
    pub start_step: usize,
    pub num_examples_seen: usize,
    pub average_loss: f32,
    pub last_accuracy: f32,
}

impl TaskInfo {
    pub fn new(task_id: usize) -> Self {
        Self {
            task_id,
            start_step: 0,
            num_examples_seen: 0,
            average_loss: 0.0,
            last_accuracy: 0.0,
        }
    }

    pub fn update_statistics(&mut self, loss: f32) {
        self.num_examples_seen += 1;
        self.average_loss = (self.average_loss * (self.num_examples_seen - 1) as f32 + loss)
            / self.num_examples_seen as f32;
    }
}

/// Automatic task-boundary detector based on input-distribution drift.
///
/// The detector keeps an exponentially weighted mean of the flattened input
/// features. When the cosine distance between an incoming batch's mean feature
/// vector and the running mean exceeds `threshold`, a boundary is reported and
/// the running mean is reset to the new batch. A change in feature width is
/// always a boundary.
pub struct TaskDetector {
    threshold: f32,
    running_mean: Option<Vec<f32>>,
    smoothing: f32,
    boundaries: usize,
}

impl TaskDetector {
    /// Create a detector that fires above `threshold` cosine distance.
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            running_mean: None,
            smoothing: 0.1,
            boundaries: 0,
        }
    }

    /// Number of boundaries reported so far.
    pub fn boundaries(&self) -> usize {
        self.boundaries
    }

    /// Mean feature vector of a batch of inputs.
    fn batch_mean(inputs: &[Tensor]) -> Result<Option<Vec<f32>>> {
        let mut accumulator: Option<Vec<f32>> = None;
        let mut count = 0usize;

        for input in inputs {
            let values = input.to_vec_f32()?;
            if values.is_empty() {
                continue;
            }
            match accumulator.as_mut() {
                Some(mean) if mean.len() == values.len() => {
                    for (slot, value) in mean.iter_mut().zip(values) {
                        *slot += value;
                    }
                },
                Some(_) => {
                    return Err(invalid_input(
                        "task detection requires all inputs in a batch to share a shape",
                    ));
                },
                None => accumulator = Some(values),
            }
            count += 1;
        }

        Ok(accumulator.map(|mut mean| {
            for value in mean.iter_mut() {
                *value /= count.max(1) as f32;
            }
            mean
        }))
    }

    fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a <= f32::EPSILON || norm_b <= f32::EPSILON {
            return 0.0;
        }
        1.0 - (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }

    /// Report a new task index when the input distribution has drifted.
    pub fn detect_task_change(
        &mut self,
        inputs: &[Tensor],
        _targets: &[Tensor],
    ) -> Result<Option<usize>> {
        let Some(batch_mean) = Self::batch_mean(inputs)? else {
            return Ok(None);
        };

        let Some(running) = self.running_mean.as_mut() else {
            self.running_mean = Some(batch_mean);
            return Ok(None);
        };

        if running.len() != batch_mean.len() {
            self.running_mean = Some(batch_mean);
            self.boundaries += 1;
            return Ok(Some(self.boundaries));
        }

        let distance = Self::cosine_distance(running, &batch_mean);
        if distance > self.threshold {
            self.running_mean = Some(batch_mean);
            self.boundaries += 1;
            return Ok(Some(self.boundaries));
        }

        for (slot, value) in running.iter_mut().zip(batch_mean) {
            *slot = (1.0 - self.smoothing) * *slot + self.smoothing * value;
        }
        Ok(None)
    }
}

/// Output from continual learning step
#[derive(Debug, Clone)]
pub struct ContinualLearningOutput {
    pub total_loss: Tensor,
    pub task_loss: Tensor,
    pub regularization_loss: Tensor,
    pub task_id: usize,
    pub memory_usage: usize,
}

/// Evaluation results for a specific task
#[derive(Debug, Clone)]
pub struct TaskEvaluation {
    pub task_id: usize,
    pub average_loss: f32,
    pub accuracy: f32,
    pub num_examples: usize,
}

/// Overall continual learning metrics
#[derive(Debug, Clone)]
pub struct ContinualLearningMetrics {
    /// Mean accuracy over every task that has stored examples, or `None` when
    /// no task has been evaluated yet.
    pub average_accuracy: Option<f32>,
    pub task_evaluations: HashMap<usize, TaskEvaluation>,
    pub memory_efficiency: f32,
    pub num_tasks_learned: usize,
    pub current_task: Option<usize>,
}

/// Utilities for continual learning
pub mod utils {
    use super::*;

    /// Create EWC configuration
    pub fn ewc_config(
        lambda: f32,
        fisher_samples: usize,
        memory_size: usize,
    ) -> ContinualLearningConfig {
        ContinualLearningConfig {
            strategy: ContinualStrategy::ElasticWeightConsolidation {
                lambda,
                fisher_samples,
            },
            memory_size,
            ..Default::default()
        }
    }

    /// Create experience replay configuration
    pub fn experience_replay_config(
        memory_size: usize,
        replay_batch_size: usize,
    ) -> ContinualLearningConfig {
        ContinualLearningConfig {
            strategy: ContinualStrategy::ExperienceReplay {
                memory_strength: 1.0,
                replay_batch_size,
            },
            memory_size,
            memory_selection: MemorySelectionStrategy::Random,
            ..Default::default()
        }
    }

    /// Create L2 regularization configuration
    pub fn l2_regularization_config(lambda: f32) -> ContinualLearningConfig {
        ContinualLearningConfig {
            strategy: ContinualStrategy::L2Regularization { lambda },
            memory_size: 0, // No memory needed for L2 regularization
            ..Default::default()
        }
    }

    /// Create progressive networks configuration
    pub fn progressive_networks_config() -> ContinualLearningConfig {
        ContinualLearningConfig {
            strategy: ContinualStrategy::ProgressiveNeuralNetworks {
                lateral_connections: true,
                adapter_layers: true,
            },
            task_specific_heads: true,
            ..Default::default()
        }
    }

    /// Compute backward transfer (improvement on previous tasks)
    pub fn compute_backward_transfer(
        evaluations_before: &HashMap<usize, TaskEvaluation>,
        evaluations_after: &HashMap<usize, TaskEvaluation>,
    ) -> f32 {
        let mut total_transfer = 0.0;
        let mut num_tasks = 0;

        for (&task_id, after_eval) in evaluations_after {
            if let Some(before_eval) = evaluations_before.get(&task_id) {
                total_transfer += after_eval.accuracy - before_eval.accuracy;
                num_tasks += 1;
            }
        }

        if num_tasks > 0 {
            total_transfer / num_tasks as f32
        } else {
            0.0
        }
    }

    /// Compute forward transfer (improvement on new tasks)
    pub fn compute_forward_transfer(baseline_accuracy: f32, continual_accuracy: f32) -> f32 {
        continual_accuracy - baseline_accuracy
    }

    /// Compute forgetting measure
    pub fn compute_forgetting(
        max_accuracies: &HashMap<usize, f32>,
        final_accuracies: &HashMap<usize, f32>,
    ) -> f32 {
        let mut total_forgetting = 0.0;
        let mut num_tasks = 0;

        for (&task_id, &max_acc) in max_accuracies {
            if let Some(&final_acc) = final_accuracies.get(&task_id) {
                total_forgetting += max_acc - final_acc;
                num_tasks += 1;
            }
        }

        if num_tasks > 0 {
            total_forgetting / num_tasks as f32
        } else {
            0.0
        }
    }
}
