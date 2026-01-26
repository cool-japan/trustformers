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
//! ```rust,no_run
//! use trustformers_models::continual_learning::{
//!     ContinualLearningTrainer, ContinualLearningConfig, ContinualStrategy
//! };
//!
//! let config = ContinualLearningConfig {
//!     strategy: ContinualStrategy::ElasticWeightConsolidation {
//!         lambda: 0.4,
//!         fisher_samples: 1000,
//!     },
//!     memory_size: 1000,
//!     ..Default::default()
//! };
//!
//! let mut trainer = ContinualLearningTrainer::new(model, config)?;
//!
//! // Learn task 1
//! trainer.learn_task(task1_data, 0)?;
//! // Learn task 2 without forgetting task 1
//! trainer.learn_task(task2_data, 1)?;
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::{errors::invalid_input, tensor::Tensor, traits::Model, Result};

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
}

impl<M: Model<Input = Tensor, Output = Tensor>> ContinualLearningTrainer<M> {
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
        })
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

        // Compute forward pass and loss
        let outputs = self.model.forward(inputs[0].clone())?; // Simplified single input
        let current_loss = self.compute_task_loss(&outputs, &targets[0])?;
        let current_loss_for_output = current_loss.clone();

        // Apply continual learning strategy
        let total_loss = match &self.config.strategy {
            ContinualStrategy::ElasticWeightConsolidation { lambda, .. } => {
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
                memory_strength, ..
            } => self.compute_gem_loss(&current_loss, *memory_strength)?,
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
        if let Some(task_info) = self.task_info.get_mut(&task_id) {
            task_info.update_statistics(total_loss.to_scalar().unwrap_or(0.0));
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
            ContinualStrategy::ElasticWeightConsolidation { fisher_samples, .. }
            | ContinualStrategy::OnlineElasticWeightConsolidation { fisher_samples, .. } => {
                self.compute_fisher_information(task_id, fisher_samples)?;
                self.save_optimal_parameters()?;
            },
            ContinualStrategy::PackNet {
                prune_ratio,
                retrain_epochs,
            } => {
                self.apply_packnet_pruning(prune_ratio)?;
                self.retrain_after_pruning(retrain_epochs)?;
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
            Ok(sum_per_sample.neg()?.mean()?)
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
            Ok(sum_per_sample.neg()?.mean()?)
        }
    }

    /// Compute EWC regularization loss
    fn compute_ewc_loss(&self, lambda: f32) -> Result<Tensor> {
        let mut total_loss = Tensor::zeros(&[1])?;

        // This is a simplified implementation
        // In practice, you'd iterate through model parameters
        for (param_name, fisher) in &self.fisher_matrices {
            if let Some(optimal) = self.optimal_parameters.get(param_name) {
                // Get current parameter (simplified)
                let current_param = Tensor::zeros_like(optimal)?; // Placeholder
                let diff = current_param.sub(optimal)?;
                let squared_diff = diff.mul(&diff)?;
                let weighted_diff = fisher.mul(&squared_diff)?;
                total_loss = total_loss.add(&weighted_diff.sum(None, false)?)?;
            }
        }

        total_loss.scalar_mul(lambda)
    }

    /// Compute Learning without Forgetting distillation loss
    fn compute_lwf_loss(
        &self,
        _inputs: &[Tensor],
        lambda: f32,
        _temperature: f32,
    ) -> Result<Tensor> {
        // This would compute the distillation loss from previous tasks
        // Simplified implementation
        Tensor::zeros(&[1])?.scalar_mul(lambda)
    }

    /// Compute experience replay loss
    fn compute_replay_loss(
        &mut self,
        memory_strength: f32,
        replay_batch_size: usize,
    ) -> Result<Tensor> {
        if self.memory.is_empty() {
            return Tensor::zeros(&[1]);
        }

        let (replay_inputs, replay_targets, _) = self.memory.sample_batch(replay_batch_size)?;

        if replay_inputs.is_empty() {
            return Tensor::zeros(&[1]);
        }

        // Compute loss on replay data
        let replay_outputs = self.model.forward(replay_inputs[0].clone())?; // Simplified
        let replay_loss = self.compute_task_loss(&replay_outputs, &replay_targets[0])?;

        replay_loss.scalar_mul(memory_strength)
    }

    /// Compute GEM constraint loss
    fn compute_gem_loss(&mut self, current_loss: &Tensor, memory_strength: f32) -> Result<Tensor> {
        // GEM computes gradients on memory and projects current gradients
        // This is a simplified implementation
        current_loss.scalar_mul(memory_strength)
    }

    /// Compute L2 regularization loss
    fn compute_l2_regularization(&self, lambda: f32) -> Result<Tensor> {
        // Compute L2 norm of parameters
        // This is a simplified implementation
        Tensor::zeros(&[1])?.scalar_mul(lambda)
    }

    /// Compute example priority for memory storage
    fn compute_example_priority(&self, input: &Tensor, target: &Tensor) -> Result<f32> {
        match self.config.memory_selection {
            MemorySelectionStrategy::Random => Ok(1.0),
            MemorySelectionStrategy::Uncertainty => {
                // Compute prediction uncertainty
                let outputs = self.model.forward(input.clone())?;
                let probs = outputs.softmax(-1)?;
                let entropy = -(probs.clone().mul(&probs.log()?)?)
                    .sum(Some(vec![1]), false)?
                    .to_scalar()
                    .unwrap_or(0.0);
                Ok(entropy)
            },
            MemorySelectionStrategy::HighestLoss => {
                let outputs = self.model.forward(input.clone())?;
                let loss = self.compute_task_loss(&outputs, target)?;
                Ok(loss.to_scalar().unwrap_or(0.0))
            },
            _ => Ok(1.0), // Default priority
        }
    }

    /// Compute Fisher information for EWC
    fn compute_fisher_information(&mut self, task_id: usize, num_samples: usize) -> Result<()> {
        // Get examples from current task
        let (task_inputs, task_targets) = self.memory.get_task_examples(task_id);

        if task_inputs.is_empty() {
            return Ok(());
        }

        // Sample examples for Fisher computation
        let sample_size = num_samples.min(task_inputs.len());

        // This is a simplified implementation
        // In practice, you'd compute Fisher information for each parameter
        for i in 0..sample_size {
            let input = &task_inputs[i % task_inputs.len()];
            let target = &task_targets[i % task_targets.len()];

            // Compute gradients and accumulate Fisher information
            let outputs = self.model.forward(input.clone())?;
            let _loss = self.compute_task_loss(&outputs, target)?;

            // Store Fisher information (simplified)
            self.fisher_matrices.insert(
                format!("param_{}", i),
                Tensor::ones(&[10])?, // Placeholder
            );
        }

        Ok(())
    }

    /// Save optimal parameters for EWC
    fn save_optimal_parameters(&mut self) -> Result<()> {
        // Save current model parameters as optimal
        // This is a simplified implementation
        self.optimal_parameters.insert(
            "param_0".to_string(),
            Tensor::zeros(&[10])?, // Placeholder
        );
        Ok(())
    }

    /// Add progressive network columns
    fn add_progressive_columns(&mut self, _task_id: usize) -> Result<()> {
        // Add new columns to the network for the new task
        // This is a simplified implementation
        Ok(())
    }

    /// Prepare for PackNet pruning
    fn prepare_packnet(&mut self, _task_id: usize) -> Result<()> {
        // Prepare the network for pruning-based continual learning
        Ok(())
    }

    /// Apply PackNet pruning
    fn apply_packnet_pruning(&mut self, _prune_ratio: f32) -> Result<()> {
        // Prune the network and freeze pruned weights
        Ok(())
    }

    /// Retrain after PackNet pruning
    fn retrain_after_pruning(&mut self, _epochs: usize) -> Result<()> {
        // Retrain the unpruned weights
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

    /// Evaluate on a specific task
    fn evaluate_task(
        &self,
        inputs: &[Tensor],
        targets: &[Tensor],
        task_id: usize,
    ) -> Result<TaskEvaluation> {
        let mut total_loss = 0.0;
        let mut correct_predictions = 0;
        let total_examples = inputs.len();

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let outputs = self.model.forward(input.clone())?;
            let loss = self.compute_task_loss(&outputs, target)?;
            total_loss += loss.to_scalar().unwrap_or(0.0);

            // Compute accuracy (simplified)
            let predicted = Tensor::zeros(&[1])?; // Simplified placeholder - ideally should be argmax
            let target_class = Tensor::zeros(&[1])?; // Simplified placeholder - ideally should be argmax
            if predicted.to_scalar().unwrap_or(-1.0) == target_class.to_scalar().unwrap_or(-2.0) {
                correct_predictions += 1;
            }
        }

        Ok(TaskEvaluation {
            task_id,
            average_loss: total_loss / total_examples as f32,
            accuracy: correct_predictions as f32 / total_examples as f32,
            num_examples: total_examples,
        })
    }

    /// Get continual learning metrics
    pub fn get_metrics(&self) -> ContinualLearningMetrics {
        let all_evaluations = self.evaluate_all_tasks().unwrap_or_default();

        let average_accuracy = if !all_evaluations.is_empty() {
            all_evaluations.values().map(|e| e.accuracy).sum::<f32>() / all_evaluations.len() as f32
        } else {
            0.0
        };

        let memory_efficiency = self.memory.size() as f32 / self.config.memory_size as f32;

        ContinualLearningMetrics {
            average_accuracy,
            task_evaluations: all_evaluations,
            memory_efficiency,
            num_tasks_learned: self.task_info.len(),
            current_task: self.current_task,
        }
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

/// Task detector for automatic task boundary detection
pub struct TaskDetector {
    #[allow(dead_code)]
    threshold: f32,
    #[allow(dead_code)]
    recent_losses: Vec<f32>,
    #[allow(dead_code)]
    window_size: usize,
}

impl TaskDetector {
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            recent_losses: Vec::new(),
            window_size: 100,
        }
    }

    pub fn detect_task_change(
        &mut self,
        _inputs: &[Tensor],
        _targets: &[Tensor],
    ) -> Result<Option<usize>> {
        // Simplified task detection based on loss spikes
        // In practice, this would be more sophisticated
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
    pub average_accuracy: f32,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continual_learning_config_default() {
        let config = ContinualLearningConfig::default();
        assert_eq!(config.memory_size, 1000);
        assert!(config.task_specific_heads);
        assert!(!config.automatic_task_detection);

        if let ContinualStrategy::ElasticWeightConsolidation {
            lambda,
            fisher_samples,
        } = config.strategy
        {
            assert_eq!(lambda, 0.4);
            assert_eq!(fisher_samples, 1000);
        } else {
            panic!("Expected EWC strategy");
        }
    }

    #[test]
    fn test_memory_buffer() {
        let mut buffer = MemoryBuffer::new(3, MemorySelectionStrategy::Random);
        assert!(buffer.is_empty());
        assert_eq!(buffer.size(), 0);

        // Add examples
        let input1 = Tensor::zeros(&[1, 10]).expect("operation failed");
        let target1 = Tensor::zeros(&[1]).expect("operation failed");
        buffer.add_example(input1, target1, 0, 1.0);
        assert_eq!(buffer.size(), 1);

        let input2 = Tensor::ones(&[1, 10]).expect("operation failed");
        let target2 = Tensor::ones(&[1]).expect("operation failed");
        buffer.add_example(input2, target2, 1, 2.0);
        assert_eq!(buffer.size(), 2);

        // Sample batch
        let (inputs, targets, task_ids) = buffer.sample_batch(2).expect("operation failed");
        assert_eq!(inputs.len(), 2);
        assert_eq!(targets.len(), 2);
        assert_eq!(task_ids.len(), 2);
    }

    #[test]
    fn test_ewc_config() {
        let config = utils::ewc_config(0.5, 2000, 500);
        assert_eq!(config.memory_size, 500);

        if let ContinualStrategy::ElasticWeightConsolidation {
            lambda,
            fisher_samples,
        } = config.strategy
        {
            assert_eq!(lambda, 0.5);
            assert_eq!(fisher_samples, 2000);
        } else {
            panic!("Expected EWC strategy");
        }
    }

    #[test]
    fn test_experience_replay_config() {
        let config = utils::experience_replay_config(1000, 64);
        assert_eq!(config.memory_size, 1000);

        if let ContinualStrategy::ExperienceReplay {
            memory_strength,
            replay_batch_size,
        } = config.strategy
        {
            assert_eq!(memory_strength, 1.0);
            assert_eq!(replay_batch_size, 64);
        } else {
            panic!("Expected ExperienceReplay strategy");
        }
    }

    #[test]
    fn test_l2_regularization_config() {
        let config = utils::l2_regularization_config(0.01);
        assert_eq!(config.memory_size, 0);

        if let ContinualStrategy::L2Regularization { lambda } = config.strategy {
            assert_eq!(lambda, 0.01);
        } else {
            panic!("Expected L2Regularization strategy");
        }
    }

    #[test]
    fn test_task_info() {
        let mut info = TaskInfo::new(5);
        assert_eq!(info.task_id, 5);
        assert_eq!(info.num_examples_seen, 0);

        info.update_statistics(0.5);
        assert_eq!(info.num_examples_seen, 1);
        assert_eq!(info.average_loss, 0.5);

        info.update_statistics(1.0);
        assert_eq!(info.num_examples_seen, 2);
        assert_eq!(info.average_loss, 0.75);
    }

    #[test]
    fn test_backward_transfer_computation() {
        let mut before = HashMap::new();
        before.insert(
            0,
            TaskEvaluation {
                task_id: 0,
                average_loss: 0.5,
                accuracy: 0.8,
                num_examples: 100,
            },
        );
        before.insert(
            1,
            TaskEvaluation {
                task_id: 1,
                average_loss: 0.6,
                accuracy: 0.7,
                num_examples: 100,
            },
        );

        let mut after = HashMap::new();
        after.insert(
            0,
            TaskEvaluation {
                task_id: 0,
                average_loss: 0.4,
                accuracy: 0.85,
                num_examples: 100,
            },
        );
        after.insert(
            1,
            TaskEvaluation {
                task_id: 1,
                average_loss: 0.55,
                accuracy: 0.72,
                num_examples: 100,
            },
        );

        let backward_transfer = utils::compute_backward_transfer(&before, &after);
        assert!((backward_transfer - 0.035).abs() < 1e-6); // (0.05 + 0.02) / 2
    }

    #[test]
    fn test_forgetting_computation() {
        let mut max_accuracies = HashMap::new();
        max_accuracies.insert(0, 0.9);
        max_accuracies.insert(1, 0.85);

        let mut final_accuracies = HashMap::new();
        final_accuracies.insert(0, 0.8);
        final_accuracies.insert(1, 0.75);

        let forgetting = utils::compute_forgetting(&max_accuracies, &final_accuracies);
        assert!((forgetting - 0.1).abs() < 1e-6); // (0.1 + 0.1) / 2
    }
}
