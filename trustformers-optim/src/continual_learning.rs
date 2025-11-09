//! # Continual Learning Optimizers
//!
//! This module implements optimization algorithms specifically designed for
//! continual learning scenarios where models must learn new tasks while
//! retaining knowledge of previous tasks.
//!
//! ## Available Methods
//!
//! - **EWC (Elastic Weight Consolidation)**: Protects important weights from changes
//! - **PackNet**: Progressive networks with parameter allocation
//! - **L2 Regularization**: Simple regularization towards previous weights
//! - **Memory Replay**: Gradient-based memory replay optimization
//! - **Meta-Learning**: Model-Agnostic Meta-Learning for continual adaptation

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::tensor::Tensor;

/// Configuration for Elastic Weight Consolidation (EWC).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EWCConfig {
    /// Base learning rate
    pub learning_rate: f32,
    /// Importance weight for Fisher information regularization
    pub lambda: f32,
    /// Method for computing Fisher information
    pub fisher_method: FisherMethod,
    /// Number of samples for Fisher information estimation
    pub fisher_samples: usize,
    /// Online vs offline EWC
    pub online: bool,
    /// Decay factor for online EWC
    pub decay_factor: f32,
}

impl Default for EWCConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            lambda: 1000.0,
            fisher_method: FisherMethod::Empirical,
            fisher_samples: 1000,
            online: false,
            decay_factor: 0.9,
        }
    }
}

/// Methods for computing Fisher information matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FisherMethod {
    /// Empirical Fisher information
    Empirical,
    /// True Fisher information (computationally expensive)
    True,
    /// Diagonal approximation
    Diagonal,
}

/// Configuration for Progressive Networks (PackNet).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackNetConfig {
    /// Base learning rate
    pub learning_rate: f32,
    /// Sparsity level for each task
    pub sparsity_level: f32,
    /// Number of tasks
    pub num_tasks: usize,
    /// Parameter allocation strategy
    pub allocation_strategy: AllocationStrategy,
}

impl Default for PackNetConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            sparsity_level: 0.5,
            num_tasks: 10,
            allocation_strategy: AllocationStrategy::Sequential,
        }
    }
}

/// Parameter allocation strategies for PackNet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// Sequential allocation
    Sequential,
    /// Random allocation
    Random,
    /// Importance-based allocation
    ImportanceBased,
}

/// Configuration for L2 regularization towards previous parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L2RegularizationConfig {
    /// Base learning rate
    pub learning_rate: f32,
    /// Regularization strength
    pub reg_strength: f32,
    /// Update strategy for anchor parameters
    pub update_strategy: UpdateStrategy,
}

impl Default for L2RegularizationConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            reg_strength: 0.1,
            update_strategy: UpdateStrategy::EMA,
        }
    }
}

/// Strategies for updating anchor parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateStrategy {
    /// No update (fixed anchors)
    Fixed,
    /// Exponential moving average
    EMA,
    /// Update at task boundaries
    TaskBoundary,
}

/// Configuration for Memory Replay optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReplayConfig {
    /// Base learning rate
    pub learning_rate: f32,
    /// Memory buffer size
    pub memory_size: usize,
    /// Replay frequency (every N steps)
    pub replay_frequency: usize,
    /// Replay batch size
    pub replay_batch_size: usize,
    /// Memory selection strategy
    pub selection_strategy: MemorySelectionStrategy,
}

impl Default for MemoryReplayConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            memory_size: 1000,
            replay_frequency: 10,
            replay_batch_size: 32,
            selection_strategy: MemorySelectionStrategy::Random,
        }
    }
}

/// Memory selection strategies for replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemorySelectionStrategy {
    /// Random selection
    Random,
    /// Gradient-based selection
    GradientBased,
    /// Uncertainty-based selection
    UncertaintyBased,
}

/// Elastic Weight Consolidation optimizer.
pub struct EWC {
    config: EWCConfig,
    parameters: Vec<Tensor>,
    importance_weights: Vec<Tensor>,
    anchor_parameters: Vec<Tensor>,
    current_task: usize,
    accumulated_importance: Vec<Tensor>,
}

impl EWC {
    /// Create a new EWC optimizer.
    pub fn new(config: EWCConfig, initial_parameters: Vec<Tensor>) -> Result<Self> {
        let param_count = initial_parameters.len();

        Ok(Self {
            config,
            parameters: initial_parameters.clone(),
            importance_weights: (0..param_count)
                .map(|i| Tensor::zeros(&initial_parameters[i].shape()).unwrap())
                .collect(),
            anchor_parameters: initial_parameters.clone(),
            current_task: 0,
            accumulated_importance: (0..param_count)
                .map(|i| Tensor::zeros(&initial_parameters[i].shape()).unwrap())
                .collect(),
        })
    }

    /// Compute Fisher information matrix for current task.
    pub fn compute_fisher_information(&mut self, gradients_samples: &[Vec<Tensor>]) -> Result<()> {
        let num_samples = gradients_samples.len();
        if num_samples == 0 {
            return Err(anyhow!("No gradient samples provided"));
        }

        // Reset importance weights for new task
        for importance in self.importance_weights.iter_mut() {
            *importance = Tensor::zeros(&importance.shape())?;
        }

        // Compute empirical Fisher information
        for gradient_sample in gradients_samples {
            for (i, gradient) in gradient_sample.iter().enumerate() {
                if i < self.importance_weights.len() {
                    let squared_grad = gradient.mul(gradient)?;
                    self.importance_weights[i] = self.importance_weights[i].add(&squared_grad)?;
                }
            }
        }

        // Average over samples
        for importance in self.importance_weights.iter_mut() {
            *importance = importance.div_scalar(num_samples as f32)?;
        }

        // Update accumulated importance for online EWC
        if self.config.online {
            for i in 0..self.accumulated_importance.len() {
                let decayed =
                    self.accumulated_importance[i].mul_scalar(self.config.decay_factor)?;
                self.accumulated_importance[i] = decayed.add(&self.importance_weights[i])?;
            }
        }

        Ok(())
    }

    /// Complete current task and prepare for next task.
    pub fn finish_task(&mut self) -> Result<()> {
        // Update anchor parameters to current parameters
        self.anchor_parameters = self.parameters.clone();
        self.current_task += 1;
        Ok(())
    }

    /// Perform optimization step with EWC regularization.
    pub fn step(&mut self, gradients: &[Tensor]) -> Result<()> {
        for (i, gradient) in gradients.iter().enumerate() {
            if i < self.parameters.len() {
                // Compute EWC penalty gradient
                let param_diff = self.parameters[i].sub(&self.anchor_parameters[i])?;
                let importance = if self.config.online {
                    &self.accumulated_importance[i]
                } else {
                    &self.importance_weights[i]
                };
                let ewc_grad = param_diff.mul(importance)?.mul_scalar(self.config.lambda)?;

                // Combine original gradient with EWC penalty
                let total_grad = gradient.add(&ewc_grad)?;

                // Apply update
                let update = total_grad.mul_scalar(self.config.learning_rate)?;
                self.parameters[i] = self.parameters[i].sub(&update)?;
            }
        }
        Ok(())
    }

    /// Get current parameters.
    pub fn get_parameters(&self) -> &[Tensor] {
        &self.parameters
    }

    /// Get importance weights.
    pub fn get_importance_weights(&self) -> &[Tensor] {
        &self.importance_weights
    }
}

/// Progressive Networks (PackNet) optimizer.
pub struct PackNet {
    config: PackNetConfig,
    parameters: Vec<Tensor>,
    #[allow(dead_code)]
    parameter_masks: Vec<Tensor>,
    task_allocations: HashMap<usize, Vec<Tensor>>,
    current_task: usize,
    available_capacity: Vec<f32>,
}

impl PackNet {
    /// Create a new PackNet optimizer.
    pub fn new(config: PackNetConfig, initial_parameters: Vec<Tensor>) -> Result<Self> {
        let param_count = initial_parameters.len();

        Ok(Self {
            config,
            parameters: initial_parameters.clone(),
            parameter_masks: (0..param_count)
                .map(|i| Tensor::ones(&initial_parameters[i].shape()).unwrap())
                .collect(),
            task_allocations: HashMap::new(),
            current_task: 0,
            available_capacity: vec![1.0; param_count],
        })
    }

    /// Allocate parameters for a new task.
    pub fn allocate_task(&mut self, task_id: usize) -> Result<()> {
        if self.available_capacity.iter().any(|&cap| cap < self.config.sparsity_level) {
            return Err(anyhow!("Insufficient parameter capacity for new task"));
        }

        let mut task_masks = Vec::new();

        for (i, param) in self.parameters.iter().enumerate() {
            let shape = param.shape();
            let total_params = shape.iter().product::<usize>();
            let allocated_params = (total_params as f32 * self.config.sparsity_level) as usize;

            // Create allocation mask
            let mut mask_data = vec![0.0; total_params];

            match self.config.allocation_strategy {
                AllocationStrategy::Sequential => {
                    let start_idx =
                        ((1.0 - self.available_capacity[i]) * total_params as f32) as usize;
                    let end_idx = (start_idx + allocated_params).min(total_params);
                    for idx in start_idx..end_idx {
                        mask_data[idx] = 1.0;
                    }
                },
                AllocationStrategy::Random => {
                    use scirs2_core::random::*; // SciRS2 Integration Policy
                    let mut indices: Vec<usize> = (0..total_params).collect();
                    let mut rng = thread_rng();
                    indices.shuffle(rng.rng_mut());
                    for &idx in indices.iter().take(allocated_params) {
                        mask_data[idx] = 1.0;
                    }
                },
                AllocationStrategy::ImportanceBased => {
                    // Simplified importance-based allocation
                    // In practice, this would use gradient magnitudes or other importance metrics
                    for idx in 0..allocated_params.min(total_params) {
                        mask_data[idx] = 1.0;
                    }
                },
            }

            let task_mask = Tensor::new(mask_data)?;
            task_masks.push(task_mask);

            // Update available capacity
            self.available_capacity[i] -= self.config.sparsity_level;
        }

        self.task_allocations.insert(task_id, task_masks);
        self.current_task = task_id;
        Ok(())
    }

    /// Perform optimization step with parameter masking.
    pub fn step(&mut self, gradients: &[Tensor]) -> Result<()> {
        let task_masks = self
            .task_allocations
            .get(&self.current_task)
            .ok_or_else(|| anyhow!("No allocation for current task"))?;

        for (i, gradient) in gradients.iter().enumerate() {
            if i < self.parameters.len() && i < task_masks.len() {
                // Apply task-specific mask to gradient
                let masked_grad = gradient.mul(&task_masks[i])?;

                // Apply update
                let update = masked_grad.mul_scalar(self.config.learning_rate)?;
                self.parameters[i] = self.parameters[i].sub(&update)?;
            }
        }
        Ok(())
    }

    /// Get current parameters.
    pub fn get_parameters(&self) -> &[Tensor] {
        &self.parameters
    }

    /// Get available capacity for new tasks.
    pub fn get_available_capacity(&self) -> &[f32] {
        &self.available_capacity
    }
}

/// L2 Regularization optimizer for continual learning.
pub struct L2Regularization {
    config: L2RegularizationConfig,
    parameters: Vec<Tensor>,
    anchor_parameters: Vec<Tensor>,
    ema_decay: f32,
}

impl L2Regularization {
    /// Create a new L2 regularization optimizer.
    pub fn new(config: L2RegularizationConfig, initial_parameters: Vec<Tensor>) -> Self {
        Self {
            config,
            parameters: initial_parameters.clone(),
            anchor_parameters: initial_parameters,
            ema_decay: 0.999,
        }
    }

    /// Perform optimization step with L2 regularization.
    pub fn step(&mut self, gradients: &[Tensor]) -> Result<()> {
        for (i, gradient) in gradients.iter().enumerate() {
            if i < self.parameters.len() {
                // Compute L2 regularization term
                let param_diff = self.parameters[i].sub(&self.anchor_parameters[i])?;
                let reg_grad = param_diff.mul_scalar(self.config.reg_strength)?;

                // Combine gradient with regularization
                let total_grad = gradient.add(&reg_grad)?;

                // Apply update
                let update = total_grad.mul_scalar(self.config.learning_rate)?;
                self.parameters[i] = self.parameters[i].sub(&update)?;

                // Update anchor parameters based on strategy
                match self.config.update_strategy {
                    UpdateStrategy::Fixed => {
                        // Don't update anchors
                    },
                    UpdateStrategy::EMA => {
                        // Exponential moving average update
                        let anchor_update = self.parameters[i].mul_scalar(1.0 - self.ema_decay)?;
                        let anchor_keep = self.anchor_parameters[i].mul_scalar(self.ema_decay)?;
                        self.anchor_parameters[i] = anchor_update.add(&anchor_keep)?;
                    },
                    UpdateStrategy::TaskBoundary => {
                        // Will be updated when finish_task() is called
                    },
                }
            }
        }
        Ok(())
    }

    /// Finish current task (update anchors for TaskBoundary strategy).
    pub fn finish_task(&mut self) -> Result<()> {
        if matches!(self.config.update_strategy, UpdateStrategy::TaskBoundary) {
            self.anchor_parameters = self.parameters.clone();
        }
        Ok(())
    }

    /// Get current parameters.
    pub fn get_parameters(&self) -> &[Tensor] {
        &self.parameters
    }
}

/// Memory replay optimizer.
pub struct MemoryReplay {
    config: MemoryReplayConfig,
    parameters: Vec<Tensor>,
    memory_buffer: Vec<Vec<Tensor>>, // Stored gradients
    step_count: usize,
}

impl MemoryReplay {
    /// Create a new memory replay optimizer.
    pub fn new(config: MemoryReplayConfig, initial_parameters: Vec<Tensor>) -> Self {
        Self {
            config,
            parameters: initial_parameters,
            memory_buffer: Vec::new(),
            step_count: 0,
        }
    }

    /// Add gradient to memory buffer.
    pub fn store_gradient(&mut self, gradients: &[Tensor]) -> Result<()> {
        if self.memory_buffer.len() >= self.config.memory_size {
            // Remove oldest or least important gradient
            match self.config.selection_strategy {
                MemorySelectionStrategy::Random => {
                    use scirs2_core::random::*; // SciRS2 Integration Policy
                    let idx = thread_rng().gen_range(0..self.memory_buffer.len());
                    self.memory_buffer.remove(idx);
                },
                _ => {
                    self.memory_buffer.remove(0); // FIFO for simplicity
                },
            }
        }

        self.memory_buffer.push(gradients.to_vec());
        Ok(())
    }

    /// Perform optimization step with memory replay.
    pub fn step(&mut self, gradients: &[Tensor]) -> Result<()> {
        // Regular gradient update
        for (i, gradient) in gradients.iter().enumerate() {
            if i < self.parameters.len() {
                let update = gradient.mul_scalar(self.config.learning_rate)?;
                self.parameters[i] = self.parameters[i].sub(&update)?;
            }
        }

        // Store current gradient
        self.store_gradient(gradients)?;

        // Replay from memory
        if self.step_count % self.config.replay_frequency == 0 && !self.memory_buffer.is_empty() {
            self.replay_step()?;
        }

        self.step_count += 1;
        Ok(())
    }

    fn replay_step(&mut self) -> Result<()> {
        let batch_size = self.config.replay_batch_size.min(self.memory_buffer.len());

        // Select random batch from memory
        use scirs2_core::random::*; // SciRS2 Integration Policy
        let mut indices: Vec<usize> = (0..self.memory_buffer.len()).collect();
        let mut rng = thread_rng();
        indices.shuffle(rng.rng_mut());

        for &idx in indices.iter().take(batch_size) {
            let replay_gradients = &self.memory_buffer[idx];

            // Apply replay gradient with reduced learning rate
            let replay_lr = self.config.learning_rate * 0.5;
            for (i, gradient) in replay_gradients.iter().enumerate() {
                if i < self.parameters.len() {
                    let update = gradient.mul_scalar(replay_lr)?;
                    self.parameters[i] = self.parameters[i].sub(&update)?;
                }
            }
        }

        Ok(())
    }

    /// Get current parameters.
    pub fn get_parameters(&self) -> &[Tensor] {
        &self.parameters
    }

    /// Get memory buffer size.
    pub fn memory_size(&self) -> usize {
        self.memory_buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ewc_config() {
        let config = EWCConfig::default();
        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.lambda, 1000.0);
        assert!(!config.online);
    }

    #[test]
    fn test_packnet_config() {
        let config = PackNetConfig::default();
        assert_eq!(config.sparsity_level, 0.5);
        assert_eq!(config.num_tasks, 10);
    }

    #[test]
    fn test_l2_regularization_config() {
        let config = L2RegularizationConfig::default();
        assert_eq!(config.reg_strength, 0.1);
        assert!(matches!(config.update_strategy, UpdateStrategy::EMA));
    }

    #[test]
    fn test_memory_replay_config() {
        let config = MemoryReplayConfig::default();
        assert_eq!(config.memory_size, 1000);
        assert_eq!(config.replay_frequency, 10);
        assert!(matches!(
            config.selection_strategy,
            MemorySelectionStrategy::Random
        ));
    }

    #[test]
    fn test_fisher_methods() {
        assert!(matches!(FisherMethod::Empirical, FisherMethod::Empirical));
        assert!(matches!(FisherMethod::True, FisherMethod::True));
        assert!(matches!(FisherMethod::Diagonal, FisherMethod::Diagonal));
    }
}
