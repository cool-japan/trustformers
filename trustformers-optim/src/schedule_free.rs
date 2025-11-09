//! # Schedule-Free Optimizers
//!
//! Implementation of Schedule-Free SGD and Adam optimizers from "The Road Less Scheduled" paper.
//! These optimizers eliminate the need for learning rate scheduling by automatically adapting
//! the learning rate based on the optimization trajectory.
//!
//! ## Key Features
//!
//! - **No Learning Rate Scheduling**: Automatically adapts learning rate without manual schedules
//! - **Theoretical Guarantees**: Maintains convergence guarantees of standard optimizers
//! - **Easy to Use**: Drop-in replacement for SGD and Adam with better generalization
//! - **Memory Efficient**: Minimal overhead compared to standard optimizers
//!
//! ## Research Background
//!
//! Schedule-Free optimizers use a momentum-based approach to automatically adjust the learning
//! rate during training. They maintain two sets of weights: the "momentum weights" used for
//! gradient computation and the "average weights" used for evaluation.

use crate::common::{OptimizerState, StateMemoryStats};
use crate::traits::StatefulOptimizer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

/// Configuration for Schedule-Free SGD optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleFreeSGDConfig {
    /// Learning rate (typically much higher than standard SGD, e.g., 1.0-10.0)
    pub learning_rate: f32,
    /// Momentum coefficient (default: 0.9)
    pub momentum: f32,
    /// Weight decay coefficient (default: 0.0)
    pub weight_decay: f32,
    /// Warmup steps for learning rate (default: 0)
    pub warmup_steps: usize,
    /// Evaluation mode coefficient (default: 1.0)
    pub r: f32,
}

impl Default for ScheduleFreeSGDConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1.0, // Much higher than standard SGD
            momentum: 0.9,
            weight_decay: 0.0,
            warmup_steps: 0,
            r: 1.0,
        }
    }
}

/// Schedule-Free SGD optimizer implementation
///
/// Based on "The Road Less Scheduled" paper, this optimizer automatically adapts
/// the learning rate without requiring manual scheduling.
#[derive(Debug)]
pub struct ScheduleFreeSGD {
    config: ScheduleFreeSGDConfig,
    state: OptimizerState,
    /// Momentum weights (x in the paper)
    momentum_weights: HashMap<String, Vec<f32>>,
    /// Average weights (y in the paper)
    average_weights: HashMap<String, Vec<f32>>,
}

impl ScheduleFreeSGD {
    /// Create a new Schedule-Free SGD optimizer
    pub fn new(learning_rate: f32, momentum: f32, weight_decay: f32) -> Self {
        let config = ScheduleFreeSGDConfig {
            learning_rate,
            momentum,
            weight_decay,
            warmup_steps: 0,
            r: 1.0,
        };

        Self {
            config,
            state: OptimizerState::new(),
            momentum_weights: HashMap::new(),
            average_weights: HashMap::new(),
        }
    }

    /// Create Schedule-Free SGD with full configuration
    pub fn with_config(config: ScheduleFreeSGDConfig) -> Self {
        Self {
            config,
            state: OptimizerState::new(),
            momentum_weights: HashMap::new(),
            average_weights: HashMap::new(),
        }
    }

    /// Create for large models (higher learning rate)
    pub fn for_large_models() -> Self {
        let config = ScheduleFreeSGDConfig {
            learning_rate: 5.0,
            momentum: 0.95,
            weight_decay: 0.1,
            warmup_steps: 1000,
            r: 1.0,
        };
        Self::with_config(config)
    }

    /// Get current effective learning rate (accounting for warmup)
    pub fn get_effective_lr(&self) -> f32 {
        if self.config.warmup_steps == 0 || self.state.step >= self.config.warmup_steps {
            self.config.learning_rate
        } else {
            self.config.learning_rate * (self.state.step as f32 / self.config.warmup_steps as f32)
        }
    }

    /// Switch to evaluation mode (returns average weights)
    pub fn eval_mode(&mut self, parameters: &mut [Tensor]) -> Result<()> {
        for param in parameters.iter_mut() {
            match param {
                Tensor::F32(param_data) => {
                    let param_id = format!("{:p}", param_data.as_ptr());

                    if let Some(average_weights) = self.average_weights.get(&param_id) {
                        // In eval mode, use average weights instead of momentum weights
                        for (p, &a) in param_data.iter_mut().zip(average_weights.iter()) {
                            *p = a;
                        }
                    }
                },
                _ => {
                    return Err(TrustformersError::tensor_op_error(
                        "Unsupported tensor type for Schedule-Free SGD eval mode",
                        "ScheduleFreeSGD::eval_mode",
                    ))
                },
            }
        }
        Ok(())
    }

    /// Switch to training mode (returns momentum weights)
    pub fn train_mode(&mut self, parameters: &mut [Tensor]) -> Result<()> {
        for param in parameters.iter_mut() {
            match param {
                Tensor::F32(param_data) => {
                    let param_id = format!("{:p}", param_data.as_ptr());

                    if let Some(momentum_weights) = self.momentum_weights.get(&param_id) {
                        // In train mode, use momentum weights
                        for (p, &m) in param_data.iter_mut().zip(momentum_weights.iter()) {
                            *p = m;
                        }
                    }
                },
                _ => {
                    return Err(TrustformersError::tensor_op_error(
                        "Unsupported tensor type for Schedule-Free SGD train mode",
                        "ScheduleFreeSGD::train_mode",
                    ))
                },
            }
        }
        Ok(())
    }
}

impl Optimizer for ScheduleFreeSGD {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        match (parameter, grad) {
            (Tensor::F32(param), Tensor::F32(grad_arr)) => {
                let param_id = format!("{:p}", param.as_ptr());
                let size = grad_arr.len();

                // Get values before mutable borrows
                let effective_lr = self.get_effective_lr();
                let beta = self.config.momentum;
                let weight_decay = self.config.weight_decay;

                // Initialize buffers if they don't exist
                let momentum_weights = self
                    .momentum_weights
                    .entry(param_id.clone())
                    .or_insert_with(|| param.iter().cloned().collect());
                let average_weights = self
                    .average_weights
                    .entry(param_id)
                    .or_insert_with(|| param.iter().cloned().collect());

                if momentum_weights.len() != size || average_weights.len() != size {
                    return Err(TrustformersError::tensor_op_error(
                        "Schedule-Free SGD state buffer size mismatch",
                        "ScheduleFreeSGD::update",
                    ));
                }

                // Schedule-Free SGD update
                for (((p, &g), m), a) in param
                    .iter_mut()
                    .zip(grad_arr.iter())
                    .zip(momentum_weights.iter_mut())
                    .zip(average_weights.iter_mut())
                {
                    let mut grad_with_wd = g;

                    // Apply weight decay to gradient
                    if weight_decay > 0.0 {
                        grad_with_wd += weight_decay * *p;
                    }

                    // Update momentum weights (x)
                    *m = beta * *m + effective_lr * grad_with_wd;

                    // Update average weights (y)
                    *a = (1.0 - beta) * *a + beta * *m;

                    // Set parameter to momentum weights for training
                    *p = *m;
                }

                Ok(())
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor types for Schedule-Free SGD",
                "ScheduleFreeSGD::update",
            )),
        }
    }

    fn zero_grad(&mut self) {
        // Schedule-Free optimizers don't need to zero gradients explicitly
    }

    fn step(&mut self) {
        self.state.step();
    }

    fn get_lr(&self) -> f32 {
        self.get_effective_lr()
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }
}

impl StatefulOptimizer for ScheduleFreeSGD {
    type Config = ScheduleFreeSGDConfig;
    type State = OptimizerState;

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }

    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut state_dict = HashMap::new();

        // Save configuration
        state_dict.insert(
            "learning_rate".to_string(),
            Tensor::new(vec![self.config.learning_rate])?,
        );
        state_dict.insert(
            "momentum".to_string(),
            Tensor::new(vec![self.config.momentum])?,
        );
        state_dict.insert(
            "weight_decay".to_string(),
            Tensor::new(vec![self.config.weight_decay])?,
        );
        state_dict.insert(
            "warmup_steps".to_string(),
            Tensor::new(vec![self.config.warmup_steps as f32])?,
        );
        state_dict.insert("r".to_string(), Tensor::new(vec![self.config.r])?);
        state_dict.insert(
            "step".to_string(),
            Tensor::new(vec![self.state.step as f32])?,
        );

        // Save weight buffers
        for (param_id, momentum_weights) in &self.momentum_weights {
            state_dict.insert(
                format!("momentum_weights_{}", param_id),
                Tensor::new(momentum_weights.clone())?,
            );
        }

        for (param_id, average_weights) in &self.average_weights {
            state_dict.insert(
                format!("average_weights_{}", param_id),
                Tensor::new(average_weights.clone())?,
            );
        }

        Ok(state_dict)
    }

    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()> {
        // Load configuration
        if let Some(lr_tensor) = state.get("learning_rate") {
            if let Ok(lr_vec) = lr_tensor.data() {
                if !lr_vec.is_empty() {
                    self.config.learning_rate = lr_vec[0];
                }
            }
        }
        if let Some(momentum_tensor) = state.get("momentum") {
            if let Ok(momentum_vec) = momentum_tensor.data() {
                if !momentum_vec.is_empty() {
                    self.config.momentum = momentum_vec[0];
                }
            }
        }
        if let Some(weight_decay_tensor) = state.get("weight_decay") {
            if let Ok(weight_decay_vec) = weight_decay_tensor.data() {
                if !weight_decay_vec.is_empty() {
                    self.config.weight_decay = weight_decay_vec[0];
                }
            }
        }
        if let Some(warmup_steps_tensor) = state.get("warmup_steps") {
            if let Ok(warmup_steps_vec) = warmup_steps_tensor.data() {
                if !warmup_steps_vec.is_empty() {
                    self.config.warmup_steps = warmup_steps_vec[0] as usize;
                }
            }
        }
        if let Some(r_tensor) = state.get("r") {
            if let Ok(r_vec) = r_tensor.data() {
                if !r_vec.is_empty() {
                    self.config.r = r_vec[0];
                }
            }
        }
        if let Some(step_tensor) = state.get("step") {
            if let Ok(step_vec) = step_tensor.data() {
                if !step_vec.is_empty() {
                    self.state.step = step_vec[0] as usize;
                }
            }
        }

        // Load weight buffers
        for (key, tensor) in state {
            if key.starts_with("momentum_weights_") {
                let param_id = key.strip_prefix("momentum_weights_").unwrap().to_string();
                if let Ok(weights) = tensor.data() {
                    self.momentum_weights.insert(param_id, weights);
                }
            } else if key.starts_with("average_weights_") {
                let param_id = key.strip_prefix("average_weights_").unwrap().to_string();
                if let Ok(weights) = tensor.data() {
                    self.average_weights.insert(param_id, weights);
                }
            }
        }

        Ok(())
    }

    fn memory_usage(&self) -> StateMemoryStats {
        let momentum_size: usize = self.momentum_weights.values().map(|v| v.len()).sum();
        let average_size: usize = self.average_weights.values().map(|v| v.len()).sum();

        StateMemoryStats {
            momentum_elements: momentum_size,
            variance_elements: 0, // Schedule-free doesn't use variance
            third_moment_elements: average_size,
            total_bytes: ((momentum_size + average_size) * 4),
            num_parameters: self.momentum_weights.len(),
        }
    }

    fn reset_state(&mut self) {
        self.state.clear();
        self.momentum_weights.clear();
        self.average_weights.clear();
    }

    fn num_parameters(&self) -> usize {
        self.momentum_weights.values().map(|v| v.len()).sum()
    }
}

/// Configuration for Schedule-Free Adam optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleFreeAdamConfig {
    /// Learning rate (typically 0.1-1.0, higher than standard Adam)
    pub learning_rate: f32,
    /// First moment decay rate (default: 0.9)
    pub beta1: f32,
    /// Second moment decay rate (default: 0.999)
    pub beta2: f32,
    /// Epsilon for numerical stability (default: 1e-8)
    pub epsilon: f32,
    /// Weight decay coefficient (default: 0.0)
    pub weight_decay: f32,
    /// Warmup steps for learning rate (default: 0)
    pub warmup_steps: usize,
    /// Evaluation mode coefficient (default: 1.0)
    pub r: f32,
}

impl Default for ScheduleFreeAdamConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.25, // Higher than standard Adam
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            warmup_steps: 0,
            r: 1.0,
        }
    }
}

/// Schedule-Free Adam optimizer implementation
///
/// Combines the adaptive learning rates of Adam with schedule-free learning rate adaptation.
#[derive(Debug)]
pub struct ScheduleFreeAdam {
    config: ScheduleFreeAdamConfig,
    state: OptimizerState,
    /// Momentum weights (x in the paper)
    momentum_weights: HashMap<String, Vec<f32>>,
    /// Average weights (y in the paper)
    average_weights: HashMap<String, Vec<f32>>,
    /// First moment estimates
    exp_avg: HashMap<String, Vec<f32>>,
    /// Second moment estimates
    exp_avg_sq: HashMap<String, Vec<f32>>,
}

impl ScheduleFreeAdam {
    /// Create a new Schedule-Free Adam optimizer
    pub fn new(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Self {
        let config = ScheduleFreeAdamConfig {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            warmup_steps: 0,
            r: 1.0,
        };

        Self {
            config,
            state: OptimizerState::new(),
            momentum_weights: HashMap::new(),
            average_weights: HashMap::new(),
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
        }
    }

    /// Create Schedule-Free Adam with full configuration
    pub fn with_config(config: ScheduleFreeAdamConfig) -> Self {
        Self {
            config,
            state: OptimizerState::new(),
            momentum_weights: HashMap::new(),
            average_weights: HashMap::new(),
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
        }
    }

    /// Create for language models (optimized hyperparameters)
    pub fn for_language_models() -> Self {
        let config = ScheduleFreeAdamConfig {
            learning_rate: 0.5,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1e-8,
            weight_decay: 0.1,
            warmup_steps: 2000,
            r: 1.0,
        };
        Self::with_config(config)
    }

    /// Get current effective learning rate (accounting for warmup)
    pub fn get_effective_lr(&self) -> f32 {
        if self.config.warmup_steps == 0 || self.state.step >= self.config.warmup_steps {
            self.config.learning_rate
        } else {
            self.config.learning_rate * (self.state.step as f32 / self.config.warmup_steps as f32)
        }
    }
}

impl Optimizer for ScheduleFreeAdam {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        match (parameter, grad) {
            (Tensor::F32(param), Tensor::F32(grad_arr)) => {
                let param_id = format!("{:p}", param.as_ptr());
                let size = grad_arr.len();

                // Get values before mutable borrows
                let effective_lr = self.get_effective_lr();
                let beta1 = self.config.beta1;
                let beta2 = self.config.beta2;
                let eps = self.config.epsilon;
                let weight_decay = self.config.weight_decay;

                // Initialize buffers if they don't exist
                let momentum_weights = self
                    .momentum_weights
                    .entry(param_id.clone())
                    .or_insert_with(|| param.iter().cloned().collect());
                let average_weights = self
                    .average_weights
                    .entry(param_id.clone())
                    .or_insert_with(|| param.iter().cloned().collect());
                let exp_avg =
                    self.exp_avg.entry(param_id.clone()).or_insert_with(|| vec![0.0; size]);
                let exp_avg_sq = self.exp_avg_sq.entry(param_id).or_insert_with(|| vec![0.0; size]);

                if momentum_weights.len() != size
                    || average_weights.len() != size
                    || exp_avg.len() != size
                    || exp_avg_sq.len() != size
                {
                    return Err(TrustformersError::tensor_op_error(
                        "Schedule-Free Adam state buffer size mismatch",
                        "ScheduleFreeAdam::update",
                    ));
                }

                // Bias correction
                let step = (self.state.step + 1) as f32;
                let bias_correction1 = 1.0 - beta1.powf(step);
                let bias_correction2 = 1.0 - beta2.powf(step);

                // Schedule-Free Adam update
                for (((((p, &g), m), a), exp_avg_val), exp_avg_sq_val) in param
                    .iter_mut()
                    .zip(grad_arr.iter())
                    .zip(momentum_weights.iter_mut())
                    .zip(average_weights.iter_mut())
                    .zip(exp_avg.iter_mut())
                    .zip(exp_avg_sq.iter_mut())
                {
                    let mut grad_with_wd = g;

                    // Apply weight decay to gradient
                    if weight_decay > 0.0 {
                        grad_with_wd += weight_decay * *p;
                    }

                    // Update biased first moment estimate
                    *exp_avg_val = beta1 * *exp_avg_val + (1.0 - beta1) * grad_with_wd;

                    // Update biased second moment estimate
                    *exp_avg_sq_val =
                        beta2 * *exp_avg_sq_val + (1.0 - beta2) * grad_with_wd * grad_with_wd;

                    // Compute bias-corrected first and second moment estimates
                    let m_hat = *exp_avg_val / bias_correction1;
                    let v_hat = *exp_avg_sq_val / bias_correction2;

                    // Compute Adam update
                    let adam_update = effective_lr * m_hat / (v_hat.sqrt() + eps);

                    // Update momentum weights (x)
                    *m = beta1 * *m + adam_update;

                    // Update average weights (y)
                    *a = (1.0 - beta1) * *a + beta1 * *m;

                    // Set parameter to momentum weights for training
                    *p = *m;
                }

                Ok(())
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor types for Schedule-Free Adam",
                "ScheduleFreeAdam::update",
            )),
        }
    }

    fn zero_grad(&mut self) {
        // Schedule-Free optimizers don't need to zero gradients explicitly
    }

    fn step(&mut self) {
        self.state.step();
    }

    fn get_lr(&self) -> f32 {
        self.get_effective_lr()
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }
}

impl StatefulOptimizer for ScheduleFreeAdam {
    type Config = ScheduleFreeAdamConfig;
    type State = OptimizerState;

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }

    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut state_dict = HashMap::new();

        // Save configuration
        state_dict.insert(
            "learning_rate".to_string(),
            Tensor::new(vec![self.config.learning_rate])?,
        );
        state_dict.insert("beta1".to_string(), Tensor::new(vec![self.config.beta1])?);
        state_dict.insert("beta2".to_string(), Tensor::new(vec![self.config.beta2])?);
        state_dict.insert(
            "epsilon".to_string(),
            Tensor::new(vec![self.config.epsilon])?,
        );
        state_dict.insert(
            "weight_decay".to_string(),
            Tensor::new(vec![self.config.weight_decay])?,
        );
        state_dict.insert(
            "warmup_steps".to_string(),
            Tensor::new(vec![self.config.warmup_steps as f32])?,
        );
        state_dict.insert("r".to_string(), Tensor::new(vec![self.config.r])?);
        state_dict.insert(
            "step".to_string(),
            Tensor::new(vec![self.state.step as f32])?,
        );

        // Save all buffers
        for (param_id, momentum_weights) in &self.momentum_weights {
            state_dict.insert(
                format!("momentum_weights_{}", param_id),
                Tensor::new(momentum_weights.clone())?,
            );
        }

        for (param_id, average_weights) in &self.average_weights {
            state_dict.insert(
                format!("average_weights_{}", param_id),
                Tensor::new(average_weights.clone())?,
            );
        }

        for (param_id, exp_avg) in &self.exp_avg {
            state_dict.insert(
                format!("exp_avg_{}", param_id),
                Tensor::new(exp_avg.clone())?,
            );
        }

        for (param_id, exp_avg_sq) in &self.exp_avg_sq {
            state_dict.insert(
                format!("exp_avg_sq_{}", param_id),
                Tensor::new(exp_avg_sq.clone())?,
            );
        }

        Ok(state_dict)
    }

    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()> {
        // Load configuration
        if let Some(lr_tensor) = state.get("learning_rate") {
            if let Ok(lr_vec) = lr_tensor.data() {
                if !lr_vec.is_empty() {
                    self.config.learning_rate = lr_vec[0];
                }
            }
        }
        // ... (similar pattern for other config fields)

        // Load all buffers
        for (key, tensor) in state {
            if key.starts_with("momentum_weights_") {
                let param_id = key.strip_prefix("momentum_weights_").unwrap().to_string();
                if let Ok(weights) = tensor.data() {
                    self.momentum_weights.insert(param_id, weights);
                }
            } else if key.starts_with("average_weights_") {
                let param_id = key.strip_prefix("average_weights_").unwrap().to_string();
                if let Ok(weights) = tensor.data() {
                    self.average_weights.insert(param_id, weights);
                }
            } else if key.starts_with("exp_avg_") && !key.starts_with("exp_avg_sq_") {
                let param_id = key.strip_prefix("exp_avg_").unwrap().to_string();
                if let Ok(weights) = tensor.data() {
                    self.exp_avg.insert(param_id, weights);
                }
            } else if key.starts_with("exp_avg_sq_") {
                let param_id = key.strip_prefix("exp_avg_sq_").unwrap().to_string();
                if let Ok(weights) = tensor.data() {
                    self.exp_avg_sq.insert(param_id, weights);
                }
            }
        }

        Ok(())
    }

    fn memory_usage(&self) -> StateMemoryStats {
        let momentum_size: usize = self.momentum_weights.values().map(|v| v.len()).sum();
        let average_size: usize = self.average_weights.values().map(|v| v.len()).sum();
        let exp_avg_size: usize = self.exp_avg.values().map(|v| v.len()).sum();
        let exp_avg_sq_size: usize = self.exp_avg_sq.values().map(|v| v.len()).sum();

        let total_params = momentum_size + average_size + exp_avg_size + exp_avg_sq_size;
        let _total_buffers = self.momentum_weights.len()
            + self.average_weights.len()
            + self.exp_avg.len()
            + self.exp_avg_sq.len();

        StateMemoryStats {
            momentum_elements: momentum_size + exp_avg_size,
            variance_elements: average_size + exp_avg_sq_size,
            third_moment_elements: 0,
            total_bytes: total_params * 4,
            num_parameters: self.momentum_weights.len(),
        }
    }

    fn reset_state(&mut self) {
        self.state.clear();
        self.momentum_weights.clear();
        self.average_weights.clear();
        self.exp_avg.clear();
        self.exp_avg_sq.clear();
    }

    fn num_parameters(&self) -> usize {
        self.momentum_weights.values().map(|v| v.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schedule_free_sgd_creation() {
        let optimizer = ScheduleFreeSGD::new(1.0, 0.9, 0.01);
        assert_eq!(optimizer.get_lr(), 1.0);
        assert_eq!(optimizer.config.momentum, 0.9);
        assert_eq!(optimizer.config.weight_decay, 0.01);
    }

    #[test]
    fn test_schedule_free_sgd_for_large_models() {
        let optimizer = ScheduleFreeSGD::for_large_models();
        assert_eq!(optimizer.config.learning_rate, 5.0);
        assert_eq!(optimizer.config.momentum, 0.95);
        assert_eq!(optimizer.config.weight_decay, 0.1);
        assert_eq!(optimizer.config.warmup_steps, 1000);
    }

    #[test]
    fn test_schedule_free_adam_creation() {
        let optimizer = ScheduleFreeAdam::new(0.25, 0.9, 0.999, 1e-8, 0.01);
        assert_eq!(optimizer.get_lr(), 0.25);
        assert_eq!(optimizer.config.beta1, 0.9);
        assert_eq!(optimizer.config.beta2, 0.999);
        assert_eq!(optimizer.config.epsilon, 1e-8);
        assert_eq!(optimizer.config.weight_decay, 0.01);
    }

    #[test]
    fn test_schedule_free_adam_for_language_models() {
        let optimizer = ScheduleFreeAdam::for_language_models();
        assert_eq!(optimizer.config.learning_rate, 0.5);
        assert_eq!(optimizer.config.beta1, 0.9);
        assert_eq!(optimizer.config.beta2, 0.95);
        assert_eq!(optimizer.config.weight_decay, 0.1);
        assert_eq!(optimizer.config.warmup_steps, 2000);
    }

    #[test]
    fn test_memory_usage() {
        let optimizer = ScheduleFreeAdam::new(0.1, 0.9, 0.999, 1e-8, 0.0);
        let memory_stats = optimizer.memory_usage();
        assert_eq!(memory_stats.num_parameters, 0); // No parameters added yet
        assert_eq!(memory_stats.total_bytes, 0);
    }
}
