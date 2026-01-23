//! # Adan Optimizer
//!
//! Adan (Adaptive Nesterov Momentum) is a cutting-edge optimizer that combines the benefits
//! of Adam and Nesterov momentum for improved convergence and training stability.
//!
//! ## Key Features
//!
//! - **Adaptive Nesterov Momentum**: Combines adaptive learning rates with Nesterov acceleration
//! - **Three Momentum Terms**: Uses first, second, and third order momentum estimates
//! - **Improved Convergence**: Better convergence properties compared to Adam/AdamW
//! - **Training Stability**: More stable training for large models and learning rates
//!
//! ## Algorithm
//!
//! Adan maintains three exponential moving averages:
//! 1. First moment (gradient momentum)
//! 2. Second moment (squared gradient momentum)
//! 3. Third moment (difference momentum for Nesterov-like acceleration)
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use trustformers_optim::Adan;
//!
//! let mut optimizer = Adan::new(
//!     1e-3,   // learning_rate
//!     0.98,   // beta1
//!     0.92,   // beta2
//!     0.99,   // beta3
//!     1e-8,   // epsilon
//!     0.02,   // weight_decay
//! );
//! ```
//!
//! ## Reference
//!
//! "Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models"
//! by Xie et al. (2022)

use crate::common::{OptimizerState, StateMemoryStats};
use crate::traits::StatefulOptimizer;
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

/// Configuration for Adan optimizer.
#[derive(Debug, Clone)]
pub struct AdanConfig {
    /// Learning rate
    pub lr: f32,
    /// First moment coefficient
    pub beta1: f32,
    /// Second moment coefficient
    pub beta2: f32,
    /// Third moment coefficient (for Nesterov-like acceleration)
    pub beta3: f32,
    /// Term added for numerical stability
    pub eps: f32,
    /// Weight decay (L2 penalty)
    pub weight_decay: f32,
    /// Whether to use bias correction
    pub bias_correction: bool,
    /// Whether to use decoupled weight decay (AdamW style)
    pub decoupled_weight_decay: bool,
}

impl Default for AdanConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.98,
            beta2: 0.92,
            beta3: 0.99,
            eps: 1e-8,
            weight_decay: 0.02,
            bias_correction: true,
            decoupled_weight_decay: true,
        }
    }
}

/// Adan (Adaptive Nesterov Momentum) optimizer.
#[derive(Debug)]
pub struct Adan {
    config: AdanConfig,
    state: OptimizerState,
    exp_avg: HashMap<String, Vec<f32>>,
    exp_avg_sq: HashMap<String, Vec<f32>>,
    exp_avg_diff: HashMap<String, Vec<f32>>,
    prev_grad: HashMap<String, Vec<f32>>,
    step_count: usize,
}

impl Adan {
    /// Creates a new Adan optimizer with default configuration.
    pub fn new(lr: f32, beta1: f32, beta2: f32, beta3: f32, eps: f32, weight_decay: f32) -> Self {
        let config = AdanConfig {
            lr,
            beta1,
            beta2,
            beta3,
            eps,
            weight_decay,
            bias_correction: true,
            decoupled_weight_decay: true,
        };

        Self::with_config(config)
    }

    /// Creates a new Adan optimizer with custom configuration.
    pub fn with_config(config: AdanConfig) -> Self {
        Self {
            config,
            state: OptimizerState::new(),
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
            exp_avg_diff: HashMap::new(),
            prev_grad: HashMap::new(),
            step_count: 0,
        }
    }

    /// Convenience constructor for Adan with recommended settings for large models.
    pub fn for_large_models(lr: f32, weight_decay: f32) -> Self {
        Self::new(lr, 0.98, 0.92, 0.99, 1e-8, weight_decay)
    }

    /// Convenience constructor for Adan with recommended settings for vision models.
    pub fn for_vision(lr: f32, weight_decay: f32) -> Self {
        Self::new(lr, 0.9, 0.999, 0.9999, 1e-8, weight_decay)
    }

    /// Convenience constructor for Adan with Adam-like settings.
    pub fn adam_like(lr: f32, weight_decay: f32) -> Self {
        Self::new(lr, 0.9, 0.999, 0.999, 1e-8, weight_decay)
    }

    /// Gets the current learning rate.
    pub fn get_lr(&self) -> f32 {
        self.config.lr
    }

    /// Sets the learning rate.
    pub fn set_lr(&mut self, lr: f32) {
        self.config.lr = lr;
    }

    /// Gets current optimizer configuration.
    pub fn config(&self) -> &AdanConfig {
        &self.config
    }

    /// Gets memory statistics for the optimizer state.
    pub fn memory_stats(&self) -> StateMemoryStats {
        let mut total_parameters = 0;
        #[allow(dead_code)]
        let mut _total_buffers = 0;

        for buffer in self.exp_avg.values() {
            total_parameters += buffer.len();
            _total_buffers += 1;
        }

        for buffer in self.exp_avg_sq.values() {
            total_parameters += buffer.len();
            _total_buffers += 1;
        }

        for buffer in self.exp_avg_diff.values() {
            total_parameters += buffer.len();
            _total_buffers += 1;
        }

        for buffer in self.prev_grad.values() {
            total_parameters += buffer.len();
            _total_buffers += 1;
        }

        StateMemoryStats {
            momentum_elements: total_parameters,
            variance_elements: total_parameters,
            third_moment_elements: total_parameters,
            total_bytes: total_parameters * 4, // f32 = 4 bytes
            num_parameters: total_parameters,
        }
    }
}

impl Optimizer for Adan {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        self.step_count += 1;

        match (parameter, grad) {
            (Tensor::F32(param), Tensor::F32(grad_data)) => {
                let param_id = format!("{:p}", param.as_ptr());
                let size = grad_data.len();

                // Initialize state if needed
                let exp_avg =
                    self.exp_avg.entry(param_id.clone()).or_insert_with(|| vec![0.0; size]);
                let exp_avg_sq =
                    self.exp_avg_sq.entry(param_id.clone()).or_insert_with(|| vec![0.0; size]);
                let exp_avg_diff =
                    self.exp_avg_diff.entry(param_id.clone()).or_insert_with(|| vec![0.0; size]);
                let prev_grad =
                    self.prev_grad.entry(param_id.clone()).or_insert_with(|| vec![0.0; size]);

                // Check buffer sizes
                if exp_avg.len() != size
                    || exp_avg_sq.len() != size
                    || exp_avg_diff.len() != size
                    || prev_grad.len() != size
                {
                    return Err(TrustformersError::tensor_op_error(
                        "Adan buffer size mismatch",
                        "Adan::update",
                    ));
                }

                // Compute bias correction factors
                let bias_correction1 = if self.config.bias_correction {
                    1.0 - self.config.beta1.powi(self.step_count as i32)
                } else {
                    1.0
                };

                let bias_correction2 = if self.config.bias_correction {
                    1.0 - self.config.beta2.powi(self.step_count as i32)
                } else {
                    1.0
                };

                let bias_correction3 = if self.config.bias_correction {
                    1.0 - self.config.beta3.powi(self.step_count as i32)
                } else {
                    1.0
                };

                // Update parameters
                for ((&g, p), ((m, v), (n, pg))) in grad_data.iter().zip(param.iter_mut()).zip(
                    exp_avg
                        .iter_mut()
                        .zip(exp_avg_sq.iter_mut())
                        .zip(exp_avg_diff.iter_mut().zip(prev_grad.iter_mut())),
                ) {
                    // Apply decoupled weight decay (AdamW style)
                    if self.config.decoupled_weight_decay && self.config.weight_decay > 0.0 {
                        *p *= 1.0 - self.config.lr * self.config.weight_decay;
                    }

                    // Apply L2 weight decay (Adam style)
                    let grad_with_decay =
                        if !self.config.decoupled_weight_decay && self.config.weight_decay > 0.0 {
                            g + self.config.weight_decay * (*p)
                        } else {
                            g
                        };

                    // Compute gradient difference (for Nesterov-like acceleration)
                    let grad_diff = grad_with_decay - *pg;

                    // Update biased first moment estimate
                    *m = self.config.beta1 * (*m) + (1.0 - self.config.beta1) * grad_with_decay;

                    // Update biased second moment estimate
                    *v = self.config.beta2 * (*v)
                        + (1.0 - self.config.beta2) * grad_with_decay * grad_with_decay;

                    // Update biased third moment estimate (difference momentum)
                    *n = self.config.beta3 * (*n) + (1.0 - self.config.beta3) * grad_diff;

                    // Store current gradient for next iteration
                    *pg = grad_with_decay;

                    // Compute bias-corrected estimates
                    let m_hat = *m / bias_correction1;
                    let v_hat = *v / bias_correction2;
                    let n_hat = *n / bias_correction3;

                    // Compute the update direction with Nesterov-like acceleration
                    let update_direction = m_hat + self.config.beta2 * n_hat;

                    // Update parameter
                    *p -= self.config.lr * update_direction / (v_hat.sqrt() + self.config.eps);
                }

                Ok(())
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Adan optimizer only supports F32 tensors",
                "Adan::update",
            )),
        }
    }

    fn zero_grad(&mut self) {
        // Adan doesn't accumulate gradients, so this is a no-op
    }

    fn step(&mut self) {
        // Update is called per parameter, so step is a no-op
    }

    fn get_lr(&self) -> f32 {
        self.config.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.lr = lr;
    }
}

impl StatefulOptimizer for Adan {
    type Config = AdanConfig;
    type State = OptimizerState;

    fn state(&self) -> &OptimizerState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut OptimizerState {
        &mut self.state
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn memory_usage(&self) -> StateMemoryStats {
        self.memory_stats()
    }

    fn reset_state(&mut self) {
        self.exp_avg.clear();
        self.exp_avg_sq.clear();
        self.exp_avg_diff.clear();
        self.prev_grad.clear();
        self.step_count = 0;
        self.state = OptimizerState::new();
    }

    fn num_parameters(&self) -> usize {
        self.exp_avg.values().map(|v| v.len()).sum()
    }

    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut dict = HashMap::new();

        for (key, value) in &self.exp_avg {
            dict.insert(format!("exp_avg_{}", key), Tensor::new(value.clone())?);
        }

        for (key, value) in &self.exp_avg_sq {
            dict.insert(format!("exp_avg_sq_{}", key), Tensor::new(value.clone())?);
        }

        for (key, value) in &self.exp_avg_diff {
            dict.insert(format!("exp_avg_diff_{}", key), Tensor::new(value.clone())?);
        }

        for (key, value) in &self.prev_grad {
            dict.insert(format!("prev_grad_{}", key), Tensor::new(value.clone())?);
        }

        dict.insert(
            "step_count".to_string(),
            Tensor::new(vec![self.step_count as f32])?,
        );

        Ok(dict)
    }

    fn load_state_dict(&mut self, state_dict: HashMap<String, Tensor>) -> Result<()> {
        // Load step count
        if let Some(Tensor::F32(data)) = state_dict.get("step_count") {
            if !data.is_empty() {
                self.step_count = data[0] as usize;
            }
        }

        // Load exp_avg
        for (key, value) in &state_dict {
            if let Some(param_key) = key.strip_prefix("exp_avg_") {
                if let Tensor::F32(data) = value {
                    self.exp_avg.insert(param_key.to_string(), data.as_slice().unwrap().to_vec());
                }
            }
        }

        // Load exp_avg_sq
        for (key, value) in &state_dict {
            if let Some(param_key) = key.strip_prefix("exp_avg_sq_") {
                if let Tensor::F32(data) = value {
                    self.exp_avg_sq
                        .insert(param_key.to_string(), data.as_slice().unwrap().to_vec());
                }
            }
        }

        // Load exp_avg_diff
        for (key, value) in &state_dict {
            if let Some(param_key) = key.strip_prefix("exp_avg_diff_") {
                if let Tensor::F32(data) = value {
                    self.exp_avg_diff
                        .insert(param_key.to_string(), data.as_slice().unwrap().to_vec());
                }
            }
        }

        // Load prev_grad
        for (key, value) in &state_dict {
            if let Some(param_key) = key.strip_prefix("prev_grad_") {
                if let Tensor::F32(data) = value {
                    self.prev_grad.insert(param_key.to_string(), data.as_slice().unwrap().to_vec());
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::tensor::Tensor;

    #[test]
    fn test_adan_creation() {
        let optimizer = Adan::new(1e-3, 0.98, 0.92, 0.99, 1e-8, 0.02);
        assert_eq!(optimizer.get_lr(), 1e-3);
        assert_eq!(optimizer.config().beta1, 0.98);
        assert_eq!(optimizer.config().beta2, 0.92);
        assert_eq!(optimizer.config().beta3, 0.99);
        assert_eq!(optimizer.config().eps, 1e-8);
        assert_eq!(optimizer.config().weight_decay, 0.02);
    }

    #[test]
    fn test_adan_for_large_models() {
        let optimizer = Adan::for_large_models(1e-3, 0.02);
        assert_eq!(optimizer.get_lr(), 1e-3);
        assert_eq!(optimizer.config().beta1, 0.98);
        assert_eq!(optimizer.config().beta2, 0.92);
        assert_eq!(optimizer.config().beta3, 0.99);
        assert_eq!(optimizer.config().weight_decay, 0.02);
    }

    #[test]
    fn test_adan_for_vision() {
        let optimizer = Adan::for_vision(1e-3, 0.02);
        assert_eq!(optimizer.get_lr(), 1e-3);
        assert_eq!(optimizer.config().beta1, 0.9);
        assert_eq!(optimizer.config().beta2, 0.999);
        assert_eq!(optimizer.config().beta3, 0.9999);
    }

    #[test]
    fn test_adan_adam_like() {
        let optimizer = Adan::adam_like(1e-3, 0.02);
        assert_eq!(optimizer.get_lr(), 1e-3);
        assert_eq!(optimizer.config().beta1, 0.9);
        assert_eq!(optimizer.config().beta2, 0.999);
        assert_eq!(optimizer.config().beta3, 0.999);
    }

    #[test]
    fn test_adan_lr_setter() {
        let mut optimizer = Adan::new(1e-3, 0.98, 0.92, 0.99, 1e-8, 0.02);
        optimizer.set_lr(2e-3);
        assert_eq!(optimizer.get_lr(), 2e-3);
    }

    #[test]
    fn test_adan_memory_stats() {
        let optimizer = Adan::new(1e-3, 0.98, 0.92, 0.99, 1e-8, 0.02);
        let stats = optimizer.memory_stats();
        assert_eq!(stats.num_parameters, 0);
        assert_eq!(stats.total_bytes, 0);
    }

    #[test]
    fn test_adan_state_dict() {
        let optimizer = Adan::new(1e-3, 0.98, 0.92, 0.99, 1e-8, 0.02);
        let state_dict = optimizer.state_dict();
        assert!(state_dict.unwrap().contains_key("step_count"));
    }

    #[test]
    fn test_adan_load_state_dict() {
        let mut optimizer = Adan::new(1e-3, 0.98, 0.92, 0.99, 1e-8, 0.02);
        let mut state_dict = HashMap::new();
        state_dict.insert("step_count".to_string(), Tensor::new(vec![10.0]).unwrap());

        optimizer.load_state_dict(state_dict).unwrap();
        assert_eq!(optimizer.step_count, 10);
    }

    #[test]
    fn test_adan_with_config() {
        let config = AdanConfig {
            lr: 2e-3,
            beta1: 0.95,
            beta2: 0.90,
            beta3: 0.95,
            eps: 1e-7,
            weight_decay: 0.01,
            bias_correction: false,
            decoupled_weight_decay: false,
        };

        let optimizer = Adan::with_config(config);
        assert_eq!(optimizer.get_lr(), 2e-3);
        assert_eq!(optimizer.config().beta1, 0.95);
        assert!(!optimizer.config().bias_correction);
        assert!(!optimizer.config().decoupled_weight_decay);
    }
}
