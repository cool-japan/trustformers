//! Common optimization operations and utilities.
//!
//! This module provides shared functionality that is used across multiple optimizers,
//! reducing code duplication and ensuring consistent behavior.
//!
//! # Features
//!
//! - **State Management**: Unified parameter state tracking
//! - **Bias Correction**: Standard bias correction calculations for momentum methods
//! - **Parameter Updates**: Common update patterns with weight decay variants
//! - **Gradient Processing**: Shared gradient manipulation utilities
//! - **Memory Management**: Efficient buffer allocation and reuse

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;

/// Unified state management for optimizer parameters.
///
/// This struct provides a consistent interface for tracking optimizer state
/// across different algorithms, reducing code duplication and memory overhead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    /// Current step counter for bias correction and scheduling
    pub step: usize,

    /// First moment estimates (momentum buffers)
    pub momentum: HashMap<String, Vec<f32>>,

    /// Second moment estimates (squared gradient buffers)
    pub variance: HashMap<String, Vec<f32>>,

    /// Optional third moment estimates (for higher-order methods)
    pub third_moment: HashMap<String, Vec<f32>>,

    /// Per-parameter step counts (for adaptive methods)
    pub param_steps: HashMap<String, usize>,

    /// Velocity buffers for optimization methods like SGD with momentum
    pub velocity: HashMap<String, Vec<f32>>,
}

impl OptimizerState {
    /// Creates a new optimizer state with empty buffers.
    pub fn new() -> Self {
        Self {
            step: 0,
            momentum: HashMap::new(),
            variance: HashMap::new(),
            third_moment: HashMap::new(),
            param_steps: HashMap::new(),
            velocity: HashMap::new(),
        }
    }

    /// Gets or creates momentum buffer for a parameter.
    pub fn get_or_create_momentum(&mut self, param_id: String, size: usize) -> &mut Vec<f32> {
        self.momentum.entry(param_id).or_insert_with(|| vec![0.0; size])
    }

    /// Gets or creates variance buffer for a parameter.
    pub fn get_or_create_variance(&mut self, param_id: String, size: usize) -> &mut Vec<f32> {
        self.variance.entry(param_id).or_insert_with(|| vec![0.0; size])
    }

    /// Gets or creates third moment buffer for a parameter.
    pub fn get_or_create_third_moment(&mut self, param_id: String, size: usize) -> &mut Vec<f32> {
        self.third_moment.entry(param_id).or_insert_with(|| vec![0.0; size])
    }

    /// Increments the global step counter.
    pub fn step(&mut self) {
        self.step += 1;
    }

    /// Increments the step counter for a specific parameter.
    pub fn step_param(&mut self, param_id: String) {
        *self.param_steps.entry(param_id).or_insert(0) += 1;
    }

    /// Gets the step count for a specific parameter.
    pub fn get_param_step(&self, param_id: &str) -> usize {
        self.param_steps.get(param_id).copied().unwrap_or(0)
    }

    /// Clears all state buffers to free memory.
    pub fn clear(&mut self) {
        self.step = 0;
        self.momentum.clear();
        self.variance.clear();
        self.third_moment.clear();
        self.param_steps.clear();
    }

    /// Gets memory usage statistics.
    pub fn memory_usage(&self) -> StateMemoryStats {
        let momentum_size: usize = self.momentum.values().map(|v| v.len()).sum();
        let variance_size: usize = self.variance.values().map(|v| v.len()).sum();
        let third_moment_size: usize = self.third_moment.values().map(|v| v.len()).sum();

        StateMemoryStats {
            momentum_elements: momentum_size,
            variance_elements: variance_size,
            third_moment_elements: third_moment_size,
            total_bytes: (momentum_size + variance_size + third_moment_size)
                * std::mem::size_of::<f32>(),
            num_parameters: self.momentum.len(),
        }
    }
}

impl Default for OptimizerState {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory usage statistics for optimizer state.
#[derive(Debug, Clone)]
pub struct StateMemoryStats {
    pub momentum_elements: usize,
    pub variance_elements: usize,
    pub third_moment_elements: usize,
    pub total_bytes: usize,
    pub num_parameters: usize,
}

/// Common bias correction utilities for momentum-based optimizers.
pub struct BiasCorrection;

impl BiasCorrection {
    /// Computes bias correction factor for exponential moving averages.
    ///
    /// Formula: 1 - beta^step
    ///
    /// # Arguments
    ///
    /// * `beta` - The exponential decay rate (e.g., 0.9 for momentum, 0.999 for variance)
    /// * `step` - The current step number (1-indexed)
    pub fn compute_correction(beta: f32, step: usize) -> f32 {
        1.0 - beta.powi(step as i32)
    }

    /// Applies bias correction to a value.
    ///
    /// # Arguments
    ///
    /// * `value` - The biased estimate
    /// * `beta` - The exponential decay rate
    /// * `step` - The current step number (1-indexed)
    pub fn apply_correction(value: f32, beta: f32, step: usize) -> f32 {
        value / Self::compute_correction(beta, step)
    }

    /// Computes both first and second moment bias corrections.
    ///
    /// # Returns
    ///
    /// Tuple of (bias_correction1, bias_correction2) for Adam-style optimizers.
    pub fn compute_adam_corrections(beta1: f32, beta2: f32, step: usize) -> (f32, f32) {
        (
            Self::compute_correction(beta1, step),
            Self::compute_correction(beta2, step),
        )
    }
}

/// Weight decay application strategies.
#[derive(Debug, Clone)]
pub enum WeightDecayMode {
    /// L2 regularization applied to gradients (traditional Adam)
    L2Regularization,
    /// Decoupled weight decay applied directly to parameters (AdamW style)
    Decoupled,
}

/// Common parameter update operations.
pub struct ParameterUpdate;

impl ParameterUpdate {
    /// Applies weight decay to gradients (L2 regularization).
    ///
    /// # Arguments
    ///
    /// * `grad` - The gradient value
    /// * `param` - The parameter value
    /// * `weight_decay` - The weight decay coefficient
    pub fn apply_l2_regularization(grad: f32, param: f32, weight_decay: f32) -> f32 {
        grad + weight_decay * param
    }

    /// Applies decoupled weight decay directly to parameter.
    ///
    /// # Arguments
    ///
    /// * `param` - The parameter value to update
    /// * `lr` - The learning rate
    /// * `weight_decay` - The weight decay coefficient
    pub fn apply_decoupled_weight_decay(param: &mut f32, lr: f32, weight_decay: f32) {
        *param *= 1.0 - lr * weight_decay;
    }

    /// Updates parameter using Adam-style formula.
    ///
    /// # Arguments
    ///
    /// * `param` - The parameter to update
    /// * `lr` - Learning rate
    /// * `m_hat` - Bias-corrected first moment
    /// * `v_hat` - Bias-corrected second moment
    /// * `eps` - Epsilon for numerical stability
    pub fn adam_update(param: &mut f32, lr: f32, m_hat: f32, v_hat: f32, eps: f32) {
        *param -= lr * m_hat / (v_hat.sqrt() + eps);
    }

    /// Updates parameter using SGD with momentum.
    ///
    /// # Arguments
    ///
    /// * `param` - The parameter to update
    /// * `lr` - Learning rate
    /// * `momentum` - Momentum buffer value
    pub fn sgd_momentum_update(param: &mut f32, lr: f32, momentum: f32) {
        *param -= lr * momentum;
    }

    /// Updates momentum buffer for SGD.
    ///
    /// # Arguments
    ///
    /// * `momentum` - The momentum buffer to update
    /// * `grad` - The gradient
    /// * `momentum_coeff` - Momentum coefficient (typically 0.9)
    /// * `dampening` - Dampening factor (typically 0.0)
    /// * `nesterov` - Whether to use Nesterov momentum
    pub fn update_sgd_momentum(
        momentum: &mut f32,
        grad: f32,
        momentum_coeff: f32,
        dampening: f32,
        nesterov: bool,
    ) -> f32 {
        *momentum = momentum_coeff * *momentum + (1.0 - dampening) * grad;
        if nesterov {
            grad + momentum_coeff * *momentum
        } else {
            *momentum
        }
    }

    /// Updates exponential moving average (for Adam-style methods).
    ///
    /// # Arguments
    ///
    /// * `ema` - The exponential moving average to update
    /// * `value` - The new value
    /// * `beta` - The decay coefficient
    pub fn update_ema(ema: &mut f32, value: f32, beta: f32) {
        *ema = beta * *ema + (1.0 - beta) * value;
    }
}

/// Gradient processing utilities.
#[derive(Debug, Clone)]
pub struct GradientProcessor;

impl GradientProcessor {
    /// Clips gradient by norm.
    ///
    /// # Arguments
    ///
    /// * `grad` - The gradient to clip
    /// * `max_norm` - Maximum allowed norm
    pub fn clip_by_norm(grad: &mut [f32], max_norm: f32) {
        let norm: f32 = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
        if norm > max_norm {
            let scale = max_norm / norm;
            for g in grad.iter_mut() {
                *g *= scale;
            }
        }
    }

    /// Clips gradient by value.
    ///
    /// # Arguments
    ///
    /// * `grad` - The gradient to clip
    /// * `min_value` - Minimum allowed value
    /// * `max_value` - Maximum allowed value
    pub fn clip_by_value(grad: &mut [f32], min_value: f32, max_value: f32) {
        for g in grad.iter_mut() {
            *g = g.clamp(min_value, max_value);
        }
    }

    /// Applies gradient scaling for mixed precision training.
    ///
    /// # Arguments
    ///
    /// * `grad` - The gradient to scale
    /// * `scale` - The scaling factor
    pub fn scale_gradient(grad: &mut [f32], scale: f32) {
        for g in grad.iter_mut() {
            *g *= scale;
        }
    }

    /// Checks for non-finite gradients (NaN or Inf).
    ///
    /// # Arguments
    ///
    /// * `grad` - The gradient to check
    ///
    /// # Returns
    ///
    /// True if all gradients are finite.
    pub fn is_finite(grad: &[f32]) -> bool {
        grad.iter().all(|g| g.is_finite())
    }
}

/// Utility functions for creating parameter IDs.
pub struct ParameterIds;

impl ParameterIds {
    /// Creates a unique parameter ID from tensor pointer.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to create ID for
    pub fn from_tensor(tensor: &Tensor) -> Result<String> {
        match tensor {
            Tensor::F32(data) => Ok(format!("{:p}", data.as_ptr())),
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor type for parameter ID",
                "from_tensor",
            )),
        }
    }

    /// Creates a parameter ID from name.
    ///
    /// # Arguments
    ///
    /// * `name` - The parameter name
    pub fn from_name(name: &str) -> String {
        name.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_state_creation() {
        let state = OptimizerState::new();
        assert_eq!(state.step, 0);
        assert!(state.momentum.is_empty());
        assert!(state.variance.is_empty());
    }

    #[test]
    fn test_bias_correction() {
        let correction1 = BiasCorrection::compute_correction(0.9, 1);
        assert!((correction1 - 0.1).abs() < 1e-6);

        let correction2 = BiasCorrection::compute_correction(0.999, 1);
        assert!((correction2 - 0.001).abs() < 1e-6);

        let corrected = BiasCorrection::apply_correction(0.09, 0.9, 1);
        assert!((corrected - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_parameter_update() {
        let mut param = 1.0;
        ParameterUpdate::apply_decoupled_weight_decay(&mut param, 0.01, 0.1);
        assert!((param - 0.999).abs() < 1e-6);

        let mut param2 = 1.0;
        ParameterUpdate::adam_update(&mut param2, 0.01, 0.1, 0.01, 1e-8);
        assert!((param2 - 0.99).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_processing() {
        let mut grad = vec![3.0, 4.0];
        GradientProcessor::clip_by_norm(&mut grad, 1.0);
        let norm: f32 = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);

        assert!(GradientProcessor::is_finite(&grad));

        let bad_grad = vec![f32::NAN, 1.0];
        assert!(!GradientProcessor::is_finite(&bad_grad));
    }

    #[test]
    fn test_memory_stats() {
        let mut state = OptimizerState::new();
        state.get_or_create_momentum("param1".to_string(), 100);
        state.get_or_create_variance("param1".to_string(), 100);

        let stats = state.memory_usage();
        assert_eq!(stats.momentum_elements, 100);
        assert_eq!(stats.variance_elements, 100);
        assert_eq!(stats.num_parameters, 1);
        assert_eq!(stats.total_bytes, 200 * std::mem::size_of::<f32>());
    }

    #[test]
    fn test_ema_update() {
        let mut ema = 0.0;
        ParameterUpdate::update_ema(&mut ema, 1.0, 0.9);
        assert!((ema - 0.1).abs() < 1e-6);

        ParameterUpdate::update_ema(&mut ema, 1.0, 0.9);
        assert!((ema - 0.19).abs() < 1e-6);
    }
}
