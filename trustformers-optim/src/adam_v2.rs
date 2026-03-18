//! # Standardized Adam and AdamW Optimizers
//!
//! This module provides updated implementations of Adam and AdamW optimizers
//! using the standardized state management and common operations framework.
//!
//! ## Improvements
//!
//! - Uses common `OptimizerState` for consistent state management
//! - Implements new trait hierarchy for better organization
//! - Leverages shared bias correction and parameter update utilities
//! - Improved memory efficiency and debugging capabilities
//! - Thread-safe and serializable state management

use crate::common::{
    BiasCorrection, OptimizerState, ParameterIds, ParameterUpdate, StateMemoryStats,
    WeightDecayMode,
};
use crate::traits::{AdaptiveMomentumOptimizer, MomentumOptimizer, StatefulOptimizer};
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

/// Configuration for Adam and AdamW optimizers.
#[derive(Debug, Clone)]
pub struct AdamConfig {
    /// Learning rate (α in the paper)
    pub lr: f32,
    /// Coefficients for computing running averages (β1, β2)
    pub betas: (f32, f32),
    /// Term added for numerical stability (ε in the paper)
    pub eps: f32,
    /// Weight decay coefficient
    pub weight_decay: f32,
    /// Weight decay mode (L2 regularization or decoupled)
    pub weight_decay_mode: WeightDecayMode,
    /// Whether to use AMSGrad variant
    pub amsgrad: bool,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
            weight_decay_mode: WeightDecayMode::Decoupled, // AdamW style by default
            amsgrad: false,
        }
    }
}

impl AdamConfig {
    /// Creates AdamW configuration (decoupled weight decay).
    pub fn adamw(lr: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            weight_decay,
            weight_decay_mode: WeightDecayMode::Decoupled,
            ..Default::default()
        }
    }

    /// Creates traditional Adam configuration (L2 regularization).
    pub fn adam(lr: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            weight_decay,
            weight_decay_mode: WeightDecayMode::L2Regularization,
            ..Default::default()
        }
    }

    /// Enables AMSGrad variant.
    pub fn with_amsgrad(mut self) -> Self {
        self.amsgrad = true;
        self
    }

    /// Sets custom beta values.
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.betas = (beta1, beta2);
        self
    }

    /// Sets custom epsilon value.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
}

/// Standardized Adam optimizer with improved state management.
///
/// This implementation uses the common state management framework and
/// implements the full trait hierarchy for better organization and extensibility.
#[derive(Debug)]
pub struct StandardizedAdam {
    /// Optimizer configuration
    config: AdamConfig,
    /// Standardized optimizer state
    state: OptimizerState,
    /// Maximum variance buffers for AMSGrad
    max_variance: HashMap<String, Vec<f32>>,
}

impl StandardizedAdam {
    /// Creates a new StandardizedAdam optimizer.
    pub fn new(config: AdamConfig) -> Self {
        Self {
            config,
            state: OptimizerState::new(),
            max_variance: HashMap::new(),
        }
    }

    /// Creates AdamW with standard configuration.
    pub fn adamw(lr: f32, weight_decay: f32) -> Self {
        Self::new(AdamConfig::adamw(lr, weight_decay))
    }

    /// Creates traditional Adam with standard configuration.
    pub fn adam(lr: f32, weight_decay: f32) -> Self {
        Self::new(AdamConfig::adam(lr, weight_decay))
    }

    /// Updates parameter with standardized operations.
    fn update_parameter(
        &mut self,
        param: &mut [f32],
        grad: &[f32],
        param_id: String,
    ) -> Result<()> {
        let size = grad.len();
        if param.len() != size {
            return Err(TrustformersError::tensor_op_error(
                "Parameter and gradient size mismatch",
                "StandardizedAdam::update_parameter",
            ));
        }

        // Increment parameter step counter first
        self.state.step_param(param_id.clone());
        let step = self.state.get_param_step(&param_id);

        // Get or create state buffers (avoid multiple mutable borrows)
        self.state.get_or_create_momentum(param_id.clone(), size);
        self.state.get_or_create_variance(param_id.clone(), size);

        // Get references to the state buffers and ensure correct size
        let momentum = self
            .state
            .momentum
            .get_mut(&param_id)
            .expect("momentum buffer must exist after get_or_create");
        let variance = self
            .state
            .variance
            .get_mut(&param_id)
            .expect("variance buffer must exist after get_or_create");

        // Ensure buffers have correct size (resize if needed)
        if momentum.len() != size {
            momentum.resize(size, 0.0);
        }
        if variance.len() != size {
            variance.resize(size, 0.0);
        }

        // For AMSGrad, maintain maximum variance
        let mut max_var = if self.config.amsgrad {
            let max_var_buf =
                self.max_variance.entry(param_id.clone()).or_insert_with(|| vec![0.0; size]);
            // Ensure correct size
            if max_var_buf.len() != size {
                max_var_buf.resize(size, 0.0);
            }
            Some(max_var_buf)
        } else {
            None
        };

        // Compute bias corrections
        let (_bias_correction1, _bias_correction2) = BiasCorrection::compute_adam_corrections(
            self.config.betas.0,
            self.config.betas.1,
            step,
        );

        // Process each element
        for i in 0..size {
            let mut grad_val = grad[i];

            // Apply weight decay
            match self.config.weight_decay_mode {
                WeightDecayMode::L2Regularization => {
                    grad_val = ParameterUpdate::apply_l2_regularization(
                        grad_val,
                        param[i],
                        self.config.weight_decay,
                    );
                },
                WeightDecayMode::Decoupled => {
                    // Will be applied after Adam update
                },
            }

            // Update exponential moving averages
            ParameterUpdate::update_ema(&mut momentum[i], grad_val, self.config.betas.0);
            ParameterUpdate::update_ema(&mut variance[i], grad_val * grad_val, self.config.betas.1);

            // For AMSGrad, maintain maximum variance
            let v_hat = if let Some(max_var) = max_var.as_mut() {
                max_var[i] = max_var[i].max(variance[i]);
                BiasCorrection::apply_correction(max_var[i], self.config.betas.1, step)
            } else {
                BiasCorrection::apply_correction(variance[i], self.config.betas.1, step)
            };

            // Apply bias-corrected Adam update
            let m_hat = BiasCorrection::apply_correction(momentum[i], self.config.betas.0, step);
            ParameterUpdate::adam_update(
                &mut param[i],
                self.config.lr,
                m_hat,
                v_hat,
                self.config.eps,
            );

            // Apply decoupled weight decay after Adam update
            if matches!(self.config.weight_decay_mode, WeightDecayMode::Decoupled) {
                ParameterUpdate::apply_decoupled_weight_decay(
                    &mut param[i],
                    self.config.lr,
                    self.config.weight_decay,
                );
            }
        }

        Ok(())
    }
}

impl Optimizer for StandardizedAdam {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        let param_id = ParameterIds::from_tensor(parameter)?;
        match (parameter, grad) {
            (Tensor::F32(param), Tensor::F32(grad_arr)) => self.update_parameter(
                param.as_slice_mut().expect("parameter array must be contiguous"),
                grad_arr.as_slice().expect("gradient array must be contiguous"),
                param_id,
            ),
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor types for StandardizedAdam",
                "StandardizedAdam::update",
            )),
        }
    }

    fn zero_grad(&mut self) {
        // No explicit gradient storage in this implementation
    }

    fn step(&mut self) {
        self.state.step();
    }

    fn get_lr(&self) -> f32 {
        self.config.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.lr = lr;
    }
}

impl StatefulOptimizer for StandardizedAdam {
    type Config = AdamConfig;
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

        // Save momentum buffers
        for (key, momentum) in &self.state.momentum {
            let tensor = Tensor::new(momentum.clone())?;
            state_dict.insert(format!("momentum.{}", key), tensor);
        }

        // Save variance buffers
        for (key, variance) in &self.state.variance {
            let tensor = Tensor::new(variance.clone())?;
            state_dict.insert(format!("variance.{}", key), tensor);
        }

        // Save max variance for AMSGrad
        if self.config.amsgrad {
            for (key, max_var) in &self.max_variance {
                let tensor = Tensor::new(max_var.clone())?;
                state_dict.insert(format!("max_variance.{}", key), tensor);
            }
        }

        // Save step counter
        state_dict.insert(
            "step".to_string(),
            Tensor::new(vec![self.state.step as f32])?,
        );

        Ok(state_dict)
    }

    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()> {
        // Load step counter
        if let Some(step_tensor) = state.get("step") {
            let step_data = step_tensor.data()?;
            if let Some(&step_val) = step_data.first() {
                self.state.step = step_val as usize;
            }
        }

        // Load momentum buffers
        for (key, tensor) in &state {
            if let Some(param_key) = key.strip_prefix("momentum.") {
                let data = tensor.data()?;
                self.state.momentum.insert(param_key.to_string(), data);
            }
        }

        // Load variance buffers
        for (key, tensor) in &state {
            if let Some(param_key) = key.strip_prefix("variance.") {
                let data = tensor.data()?;
                self.state.variance.insert(param_key.to_string(), data);
            }
        }

        // Load max variance buffers for AMSGrad
        if self.config.amsgrad {
            for (key, tensor) in &state {
                if let Some(param_key) = key.strip_prefix("max_variance.") {
                    let data = tensor.data()?;
                    self.max_variance.insert(param_key.to_string(), data);
                }
            }
        }

        Ok(())
    }

    fn memory_usage(&self) -> StateMemoryStats {
        let mut stats = self.state.memory_usage();

        // Add AMSGrad max variance memory usage
        if self.config.amsgrad {
            let max_var_elements: usize = self.max_variance.values().map(|v| v.len()).sum();
            stats.third_moment_elements = max_var_elements;
            stats.total_bytes += max_var_elements * std::mem::size_of::<f32>();
        }

        stats
    }

    fn reset_state(&mut self) {
        self.state.clear();
        self.max_variance.clear();
    }

    fn num_parameters(&self) -> usize {
        self.state.momentum.len()
    }
}

impl MomentumOptimizer for StandardizedAdam {
    fn momentum_coeff(&self) -> f32 {
        self.config.betas.0
    }

    fn set_momentum_coeff(&mut self, coeff: f32) {
        self.config.betas.0 = coeff;
    }

    fn momentum_buffers(&self) -> &HashMap<String, Vec<f32>> {
        &self.state.momentum
    }

    fn clear_momentum(&mut self) {
        self.state.momentum.clear();
    }
}

impl AdaptiveMomentumOptimizer for StandardizedAdam {
    fn variance_coeff(&self) -> f32 {
        self.config.betas.1
    }

    fn set_variance_coeff(&mut self, coeff: f32) {
        self.config.betas.1 = coeff;
    }

    fn epsilon(&self) -> f32 {
        self.config.eps
    }

    fn set_epsilon(&mut self, eps: f32) {
        self.config.eps = eps;
    }

    fn variance_buffers(&self) -> &HashMap<String, Vec<f32>> {
        &self.state.variance
    }

    fn clear_variance(&mut self) {
        self.state.variance.clear();
        if self.config.amsgrad {
            self.max_variance.clear();
        }
    }

    fn apply_bias_correction(&self, momentum: f32, variance: f32, step: usize) -> (f32, f32) {
        let m_hat = BiasCorrection::apply_correction(momentum, self.config.betas.0, step);
        let v_hat = BiasCorrection::apply_correction(variance, self.config.betas.1, step);
        (m_hat, v_hat)
    }
}

/// Type alias for AdamW using standardized implementation.
pub type StandardizedAdamW = StandardizedAdam;

/// Helper functions for creating standardized optimizers.
impl StandardizedAdam {
    /// Creates AdamW with transformer-specific defaults.
    pub fn for_transformers(lr: f32) -> Self {
        Self::adamw(lr, 0.01) // Standard weight decay for transformers
    }

    /// Creates Adam for fine-tuning with lower learning rate.
    pub fn for_fine_tuning(lr: f32) -> Self {
        Self::adam(lr, 0.0).with_betas(0.9, 0.999)
    }

    /// Creates AMSGrad variant for more stable training.
    pub fn amsgrad(lr: f32, weight_decay: f32) -> Self {
        Self::adamw(lr, weight_decay).with_amsgrad()
    }

    /// Creates configuration for mixed precision training.
    pub fn for_mixed_precision(lr: f32, weight_decay: f32) -> Self {
        Self::adamw(lr, weight_decay).with_eps(1e-6) // Larger epsilon for FP16
    }

    /// Set beta parameters for momentum and variance estimation.
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.config.betas = (beta1, beta2);
        self
    }

    /// Enable AMSGrad variant.
    pub fn with_amsgrad(mut self) -> Self {
        self.config.amsgrad = true;
        self
    }

    /// Set epsilon for numerical stability.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.config.eps = eps;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam_config_creation() {
        let config = AdamConfig::adamw(1e-3, 0.01);
        assert_eq!(config.lr, 1e-3);
        assert_eq!(config.weight_decay, 0.01);
        assert!(matches!(
            config.weight_decay_mode,
            WeightDecayMode::Decoupled
        ));

        let adam_config = AdamConfig::adam(1e-3, 0.01);
        assert!(matches!(
            adam_config.weight_decay_mode,
            WeightDecayMode::L2Regularization
        ));
    }

    #[test]
    fn test_standardized_adam_creation() {
        let optimizer = StandardizedAdam::adamw(1e-3, 0.01);
        assert_eq!(optimizer.get_lr(), 1e-3);
        assert_eq!(optimizer.config().weight_decay, 0.01);
        assert_eq!(optimizer.num_parameters(), 0);
    }

    #[test]
    fn test_memory_usage_tracking() {
        let optimizer = StandardizedAdam::adamw(1e-3, 0.01);
        let stats = optimizer.memory_usage();
        assert_eq!(stats.momentum_elements, 0);
        assert_eq!(stats.variance_elements, 0);
        assert_eq!(stats.num_parameters, 0);
    }

    #[test]
    fn test_trait_implementations() {
        let mut optimizer = StandardizedAdam::adamw(1e-3, 0.01);

        // Test MomentumOptimizer trait
        assert_eq!(optimizer.momentum_coeff(), 0.9);
        optimizer.set_momentum_coeff(0.95);
        assert_eq!(optimizer.momentum_coeff(), 0.95);

        // Test AdaptiveMomentumOptimizer trait
        assert_eq!(optimizer.variance_coeff(), 0.999);
        optimizer.set_variance_coeff(0.995);
        assert_eq!(optimizer.variance_coeff(), 0.995);

        assert_eq!(optimizer.epsilon(), 1e-8);
        optimizer.set_epsilon(1e-6);
        assert_eq!(optimizer.epsilon(), 1e-6);
    }

    #[test]
    fn test_convenience_constructors() {
        let transformer_opt = StandardizedAdam::for_transformers(1e-4);
        assert_eq!(transformer_opt.config().lr, 1e-4);
        assert_eq!(transformer_opt.config().weight_decay, 0.01);

        let fine_tune_opt = StandardizedAdam::for_fine_tuning(1e-5);
        assert_eq!(fine_tune_opt.config().lr, 1e-5);
        assert_eq!(fine_tune_opt.config().weight_decay, 0.0);

        let amsgrad_opt = StandardizedAdam::amsgrad(1e-3, 0.01);
        assert!(amsgrad_opt.config().amsgrad);

        let mixed_precision_opt = StandardizedAdam::for_mixed_precision(1e-3, 0.01);
        assert_eq!(mixed_precision_opt.config().eps, 1e-6);
    }

    #[test]
    fn test_state_management() {
        let mut optimizer = StandardizedAdam::adamw(1e-3, 0.01);

        // Test state access
        assert_eq!(optimizer.state().step, 0);
        optimizer.step();
        assert_eq!(optimizer.state().step, 1);

        // Test state reset
        optimizer.reset_state();
        assert_eq!(optimizer.state().step, 0);
        assert!(optimizer.state().momentum.is_empty());
        assert!(optimizer.state().variance.is_empty());
    }
}
