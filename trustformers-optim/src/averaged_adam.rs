//! # Averaged Adam Optimizer (2025)
//!
//! This module implements the Averaged Adam optimizer from "Averaged Adam accelerates
//! stochastic optimization..." (arXiv:2501.06081, January 10, 2025).
//!
//! ## Key Innovation: Polyak-Ruppert Averaging
//!
//! Averaged Adam combines the adaptive learning rate benefits of Adam with the
//! enhanced convergence properties of Polyak-Ruppert averaging. The algorithm
//! maintains both the standard Adam parameters and their averaged versions.
//!
//! ## Algorithm
//!
//! The Averaged Adam update rule:
//! ```text
//! Standard Adam updates:
//! m_t = β1 * m_{t-1} + (1 - β1) * g_t
//! v_t = β2 * v_{t-1} + (1 - β2) * g_t²
//! θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
//!
//! Polyak-Ruppert averaging:
//! θ̄_t = γ * θ̄_{t-1} + (1 - γ) * θ_t
//! ```
//!
//! Where γ is the averaging coefficient, typically close to 1.0.
//!
//! ## Performance Benefits
//!
//! Research demonstrates superior performance across multiple domains:
//! - Physics-Informed Neural Networks (PINNs)
//! - Deep backward stochastic differential equations
//! - Deep Kolmogorov approximations for PDEs
//! - Optimal control problems
//! - Image classification (ResNet on CIFAR-10)
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use trustformers_optim::AveragedAdam;
//! use trustformers_core::traits::Optimizer;
//!
//! // Standard configuration
//! let mut optimizer = AveragedAdam::new(
//!     1e-3,           // learning rate
//!     (0.9, 0.999),   // (β1, β2) for momentum
//!     1e-8,           // epsilon
//!     0.01,           // weight decay
//!     0.999,          // averaging coefficient γ
//! );
//!
//! // PINN-optimized configuration
//! let mut pinn_optimizer = AveragedAdam::for_pinn_training();
//!
//! // Use averaged parameters for evaluation
//! optimizer.use_averaged_parameters(true);
//! ```

use crate::common::{OptimizerState, StateMemoryStats};
use crate::traits::StatefulOptimizer;
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

/// Configuration for Averaged Adam optimizer.
#[derive(Debug, Clone)]
pub struct AveragedAdamConfig {
    /// Learning rate (α in the paper)
    pub lr: f32,
    /// Coefficients for computing running averages (β1, β2)
    pub betas: (f32, f32),
    /// Term added for numerical stability (ε in the paper)
    pub eps: f32,
    /// Decoupled weight decay coefficient
    pub weight_decay: f32,
    /// Polyak-Ruppert averaging coefficient (γ)
    pub averaging_coeff: f32,
    /// Whether to use averaged parameters for updates
    pub use_averaged: bool,
    /// Warmup steps before starting averaging
    pub averaging_warmup: usize,
}

impl Default for AveragedAdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.01,
            averaging_coeff: 0.999,
            use_averaged: false,
            averaging_warmup: 100,
        }
    }
}

/// State for a single parameter in Averaged Adam.
#[derive(Debug, Clone)]
pub struct AveragedAdamParamState {
    /// First moment estimate (momentum)
    pub momentum: Vec<f32>,
    /// Second moment estimate (uncentered variance)
    pub variance: Vec<f32>,
    /// Averaged parameters (Polyak-Ruppert)
    pub averaged_params: Vec<f32>,
}

impl AveragedAdamParamState {
    pub fn new(size: usize) -> Self {
        Self {
            momentum: vec![0.0; size],
            variance: vec![0.0; size],
            averaged_params: vec![0.0; size],
        }
    }

    pub fn memory_usage(&self) -> StateMemoryStats {
        let momentum_elements = self.momentum.len();
        let variance_elements = self.variance.len();
        let averaged_elements = self.averaged_params.len();

        StateMemoryStats {
            momentum_elements,
            variance_elements,
            third_moment_elements: averaged_elements,
            total_bytes: (momentum_elements + variance_elements + averaged_elements)
                * std::mem::size_of::<f32>()
                + std::mem::size_of::<Self>(),
            num_parameters: momentum_elements,
        }
    }
}

/// Averaged Adam optimizer with Polyak-Ruppert averaging for enhanced convergence.
///
/// Implements the algorithm from "Averaged Adam accelerates stochastic optimization..."
/// (arXiv:2501.06081, 2025), combining Adam's adaptive learning rates with the
/// superior convergence properties of parameter averaging.
#[derive(Debug, Clone)]
pub struct AveragedAdam {
    /// Configuration for this optimizer
    config: AveragedAdamConfig,
    /// Optimizer state tracking steps
    state: OptimizerState,
    /// Per-parameter state (momentum, variance, averaged params)
    param_states: HashMap<String, AveragedAdamParamState>,
}

impl AveragedAdam {
    /// Creates a new Averaged Adam optimizer.
    ///
    /// # Arguments
    /// * `lr` - Learning rate (typically 1e-4 to 1e-3)
    /// * `betas` - Coefficients for momentum and variance (β1, β2)
    /// * `eps` - Numerical stability term
    /// * `weight_decay` - Decoupled weight decay coefficient
    /// * `averaging_coeff` - Polyak-Ruppert averaging coefficient (typically 0.999)
    pub fn new(
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
        averaging_coeff: f32,
    ) -> Self {
        let config = AveragedAdamConfig {
            lr,
            betas,
            eps,
            weight_decay,
            averaging_coeff,
            use_averaged: false,
            averaging_warmup: 100,
        };

        Self {
            config,
            state: OptimizerState::new(),
            param_states: HashMap::new(),
        }
    }

    /// Creates an Averaged Adam optimizer optimized for Physics-Informed Neural Networks.
    ///
    /// This preset uses hyperparameters that have shown superior performance
    /// for PINN training compared to standard Adam.
    pub fn for_pinn_training() -> Self {
        let config = AveragedAdamConfig {
            lr: 1e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 1e-4,
            averaging_coeff: 0.9999, // Higher averaging for PINNs
            use_averaged: false,
            averaging_warmup: 200, // Longer warmup for stability
        };

        Self {
            config,
            state: OptimizerState::new(),
            param_states: HashMap::new(),
        }
    }

    /// Creates an optimizer for image classification tasks.
    pub fn for_image_classification() -> Self {
        let config = AveragedAdamConfig {
            lr: 1e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.01,
            averaging_coeff: 0.999,
            use_averaged: false,
            averaging_warmup: 100,
        };

        Self {
            config,
            state: OptimizerState::new(),
            param_states: HashMap::new(),
        }
    }

    /// Creates an optimizer for optimal control problems.
    pub fn for_optimal_control() -> Self {
        let config = AveragedAdamConfig {
            lr: 5e-4,
            betas: (0.95, 0.999), // Higher β1 for control
            eps: 1e-8,
            weight_decay: 1e-5, // Lower weight decay
            averaging_coeff: 0.9995,
            use_averaged: false,
            averaging_warmup: 50, // Faster averaging start
        };

        Self {
            config,
            state: OptimizerState::new(),
            param_states: HashMap::new(),
        }
    }

    /// Sets whether to use averaged parameters for gradient updates.
    ///
    /// When true, uses the Polyak-Ruppert averaged parameters for forward pass.
    /// Typically enabled during evaluation or later stages of training.
    pub fn use_averaged_parameters(&mut self, use_averaged: bool) {
        self.config.use_averaged = use_averaged;
    }

    /// Returns the current averaged parameters for a given parameter name.
    ///
    /// Useful for evaluation or analysis of the averaging process.
    pub fn get_averaged_parameters(&self, param_name: &str) -> Option<&Vec<f32>> {
        self.param_states.get(param_name).map(|state| &state.averaged_params)
    }

    /// Computes the averaging factor for the current step.
    ///
    /// Uses warmup to gradually increase the averaging influence.
    fn compute_averaging_factor(&self) -> f32 {
        if self.state.step < self.config.averaging_warmup {
            // Linear warmup from 0 to averaging_coeff
            let warmup_progress = self.state.step as f32 / self.config.averaging_warmup as f32;
            warmup_progress * self.config.averaging_coeff
        } else {
            self.config.averaging_coeff
        }
    }

    /// Updates averaged parameters using Polyak-Ruppert averaging.
    fn update_averaged_parameters(
        &mut self,
        param_name: &str,
        current_params: &[f32],
    ) -> Result<()> {
        let gamma = self.compute_averaging_factor();

        if let Some(param_state) = self.param_states.get_mut(param_name) {
            if param_state.averaged_params.len() != current_params.len() {
                // Initialize averaged parameters if size mismatch
                param_state.averaged_params = current_params.to_vec();
            } else {
                // Polyak-Ruppert averaging: θ̄_t = γ * θ̄_{t-1} + (1 - γ) * θ_t
                for (avg_param, &curr_param) in
                    param_state.averaged_params.iter_mut().zip(current_params.iter())
                {
                    *avg_param = gamma * (*avg_param) + (1.0 - gamma) * curr_param;
                }
            }
        }
        Ok(())
    }
}

impl Optimizer for AveragedAdam {
    fn update(&mut self, param: &mut Tensor, gradient: &Tensor) -> Result<()> {
        // Get parameter data (bind to variable before taking pointer)
        let param_data = param.data()?;
        let param_name = format!("{:p}", param_data.as_ptr());
        let grad_data = gradient.data()?;

        if param_data.len() != grad_data.len() {
            return Err(TrustformersError::tensor_op_error(
                "Parameter and gradient size mismatch",
                "AveragedAdam::update",
            ));
        }

        let param_size = param_data.len();

        // Initialize parameter state if needed
        if !self.param_states.contains_key(&param_name) {
            let mut param_state = AveragedAdamParamState::new(param_size);
            // Initialize averaged parameters with current parameters
            param_state.averaged_params = param_data.clone();
            self.param_states.insert(param_name.to_string(), param_state);
        }

        let param_state = self.param_states.get_mut(&param_name).ok_or_else(|| {
            TrustformersError::tensor_op_error(
                "Parameter state not found after insertion",
                "AveragedAdam::update",
            )
        })?;

        // Ensure state buffers have correct size
        if param_state.momentum.len() != param_size {
            param_state.momentum.resize(param_size, 0.0);
            param_state.variance.resize(param_size, 0.0);
            param_state.averaged_params.resize(param_size, 0.0);
        }

        let step = self.state.step as f32 + 1.0;
        let (beta1, beta2) = self.config.betas;

        // Apply weight decay to gradients
        let mut effective_grad = grad_data.clone();
        if self.config.weight_decay > 0.0 {
            for (grad, &param_val) in effective_grad.iter_mut().zip(param_data.iter()) {
                *grad += self.config.weight_decay * param_val;
            }
        }

        // Adam updates
        let mut updated_params = param_data.clone();
        for i in 0..param_size {
            // Update biased first moment estimate
            param_state.momentum[i] =
                beta1 * param_state.momentum[i] + (1.0 - beta1) * effective_grad[i];

            // Update biased second moment estimate
            param_state.variance[i] =
                beta2 * param_state.variance[i] + (1.0 - beta2) * effective_grad[i].powi(2);

            // Bias correction
            let m_hat = param_state.momentum[i] / (1.0 - beta1.powf(step));
            let v_hat = param_state.variance[i] / (1.0 - beta2.powf(step));

            // Parameter update
            updated_params[i] -= self.config.lr * m_hat / (v_hat.sqrt() + self.config.eps);
        }

        // Update averaged parameters using Polyak-Ruppert averaging
        self.update_averaged_parameters(&param_name, &updated_params)?;

        // Use averaged parameters if enabled
        let final_params =
            if self.config.use_averaged && self.state.step >= self.config.averaging_warmup {
                // Get fresh reference to param_state after update
                let param_state = self.param_states.get(&param_name).ok_or_else(|| {
                    TrustformersError::tensor_op_error(
                        "Parameter state not found",
                        "AveragedAdam::update",
                    )
                })?;
                param_state.averaged_params.clone()
            } else {
                updated_params.clone()
            };

        // Create new tensor with updated parameters
        let updated_tensor = Tensor::new(final_params)?;
        *param = updated_tensor;

        Ok(())
    }

    fn zero_grad(&mut self) {
        // Averaged Adam doesn't need to clear gradients
    }

    fn step(&mut self) {
        self.state.step += 1;
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.lr = lr;
    }

    fn get_lr(&self) -> f32 {
        self.config.lr
    }
}

impl StatefulOptimizer for AveragedAdam {
    type Config = AveragedAdamConfig;
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
        state_dict.insert("lr".to_string(), Tensor::new(vec![self.config.lr])?);
        state_dict.insert("beta1".to_string(), Tensor::new(vec![self.config.betas.0])?);
        state_dict.insert("beta2".to_string(), Tensor::new(vec![self.config.betas.1])?);
        state_dict.insert("eps".to_string(), Tensor::new(vec![self.config.eps])?);
        state_dict.insert(
            "weight_decay".to_string(),
            Tensor::new(vec![self.config.weight_decay])?,
        );
        state_dict.insert(
            "averaging_coeff".to_string(),
            Tensor::new(vec![self.config.averaging_coeff])?,
        );
        state_dict.insert(
            "step".to_string(),
            Tensor::new(vec![self.state.step as f32])?,
        );

        for (param_name, param_state) in &self.param_states {
            // Store momentum
            state_dict.insert(
                format!("momentum_{}", param_name),
                Tensor::new(param_state.momentum.clone())?,
            );
            // Store variance
            state_dict.insert(
                format!("variance_{}", param_name),
                Tensor::new(param_state.variance.clone())?,
            );
            // Store averaged parameters
            state_dict.insert(
                format!("averaged_{}", param_name),
                Tensor::new(param_state.averaged_params.clone())?,
            );
        }

        Ok(state_dict)
    }

    fn load_state_dict(&mut self, state_dict: HashMap<String, Tensor>) -> Result<()> {
        // Load configuration
        if let Some(lr_tensor) = state_dict.get("lr") {
            let lr_data = lr_tensor.data()?;
            if !lr_data.is_empty() {
                self.config.lr = lr_data[0];
            }
        }

        if let Some(step_tensor) = state_dict.get("step") {
            let step_data = step_tensor.data()?;
            if !step_data.is_empty() {
                self.state.step = step_data[0] as usize;
            }
        }

        // Collect parameter names
        let mut param_names = std::collections::HashSet::new();
        for key in state_dict.keys() {
            if let Some(underscore_idx) = key.find('_') {
                let prefix = &key[..underscore_idx];
                if prefix == "momentum" || prefix == "variance" || prefix == "averaged" {
                    let param_name = &key[underscore_idx + 1..];
                    param_names.insert(param_name.to_string());
                }
            }
        }

        // Load parameter states
        for param_name in param_names {
            let momentum_key = format!("momentum_{}", param_name);
            let variance_key = format!("variance_{}", param_name);
            let averaged_key = format!("averaged_{}", param_name);

            if let (Some(momentum_tensor), Some(variance_tensor), Some(averaged_tensor)) = (
                state_dict.get(&momentum_key),
                state_dict.get(&variance_key),
                state_dict.get(&averaged_key),
            ) {
                let param_state = AveragedAdamParamState {
                    momentum: momentum_tensor.data()?,
                    variance: variance_tensor.data()?,
                    averaged_params: averaged_tensor.data()?,
                };
                self.param_states.insert(param_name, param_state);
            }
        }

        Ok(())
    }

    fn memory_usage(&self) -> StateMemoryStats {
        let mut total_momentum = 0;
        let mut total_variance = 0;
        let mut total_averaged = 0;
        let mut total_params = 0;

        for param_state in self.param_states.values() {
            let state_stats = param_state.memory_usage();
            total_momentum += state_stats.momentum_elements;
            total_variance += state_stats.variance_elements;
            total_averaged += state_stats.third_moment_elements;
            total_params += state_stats.num_parameters;
        }

        let total_bytes = (total_momentum + total_variance + total_averaged)
            * std::mem::size_of::<f32>()
            + std::mem::size_of::<AveragedAdamConfig>()
            + std::mem::size_of::<OptimizerState>();

        StateMemoryStats {
            momentum_elements: total_momentum,
            variance_elements: total_variance,
            third_moment_elements: total_averaged,
            total_bytes,
            num_parameters: total_params,
        }
    }

    fn reset_state(&mut self) {
        self.param_states.clear();
        self.state = OptimizerState::new();
    }

    fn num_parameters(&self) -> usize {
        self.param_states.values().map(|state| state.momentum.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::tensor::Tensor;

    #[test]
    fn test_averaged_adam_creation() {
        let optimizer = AveragedAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01, 0.999);
        assert_eq!(optimizer.get_lr(), 1e-3);
        assert_eq!(optimizer.config.betas, (0.9, 0.999));
        assert_eq!(optimizer.config.averaging_coeff, 0.999);
    }

    #[test]
    fn test_pinn_preset() {
        let optimizer = AveragedAdam::for_pinn_training();
        assert_eq!(optimizer.config.averaging_coeff, 0.9999);
        assert_eq!(optimizer.config.averaging_warmup, 200);
    }

    #[test]
    fn test_image_classification_preset() {
        let optimizer = AveragedAdam::for_image_classification();
        assert_eq!(optimizer.config.weight_decay, 0.01);
        assert_eq!(optimizer.config.averaging_coeff, 0.999);
    }

    #[test]
    fn test_optimal_control_preset() {
        let optimizer = AveragedAdam::for_optimal_control();
        assert_eq!(optimizer.config.betas.0, 0.95);
        assert_eq!(optimizer.config.averaging_warmup, 50);
    }

    #[test]
    fn test_parameter_update() {
        let mut optimizer = AveragedAdam::new(0.1, (0.9, 0.999), 1e-8, 0.0, 0.999);
        let mut param = Tensor::new(vec![1.0, 2.0, 3.0]).unwrap();
        let grad = Tensor::new(vec![0.1, 0.2, 0.3]).unwrap();

        optimizer.update(&mut param, &grad).unwrap();
        optimizer.step();

        // Check that parameters were updated
        let updated_data = param.data().unwrap();
        assert!(updated_data[0] < 1.0);
        assert!(updated_data[1] < 2.0);
        assert!(updated_data[2] < 3.0);
    }

    #[test]
    fn test_averaged_parameters() {
        let mut optimizer = AveragedAdam::new(0.1, (0.9, 0.999), 1e-8, 0.0, 0.9);
        let mut param = Tensor::new(vec![1.0, 2.0]).unwrap();
        let grad = Tensor::new(vec![0.1, 0.2]).unwrap();

        // Perform several updates
        for _ in 0..10 {
            optimizer.update(&mut param, &grad).unwrap();
            optimizer.step();
        }

        // Check that averaged parameters exist - get the first parameter key
        let param_keys: Vec<String> = optimizer.param_states.keys().cloned().collect();
        assert!(!param_keys.is_empty());
        let first_param_key = &param_keys[0];
        let averaged = optimizer.get_averaged_parameters(first_param_key);
        assert!(averaged.is_some());
        assert_eq!(averaged.unwrap().len(), 2);
    }

    #[test]
    fn test_use_averaged_parameters() {
        let mut optimizer = AveragedAdam::new(0.1, (0.9, 0.999), 1e-8, 0.0, 0.9);
        optimizer.config.averaging_warmup = 0; // Disable warmup for test

        let mut param = Tensor::new(vec![1.0]).unwrap();
        let grad = Tensor::new(vec![0.1]).unwrap();

        // Initial update without averaging
        optimizer.update(&mut param, &grad).unwrap();
        optimizer.step();
        let standard_value = param.data().unwrap()[0];

        // Reset and use averaged parameters
        param = Tensor::new(vec![1.0]).unwrap();
        optimizer.reset_state();
        optimizer.use_averaged_parameters(true);

        optimizer.update(&mut param, &grad).unwrap();
        optimizer.step();
        let averaged_value = param.data().unwrap()[0];

        // Values should be different when using averaging
        assert_ne!(standard_value, averaged_value);
    }

    #[test]
    fn test_state_dict_operations() {
        let mut optimizer = AveragedAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01, 0.999);
        let mut param = Tensor::new(vec![1.0, 2.0]).unwrap();
        let grad = Tensor::new(vec![0.1, 0.2]).unwrap();

        // Perform update to create state
        optimizer.update(&mut param, &grad).unwrap();
        optimizer.step();

        // Save state
        let state_dict = optimizer.state_dict().unwrap();
        assert!(state_dict.contains_key("lr"));
        assert!(state_dict.contains_key("step"));

        // Check that at least one parameter state exists (momentum/variance/averaged)
        let has_momentum = state_dict.keys().any(|k| k.starts_with("momentum_"));
        let has_variance = state_dict.keys().any(|k| k.starts_with("variance_"));
        let has_averaged = state_dict.keys().any(|k| k.starts_with("averaged_"));
        assert!(has_momentum);
        assert!(has_variance);
        assert!(has_averaged);

        // Create new optimizer and load state
        let mut new_optimizer = AveragedAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01, 0.999);
        new_optimizer.load_state_dict(state_dict).unwrap();

        // Check state was loaded correctly
        assert_eq!(new_optimizer.state().step, optimizer.state().step);
        assert!(!new_optimizer.param_states.is_empty());
    }

    #[test]
    fn test_memory_usage() {
        let mut optimizer = AveragedAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01, 0.999);
        let mut param = Tensor::new(vec![1.0; 100]).unwrap();
        let grad = Tensor::new(vec![0.1; 100]).unwrap();

        optimizer.update(&mut param, &grad).unwrap();

        let memory_stats = optimizer.memory_usage();
        assert!(memory_stats.momentum_elements > 0);
        assert!(memory_stats.variance_elements > 0);
        assert!(memory_stats.third_moment_elements > 0);
        assert!(memory_stats.total_bytes > 0);

        // Should have memory for momentum, variance, and averaged parameters
        assert_eq!(memory_stats.momentum_elements, 100);
        assert_eq!(memory_stats.variance_elements, 100);
        assert_eq!(memory_stats.third_moment_elements, 100);
    }

    #[test]
    fn test_averaging_warmup() {
        let mut optimizer = AveragedAdam::new(0.1, (0.9, 0.999), 1e-8, 0.0, 0.999);
        optimizer.config.averaging_warmup = 5;

        // Before warmup, averaging factor should be lower
        let early_factor = optimizer.compute_averaging_factor();
        assert!(early_factor < 0.999);

        // Advance past warmup
        for _ in 0..10 {
            optimizer.step();
        }

        let late_factor = optimizer.compute_averaging_factor();
        assert_eq!(late_factor, 0.999);
    }

    #[test]
    fn test_num_parameters() {
        let mut optimizer = AveragedAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01, 0.999);
        assert_eq!(optimizer.num_parameters(), 0);

        let mut param1 = Tensor::new(vec![1.0; 10]).unwrap();
        let grad1 = Tensor::new(vec![0.1; 10]).unwrap();
        optimizer.update(&mut param1, &grad1).unwrap();
        assert_eq!(optimizer.num_parameters(), 10);

        let mut param2 = Tensor::new(vec![2.0; 20]).unwrap();
        let grad2 = Tensor::new(vec![0.2; 20]).unwrap();
        optimizer.update(&mut param2, &grad2).unwrap();
        assert_eq!(optimizer.num_parameters(), 30);
    }
}
