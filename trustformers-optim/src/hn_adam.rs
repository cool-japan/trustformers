//! # HN-Adam Optimizer
//!
//! This module implements the HN-Adam (Hybrid Norm Adam) optimizer, a recent modification
//! of the Adam algorithm that automatically adjusts the step size of parameter updates
//! based on the norm value of the parameter update formula according to gradient values.
//!
//! ## Algorithm Description
//!
//! HN-Adam improves upon the standard Adam algorithm by:
//! - Automatically adjusting the step size during training epochs
//! - Using the norm of parameter updates to guide adaptation
//! - Improving convergence speed and accuracy compared to standard Adam
//!
//! The HN-Adam update rule:
//! ```text
//! m_t = β1 * m_{t-1} + (1 - β1) * g_t
//! v_t = β2 * v_{t-1} + (1 - β2) * g_t²
//!
//! m̂_t = m_t / (1 - β1^t)  (bias correction)
//! v̂_t = v_t / (1 - β2^t)  (bias correction)
//!
//! update = m̂_t / (√v̂_t + ε)
//! norm_factor = ||update|| / (||update|| + adaptation_threshold)
//! adaptive_lr = α * norm_factor
//!
//! θ_t = θ_{t-1} - adaptive_lr * update
//! ```
//!
//! ## Key Features
//!
//! - **Adaptive Step Size**: Automatically adjusts learning rate based on update norms
//! - **Improved Convergence**: Often converges faster than standard Adam
//! - **Better Accuracy**: Can achieve higher final accuracy on many tasks
//! - **Robust Training**: More stable training across different learning rate ranges
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use trustformers_optim::{HNAdam, HNAdamConfig};
//! use trustformers_core::traits::Optimizer;
//!
//! // Create HN-Adam optimizer
//! let mut optimizer = HNAdam::new(
//!     1e-3,           // learning rate
//!     (0.9, 0.999),   // (β1, β2)
//!     1e-8,           // epsilon
//!     0.01,           // weight decay
//!     0.1,            // adaptation threshold
//! );
//!
//! // Or use configuration
//! let config = HNAdamConfig {
//!     lr: 1e-3,
//!     betas: (0.9, 0.999),
//!     eps: 1e-8,
//!     weight_decay: 0.01,
//!     adaptation_threshold: 0.1,
//!     amsgrad: false,
//! };
//! let optimizer = HNAdam::with_config(config);
//! ```
//!
//! ## Research Background
//!
//! Based on recent research (2024) that shows HN-Adam can outperform standard Adam
//! in terms of both convergence speed and final accuracy across various datasets
//! including MNIST, CIFAR-10, and other computer vision tasks.

use std::collections::HashMap;
use trustformers_core::{
    errors::{Result, TrustformersError},
    tensor::Tensor,
    traits::Optimizer,
};

use crate::{
    common::{BiasCorrection, OptimizerState, StateMemoryStats},
    traits::StatefulOptimizer,
};

/// Configuration for HN-Adam optimizer
#[derive(Debug, Clone)]
pub struct HNAdamConfig {
    /// Learning rate (default: 1e-3)
    pub lr: f32,
    /// Exponential decay rates for moment estimates (default: (0.9, 0.999))
    pub betas: (f32, f32),
    /// Term added to denominator for numerical stability (default: 1e-8)
    pub eps: f32,
    /// Weight decay coefficient (default: 0.0)
    pub weight_decay: f32,
    /// Threshold for adaptive step size adjustment (default: 0.1)
    pub adaptation_threshold: f32,
    /// Whether to use AMSGrad variant (default: false)
    pub amsgrad: bool,
}

impl Default for HNAdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
            adaptation_threshold: 0.1,
            amsgrad: false,
        }
    }
}

/// HN-Adam optimizer with adaptive step size based on update norms
#[derive(Debug)]
pub struct HNAdam {
    config: HNAdamConfig,
    state: OptimizerState,
    step_count: usize,
    // Store max variance for AMSGrad variant
    max_variance: HashMap<String, Vec<f32>>,
}

impl HNAdam {
    /// Create a new HN-Adam optimizer with specified parameters
    pub fn new(
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
        adaptation_threshold: f32,
    ) -> Self {
        let config = HNAdamConfig {
            lr,
            betas,
            eps,
            weight_decay,
            adaptation_threshold,
            amsgrad: false,
        };
        Self::with_config(config)
    }

    /// Create HN-Adam optimizer with configuration struct
    pub fn with_config(config: HNAdamConfig) -> Self {
        Self {
            config,
            state: OptimizerState::new(),
            step_count: 0,
            max_variance: HashMap::new(),
        }
    }

    /// Create HN-Adam with AMSGrad variant
    pub fn with_amsgrad(
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
        adaptation_threshold: f32,
    ) -> Self {
        let config = HNAdamConfig {
            lr,
            betas,
            eps,
            weight_decay,
            adaptation_threshold,
            amsgrad: true,
        };
        Self::with_config(config)
    }

    /// Create HN-Adam optimized for transformer training
    pub fn for_transformers() -> Self {
        Self::with_config(HNAdamConfig {
            lr: 1e-4,
            betas: (0.9, 0.98),
            eps: 1e-9,
            weight_decay: 0.01,
            adaptation_threshold: 0.05,
            amsgrad: false,
        })
    }

    /// Create HN-Adam optimized for computer vision tasks
    pub fn for_vision() -> Self {
        Self::with_config(HNAdamConfig {
            lr: 1e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 1e-4,
            adaptation_threshold: 0.1,
            amsgrad: false,
        })
    }

    /// Calculate adaptive learning rate based on update norm
    fn calculate_adaptive_lr(&self, update_norm: f32) -> f32 {
        let norm_factor = update_norm / (update_norm + self.config.adaptation_threshold);
        self.config.lr * norm_factor.max(0.1) // Minimum 10% of original LR
    }

    /// Calculate the L2 norm of a vector
    fn vector_norm(&self, data: &[f32]) -> f32 {
        let norm_squared: f32 = data.iter().map(|x| x * x).sum();
        norm_squared.sqrt()
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> StateMemoryStats {
        let momentum_elements: usize = self.state.momentum.values().map(|v| v.len()).sum();
        let variance_elements: usize = self.state.variance.values().map(|v| v.len()).sum();
        let third_moment_elements: usize = self.state.third_moment.values().map(|v| v.len()).sum();

        let total_bytes = (momentum_elements + variance_elements + third_moment_elements)
            * std::mem::size_of::<f32>();

        StateMemoryStats {
            momentum_elements,
            variance_elements,
            third_moment_elements,
            total_bytes,
            num_parameters: self.state.momentum.len(),
        }
    }

    /// Get current adaptation threshold
    pub fn adaptation_threshold(&self) -> f32 {
        self.config.adaptation_threshold
    }

    /// Set adaptation threshold
    pub fn set_adaptation_threshold(&mut self, threshold: f32) {
        self.config.adaptation_threshold = threshold.max(0.001); // Minimum threshold
    }

    /// Generate parameter ID for a given parameter
    fn get_param_id(&self, param: &Tensor) -> String {
        format!("param_{:p}", param as *const _)
    }
}

impl StatefulOptimizer for HNAdam {
    type Config = HNAdamConfig;
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
            "adaptation_threshold".to_string(),
            Tensor::new(vec![self.config.adaptation_threshold])?,
        );
        state_dict.insert(
            "step_count".to_string(),
            Tensor::new(vec![self.step_count as f32])?,
        );

        // Save optimizer states - convert Vec<f32> to Tensor
        for (param_id, momentum) in &self.state.momentum {
            state_dict.insert(
                format!("momentum_{}", param_id),
                Tensor::new(momentum.clone())?,
            );
        }

        for (param_id, variance) in &self.state.variance {
            state_dict.insert(
                format!("variance_{}", param_id),
                Tensor::new(variance.clone())?,
            );
        }

        if self.config.amsgrad {
            for (param_id, max_var) in &self.max_variance {
                state_dict.insert(
                    format!("max_variance_{}", param_id),
                    Tensor::new(max_var.clone())?,
                );
            }
        }

        Ok(state_dict)
    }

    fn load_state_dict(&mut self, state_dict: HashMap<String, Tensor>) -> Result<()> {
        // Load configuration
        if let Some(lr_tensor) = state_dict.get("lr") {
            self.config.lr = lr_tensor.data()?[0];
        }
        if let Some(beta1_tensor) = state_dict.get("beta1") {
            self.config.betas.0 = beta1_tensor.data()?[0];
        }
        if let Some(beta2_tensor) = state_dict.get("beta2") {
            self.config.betas.1 = beta2_tensor.data()?[0];
        }
        if let Some(step_tensor) = state_dict.get("step_count") {
            self.step_count = step_tensor.data()?[0] as usize;
        }

        // Load optimizer states - convert Tensor to Vec<f32>
        for (key, tensor) in state_dict {
            if key.starts_with("momentum_") {
                let param_id = key.strip_prefix("momentum_").unwrap().to_string();
                self.state.momentum.insert(param_id, tensor.data()?);
            } else if key.starts_with("variance_") {
                let param_id = key.strip_prefix("variance_").unwrap().to_string();
                self.state.variance.insert(param_id, tensor.data()?);
            } else if key.starts_with("max_variance_") && self.config.amsgrad {
                let param_id = key.strip_prefix("max_variance_").unwrap().to_string();
                self.max_variance.insert(param_id, tensor.data()?);
            }
        }

        Ok(())
    }

    fn memory_usage(&self) -> StateMemoryStats {
        self.memory_stats()
    }

    fn reset_state(&mut self) {
        self.state = OptimizerState::new();
        self.step_count = 0;
        self.max_variance.clear();
    }

    fn num_parameters(&self) -> usize {
        self.state.momentum.len()
    }
}

impl Optimizer for HNAdam {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        self.step_count += 1;
        let step = self.step_count;

        let param_id = self.get_param_id(parameter);

        match (parameter, grad) {
            (Tensor::F32(param_data), Tensor::F32(grad_data)) => {
                if param_data.len() != grad_data.len() {
                    return Err(TrustformersError::tensor_op_error(
                        "Parameter and gradient tensors must have the same size",
                        "HNAdam::update",
                    ));
                }

                let param_size = param_data.len();

                // Apply weight decay if specified
                if self.config.weight_decay > 0.0 {
                    for i in 0..param_size {
                        param_data[i] -= self.config.weight_decay * param_data[i];
                    }
                }

                // Compute bias correction factors
                let (bias_correction1, bias_correction2) = BiasCorrection::compute_adam_corrections(
                    self.config.betas.0,
                    self.config.betas.1,
                    step,
                );

                // Initialize buffers if needed
                if !self.state.momentum.contains_key(&param_id) {
                    self.state.momentum.insert(param_id.clone(), vec![0.0; param_size]);
                }
                if !self.state.variance.contains_key(&param_id) {
                    self.state.variance.insert(param_id.clone(), vec![0.0; param_size]);
                }

                // Update momentum and variance, compute parameter updates
                let mut update_values = Vec::with_capacity(param_size);

                // Get momentum and variance values for computation
                let momentum_values = self.state.momentum.get(&param_id).unwrap();
                let variance_values = self.state.variance.get(&param_id).unwrap();

                let mut new_momentum = vec![0.0; param_size];
                let mut new_variance = vec![0.0; param_size];

                for i in 0..param_size {
                    // Update momentum (first moment)
                    new_momentum[i] = self.config.betas.0 * momentum_values[i]
                        + (1.0 - self.config.betas.0) * grad_data[i];

                    // Update variance (second moment)
                    new_variance[i] = self.config.betas.1 * variance_values[i]
                        + (1.0 - self.config.betas.1) * grad_data[i] * grad_data[i];

                    // Bias-corrected estimates
                    let momentum_corrected = new_momentum[i] / bias_correction1;
                    let variance_corrected = new_variance[i] / bias_correction2;

                    // Handle AMSGrad variant
                    let final_variance = if self.config.amsgrad {
                        let max_var = self
                            .max_variance
                            .entry(param_id.clone())
                            .or_insert_with(|| vec![0.0; param_size]);
                        max_var[i] = max_var[i].max(variance_corrected);
                        max_var[i]
                    } else {
                        variance_corrected
                    };

                    // Calculate update direction
                    let update = momentum_corrected / (final_variance.sqrt() + self.config.eps);
                    update_values.push(update);
                }

                // Store updated momentum and variance
                self.state.momentum.insert(param_id.clone(), new_momentum);
                self.state.variance.insert(param_id, new_variance);

                // Calculate adaptive learning rate based on update norm
                let update_norm = self.vector_norm(&update_values);
                let adaptive_lr = self.calculate_adaptive_lr(update_norm);

                // Apply parameter updates
                for i in 0..param_size {
                    param_data[i] -= adaptive_lr * update_values[i];
                }

                Ok(())
            },
            _ => Err(TrustformersError::tensor_op_error(
                "HN-Adam optimizer only supports F32 tensors",
                "HNAdam::update",
            )),
        }
    }

    fn step(&mut self) {
        // Update is handled in the update method
    }

    fn zero_grad(&mut self) {
        // Gradient zeroing is typically handled by the training framework
    }

    fn get_lr(&self) -> f32 {
        self.config.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.lr = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hn_adam_creation() {
        let optimizer = HNAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01, 0.1);
        assert_eq!(optimizer.get_lr(), 1e-3);
        assert_eq!(optimizer.adaptation_threshold(), 0.1);
    }

    #[test]
    fn test_hn_adam_config() {
        let config = HNAdamConfig {
            lr: 2e-3,
            betas: (0.95, 0.995),
            eps: 1e-9,
            weight_decay: 0.02,
            adaptation_threshold: 0.05,
            amsgrad: true,
        };
        let optimizer = HNAdam::with_config(config.clone());
        assert_eq!(optimizer.get_lr(), 2e-3);
        assert_eq!(optimizer.config.betas, (0.95, 0.995));
        assert!(optimizer.config.amsgrad);
    }

    #[test]
    fn test_hn_adam_presets() {
        let transformer_opt = HNAdam::for_transformers();
        assert_eq!(transformer_opt.get_lr(), 1e-4);
        assert_eq!(transformer_opt.config.betas, (0.9, 0.98));

        let vision_opt = HNAdam::for_vision();
        assert_eq!(vision_opt.get_lr(), 1e-3);
        assert_eq!(vision_opt.config.adaptation_threshold, 0.1);
    }

    #[test]
    fn test_adaptive_lr_calculation() {
        let optimizer = HNAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.0, 0.1);

        // Test different update norms
        let small_norm = 0.01;
        let large_norm = 1.0;

        let adaptive_lr_small = optimizer.calculate_adaptive_lr(small_norm);
        let adaptive_lr_large = optimizer.calculate_adaptive_lr(large_norm);

        // Larger norms should get higher adaptive learning rates
        assert!(adaptive_lr_large > adaptive_lr_small);

        // Both should be at least 10% of original LR
        assert!(adaptive_lr_small >= 0.1 * optimizer.get_lr());
        assert!(adaptive_lr_large >= 0.1 * optimizer.get_lr());
    }

    #[test]
    fn test_hn_adam_update() -> Result<()> {
        let mut optimizer = HNAdam::new(1e-2, (0.9, 0.999), 1e-8, 0.0, 0.1);

        let mut param = Tensor::new(vec![1.0, 2.0, 3.0])?;
        let grad = Tensor::new(vec![0.1, 0.2, 0.3])?;

        let original_data = param.data()?;

        optimizer.update(&mut param, &grad)?;

        let updated_data = param.data()?;

        // Parameters should have changed
        for (orig, updated) in original_data.iter().zip(updated_data.iter()) {
            assert_ne!(orig, updated);
            // Should move in opposite direction of gradient
            assert!(updated < orig);
        }

        Ok(())
    }

    #[test]
    fn test_stateful_optimizer_trait() -> Result<()> {
        let optimizer = HNAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01, 0.1);

        // Test StatefulOptimizer methods
        assert_eq!(optimizer.config().lr, 1e-3);
        assert_eq!(optimizer.num_parameters(), 0);

        let memory_stats = optimizer.memory_usage();
        assert_eq!(memory_stats.momentum_elements, 0);
        assert_eq!(memory_stats.variance_elements, 0);

        Ok(())
    }

    #[test]
    fn test_amsgrad_variant() -> Result<()> {
        let mut optimizer = HNAdam::with_amsgrad(1e-3, (0.9, 0.999), 1e-8, 0.0, 0.1);
        assert!(optimizer.config.amsgrad);

        let mut param = Tensor::new(vec![1.0, 2.0])?;
        let grad = Tensor::new(vec![0.1, 0.2])?;

        // Run a few updates to test AMSGrad functionality
        for _ in 0..3 {
            optimizer.update(&mut param, &grad)?;
        }

        // Should have max_variance buffers
        assert!(!optimizer.max_variance.is_empty());

        Ok(())
    }

    #[test]
    fn test_state_dict_save_load() -> Result<()> {
        let mut optimizer1 = HNAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01, 0.1);
        let mut param = Tensor::new(vec![1.0, 2.0])?;
        let grad = Tensor::new(vec![0.1, 0.2])?;

        // Run some updates
        for _ in 0..5 {
            optimizer1.update(&mut param, &grad)?;
        }

        // Save state
        let state_dict = optimizer1.state_dict()?;

        // Create new optimizer and load state
        let mut optimizer2 = HNAdam::new(1e-4, (0.8, 0.9), 1e-9, 0.0, 0.2);
        optimizer2.load_state_dict(state_dict)?;

        // Check that state was loaded correctly
        assert_eq!(optimizer2.get_lr(), optimizer1.get_lr());
        assert_eq!(optimizer2.config.betas, optimizer1.config.betas);
        assert_eq!(optimizer2.step_count, optimizer1.step_count);

        Ok(())
    }

    #[test]
    fn test_threshold_adjustment() {
        let mut optimizer = HNAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.0, 0.1);

        optimizer.set_adaptation_threshold(0.2);
        assert_eq!(optimizer.adaptation_threshold(), 0.2);

        // Test minimum threshold enforcement
        optimizer.set_adaptation_threshold(0.0001);
        assert_eq!(optimizer.adaptation_threshold(), 0.001);
    }

    #[test]
    fn test_memory_stats() {
        let optimizer = HNAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.0, 0.1);
        let stats = optimizer.memory_stats();

        // Initially should have no parameters
        assert_eq!(stats.momentum_elements, 0);
        assert_eq!(stats.variance_elements, 0);
        assert_eq!(stats.num_parameters, 0);
        assert_eq!(stats.total_bytes, 0);
    }

    #[test]
    fn test_reset_state() {
        let mut optimizer = HNAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.0, 0.1);

        // Add some state
        optimizer.step_count = 10;
        optimizer.state.momentum.insert("test".to_string(), vec![1.0, 2.0]);

        // Reset and verify
        optimizer.reset_state();
        assert_eq!(optimizer.step_count, 0);
        assert!(optimizer.state.momentum.is_empty());
        assert!(optimizer.max_variance.is_empty());
    }
}
