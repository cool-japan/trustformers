//! # Lion Optimizer (EvoLved Sign Momentum)
//!
//! This module implements the Lion optimizer, a recent optimization algorithm
//! from Google Research that uses sign-based momentum for efficient training.
//!
//! ## Lion
//!
//! Lion uses the sign of interpolated momentum for parameter updates:
//! - Simpler than Adam (no second moment estimation)
//! - More memory efficient (only stores momentum)
//! - Often achieves better performance on large models
//! - More stable training with larger learning rates
//!
//! The Lion update rule:
//! ```text
//! m_t = β1 * m_{t-1} + (1 - β1) * g_t
//! θ_t = θ_{t-1} - η * (sign(m_t) + λ * θ_{t-1})
//! ```
//!
//! Reference: "Symbolic Discovery of Optimization Algorithms"
//! by Chen et al. (2023) - Google Research
//!
//! ## Key Features
//!
//! - **Memory Efficient**: Only stores momentum (no variance tracking)
//! - **Simple Updates**: Uses sign of momentum instead of raw values
//! - **Robust**: More stable with larger learning rates
//! - **Performance**: Often outperforms Adam on large models
//! - **Gradient Clipping**: Built-in gradient norm clipping
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use trustformers_optim::Lion;
//! use trustformers_core::traits::Optimizer;
//!
//! // Create Lion optimizer
//! let mut optimizer = Lion::new(
//!     1e-4,           // Learning rate (typically lower than Adam)
//!     (0.9, 0.99),    // (β1, β2) - β2 for momentum update
//!     0.01,           // Weight decay coefficient
//! );
//! ```
//!
//! ## Hyperparameter Guidelines
//!
//! ### Learning Rate
//! - Use lower learning rates than Adam (typically 3-10x smaller)
//! - Good starting point: 1e-4 to 3e-4 for transformers
//! - Can often use larger learning rates than other optimizers
//!
//! ### Beta Values
//! - β1 (momentum coefficient): 0.9 (default, robust across tasks)
//! - β2 (momentum interpolation): 0.99 (affects momentum interpolation)
//! - Less sensitive to β values than Adam
//!
//! ### Weight Decay
//! - Similar to AdamW: 0.01 to 0.1
//! - Applied directly to parameters (decoupled)
//! - Often needs less weight decay than Adam
//!
//! ## Memory Usage
//!
//! - Lion: O(n) memory for parameters of size n
//! - Adam: O(2n) memory (momentum + variance)
//! - Significant memory savings for large models

use crate::common::{OptimizerState, StateMemoryStats};
use crate::traits::StatefulOptimizer;
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

/// Configuration for Lion optimizer.
#[derive(Debug, Clone)]
pub struct LionConfig {
    /// Learning rate (typically 3-10x smaller than Adam)
    pub lr: f32,
    /// Momentum coefficients (β1, β2)
    pub betas: (f32, f32),
    /// Weight decay coefficient
    pub weight_decay: f32,
    /// Gradient clipping threshold (0.0 disables clipping)
    pub grad_clip: f32,
}

impl Default for LionConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            betas: (0.9, 0.99),
            weight_decay: 0.01,
            grad_clip: 1.0,
        }
    }
}

/// Lion optimizer with sign-based momentum updates.
///
/// Implements the Lion algorithm from "Symbolic Discovery of Optimization Algorithms"
/// by Chen et al. (2023). This optimizer uses the sign of interpolated momentum
/// for parameter updates, making it more memory efficient than Adam while often
/// achieving better performance.
#[derive(Debug)]
pub struct Lion {
    /// Configuration for this optimizer
    config: LionConfig,
    /// Optimizer state tracking steps
    state: OptimizerState,
    /// Momentum estimates (m_t)
    momentum: HashMap<String, Vec<f32>>,
}

impl Lion {
    /// Creates a new Lion optimizer.
    ///
    /// # Arguments
    ///
    /// * `lr` - Learning rate (typical: 1e-4 to 3e-4, lower than Adam)
    /// * `betas` - Momentum coefficients (typical: (0.9, 0.99))
    /// * `weight_decay` - Weight decay coefficient (typical: 0.01)
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_optim::Lion;
    /// let optimizer = Lion::new(1e-4, (0.9, 0.99), 0.01);
    /// ```
    pub fn new(lr: f32, betas: (f32, f32), weight_decay: f32) -> Self {
        Self {
            config: LionConfig {
                lr,
                betas,
                weight_decay,
                grad_clip: 1.0,
            },
            state: OptimizerState::new(),
            momentum: HashMap::new(),
        }
    }

    /// Creates a new Lion optimizer with gradient clipping.
    ///
    /// # Arguments
    ///
    /// * `lr` - Learning rate
    /// * `betas` - Momentum coefficients
    /// * `weight_decay` - Weight decay coefficient
    /// * `grad_clip` - Gradient clipping threshold
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_optim::Lion;
    /// let optimizer = Lion::with_grad_clip(1e-4, (0.9, 0.99), 0.01, 1.0);
    /// ```
    pub fn with_grad_clip(lr: f32, betas: (f32, f32), weight_decay: f32, grad_clip: f32) -> Self {
        Self {
            config: LionConfig {
                lr,
                betas,
                weight_decay,
                grad_clip,
            },
            state: OptimizerState::new(),
            momentum: HashMap::new(),
        }
    }

    /// Creates a new Lion optimizer from configuration.
    pub fn from_config(config: LionConfig) -> Self {
        Self {
            config,
            state: OptimizerState::new(),
            momentum: HashMap::new(),
        }
    }
}

impl Optimizer for Lion {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        match (parameter, grad) {
            (Tensor::F32(param), Tensor::F32(grad_arr)) => {
                let param_id = format!("{:p}", param.as_ptr());
                let size = grad_arr.len();

                // Initialize momentum if not exists
                let momentum = self.momentum.entry(param_id).or_insert_with(|| vec![0.0; size]);

                if momentum.len() != size {
                    return Err(TrustformersError::tensor_op_error(
                        "Lion momentum buffer size mismatch",
                        "Lion::update",
                    ));
                }

                // Compute gradient norm for clipping
                let grad_norm: f32 = grad_arr.iter().map(|g| g * g).sum::<f32>().sqrt();
                let clip_coeff = if self.config.grad_clip > 0.0 && grad_norm > self.config.grad_clip
                {
                    self.config.grad_clip / grad_norm
                } else {
                    1.0
                };

                // Lion algorithm implementation
                for ((p, &g), m) in param.iter_mut().zip(grad_arr.iter()).zip(momentum.iter_mut()) {
                    // Apply gradient clipping
                    let clipped_g = g * clip_coeff;
                    // Compute interpolated momentum: β2 * m + (1 - β2) * clipped_g
                    let interpolated_momentum =
                        self.config.betas.1 * *m + (1.0 - self.config.betas.1) * clipped_g;

                    // Compute update using sign of interpolated momentum
                    let sign_update = if interpolated_momentum > 0.0 {
                        1.0
                    } else if interpolated_momentum < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };

                    // Apply weight decay directly to parameters (decoupled)
                    if self.config.weight_decay != 0.0 {
                        *p -= self.config.lr * self.config.weight_decay * *p;
                    }

                    // Apply Lion update: θ = θ - η * sign(interpolated_momentum)
                    *p -= self.config.lr * sign_update;

                    // Update momentum: m = β1 * m + (1 - β1) * clipped_g
                    *m = self.config.betas.0 * *m + (1.0 - self.config.betas.0) * clipped_g;
                }

                Ok(())
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor types for Lion",
                "Lion::update",
            )),
        }
    }

    fn zero_grad(&mut self) {}

    fn step(&mut self) {
        self.state.step += 1;
    }

    fn get_lr(&self) -> f32 {
        self.config.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.lr = lr;
    }
}

impl StatefulOptimizer for Lion {
    type Config = LionConfig;
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
        state_dict.insert(
            "weight_decay".to_string(),
            Tensor::new(vec![self.config.weight_decay])?,
        );
        state_dict.insert(
            "grad_clip".to_string(),
            Tensor::new(vec![self.config.grad_clip])?,
        );
        state_dict.insert(
            "step".to_string(),
            Tensor::new(vec![self.state.step as f32])?,
        );

        // Save momentum buffers
        for (param_id, momentum) in &self.momentum {
            state_dict.insert(
                format!("momentum_{}", param_id),
                Tensor::new(momentum.clone())?,
            );
        }

        Ok(state_dict)
    }

    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()> {
        // Load configuration
        if let Some(lr_tensor) = state.get("lr") {
            if let Ok(lr_vec) = lr_tensor.data() {
                if !lr_vec.is_empty() {
                    self.config.lr = lr_vec[0];
                }
            }
        }

        if let Some(beta1_tensor) = state.get("beta1") {
            if let Ok(beta1_vec) = beta1_tensor.data() {
                if !beta1_vec.is_empty() {
                    self.config.betas.0 = beta1_vec[0];
                }
            }
        }

        if let Some(beta2_tensor) = state.get("beta2") {
            if let Ok(beta2_vec) = beta2_tensor.data() {
                if !beta2_vec.is_empty() {
                    self.config.betas.1 = beta2_vec[0];
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

        if let Some(grad_clip_tensor) = state.get("grad_clip") {
            if let Ok(grad_clip_vec) = grad_clip_tensor.data() {
                if !grad_clip_vec.is_empty() {
                    self.config.grad_clip = grad_clip_vec[0];
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

        // Load momentum buffers
        for (key, tensor) in state.iter() {
            if key.starts_with("momentum_") {
                let param_id = key.trim_start_matches("momentum_");
                if let Ok(momentum) = tensor.data() {
                    self.momentum.insert(param_id.to_string(), momentum.clone());
                }
            }
        }

        Ok(())
    }

    fn memory_usage(&self) -> StateMemoryStats {
        let mut momentum_elements = 0;

        for momentum in self.momentum.values() {
            momentum_elements += momentum.len();
        }

        let total_elements = momentum_elements;
        let total_bytes = total_elements * std::mem::size_of::<f32>();

        StateMemoryStats {
            momentum_elements,
            variance_elements: 0, // Lion doesn't use variance
            third_moment_elements: 0,
            total_bytes,
            num_parameters: momentum_elements,
        }
    }

    fn reset_state(&mut self) {
        self.state.step = 0;
        self.momentum.clear();
    }

    fn num_parameters(&self) -> usize {
        self.momentum.values().map(|v| v.len()).sum()
    }
}

impl Default for Lion {
    fn default() -> Self {
        Self::new(1e-4, (0.9, 0.99), 0.01)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lion_creation() {
        let optimizer = Lion::new(1e-4, (0.9, 0.99), 0.01);
        assert_eq!(optimizer.config.lr, 1e-4);
        assert_eq!(optimizer.config.betas, (0.9, 0.99));
        assert_eq!(optimizer.config.weight_decay, 0.01);
    }

    #[test]
    fn test_lion_with_grad_clip() {
        let optimizer = Lion::with_grad_clip(1e-4, (0.9, 0.99), 0.01, 1.0);
        assert_eq!(optimizer.config.grad_clip, 1.0);
    }

    #[test]
    fn test_lion_default() {
        let optimizer = Lion::default();
        assert_eq!(optimizer.config.lr, 1e-4);
        assert_eq!(optimizer.config.betas, (0.9, 0.99));
        assert_eq!(optimizer.config.weight_decay, 0.01);
    }

    #[test]
    fn test_gradient_clipping() {
        let optimizer = Lion::with_grad_clip(1e-4, (0.9, 0.99), 0.01, 1.0);
        let gradients = vec![2.0, -3.0, 1.5]; // Norm = sqrt(4 + 9 + 2.25) = ~3.91

        let grad_norm: f32 = gradients.iter().map(|g| g * g).sum::<f32>().sqrt();
        let clip_coeff =
            if optimizer.config.grad_clip > 0.0 && grad_norm > optimizer.config.grad_clip {
                optimizer.config.grad_clip / grad_norm
            } else {
                1.0
            };

        // Should be clipped since norm > 1.0
        assert!(clip_coeff < 1.0);
        let clipped_gradients: Vec<f32> = gradients.iter().map(|g| g * clip_coeff).collect();
        let new_norm: f32 = clipped_gradients.iter().map(|g| g * g).sum::<f32>().sqrt();
        assert!((new_norm - 1.0).abs() < 1e-6); // Should be approximately 1.0
    }

    #[test]
    fn test_no_gradient_clipping() {
        let optimizer = Lion::with_grad_clip(1e-4, (0.9, 0.99), 0.01, 0.0); // Disabled
        let gradients = vec![2.0, -3.0, 1.5];

        let grad_norm: f32 = gradients.iter().map(|g| g * g).sum::<f32>().sqrt();
        let clip_coeff =
            if optimizer.config.grad_clip > 0.0 && grad_norm > optimizer.config.grad_clip {
                optimizer.config.grad_clip / grad_norm
            } else {
                1.0
            };

        assert_eq!(clip_coeff, 1.0);
        // Gradients should remain unchanged when clipping is disabled
    }

    #[test]
    fn test_memory_efficiency() {
        let optimizer = Lion::default();
        let stats = optimizer.memory_usage();

        // Initially should have no memory usage
        assert_eq!(stats.total_bytes, 0);
        assert_eq!(stats.momentum_elements, 0);
        assert_eq!(stats.variance_elements, 0); // Lion doesn't use variance
    }

    #[test]
    fn test_state_persistence() {
        let mut optimizer = Lion::new(1e-4, (0.9, 0.99), 0.01);
        optimizer.state.step = 100;

        // Test that state is maintained
        assert_eq!(optimizer.state.step, 100);

        // Test reset
        optimizer.reset_state();
        assert_eq!(optimizer.state.step, 0);
        assert!(optimizer.momentum.is_empty());
    }
}
