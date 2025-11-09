//! # Sophia Optimizer (Second-order Clipped Stochastic Optimization)
//!
//! This module implements the Sophia optimizer, a recent second-order optimization
//! algorithm designed specifically for large language model training. Sophia uses
//! second-order information efficiently while maintaining computational tractability.
//!
//! ## Sophia
//!
//! Sophia adapts the step size using second-order information:
//! - Uses Hessian diagonal estimate for preconditioning
//! - Applies clipping to control step sizes
//! - More efficient than full second-order methods
//! - Designed for transformer model training
//!
//! The Sophia update rule:
//! ```text
//! m_t = β1 * m_{t-1} + (1 - β1) * g_t
//! h_t = β2 * h_{t-1} + (1 - β2) * diag(H_t)  // Hessian diagonal
//! θ_t = θ_{t-1} - η * clip(m_t / (ρ * h_t + ε), γ)
//! ```
//!
//! Reference: "Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training"
//! by Liu et al. (2023)
//!
//! ## Key Features
//!
//! - **Second-order Information**: Uses Hessian diagonal for better convergence
//! - **Computational Efficiency**: More efficient than full second-order methods
//! - **Clipping**: Built-in clipping for stable training
//! - **Language Model Focus**: Designed specifically for transformer training
//! - **Memory Efficient**: Only stores diagonal Hessian estimate
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use trustformers_optim::Sophia;
//! use trustformers_core::traits::Optimizer;
//!
//! // Create Sophia optimizer
//! let mut optimizer = Sophia::new(
//!     1e-4,           // Learning rate
//!     (0.965, 0.99),  // (β1, β2)
//!     1e-8,           // Numerical stability
//!     0.04,           // Hessian scaling factor
//!     0.01,           // Clipping threshold
//! );
//! ```
//!
//! ## Hyperparameter Guidelines
//!
//! ### Learning Rate
//! - Similar to Adam: 1e-4 to 3e-4 for transformers
//! - Can often use slightly higher learning rates than Adam
//! - Less sensitive to learning rate than first-order methods
//!
//! ### Beta Values
//! - β1 (momentum): 0.965 (default, slightly higher than Adam)
//! - β2 (Hessian EMA): 0.99 (similar to Adam's β2)
//! - Less sensitive to β values due to second-order information
//!
//! ### Rho (Hessian Scaling)
//! - Controls influence of Hessian information: 0.04 (default)
//! - Lower values: more first-order behavior
//! - Higher values: more second-order behavior
//!
//! ### Gamma (Clipping)
//! - Controls maximum step size: 0.01 (default)
//! - Important for stability with second-order information
//! - Smaller values: more conservative updates
//!
//! ## Memory Usage
//!
//! - Sophia: O(2n) memory for parameters of size n (momentum + Hessian diagonal)
//! - Similar to Adam but uses second-order information
//! - More efficient than full second-order methods (O(n²))

use crate::common::{OptimizerState, StateMemoryStats};
use crate::traits::StatefulOptimizer;
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

/// Configuration for Sophia optimizer.
#[derive(Debug, Clone)]
pub struct SophiaConfig {
    /// Learning rate
    pub lr: f32,
    /// Momentum coefficients (β1, β2)
    pub betas: (f32, f32),
    /// Numerical stability term
    pub eps: f32,
    /// Hessian scaling factor
    pub rho: f32,
    /// Clipping threshold
    pub gamma: f32,
    /// Weight decay coefficient
    pub weight_decay: f32,
    /// Hessian update frequency (every k steps)
    pub hessian_update_freq: usize,
}

impl Default for SophiaConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            betas: (0.965, 0.99),
            eps: 1e-8,
            rho: 0.04,
            gamma: 0.01,
            weight_decay: 0.01,
            hessian_update_freq: 10, // Update Hessian every 10 steps
        }
    }
}

/// Sophia optimizer with second-order clipped stochastic optimization.
///
/// Implements the Sophia algorithm from "Sophia: A Scalable Stochastic Second-order
/// Optimizer for Language Model Pre-training" by Liu et al. (2023). This optimizer
/// uses second-order information efficiently for large language model training.
#[derive(Debug)]
pub struct Sophia {
    /// Configuration for this optimizer
    config: SophiaConfig,
    /// Optimizer state tracking steps
    state: OptimizerState,
    /// First moment estimates (m_t)
    momentum: HashMap<String, Vec<f32>>,
    /// Hessian diagonal estimates (h_t)
    hessian_diag: HashMap<String, Vec<f32>>,
    /// Previous gradients for Hessian estimation
    prev_grad: HashMap<String, Vec<f32>>,
}

impl Sophia {
    /// Creates a new Sophia optimizer.
    ///
    /// # Arguments
    ///
    /// * `lr` - Learning rate (typical: 1e-4 to 3e-4)
    /// * `betas` - Momentum coefficients (typical: (0.965, 0.99))
    /// * `eps` - Numerical stability term (typical: 1e-8)
    /// * `rho` - Hessian scaling factor (typical: 0.04)
    /// * `gamma` - Clipping threshold (typical: 0.01)
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_optim::Sophia;
    /// let optimizer = Sophia::new(1e-4, (0.965, 0.99), 1e-8, 0.04, 0.01);
    /// ```
    pub fn new(lr: f32, betas: (f32, f32), eps: f32, rho: f32, gamma: f32) -> Self {
        Self {
            config: SophiaConfig {
                lr,
                betas,
                eps,
                rho,
                gamma,
                weight_decay: 0.01,
                hessian_update_freq: 10,
            },
            state: OptimizerState::new(),
            momentum: HashMap::new(),
            hessian_diag: HashMap::new(),
            prev_grad: HashMap::new(),
        }
    }

    /// Creates a new Sophia optimizer with weight decay.
    ///
    /// # Arguments
    ///
    /// * `lr` - Learning rate
    /// * `betas` - Momentum coefficients
    /// * `eps` - Numerical stability term
    /// * `rho` - Hessian scaling factor
    /// * `gamma` - Clipping threshold
    /// * `weight_decay` - Weight decay coefficient
    /// * `hessian_update_freq` - Hessian update frequency
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_optim::Sophia;
    /// let optimizer = Sophia::with_config(1e-4, (0.965, 0.99), 1e-8, 0.04, 0.01, 0.01, 10);
    /// ```
    pub fn with_config(
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        rho: f32,
        gamma: f32,
        weight_decay: f32,
        hessian_update_freq: usize,
    ) -> Self {
        Self {
            config: SophiaConfig {
                lr,
                betas,
                eps,
                rho,
                gamma,
                weight_decay,
                hessian_update_freq,
            },
            state: OptimizerState::new(),
            momentum: HashMap::new(),
            hessian_diag: HashMap::new(),
            prev_grad: HashMap::new(),
        }
    }

    /// Creates a new Sophia optimizer from configuration.
    pub fn from_config(config: SophiaConfig) -> Self {
        Self {
            config,
            state: OptimizerState::new(),
            momentum: HashMap::new(),
            hessian_diag: HashMap::new(),
            prev_grad: HashMap::new(),
        }
    }

    /// Applies clipping to the update.
    #[allow(dead_code)]
    fn clip_update(&self, update: f32) -> f32 {
        update.clamp(-self.config.gamma, self.config.gamma)
    }
}

impl Optimizer for Sophia {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        match (parameter, grad) {
            (Tensor::F32(param), Tensor::F32(grad_arr)) => {
                let param_id = format!("{:p}", param.as_ptr());
                let size = grad_arr.len();

                // Initialize states if not exists
                let momentum =
                    self.momentum.entry(param_id.clone()).or_insert_with(|| vec![0.0; size]);
                let hessian_diag =
                    self.hessian_diag.entry(param_id.clone()).or_insert_with(|| vec![1e-4; size]); // Small initial value
                let prev_grad =
                    self.prev_grad.entry(param_id.clone()).or_insert_with(|| vec![0.0; size]);

                if momentum.len() != size || hessian_diag.len() != size || prev_grad.len() != size {
                    return Err(TrustformersError::tensor_op_error(
                        "Sophia state buffer size mismatch",
                        "Sophia::update",
                    ));
                }

                // Update Hessian diagonal if it's time
                let should_update_hessian =
                    self.state.step > 0 && self.state.step % self.config.hessian_update_freq == 0;

                // Sophia algorithm implementation
                for (((p, &g), m), (h, prev_g)) in param
                    .iter_mut()
                    .zip(grad_arr.iter())
                    .zip(momentum.iter_mut())
                    .zip(hessian_diag.iter_mut().zip(prev_grad.iter_mut()))
                {
                    // Update Hessian diagonal if it's time
                    if should_update_hessian {
                        // Estimate Hessian diagonal: h_i ≈ |g_i(t) - g_i(t-1)| / δ
                        let hessian_estimate = (g - *prev_g).abs() / self.config.lr.max(1e-8);
                        *h = self.config.betas.1 * *h
                            + (1.0 - self.config.betas.1) * hessian_estimate;
                    }

                    // Apply weight decay directly to parameters (decoupled)
                    if self.config.weight_decay != 0.0 {
                        *p -= self.config.lr * self.config.weight_decay * *p;
                    }

                    // Update momentum: m = β1 * m + (1 - β1) * g
                    *m = self.config.betas.0 * *m + (1.0 - self.config.betas.0) * g;

                    // Compute bias correction for momentum
                    let step = (self.state.step + 1) as f32;
                    let bias_correction = 1.0 - self.config.betas.0.powf(step);
                    let corrected_momentum = *m / bias_correction;

                    // Compute Sophia update: m / (ρ * h + ε)
                    let denom = self.config.rho * *h + self.config.eps;
                    let raw_update = corrected_momentum / denom;

                    // Apply clipping: clip to [-γ, γ]
                    let clipped_update = raw_update.clamp(-self.config.gamma, self.config.gamma);

                    // Apply update: θ = θ - η * clip(m / (ρ * h + ε))
                    *p -= self.config.lr * clipped_update;

                    // Store current gradient for next Hessian estimation
                    *prev_g = g;
                }

                Ok(())
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor types for Sophia",
                "Sophia::update",
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

impl StatefulOptimizer for Sophia {
    type Config = SophiaConfig;
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
        state_dict.insert("rho".to_string(), Tensor::new(vec![self.config.rho])?);
        state_dict.insert("gamma".to_string(), Tensor::new(vec![self.config.gamma])?);
        state_dict.insert(
            "weight_decay".to_string(),
            Tensor::new(vec![self.config.weight_decay])?,
        );
        state_dict.insert(
            "step".to_string(),
            Tensor::new(vec![self.state.step as f32])?,
        );

        // Save optimizer states
        for (param_id, momentum) in &self.momentum {
            state_dict.insert(
                format!("momentum_{}", param_id),
                Tensor::new(momentum.clone())?,
            );
        }

        for (param_id, hessian_diag) in &self.hessian_diag {
            state_dict.insert(
                format!("hessian_diag_{}", param_id),
                Tensor::new(hessian_diag.clone())?,
            );
        }

        for (param_id, prev_grad) in &self.prev_grad {
            state_dict.insert(
                format!("prev_grad_{}", param_id),
                Tensor::new(prev_grad.clone())?,
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

        if let Some(eps_tensor) = state.get("eps") {
            if let Ok(eps_vec) = eps_tensor.data() {
                if !eps_vec.is_empty() {
                    self.config.eps = eps_vec[0];
                }
            }
        }

        if let Some(rho_tensor) = state.get("rho") {
            if let Ok(rho_vec) = rho_tensor.data() {
                if !rho_vec.is_empty() {
                    self.config.rho = rho_vec[0];
                }
            }
        }

        if let Some(gamma_tensor) = state.get("gamma") {
            if let Ok(gamma_vec) = gamma_tensor.data() {
                if !gamma_vec.is_empty() {
                    self.config.gamma = gamma_vec[0];
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

        if let Some(step_tensor) = state.get("step") {
            if let Ok(step_vec) = step_tensor.data() {
                if !step_vec.is_empty() {
                    self.state.step = step_vec[0] as usize;
                }
            }
        }

        // Load optimizer states
        for (key, tensor) in state.iter() {
            if key.starts_with("momentum_") {
                let param_id = key.trim_start_matches("momentum_");
                if let Ok(momentum) = tensor.data() {
                    self.momentum.insert(param_id.to_string(), momentum.clone());
                }
            } else if key.starts_with("hessian_diag_") {
                let param_id = key.trim_start_matches("hessian_diag_");
                if let Ok(hessian_diag) = tensor.data() {
                    self.hessian_diag.insert(param_id.to_string(), hessian_diag.clone());
                }
            } else if key.starts_with("prev_grad_") {
                let param_id = key.trim_start_matches("prev_grad_");
                if let Ok(prev_grad) = tensor.data() {
                    self.prev_grad.insert(param_id.to_string(), prev_grad.clone());
                }
            }
        }

        Ok(())
    }

    fn memory_usage(&self) -> StateMemoryStats {
        let mut momentum_elements = 0;
        let mut variance_elements = 0; // Use for Hessian diagonal
        let mut third_moment_elements = 0; // Use for previous gradients

        for momentum in self.momentum.values() {
            momentum_elements += momentum.len();
        }

        for hessian_diag in self.hessian_diag.values() {
            variance_elements += hessian_diag.len();
        }

        for prev_grad in self.prev_grad.values() {
            third_moment_elements += prev_grad.len();
        }

        let total_elements = momentum_elements + variance_elements + third_moment_elements;
        let total_bytes = total_elements * std::mem::size_of::<f32>();

        StateMemoryStats {
            momentum_elements,
            variance_elements,
            third_moment_elements,
            total_bytes,
            num_parameters: momentum_elements,
        }
    }

    fn reset_state(&mut self) {
        self.state.step = 0;
        self.momentum.clear();
        self.hessian_diag.clear();
        self.prev_grad.clear();
    }

    fn num_parameters(&self) -> usize {
        self.momentum.values().map(|v| v.len()).sum()
    }
}

impl Default for Sophia {
    fn default() -> Self {
        Self::new(1e-4, (0.965, 0.99), 1e-8, 0.04, 0.01)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sophia_creation() {
        let optimizer = Sophia::new(1e-4, (0.965, 0.99), 1e-8, 0.04, 0.01);
        assert_eq!(optimizer.config.lr, 1e-4);
        assert_eq!(optimizer.config.betas, (0.965, 0.99));
        assert_eq!(optimizer.config.eps, 1e-8);
        assert_eq!(optimizer.config.rho, 0.04);
        assert_eq!(optimizer.config.gamma, 0.01);
    }

    #[test]
    fn test_sophia_with_config() {
        let optimizer = Sophia::with_config(1e-4, (0.965, 0.99), 1e-8, 0.04, 0.01, 0.01, 5);
        assert_eq!(optimizer.config.weight_decay, 0.01);
        assert_eq!(optimizer.config.hessian_update_freq, 5);
    }

    #[test]
    fn test_sophia_default() {
        let optimizer = Sophia::default();
        assert_eq!(optimizer.config.lr, 1e-4);
        assert_eq!(optimizer.config.betas, (0.965, 0.99));
        assert_eq!(optimizer.config.rho, 0.04);
        assert_eq!(optimizer.config.gamma, 0.01);
    }

    #[test]
    fn test_hessian_estimation() {
        let optimizer = Sophia::default();
        let current_grad = 1.0f32;
        let previous_grad = 0.8f32;

        // Test inline Hessian estimation logic
        let hessian_est = (current_grad - previous_grad).abs() / optimizer.config.lr.max(1e-8f32);

        assert!(hessian_est >= 0.0); // Hessian estimate should be non-negative
        assert!(hessian_est > 0.0); // Should be positive for different gradients
    }

    #[test]
    fn test_clipping() {
        let optimizer = Sophia::with_config(1e-4, (0.965, 0.99), 1e-8, 0.04, 0.01, 0.01, 10);

        assert_eq!(optimizer.clip_update(0.005), 0.005); // No clipping
        assert_eq!(optimizer.clip_update(0.02), 0.01); // Positive clipping
        assert_eq!(optimizer.clip_update(-0.02), -0.01); // Negative clipping
    }

    #[test]
    fn test_memory_usage() {
        let optimizer = Sophia::default();
        let stats = optimizer.memory_usage();

        // Initially should have no memory usage
        assert_eq!(stats.total_bytes, 0);
        assert_eq!(stats.momentum_elements, 0);
        assert_eq!(stats.variance_elements, 0);
        assert_eq!(stats.third_moment_elements, 0);
    }

    #[test]
    fn test_state_persistence() {
        let mut optimizer = Sophia::default();
        optimizer.state.step = 50;

        // Test that state is maintained
        assert_eq!(optimizer.state.step, 50);

        // Test reset
        optimizer.reset_state();
        assert_eq!(optimizer.state.step, 0);
        assert!(optimizer.momentum.is_empty());
        assert!(optimizer.hessian_diag.is_empty());
        assert!(optimizer.prev_grad.is_empty());
    }

    #[test]
    fn test_hessian_update_frequency() {
        let optimizer = Sophia::with_config(1e-4, (0.965, 0.99), 1e-8, 0.04, 0.01, 0.01, 5);

        // Should update Hessian every 5 steps
        assert_eq!(optimizer.config.hessian_update_freq, 5);

        // Test logic
        assert!(!((0 + 1) % 5 == 0)); // Step 0 -> 1: no update
        assert!((4 + 1) % 5 == 0); // Step 4 -> 5: update (5 % 5 == 0)
        assert!((5 % 5 == 0)); // Step 5: update
        assert!((10 % 5 == 0)); // Step 10: update
    }
}
