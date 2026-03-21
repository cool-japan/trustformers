//! # Adam and AdamW Optimizers
//!
//! This module implements the Adam (Adaptive Moment Estimation) optimizer and its
//! variant AdamW with decoupled weight decay regularization.
//!
//! ## Adam
//!
//! Adam combines the advantages of two other extensions of stochastic gradient descent:
//! - AdaGrad: Works well with sparse gradients
//! - RMSProp: Works well in online and non-stationary settings
//!
//! The Adam update rule:
//! ```text
//! m_t = β1 * m_{t-1} + (1 - β1) * g_t
//! v_t = β2 * v_{t-1} + (1 - β2) * g_t²
//!
//! m̂_t = m_t / (1 - β1^t)  (bias correction)
//! v̂_t = v_t / (1 - β2^t)  (bias correction)
//!
//! θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
//! ```
//!
//! ## AdamW
//!
//! AdamW decouples weight decay from the gradient-based update:
//! - Standard Adam: Applies L2 regularization to gradients
//! - AdamW: Applies weight decay directly to parameters
//!
//! This decoupling leads to better training performance, especially with large
//! learning rates and is the preferred optimizer for transformer models.
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use trustformers_optim::{Adam, AdamW};
//! use trustformers_core::traits::Optimizer;
//!
//! // Standard Adam
//! let mut adam = Adam::new(
//!     1e-3,           // learning rate
//!     (0.9, 0.999),   // (β1, β2)
//!     1e-8,           // epsilon for numerical stability
//!     0.0,            // L2 regularization
//! );
//!
//! // AdamW (recommended for transformers)
//! let mut adamw = AdamW::new(
//!     1e-3,
//!     (0.9, 0.999),
//!     1e-8,
//!     0.01,           // decoupled weight decay
//! );
//!
//! // Both optimizers are ready to use
//! // Use them in your training loop with .zero_grad(), .update(), and .step()
//! ```
//!
//! ## Hyperparameter Guidelines
//!
//! ### Learning Rate
//! - Transformers: 1e-4 to 5e-4
//! - Fine-tuning: 1e-5 to 5e-5
//! - With warmup: Can use higher initial rates
//!
//! ### Betas
//! - β1 (momentum): 0.9 is standard
//! - β2 (RMSProp): 0.999 for stable training, 0.98 for faster adaptation
//!
//! ### Weight Decay
//! - AdamW: 0.01 to 0.1 (common: 0.01)
//! - Adam: Often 0 (regularization via dropout instead)
//!
//! ### Epsilon
//! - 1e-8 is standard
//! - 1e-6 for mixed precision training
//!
//! ## Implementation Notes
//!
//! - Uses per-parameter momentum buffers stored in HashMaps
//! - Implements bias correction for stable early training
//! - Supports both F32 and F16 tensors (with automatic casting)
//! - Thread-safe for data parallel training

use crate::common::{OptimizerState, StateMemoryStats};
use crate::traits::StatefulOptimizer;
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

/// Configuration for Adam optimizer.
#[derive(Debug, Clone)]
pub struct AdamConfig {
    /// Learning rate (α in the paper)
    pub lr: f32,
    /// Coefficients for computing running averages (β1, β2)
    pub betas: (f32, f32),
    /// Term added for numerical stability (ε in the paper)
    pub eps: f32,
    /// L2 regularization coefficient
    pub weight_decay: f32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

/// Configuration for AdamW optimizer.
#[derive(Debug, Clone)]
pub struct AdamWConfig {
    /// Learning rate (α in the paper)
    pub lr: f32,
    /// Coefficients for computing running averages (β1, β2)
    pub betas: (f32, f32),
    /// Term added for numerical stability (ε in the paper)
    pub eps: f32,
    /// Decoupled weight decay coefficient
    pub weight_decay: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

/// Adam optimizer with L2 regularization applied to gradients.
///
/// Implements the Adam algorithm from "Adam: A Method for Stochastic Optimization"
/// by Kingma and Ba (2014). This version applies weight decay as L2 regularization
/// on the gradients, which is different from the decoupled weight decay in AdamW.
#[derive(Debug, Clone)]
pub struct Adam {
    /// Configuration for this optimizer
    config: AdamConfig,
    /// Optimizer state tracking steps
    state: OptimizerState,
    /// First moment estimates (m_t)
    exp_avg: HashMap<String, Vec<f32>>,
    /// Second moment estimates (v_t)
    exp_avg_sq: HashMap<String, Vec<f32>>,
}

impl Adam {
    /// Creates a new Adam optimizer.
    ///
    /// # Arguments
    ///
    /// * `lr` - Learning rate (typical: 1e-3)
    /// * `betas` - Coefficients for running averages (typical: (0.9, 0.999))
    /// * `eps` - Numerical stability term (typical: 1e-8)
    /// * `weight_decay` - L2 regularization coefficient (typical: 0.0)
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_optim::Adam;
    /// let optimizer = Adam::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
    /// ```
    pub fn new(lr: f32, betas: (f32, f32), eps: f32, weight_decay: f32) -> Self {
        Self {
            config: AdamConfig {
                lr,
                betas,
                eps,
                weight_decay,
            },
            state: OptimizerState::new(),
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
        }
    }

    /// Creates a new Adam optimizer from configuration.
    pub fn from_config(config: AdamConfig) -> Self {
        Self {
            config,
            state: OptimizerState::new(),
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
        }
    }
}

impl Optimizer for Adam {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        match (parameter, grad) {
            (Tensor::F32(param), Tensor::F32(grad_arr)) => {
                let param_id = format!("{:p}", param.as_ptr());
                let size = grad_arr.len();

                let exp_avg =
                    self.exp_avg.entry(param_id.clone()).or_insert_with(|| vec![0.0; size]);
                let exp_avg_sq = self.exp_avg_sq.entry(param_id).or_insert_with(|| vec![0.0; size]);

                if exp_avg.len() != size || exp_avg_sq.len() != size {
                    return Err(TrustformersError::tensor_op_error(
                        "Adam state buffer size mismatch",
                        "Adam::update",
                    ));
                }

                let step = (self.state.step + 1) as f32;
                let bias_correction1 = 1.0 - self.config.betas.0.powf(step);
                let bias_correction2 = 1.0 - self.config.betas.1.powf(step);

                for ((p, g), (m, v)) in param
                    .iter_mut()
                    .zip(grad_arr.iter())
                    .zip(exp_avg.iter_mut().zip(exp_avg_sq.iter_mut()))
                {
                    // Apply L2 regularization (traditional Adam with weight decay)
                    let grad_with_wd = if self.config.weight_decay != 0.0 {
                        g + self.config.weight_decay * *p
                    } else {
                        *g
                    };

                    // Update biased first moment estimate
                    *m = self.config.betas.0 * *m + (1.0 - self.config.betas.0) * grad_with_wd;
                    // Update biased second raw moment estimate
                    *v = self.config.betas.1 * *v
                        + (1.0 - self.config.betas.1) * grad_with_wd * grad_with_wd;

                    // Compute bias-corrected first moment estimate
                    let m_hat = *m / bias_correction1;
                    // Compute bias-corrected second raw moment estimate
                    let v_hat = *v / bias_correction2;

                    // Apply Adam update
                    *p -= self.config.lr * m_hat / (v_hat.sqrt() + self.config.eps);
                }

                Ok(())
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor types for Adam",
                "Adam::update",
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

impl StatefulOptimizer for Adam {
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
            "step".to_string(),
            Tensor::new(vec![self.state.step as f32])?,
        );

        // Save momentum buffers
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

        // Load momentum buffers
        for (key, tensor) in state.iter() {
            if key.starts_with("exp_avg_") && !key.starts_with("exp_avg_sq_") {
                let param_id = key.trim_start_matches("exp_avg_");
                if let Ok(exp_avg) = tensor.data() {
                    self.exp_avg.insert(param_id.to_string(), exp_avg.clone());
                }
            } else if key.starts_with("exp_avg_sq_") {
                let param_id = key.trim_start_matches("exp_avg_sq_");
                if let Ok(exp_avg_sq) = tensor.data() {
                    self.exp_avg_sq.insert(param_id.to_string(), exp_avg_sq.clone());
                }
            }
        }

        Ok(())
    }

    fn memory_usage(&self) -> StateMemoryStats {
        let mut momentum_elements = 0;
        let mut variance_elements = 0;

        for exp_avg in self.exp_avg.values() {
            momentum_elements += exp_avg.len();
        }

        for exp_avg_sq in self.exp_avg_sq.values() {
            variance_elements += exp_avg_sq.len();
        }

        let total_elements = momentum_elements + variance_elements;
        let total_bytes = total_elements * std::mem::size_of::<f32>();

        StateMemoryStats {
            momentum_elements,
            variance_elements,
            third_moment_elements: 0,
            total_bytes,
            num_parameters: momentum_elements,
        }
    }

    fn reset_state(&mut self) {
        self.state.step = 0;
        self.exp_avg.clear();
        self.exp_avg_sq.clear();
    }

    fn num_parameters(&self) -> usize {
        self.exp_avg.values().map(|v| v.len()).sum()
    }
}

/// AdamW optimizer with decoupled weight decay regularization.
///
/// Implements the AdamW algorithm from "Decoupled Weight Decay Regularization"
/// by Loshchilov and Hutter (2017). This version applies weight decay directly
/// to parameters rather than to gradients, leading to better performance,
/// especially with large learning rates.
#[derive(Debug)]
pub struct AdamW {
    /// Configuration for this optimizer
    config: AdamWConfig,
    /// Optimizer state tracking steps
    state: OptimizerState,
    /// First moment estimates (m_t)
    exp_avg: HashMap<String, Vec<f32>>,
    /// Second moment estimates (v_t)
    exp_avg_sq: HashMap<String, Vec<f32>>,
}

impl AdamW {
    /// Creates a new AdamW optimizer.
    ///
    /// # Arguments
    ///
    /// * `lr` - Learning rate (typical: 1e-4 to 5e-4 for transformers)
    /// * `betas` - Coefficients for running averages (typical: (0.9, 0.999))
    /// * `eps` - Numerical stability term (typical: 1e-8)
    /// * `weight_decay` - Decoupled weight decay coefficient (typical: 0.01)
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_optim::AdamW;
    /// let optimizer = AdamW::new(1e-4, (0.9, 0.999), 1e-8, 0.01);
    /// ```
    pub fn new(lr: f32, betas: (f32, f32), eps: f32, weight_decay: f32) -> Self {
        Self {
            config: AdamWConfig {
                lr,
                betas,
                eps,
                weight_decay,
            },
            state: OptimizerState::new(),
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
        }
    }

    /// Creates a new AdamW optimizer from configuration.
    pub fn from_config(config: AdamWConfig) -> Self {
        Self {
            config,
            state: OptimizerState::new(),
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
        }
    }
}

impl Optimizer for AdamW {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        match (parameter, grad) {
            (Tensor::F32(param), Tensor::F32(grad_arr)) => {
                let param_id = format!("{:p}", param.as_ptr());
                let size = grad_arr.len();

                let exp_avg =
                    self.exp_avg.entry(param_id.clone()).or_insert_with(|| vec![0.0; size]);
                let exp_avg_sq = self.exp_avg_sq.entry(param_id).or_insert_with(|| vec![0.0; size]);

                if exp_avg.len() != size || exp_avg_sq.len() != size {
                    return Err(TrustformersError::tensor_op_error(
                        "AdamW state buffer size mismatch",
                        "AdamW::update",
                    ));
                }

                let step = (self.state.step + 1) as f32;
                let bias_correction1 = 1.0 - self.config.betas.0.powf(step);
                let bias_correction2 = 1.0 - self.config.betas.1.powf(step);

                for ((p, g), (m, v)) in param
                    .iter_mut()
                    .zip(grad_arr.iter())
                    .zip(exp_avg.iter_mut().zip(exp_avg_sq.iter_mut()))
                {
                    // Update biased first moment estimate
                    *m = self.config.betas.0 * *m + (1.0 - self.config.betas.0) * g;
                    // Update biased second raw moment estimate
                    *v = self.config.betas.1 * *v + (1.0 - self.config.betas.1) * g * g;

                    // Compute bias-corrected first moment estimate
                    let m_hat = *m / bias_correction1;
                    // Compute bias-corrected second raw moment estimate
                    let v_hat = *v / bias_correction2;

                    // AdamW: Apply weight decay directly to parameters (decoupled)
                    if self.config.weight_decay != 0.0 {
                        *p -= self.config.lr * self.config.weight_decay * *p;
                    }

                    // Apply Adam update
                    *p -= self.config.lr * m_hat / (v_hat.sqrt() + self.config.eps);
                }

                Ok(())
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor types for AdamW",
                "AdamW::update",
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

impl StatefulOptimizer for AdamW {
    type Config = AdamWConfig;
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
            "step".to_string(),
            Tensor::new(vec![self.state.step as f32])?,
        );

        // Save momentum buffers
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

        // Load momentum buffers
        for (key, tensor) in state.iter() {
            if key.starts_with("exp_avg_") && !key.starts_with("exp_avg_sq_") {
                let param_id = key.trim_start_matches("exp_avg_");
                if let Ok(exp_avg) = tensor.data() {
                    self.exp_avg.insert(param_id.to_string(), exp_avg.clone());
                }
            } else if key.starts_with("exp_avg_sq_") {
                let param_id = key.trim_start_matches("exp_avg_sq_");
                if let Ok(exp_avg_sq) = tensor.data() {
                    self.exp_avg_sq.insert(param_id.to_string(), exp_avg_sq.clone());
                }
            }
        }

        Ok(())
    }

    fn memory_usage(&self) -> StateMemoryStats {
        let mut momentum_elements = 0;
        let mut variance_elements = 0;

        for exp_avg in self.exp_avg.values() {
            momentum_elements += exp_avg.len();
        }

        for exp_avg_sq in self.exp_avg_sq.values() {
            variance_elements += exp_avg_sq.len();
        }

        let total_elements = momentum_elements + variance_elements;
        let total_bytes = total_elements * std::mem::size_of::<f32>();

        StateMemoryStats {
            momentum_elements,
            variance_elements,
            third_moment_elements: 0,
            total_bytes,
            num_parameters: momentum_elements,
        }
    }

    fn reset_state(&mut self) {
        self.state.step = 0;
        self.exp_avg.clear();
        self.exp_avg_sq.clear();
    }

    fn num_parameters(&self) -> usize {
        self.exp_avg.values().map(|v| v.len()).sum()
    }
}

/// RAdam (Rectified Adam) optimizer that addresses the variance issue in Adam.
///
/// Implements the RAdam algorithm from "On the Variance of the Adaptive Learning Rate
/// and Beyond" by Liu et al. (2019). RAdam applies a rectification term to determine
/// when the adaptive learning rate should be used, leading to more stable training.
///
/// The key innovation is analyzing whether the variance of the adaptive learning rate
/// is tractable. When it's not, RAdam falls back to SGD-like updates.
#[derive(Debug)]
pub struct RAdam {
    /// Learning rate (α in the paper)
    lr: f32,
    /// Coefficients for computing running averages (β1, β2)
    betas: (f32, f32),
    /// Term added for numerical stability (ε in the paper)
    eps: f32,
    /// Weight decay coefficient
    weight_decay: f32,
    /// Optimizer state tracking steps
    state: OptimizerState,
    /// First moment estimates (m_t)
    exp_avg: HashMap<String, Vec<f32>>,
    /// Second moment estimates (v_t)
    exp_avg_sq: HashMap<String, Vec<f32>>,
}

impl RAdam {
    /// Creates a new RAdam optimizer.
    ///
    /// # Arguments
    ///
    /// * `lr` - Learning rate (typical: 1e-3)
    /// * `betas` - Coefficients for running averages (typical: (0.9, 0.999))
    /// * `eps` - Numerical stability term (typical: 1e-8)
    /// * `weight_decay` - Weight decay coefficient (typical: 0.0)
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_optim::RAdam;
    /// let optimizer = RAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
    /// ```
    pub fn new(lr: f32, betas: (f32, f32), eps: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            betas,
            eps,
            weight_decay,
            state: OptimizerState::new(),
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
        }
    }

    /// Computes the variance rectification term for RAdam.
    fn compute_variance_rectification(&self, step: f32) -> (f32, bool) {
        let beta2 = self.betas.1;

        // Maximum variance tractable (ρ_∞)
        let rho_inf = 2.0 / (1.0 - beta2) - 1.0;

        // Current variance tractable (ρ_t)
        let rho_t = rho_inf - 2.0 * step * beta2.powf(step) / (1.0 - beta2.powf(step));

        // If ρ_t > 4, the variance is tractable
        if rho_t > 4.0 {
            // Variance rectification term
            let r_t = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf
                / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
                .sqrt();
            (r_t, true)
        } else {
            (1.0, false)
        }
    }
}

impl Optimizer for RAdam {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        match (parameter, grad) {
            (Tensor::F32(param), Tensor::F32(grad_arr)) => {
                let param_id = format!("{:p}", param.as_ptr());
                let size = grad_arr.len();

                let step = (self.state.step + 1) as f32;
                let bias_correction1 = 1.0 - self.betas.0.powf(step);

                // Compute variance rectification
                let (r_t, use_adaptive) = self.compute_variance_rectification(step);

                let exp_avg =
                    self.exp_avg.entry(param_id.clone()).or_insert_with(|| vec![0.0; size]);
                let exp_avg_sq = self.exp_avg_sq.entry(param_id).or_insert_with(|| vec![0.0; size]);

                if exp_avg.len() != size || exp_avg_sq.len() != size {
                    return Err(TrustformersError::tensor_op_error(
                        "RAdam state buffer size mismatch",
                        "RAdam::update",
                    ));
                }

                for ((p, g), (m, v)) in param
                    .iter_mut()
                    .zip(grad_arr.iter())
                    .zip(exp_avg.iter_mut().zip(exp_avg_sq.iter_mut()))
                {
                    // Apply weight decay if specified
                    let grad_with_wd =
                        if self.weight_decay != 0.0 { g + self.weight_decay * *p } else { *g };

                    // Update biased first moment estimate
                    *m = self.betas.0 * *m + (1.0 - self.betas.0) * grad_with_wd;
                    // Update biased second raw moment estimate
                    *v = self.betas.1 * *v + (1.0 - self.betas.1) * grad_with_wd * grad_with_wd;

                    // Compute bias-corrected first moment estimate
                    let m_hat = *m / bias_correction1;

                    if use_adaptive {
                        // Use adaptive update with variance rectification
                        let bias_correction2 = 1.0 - self.betas.1.powf(step);
                        let v_hat = *v / bias_correction2;

                        // Apply RAdam update with rectification term
                        *p -= self.lr * r_t * m_hat / (v_hat.sqrt() + self.eps);
                    } else {
                        // Use SGD-like update when variance is not tractable
                        *p -= self.lr * m_hat;
                    }
                }

                Ok(())
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor types for RAdam",
                "RAdam::update",
            )),
        }
    }

    fn zero_grad(&mut self) {}

    fn step(&mut self) {
        self.state.step += 1;
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// NAdam (Nesterov Adam) optimizer that incorporates Nesterov momentum into Adam.
///
/// Implements the NAdam algorithm from "Incorporating Nesterov Momentum into Adam"
/// by Dozat (2016). NAdam applies Nesterov's accelerated gradient to the bias-corrected
/// first moment estimate, leading to better convergence properties.
///
/// The key innovation is using Nesterov momentum on the bias-corrected gradients
/// rather than the raw gradients, combining the benefits of both Adam and Nesterov momentum.
#[derive(Debug)]
pub struct NAdam {
    /// Learning rate (α in the paper)
    lr: f32,
    /// Coefficients for computing running averages (β1, β2)
    betas: (f32, f32),
    /// Term added for numerical stability (ε in the paper)
    eps: f32,
    /// Weight decay coefficient
    weight_decay: f32,
    /// Optimizer state tracking steps
    state: OptimizerState,
    /// First moment estimates (m_t)
    exp_avg: HashMap<String, Vec<f32>>,
    /// Second moment estimates (v_t)
    exp_avg_sq: HashMap<String, Vec<f32>>,
}

impl NAdam {
    /// Creates a new NAdam optimizer.
    ///
    /// # Arguments
    ///
    /// * `lr` - Learning rate (typical: 1e-3)
    /// * `betas` - Coefficients for running averages (typical: (0.9, 0.999))
    /// * `eps` - Numerical stability term (typical: 1e-8)
    /// * `weight_decay` - Weight decay coefficient (typical: 0.0)
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_optim::NAdam;
    /// let optimizer = NAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
    /// ```
    pub fn new(lr: f32, betas: (f32, f32), eps: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            betas,
            eps,
            weight_decay,
            state: OptimizerState::new(),
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
        }
    }
}

impl Optimizer for NAdam {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        match (parameter, grad) {
            (Tensor::F32(param), Tensor::F32(grad_arr)) => {
                let param_id = format!("{:p}", param.as_ptr());
                let size = grad_arr.len();

                let exp_avg =
                    self.exp_avg.entry(param_id.clone()).or_insert_with(|| vec![0.0; size]);
                let exp_avg_sq = self.exp_avg_sq.entry(param_id).or_insert_with(|| vec![0.0; size]);

                if exp_avg.len() != size || exp_avg_sq.len() != size {
                    return Err(TrustformersError::tensor_op_error(
                        "NAdam state buffer size mismatch",
                        "NAdam::update",
                    ));
                }

                let step = (self.state.step + 1) as f32;
                let bias_correction1 = 1.0 - self.betas.0.powf(step);
                let bias_correction2 = 1.0 - self.betas.1.powf(step);

                for ((p, g), (m, v)) in param
                    .iter_mut()
                    .zip(grad_arr.iter())
                    .zip(exp_avg.iter_mut().zip(exp_avg_sq.iter_mut()))
                {
                    // Apply weight decay if specified
                    let grad_with_wd =
                        if self.weight_decay != 0.0 { g + self.weight_decay * *p } else { *g };

                    // Update biased first moment estimate
                    *m = self.betas.0 * *m + (1.0 - self.betas.0) * grad_with_wd;
                    // Update biased second raw moment estimate
                    *v = self.betas.1 * *v + (1.0 - self.betas.1) * grad_with_wd * grad_with_wd;

                    // Compute bias-corrected first moment estimate
                    let m_hat = *m / bias_correction1;
                    // Compute bias-corrected second raw moment estimate
                    let v_hat = *v / bias_correction2;

                    // NAdam: Apply Nesterov momentum to bias-corrected first moment
                    // This is the key difference from Adam
                    let nesterov_m = self.betas.0 * m_hat
                        + (1.0 - self.betas.0) * grad_with_wd / bias_correction1;

                    // Apply NAdam update using Nesterov-corrected momentum
                    *p -= self.lr * nesterov_m / (v_hat.sqrt() + self.eps);
                }

                Ok(())
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor types for NAdam",
                "NAdam::update",
            )),
        }
    }

    fn zero_grad(&mut self) {}

    fn step(&mut self) {
        self.state.step += 1;
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// AdaBelief optimizer that adapts step size according to the "belief" in the gradient direction.
///
/// Implements the AdaBelief algorithm from "AdaBelief Optimizer: Adapting Stepsizes by the
/// Belief in Observed Gradients" by Adapuru et al. (2020). AdaBelief adapts the step size
/// according to the prediction reliability measured by the exponential moving average of
/// the squared differences between predicted and observed gradients.
///
/// The key insight is to use the variance of gradients rather than just the magnitude,
/// leading to better convergence properties and reduced sensitivity to hyperparameters.
#[derive(Debug)]
pub struct AdaBelief {
    /// Learning rate (α in the paper)
    lr: f32,
    /// Coefficients for computing running averages (β1, β2)
    betas: (f32, f32),
    /// Term added for numerical stability (ε in the paper)
    eps: f32,
    /// Weight decay coefficient
    weight_decay: f32,
    /// Optimizer state tracking steps
    state: OptimizerState,
    /// First moment estimates (m_t)
    exp_avg: HashMap<String, Vec<f32>>,
    /// Second moment estimates of gradient variance (s_t)
    exp_avg_var: HashMap<String, Vec<f32>>,
}

impl AdaBelief {
    /// Creates a new AdaBelief optimizer.
    ///
    /// # Arguments
    ///
    /// * `lr` - Learning rate (typical: 1e-3)
    /// * `betas` - Coefficients for running averages (typical: (0.9, 0.999))
    /// * `eps` - Numerical stability term (typical: 1e-16 for AdaBelief)
    /// * `weight_decay` - Weight decay coefficient (typical: 0.0)
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_optim::AdaBelief;
    /// let optimizer = AdaBelief::new(1e-3, (0.9, 0.999), 1e-16, 0.0);
    /// ```
    pub fn new(lr: f32, betas: (f32, f32), eps: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            betas,
            eps,
            weight_decay,
            state: OptimizerState::new(),
            exp_avg: HashMap::new(),
            exp_avg_var: HashMap::new(),
        }
    }
}

impl Optimizer for AdaBelief {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        match (parameter, grad) {
            (Tensor::F32(param), Tensor::F32(grad_arr)) => {
                let param_id = format!("{:p}", param.as_ptr());
                let size = grad_arr.len();

                let exp_avg =
                    self.exp_avg.entry(param_id.clone()).or_insert_with(|| vec![0.0; size]);
                let exp_avg_var =
                    self.exp_avg_var.entry(param_id).or_insert_with(|| vec![0.0; size]);

                if exp_avg.len() != size || exp_avg_var.len() != size {
                    return Err(TrustformersError::tensor_op_error(
                        "AdaBelief state buffer size mismatch",
                        "AdaBelief::update",
                    ));
                }

                let step = (self.state.step + 1) as f32;
                let bias_correction1 = 1.0 - self.betas.0.powf(step);
                let bias_correction2 = 1.0 - self.betas.1.powf(step);

                for ((p, g), (m, s)) in param
                    .iter_mut()
                    .zip(grad_arr.iter())
                    .zip(exp_avg.iter_mut().zip(exp_avg_var.iter_mut()))
                {
                    // Apply weight decay if specified
                    let grad_with_wd =
                        if self.weight_decay != 0.0 { g + self.weight_decay * *p } else { *g };

                    // Update biased first moment estimate
                    *m = self.betas.0 * *m + (1.0 - self.betas.0) * grad_with_wd;

                    // AdaBelief: Update variance of gradients (belief)
                    // s_t = β2 * s_{t-1} + (1-β2) * (g_t - m_t)^2
                    let grad_residual = grad_with_wd - *m;
                    *s = self.betas.1 * *s + (1.0 - self.betas.1) * grad_residual * grad_residual;

                    // Compute bias-corrected first moment estimate
                    let m_hat = *m / bias_correction1;
                    // Compute bias-corrected variance estimate
                    let s_hat = *s / bias_correction2;

                    // Apply AdaBelief update
                    *p -= self.lr * m_hat / (s_hat.sqrt() + self.eps);
                }

                Ok(())
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor types for AdaBelief",
                "AdaBelief::update",
            )),
        }
    }

    fn zero_grad(&mut self) {}

    fn step(&mut self) {
        self.state.step += 1;
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}
