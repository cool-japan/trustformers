//! # AdaMax+ Optimizer
//!
//! Implementation of AdaMax+ (Enhanced AdaMax with Momentum Scheduling), an advanced
//! variant of the AdaMax optimizer that incorporates adaptive momentum scheduling,
//! improved numerical stability, and enhanced convergence properties.
//!
//! ## Key Features
//!
//! - **Adaptive Momentum Scheduling**: Dynamic β₁ adjustment based on training progress
//! - **Enhanced Numerical Stability**: Improved handling of extreme gradient values
//! - **Convergence Acceleration**: Advanced momentum scheduling for faster convergence
//! - **Variance-Aware Updates**: Optional variance tracking for better adaptation
//!
//! ## Algorithm Description
//!
//! AdaMax+ extends the standard AdaMax algorithm with:
//! 1. Adaptive momentum scheduling based on gradient variance
//! 2. Enhanced infinity norm computation with outlier handling
//! 3. Learning rate warm-up and scheduling capabilities
//! 4. Optional bias correction improvements
//!
//! The AdaMax+ update rule:
//! ```text
//! # Adaptive momentum parameter
//! β₁_t = β₁_base * (1 - α * variance_factor)
//!
//! # First moment estimation
//! m_t = β₁_t * m_{t-1} + (1 - β₁_t) * g_t
//!
//! # Infinity norm with outlier handling
//! u_t = max(β₂ * u_{t-1}, |g_t|_∞)
//!
//! # Enhanced bias correction
//! m̂_t = m_t / (1 - β₁_t^t)
//!
//! # Parameter update with warm-up
//! lr_t = lr * min(1, t / warmup_steps)
//! θ_t = θ_{t-1} - (lr_t / u_t) * m̂_t
//! ```
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use trustformers_optim::{AdaMaxPlus, AdaMaxPlusConfig};
//! use trustformers_core::traits::Optimizer;
//!
//! // Create AdaMax+ optimizer with default settings
//! let mut optimizer = AdaMaxPlus::new(
//!     1e-3,      // learning rate
//!     (0.9, 0.999), // (β₁, β₂)
//!     1e-8,      // epsilon
//!     0.01,      // weight decay
//! );
//!
//! // Or create with advanced configuration
//! let config = AdaMaxPlusConfig::new()
//!     .learning_rate(0.002)
//!     .betas((0.95, 0.999))
//!     .enable_adaptive_momentum(true)
//!     .warmup_steps(1000)
//!     .variance_tracking(true);
//!
//! let mut optimizer = AdaMaxPlus::from_config(config);
//! ```
//!
//! ## Research Foundation
//!
//! This implementation builds on:
//! - Original AdaMax algorithm (Kingma & Ba, 2014)
//! - Adaptive momentum scheduling techniques
//! - Recent advances in optimization stability and convergence

use crate::common::{OptimizerState, StateMemoryStats};
use crate::traits::StatefulOptimizer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::errors::Result;
use trustformers_core::{tensor::Tensor, traits::Optimizer};

/// Configuration for AdaMax+ optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaMaxPlusConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Exponential decay rates for moment estimates (β₁, β₂)
    pub betas: (f32, f32),
    /// Small constant for numerical stability
    pub epsilon: f32,
    /// Weight decay (L2 regularization)
    pub weight_decay: f32,
    /// Enable adaptive momentum scheduling
    pub adaptive_momentum: bool,
    /// Momentum adaptation strength
    pub momentum_adaptation_strength: f32,
    /// Number of warm-up steps
    pub warmup_steps: usize,
    /// Enable variance tracking for momentum adaptation
    pub variance_tracking: bool,
    /// Bias correction improvement factor
    pub bias_correction_factor: f32,
    /// Outlier handling threshold for infinity norm
    pub outlier_threshold: f32,
}

impl Default for AdaMaxPlusConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            betas: (0.9, 0.999),
            epsilon: 1e-8,
            weight_decay: 0.0,
            adaptive_momentum: true,
            momentum_adaptation_strength: 0.1,
            warmup_steps: 0,
            variance_tracking: true,
            bias_correction_factor: 1.0,
            outlier_threshold: 10.0,
        }
    }
}

impl AdaMaxPlusConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set beta parameters
    pub fn betas(mut self, betas: (f32, f32)) -> Self {
        self.betas = betas;
        self
    }

    /// Set epsilon
    pub fn epsilon(mut self, eps: f32) -> Self {
        self.epsilon = eps;
        self
    }

    /// Set weight decay
    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Enable/disable adaptive momentum
    pub fn enable_adaptive_momentum(mut self, enabled: bool) -> Self {
        self.adaptive_momentum = enabled;
        self
    }

    /// Set momentum adaptation strength
    pub fn momentum_adaptation_strength(mut self, strength: f32) -> Self {
        self.momentum_adaptation_strength = strength;
        self
    }

    /// Set warmup steps
    pub fn warmup_steps(mut self, steps: usize) -> Self {
        self.warmup_steps = steps;
        self
    }

    /// Enable/disable variance tracking
    pub fn variance_tracking(mut self, enabled: bool) -> Self {
        self.variance_tracking = enabled;
        self
    }

    /// Set bias correction factor
    pub fn bias_correction_factor(mut self, factor: f32) -> Self {
        self.bias_correction_factor = factor;
        self
    }

    /// Set outlier threshold
    pub fn outlier_threshold(mut self, threshold: f32) -> Self {
        self.outlier_threshold = threshold;
        self
    }
}

/// State for a single parameter in AdaMax+ optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaMaxPlusParameterState {
    /// First moment estimate (momentum)
    pub momentum: Vec<f32>,
    /// Infinity norm estimate
    pub inf_norm: f32,
    /// Gradient variance (if variance tracking is enabled)
    pub gradient_variance: f32,
    /// Step count for this parameter
    pub step_count: usize,
    /// Exponential moving average of gradients for variance computation
    pub grad_ema: Option<Vec<f32>>,
    /// Exponential moving average of squared gradients for variance computation
    pub grad_sq_ema: Option<Vec<f32>>,
}

/// AdaMax+ optimizer state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaMaxPlusState {
    /// Common optimizer state (momentum, variance, etc.)
    pub state: OptimizerState,
    /// Configuration
    pub config: AdaMaxPlusConfig,
    /// Global step count
    pub step_count: usize,
    /// Parameter-specific infinity norms
    pub inf_norms: HashMap<String, f32>,
    /// Gradient variances (if tracking enabled)
    pub gradient_variances: HashMap<String, f32>,
    /// Parameter step counts
    pub param_step_counts: HashMap<String, usize>,
}

impl AdaMaxPlusState {
    /// Create new optimizer state
    pub fn new(config: AdaMaxPlusConfig) -> Self {
        Self {
            state: OptimizerState::new(),
            config,
            step_count: 0,
            inf_norms: HashMap::new(),
            gradient_variances: HashMap::new(),
            param_step_counts: HashMap::new(),
        }
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        // Calculate approximate memory usage
        let momentum_size = self.state.momentum.values().map(|v| v.len() * 4).sum::<usize>(); // 4 bytes per f32
        let variance_size = self.state.variance.values().map(|v| v.len() * 4).sum::<usize>();
        let inf_norms_size = self.inf_norms.len() * 4; // 4 bytes per f32
        let gradient_variances_size = self.gradient_variances.len() * 4;
        let param_step_counts_size = self.param_step_counts.len() * 8; // 8 bytes per usize

        momentum_size
            + variance_size
            + inf_norms_size
            + gradient_variances_size
            + param_step_counts_size
    }
}

/// AdaMax+ optimizer implementation
pub struct AdaMaxPlus {
    state: AdaMaxPlusState,
}

impl AdaMaxPlus {
    /// Create a new AdaMax+ optimizer with basic parameters
    pub fn new(learning_rate: f32, betas: (f32, f32), epsilon: f32, weight_decay: f32) -> Self {
        let config = AdaMaxPlusConfig {
            learning_rate,
            betas,
            epsilon,
            weight_decay,
            ..Default::default()
        };

        Self {
            state: AdaMaxPlusState::new(config),
        }
    }

    /// Create AdaMax+ optimizer from configuration
    pub fn from_config(config: AdaMaxPlusConfig) -> Self {
        Self {
            state: AdaMaxPlusState::new(config),
        }
    }

    /// Create AdaMax+ optimized for large language models
    pub fn for_large_models() -> Self {
        let config = AdaMaxPlusConfig::new()
            .learning_rate(0.0002)
            .betas((0.9, 0.999))
            .enable_adaptive_momentum(true)
            .warmup_steps(10000)
            .variance_tracking(true)
            .weight_decay(0.1);

        Self::from_config(config)
    }

    /// Create AdaMax+ optimized for fast training
    pub fn for_fast_training() -> Self {
        let config = AdaMaxPlusConfig::new()
            .learning_rate(0.003)
            .betas((0.95, 0.999))
            .enable_adaptive_momentum(true)
            .momentum_adaptation_strength(0.2)
            .warmup_steps(500);

        Self::from_config(config)
    }

    /// Create AdaMax+ optimized for stable training
    pub fn for_stable_training() -> Self {
        let config = AdaMaxPlusConfig::new()
            .learning_rate(0.001)
            .betas((0.9, 0.999))
            .enable_adaptive_momentum(false)
            .variance_tracking(false)
            .bias_correction_factor(1.2)
            .outlier_threshold(5.0);

        Self::from_config(config)
    }

    /// Compute adaptive momentum parameter
    fn compute_adaptive_momentum(&self, param_id: String) -> f32 {
        if !self.state.config.adaptive_momentum {
            return self.state.config.betas.0;
        }

        let base_beta1 = self.state.config.betas.0;
        let adaptation_strength = self.state.config.momentum_adaptation_strength;

        // Use gradient variance to adapt momentum
        let variance_factor = if self.state.config.variance_tracking {
            self.state.gradient_variances.get(&param_id).copied().unwrap_or(0.0).min(1.0)
        } else {
            0.0
        };

        // Adaptive momentum: higher variance -> lower momentum for better adaptation
        let adaptive_beta1 = base_beta1 * (1.0 - adaptation_strength * variance_factor);
        adaptive_beta1.clamp(0.1, 0.99) // Clamp to reasonable range
    }

    /// Compute learning rate with warm-up
    fn compute_effective_learning_rate(&self) -> f32 {
        let base_lr = self.state.config.learning_rate;

        if self.state.config.warmup_steps == 0 {
            return base_lr;
        }

        let warmup_factor = if self.state.step_count <= self.state.config.warmup_steps {
            self.state.step_count as f32 / self.state.config.warmup_steps as f32
        } else {
            1.0
        };

        base_lr * warmup_factor
    }

    /// Update gradient variance tracking
    fn update_gradient_variance(&mut self, param_id: String, gradient: &Tensor) -> Result<()> {
        if !self.state.config.variance_tracking {
            return Ok(());
        }

        let beta1 = self.state.config.betas.0;
        let beta2 = self.state.config.betas.1;

        let gradient_data = gradient.data()?;
        let param_size = gradient_data.len();

        // Get or initialize variance tracking buffers
        let grad_ema = self
            .state
            .state
            .get_or_create_momentum(format!("{}_grad_ema", param_id), param_size)
            .clone();
        let grad_sq_ema = self
            .state
            .state
            .get_or_create_variance(format!("{}_grad_sq_ema", param_id), param_size)
            .clone();

        // Update gradient EMA: m = β₁ * m + (1 - β₁) * g
        let updated_grad_ema: Vec<f32> = grad_ema
            .iter()
            .zip(gradient_data.iter())
            .map(|(&m, &g)| beta1 * m + (1.0 - beta1) * g)
            .collect();

        // Update squared gradient EMA: v = β₂ * v + (1 - β₂) * g²
        let updated_grad_sq_ema: Vec<f32> = grad_sq_ema
            .iter()
            .zip(gradient_data.iter())
            .map(|(&v, &g)| beta2 * v + (1.0 - beta2) * g * g)
            .collect();

        // Compute variance: Var[g] = E[g²] - E[g]²
        let variance: f32 = updated_grad_sq_ema
            .iter()
            .zip(updated_grad_ema.iter())
            .map(|(&sq_ema, &ema)| sq_ema - ema * ema)
            .sum::<f32>()
            / param_size as f32;

        // Store updated values
        self.state
            .state
            .momentum
            .insert(format!("{}_grad_ema", param_id), updated_grad_ema);
        self.state
            .state
            .variance
            .insert(format!("{}_grad_sq_ema", param_id), updated_grad_sq_ema);
        self.state.gradient_variances.insert(param_id, variance);

        Ok(())
    }
}

impl Optimizer for AdaMaxPlus {
    fn step(&mut self) {
        // Default step implementation - parameters are updated via update() calls
    }

    fn zero_grad(&mut self) {
        // Clear gradients - implementation specific to the framework
    }

    fn update(&mut self, parameter: &mut Tensor, gradient: &Tensor) -> Result<()> {
        let param_id = format!("{:p}", parameter.data()?.as_ptr());
        self.state.step_count += 1;

        // Get parameter size for initialization
        let param_size = parameter.data()?.len();

        // Get or initialize momentum using OptimizerState methods
        let momentum_data = {
            let momentum_buffer =
                self.state.state.get_or_create_momentum(param_id.clone(), param_size);
            momentum_buffer.clone()
        };

        // Update gradient variance if enabled
        if self.state.config.variance_tracking {
            self.update_gradient_variance(param_id.clone(), gradient)?;
        }

        // Apply weight decay to gradient if specified
        let effective_gradient = if self.state.config.weight_decay > 0.0 {
            gradient.add(&parameter.mul_scalar(self.state.config.weight_decay)?)?
        } else {
            gradient.clone()
        };

        // Get adaptive momentum parameter
        let adaptive_beta1 = self.compute_adaptive_momentum(param_id.clone());
        let beta2 = self.state.config.betas.1;

        // Update momentum: m_t = β₁_adaptive * m_{t-1} + (1 - β₁_adaptive) * g_t
        let gradient_data = effective_gradient.data()?;
        let updated_momentum: Vec<f32> = momentum_data
            .iter()
            .zip(gradient_data.iter())
            .map(|(&m, &g)| adaptive_beta1 * m + (1.0 - adaptive_beta1) * g)
            .collect();

        // Update infinity norm with outlier handling
        let grad_inf_norm = gradient_data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let clamped_grad_norm = grad_inf_norm.min(self.state.config.outlier_threshold);
        let current_inf_norm = self.state.inf_norms.get(&param_id).copied().unwrap_or(0.0);
        let new_inf_norm = (beta2 * current_inf_norm).max(clamped_grad_norm);
        self.state.inf_norms.insert(param_id.clone(), new_inf_norm);

        // Get parameter step count
        let step_count = self.state.param_step_counts.entry(param_id.clone()).or_insert(0);
        *step_count += 1;

        // Enhanced bias correction
        let bias_correction = 1.0 - adaptive_beta1.powi(*step_count as i32);
        let bias_corrected_momentum: Vec<f32> = updated_momentum
            .iter()
            .map(|&m| m / (bias_correction * self.state.config.bias_correction_factor))
            .collect();

        // Compute effective learning rate with warm-up
        let effective_lr = self.compute_effective_learning_rate();

        // Compute step size with numerical stability
        let step_size = effective_lr / (new_inf_norm + self.state.config.epsilon);

        // Update parameters: θ_t = θ_{t-1} - step_size * m̂_t
        let param_data = parameter.data()?;
        let updated_params: Vec<f32> = param_data
            .iter()
            .zip(bias_corrected_momentum.iter())
            .map(|(&p, &m)| p - step_size * m)
            .collect();

        *parameter = Tensor::new(updated_params)?;

        // Store updated momentum
        self.state.state.momentum.insert(param_id, updated_momentum);

        Ok(())
    }

    fn set_lr(&mut self, lr: f32) {
        self.state.config.learning_rate = lr;
    }

    fn get_lr(&self) -> f32 {
        self.state.config.learning_rate
    }
}

impl StatefulOptimizer for AdaMaxPlus {
    type Config = AdaMaxPlusConfig;
    type State = AdaMaxPlusState;

    fn config(&self) -> &Self::Config {
        &self.state.config
    }

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }

    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut state_dict = HashMap::new();

        // Convert momentum buffers to tensors
        for (key, buffer) in &self.state.state.momentum {
            let tensor = Tensor::new(buffer.clone())?;
            state_dict.insert(format!("{}_momentum", key), tensor);
        }

        // Convert variance buffers to tensors (if any)
        for (key, buffer) in &self.state.state.variance {
            let tensor = Tensor::new(buffer.clone())?;
            state_dict.insert(format!("{}_variance", key), tensor);
        }

        // Add infinity norms
        for (key, &inf_norm) in &self.state.inf_norms {
            let tensor = Tensor::new(vec![inf_norm])?;
            state_dict.insert(format!("{}_inf_norm", key), tensor);
        }

        // Add gradient variances
        for (key, &variance) in &self.state.gradient_variances {
            let tensor = Tensor::new(vec![variance])?;
            state_dict.insert(format!("{}_gradient_variance", key), tensor);
        }

        // Add parameter step counts
        for (key, &step_count) in &self.state.param_step_counts {
            let tensor = Tensor::new(vec![step_count as f32])?;
            state_dict.insert(format!("{}_step_count", key), tensor);
        }

        // Add global step count
        let step_tensor = Tensor::new(vec![self.state.step_count as f32])?;
        state_dict.insert("step_count".to_string(), step_tensor);

        Ok(state_dict)
    }

    fn load_state_dict(&mut self, state_dict: HashMap<String, Tensor>) -> Result<()> {
        for (key, tensor) in state_dict {
            let data = tensor.data()?;

            if key == "step_count" {
                if let Some(&count) = data.first() {
                    self.state.step_count = count as usize;
                }
            } else if let Some(param_id) = key.strip_suffix("_momentum") {
                self.state.state.momentum.insert(param_id.to_string(), data.clone());
            } else if let Some(param_id) = key.strip_suffix("_variance") {
                self.state.state.variance.insert(param_id.to_string(), data.clone());
            } else if let Some(param_id) = key.strip_suffix("_inf_norm") {
                if let Some(&inf_norm) = data.first() {
                    self.state.inf_norms.insert(param_id.to_string(), inf_norm);
                }
            } else if let Some(param_id) = key.strip_suffix("_gradient_variance") {
                if let Some(&variance) = data.first() {
                    self.state.gradient_variances.insert(param_id.to_string(), variance);
                }
            } else if let Some(param_id) = key.strip_suffix("_step_count") {
                if let Some(&step_count) = data.first() {
                    self.state.param_step_counts.insert(param_id.to_string(), step_count as usize);
                }
            }
        }

        Ok(())
    }

    fn memory_usage(&self) -> StateMemoryStats {
        StateMemoryStats {
            momentum_elements: self.state.state.momentum.values().map(|v| v.len()).sum::<usize>(),
            variance_elements: self.state.state.variance.values().map(|v| v.len()).sum::<usize>(),
            third_moment_elements: 0, // AdaMax+ doesn't use third moments
            total_bytes: self.state.memory_usage(),
            num_parameters: self.state.state.momentum.len(),
        }
    }

    fn reset_state(&mut self) {
        self.state.state.clear();
        self.state.step_count = 0;
        self.state.inf_norms.clear();
        self.state.gradient_variances.clear();
        self.state.param_step_counts.clear();
    }

    fn num_parameters(&self) -> usize {
        self.state.state.momentum.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::tensor::Tensor;

    #[test]
    fn test_adamax_plus_creation() {
        let optimizer = AdaMaxPlus::new(0.001, (0.9, 0.999), 1e-8, 0.01);
        assert_eq!(optimizer.get_lr(), 0.001);
        assert_eq!(optimizer.state.config.betas, (0.9, 0.999));
        assert_eq!(optimizer.state.config.epsilon, 1e-8);
        assert_eq!(optimizer.state.config.weight_decay, 0.01);
    }

    #[test]
    fn test_adamax_plus_config() {
        let config = AdaMaxPlusConfig::new()
            .learning_rate(0.002)
            .betas((0.95, 0.999))
            .enable_adaptive_momentum(true)
            .warmup_steps(1000);

        let optimizer = AdaMaxPlus::from_config(config);
        assert_eq!(optimizer.get_lr(), 0.002);
        assert_eq!(optimizer.state.config.betas, (0.95, 0.999));
        assert!(optimizer.state.config.adaptive_momentum);
        assert_eq!(optimizer.state.config.warmup_steps, 1000);
    }

    #[test]
    fn test_adamax_plus_presets() {
        let llm_optimizer = AdaMaxPlus::for_large_models();
        assert_eq!(llm_optimizer.get_lr(), 0.0002);
        assert_eq!(llm_optimizer.state.config.warmup_steps, 10000);
        assert!(llm_optimizer.state.config.adaptive_momentum);

        let fast_optimizer = AdaMaxPlus::for_fast_training();
        assert_eq!(fast_optimizer.get_lr(), 0.003);
        assert_eq!(
            fast_optimizer.state.config.momentum_adaptation_strength,
            0.2
        );

        let stable_optimizer = AdaMaxPlus::for_stable_training();
        assert!(!stable_optimizer.state.config.adaptive_momentum);
        assert!(!stable_optimizer.state.config.variance_tracking);
    }

    #[test]
    fn test_adamax_plus_step() -> Result<()> {
        let mut optimizer = AdaMaxPlus::new(0.01, (0.9, 0.999), 1e-8, 0.0);

        // Create test parameters and gradients directly
        let mut param = Tensor::ones(&[2, 2])?;
        let grad = Tensor::new(vec![0.1, 0.2, 0.3, 0.4])?;

        // Store original parameters
        let original_data = param.data()?.clone();

        // Perform optimization step
        optimizer.update(&mut param, &grad)?;

        // Check that parameters were updated
        let param_data = param.data()?;
        assert!(param_data.iter().zip(original_data.iter()).all(|(&new, &orig)| new != orig)); // Parameters should change

        Ok(())
    }

    #[test]
    fn test_warmup_learning_rate() {
        let mut optimizer =
            AdaMaxPlus::from_config(AdaMaxPlusConfig::new().learning_rate(0.001).warmup_steps(100));

        // At step 0, effective LR should be 0
        assert_eq!(optimizer.compute_effective_learning_rate(), 0.0);

        // At step 50, effective LR should be 50% of base LR
        optimizer.state.step_count = 50;
        assert!((optimizer.compute_effective_learning_rate() - 0.0005).abs() < 1e-9);

        // At step 100, effective LR should be 100% of base LR
        optimizer.state.step_count = 100;
        assert!((optimizer.compute_effective_learning_rate() - 0.001).abs() < 1e-9);

        // Beyond warmup, should remain at base LR
        optimizer.state.step_count = 200;
        assert!((optimizer.compute_effective_learning_rate() - 0.001).abs() < 1e-9);
    }

    #[test]
    fn test_adaptive_momentum() {
        let optimizer = AdaMaxPlus::from_config(
            AdaMaxPlusConfig::new()
                .enable_adaptive_momentum(true)
                .momentum_adaptation_strength(0.2),
        );

        // Test with low variance (should use higher momentum)
        let param_id = "test_param".to_string();

        // Simulate low variance by setting it in the optimizer's gradient_variances
        let mut test_optimizer = optimizer;
        test_optimizer.state.gradient_variances.insert(param_id.clone(), 0.1);

        let adaptive_beta1 = test_optimizer.compute_adaptive_momentum(param_id.clone());
        assert!(adaptive_beta1 > 0.85); // Should be close to base beta1 (0.9)

        // Test with high variance (should use lower momentum)
        test_optimizer.state.gradient_variances.insert(param_id.clone(), 0.8);

        let adaptive_beta1_high = test_optimizer.compute_adaptive_momentum(param_id);
        assert!(adaptive_beta1_high < adaptive_beta1); // Should be lower than low variance case
    }

    #[test]
    fn test_state_dict_save_load() -> Result<()> {
        let mut optimizer = AdaMaxPlus::new(0.001, (0.9, 0.999), 1e-8, 0.01);

        // Create and process some parameters directly
        let mut param = Tensor::ones(&[2])?;
        let grad = Tensor::new(vec![0.1, 0.2])?;
        optimizer.update(&mut param, &grad)?;

        // Save state
        let state_dict = optimizer.state_dict()?;
        assert!(!state_dict.is_empty());

        // Create new optimizer and load state
        let mut new_optimizer = AdaMaxPlus::new(0.002, (0.8, 0.99), 1e-7, 0.02);
        new_optimizer.load_state_dict(state_dict)?;

        // Check that state was loaded correctly (config doesn't change during load)
        assert_eq!(new_optimizer.get_lr(), 0.002); // Should keep new config
        assert_eq!(new_optimizer.state.config.betas, (0.8, 0.99));
        assert!(new_optimizer.state.step_count > 0);

        Ok(())
    }

    #[test]
    fn test_zero_grad() -> Result<()> {
        let mut optimizer = AdaMaxPlus::new(0.001, (0.9, 0.999), 1e-8, 0.0);

        // Test that zero_grad doesn't crash (implementation depends on framework)
        optimizer.zero_grad();

        // Since gradient tracking isn't implemented yet, we just ensure the method exists
        // and can be called without errors
        assert_eq!(optimizer.get_lr(), 0.001);

        Ok(())
    }

    #[test]
    fn test_memory_usage_tracking() {
        let optimizer = AdaMaxPlus::new(0.001, (0.9, 0.999), 1e-8, 0.0);
        let memory_usage = optimizer.memory_usage();
        assert_eq!(memory_usage.total_bytes, 0); // Should start at 0
    }

    #[test]
    fn test_lr_get_set() {
        let mut optimizer = AdaMaxPlus::new(0.001, (0.9, 0.999), 1e-8, 0.0);
        assert_eq!(optimizer.get_lr(), 0.001);

        optimizer.set_lr(0.002);
        assert_eq!(optimizer.get_lr(), 0.002);
    }
}
