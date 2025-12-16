//! # BGE-Adam Optimizer
//!
//! Implementation of BGE-Adam (BGE-Adam Optimization Algorithm Based on Entropy Weighting
//! and Adaptive Gradient Strategy) from 2024 research. This optimizer enhances the standard
//! Adam algorithm with entropy weighting and adaptive gradient strategies to improve
//! convergence speed, robustness, and adaptability across various training conditions.
//!
//! ## Key Features
//!
//! - **Entropy Weighting**: Uses information entropy to weight gradient components
//! - **Adaptive Gradient Strategy**: Dynamically adjusts gradient processing
//! - **Enhanced Robustness**: Better handling of diverse training scenarios
//! - **Improved Convergence**: Faster and more stable convergence than standard Adam
//!
//! ## Algorithm Description
//!
//! BGE-Adam extends Adam with:
//! 1. Entropy-based weighting of gradient components
//! 2. Adaptive gradient scaling based on historical information
//! 3. Dynamic adjustment of momentum and variance parameters
//!
//! The BGE-Adam update rule:
//! ```text
//! # Entropy calculation for gradient weighting
//! p_i = |g_i| / Σ|g_j|  (normalized gradient magnitudes)
//! H = -Σ(p_i * log(p_i + ε))  (entropy)
//! w_i = exp(-α * H * p_i)  (entropy weights)
//!
//! # Weighted gradients
//! g_weighted = w ⊙ g  (element-wise multiplication)
//!
//! # Adaptive parameters
//! β1_adaptive = β1 * (1 + γ * H)
//! β2_adaptive = β2 * (1 - δ * H)
//!
//! # Standard Adam updates with adaptive parameters
//! m_t = β1_adaptive * m_{t-1} + (1 - β1_adaptive) * g_weighted
//! v_t = β2_adaptive * v_{t-1} + (1 - β2_adaptive) * g_weighted²
//!
//! # Bias correction with adaptive parameters
//! m̂_t = m_t / (1 - β1_adaptive^t)
//! v̂_t = v_t / (1 - β2_adaptive^t)
//!
//! # Parameter update
//! θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
//! ```
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use trustformers_optim::{BGEAdam, BGEAdamConfig};
//! use trustformers_core::traits::Optimizer;
//!
//! // Create BGE-Adam optimizer with default settings
//! let mut optimizer = BGEAdam::new(
//!     1e-3,    // learning rate
//!     (0.9, 0.999), // (β1, β2)
//!     1e-8,    // epsilon
//!     0.01,    // weight decay
//!     0.1,     // entropy scaling factor (α)
//!     0.05,    // β1 adaptation factor (γ)
//!     0.05,    // β2 adaptation factor (δ)
//! );
//! ```

use crate::common::{OptimizerState, StateMemoryStats};
use crate::traits::StatefulOptimizer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::errors::Result;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

/// Configuration for BGE-Adam optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BGEAdamConfig {
    /// Learning rate (default: 1e-3)
    pub learning_rate: f32,
    /// Coefficient for computing first moment (default: 0.9)
    pub beta1: f32,
    /// Coefficient for computing second moment (default: 0.999)
    pub beta2: f32,
    /// Small constant for numerical stability (default: 1e-8)
    pub epsilon: f32,
    /// Weight decay coefficient (default: 0.01)
    pub weight_decay: f32,
    /// Entropy scaling factor for gradient weighting (default: 0.1)
    pub entropy_scaling: f32,
    /// Adaptation factor for β1 based on entropy (default: 0.05)
    pub beta1_adaptation: f32,
    /// Adaptation factor for β2 based on entropy (default: 0.05)
    pub beta2_adaptation: f32,
    /// Minimum entropy value for numerical stability (default: 1e-6)
    pub min_entropy: f32,
    /// Use bias correction (default: true)
    pub bias_correction: bool,
    /// Enable entropy weighting (default: true)
    pub entropy_weighting: bool,
    /// Enable adaptive parameters (default: true)
    pub adaptive_parameters: bool,
}

impl Default for BGEAdamConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            entropy_scaling: 0.1,
            beta1_adaptation: 0.05,
            beta2_adaptation: 0.05,
            min_entropy: 1e-6,
            bias_correction: true,
            entropy_weighting: true,
            adaptive_parameters: true,
        }
    }
}

/// BGE-Adam optimizer implementation with entropy weighting and adaptive gradient strategy
pub struct BGEAdam {
    config: BGEAdamConfig,
    state: OptimizerState,
    step_count: usize,
    entropy_history: Vec<f32>,
    max_entropy_history: usize,
}

impl BGEAdam {
    /// Create a new BGE-Adam optimizer with specified parameters
    pub fn new(
        learning_rate: f32,
        betas: (f32, f32),
        epsilon: f32,
        weight_decay: f32,
        entropy_scaling: f32,
        beta1_adaptation: f32,
        beta2_adaptation: f32,
    ) -> Self {
        let config = BGEAdamConfig {
            learning_rate,
            beta1: betas.0,
            beta2: betas.1,
            epsilon,
            weight_decay,
            entropy_scaling,
            beta1_adaptation,
            beta2_adaptation,
            ..Default::default()
        };

        Self {
            config,
            state: OptimizerState::new(),
            step_count: 0,
            entropy_history: Vec::new(),
            max_entropy_history: 100, // Keep last 100 entropy values
        }
    }

    /// Create BGE-Adam with configuration optimized for large language models
    pub fn for_large_models() -> Self {
        Self::new(
            3e-4,        // Lower learning rate for stability
            (0.9, 0.95), // Lower β2 for better long-range dependencies
            1e-8,
            0.1,  // Higher weight decay
            0.15, // Increased entropy scaling
            0.08, // Higher adaptation factors
            0.08,
        )
    }

    /// Create BGE-Adam with configuration optimized for computer vision tasks
    pub fn for_vision() -> Self {
        Self::new(
            1e-3,         // Standard learning rate
            (0.9, 0.999), // Standard beta values
            1e-8,
            0.05, // Moderate weight decay
            0.1,  // Standard entropy scaling
            0.05, // Standard adaptation factors
            0.05,
        )
    }

    /// Create BGE-Adam with enhanced robustness settings
    pub fn for_robust_training() -> Self {
        Self::new(
            5e-4,          // Conservative learning rate
            (0.95, 0.999), // Higher β1 for more smoothing
            1e-6,          // Smaller epsilon for precision
            0.02,          // Moderate weight decay
            0.2,           // Higher entropy scaling for more adaptation
            0.1,           // Higher adaptation factors
            0.1,
        )
    }

    /// Calculate entropy of gradient components for weighting
    fn calculate_gradient_entropy(&self, gradients: &Tensor) -> Result<f32> {
        let grad_data = gradients.data()?;

        // Calculate absolute gradient magnitudes
        let abs_grads: Vec<f32> = grad_data.iter().map(|&g| g.abs()).collect();
        let sum_abs_grads: f32 = abs_grads.iter().sum();

        if sum_abs_grads < self.config.epsilon {
            return Ok(self.config.min_entropy);
        }

        // Calculate normalized probabilities
        let probabilities: Vec<f32> =
            abs_grads.iter().map(|&abs_g| abs_g / sum_abs_grads).collect();

        // Calculate entropy: H = -Σ(p_i * log(p_i + ε))
        let entropy =
            probabilities
                .iter()
                .map(|&p| {
                    if p > self.config.epsilon {
                        -p * (p + self.config.epsilon).ln()
                    } else {
                        0.0
                    }
                })
                .sum::<f32>();

        Ok(entropy.max(self.config.min_entropy))
    }

    /// Apply entropy weighting to gradients
    fn apply_entropy_weighting(&self, gradients: &Tensor, entropy: f32) -> Result<Tensor> {
        if !self.config.entropy_weighting {
            return Ok(gradients.clone());
        }

        let grad_data = gradients.data()?;
        let sum_abs_grads: f32 = grad_data.iter().map(|&g| g.abs()).sum();

        if sum_abs_grads < self.config.epsilon {
            return Ok(gradients.clone());
        }

        // Calculate entropy weights: w_i = exp(-α * H * p_i)
        let weighted_data: Vec<f32> = grad_data
            .iter()
            .map(|&g| {
                let p_i = g.abs() / sum_abs_grads;
                let weight = (-self.config.entropy_scaling * entropy * p_i).exp();
                g * weight
            })
            .collect();

        Tensor::new(weighted_data)
    }

    /// Get adaptive beta parameters based on entropy
    fn get_adaptive_betas(&self, entropy: f32) -> (f32, f32) {
        if !self.config.adaptive_parameters {
            return (self.config.beta1, self.config.beta2);
        }

        let beta1_adaptive = self.config.beta1 * (1.0 + self.config.beta1_adaptation * entropy);
        let beta2_adaptive = self.config.beta2 * (1.0 - self.config.beta2_adaptation * entropy);

        // Clamp to reasonable ranges
        let beta1_adaptive = beta1_adaptive.clamp(0.1, 0.99);
        let beta2_adaptive = beta2_adaptive.clamp(0.9, 0.9999);

        (beta1_adaptive, beta2_adaptive)
    }

    /// Update entropy history for adaptive behavior
    fn update_entropy_history(&mut self, entropy: f32) {
        self.entropy_history.push(entropy);

        // Keep only recent history
        if self.entropy_history.len() > self.max_entropy_history {
            self.entropy_history.remove(0);
        }
    }

    /// Get average entropy from recent history
    pub fn get_average_entropy(&self) -> f32 {
        if self.entropy_history.is_empty() {
            0.0
        } else {
            self.entropy_history.iter().sum::<f32>() / self.entropy_history.len() as f32
        }
    }

    /// Get entropy statistics for monitoring
    pub fn get_entropy_stats(&self) -> (f32, f32, f32) {
        if self.entropy_history.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let min_entropy = self.entropy_history.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_entropy = self.entropy_history.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let avg_entropy = self.get_average_entropy();

        (min_entropy, max_entropy, avg_entropy)
    }
}

impl Optimizer for BGEAdam {
    fn zero_grad(&mut self) {
        // Clear gradients - implementation specific to the framework
    }

    fn update(&mut self, parameter: &mut Tensor, gradient: &Tensor) -> Result<()> {
        let param_data = parameter.data()?;
        let param_id = format!("{:p}", param_data.as_ptr());
        self.step_count += 1;

        // Calculate gradient entropy
        let entropy = self.calculate_gradient_entropy(gradient)?;
        self.update_entropy_history(entropy);

        // Apply entropy weighting to gradients
        let weighted_gradients = self.apply_entropy_weighting(gradient, entropy)?;

        // Get adaptive beta parameters
        let (beta1_adaptive, beta2_adaptive) = self.get_adaptive_betas(entropy);

        // Get parameter size for initialization
        let param_size = parameter.data()?.len();

        // Get or initialize momentum and variance using OptimizerState methods
        let momentum_data = {
            let momentum_buffer = self.state.get_or_create_momentum(param_id.clone(), param_size);
            momentum_buffer.clone()
        };
        let variance_data = {
            let variance_buffer = self.state.get_or_create_variance(param_id.clone(), param_size);
            variance_buffer.clone()
        };

        let momentum = Tensor::new(momentum_data)?;
        let variance = Tensor::new(variance_data)?;

        // Update momentum: m_t = β1_adaptive * m_{t-1} + (1 - β1_adaptive) * g_weighted
        let momentum_data = momentum.data()?;
        let weighted_grad_data = weighted_gradients.data()?;
        let new_momentum_data: Vec<f32> = momentum_data
            .iter()
            .zip(weighted_grad_data.iter())
            .map(|(&m, &g)| beta1_adaptive * m + (1.0 - beta1_adaptive) * g)
            .collect();
        let new_momentum = Tensor::new(new_momentum_data)?;

        // Update variance: v_t = β2_adaptive * v_{t-1} + (1 - β2_adaptive) * g_weighted²
        let variance_data = variance.data()?;
        let new_variance_data: Vec<f32> = variance_data
            .iter()
            .zip(weighted_grad_data.iter())
            .map(|(&v, &g)| beta2_adaptive * v + (1.0 - beta2_adaptive) * g * g)
            .collect();
        let new_variance = Tensor::new(new_variance_data)?;

        // Store updated states back to the optimizer state
        let new_momentum_data = new_momentum.data()?;
        let new_variance_data = new_variance.data()?;

        self.state.momentum.insert(param_id.clone(), new_momentum_data.clone());
        self.state.variance.insert(param_id.clone(), new_variance_data.clone());

        // Apply bias correction if enabled
        let (corrected_momentum, corrected_variance) = if self.config.bias_correction {
            let step_f32 = self.step_count as f32;
            let momentum_correction = 1.0 - beta1_adaptive.powf(step_f32);
            let variance_correction = 1.0 - beta2_adaptive.powf(step_f32);

            let momentum_data = new_momentum.data()?;
            let variance_data = new_variance.data()?;

            let corrected_momentum_data: Vec<f32> =
                momentum_data.iter().map(|&m| m / momentum_correction).collect();
            let corrected_variance_data: Vec<f32> =
                variance_data.iter().map(|&v| v / variance_correction).collect();

            (
                Tensor::new(corrected_momentum_data)?,
                Tensor::new(corrected_variance_data)?,
            )
        } else {
            (new_momentum, new_variance)
        };

        // Calculate parameter update
        let param_data = parameter.data()?;
        let corrected_momentum_data = corrected_momentum.data()?;
        let corrected_variance_data = corrected_variance.data()?;

        let updated_params: Vec<f32> = param_data
            .iter()
            .zip(corrected_momentum_data.iter())
            .zip(corrected_variance_data.iter())
            .map(|((&p, &m), &v)| {
                let update = m / (v.sqrt() + self.config.epsilon);
                let weight_decay_term = self.config.weight_decay * p;
                p - self.config.learning_rate * (update + weight_decay_term)
            })
            .collect();

        // Update parameter tensor
        *parameter = Tensor::new(updated_params)?;

        Ok(())
    }

    fn step(&mut self) {
        // Step is handled in update() method
        self.state.step();
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }

    fn get_lr(&self) -> f32 {
        self.config.learning_rate
    }
}

impl StatefulOptimizer for BGEAdam {
    type Config = BGEAdamConfig;
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

        // Convert momentum buffers
        for (key, buffer) in &self.state.momentum {
            let tensor = Tensor::new(buffer.clone())?;
            state_dict.insert(format!("{}_momentum", key), tensor);
        }

        // Convert variance buffers
        for (key, buffer) in &self.state.variance {
            let tensor = Tensor::new(buffer.clone())?;
            state_dict.insert(format!("{}_variance", key), tensor);
        }

        // Add entropy history
        let entropy_tensor = Tensor::new(self.entropy_history.clone())?;
        state_dict.insert("entropy_history".to_string(), entropy_tensor);

        let step_tensor = Tensor::new(vec![self.step_count as f32])?;
        state_dict.insert("step_count".to_string(), step_tensor);

        Ok(state_dict)
    }

    fn load_state_dict(&mut self, state_dict: HashMap<String, Tensor>) -> Result<()> {
        for (key, tensor) in state_dict {
            let data = tensor.data()?;

            if key == "entropy_history" {
                self.entropy_history = data.clone();
            } else if key == "step_count" {
                if let Some(&count) = data.first() {
                    self.step_count = count as usize;
                }
            } else if key.ends_with("_momentum") {
                let param_key = key.strip_suffix("_momentum").unwrap().to_string();
                self.state.momentum.insert(param_key, data.clone());
            } else if key.ends_with("_variance") {
                let param_key = key.strip_suffix("_variance").unwrap().to_string();
                self.state.variance.insert(param_key, data.clone());
            }
        }
        Ok(())
    }

    fn memory_usage(&self) -> StateMemoryStats {
        let momentum_size: usize = self.state.momentum.values().map(|v| v.len()).sum();
        let variance_size: usize = self.state.variance.values().map(|v| v.len()).sum();
        let entropy_size = self.entropy_history.len();

        let total_bytes =
            (momentum_size + variance_size + entropy_size) * std::mem::size_of::<f32>();

        StateMemoryStats {
            momentum_elements: momentum_size,
            variance_elements: variance_size,
            third_moment_elements: 0,
            total_bytes,
            num_parameters: self.state.momentum.len().max(self.state.variance.len()),
        }
    }

    fn reset_state(&mut self) {
        self.state.clear();
        self.step_count = 0;
        self.entropy_history.clear();
    }

    fn num_parameters(&self) -> usize {
        self.state.momentum.len().max(self.state.variance.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bge_adam_creation() {
        let optimizer = BGEAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01, 0.1, 0.05, 0.05);
        assert_eq!(optimizer.config.learning_rate, 1e-3);
        assert_eq!(optimizer.config.beta1, 0.9);
        assert_eq!(optimizer.config.beta2, 0.999);
        assert_eq!(optimizer.step_count, 0);
    }

    #[test]
    fn test_bge_adam_presets() {
        let llm_optimizer = BGEAdam::for_large_models();
        assert_eq!(llm_optimizer.config.learning_rate, 3e-4);
        assert_eq!(llm_optimizer.config.beta2, 0.95);

        let vision_optimizer = BGEAdam::for_vision();
        assert_eq!(vision_optimizer.config.learning_rate, 1e-3);
        assert_eq!(vision_optimizer.config.beta2, 0.999);

        let robust_optimizer = BGEAdam::for_robust_training();
        assert_eq!(robust_optimizer.config.learning_rate, 5e-4);
        assert_eq!(robust_optimizer.config.beta1, 0.95);
    }

    #[test]
    fn test_entropy_calculation() -> Result<()> {
        let optimizer = BGEAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01, 0.1, 0.05, 0.05);
        let gradients = Tensor::new(vec![1.0, 2.0, 1.0, 0.5])?;
        let entropy = optimizer.calculate_gradient_entropy(&gradients)?;
        assert!(entropy > 0.0);
        Ok(())
    }

    #[test]
    fn test_entropy_weighting() -> Result<()> {
        let optimizer = BGEAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01, 0.1, 0.05, 0.05);
        let gradients = Tensor::new(vec![1.0, 2.0, 1.0, 0.5])?;
        let entropy = 1.0;
        let weighted_gradients = optimizer.apply_entropy_weighting(&gradients, entropy)?;

        // Weighted gradients should be different from original
        let orig_data = gradients.data()?;
        let weighted_data = weighted_gradients.data()?;
        assert_ne!(orig_data, weighted_data);
        Ok(())
    }

    #[test]
    fn test_adaptive_betas() {
        let optimizer = BGEAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01, 0.1, 0.05, 0.05);
        let entropy = 1.0;
        let (beta1_adaptive, beta2_adaptive) = optimizer.get_adaptive_betas(entropy);

        // Beta1 should increase with entropy, beta2 should decrease
        assert!(beta1_adaptive > 0.9);
        assert!(beta2_adaptive < 0.999);
        assert!(beta1_adaptive < 0.99);
        assert!(beta2_adaptive > 0.9);
    }

    #[test]
    fn test_entropy_history() {
        let mut optimizer = BGEAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01, 0.1, 0.05, 0.05);

        optimizer.update_entropy_history(1.0);
        optimizer.update_entropy_history(1.5);
        optimizer.update_entropy_history(0.8);

        assert_eq!(optimizer.entropy_history.len(), 3);
        assert_relative_eq!(optimizer.get_average_entropy(), 1.1, epsilon = 1e-6);

        let (min_entropy, max_entropy, avg_entropy) = optimizer.get_entropy_stats();
        assert_relative_eq!(min_entropy, 0.8, epsilon = 1e-6);
        assert_relative_eq!(max_entropy, 1.5, epsilon = 1e-6);
        assert_relative_eq!(avg_entropy, 1.1, epsilon = 1e-6);
    }

    #[test]
    fn test_lr_setter_getter() {
        let mut optimizer = BGEAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01, 0.1, 0.05, 0.05);
        assert_eq!(optimizer.get_lr(), 1e-3);

        optimizer.set_lr(2e-3);
        assert_eq!(optimizer.get_lr(), 2e-3);
    }

    #[test]
    fn test_state_dict_operations() -> Result<()> {
        let mut optimizer = BGEAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01, 0.1, 0.05, 0.05);

        // Add some entropy history and step count
        optimizer.update_entropy_history(1.0);
        optimizer.update_entropy_history(1.5);
        optimizer.step_count = 10;

        let state_dict = optimizer.state_dict()?;
        assert!(state_dict.contains_key("entropy_history"));
        assert!(state_dict.contains_key("step_count"));

        let mut new_optimizer = BGEAdam::new(2e-3, (0.8, 0.99), 1e-7, 0.02, 0.2, 0.1, 0.1);
        new_optimizer.load_state_dict(state_dict)?;

        assert_eq!(new_optimizer.entropy_history.len(), 2);
        assert_eq!(new_optimizer.step_count, 10);
        assert_relative_eq!(new_optimizer.get_average_entropy(), 1.25, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_reset() {
        let mut optimizer = BGEAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01, 0.1, 0.05, 0.05);

        optimizer.update_entropy_history(1.0);
        optimizer.step_count = 5;

        optimizer.reset_state();

        assert_eq!(optimizer.step_count, 0);
        assert_eq!(optimizer.entropy_history.len(), 0);
        assert_eq!(optimizer.state.momentum.len(), 0);
        assert_eq!(optimizer.state.variance.len(), 0);
    }

    #[test]
    fn test_memory_usage() {
        let optimizer = BGEAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01, 0.1, 0.05, 0.05);
        let stats = optimizer.memory_usage();

        assert_eq!(stats.total_bytes, 0); // No parameters yet
        assert_eq!(stats.num_parameters, 0);
        assert_eq!(stats.momentum_elements, 0);
        assert_eq!(stats.variance_elements, 0);
    }

    #[test]
    fn test_config_access() {
        let optimizer = BGEAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01, 0.1, 0.05, 0.05);
        let config = optimizer.config();

        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.beta1, 0.9);
        assert_eq!(config.beta2, 0.999);
        assert_eq!(config.entropy_scaling, 0.1);
        assert_eq!(config.beta1_adaptation, 0.05);
        assert_eq!(config.beta2_adaptation, 0.05);
    }

    #[test]
    fn test_disabled_features() {
        let mut config = BGEAdamConfig::default();
        config.entropy_weighting = false;
        config.adaptive_parameters = false;

        let optimizer = BGEAdam {
            config,
            state: OptimizerState::new(),
            step_count: 0,
            entropy_history: Vec::new(),
            max_entropy_history: 100,
        };

        // Test that entropy weighting is disabled
        let gradients = Tensor::new(vec![1.0, 2.0, 1.0, 0.5]).unwrap();
        let weighted = optimizer.apply_entropy_weighting(&gradients, 1.0).unwrap();
        let grad_data = gradients.data().unwrap();
        let weighted_data = weighted.data().unwrap();
        assert_eq!(grad_data, weighted_data); // Should be unchanged

        // Test that adaptive parameters are disabled
        let (beta1, beta2) = optimizer.get_adaptive_betas(1.0);
        assert_eq!(beta1, optimizer.config.beta1);
        assert_eq!(beta2, optimizer.config.beta2);
    }
}
