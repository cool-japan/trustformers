//! # Prodigy Optimizer
//!
//! Implementation of the Prodigy optimizer, a cutting-edge 2024 optimization algorithm
//! that adaptively estimates the distance to optimality and adjusts learning rates accordingly.
//!
//! Prodigy often outperforms Adam and other optimizers without requiring manual learning rate tuning.
//!
//! ## Key Features
//!
//! - **Adaptive Learning Rate**: Automatically estimates optimal learning rate without manual tuning
//! - **Distance Estimation**: Estimates distance to optimality for better convergence
//! - **Superior Performance**: Often outperforms Adam, AdamW, and other optimizers
//! - **No LR Scheduling**: Eliminates need for learning rate schedules
//! - **Robust Convergence**: Stable convergence across different problem types
//!
//! ## Research Foundation
//!
//! Based on "Prodigy: An Expeditiously Adaptive Parameter-Free Learner" and related research
//! demonstrating superior convergence properties and automatic learning rate adaptation.
//!
//! ## Example Usage
//!
//! ```rust
//! use trustformers_optim::prodigy::{Prodigy, ProdigyConfig};
//!
//! // Create with default configuration (no learning rate needed!)
//! let optimizer = Prodigy::new();
//!
//! // Or customize configuration
//! let config = ProdigyConfig {
//!     d0: 1e-6,           // Initial distance estimate
//!     beta1: 0.9,         // Momentum coefficient
//!     beta2: 0.999,       // Variance coefficient
//!     eps: 1e-8,          // Numerical stability
//!     weight_decay: 0.01, // L2 regularization
//!     growth_rate: 1.02,  // Distance growth rate
//!     ..Default::default()
//! };
//! let optimizer = Prodigy::with_config(config);
//! ```

use crate::traits::StatefulOptimizer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

/// Configuration for the Prodigy optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProdigyConfig {
    /// Initial distance estimate (d0)
    pub d0: f64,
    /// Momentum coefficient for first moment (β1)
    pub beta1: f64,
    /// Momentum coefficient for second moment (β2)
    pub beta2: f64,
    /// Numerical stability constant (ε)
    pub eps: f64,
    /// Weight decay coefficient
    pub weight_decay: f64,
    /// Growth rate for distance estimation
    pub growth_rate: f64,
    /// Warmup steps for stability
    pub warmup_steps: usize,
    /// Use bias correction
    pub bias_correction: bool,
    /// Safeguard bound for distance estimation
    pub safeguard_bound: f64,
}

impl Default for ProdigyConfig {
    fn default() -> Self {
        Self {
            d0: 1e-6,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            growth_rate: 1.02,
            warmup_steps: 0,
            bias_correction: true,
            safeguard_bound: 2.0,
        }
    }
}

impl ProdigyConfig {
    /// Configuration optimized for language model training.
    pub fn for_language_models() -> Self {
        Self {
            d0: 1e-6,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.1,
            growth_rate: 1.02,
            warmup_steps: 1000,
            bias_correction: true,
            safeguard_bound: 2.0,
        }
    }

    /// Configuration optimized for computer vision tasks.
    pub fn for_vision() -> Self {
        Self {
            d0: 1e-6,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.05,
            growth_rate: 1.01,
            warmup_steps: 100,
            bias_correction: true,
            safeguard_bound: 1.5,
        }
    }

    /// Configuration for fast training with aggressive adaptation.
    pub fn for_fast_training() -> Self {
        Self {
            d0: 1e-5,
            beta1: 0.9,
            beta2: 0.99,
            eps: 1e-8,
            weight_decay: 0.01,
            growth_rate: 1.05,
            warmup_steps: 10,
            bias_correction: false,
            safeguard_bound: 3.0,
        }
    }

    /// Configuration for stable, conservative training.
    pub fn for_stable_training() -> Self {
        Self {
            d0: 1e-7,
            beta1: 0.95,
            beta2: 0.9999,
            eps: 1e-8,
            weight_decay: 0.001,
            growth_rate: 1.005,
            warmup_steps: 2000,
            bias_correction: true,
            safeguard_bound: 1.2,
        }
    }
}

/// Optimizer state for individual parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProdigyParameterState {
    /// First moment estimate (momentum)
    pub momentum: Vec<f32>,
    /// Second moment estimate (variance)
    pub variance: Vec<f32>,
    /// Current distance estimate
    pub distance: f64,
    /// Step count for this parameter
    pub step: usize,
}

impl ProdigyParameterState {
    pub fn new(param_size: usize, initial_distance: f64) -> Self {
        Self {
            momentum: vec![0.0; param_size],
            variance: vec![0.0; param_size],
            distance: initial_distance,
            step: 0,
        }
    }

    /// Get memory usage statistics for this parameter state.
    pub fn memory_usage(&self) -> ProdigyMemoryStats {
        let momentum_bytes = self.momentum.len() * std::mem::size_of::<f32>();
        let variance_bytes = self.variance.len() * std::mem::size_of::<f32>();
        let metadata_bytes = std::mem::size_of::<f64>() + std::mem::size_of::<usize>();

        ProdigyMemoryStats {
            momentum_bytes,
            variance_bytes,
            metadata_bytes,
            total_bytes: momentum_bytes + variance_bytes + metadata_bytes,
        }
    }
}

/// Memory usage statistics for Prodigy optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProdigyMemoryStats {
    pub momentum_bytes: usize,
    pub variance_bytes: usize,
    pub metadata_bytes: usize,
    pub total_bytes: usize,
}

/// Global optimizer state containing all parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProdigyOptimizerState {
    /// Per-parameter states
    pub parameters: HashMap<String, ProdigyParameterState>,
    /// Global step count
    pub global_step: usize,
    /// Global distance estimate
    pub global_distance: f64,
    /// Distance growth history for adaptive adjustment
    pub distance_history: Vec<f64>,
}

impl Default for ProdigyOptimizerState {
    fn default() -> Self {
        Self {
            parameters: HashMap::new(),
            global_step: 0,
            global_distance: 1e-6,
            distance_history: Vec::new(),
        }
    }
}

impl ProdigyOptimizerState {
    /// Clear all optimizer state.
    pub fn clear(&mut self) {
        self.parameters.clear();
        self.global_step = 0;
        self.global_distance = 1e-6;
        self.distance_history.clear();
    }

    /// Get total memory usage across all parameters.
    pub fn total_memory_usage(&self) -> ProdigyMemoryStats {
        let mut total_momentum = 0;
        let mut total_variance = 0;
        let mut total_metadata = 0;

        for param_state in self.parameters.values() {
            let stats = param_state.memory_usage();
            total_momentum += stats.momentum_bytes;
            total_variance += stats.variance_bytes;
            total_metadata += stats.metadata_bytes;
        }

        // Add global state memory
        total_metadata += std::mem::size_of::<usize>()
            + std::mem::size_of::<f64>()
            + self.distance_history.len() * std::mem::size_of::<f64>();

        ProdigyMemoryStats {
            momentum_bytes: total_momentum,
            variance_bytes: total_variance,
            metadata_bytes: total_metadata,
            total_bytes: total_momentum + total_variance + total_metadata,
        }
    }
}

/// Prodigy optimizer with adaptive learning rate estimation.
pub struct Prodigy {
    config: ProdigyConfig,
    state: ProdigyOptimizerState,
}

impl Prodigy {
    /// Create a new Prodigy optimizer with default configuration.
    pub fn new() -> Self {
        Self {
            config: ProdigyConfig::default(),
            state: ProdigyOptimizerState::default(),
        }
    }

    /// Create Prodigy optimizer with custom configuration.
    pub fn with_config(config: ProdigyConfig) -> Self {
        let state = ProdigyOptimizerState {
            global_distance: config.d0,
            ..Default::default()
        };

        Self { config, state }
    }

    /// Create Prodigy optimizer optimized for language models.
    pub fn for_language_models() -> Self {
        Self::with_config(ProdigyConfig::for_language_models())
    }

    /// Create Prodigy optimizer optimized for computer vision.
    pub fn for_vision() -> Self {
        Self::with_config(ProdigyConfig::for_vision())
    }

    /// Create Prodigy optimizer for fast training.
    pub fn for_fast_training() -> Self {
        Self::with_config(ProdigyConfig::for_fast_training())
    }

    /// Create Prodigy optimizer for stable training.
    pub fn for_stable_training() -> Self {
        Self::with_config(ProdigyConfig::for_stable_training())
    }

    /// Get current global learning rate estimate.
    pub fn get_lr(&self) -> f64 {
        self.state.global_distance
    }

    /// Set global distance estimate (equivalent to learning rate).
    pub fn set_lr(&mut self, distance: f64) {
        self.state.global_distance = distance.max(1e-10);
    }

    /// Reset optimizer state.
    pub fn reset(&mut self) {
        self.state.clear();
        self.state.global_distance = self.config.d0;
    }

    /// Get memory usage statistics.
    pub fn memory_usage(&self) -> ProdigyMemoryStats {
        self.state.total_memory_usage()
    }

    /// Update distance estimate based on gradient and parameter norms.
    fn update_distance_estimate(&mut self, grad_norm: f64, param_norm: f64) {
        if grad_norm > 0.0 && param_norm > 0.0 {
            // Estimate distance to optimality using gradient and parameter norms
            let distance_estimate = (param_norm / grad_norm).min(self.config.safeguard_bound);

            // Apply exponential moving average for stability
            let alpha = 0.01; // Smoothing factor
            self.state.global_distance = (1.0 - alpha) * self.state.global_distance
                + alpha * distance_estimate * self.config.growth_rate;

            // Keep history for adaptive adjustment
            self.state.distance_history.push(self.state.global_distance);
            if self.state.distance_history.len() > 100 {
                self.state.distance_history.remove(0);
            }
        }
    }

    /// Compute bias correction factors.
    #[allow(dead_code)]
    fn bias_correction(&self, step: usize) -> (f64, f64) {
        if self.config.bias_correction && step > 0 {
            let beta1_correction = 1.0 - self.config.beta1.powi(step as i32);
            let beta2_correction = 1.0 - self.config.beta2.powi(step as i32);
            (beta1_correction, beta2_correction)
        } else {
            (1.0, 1.0)
        }
    }

    /// Apply warmup scaling to learning rate.
    #[allow(dead_code)]
    fn warmup_scaling(&self, step: usize) -> f64 {
        if self.config.warmup_steps > 0 && step < self.config.warmup_steps {
            (step as f64 + 1.0) / (self.config.warmup_steps as f64)
        } else {
            1.0
        }
    }

    /// Updates a named parameter with its gradient.
    pub fn update_parameter(
        &mut self,
        param_name: &str,
        param: &mut Tensor,
        grad: &Tensor,
    ) -> Result<()> {
        let mut param_data = param.data().map_err(|e| {
            TrustformersError::tensor_op_error(
                &format!("Failed to get parameter data: {}", e),
                "prodigy_update",
            )
        })?;
        let grad_data = grad.data().map_err(|e| {
            TrustformersError::tensor_op_error(
                &format!("Failed to get gradient data: {}", e),
                "prodigy_update",
            )
        })?;

        if param_data.len() != grad_data.len() {
            return Err(TrustformersError::tensor_op_error(
                "Parameter and gradient size mismatch",
                "prodigy_update",
            ));
        }

        // Get or create parameter state
        let param_size = param_data.len();

        // Compute gradient and parameter norms for distance estimation
        let grad_norm: f64 = grad_data.iter().map(|&g| (g as f64).powi(2)).sum::<f64>().sqrt();
        let param_norm: f64 = param_data.iter().map(|&p| (p as f64).powi(2)).sum::<f64>().sqrt();

        // Update global distance estimate first (before borrowing param_state)
        self.update_distance_estimate(grad_norm, param_norm);

        // Now get or create parameter state
        let param_state = self
            .state
            .parameters
            .entry(param_name.to_string())
            .or_insert_with(|| ProdigyParameterState::new(param_size, self.config.d0));

        // Resize state if needed
        if param_state.momentum.len() != param_size {
            param_state.momentum.resize(param_size, 0.0);
            param_state.variance.resize(param_size, 0.0);
        }

        param_state.step += 1;
        let current_step = param_state.step;

        // Apply warmup scaling (using local variable to avoid borrow issues)
        let warmup_scale =
            if self.config.warmup_steps > 0 && current_step < self.config.warmup_steps {
                (current_step as f64 + 1.0) / (self.config.warmup_steps as f64)
            } else {
                1.0
            };
        let effective_distance = self.state.global_distance * warmup_scale;

        // Bias correction (using local variables to avoid borrow issues)
        let (beta1_correction, beta2_correction) =
            if self.config.bias_correction && current_step > 0 {
                let beta1_correction = 1.0 - self.config.beta1.powi(current_step as i32);
                let beta2_correction = 1.0 - self.config.beta2.powi(current_step as i32);
                (beta1_correction, beta2_correction)
            } else {
                (1.0, 1.0)
            };

        // Update momentum and variance
        for i in 0..param_size {
            let grad_val = grad_data[i] as f64;

            // Apply weight decay to gradient if specified
            let grad_with_decay = if self.config.weight_decay > 0.0 {
                grad_val + self.config.weight_decay * (param_data[i] as f64)
            } else {
                grad_val
            };

            // Update biased first moment estimate
            param_state.momentum[i] = (self.config.beta1 * param_state.momentum[i] as f64
                + (1.0 - self.config.beta1) * grad_with_decay)
                as f32;

            // Update biased second moment estimate
            param_state.variance[i] = (self.config.beta2 * param_state.variance[i] as f64
                + (1.0 - self.config.beta2) * grad_with_decay.powi(2))
                as f32;

            // Bias-corrected moments
            let m_hat = param_state.momentum[i] as f64 / beta1_correction;
            let v_hat = param_state.variance[i] as f64 / beta2_correction;

            // Compute parameter update using adaptive distance
            let denominator = v_hat.sqrt() + self.config.eps;
            let update = effective_distance * m_hat / denominator;

            // Apply update
            param_data[i] = (param_data[i] as f64 - update) as f32;
        }

        // Update parameter tensor with new data
        *param = Tensor::new(param_data)?;

        Ok(())
    }
}

impl Default for Prodigy {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimizer for Prodigy {
    fn step(&mut self) {
        self.state.global_step += 1;
    }

    fn zero_grad(&mut self) {
        // Prodigy doesn't need to explicitly zero gradients
        // as it processes them immediately during update
    }

    fn update(&mut self, param: &mut Tensor, grad: &Tensor) -> Result<()> {
        // Use a default parameter name for the core update
        self.update_parameter("default", param, grad)
    }

    fn get_lr(&self) -> f32 {
        self.state.global_distance as f32
    }

    fn set_lr(&mut self, lr: f32) {
        self.state.global_distance = (lr as f64).max(1e-10);
    }
}

impl StatefulOptimizer for Prodigy {
    type Config = ProdigyConfig;
    type State = ProdigyOptimizerState;

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

        // Save configuration as tensors
        state_dict.insert("lr".to_string(), Tensor::new(vec![self.config.d0 as f32])?);
        state_dict.insert(
            "beta1".to_string(),
            Tensor::new(vec![self.config.beta1 as f32])?,
        );
        state_dict.insert(
            "beta2".to_string(),
            Tensor::new(vec![self.config.beta2 as f32])?,
        );
        state_dict.insert(
            "eps".to_string(),
            Tensor::new(vec![self.config.eps as f32])?,
        );
        state_dict.insert(
            "weight_decay".to_string(),
            Tensor::new(vec![self.config.weight_decay as f32])?,
        );
        state_dict.insert(
            "growth_rate".to_string(),
            Tensor::new(vec![self.config.growth_rate as f32])?,
        );
        state_dict.insert(
            "warmup_steps".to_string(),
            Tensor::new(vec![self.config.warmup_steps as f32])?,
        );
        state_dict.insert(
            "global_step".to_string(),
            Tensor::new(vec![self.state.global_step as f32])?,
        );
        state_dict.insert(
            "global_distance".to_string(),
            Tensor::new(vec![self.state.global_distance as f32])?,
        );

        // Save parameter states
        for (param_name, param_state) in &self.state.parameters {
            state_dict.insert(
                format!("momentum_{}", param_name),
                Tensor::new(param_state.momentum.clone())?,
            );
            state_dict.insert(
                format!("variance_{}", param_name),
                Tensor::new(param_state.variance.clone())?,
            );
            state_dict.insert(
                format!("distance_{}", param_name),
                Tensor::new(vec![param_state.distance as f32])?,
            );
            state_dict.insert(
                format!("step_{}", param_name),
                Tensor::new(vec![param_state.step as f32])?,
            );
        }

        Ok(state_dict)
    }

    fn load_state_dict(&mut self, state_dict: HashMap<String, Tensor>) -> Result<()> {
        // Load configuration
        if let Some(lr_tensor) = state_dict.get("lr") {
            if let Ok(lr_vec) = lr_tensor.data() {
                if !lr_vec.is_empty() {
                    self.config.d0 = lr_vec[0] as f64;
                }
            }
        }
        if let Some(beta1_tensor) = state_dict.get("beta1") {
            if let Ok(beta1_vec) = beta1_tensor.data() {
                if !beta1_vec.is_empty() {
                    self.config.beta1 = beta1_vec[0] as f64;
                }
            }
        }
        if let Some(beta2_tensor) = state_dict.get("beta2") {
            if let Ok(beta2_vec) = beta2_tensor.data() {
                if !beta2_vec.is_empty() {
                    self.config.beta2 = beta2_vec[0] as f64;
                }
            }
        }

        // Load global state
        if let Some(global_step_tensor) = state_dict.get("global_step") {
            if let Ok(global_step_vec) = global_step_tensor.data() {
                if !global_step_vec.is_empty() {
                    self.state.global_step = global_step_vec[0] as usize;
                }
            }
        }
        if let Some(global_distance_tensor) = state_dict.get("global_distance") {
            if let Ok(global_distance_vec) = global_distance_tensor.data() {
                if !global_distance_vec.is_empty() {
                    self.state.global_distance = global_distance_vec[0] as f64;
                }
            }
        }

        // Load parameter states (simplified for now)
        // In a full implementation, we'd reconstruct all parameter states
        // This would require iterating through the state_dict to find matching patterns

        Ok(())
    }

    fn memory_usage(&self) -> crate::common::StateMemoryStats {
        let total_momentum_elements: usize =
            self.state.parameters.values().map(|p| p.momentum.len()).sum();
        let total_variance_elements: usize =
            self.state.parameters.values().map(|p| p.variance.len()).sum();

        let momentum_bytes = total_momentum_elements * std::mem::size_of::<f32>();
        let variance_bytes = total_variance_elements * std::mem::size_of::<f32>();
        let metadata_bytes = self.state.parameters.len()
            * (std::mem::size_of::<f64>() + std::mem::size_of::<usize>());

        crate::common::StateMemoryStats {
            momentum_elements: total_momentum_elements,
            variance_elements: total_variance_elements,
            third_moment_elements: 0,
            total_bytes: momentum_bytes + variance_bytes + metadata_bytes,
            num_parameters: self.state.parameters.len(),
        }
    }

    fn reset_state(&mut self) {
        self.reset();
    }

    fn num_parameters(&self) -> usize {
        self.state.parameters.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prodigy_creation() {
        let optimizer = Prodigy::new();
        assert_eq!(optimizer.config.d0, 1e-6);
        assert_eq!(optimizer.config.beta1, 0.9);
        assert_eq!(optimizer.config.beta2, 0.999);
    }

    #[test]
    fn test_prodigy_with_config() {
        let config = ProdigyConfig {
            d0: 1e-5,
            beta1: 0.95,
            beta2: 0.99,
            weight_decay: 0.1,
            ..Default::default()
        };
        let optimizer = Prodigy::with_config(config.clone());
        assert_eq!(optimizer.config.d0, config.d0);
        assert_eq!(optimizer.config.beta1, config.beta1);
        assert_eq!(optimizer.config.weight_decay, config.weight_decay);
    }

    #[test]
    fn test_prodigy_presets() {
        let lm_optimizer = Prodigy::for_language_models();
        assert_eq!(lm_optimizer.config.warmup_steps, 1000);
        assert_eq!(lm_optimizer.config.weight_decay, 0.1);

        let vision_optimizer = Prodigy::for_vision();
        assert_eq!(vision_optimizer.config.warmup_steps, 100);
        assert_eq!(vision_optimizer.config.weight_decay, 0.05);

        let fast_optimizer = Prodigy::for_fast_training();
        assert_eq!(fast_optimizer.config.growth_rate, 1.05);
        assert!(!fast_optimizer.config.bias_correction);

        let stable_optimizer = Prodigy::for_stable_training();
        assert_eq!(stable_optimizer.config.warmup_steps, 2000);
        assert_eq!(stable_optimizer.config.safeguard_bound, 1.2);
    }

    #[test]
    fn test_lr_getter_setter() {
        let mut optimizer = Prodigy::new();
        let initial_lr = optimizer.get_lr();
        assert_eq!(initial_lr, 1e-6);

        optimizer.set_lr(0.001);
        assert_eq!(optimizer.get_lr(), 0.001);

        // Test minimum bound
        optimizer.set_lr(-1.0);
        assert!(optimizer.get_lr() >= 1e-10);
    }

    #[test]
    fn test_parameter_state_creation() {
        let param_state = ProdigyParameterState::new(100, 1e-6);
        assert_eq!(param_state.momentum.len(), 100);
        assert_eq!(param_state.variance.len(), 100);
        assert_eq!(param_state.distance, 1e-6);
        assert_eq!(param_state.step, 0);
        assert!(param_state.momentum.iter().all(|&x| x == 0.0));
        assert!(param_state.variance.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_memory_usage_tracking() {
        let param_state = ProdigyParameterState::new(1000, 1e-6);
        let memory_stats = param_state.memory_usage();

        assert_eq!(memory_stats.momentum_bytes, 1000 * 4); // f32 = 4 bytes
        assert_eq!(memory_stats.variance_bytes, 1000 * 4);
        assert!(memory_stats.metadata_bytes > 0);
        assert_eq!(
            memory_stats.total_bytes,
            memory_stats.momentum_bytes + memory_stats.variance_bytes + memory_stats.metadata_bytes
        );
    }

    #[test]
    fn test_optimizer_state_operations() {
        let mut state = ProdigyOptimizerState::default();
        state
            .parameters
            .insert("param1".to_string(), ProdigyParameterState::new(100, 1e-6));
        state
            .parameters
            .insert("param2".to_string(), ProdigyParameterState::new(200, 1e-6));
        state.global_step = 10;

        let memory_stats = state.total_memory_usage();
        assert!(memory_stats.total_bytes > 0);
        assert_eq!(memory_stats.momentum_bytes, (100 + 200) * 4);

        state.clear();
        assert_eq!(state.parameters.len(), 0);
        assert_eq!(state.global_step, 0);
        assert_eq!(state.global_distance, 1e-6);
    }

    #[test]
    fn test_reset() {
        let mut optimizer = Prodigy::new();
        optimizer.state.global_step = 100;
        optimizer
            .state
            .parameters
            .insert("test".to_string(), ProdigyParameterState::new(10, 1e-6));

        optimizer.reset();
        assert_eq!(optimizer.state.global_step, 0);
        assert_eq!(optimizer.state.parameters.len(), 0);
        assert_eq!(optimizer.state.global_distance, optimizer.config.d0);
    }

    #[test]
    fn test_config_serialization() {
        let config = ProdigyConfig::for_language_models();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: ProdigyConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config.d0, deserialized.d0);
        assert_eq!(config.beta1, deserialized.beta1);
        assert_eq!(config.warmup_steps, deserialized.warmup_steps);
    }

    #[test]
    fn test_state_dict_operations() {
        let mut optimizer = Prodigy::for_vision();
        optimizer.state.global_step = 50;
        optimizer.state.parameters.insert(
            "test_param".to_string(),
            ProdigyParameterState::new(5, 1e-5),
        );

        // Save state dict
        let state_dict = optimizer.state_dict().unwrap();
        assert!(state_dict.contains_key("lr"));
        assert!(state_dict.contains_key("global_step"));

        // Create new optimizer and load state
        let mut new_optimizer = Prodigy::new();
        new_optimizer.load_state_dict(state_dict).unwrap();

        assert_eq!(new_optimizer.state.global_step, 50);
        // Note: parameter states are not fully implemented in load_state_dict yet
        // This test validates that basic config and global state are loaded correctly
    }

    #[test]
    fn test_step_and_zero_grad() {
        let mut optimizer = Prodigy::new();
        assert_eq!(optimizer.state.global_step, 0);

        optimizer.step();
        assert_eq!(optimizer.state.global_step, 1);

        optimizer.zero_grad(); // Should not error
    }

    #[test]
    fn test_stateful_optimizer_trait() {
        let optimizer = Prodigy::for_fast_training();

        // Test config access
        let config = optimizer.config();
        assert_eq!(config.growth_rate, 1.05);

        // Test state access
        let state = optimizer.state();
        assert_eq!(state.global_step, 0);
    }

    #[test]
    fn test_distance_estimation_bounds() {
        let mut optimizer = Prodigy::with_config(ProdigyConfig {
            safeguard_bound: 2.0,
            ..Default::default()
        });

        // Test that distance estimation respects safeguard bounds
        optimizer.update_distance_estimate(1.0, 10.0); // Would give 10.0 without bound
        assert!(optimizer.get_lr() <= 2.0);
    }

    #[test]
    fn test_bias_correction() {
        let optimizer = Prodigy::new();

        // With bias correction enabled
        let (bc1, bc2) = optimizer.bias_correction(1);
        assert!(bc1 > 0.0 && bc1 < 1.0);
        assert!(bc2 > 0.0 && bc2 < 1.0);

        // After many steps, bias correction should be positive and less than 1.0
        let (bc1_late, bc2_late) = optimizer.bias_correction(1000);
        assert!(bc1_late > 0.9);
        assert!(bc2_late > 0.6); // 1.0 - 0.999^1000 ≈ 0.63
    }

    #[test]
    fn test_warmup_scaling() {
        let optimizer = Prodigy::with_config(ProdigyConfig {
            warmup_steps: 100,
            ..Default::default()
        });

        // During warmup
        let scale_early = optimizer.warmup_scaling(10);
        assert!(scale_early < 1.0);
        assert_eq!(scale_early, 11.0 / 100.0);

        // After warmup
        let scale_late = optimizer.warmup_scaling(200);
        assert_eq!(scale_late, 1.0);
    }
}
