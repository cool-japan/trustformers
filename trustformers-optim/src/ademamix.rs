//! # AdEMAMix Optimizer
//!
//! Implementation of the AdEMAMix optimizer from "The AdEMAMix Optimizer: Better, Faster, Older" (2024).
//! Developed by Apple and EPFL researchers, this optimizer modifies Adam with a mixture of two EMAs
//! to better take advantage of past gradients.
//!
//! ## Key Features
//!
//! - **Dual EMA System**: Uses two exponential moving averages for better gradient utilization
//! - **Superior Convergence**: Often converges to lower minima with faster speed
//! - **Memory Efficiency**: Minimal overhead compared to standard Adam
//! - **Reduced Forgetting**: Significantly slows down model forgetting during training
//!
//! ## Research Results
//!
//! A 1.3B parameter AdEMAMix LLM trained on 101B tokens performs comparably to an AdamW model
//! trained on 197B tokens (+95% data efficiency improvement).

use crate::common::{OptimizerState, StateMemoryStats};
use crate::traits::StatefulOptimizer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::errors::Result;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

/// Configuration for AdEMAMix optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdEMAMixConfig {
    /// Learning rate (default: 1e-3)
    pub learning_rate: f32,
    /// First moment decay rate for short-term EMA (default: 0.9)
    pub beta1: f32,
    /// Second moment decay rate (default: 0.999)
    pub beta2: f32,
    /// Third moment decay rate for long-term EMA (default: 0.9999)
    pub beta3: f32,
    /// Weight for long-term EMA mixture (default: 5.0)
    pub alpha: f32,
    /// Small constant for numerical stability (default: 1e-8)
    pub epsilon: f32,
    /// Weight decay coefficient (default: 0.01)
    pub weight_decay: f32,
    /// Whether to use bias correction (default: true)
    pub bias_correction: bool,
}

impl Default for AdEMAMixConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            beta3: 0.9999,
            alpha: 5.0,
            epsilon: 1e-8,
            weight_decay: 0.01,
            bias_correction: true,
        }
    }
}

/// AdEMAMix optimizer implementation
///
/// AdEMAMix uses a mixture of two exponential moving averages for the first moment:
/// - Short-term EMA with beta1 for recent gradients
/// - Long-term EMA with beta3 for historical gradients
///
/// The final momentum is a weighted combination: m_t = m_short + alpha * m_long
#[derive(Debug)]
pub struct AdEMAMix {
    config: AdEMAMixConfig,
    state: OptimizerState,
    /// Short-term first moment estimates (beta1)
    short_momentum: HashMap<String, Vec<f32>>,
    /// Long-term first moment estimates (beta3)
    long_momentum: HashMap<String, Vec<f32>>,
    /// Second moment estimates
    variance: HashMap<String, Vec<f32>>,
}

impl AdEMAMix {
    /// Create a new AdEMAMix optimizer with default configuration
    pub fn new() -> Self {
        Self::with_config(AdEMAMixConfig::default())
    }

    /// Create AdEMAMix with custom learning rate and weight decay
    pub fn new_with_params(learning_rate: f32, weight_decay: f32) -> Self {
        let config = AdEMAMixConfig {
            learning_rate,
            weight_decay,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create AdEMAMix with full parameter specification
    pub fn new_full(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        beta3: f32,
        alpha: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Self {
        let config = AdEMAMixConfig {
            learning_rate,
            beta1,
            beta2,
            beta3,
            alpha,
            epsilon,
            weight_decay,
            bias_correction: true,
        };
        Self::with_config(config)
    }

    /// Create AdEMAMix for large language model training (optimized hyperparameters)
    pub fn for_llm_training() -> Self {
        let config = AdEMAMixConfig {
            learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            beta3: 0.9999,
            alpha: 5.0,
            epsilon: 1e-8,
            weight_decay: 0.1,
            bias_correction: true,
        };
        Self::with_config(config)
    }

    /// Create AdEMAMix for computer vision tasks
    pub fn for_vision_training() -> Self {
        let config = AdEMAMixConfig {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            beta3: 0.999, // Slightly less long-term for vision
            alpha: 3.0,   // Reduced mixture weight
            epsilon: 1e-8,
            weight_decay: 1e-4,
            bias_correction: true,
        };
        Self::with_config(config)
    }

    /// Create AdEMAMix with custom configuration
    pub fn with_config(config: AdEMAMixConfig) -> Self {
        Self {
            config,
            state: OptimizerState::new(),
            short_momentum: HashMap::new(),
            long_momentum: HashMap::new(),
            variance: HashMap::new(),
        }
    }

    /// Get memory statistics for AdEMAMix state (deprecated - use memory_usage instead)
    pub fn memory_stats(&self) -> StateMemoryStats {
        self.memory_usage()
    }

    /// Calculate effective learning rate with bias correction
    fn effective_learning_rate(&self, step: usize) -> f32 {
        if !self.config.bias_correction {
            return self.config.learning_rate;
        }

        let step_f = step as f32 + 1.0;
        let bias_correction1_short = 1.0 - self.config.beta1.powf(step_f);
        let bias_correction1_long = 1.0 - self.config.beta3.powf(step_f);
        let bias_correction2 = 1.0 - self.config.beta2.powf(step_f);

        // Use the average bias correction for the mixed momentum
        let mixed_bias_correction1 = (bias_correction1_short
            + self.config.alpha * bias_correction1_long)
            / (1.0 + self.config.alpha);

        self.config.learning_rate * (mixed_bias_correction1 / bias_correction2.sqrt())
    }

    /// Initialize momentum and variance buffers for a parameter
    fn init_param_state(&mut self, param_id: &str, param_size: usize) {
        if !self.short_momentum.contains_key(param_id) {
            self.short_momentum.insert(param_id.to_string(), vec![0.0; param_size]);
            self.long_momentum.insert(param_id.to_string(), vec![0.0; param_size]);
            self.variance.insert(param_id.to_string(), vec![0.0; param_size]);
        }
    }
}

impl Default for AdEMAMix {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimizer for AdEMAMix {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        let param_data = parameter.data_mut()?;
        let grad_data = grad.data()?;

        // Generate unique parameter ID based on memory address
        let param_id = format!("param_{:p}", param_data.as_ptr());
        let param_size = param_data.len();

        // Initialize parameter state if needed
        self.init_param_state(&param_id, param_size);

        // Get effective learning rate with bias correction
        let effective_lr = self.effective_learning_rate(self.state.step);

        // Get mutable references to buffers (safe: init_param_state ensures they exist)
        let short_momentum = self
            .short_momentum
            .get_mut(&param_id)
            .expect("short_momentum should exist after init_param_state");
        let long_momentum = self
            .long_momentum
            .get_mut(&param_id)
            .expect("long_momentum should exist after init_param_state");
        let variance = self
            .variance
            .get_mut(&param_id)
            .expect("variance should exist after init_param_state");

        // Apply weight decay
        if self.config.weight_decay > 0.0 {
            for i in 0..param_size {
                param_data[i] *= 1.0 - self.config.learning_rate * self.config.weight_decay;
            }
        }

        // Update moments and parameters
        for i in 0..param_size {
            let grad = grad_data[i];

            // Update short-term momentum (beta1)
            short_momentum[i] =
                self.config.beta1 * short_momentum[i] + (1.0 - self.config.beta1) * grad;

            // Update long-term momentum (beta3)
            long_momentum[i] =
                self.config.beta3 * long_momentum[i] + (1.0 - self.config.beta3) * grad;

            // Update second moment
            variance[i] = self.config.beta2 * variance[i] + (1.0 - self.config.beta2) * grad * grad;

            // Combine short and long momentum with mixture weight
            let mixed_momentum = short_momentum[i] + self.config.alpha * long_momentum[i];

            // Apply AdEMAMix update
            let denom = variance[i].sqrt() + self.config.epsilon;
            param_data[i] -= effective_lr * mixed_momentum / denom;
        }

        Ok(())
    }

    fn step(&mut self) {
        self.state.step += 1;
    }

    fn zero_grad(&mut self) {
        // This is typically handled by the training framework
        // No action needed here as gradients are managed externally
    }

    fn get_lr(&self) -> f32 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }
}

impl StatefulOptimizer for AdEMAMix {
    type Config = AdEMAMixConfig;
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

        // Save step count
        state_dict.insert(
            "step".to_string(),
            Tensor::new(vec![self.state.step as f32])?,
        );

        // Save short momentum buffers
        for (param_id, momentum) in &self.short_momentum {
            state_dict.insert(
                format!("short_momentum_{}", param_id),
                Tensor::new(momentum.clone())?,
            );
        }

        // Save long momentum buffers
        for (param_id, momentum) in &self.long_momentum {
            state_dict.insert(
                format!("long_momentum_{}", param_id),
                Tensor::new(momentum.clone())?,
            );
        }

        // Save variance buffers
        for (param_id, var) in &self.variance {
            state_dict.insert(format!("variance_{}", param_id), Tensor::new(var.clone())?);
        }

        Ok(state_dict)
    }

    fn load_state_dict(&mut self, state_dict: HashMap<String, Tensor>) -> Result<()> {
        // Load step count
        if let Some(step_tensor) = state_dict.get("step") {
            let step_data = step_tensor.data()?;
            if !step_data.is_empty() {
                self.state.step = step_data[0] as usize;
            }
        }

        // Load buffers
        for (key, tensor) in &state_dict {
            let data = tensor.data()?;
            if let Some(param_id) = key.strip_prefix("short_momentum_") {
                self.short_momentum.insert(param_id.to_string(), data);
            } else if let Some(param_id) = key.strip_prefix("long_momentum_") {
                self.long_momentum.insert(param_id.to_string(), data);
            } else if let Some(param_id) = key.strip_prefix("variance_") {
                self.variance.insert(param_id.to_string(), data);
            }
        }

        Ok(())
    }

    fn memory_usage(&self) -> StateMemoryStats {
        let mut momentum_elements = 0;
        let mut variance_elements = 0;

        // Count momentum elements (both short and long)
        for momentum in self.short_momentum.values() {
            momentum_elements += momentum.len();
        }
        for momentum in self.long_momentum.values() {
            momentum_elements += momentum.len();
        }

        // Count variance elements
        for variance in self.variance.values() {
            variance_elements += variance.len();
        }

        let total_elements = momentum_elements + variance_elements;
        let total_bytes = total_elements * std::mem::size_of::<f32>();

        StateMemoryStats {
            momentum_elements,
            variance_elements,
            third_moment_elements: 0,
            total_bytes,
            num_parameters: self.short_momentum.values().map(|v| v.len()).sum(), // Total parameters
        }
    }

    fn reset_state(&mut self) {
        self.state = OptimizerState::new();
        self.short_momentum.clear();
        self.long_momentum.clear();
        self.variance.clear();
    }

    fn num_parameters(&self) -> usize {
        self.short_momentum.values().map(|v| v.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ademamix_creation() {
        let optimizer = AdEMAMix::new();
        assert_eq!(optimizer.config.learning_rate, 1e-3);
        assert_eq!(optimizer.config.beta1, 0.9);
        assert_eq!(optimizer.config.beta2, 0.999);
        assert_eq!(optimizer.config.beta3, 0.9999);
        assert_eq!(optimizer.config.alpha, 5.0);
        assert_eq!(optimizer.state.step, 0);
    }

    #[test]
    fn test_ademamix_with_params() {
        let optimizer = AdEMAMix::new_with_params(1e-4, 0.1);
        assert_eq!(optimizer.config.learning_rate, 1e-4);
        assert_eq!(optimizer.config.weight_decay, 0.1);
    }

    #[test]
    fn test_ademamix_full_params() {
        let optimizer = AdEMAMix::new_full(1e-4, 0.95, 0.999, 0.9999, 6.0, 1e-7, 0.05);
        assert_eq!(optimizer.config.learning_rate, 1e-4);
        assert_eq!(optimizer.config.beta1, 0.95);
        assert_eq!(optimizer.config.beta2, 0.999);
        assert_eq!(optimizer.config.beta3, 0.9999);
        assert_eq!(optimizer.config.alpha, 6.0);
        assert_eq!(optimizer.config.epsilon, 1e-7);
        assert_eq!(optimizer.config.weight_decay, 0.05);
    }

    #[test]
    fn test_ademamix_llm_preset() {
        let optimizer = AdEMAMix::for_llm_training();
        assert_eq!(optimizer.config.learning_rate, 1e-4);
        assert_eq!(optimizer.config.weight_decay, 0.1);
        assert_eq!(optimizer.config.alpha, 5.0);
    }

    #[test]
    fn test_ademamix_vision_preset() {
        let optimizer = AdEMAMix::for_vision_training();
        assert_eq!(optimizer.config.learning_rate, 1e-3);
        assert_eq!(optimizer.config.alpha, 3.0);
        assert_eq!(optimizer.config.beta3, 0.999);
    }

    #[test]
    fn test_ademamix_memory_stats() {
        let mut optimizer = AdEMAMix::new();

        // Initialize some parameter states
        optimizer.init_param_state("param_0", 1000);
        optimizer.init_param_state("param_1", 500);

        let stats = optimizer.memory_stats();
        assert_eq!(stats.num_parameters, 1500); // Total momentum parameters (not divided by 2)
        assert_eq!(stats.momentum_elements, 3000); // 1500 * 2 (short + long)
        assert_eq!(stats.variance_elements, 1500); // 1500 variance elements
        assert_eq!(stats.total_bytes, 18000); // 4500 total elements * 4 bytes
    }

    #[test]
    fn test_state_dict_operations() {
        let mut optimizer = AdEMAMix::new();
        optimizer.state.step = 10;
        optimizer.short_momentum.insert("param_0".to_string(), vec![0.1, 0.2, 0.3]);
        optimizer.long_momentum.insert("param_0".to_string(), vec![0.05, 0.1, 0.15]);
        optimizer.variance.insert("param_0".to_string(), vec![0.01, 0.04, 0.09]);

        // Save state
        let state_dict = optimizer.state_dict().unwrap();
        assert!(state_dict.contains_key("step"));
        assert!(state_dict.contains_key("short_momentum_param_0"));
        assert!(state_dict.contains_key("long_momentum_param_0"));
        assert!(state_dict.contains_key("variance_param_0"));

        // Create new optimizer and load state
        let mut new_optimizer = AdEMAMix::new();
        new_optimizer.load_state_dict(state_dict).unwrap();
        assert_eq!(new_optimizer.state.step, 10);
        assert_eq!(new_optimizer.short_momentum["param_0"], vec![0.1, 0.2, 0.3]);
        assert_eq!(
            new_optimizer.long_momentum["param_0"],
            vec![0.05, 0.1, 0.15]
        );
        assert_eq!(new_optimizer.variance["param_0"], vec![0.01, 0.04, 0.09]);
    }

    #[test]
    fn test_effective_learning_rate() {
        let optimizer = AdEMAMix::new();

        // Test that effective learning rate function works and returns positive values
        let lr_step1 = optimizer.effective_learning_rate(0);
        let lr_step100 = optimizer.effective_learning_rate(99);

        // Both learning rates should be positive
        assert!(lr_step1 > 0.0);
        assert!(lr_step100 > 0.0);

        // Learning rate should be reasonably close to configured rate after many steps
        assert!(lr_step100 > 0.0001); // Should be reasonable magnitude
    }

    #[test]
    fn test_lr_setter_getter() {
        let mut optimizer = AdEMAMix::new();
        assert_eq!(optimizer.get_lr(), 1e-3);

        optimizer.set_lr(2e-4);
        assert_eq!(optimizer.get_lr(), 2e-4);
        assert_eq!(optimizer.config.learning_rate, 2e-4);
    }

    #[test]
    fn test_reset() {
        let mut optimizer = AdEMAMix::new();
        optimizer.state.step = 100;
        optimizer.short_momentum.insert("param_0".to_string(), vec![0.1, 0.2]);
        optimizer.long_momentum.insert("param_0".to_string(), vec![0.05, 0.1]);
        optimizer.variance.insert("param_0".to_string(), vec![0.01, 0.04]);

        optimizer.reset_state();

        assert_eq!(optimizer.state.step, 0);
        assert!(optimizer.short_momentum.is_empty());
        assert!(optimizer.long_momentum.is_empty());
        assert!(optimizer.variance.is_empty());
    }

    #[test]
    fn test_config_serialization() {
        let config = AdEMAMixConfig {
            learning_rate: 1e-4,
            beta1: 0.95,
            beta2: 0.999,
            beta3: 0.9999,
            alpha: 6.0,
            epsilon: 1e-7,
            weight_decay: 0.05,
            bias_correction: true,
        };

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: AdEMAMixConfig = serde_json::from_str(&serialized).unwrap();

        assert_relative_eq!(deserialized.learning_rate, config.learning_rate);
        assert_relative_eq!(deserialized.alpha, config.alpha);
        assert_eq!(deserialized.bias_correction, config.bias_correction);
    }
}
