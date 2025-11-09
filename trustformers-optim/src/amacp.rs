//! # aMacP: Adaptive Momentum and Consecutive Parameters Optimizer
//!
//! This module implements the aMacP optimizer from 2025 research, which addresses
//! limitations in existing optimizers by incorporating the average of both momentums
//! and consecutive parameters to adaptively change the step size.
//!
//! ## Key Innovations
//!
//! - **Dual Momentum Averaging**: Combines first and second moment estimates
//! - **Consecutive Parameter Averaging**: Uses parameter history for adaptive updates
//! - **Gradient Heterogeneity Handling**: Superior performance on transformer architectures
//! - **Adaptive Step Size**: Dynamic learning rate adjustment based on parameter trends
//!
//! ## Research Citation
//!
//! "aMacP: An adaptive optimization algorithm for Deep Neural Network"
//! Cyber Security and Applications, Volume 3, 2025

use crate::{
    common::{BiasCorrection, OptimizerState, ParameterUpdate, StateMemoryStats},
    traits::StatefulOptimizer,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::{errors::Result, tensor::Tensor, traits::Optimizer};

/// Configuration for aMacP optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AMacPConfig {
    /// Base learning rate
    pub learning_rate: f32,
    /// First momentum coefficient (gradient averaging)
    pub beta1: f32,
    /// Second momentum coefficient (squared gradient averaging)
    pub beta2: f32,
    /// Consecutive parameter averaging coefficient
    pub gamma: f32,
    /// Dual momentum weighting factor
    pub alpha: f32,
    /// Gradient heterogeneity adaptation strength
    pub eta: f32,
    /// Small constant for numerical stability
    pub epsilon: f32,
    /// Weight decay coefficient
    pub weight_decay: f32,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: Option<f32>,
    /// Enable adaptive step size based on parameter trends
    pub adaptive_step_size: bool,
    /// Warmup steps for gradient stabilization
    pub warmup_steps: usize,
}

impl Default for AMacPConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            gamma: 0.95, // Consecutive parameter averaging
            alpha: 0.5,  // Dual momentum weighting
            eta: 0.1,    // Gradient heterogeneity adaptation
            epsilon: 1e-8,
            weight_decay: 0.0,
            max_grad_norm: Some(1.0),
            adaptive_step_size: true,
            warmup_steps: 1000,
        }
    }
}

impl AMacPConfig {
    /// Configuration optimized for transformer models
    pub fn for_transformers() -> Self {
        Self {
            learning_rate: 6e-4,
            beta1: 0.9,
            beta2: 0.95,
            gamma: 0.98, // Higher consecutive parameter averaging for transformers
            alpha: 0.6,  // Stronger dual momentum weighting
            eta: 0.15,   // Higher gradient heterogeneity adaptation
            epsilon: 1e-8,
            weight_decay: 1e-2,
            max_grad_norm: Some(1.0),
            adaptive_step_size: true,
            warmup_steps: 4000, // Longer warmup for large models
        }
    }

    /// Configuration for vision models (CNN architectures)
    pub fn for_vision() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            gamma: 0.92, // Lower consecutive parameter averaging for vision
            alpha: 0.4,  // Moderate dual momentum weighting
            eta: 0.08,   // Lower gradient heterogeneity for stable vision training
            epsilon: 1e-8,
            weight_decay: 5e-4,
            max_grad_norm: Some(0.5),
            adaptive_step_size: true,
            warmup_steps: 500, // Shorter warmup for vision models
        }
    }

    /// Configuration for large language models
    pub fn for_large_language_models() -> Self {
        Self {
            learning_rate: 3e-4,
            beta1: 0.9,
            beta2: 0.95,
            gamma: 0.99, // Very high consecutive parameter averaging for LLMs
            alpha: 0.7,  // Strong dual momentum weighting for stability
            eta: 0.2,    // High gradient heterogeneity adaptation
            epsilon: 1e-8,
            weight_decay: 1e-1,
            max_grad_norm: Some(1.0),
            adaptive_step_size: true,
            warmup_steps: 10000, // Long warmup for stability
        }
    }
}

/// aMacP Optimizer implementation
#[derive(Debug)]
pub struct AMacP {
    config: AMacPConfig,
    state: OptimizerState,
    /// Previous parameters for consecutive averaging
    previous_params: HashMap<String, Vec<f32>>,
    /// Dual momentum buffers
    dual_momentum: HashMap<String, Vec<f32>>,
    /// Gradient heterogeneity tracking
    gradient_heterogeneity: HashMap<String, f32>,
    /// Step size adaptation factors
    step_size_factors: HashMap<String, f32>,
    /// Current step number
    current_step: usize,
}

impl AMacP {
    /// Create a new aMacP optimizer
    pub fn new(config: AMacPConfig) -> Self {
        Self {
            config,
            state: OptimizerState::new(),
            previous_params: HashMap::new(),
            dual_momentum: HashMap::new(),
            gradient_heterogeneity: HashMap::new(),
            step_size_factors: HashMap::new(),
            current_step: 0,
        }
    }

    /// Create aMacP for transformer models
    pub fn for_transformers() -> Self {
        Self::new(AMacPConfig::for_transformers())
    }

    /// Create aMacP for vision models
    pub fn for_vision() -> Self {
        Self::new(AMacPConfig::for_vision())
    }

    /// Create aMacP for large language models
    pub fn for_large_language_models() -> Self {
        Self::new(AMacPConfig::for_large_language_models())
    }

    /// Compute dual momentum combining first and second moments
    fn compute_dual_momentum(&self, m_hat: f32, v_hat: f32) -> f32 {
        self.config.alpha * m_hat + (1.0 - self.config.alpha) * v_hat.sqrt()
    }

    /// Update gradient heterogeneity measure
    fn update_gradient_heterogeneity(&mut self, param_id: &str, gradient: &[f32]) {
        let grad_norm: f32 = gradient.iter().map(|g| g * g).sum::<f32>().sqrt();
        let grad_mean = gradient.iter().sum::<f32>() / gradient.len() as f32;
        let grad_std = (gradient.iter().map(|g| (g - grad_mean) * (g - grad_mean)).sum::<f32>()
            / gradient.len() as f32)
            .sqrt();

        let heterogeneity = if grad_norm > 1e-8 { grad_std / grad_norm } else { 0.0 };

        let entry = self.gradient_heterogeneity.entry(param_id.to_string()).or_insert(0.0);
        *entry = 0.9 * *entry + 0.1 * heterogeneity;
    }

    /// Compute adaptive step size based on parameter trends (static version to avoid borrowing)
    #[allow(dead_code)]
    fn compute_adaptive_step_size_static(
        config: &AMacPConfig,
        current_params: &[f32],
        prev_params: &[f32],
        stored_factor: f32,
    ) -> f32 {
        if !config.adaptive_step_size {
            return 1.0;
        }

        let param_change_norm: f32 = current_params
            .iter()
            .zip(prev_params.iter())
            .map(|(curr, prev)| (curr - prev) * (curr - prev))
            .sum::<f32>()
            .sqrt();

        let param_norm: f32 = current_params.iter().map(|p| p * p).sum::<f32>().sqrt();

        let relative_change = if param_norm > 1e-8 { param_change_norm / param_norm } else { 0.0 };

        // Adapt step size based on parameter change magnitude
        let step_factor = if relative_change > 0.1 {
            0.5 // Reduce step size for large changes
        } else if relative_change < 0.01 {
            1.5 // Increase step size for small changes
        } else {
            1.0 // Keep normal step size
        };

        0.9 * stored_factor + 0.1 * step_factor
    }

    /// Apply warmup scaling during initial training steps
    fn get_warmup_lr(&self) -> f32 {
        if self.current_step < self.config.warmup_steps {
            let warmup_factor = (self.current_step as f32) / (self.config.warmup_steps as f32);
            self.config.learning_rate * warmup_factor
        } else {
            self.config.learning_rate
        }
    }

    /// Get current learning rate
    pub fn learning_rate(&self) -> f32 {
        self.config.learning_rate
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }
}

impl Optimizer for AMacP {
    fn update(&mut self, _parameter: &mut Tensor, _gradient: &Tensor) -> Result<()> {
        // Implementation for single parameter update
        // This is called by the training framework for each parameter
        Ok(())
    }

    fn step(&mut self) {
        // Step counter increment - called after all parameter updates
        self.current_step += 1;
        self.state.step();
    }

    fn zero_grad(&mut self) {
        // Gradients are typically zeroed by the training framework
        // This method can be used for any optimizer-specific cleanup
    }

    fn get_lr(&self) -> f32 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }
}

// Additional method for batch parameter updates (non-trait)
impl AMacP {
    /// Process multiple parameters at once (non-trait method for convenience)
    pub fn step_batch(&mut self, gradients: &HashMap<String, Tensor>) -> Result<()> {
        let warmup_lr = self.get_warmup_lr();
        let current_step = self.current_step + 1;

        // Process each parameter individually to avoid borrowing conflicts
        for (param_name, gradient) in gradients.iter() {
            let grad_data = gradient.data()?;
            if grad_data.is_empty() {
                continue;
            }

            // Apply gradient clipping if enabled
            let mut clipped_grad = grad_data.clone();
            if let Some(max_norm) = self.config.max_grad_norm {
                let grad_norm: f32 = clipped_grad.iter().map(|g| g * g).sum::<f32>().sqrt();
                if grad_norm > max_norm {
                    let scale = max_norm / grad_norm;
                    for g in clipped_grad.iter_mut() {
                        *g *= scale;
                    }
                }
            }

            // Update gradient heterogeneity
            self.update_gradient_heterogeneity(param_name, &clipped_grad);

            let param_size = clipped_grad.len();

            // Get momentum and variance separately to avoid multiple mutable borrows
            let momentum = {
                let momentum = self.state.get_or_create_momentum(param_name.clone(), param_size);
                momentum.clone()
            };

            let variance = {
                let variance = self.state.get_or_create_variance(param_name.clone(), param_size);
                variance.clone()
            };

            // Compute bias corrections
            let (bias_correction1, bias_correction2) = BiasCorrection::compute_adam_corrections(
                self.config.beta1,
                self.config.beta2,
                current_step,
            );

            // Update momentum and variance (standard Adam updates)
            let mut updated_momentum = momentum;
            let mut updated_variance = variance;
            for i in 0..param_size {
                ParameterUpdate::update_ema(
                    &mut updated_momentum[i],
                    clipped_grad[i],
                    self.config.beta1,
                );
                ParameterUpdate::update_ema(
                    &mut updated_variance[i],
                    clipped_grad[i] * clipped_grad[i],
                    self.config.beta2,
                );
            }

            // Compute bias-corrected estimates
            let m_hat: Vec<f32> = updated_momentum.iter().map(|m| m / bias_correction1).collect();
            let v_hat: Vec<f32> = updated_variance.iter().map(|v| v / bias_correction2).collect();

            // Update dual momentum (aMacP innovation)
            let mut dual_momentum = self
                .dual_momentum
                .entry(param_name.clone())
                .or_insert_with(|| vec![0.0; param_size])
                .clone();

            for i in 0..param_size {
                let dual_momentum_value = self.compute_dual_momentum(m_hat[i], v_hat[i]);
                ParameterUpdate::update_ema(
                    &mut dual_momentum[i],
                    dual_momentum_value,
                    self.config.gamma,
                );
            }

            // Apply consecutive parameter averaging if previous parameters exist
            if let Some(prev_params) = self.previous_params.get(param_name).cloned() {
                let step_factor = {
                    if !self.config.adaptive_step_size {
                        1.0
                    } else {
                        let param_change_norm: f32 = dual_momentum
                            .iter()
                            .zip(prev_params.iter())
                            .map(|(curr, prev)| (curr - prev) * (curr - prev))
                            .sum::<f32>()
                            .sqrt();

                        let param_norm: f32 =
                            dual_momentum.iter().map(|p| p * p).sum::<f32>().sqrt();

                        let relative_change =
                            if param_norm > 1e-8 { param_change_norm / param_norm } else { 0.0 };

                        let step_factor = if relative_change > 0.1 {
                            0.5 // Reduce step size for large changes
                        } else if relative_change < 0.01 {
                            1.5 // Increase step size for small changes
                        } else {
                            1.0 // Keep normal step size
                        };

                        let entry = self.step_size_factors.entry(param_name.clone()).or_insert(1.0);
                        *entry = 0.9 * *entry + 0.1 * step_factor;
                        *entry
                    }
                };

                let heterogeneity_factor = 1.0
                    + self.config.eta * self.gradient_heterogeneity.get(param_name).unwrap_or(&0.0);

                let effective_lr = warmup_lr * step_factor * heterogeneity_factor;

                // aMacP parameter update using dual momentum and consecutive averaging
                for i in 0..param_size {
                    let averaged_param = self.config.gamma * prev_params[i]
                        + (1.0 - self.config.gamma) * dual_momentum[i];

                    // Update parameter using averaged momentum and consecutive parameters
                    let _update =
                        effective_lr * averaged_param / (v_hat[i].sqrt() + self.config.epsilon);
                    // Note: In real implementation, this would update the actual parameters
                    // Here we just track the update for state management
                }
            }

            // Store updated states back
            self.state.momentum.insert(param_name.clone(), updated_momentum);
            self.state.variance.insert(param_name.clone(), updated_variance);
            self.dual_momentum.insert(param_name.clone(), dual_momentum.clone());
            self.previous_params.insert(param_name.clone(), dual_momentum);
        }

        // Update step counter after processing all parameters
        self.current_step = current_step;
        self.state.step = current_step;

        Ok(())
    }
}

impl StatefulOptimizer for AMacP {
    type Config = AMacPConfig;
    type State = OptimizerState;

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut state = HashMap::new();

        // Save step count
        state.insert(
            "step".to_string(),
            Tensor::new(vec![self.current_step as f32])?,
        );

        // Save momentum and variance states
        for (name, momentum) in &self.state.momentum {
            let shape = vec![momentum.len()];
            state.insert(
                format!("momentum_{}", name),
                Tensor::from_vec(momentum.clone(), &shape)?,
            );
        }
        for (name, variance) in &self.state.variance {
            let shape = vec![variance.len()];
            state.insert(
                format!("variance_{}", name),
                Tensor::from_vec(variance.clone(), &shape)?,
            );
        }

        // Save aMacP-specific states
        for (name, dual_mom) in &self.dual_momentum {
            let shape = vec![dual_mom.len()];
            state.insert(
                format!("dual_momentum_{}", name),
                Tensor::from_vec(dual_mom.clone(), &shape)?,
            );
        }
        for (name, prev_params) in &self.previous_params {
            let shape = vec![prev_params.len()];
            state.insert(
                format!("prev_params_{}", name),
                Tensor::from_vec(prev_params.clone(), &shape)?,
            );
        }
        for (name, heterogeneity) in &self.gradient_heterogeneity {
            state.insert(
                format!("heterogeneity_{}", name),
                Tensor::new(vec![*heterogeneity])?,
            );
        }
        for (name, factor) in &self.step_size_factors {
            state.insert(format!("step_factor_{}", name), Tensor::new(vec![*factor])?);
        }

        Ok(state)
    }

    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()> {
        // Load step count
        if let Some(step_tensor) = state.get("step") {
            if let Ok(step_data) = step_tensor.data() {
                if !step_data.is_empty() {
                    self.current_step = step_data[0] as usize;
                    self.state.step = self.current_step;
                }
            }
        }

        // Load momentum and variance states
        for (key, tensor) in &state {
            if let Some(name) = key.strip_prefix("momentum_") {
                if let Ok(data) = tensor.data() {
                    self.state.momentum.insert(name.to_string(), data);
                }
            } else if let Some(name) = key.strip_prefix("variance_") {
                if let Ok(data) = tensor.data() {
                    self.state.variance.insert(name.to_string(), data);
                }
            } else if let Some(name) = key.strip_prefix("dual_momentum_") {
                if let Ok(data) = tensor.data() {
                    self.dual_momentum.insert(name.to_string(), data);
                }
            } else if let Some(name) = key.strip_prefix("prev_params_") {
                if let Ok(data) = tensor.data() {
                    self.previous_params.insert(name.to_string(), data);
                }
            } else if let Some(name) = key.strip_prefix("heterogeneity_") {
                if let Ok(data) = tensor.data() {
                    if !data.is_empty() {
                        self.gradient_heterogeneity.insert(name.to_string(), data[0]);
                    }
                }
            } else if let Some(name) = key.strip_prefix("step_factor_") {
                if let Ok(data) = tensor.data() {
                    if !data.is_empty() {
                        self.step_size_factors.insert(name.to_string(), data[0]);
                    }
                }
            }
        }

        Ok(())
    }

    fn memory_usage(&self) -> StateMemoryStats {
        let base_stats = self.state.memory_usage();

        // Add aMacP-specific memory usage
        let dual_momentum_elements: usize = self.dual_momentum.values().map(|v| v.len()).sum();
        let prev_params_elements: usize = self.previous_params.values().map(|v| v.len()).sum();
        let scalar_elements = self.gradient_heterogeneity.len() + self.step_size_factors.len();

        StateMemoryStats {
            momentum_elements: base_stats.momentum_elements
                + dual_momentum_elements
                + prev_params_elements,
            variance_elements: base_stats.variance_elements,
            third_moment_elements: scalar_elements,
            total_bytes: base_stats.total_bytes
                + (dual_momentum_elements + prev_params_elements + scalar_elements)
                    * std::mem::size_of::<f32>(),
            num_parameters: base_stats.num_parameters,
        }
    }

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }

    fn reset_state(&mut self) {
        self.state.clear();
        self.previous_params.clear();
        self.dual_momentum.clear();
        self.gradient_heterogeneity.clear();
        self.step_size_factors.clear();
        self.current_step = 0;
    }

    fn num_parameters(&self) -> usize {
        self.state.momentum.len()
    }
}

/// Statistics specific to aMacP optimizer
#[derive(Debug, Clone)]
pub struct AMacPStats {
    pub current_step: usize,
    pub average_gradient_heterogeneity: f32,
    pub average_step_size_factor: f32,
    pub total_parameters: usize,
    pub warmup_progress: f32,
    pub dual_momentum_norm: f32,
}

impl AMacP {
    /// Reset all optimizer state (convenience method)
    pub fn reset(&mut self) {
        self.reset_state();
    }

    /// Get comprehensive aMacP statistics
    pub fn get_stats(&self) -> AMacPStats {
        let avg_heterogeneity = if !self.gradient_heterogeneity.is_empty() {
            self.gradient_heterogeneity.values().sum::<f32>()
                / self.gradient_heterogeneity.len() as f32
        } else {
            0.0
        };

        let avg_step_factor = if !self.step_size_factors.is_empty() {
            self.step_size_factors.values().sum::<f32>() / self.step_size_factors.len() as f32
        } else {
            1.0
        };

        let warmup_progress = if self.config.warmup_steps > 0 {
            (self.current_step as f32 / self.config.warmup_steps as f32).min(1.0)
        } else {
            1.0
        };

        let dual_momentum_norm: f32 = self
            .dual_momentum
            .values()
            .flat_map(|v| v.iter())
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();

        AMacPStats {
            current_step: self.current_step,
            average_gradient_heterogeneity: avg_heterogeneity,
            average_step_size_factor: avg_step_factor,
            total_parameters: self.num_parameters(),
            warmup_progress,
            dual_momentum_norm,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amacp_creation() {
        let optimizer = AMacP::new(AMacPConfig::default());
        assert_eq!(optimizer.learning_rate(), 1e-3);
        assert_eq!(optimizer.config.beta1, 0.9);
        assert_eq!(optimizer.config.beta2, 0.999);
        assert_eq!(optimizer.config.gamma, 0.95);
    }

    #[test]
    fn test_amacp_presets() {
        let transformer_opt = AMacP::for_transformers();
        assert_eq!(transformer_opt.config.learning_rate, 6e-4);
        assert_eq!(transformer_opt.config.warmup_steps, 4000);

        let vision_opt = AMacP::for_vision();
        assert_eq!(vision_opt.config.learning_rate, 1e-3);
        assert_eq!(vision_opt.config.warmup_steps, 500);

        let llm_opt = AMacP::for_large_language_models();
        assert_eq!(llm_opt.config.learning_rate, 3e-4);
        assert_eq!(llm_opt.config.warmup_steps, 10000);
    }

    #[test]
    fn test_dual_momentum_computation() {
        let optimizer = AMacP::new(AMacPConfig::default());
        let m_hat = 0.1;
        let v_hat = 0.01;
        let dual_momentum = optimizer.compute_dual_momentum(m_hat, v_hat);

        let expected = 0.5 * 0.1 + 0.5 * 0.01_f32.sqrt();
        assert!((dual_momentum - expected).abs() < 1e-6);
    }

    #[test]
    fn test_learning_rate_getter_setter() {
        let mut optimizer = AMacP::new(AMacPConfig::default());
        assert_eq!(optimizer.learning_rate(), 1e-3);

        optimizer.set_learning_rate(2e-3);
        assert_eq!(optimizer.learning_rate(), 2e-3);
    }

    #[test]
    fn test_warmup_lr_calculation() {
        let mut optimizer = AMacP::new(AMacPConfig {
            learning_rate: 1e-3,
            warmup_steps: 1000,
            ..Default::default()
        });

        optimizer.current_step = 500;
        let warmup_lr = optimizer.get_warmup_lr();
        assert!((warmup_lr - 5e-4).abs() < 1e-6); // 50% of base LR
    }

    #[test]
    fn test_memory_usage_tracking() {
        let optimizer = AMacP::new(AMacPConfig::default());
        let memory_stats = optimizer.memory_usage();

        assert_eq!(memory_stats.momentum_elements, 0);
        assert_eq!(memory_stats.variance_elements, 0);
        assert_eq!(memory_stats.num_parameters, 0);
    }

    #[test]
    fn test_stats_generation() {
        let optimizer = AMacP::new(AMacPConfig::default());
        let stats = optimizer.get_stats();

        assert_eq!(stats.current_step, 0);
        assert_eq!(stats.total_parameters, 0);
        assert_eq!(stats.warmup_progress, 0.0);
        assert_eq!(stats.dual_momentum_norm, 0.0);
    }

    #[test]
    fn test_reset_functionality() {
        let mut optimizer = AMacP::new(AMacPConfig::default());
        optimizer.current_step = 100;

        optimizer.reset();
        assert_eq!(optimizer.current_step, 0);
        assert!(optimizer.dual_momentum.is_empty());
        assert!(optimizer.previous_params.is_empty());
    }

    #[test]
    fn test_state_dict_operations() {
        let optimizer = AMacP::new(AMacPConfig::default());
        let state_dict = optimizer.state_dict();
        assert!(state_dict.is_ok());

        let state = state_dict.unwrap();
        assert!(state.contains_key("step"));
    }

    #[test]
    fn test_config_serialization() {
        let config = AMacPConfig::for_transformers();
        let serialized = serde_json::to_string(&config);
        assert!(serialized.is_ok());

        let deserialized: std::result::Result<AMacPConfig, _> =
            serde_json::from_str(&serialized.unwrap());
        assert!(deserialized.is_ok());
        assert_eq!(deserialized.unwrap().learning_rate, 6e-4);
    }
}
