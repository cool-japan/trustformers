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

// reason: research-stage module — reserved API/scaffolding fields and methods
// retained intentionally for in-progress features; not yet on active call paths.
#![allow(dead_code)]

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

    /// Averages the two momenta into one descent direction.
    ///
    /// `α·m̂ + (1 − α)·m̂/(√v̂ + ε)` blends the raw (heavy-ball) first moment with the
    /// Adam-preconditioned one. Both terms carry the sign of `m̂`, so the blend is
    /// always a descent direction — the earlier `(1 − α)·√v̂` term was unsigned and
    /// therefore pushed every coordinate in the positive direction regardless of the
    /// gradient.
    fn compute_dual_momentum(&self, m_hat: f32, v_hat: f32) -> f32 {
        let preconditioned = m_hat / (v_hat.sqrt() + self.config.epsilon);
        self.config.alpha * m_hat + (1.0 - self.config.alpha) * preconditioned
    }

    /// Learning rate after warmup scaling at an explicit step index.
    fn warmup_lr_at(&self, step: usize) -> f32 {
        if self.config.warmup_steps > 0 && step < self.config.warmup_steps {
            self.config.learning_rate * (step as f32) / (self.config.warmup_steps as f32)
        } else {
            self.config.learning_rate
        }
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
        self.warmup_lr_at(self.current_step)
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

impl AMacP {
    /// Applies one aMacP step to a single parameter tensor, in place.
    ///
    /// The rule implemented here — the module's reading of Kumar et al. (2025), whose
    /// contribution is "the average of both momentums and consecutive parameters to
    /// adaptively change the step size" — is:
    ///
    /// ```text
    /// g    = clip(∇L)
    /// m    = β1·m + (1 − β1)·g          v = β2·v + (1 − β2)·g²
    /// m̂    = m/(1 − β1ᵗ)                v̂ = v/(1 − β2ᵗ)
    /// d    = α·m̂ + (1 − α)·m̂/(√v̂ + ε)   (average of both momenta)
    /// w̄    = γ·w̄ + (1 − γ)·w             (average of consecutive parameters)
    /// s    = f(‖w − w̄‖ / ‖w‖)            (adaptive step factor)
    /// h    = EMA of grad std / grad norm (gradient heterogeneity)
    /// η    = warmup(t) · s · (1 + η_cfg·h)
    /// w   -= η·(d + λ·w)
    /// ```
    ///
    /// Every configuration knob therefore has an observable effect on the update.
    fn apply_update(&mut self, key: &str, param: &mut [f32], grad: &[f32]) -> Result<()> {
        if param.len() != grad.len() {
            return Err(trustformers_core::errors::TrustformersError::invalid_input(
                format!(
                    "aMacP: parameter has {} elements but the gradient has {}",
                    param.len(),
                    grad.len()
                ),
            ));
        }
        if param.is_empty() {
            return Ok(());
        }

        let local_step = self.state.param_steps.entry(key.to_string()).or_insert(0);
        *local_step += 1;
        let local_step = *local_step;

        // 1. Optional gradient clipping by global norm of this tensor.
        let mut clipped: Vec<f32> = grad.to_vec();
        if let Some(max_norm) = self.config.max_grad_norm {
            let norm: f32 = clipped.iter().map(|g| g * g).sum::<f32>().sqrt();
            if norm > max_norm && norm > 0.0 {
                let scale = max_norm / norm;
                for value in clipped.iter_mut() {
                    *value *= scale;
                }
            }
        }

        // 2. Gradient heterogeneity tracking (drives the adaptation strength `eta`).
        self.update_gradient_heterogeneity(key, &clipped);

        // 3. Adam-style first and second moments.
        let (bias_correction1, bias_correction2) = BiasCorrection::compute_adam_corrections(
            self.config.beta1,
            self.config.beta2,
            local_step,
        );

        let size = param.len();
        let mut momentum = self.state.get_or_create_momentum(key.to_string(), size).clone();
        let mut variance = self.state.get_or_create_variance(key.to_string(), size).clone();
        for index in 0..size {
            ParameterUpdate::update_ema(&mut momentum[index], clipped[index], self.config.beta1);
            ParameterUpdate::update_ema(
                &mut variance[index],
                clipped[index] * clipped[index],
                self.config.beta2,
            );
        }

        // 4. Average of both momenta.
        let mut dual = self.dual_momentum.get(key).cloned().unwrap_or_else(|| vec![0.0_f32; size]);
        for index in 0..size {
            let m_hat = momentum[index] / bias_correction1;
            let v_hat = variance[index] / bias_correction2;
            dual[index] = self.compute_dual_momentum(m_hat, v_hat);
        }

        // 5. Average of consecutive parameters, and the step factor it induces.
        let averaged = match self.previous_params.get(key) {
            Some(previous) if previous.len() == size => {
                let mut averaged = previous.clone();
                for index in 0..size {
                    averaged[index] = self.config.gamma * previous[index]
                        + (1.0 - self.config.gamma) * param[index];
                }
                averaged
            },
            // First sighting: the running average starts at the current parameter.
            _ => param.to_vec(),
        };

        let stored_factor = self.step_size_factors.get(key).copied().unwrap_or(1.0);
        let step_factor =
            Self::compute_adaptive_step_size_static(&self.config, param, &averaged, stored_factor);
        self.step_size_factors.insert(key.to_string(), step_factor);

        // 6. Effective learning rate.
        let heterogeneity = self.gradient_heterogeneity.get(key).copied().unwrap_or(0.0);
        let heterogeneity_factor = 1.0 + self.config.eta * heterogeneity;
        let effective_lr = self.warmup_lr_at(local_step) * step_factor * heterogeneity_factor;

        // 7. The actual parameter write.
        for index in 0..size {
            param[index] -= effective_lr * (dual[index] + self.config.weight_decay * param[index]);
        }

        // 8. Persist state.
        self.state.momentum.insert(key.to_string(), momentum);
        self.state.variance.insert(key.to_string(), variance);
        self.dual_momentum.insert(key.to_string(), dual);
        self.previous_params.insert(key.to_string(), averaged);

        Ok(())
    }

    /// Updates one parameter that carries a stable caller-supplied name.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor is not `f32`-readable or shapes disagree.
    pub fn update_named(
        &mut self,
        name: &str,
        parameter: &mut Tensor,
        gradient: &Tensor,
    ) -> Result<()> {
        let id = self.state.params.id_for_named_tensor(name, parameter)?;
        let key = self
            .state
            .params
            .key(id)
            .map(str::to_string)
            .unwrap_or_else(|| format!("n:{name}"));

        let mut values = parameter.data_f32()?;
        let grad = gradient.data_f32()?;
        self.apply_update(&key, &mut values, &grad)?;
        parameter.set_data_f32(&values)?;
        self.state.params.rebind(id, parameter)?;
        Ok(())
    }
}

impl Optimizer for AMacP {
    fn update(&mut self, parameter: &mut Tensor, gradient: &Tensor) -> Result<()> {
        let id = self.state.params.id_for_tensor(parameter)?;
        let key = self
            .state
            .params
            .key(id)
            .map(str::to_string)
            .unwrap_or_else(|| format!("p:{}", id.index()));

        let mut values = parameter.data_f32()?;
        let grad = gradient.data_f32()?;
        self.apply_update(&key, &mut values, &grad)?;
        parameter.set_data_f32(&values)?;
        self.state.params.rebind(id, parameter)?;
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
    /// Updates a whole named parameter set at once.
    ///
    /// Parameters are keyed by name, which is the durable state identity (see
    /// [`crate::param_id`]). A gradient without a matching parameter is an error
    /// rather than a silent skip.
    ///
    /// # Errors
    ///
    /// Returns an error when a gradient has no matching parameter, when shapes
    /// disagree, or when a tensor is not `f32`-readable.
    pub fn step_batch(
        &mut self,
        parameters: &mut HashMap<String, Tensor>,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<()> {
        // Deterministic visit order keeps registration indices reproducible.
        let mut names: Vec<String> = gradients.keys().cloned().collect();
        names.sort();

        for name in names {
            let gradient = gradients.get(&name).ok_or_else(|| {
                trustformers_core::errors::TrustformersError::invalid_input(format!(
                    "aMacP: gradient '{name}' disappeared during iteration"
                ))
            })?;
            if gradient.is_empty() {
                continue;
            }
            let parameter = parameters.get_mut(&name).ok_or_else(|| {
                trustformers_core::errors::TrustformersError::invalid_input(format!(
                    "aMacP: no parameter named '{name}' to apply its gradient to"
                ))
            })?;

            let id = self.state.params.id_for_named_tensor(&name, parameter)?;
            let key = self
                .state
                .params
                .key(id)
                .map(str::to_string)
                .unwrap_or_else(|| format!("n:{name}"));

            let mut values = parameter.data_f32()?;
            let grad = gradient.data_f32()?;
            self.apply_update(&key, &mut values, &grad)?;
            parameter.set_data_f32(&values)?;
            self.state.params.rebind(id, parameter)?;
        }

        self.current_step += 1;
        self.state.step = self.current_step;

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

        // α·m̂ + (1 − α)·m̂/(√v̂ + ε): both terms keep the sign of m̂.
        let expected = 0.5 * 0.1 + 0.5 * (0.1 / (0.01_f32.sqrt() + 1e-8));
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

        let state = state_dict.expect("Operation failed in test");
        assert!(state.contains_key("step"));
    }

    fn plain_config(lr: f32) -> AMacPConfig {
        AMacPConfig {
            learning_rate: lr,
            beta1: 0.9,
            beta2: 0.999,
            gamma: 0.95,
            alpha: 0.5,
            eta: 0.0,
            epsilon: 1e-8,
            weight_decay: 0.0,
            max_grad_norm: None,
            adaptive_step_size: false,
            warmup_steps: 0,
        }
    }

    fn tensor(values: &[f32]) -> Tensor {
        Tensor::from_vec(values.to_vec(), &[values.len()]).expect("tensor")
    }

    /// Regression: `Optimizer::update` was an empty `Ok(())` stub and `step_batch`
    /// received no parameters, so aMacP could not write to anything.
    #[test]
    fn update_moves_the_parameter() {
        let mut optimizer = AMacP::new(plain_config(0.1));
        let mut param = tensor(&[1.0, -1.0]);
        let grad = tensor(&[1.0, -1.0]);

        optimizer.update(&mut param, &grad).expect("update");
        let after = param.data_f32().expect("data");
        assert!(after[0] < 1.0, "positive gradient must decrease the weight");
        assert!(
            after[1] > -1.0,
            "negative gradient must increase the weight"
        );
    }

    /// Exact hand-computed steps. With `α = 0.5`, `ε → 0` and a constant unit
    /// gradient the bias-corrected moments satisfy `m̂ = v̂ = 1`, so the dual momentum
    /// is `0.5·1 + 0.5·1 = 1` and each step subtracts exactly `lr`.
    #[test]
    fn two_steps_match_hand_computation() {
        let mut optimizer = AMacP::new(plain_config(0.1));
        let mut param = tensor(&[1.0]);
        let grad = tensor(&[1.0]);

        optimizer.update_named("w", &mut param, &grad).expect("step 1");
        let after_one = param.data_f32().expect("data")[0];
        assert!((after_one - 0.9).abs() < 1e-5, "got {after_one}");

        optimizer.update_named("w", &mut param, &grad).expect("step 2");
        let after_two = param.data_f32().expect("data")[0];
        assert!((after_two - 0.8).abs() < 1e-5, "got {after_two}");
    }

    /// The dual momentum must follow the gradient sign in both directions.
    #[test]
    fn dual_momentum_is_a_descent_direction() {
        let optimizer = AMacP::new(plain_config(0.1));
        assert!(optimizer.compute_dual_momentum(-0.5, 0.25) < 0.0);
        assert!(optimizer.compute_dual_momentum(0.5, 0.25) > 0.0);
    }

    /// Weight decay must reach the parameters even when the gradient is zero.
    #[test]
    fn weight_decay_shrinks_a_zero_gradient_parameter() {
        let mut config = plain_config(0.1);
        config.weight_decay = 0.5;
        let mut optimizer = AMacP::new(config);
        let mut param = tensor(&[10.0]);
        let grad = tensor(&[0.0]);

        optimizer.update_named("w", &mut param, &grad).expect("update");
        let after = param.data_f32().expect("data")[0];
        assert!((after - 9.5).abs() < 1e-4, "expected 9.5, got {after}");
    }

    /// Regression: `step_batch` took gradients only and never wrote a parameter.
    #[test]
    fn step_batch_updates_every_parameter() {
        let mut optimizer = AMacP::new(plain_config(0.1));
        let mut params = HashMap::new();
        params.insert("a".to_string(), tensor(&[1.0]));
        params.insert("b".to_string(), tensor(&[2.0]));
        let mut grads = HashMap::new();
        grads.insert("a".to_string(), tensor(&[1.0]));
        grads.insert("b".to_string(), tensor(&[1.0]));

        optimizer.step_batch(&mut params, &grads).expect("step_batch");

        assert!((params["a"].data_f32().expect("data")[0] - 0.9).abs() < 1e-5);
        assert!((params["b"].data_f32().expect("data")[0] - 1.9).abs() < 1e-5);
        assert_eq!(optimizer.current_step, 1);
    }

    /// A gradient with no matching parameter is an error, never a silent skip.
    #[test]
    fn step_batch_rejects_orphan_gradients() {
        let mut optimizer = AMacP::new(plain_config(0.1));
        let mut params = HashMap::new();
        params.insert("a".to_string(), tensor(&[1.0]));
        let mut grads = HashMap::new();
        grads.insert("ghost".to_string(), tensor(&[1.0]));

        assert!(optimizer.step_batch(&mut params, &grads).is_err());
    }

    /// The consecutive-parameter average must track the real iterates, not the
    /// dual-momentum buffer it used to be fed.
    #[test]
    fn consecutive_parameter_average_tracks_the_iterates() {
        let mut optimizer = AMacP::new(plain_config(0.1));
        let mut param = tensor(&[1.0]);
        let grad = tensor(&[1.0]);

        optimizer.update_named("w", &mut param, &grad).expect("step 1");
        let first = optimizer.previous_params.get("n:w").and_then(|v| v.first()).copied();
        assert_eq!(
            first,
            Some(1.0),
            "the first average is the initial parameter"
        );

        optimizer.update_named("w", &mut param, &grad).expect("step 2");
        // w̄ = 0.95·1.0 + 0.05·0.9 = 0.995
        let second = optimizer.previous_params.get("n:w").and_then(|v| v.first()).copied();
        let second = second.expect("average present");
        assert!((second - 0.995).abs() < 1e-5, "got {second}");
    }

    /// Convergence smoke test on the quadratic bowl `f(x) = Σ x²` (`∇f = 2x`).
    #[test]
    fn descends_a_quadratic_bowl() {
        let mut optimizer = AMacP::new(plain_config(0.05));
        let mut param = tensor(&[3.0, -4.0]);
        let initial: f32 = param.data_f32().expect("data").iter().map(|v| v * v).sum();

        for _ in 0..400 {
            let values = param.data_f32().expect("data");
            let grad = tensor(&values.iter().map(|v| 2.0 * v).collect::<Vec<_>>());
            optimizer.update_named("w", &mut param, &grad).expect("step");
        }

        let final_loss: f32 = param.data_f32().expect("data").iter().map(|v| v * v).sum();
        assert!(
            final_loss < initial * 0.05,
            "loss must fall: {initial} -> {final_loss}"
        );
    }

    #[test]
    fn test_config_serialization() {
        let config = AMacPConfig::for_transformers();
        let serialized = serde_json::to_string(&config);
        assert!(serialized.is_ok());

        let deserialized: std::result::Result<AMacPConfig, _> =
            serde_json::from_str(&serialized.expect("Deserialization failed"));
        assert!(deserialized.is_ok());
        assert_eq!(
            deserialized.expect("Operation failed in test").learning_rate,
            6e-4
        );
    }
}
