//! # NovoGrad: Memory-Efficient Adaptive Optimizer
//!
//! NovoGrad is an adaptive gradient method designed for large-scale deep learning
//! training. Its key innovation lies in performing gradient normalization per
//! parameter tensor (layer) rather than per individual weight element, providing
//! significant memory savings for large models.
//!
//! ## Key Features
//!
//! - **Layer-wise Gradient Normalization**: Reduces memory requirements dramatically
//! - **Memory Efficient**: O(L) memory complexity where L is number of layers
//! - **Large-scale Training**: Optimized for models with millions/billions of parameters
//! - **Adaptive Learning**: Combines benefits of Adam with reduced memory footprint
//! - **Gradient Clipping**: Built-in gradient norm clipping for training stability
//!
//! ## Research Reference
//!
//! "Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training
//! of Deep Networks" - Ginsburg et al., 2019.
//!
//! ## Deviation from the paper
//!
//! [`NovoGradConfig::default`] and the task presets enable three extensions the paper
//! does not describe: bias correction, adaptive weight decay and a Nesterov-style
//! look-ahead blend (`memory_factor`). Use [`NovoGradConfig::paper_default`] for the
//! unmodified rule of Ginsburg et al.

use crate::{
    common::{BiasCorrection, OptimizerState, StateMemoryStats},
    traits::StatefulOptimizer,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::{errors::Result, tensor::Tensor, traits::Optimizer};

/// Configuration for NovoGrad optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NovoGradConfig {
    /// Base learning rate
    pub learning_rate: f32,
    /// First momentum coefficient (exponential moving average of gradients)
    pub beta1: f32,
    /// Second momentum coefficient (exponential moving average of layer norms)
    pub beta2: f32,
    /// Small constant for numerical stability
    pub epsilon: f32,
    /// Weight decay coefficient (L2 regularization)
    pub weight_decay: f32,
    /// Gradient clipping threshold (None = no clipping)
    pub grad_clipping: Option<f32>,
    /// Use bias correction for momentum estimates
    pub bias_correction: bool,
    /// Adaptive weight decay based on layer size
    pub adaptive_weight_decay: bool,
    /// Nesterov-style look-ahead coefficient blended into the update direction.
    ///
    /// The update applied is
    /// `m̂ + memory_factor · (g / (√v̂ + ε) + λ·w)`, so `memory_factor = 0`
    /// reproduces Ginsburg et al. exactly and larger values add a fraction of the
    /// current normalised gradient on top of the momentum. Use
    /// [`NovoGradConfig::paper_default`] for the unmodified paper rule.
    pub memory_factor: f32,
    /// Enable layer-wise adaptive learning rates
    pub layer_wise_adaptation: bool,
}

impl Default for NovoGradConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.95, // Higher than Adam for better convergence
            beta2: 0.98, // Layer-wise second moment coefficient
            epsilon: 1e-8,
            weight_decay: 0.0,
            grad_clipping: Some(1.0),
            bias_correction: true,
            adaptive_weight_decay: true,
            memory_factor: 0.8,
            layer_wise_adaptation: true,
        }
    }
}

impl NovoGradConfig {
    /// Exactly the update rule of Ginsburg et al. (2019), with no look-ahead blend,
    /// no adaptive learning-rate rescaling and no adaptive weight decay.
    ///
    /// ```text
    /// v_l = β2·v_l + (1 − β2)·‖g_l‖²
    /// m_l = β1·m_l + (g_l / (√v_l + ε) + λ·w_l)
    /// w_l = w_l − η·m_l
    /// ```
    pub fn paper_default() -> Self {
        Self {
            learning_rate: 1e-2,
            beta1: 0.95,
            beta2: 0.98,
            epsilon: 1e-8,
            weight_decay: 0.0,
            grad_clipping: None,
            bias_correction: false,
            adaptive_weight_decay: false,
            memory_factor: 0.0,
            layer_wise_adaptation: false,
        }
    }

    /// Configuration optimized for very large language models (>1B parameters)
    pub fn for_large_language_models() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.95,
            beta2: 0.999,  // More conservative for large models
            epsilon: 1e-6, // Better numerical stability for large models
            weight_decay: 1e-2,
            grad_clipping: Some(1.0),
            bias_correction: true,
            adaptive_weight_decay: true,
            memory_factor: 0.9, // Strong look-ahead blend
            layer_wise_adaptation: true,
        }
    }

    /// Configuration for computer vision models with batch normalization
    pub fn for_vision_models() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.9, // Standard momentum for vision
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 1e-4,
            grad_clipping: Some(2.0), // Higher clipping for vision models
            bias_correction: true,
            adaptive_weight_decay: false, // Fixed weight decay for vision
            memory_factor: 0.7,
            layer_wise_adaptation: false, // Standard adaptation for vision
        }
    }

    /// Configuration for memory-constrained environments
    pub fn for_memory_constrained() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.95,
            beta2: 0.98,
            epsilon: 1e-8,
            weight_decay: 0.0,
            grad_clipping: Some(1.0),
            bias_correction: false, // Disable for memory savings
            adaptive_weight_decay: false,
            memory_factor: 1.0, // Full look-ahead blend
            layer_wise_adaptation: false,
        }
    }

    /// Configuration for scientific computing and neural ODEs
    pub fn for_scientific_computing() -> Self {
        Self {
            learning_rate: 1e-4, // Conservative LR for scientific applications
            beta1: 0.99,         // High momentum for smooth optimization
            beta2: 0.999,
            epsilon: 1e-10,           // Higher precision for scientific computing
            weight_decay: 1e-6,       // Minimal regularization
            grad_clipping: Some(0.5), // Tight gradient clipping
            bias_correction: true,
            adaptive_weight_decay: true,
            memory_factor: 0.8,
            layer_wise_adaptation: true,
        }
    }
}

/// NovoGrad optimizer implementation with layer-wise gradient normalization
#[derive(Debug)]
pub struct NovoGrad {
    config: NovoGradConfig,
    state: OptimizerState,
    /// Layer-wise second moment estimates (v)
    layer_second_moments: HashMap<String, f32>,
    /// Layer-wise gradient norms for statistics
    layer_grad_norms: HashMap<String, f32>,
    /// Adaptive learning rate factors per layer
    layer_lr_factors: HashMap<String, f32>,
    /// Current step number
    current_step: usize,
    /// Total number of parameters for memory tracking
    total_parameters: usize,
}

impl NovoGrad {
    /// Create a new NovoGrad optimizer
    pub fn new(config: NovoGradConfig) -> Self {
        Self {
            config,
            state: OptimizerState::new(),
            layer_second_moments: HashMap::new(),
            layer_grad_norms: HashMap::new(),
            layer_lr_factors: HashMap::new(),
            current_step: 0,
            total_parameters: 0,
        }
    }

    /// Create NovoGrad for large language models
    pub fn for_large_language_models() -> Self {
        Self::new(NovoGradConfig::for_large_language_models())
    }

    /// Create NovoGrad for vision models
    pub fn for_vision_models() -> Self {
        Self::new(NovoGradConfig::for_vision_models())
    }

    /// Create NovoGrad for memory-constrained environments
    pub fn for_memory_constrained() -> Self {
        Self::new(NovoGradConfig::for_memory_constrained())
    }

    /// Create NovoGrad for scientific computing
    pub fn for_scientific_computing() -> Self {
        Self::new(NovoGradConfig::for_scientific_computing())
    }

    /// Compute layer-wise gradient norm (NovoGrad's key innovation)
    fn compute_layer_grad_norm(&self, gradient: &[f32]) -> f32 {
        let grad_norm_squared: f32 = gradient.iter().map(|g| g * g).sum();
        grad_norm_squared.sqrt()
    }

    /// Apply layer-wise adaptive learning rate
    fn compute_adaptive_lr(&mut self, layer_id: &str, grad_norm: f32) -> f32 {
        if !self.config.layer_wise_adaptation {
            return self.config.learning_rate;
        }

        // Adaptive learning rate based on layer gradient norms
        let base_lr = self.config.learning_rate;
        let prev_norm = self.layer_grad_norms.get(layer_id).copied().unwrap_or(1.0);

        // Compute adaptation factor based on gradient norm change
        let norm_ratio = if prev_norm > 1e-8 { grad_norm / prev_norm } else { 1.0 };

        // Adaptive factor: decrease LR if gradients are growing, increase if shrinking
        let adaptation_factor = if norm_ratio > 1.2 {
            0.8 // Reduce LR for growing gradients
        } else if norm_ratio < 0.8 {
            1.1 // Increase LR for shrinking gradients
        } else {
            1.0 // Keep LR stable
        };

        // Smooth the adaptation factor
        let current_factor = self.layer_lr_factors.get(layer_id).copied().unwrap_or(1.0);
        let new_factor = 0.9 * current_factor + 0.1 * adaptation_factor;
        self.layer_lr_factors.insert(layer_id.to_string(), new_factor);

        base_lr * new_factor
    }

    /// Apply adaptive weight decay based on layer size
    fn compute_adaptive_weight_decay(&self, layer_size: usize) -> f32 {
        if !self.config.adaptive_weight_decay {
            return self.config.weight_decay;
        }

        // Reduce weight decay for larger layers to prevent over-regularization
        let size_factor = (layer_size as f32).sqrt();
        let adapted_wd = self.config.weight_decay / (1.0 + size_factor * 0.001);
        adapted_wd.max(self.config.weight_decay * 0.1) // Minimum 10% of original
    }

    /// Get memory efficiency statistics
    pub fn memory_efficiency(&self) -> MemoryEfficiencyStats {
        let traditional_adam_memory = self.total_parameters * 2 * std::mem::size_of::<f32>(); // m + v
        let novograd_memory = self.state.momentum.values().map(|v| v.len()).sum::<usize>()
            * std::mem::size_of::<f32>()
            + self.layer_second_moments.len() * std::mem::size_of::<f32>();

        let memory_savings = if traditional_adam_memory > 0 {
            1.0 - (novograd_memory as f32) / (traditional_adam_memory as f32)
        } else {
            0.0
        };

        MemoryEfficiencyStats {
            traditional_adam_memory_bytes: traditional_adam_memory,
            novograd_memory_bytes: novograd_memory,
            memory_savings_ratio: memory_savings,
            layer_count: self.layer_second_moments.len(),
            average_layer_size: if !self.layer_second_moments.is_empty() {
                self.total_parameters / self.layer_second_moments.len()
            } else {
                0
            },
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

/// Memory efficiency statistics for NovoGrad
#[derive(Debug, Clone)]
pub struct MemoryEfficiencyStats {
    pub traditional_adam_memory_bytes: usize,
    pub novograd_memory_bytes: usize,
    pub memory_savings_ratio: f32,
    pub layer_count: usize,
    pub average_layer_size: usize,
}

impl NovoGrad {
    /// Applies one NovoGrad step to a single layer, in place.
    ///
    /// `key` is the stable state key from [`crate::param_id::ParamRegistry`]; the
    /// layer-wise second moment, first moment and step counter are all stored under it.
    ///
    /// ```text
    /// g       = clip(∇L)                              (optional)
    /// v_l     = β2·v_l + (1 − β2)·‖g‖²                (one scalar per layer)
    /// m       = β1·m + (g / (√v̂ + ε) + λ·w)
    /// w      -= η_l · (m̂ + memory_factor · (g / (√v̂ + ε) + λ·w))
    /// ```
    fn apply_layer_update(&mut self, key: &str, param: &mut [f32], grad: &[f32]) -> Result<()> {
        if param.len() != grad.len() {
            return Err(trustformers_core::errors::TrustformersError::invalid_input(
                format!(
                    "NovoGrad: parameter has {} elements but the gradient has {}",
                    param.len(),
                    grad.len()
                ),
            ));
        }
        if param.is_empty() {
            return Ok(());
        }

        // Per-layer step counter drives bias correction independently of how many
        // other layers have been visited.
        let local_step = self.state.param_steps.entry(key.to_string()).or_insert(0);
        *local_step += 1;
        let local_step = *local_step;

        // 1. Optional gradient clipping (layer-wise, by norm).
        let mut clipped: Vec<f32> = grad.to_vec();
        if let Some(clip_value) = self.config.grad_clipping {
            let norm = self.compute_layer_grad_norm(&clipped);
            if norm > clip_value && norm > 0.0 {
                let scale = clip_value / norm;
                for g in clipped.iter_mut() {
                    *g *= scale;
                }
            }
        }

        // 2. Layer-wise gradient norm — NovoGrad's key quantity.
        let grad_norm = self.compute_layer_grad_norm(&clipped);
        self.layer_grad_norms.insert(key.to_string(), grad_norm);

        // 3. Layer-wise second moment. The paper initialises v with ‖g‖² on the very
        //    first step rather than starting from zero.
        let layer_v = match self.layer_second_moments.get(key).copied() {
            Some(previous) => {
                self.config.beta2 * previous + (1.0 - self.config.beta2) * grad_norm * grad_norm
            },
            None => grad_norm * grad_norm,
        };
        self.layer_second_moments.insert(key.to_string(), layer_v);

        let (bias_correction1, bias_correction2) = if self.config.bias_correction {
            BiasCorrection::compute_adam_corrections(
                self.config.beta1,
                self.config.beta2,
                local_step,
            )
        } else {
            (1.0, 1.0)
        };

        let v_hat = layer_v / bias_correction2;
        let denominator = v_hat.sqrt() + self.config.epsilon;

        let adaptive_wd = self.compute_adaptive_weight_decay(param.len());
        let adaptive_lr = self.compute_adaptive_lr(key, grad_norm);

        let momentum = self.state.get_or_create_momentum(key.to_string(), param.len());

        for index in 0..param.len() {
            // Normalised gradient plus decoupled weight decay on the real parameter.
            let normalized = clipped[index] / denominator + adaptive_wd * param[index];

            // NovoGrad's first moment accumulates the *normalised* gradient.
            momentum[index] = self.config.beta1 * momentum[index] + normalized;
            let m_hat = momentum[index] / bias_correction1;

            let direction = m_hat + self.config.memory_factor * normalized;
            param[index] -= adaptive_lr * direction;
        }

        self.total_parameters =
            self.state.momentum.values().map(|buffer| buffer.len()).sum::<usize>();

        Ok(())
    }

    /// Updates one parameter that carries a stable caller-supplied name.
    ///
    /// Prefer this over [`Optimizer::update`]: NovoGrad's state is layer-wise, and a
    /// name is the most durable way to identify a layer across processes.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor is not `f32`-readable or the shapes disagree.
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
        self.apply_layer_update(&key, &mut values, &grad)?;
        parameter.set_data_f32(&values)?;
        self.state.params.rebind(id, parameter)?;
        Ok(())
    }
}

impl Optimizer for NovoGrad {
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
        self.apply_layer_update(&key, &mut values, &grad)?;
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
    }

    fn get_lr(&self) -> f32 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }
}

// Additional method for batch parameter updates (non-trait)
impl NovoGrad {
    /// Updates a whole named parameter set at once with NovoGrad's layer-wise rule.
    ///
    /// Parameters are keyed by name, which is the durable identity used for the
    /// layer-wise state (see [`crate::param_id`]). Every entry present in both maps
    /// is updated in place; gradients without a matching parameter are an error,
    /// because silently skipping them is how the previous no-op version hid itself.
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
        self.current_step += 1;

        // Deterministic visit order keeps anonymous registration indices reproducible.
        let mut names: Vec<&String> = gradients.keys().collect();
        names.sort();

        for name in names {
            let gradient = gradients.get(name).ok_or_else(|| {
                trustformers_core::errors::TrustformersError::invalid_input(format!(
                    "NovoGrad: gradient '{name}' disappeared during iteration"
                ))
            })?;
            let parameter = parameters.get_mut(name).ok_or_else(|| {
                trustformers_core::errors::TrustformersError::invalid_input(format!(
                    "NovoGrad: no parameter named '{name}' to apply its gradient to"
                ))
            })?;

            if gradient.is_empty() {
                continue;
            }

            let id = self.state.params.id_for_named_tensor(name, parameter)?;
            let key = self
                .state
                .params
                .key(id)
                .map(str::to_string)
                .unwrap_or_else(|| format!("n:{name}"));

            let mut values = parameter.data_f32()?;
            let grad = gradient.data_f32()?;
            self.apply_layer_update(&key, &mut values, &grad)?;
            parameter.set_data_f32(&values)?;
            self.state.params.rebind(id, parameter)?;
        }

        Ok(())
    }
}

impl StatefulOptimizer for NovoGrad {
    type Config = NovoGradConfig;
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
        let mut state = HashMap::new();

        // Save step count
        state.insert(
            "step".to_string(),
            Tensor::new(vec![self.current_step as f32])?,
        );

        // Save momentum states
        for (name, momentum) in &self.state.momentum {
            let shape = vec![momentum.len()];
            state.insert(
                format!("momentum_{}", name),
                Tensor::from_vec(momentum.clone(), &shape)?,
            );
        }

        // Save NovoGrad-specific states (layer-wise second moments)
        for (name, v) in &self.layer_second_moments {
            state.insert(format!("layer_v_{}", name), Tensor::new(vec![*v])?);
        }

        // Save layer-wise learning rate factors
        for (name, factor) in &self.layer_lr_factors {
            state.insert(format!("lr_factor_{}", name), Tensor::new(vec![*factor])?);
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

        // Load momentum states
        for (key, tensor) in &state {
            if let Some(name) = key.strip_prefix("momentum_") {
                if let Ok(data) = tensor.data() {
                    self.state.momentum.insert(name.to_string(), data);
                }
            } else if let Some(name) = key.strip_prefix("layer_v_") {
                if let Ok(data) = tensor.data() {
                    if !data.is_empty() {
                        self.layer_second_moments.insert(name.to_string(), data[0]);
                    }
                }
            } else if let Some(name) = key.strip_prefix("lr_factor_") {
                if let Ok(data) = tensor.data() {
                    if !data.is_empty() {
                        self.layer_lr_factors.insert(name.to_string(), data[0]);
                    }
                }
            }
        }

        Ok(())
    }

    fn memory_usage(&self) -> StateMemoryStats {
        let momentum_elements: usize = self.state.momentum.values().map(|v| v.len()).sum();
        let layer_elements = self.layer_second_moments.len() + self.layer_lr_factors.len();

        StateMemoryStats {
            momentum_elements,
            variance_elements: 0, // NovoGrad doesn't use per-parameter variance
            third_moment_elements: layer_elements, // Layer-wise second moments
            total_bytes: momentum_elements * std::mem::size_of::<f32>()
                + layer_elements * std::mem::size_of::<f32>(),
            num_parameters: self.state.momentum.len(),
        }
    }

    fn reset_state(&mut self) {
        self.state.clear();
        self.layer_second_moments.clear();
        self.layer_grad_norms.clear();
        self.layer_lr_factors.clear();
        self.current_step = 0;
        self.total_parameters = 0;
    }

    fn num_parameters(&self) -> usize {
        self.state.momentum.len()
    }
}

/// Comprehensive NovoGrad statistics
#[derive(Debug, Clone)]
pub struct NovoGradStats {
    pub current_step: usize,
    pub total_parameters: usize,
    pub layer_count: usize,
    pub average_grad_norm: f32,
    pub max_grad_norm: f32,
    pub min_grad_norm: f32,
    pub memory_efficiency: MemoryEfficiencyStats,
    pub adaptive_lr_range: (f32, f32), // (min, max) adaptive learning rates
}

impl NovoGrad {
    /// Reset all optimizer state (convenience method)
    pub fn reset(&mut self) {
        self.reset_state();
    }

    /// Get comprehensive NovoGrad statistics
    pub fn get_stats(&self) -> NovoGradStats {
        let grad_norms: Vec<f32> = self.layer_grad_norms.values().copied().collect();
        let lr_factors: Vec<f32> = self.layer_lr_factors.values().copied().collect();

        let avg_grad_norm = if !grad_norms.is_empty() {
            grad_norms.iter().sum::<f32>() / grad_norms.len() as f32
        } else {
            0.0
        };

        let (min_grad_norm, max_grad_norm) = if !grad_norms.is_empty() {
            let min = grad_norms.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max = grad_norms.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            (min, max)
        } else {
            (0.0, 0.0)
        };

        let adaptive_lr_range = if !lr_factors.is_empty() {
            let min_factor = lr_factors.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_factor = lr_factors.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            (
                self.config.learning_rate * min_factor,
                self.config.learning_rate * max_factor,
            )
        } else {
            (self.config.learning_rate, self.config.learning_rate)
        };

        NovoGradStats {
            current_step: self.current_step,
            total_parameters: self.total_parameters,
            layer_count: self.layer_second_moments.len(),
            average_grad_norm: avg_grad_norm,
            max_grad_norm,
            min_grad_norm,
            memory_efficiency: self.memory_efficiency(),
            adaptive_lr_range,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_novograd_creation() {
        let optimizer = NovoGrad::new(NovoGradConfig::default());
        assert_eq!(optimizer.learning_rate(), 1e-3);
        assert_eq!(optimizer.config.beta1, 0.95);
        assert_eq!(optimizer.config.beta2, 0.98);
    }

    #[test]
    fn test_novograd_presets() {
        let llm_opt = NovoGrad::for_large_language_models();
        assert_eq!(llm_opt.config.beta2, 0.999);
        assert_eq!(llm_opt.config.memory_factor, 0.9);

        let vision_opt = NovoGrad::for_vision_models();
        assert_eq!(vision_opt.config.beta1, 0.9);
        assert!(!vision_opt.config.layer_wise_adaptation);

        let memory_opt = NovoGrad::for_memory_constrained();
        assert_eq!(memory_opt.config.memory_factor, 1.0);
        assert!(!memory_opt.config.bias_correction);

        let sci_opt = NovoGrad::for_scientific_computing();
        assert_eq!(sci_opt.config.learning_rate, 1e-4);
        assert_eq!(sci_opt.config.epsilon, 1e-10);
    }

    #[test]
    fn test_layer_grad_norm_computation() {
        let optimizer = NovoGrad::new(NovoGradConfig::default());
        let gradient = vec![3.0, 4.0]; // Norm should be 5.0
        let norm = optimizer.compute_layer_grad_norm(&gradient);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_adaptive_weight_decay() {
        let optimizer = NovoGrad::new(NovoGradConfig {
            adaptive_weight_decay: true,
            weight_decay: 1e-4,
            ..Default::default()
        });

        let small_layer_wd = optimizer.compute_adaptive_weight_decay(100);
        let large_layer_wd = optimizer.compute_adaptive_weight_decay(10000);

        // Larger layers should have smaller weight decay
        assert!(large_layer_wd < small_layer_wd);
        assert!(large_layer_wd >= 1e-5); // Should not be less than 10% of original
    }

    #[test]
    fn test_learning_rate_getter_setter() {
        let mut optimizer = NovoGrad::new(NovoGradConfig::default());
        assert_eq!(optimizer.learning_rate(), 1e-3);

        optimizer.set_learning_rate(2e-3);
        assert_eq!(optimizer.learning_rate(), 2e-3);
    }

    #[test]
    fn test_memory_efficiency_tracking() {
        let optimizer = NovoGrad::new(NovoGradConfig::default());
        let efficiency = optimizer.memory_efficiency();

        assert_eq!(efficiency.layer_count, 0);
        assert_eq!(efficiency.average_layer_size, 0);
        assert_eq!(efficiency.novograd_memory_bytes, 0);
    }

    #[test]
    fn test_memory_usage_tracking() {
        let optimizer = NovoGrad::new(NovoGradConfig::default());
        let memory_stats = optimizer.memory_usage();

        assert_eq!(memory_stats.momentum_elements, 0);
        assert_eq!(memory_stats.variance_elements, 0); // NovoGrad doesn't use per-param variance
        assert_eq!(memory_stats.num_parameters, 0);
    }

    #[test]
    fn test_stats_generation() {
        let optimizer = NovoGrad::new(NovoGradConfig::default());
        let stats = optimizer.get_stats();

        assert_eq!(stats.current_step, 0);
        assert_eq!(stats.total_parameters, 0);
        assert_eq!(stats.layer_count, 0);
        assert_eq!(stats.average_grad_norm, 0.0);
    }

    #[test]
    fn test_reset_functionality() {
        let mut optimizer = NovoGrad::new(NovoGradConfig::default());
        optimizer.current_step = 100;
        optimizer.layer_second_moments.insert("test".to_string(), 0.5);

        optimizer.reset();
        assert_eq!(optimizer.current_step, 0);
        assert!(optimizer.layer_second_moments.is_empty());
    }

    #[test]
    fn test_state_dict_operations() {
        let optimizer = NovoGrad::new(NovoGradConfig::default());
        let state_dict = optimizer.state_dict();
        assert!(state_dict.is_ok());

        let state = state_dict.expect("Operation failed in test");
        assert!(state.contains_key("step"));
    }

    fn paper_optimizer(lr: f32) -> NovoGrad {
        NovoGrad::new(NovoGradConfig {
            learning_rate: lr,
            ..NovoGradConfig::paper_default()
        })
    }

    fn tensor(values: &[f32]) -> Tensor {
        Tensor::from_vec(values.to_vec(), &[values.len()]).expect("tensor")
    }

    /// Regression: `Optimizer::update` used to be an empty `Ok(())` stub, so no code
    /// path in this module ever wrote to a parameter.
    #[test]
    fn update_moves_the_parameter() {
        let mut optimizer = paper_optimizer(0.1);
        let mut param = tensor(&[3.0, 4.0]);
        let grad = tensor(&[3.0, 4.0]);

        let before = param.data_f32().expect("data");
        optimizer.update(&mut param, &grad).expect("update");
        let after = param.data_f32().expect("data");
        assert!(
            after[0] < before[0],
            "parameter must descend: {before:?} -> {after:?}"
        );
        assert!(
            after[1] < before[1],
            "parameter must descend: {before:?} -> {after:?}"
        );
    }

    /// Exact hand-computed NovoGrad steps for `lr = 0.1`, `β1 = 0.95`, `β2 = 0.98`.
    ///
    /// Step 1: `‖g‖ = 5`, `v = 25` (paper initialisation), `ĝ = g/5 = [0.6, 0.8]`,
    /// `m = ĝ`, so `w = [3, 4] − 0.1·[0.6, 0.8] = [2.94, 3.92]`.
    ///
    /// Step 2: `v = 0.98·25 + 0.02·25 = 25`, `m = 0.95·[0.6, 0.8] + [0.6, 0.8]
    /// = [1.17, 1.56]`, so `w = [2.94, 3.92] − 0.1·[1.17, 1.56] = [2.823, 3.764]`.
    #[test]
    fn two_steps_match_hand_computation() {
        let mut optimizer = paper_optimizer(0.1);
        let mut param = tensor(&[3.0, 4.0]);
        let grad = tensor(&[3.0, 4.0]);

        optimizer.update_named("layer", &mut param, &grad).expect("step 1");
        let after_one = param.data_f32().expect("data");
        assert!((after_one[0] - 2.94).abs() < 1e-5, "got {}", after_one[0]);
        assert!((after_one[1] - 3.92).abs() < 1e-5, "got {}", after_one[1]);

        optimizer.update_named("layer", &mut param, &grad).expect("step 2");
        let after_two = param.data_f32().expect("data");
        assert!((after_two[0] - 2.823).abs() < 1e-4, "got {}", after_two[0]);
        assert!((after_two[1] - 3.764).abs() < 1e-4, "got {}", after_two[1]);
    }

    /// The layer-wise second moment is one scalar per layer, not one per element.
    #[test]
    fn second_moment_is_layer_wise() {
        let mut optimizer = paper_optimizer(0.1);
        let mut param = tensor(&[1.0; 16]);
        let grad = tensor(&[0.25; 16]);

        optimizer.update_named("block", &mut param, &grad).expect("update");
        assert_eq!(optimizer.layer_second_moments.len(), 1);
        let v = optimizer.layer_second_moments.get("n:block").copied().expect("v");
        // ‖g‖² = 16 · 0.0625 = 1.0
        assert!((v - 1.0).abs() < 1e-5, "got {v}");
    }

    /// Regression: `step_batch` used to take gradients only and could not write.
    #[test]
    fn step_batch_updates_every_parameter() {
        let mut optimizer = paper_optimizer(0.1);
        let mut params = HashMap::new();
        params.insert("a".to_string(), tensor(&[1.0, 1.0]));
        params.insert("b".to_string(), tensor(&[2.0, 2.0]));
        let before_a = params["a"].data_f32().expect("data");
        let before_b = params["b"].data_f32().expect("data");

        let mut grads = HashMap::new();
        grads.insert("a".to_string(), tensor(&[1.0, 1.0]));
        grads.insert("b".to_string(), tensor(&[1.0, 1.0]));

        optimizer.step_batch(&mut params, &grads).expect("step_batch");

        let after_a = params["a"].data_f32().expect("data");
        let after_b = params["b"].data_f32().expect("data");
        assert!(after_a[0] < before_a[0], "'a' must move");
        assert!(after_b[0] < before_b[0], "'b' must move");
        assert_eq!(optimizer.layer_second_moments.len(), 2);
    }

    /// A gradient with no matching parameter is an error, never a silent skip.
    #[test]
    fn step_batch_rejects_orphan_gradients() {
        let mut optimizer = paper_optimizer(0.1);
        let mut params = HashMap::new();
        params.insert("a".to_string(), tensor(&[1.0]));
        let mut grads = HashMap::new();
        grads.insert("missing".to_string(), tensor(&[1.0]));

        assert!(optimizer.step_batch(&mut params, &grads).is_err());
    }

    /// Weight decay must use the real parameter value, not the old `* 0.0` placeholder.
    #[test]
    fn weight_decay_uses_real_parameter_values() {
        let mut optimizer = NovoGrad::new(NovoGradConfig {
            learning_rate: 0.1,
            weight_decay: 0.5,
            ..NovoGradConfig::paper_default()
        });
        let mut param = tensor(&[10.0]);
        // A zero gradient means the *only* source of movement is the decay term.
        let grad = tensor(&[0.0]);

        optimizer.update_named("w", &mut param, &grad).expect("update");
        let after = param.data_f32().expect("data")[0];
        assert!(
            after < 10.0,
            "weight decay must shrink the parameter, got {after}"
        );
    }

    /// Convergence smoke test on the quadratic bowl `f(x) = Σ x²` (`∇f = 2x`).
    #[test]
    fn descends_a_quadratic_bowl() {
        let mut optimizer = paper_optimizer(0.05);
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
        let config = NovoGradConfig::for_large_language_models();
        let serialized = serde_json::to_string(&config);
        assert!(serialized.is_ok());

        let deserialized: std::result::Result<NovoGradConfig, _> =
            serde_json::from_str(&serialized.expect("Deserialization failed"));
        assert!(deserialized.is_ok());
        assert_eq!(deserialized.expect("Operation failed in test").beta2, 0.999);
    }
}
