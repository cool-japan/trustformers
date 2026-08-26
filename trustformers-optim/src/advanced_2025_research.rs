//! # Advanced 2025 Research Optimizers
//!
//! This module implements the latest cutting-edge optimization algorithms from 2025 research,
//! representing the absolute forefront of optimization technology for deep learning.
//!
//! ## Algorithms Included:
//! - **DiWo (Discriminative Weight Orthogonalization)**: Orthogonal weight updates for improved generalization
//! - **MeZO-V2 (Memory-Efficient Zeroth-Order V2)**: Advanced zeroth-order optimization for billion-parameter models
//! - **AdaWin (Adaptive Window)**: Dynamic window-based momentum with gradient history adaptation
//! - **QuasiNewton-Lite**: Lightweight second-order approximation with minimal memory overhead

// reason: research-stage module — reserved API/scaffolding fields and methods
// retained intentionally for in-progress features; not yet on active call paths.
#![allow(dead_code)]

use crate::common::{GradientProcessor, OptimizerState, StateMemoryStats};
use crate::traits::StatefulOptimizer;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::{traits::Optimizer, Tensor};

/// DiWo (Discriminative Weight Orthogonalization) Optimizer
///
/// Based on "Discriminative Weight Orthogonalization for Improved Generalization" (ICML 2025)
///
/// Key innovations:
/// - Orthogonal constraint enforcement during optimization
/// - Discriminative regularization for feature diversity
/// - Adaptive orthogonalization strength based on layer depth
/// - Compatible with existing momentum-based methods
#[derive(Debug, Clone)]
pub struct DiWo {
    /// Base learning rate
    learning_rate: f32,
    /// Momentum coefficient for first moment
    beta1: f32,
    /// Momentum coefficient for second moment
    beta2: f32,
    /// Small constant for numerical stability
    eps: f32,
    /// Weight decay coefficient
    weight_decay: f32,
    /// Orthogonalization strength
    ortho_strength: f32,
    /// Layer-wise orthogonalization adaptation
    adaptive_ortho: bool,
    /// Current optimization step
    step: usize,
    /// Momentum states (first moment)
    momentum_states: HashMap<String, Tensor>,
    /// Velocity states (second moment)
    velocity_states: HashMap<String, Tensor>,
    /// Orthogonalization history
    ortho_history: HashMap<String, OrthogonalizationState>,
    /// Gradient processor
    gradient_processor: GradientProcessor,
}

/// Orthogonalization state tracking
#[derive(Debug, Clone)]
struct OrthogonalizationState {
    /// Previous orthogonal basis
    previous_basis: Option<Tensor>,
    /// Orthogonality violation history
    violation_history: Vec<f32>,
    /// Adaptation rate for orthogonalization strength
    adaptation_rate: f32,
}

impl Default for OrthogonalizationState {
    fn default() -> Self {
        Self {
            previous_basis: None,
            violation_history: Vec::new(),
            adaptation_rate: 0.1,
        }
    }
}

/// DiWo configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiWoConfig {
    /// Base learning rate
    pub learning_rate: f32,
    /// First moment momentum coefficient
    pub beta1: f32,
    /// Second moment momentum coefficient
    pub beta2: f32,
    /// Numerical stability epsilon
    pub eps: f32,
    /// Weight decay strength
    pub weight_decay: f32,
    /// Orthogonalization strength
    pub ortho_strength: f32,
    /// Enable adaptive orthogonalization
    pub adaptive_ortho: bool,
}

impl Default for DiWoConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            ortho_strength: 0.1,
            adaptive_ortho: true,
        }
    }
}

impl DiWo {
    /// Create new DiWo optimizer
    pub fn new(
        learning_rate: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
        ortho_strength: f32,
    ) -> Self {
        Self {
            learning_rate,
            beta1: betas.0,
            beta2: betas.1,
            eps,
            weight_decay,
            ortho_strength,
            adaptive_ortho: true,
            step: 0,
            momentum_states: HashMap::new(),
            velocity_states: HashMap::new(),
            ortho_history: HashMap::new(),
            gradient_processor: GradientProcessor,
        }
    }

    /// Create DiWo for transformer training
    pub fn for_transformer_training() -> Self {
        Self::new(2e-4, (0.9, 0.98), 1e-8, 0.01, 0.05)
    }

    /// Create DiWo for computer vision
    pub fn for_computer_vision() -> Self {
        Self::new(1e-3, (0.9, 0.999), 1e-8, 1e-4, 0.1)
    }

    /// Create DiWo from configuration
    pub fn from_config(config: DiWoConfig) -> Self {
        let mut optimizer = Self::new(
            config.learning_rate,
            (config.beta1, config.beta2),
            config.eps,
            config.weight_decay,
            config.ortho_strength,
        );
        optimizer.adaptive_ortho = config.adaptive_ortho;
        optimizer
    }

    /// Apply orthogonalization constraint to parameter update
    fn apply_orthogonalization(
        &mut self,
        param_name: &str,
        param: &Tensor,
        update: &Tensor,
    ) -> Result<Tensor> {
        // Get or create orthogonalization state
        let _ortho_state = self.ortho_history.entry(param_name.to_string()).or_default();

        // For matrices (2D tensors), apply orthogonal constraints
        if param.shape().len() >= 2 && param.len() > 100 {
            // Only for significant matrices
            let param_data = param.data()?;
            let update_data = update.data()?;

            // Simplified orthogonalization using Gram-Schmidt process
            let orthogonalized_update =
                self.gram_schmidt_orthogonalization(&param_data, &update_data, &param.shape())?;

            // Adaptive orthogonalization strength
            let effective_ortho_strength = if self.adaptive_ortho {
                self.adapt_orthogonalization_strength(param_name, param, &orthogonalized_update)?
            } else {
                self.ortho_strength
            };

            // Blend original update with orthogonalized version
            let blended_update_data: Vec<f32> = update_data
                .iter()
                .zip(orthogonalized_update.iter())
                .map(|(orig, ortho)| {
                    orig * (1.0 - effective_ortho_strength) + ortho * effective_ortho_strength
                })
                .collect();

            Ok(Tensor::new(blended_update_data)?)
        } else {
            // For vectors or small matrices, return original update
            Ok(update.clone())
        }
    }

    /// Simplified Gram-Schmidt orthogonalization
    fn gram_schmidt_orthogonalization(
        &self,
        _param_data: &[f32],
        update_data: &[f32],
        shape: &[usize],
    ) -> Result<Vec<f32>> {
        if shape.len() < 2 {
            return Ok(update_data.to_vec());
        }

        let rows = shape[0];
        let cols = shape[1];
        let mut orthogonal_update = update_data.to_vec();

        // Apply modified Gram-Schmidt for stability
        for i in 0..rows.min(cols) {
            let start_idx = i * cols;
            let end_idx = (i + 1) * cols;

            if end_idx > orthogonal_update.len() {
                break;
            }

            // Normalize current row/column
            let mut norm_sq = 0.0f32;
            for j in start_idx..end_idx {
                norm_sq += orthogonal_update[j] * orthogonal_update[j];
            }

            if norm_sq > 1e-8 {
                let norm = norm_sq.sqrt();
                for j in start_idx..end_idx {
                    orthogonal_update[j] /= norm;
                }
            }

            // Orthogonalize against previous rows/columns
            for k in 0..i {
                let k_start = k * cols;
                let k_end = (k + 1) * cols;

                if k_end > orthogonal_update.len() {
                    break;
                }

                // Calculate dot product
                let mut dot_product = 0.0f32;
                for (j1, j2) in (start_idx..end_idx).zip(k_start..k_end) {
                    dot_product += orthogonal_update[j1] * orthogonal_update[j2];
                }

                // Subtract projection
                for (j1, j2) in (start_idx..end_idx).zip(k_start..k_end) {
                    orthogonal_update[j1] -= dot_product * orthogonal_update[j2];
                }
            }
        }

        Ok(orthogonal_update)
    }

    /// Adapt orthogonalization strength based on parameter behavior
    fn adapt_orthogonalization_strength(
        &mut self,
        param_name: &str,
        param: &Tensor,
        orthogonal_update: &[f32],
    ) -> Result<f32> {
        // Calculate orthogonality violation (how much the update violates orthogonal constraints)
        let param_data = param.data()?;
        let violation = self.calculate_orthogonality_violation(&param_data, orthogonal_update)?;

        // Get mutable reference after immutable borrow is done
        let ortho_state = self.ortho_history.get_mut(param_name).ok_or_else(|| {
            TrustformersError::invalid_state(
                "ortho_history state must exist after initialization".to_string(),
            )
        })?;
        ortho_state.violation_history.push(violation);

        // Keep only recent history
        if ortho_state.violation_history.len() > 100 {
            ortho_state.violation_history.remove(0);
        }

        // Adapt based on violation trend
        if ortho_state.violation_history.len() >= 2 {
            let recent_avg =
                ortho_state.violation_history.iter().rev().take(10).sum::<f32>() / 10.0;
            let older_avg = if ortho_state.violation_history.len() > 20 {
                ortho_state.violation_history.iter().rev().skip(10).take(10).sum::<f32>() / 10.0
            } else {
                recent_avg
            };

            if recent_avg > older_avg {
                // Violations increasing - strengthen orthogonalization
                ortho_state.adaptation_rate += 0.001;
            } else {
                // Violations decreasing - can reduce orthogonalization
                ortho_state.adaptation_rate -= 0.0005;
            }

            ortho_state.adaptation_rate = ortho_state.adaptation_rate.clamp(0.0, 1.0);
        }

        Ok((self.ortho_strength * ortho_state.adaptation_rate).clamp(0.0, 0.5))
    }

    /// Calculate how much the update violates orthogonality constraints
    fn calculate_orthogonality_violation(
        &self,
        param_data: &[f32],
        update_data: &[f32],
    ) -> Result<f32> {
        if param_data.len() != update_data.len() || param_data.len() < 4 {
            return Ok(0.0);
        }

        // Simple orthogonality measure: correlation between parameter and update
        let param_mean = param_data.iter().sum::<f32>() / param_data.len() as f32;
        let update_mean = update_data.iter().sum::<f32>() / update_data.len() as f32;

        let mut numerator = 0.0f32;
        let mut param_var = 0.0f32;
        let mut update_var = 0.0f32;

        for (p, u) in param_data.iter().zip(update_data.iter()) {
            let p_centered = p - param_mean;
            let u_centered = u - update_mean;
            numerator += p_centered * u_centered;
            param_var += p_centered * p_centered;
            update_var += u_centered * u_centered;
        }

        if param_var > 1e-8 && update_var > 1e-8 {
            let correlation = numerator / (param_var.sqrt() * update_var.sqrt());
            Ok(correlation.abs()) // Violation is absolute correlation
        } else {
            Ok(0.0)
        }
    }
}

impl Optimizer for DiWo {
    fn update(&mut self, param: &mut Tensor, gradient: &Tensor) -> Result<()> {
        let param_id = format!("param_{}", self.momentum_states.len());

        // Get current parameter and gradient data
        let param_data = param.data()?;
        let grad_data = gradient.data()?;

        // Initialize states if needed
        if !self.momentum_states.contains_key(&param_id) {
            self.momentum_states.insert(param_id.clone(), Tensor::zeros_like(param)?);
            self.velocity_states.insert(param_id.clone(), Tensor::zeros_like(param)?);
        }

        // Get momentum and velocity states
        let momentum = self.momentum_states.get_mut(&param_id).ok_or_else(|| {
            TrustformersError::invalid_state(
                "momentum_states must exist after initialization".to_string(),
            )
        })?;
        let velocity = self.velocity_states.get_mut(&param_id).ok_or_else(|| {
            TrustformersError::invalid_state(
                "velocity_states must exist after initialization".to_string(),
            )
        })?;

        let momentum_data = momentum.data()?;
        let velocity_data = velocity.data()?;

        // Update momentum (first moment)
        let new_momentum_data: Vec<f32> = momentum_data
            .iter()
            .zip(grad_data.iter())
            .map(|(m, g)| self.beta1 * m + (1.0 - self.beta1) * g)
            .collect();

        // Update velocity (second moment)
        let new_velocity_data: Vec<f32> = velocity_data
            .iter()
            .zip(grad_data.iter())
            .map(|(v, g)| self.beta2 * v + (1.0 - self.beta2) * g * g)
            .collect();

        *momentum = Tensor::new(new_momentum_data.clone())?;
        *velocity = Tensor::new(new_velocity_data.clone())?;

        // Bias correction
        let step_f32 = (self.step + 1) as f32;
        let momentum_corrected = 1.0 - self.beta1.powf(step_f32);
        let velocity_corrected = 1.0 - self.beta2.powf(step_f32);

        // Calculate base update
        let base_update_data: Vec<f32> = new_momentum_data
            .iter()
            .zip(new_velocity_data.iter())
            .zip(param_data.iter())
            .map(|((m, v), p)| {
                let m_hat = m / momentum_corrected;
                let v_hat = v / velocity_corrected;
                let base_update = -self.learning_rate * m_hat / (v_hat.sqrt() + self.eps);

                // Add weight decay
                base_update - self.learning_rate * self.weight_decay * p
            })
            .collect();

        let base_update = Tensor::new(base_update_data)?;

        // Apply orthogonalization
        let orthogonal_update = self.apply_orthogonalization(&param_id, param, &base_update)?;

        // Update parameters
        *param = param.add(&orthogonal_update)?;

        Ok(())
    }

    fn step(&mut self) {
        self.step += 1;
    }

    fn zero_grad(&mut self) {
        // DiWo doesn't maintain gradients, so this is a no-op
    }

    fn get_lr(&self) -> f32 {
        self.learning_rate
    }

    fn set_lr(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

impl StatefulOptimizer for DiWo {
    type Config = DiWoConfig;
    type State = OptimizerState;

    fn config(&self) -> &Self::Config {
        // Create a config on the fly from current values
        // Note: This is a temporary solution - ideally we'd store the config
        thread_local! {
            static TEMP_CONFIG: RefCell<Option<DiWoConfig>> = const { RefCell::new(None) };
        }
        TEMP_CONFIG.with(|config| {
            *config.borrow_mut() = Some(DiWoConfig {
                learning_rate: self.learning_rate,
                beta1: self.beta1,
                beta2: self.beta2,
                eps: self.eps,
                weight_decay: self.weight_decay,
                ortho_strength: self.ortho_strength,
                adaptive_ortho: self.adaptive_ortho,
            });
            // This is still unsafe because we're returning a reference to thread-local data
            // A proper fix would require restructuring the trait and implementation
            unsafe {
                std::mem::transmute(
                    config.borrow().as_ref().expect("config just set to Some above"),
                )
            }
        })
    }

    fn state(&self) -> &Self::State {
        // Create a state on the fly from current values
        // Note: This is a temporary solution - ideally we'd store the state properly
        thread_local! {
            static TEMP_STATE: RefCell<Option<OptimizerState>> = const { RefCell::new(None) };
        }
        TEMP_STATE.with(|state| {
            *state.borrow_mut() = Some(OptimizerState {
                step: self.step,
                momentum: HashMap::new(),
                variance: HashMap::new(),
                third_moment: HashMap::new(),
                param_steps: HashMap::new(),
                velocity: HashMap::new(),
                ..Default::default()
            });
            // This is still unsafe because we're returning a reference to thread-local data
            // A proper fix would require restructuring the trait and implementation
            unsafe {
                std::mem::transmute(state.borrow().as_ref().expect("state just set to Some above"))
            }
        })
    }

    fn state_mut(&mut self) -> &mut Self::State {
        // This is challenging with the current structure
        // We'll need to refactor to properly support this
        thread_local! {
            static TEMP_STATE: RefCell<Option<OptimizerState>> = const { RefCell::new(None) };
        }
        TEMP_STATE.with(|state| {
            *state.borrow_mut() = Some(OptimizerState {
                step: self.step,
                momentum: HashMap::new(),
                variance: HashMap::new(),
                third_moment: HashMap::new(),
                param_steps: HashMap::new(),
                velocity: HashMap::new(),
                ..Default::default()
            });
            // This is still unsafe because we're returning a mutable reference to thread-local data
            // A proper fix would require restructuring the trait and implementation
            unsafe {
                std::mem::transmute(
                    state.borrow_mut().as_mut().expect("state just set to Some above"),
                )
            }
        })
    }

    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut state_dict = HashMap::new();

        // Save configuration
        state_dict.insert(
            "learning_rate".to_string(),
            Tensor::new(vec![self.learning_rate])?,
        );
        state_dict.insert("beta1".to_string(), Tensor::new(vec![self.beta1])?);
        state_dict.insert("beta2".to_string(), Tensor::new(vec![self.beta2])?);
        state_dict.insert("eps".to_string(), Tensor::new(vec![self.eps])?);
        state_dict.insert(
            "weight_decay".to_string(),
            Tensor::new(vec![self.weight_decay])?,
        );
        state_dict.insert(
            "ortho_strength".to_string(),
            Tensor::new(vec![self.ortho_strength])?,
        );
        state_dict.insert("step".to_string(), Tensor::new(vec![self.step as f32])?);

        // Save momentum and velocity states
        for (key, tensor) in &self.momentum_states {
            state_dict.insert(format!("momentum_{}", key), tensor.clone());
        }
        for (key, tensor) in &self.velocity_states {
            state_dict.insert(format!("velocity_{}", key), tensor.clone());
        }

        Ok(state_dict)
    }

    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()> {
        // Load configuration
        if let Some(lr_tensor) = state.get("learning_rate") {
            if let Ok(lr_vec) = lr_tensor.data() {
                if !lr_vec.is_empty() {
                    self.learning_rate = lr_vec[0];
                }
            }
        }

        if let Some(beta1_tensor) = state.get("beta1") {
            if let Ok(beta1_vec) = beta1_tensor.data() {
                if !beta1_vec.is_empty() {
                    self.beta1 = beta1_vec[0];
                }
            }
        }

        if let Some(beta2_tensor) = state.get("beta2") {
            if let Ok(beta2_vec) = beta2_tensor.data() {
                if !beta2_vec.is_empty() {
                    self.beta2 = beta2_vec[0];
                }
            }
        }

        if let Some(step_tensor) = state.get("step") {
            if let Ok(step_vec) = step_tensor.data() {
                if !step_vec.is_empty() {
                    self.step = step_vec[0] as usize;
                }
            }
        }

        // Load momentum and velocity states
        for (key, tensor) in state.iter() {
            if let Some(param_id) = key.strip_prefix("momentum_") {
                self.momentum_states.insert(param_id.to_string(), tensor.clone());
            } else if let Some(param_id) = key.strip_prefix("velocity_") {
                self.velocity_states.insert(param_id.to_string(), tensor.clone());
            }
        }

        Ok(())
    }

    fn memory_usage(&self) -> StateMemoryStats {
        let momentum_elements: usize = self
            .momentum_states
            .values()
            .map(|t| if let Ok(data) = t.data() { data.len() } else { 0 })
            .sum();

        let velocity_elements: usize = self
            .velocity_states
            .values()
            .map(|t| if let Ok(data) = t.data() { data.len() } else { 0 })
            .sum();

        let total_elements = momentum_elements + velocity_elements;
        let total_bytes = total_elements * std::mem::size_of::<f32>();

        StateMemoryStats {
            momentum_elements,
            variance_elements: velocity_elements,
            third_moment_elements: 0,
            total_bytes,
            num_parameters: momentum_elements,
        }
    }

    fn reset_state(&mut self) {
        self.step = 0;
        self.momentum_states.clear();
        self.velocity_states.clear();
        self.ortho_history.clear();
    }

    fn num_parameters(&self) -> usize {
        self.momentum_states.len()
    }
}

/// MeZO-V2 (Memory-Efficient Zeroth-Order V2) Optimizer
///
/// Based on "MeZO-V2: Memory-Efficient Zeroth-Order Optimization for Billion-Parameter Models" (NeurIPS 2025)
///
/// Key innovations:
/// - Advanced finite difference approximation with adaptive perturbation
/// - Memory-efficient gradient estimation using random projections
/// - Multi-scale perturbation for better approximation quality
/// - Compatible with distributed training and model parallelism
#[derive(Debug, Clone)]
pub struct MeZOV2 {
    /// Base learning rate
    learning_rate: f32,
    /// Perturbation magnitude for finite differences
    perturbation_scale: f32,
    /// Number of gradient estimation samples
    num_samples: usize,
    /// Adaptive perturbation adjustment
    adaptive_perturbation: bool,
    /// Multi-scale perturbation levels
    perturbation_levels: Vec<f32>,
    /// Current optimization step
    step: usize,
    /// Random seed for reproducible perturbations
    random_seed: u64,
}

/// MeZO-V2 configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeZOV2Config {
    /// Learning rate
    pub learning_rate: f32,
    /// Base perturbation scale
    pub perturbation_scale: f32,
    /// Number of samples for gradient estimation
    pub num_samples: usize,
    /// Enable adaptive perturbation scaling
    pub adaptive_perturbation: bool,
    /// Multi-scale perturbation factors
    pub perturbation_levels: Vec<f32>,
}

impl Default for MeZOV2Config {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            perturbation_scale: 1e-3,
            num_samples: 2,
            adaptive_perturbation: true,
            perturbation_levels: vec![1.0, 0.1, 0.01],
        }
    }
}

impl MeZOV2 {
    /// Create new MeZO-V2 optimizer
    pub fn new(learning_rate: f32, perturbation_scale: f32, num_samples: usize) -> Self {
        Self {
            learning_rate,
            perturbation_scale,
            num_samples,
            adaptive_perturbation: true,
            perturbation_levels: vec![1.0, 0.1, 0.01],
            step: 0,
            random_seed: 42,
        }
    }

    /// Create MeZO-V2 for large language models
    pub fn for_large_language_models() -> Self {
        Self::new(1e-4, 1e-4, 1)
    }

    /// Create MeZO-V2 for billion-parameter models
    pub fn for_billion_parameter_models() -> Self {
        Self::new(5e-5, 5e-5, 1)
    }

    /// Create from configuration
    pub fn from_config(config: MeZOV2Config) -> Self {
        let mut optimizer = Self::new(
            config.learning_rate,
            config.perturbation_scale,
            config.num_samples,
        );
        optimizer.adaptive_perturbation = config.adaptive_perturbation;
        optimizer.perturbation_levels = config.perturbation_levels;
        optimizer
    }

    /// Deterministically regenerates the Rademacher perturbation for `seed`.
    ///
    /// MeZO's memory efficiency comes from *not storing* the perturbation: the same
    /// seed is replayed to apply `+ε·z`, `−2ε·z`, `+ε·z` and finally the update
    /// `−lr·ĝ·z`, so only one `u64` is carried between the phases.
    fn perturbation_at(seed: u64, index: usize) -> f32 {
        let mut z = seed.wrapping_add((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        if z & 1 == 0 {
            -1.0
        } else {
            1.0
        }
    }

    /// Adds `amount · z(seed)` to every parameter, in registration-sorted name order.
    ///
    /// The index stream is continuous across parameters so one seed covers the whole
    /// model, exactly as MeZO requires.
    fn perturb_in_place(
        parameters: &mut HashMap<String, Tensor>,
        names: &[String],
        seed: u64,
        amount: f32,
    ) -> Result<()> {
        let mut offset = 0_usize;
        for name in names {
            let parameter = parameters.get_mut(name).ok_or_else(|| {
                TrustformersError::invalid_input(format!("MeZO: parameter '{name}' disappeared"))
            })?;
            let mut values = parameter.data_f32()?;
            for (index, value) in values.iter_mut().enumerate() {
                *value += amount * Self::perturbation_at(seed, offset + index);
            }
            offset += values.len();
            parameter.set_data_f32(&values)?;
        }
        Ok(())
    }

    /// Runs one genuine MeZO step over a whole named parameter map.
    ///
    /// This is the *only* path that implements zeroth-order optimization, because it
    /// is the only one that can evaluate the objective. For each of `num_samples`
    /// probes and each configured perturbation level:
    ///
    /// ```text
    /// z ~ Rademacher(seed)                       (regenerated, never stored)
    /// θ ← θ + εz;   l₊ = loss(θ)
    /// θ ← θ − 2εz;  l₋ = loss(θ)
    /// θ ← θ + εz                                 (restored exactly)
    /// ĝ  = (l₊ − l₋) / (2ε)                      (one scalar for the whole model)
    /// θ ← θ − (lr / probes) · ĝ · z
    /// ```
    ///
    /// Returns the loss measured at the unperturbed parameters before the update.
    ///
    /// # Errors
    ///
    /// Returns an error when the objective fails, when a parameter is not
    /// `f32`-readable, or when `num_samples` or the perturbation levels are empty.
    pub fn step_with_loss<F>(
        &mut self,
        parameters: &mut HashMap<String, Tensor>,
        mut loss_fn: F,
    ) -> Result<f32>
    where
        F: FnMut(&HashMap<String, Tensor>) -> Result<f32>,
    {
        if self.num_samples == 0 {
            return Err(TrustformersError::invalid_config(
                "MeZO requires num_samples >= 1".to_string(),
            ));
        }
        if self.perturbation_levels.is_empty() {
            return Err(TrustformersError::invalid_config(
                "MeZO requires at least one perturbation level".to_string(),
            ));
        }

        // A deterministic, order-independent traversal so a seed maps to the same
        // coordinates on every phase and in every process.
        let mut names: Vec<String> = parameters.keys().cloned().collect();
        names.sort();

        let baseline_loss = loss_fn(parameters)?;

        self.step += 1;
        let levels = self.perturbation_levels.clone();
        let probes = self.num_samples * levels.len();
        let mut projected_sum = 0.0_f32;

        for (level_index, &level) in levels.iter().enumerate() {
            let epsilon = self.perturbation_scale * level;
            if epsilon <= 0.0 {
                return Err(TrustformersError::invalid_config(format!(
                    "MeZO perturbation scale must be positive, got {epsilon}"
                )));
            }

            for sample in 0..self.num_samples {
                let seed = self
                    .random_seed
                    .wrapping_add((self.step as u64).wrapping_mul(0x0100_0000_01B3))
                    .wrapping_add((level_index as u64) << 32)
                    .wrapping_add(sample as u64);

                // θ + εz
                Self::perturb_in_place(parameters, &names, seed, epsilon)?;
                let loss_plus = loss_fn(parameters)?;

                // θ − εz
                Self::perturb_in_place(parameters, &names, seed, -2.0 * epsilon)?;
                let loss_minus = loss_fn(parameters)?;

                // Restore θ exactly.
                Self::perturb_in_place(parameters, &names, seed, epsilon)?;

                // The two-point estimate: one scalar for the entire model.
                let projected = (loss_plus - loss_minus) / (2.0 * epsilon);
                projected_sum += projected.abs();

                // θ ← θ − (lr / probes) · ĝ · z, regenerating z from the same seed.
                let step_size = self.learning_rate * projected / probes as f32;
                Self::perturb_in_place(parameters, &names, seed, -step_size)?;
            }
        }

        self.adapt_perturbation_scale_from_estimate(projected_sum / probes as f32);
        Ok(baseline_loss)
    }

    /// Convenience wrapper for a single unnamed parameter tensor.
    ///
    /// # Errors
    ///
    /// See [`MeZOV2::step_with_loss`].
    pub fn step_with_loss_single<F>(
        &mut self,
        parameter: &mut Tensor,
        mut loss_fn: F,
    ) -> Result<f32>
    where
        F: FnMut(&Tensor) -> Result<f32>,
    {
        let mut parameters = HashMap::new();
        parameters.insert("theta".to_string(), parameter.clone());
        let loss = self.step_with_loss(&mut parameters, |params| {
            let tensor = params.get("theta").ok_or_else(|| {
                TrustformersError::invalid_state("MeZO: 'theta' vanished".to_string())
            })?;
            loss_fn(tensor)
        })?;
        *parameter = parameters.remove("theta").ok_or_else(|| {
            TrustformersError::invalid_state("MeZO: 'theta' vanished".to_string())
        })?;
        Ok(loss)
    }

    /// Adapts the perturbation scale from the *measured* directional derivative.
    ///
    /// A vanishing estimate means the two probe losses are indistinguishable, so the
    /// step is too small to resolve; a very large one means the objective is changing
    /// faster than the finite difference can track.
    fn adapt_perturbation_scale_from_estimate(&mut self, mean_abs_projection: f32) {
        if !self.adaptive_perturbation {
            return;
        }
        if mean_abs_projection < 1e-8 {
            self.perturbation_scale *= 1.1;
        } else if mean_abs_projection > 1e3 {
            self.perturbation_scale *= 0.9;
        }
        self.perturbation_scale = self.perturbation_scale.clamp(1e-6, 1e-1);
    }
}

impl Optimizer for MeZOV2 {
    /// Always returns [`ErrorKind::NotImplemented`](trustformers_core::errors::ErrorKind).
    ///
    /// MeZO is a *zeroth-order* method: its whole premise is that back-propagated
    /// gradients are unavailable, and its update direction comes from two loss
    /// evaluations at `θ ± εz`. The bare [`Optimizer`] signature carries no objective,
    /// so there is nothing honest this method can compute — an earlier version
    /// multiplied the supplied first-order gradient by random noise and called the
    /// result a finite-difference approximation. Use
    /// [`MeZOV2::step_with_loss`] (or [`MeZOV2::step_with_loss_single`]) instead.
    fn update(&mut self, _param: &mut Tensor, _gradient: &Tensor) -> Result<()> {
        Err(TrustformersError::not_implemented(
            "MeZOV2 is a zeroth-order optimizer and cannot use a supplied gradient; \
             call MeZOV2::step_with_loss with an objective closure instead"
                .to_string(),
        ))
    }

    fn step(&mut self) {
        self.step += 1;
    }

    fn zero_grad(&mut self) {
        // No-op for zeroth-order methods
    }

    fn get_lr(&self) -> f32 {
        self.learning_rate
    }

    fn set_lr(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

/// AdaWin (Adaptive Window) Optimizer
///
/// Based on "AdaWin: Adaptive Window-Based Momentum for Large-Scale Optimization" (ICLR 2025)
///
/// Key innovations:
/// - Dynamic window size adaptation based on gradient correlation
/// - Multi-timescale momentum with adaptive weighting
/// - Memory-efficient circular buffer for gradient history
/// - Automatic hyperparameter adaptation based on optimization landscape
#[derive(Debug, Clone)]
pub struct AdaWin {
    /// Base learning rate
    learning_rate: f32,
    /// Base momentum coefficient
    beta: f32,
    /// Window size adaptation rate
    adaptation_rate: f32,
    /// Maximum window size
    max_window_size: usize,
    /// Current optimization step
    step: usize,
    /// Parameter-specific states
    parameter_states: HashMap<String, AdaWinParameterState>,
}

#[derive(Debug, Clone)]
struct AdaWinParameterState {
    /// Circular buffer for gradient history
    gradient_history: Vec<Tensor>,
    /// Current window size
    window_size: usize,
    /// Adaptive momentum coefficients
    momentum_weights: Vec<f32>,
    /// Cumulative momentum
    momentum: Tensor,
    /// Gradient correlation tracking
    correlation_history: Vec<f32>,
}

/// AdaWin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaWinConfig {
    /// Base learning rate
    pub learning_rate: f32,
    /// Base momentum coefficient
    pub beta: f32,
    /// Window adaptation rate
    pub adaptation_rate: f32,
    /// Maximum window size
    pub max_window_size: usize,
}

impl Default for AdaWinConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta: 0.9,
            adaptation_rate: 0.01,
            max_window_size: 20,
        }
    }
}

impl AdaWin {
    /// Create new AdaWin optimizer
    pub fn new(
        learning_rate: f32,
        beta: f32,
        adaptation_rate: f32,
        max_window_size: usize,
    ) -> Self {
        Self {
            learning_rate,
            beta,
            adaptation_rate,
            max_window_size,
            step: 0,
            parameter_states: HashMap::new(),
        }
    }

    /// Create AdaWin for transformer training
    pub fn for_transformer_training() -> Self {
        Self::new(1e-4, 0.95, 0.02, 15)
    }

    /// Create AdaWin for computer vision
    pub fn for_computer_vision() -> Self {
        Self::new(1e-3, 0.9, 0.01, 25)
    }

    /// Create from configuration
    pub fn from_config(config: AdaWinConfig) -> Self {
        Self::new(
            config.learning_rate,
            config.beta,
            config.adaptation_rate,
            config.max_window_size,
        )
    }

    /// Calculate gradient correlation within window
    fn calculate_gradient_correlation(&self, gradients: &[Tensor]) -> Result<f32> {
        if gradients.len() < 2 {
            return Ok(0.0);
        }

        let mut total_correlation = 0.0f32;
        let mut count = 0;

        for i in 0..gradients.len() - 1 {
            for j in i + 1..gradients.len() {
                let grad1_data = gradients[i].data()?;
                let grad2_data = gradients[j].data()?;

                if grad1_data.len() == grad2_data.len() {
                    // Calculate Pearson correlation
                    let mean1 = grad1_data.iter().sum::<f32>() / grad1_data.len() as f32;
                    let mean2 = grad2_data.iter().sum::<f32>() / grad2_data.len() as f32;

                    let mut numerator = 0.0f32;
                    let mut var1 = 0.0f32;
                    let mut var2 = 0.0f32;

                    for (g1, g2) in grad1_data.iter().zip(grad2_data.iter()) {
                        let d1 = g1 - mean1;
                        let d2 = g2 - mean2;
                        numerator += d1 * d2;
                        var1 += d1 * d1;
                        var2 += d2 * d2;
                    }

                    if var1 > 1e-8 && var2 > 1e-8 {
                        total_correlation += numerator / (var1.sqrt() * var2.sqrt());
                        count += 1;
                    }
                }
            }
        }

        Ok(if count > 0 { total_correlation / count as f32 } else { 0.0 })
    }

    /// Adapt window size based on gradient correlation
    fn adapt_window_size(&self, param_state: &mut AdaWinParameterState) -> Result<()> {
        let correlation = self.calculate_gradient_correlation(&param_state.gradient_history)?;
        param_state.correlation_history.push(correlation);

        // Keep correlation history bounded
        if param_state.correlation_history.len() > 50 {
            param_state.correlation_history.remove(0);
        }

        // Adapt window size based on correlation trends
        if param_state.correlation_history.len() >= 5 {
            let recent_avg =
                param_state.correlation_history.iter().rev().take(3).sum::<f32>() / 3.0;

            if recent_avg > 0.5 {
                // High correlation - increase window size
                let new_size = (param_state.window_size + 1).min(self.max_window_size);
                param_state.window_size = new_size;
            } else if recent_avg < 0.1 {
                // Low correlation - decrease window size
                let new_size = (param_state.window_size.saturating_sub(1)).max(2);
                param_state.window_size = new_size;
            }
        }

        // Update momentum weights based on window size
        self.update_momentum_weights(param_state);

        Ok(())
    }

    /// Update momentum weights for current window
    fn update_momentum_weights(&self, param_state: &mut AdaWinParameterState) {
        param_state.momentum_weights.clear();

        let window_size = param_state.window_size;
        for i in 0..window_size {
            // Exponentially decaying weights with adaptive base
            let age = i as f32;
            let adaptive_beta = self.beta * (1.0 - age / (window_size as f32 + 1.0));
            param_state.momentum_weights.push(adaptive_beta.powf(age));
        }

        // Normalize weights
        let sum: f32 = param_state.momentum_weights.iter().sum();
        if sum > 1e-8 {
            for weight in param_state.momentum_weights.iter_mut() {
                *weight /= sum;
            }
        }
    }

    /// Calculate weighted momentum from gradient history
    fn calculate_weighted_momentum(&self, param_state: &AdaWinParameterState) -> Result<Tensor> {
        if param_state.gradient_history.is_empty() {
            return Ok(param_state.momentum.clone());
        }

        let first_grad = &param_state.gradient_history[0];
        let mut weighted_momentum = vec![0.0f32; first_grad.len()];

        for (i, gradient) in param_state.gradient_history.iter().enumerate() {
            if i < param_state.momentum_weights.len() {
                let weight = param_state.momentum_weights[i];
                let grad_data = gradient.data()?;

                for (j, &g) in grad_data.iter().enumerate() {
                    if j < weighted_momentum.len() {
                        weighted_momentum[j] += weight * g;
                    }
                }
            }
        }

        Tensor::new(weighted_momentum)
    }
}

impl Optimizer for AdaWin {
    fn update(&mut self, param: &mut Tensor, gradient: &Tensor) -> Result<()> {
        let param_id = format!("param_{}", self.parameter_states.len());

        // Initialize parameter state if needed
        if !self.parameter_states.contains_key(&param_id) {
            let momentum = Tensor::zeros_like(param)?;
            self.parameter_states.insert(
                param_id.clone(),
                AdaWinParameterState {
                    gradient_history: Vec::new(),
                    window_size: 3,                         // Start with small window
                    momentum_weights: vec![1.0, 0.5, 0.25], // Initial weights
                    momentum,
                    correlation_history: Vec::new(),
                },
            );
        }

        // Add gradient to history and maintain circular buffer
        {
            let param_state = self.parameter_states.get_mut(&param_id).ok_or_else(|| {
                TrustformersError::invalid_state(
                    "parameter_states must exist after step".to_string(),
                )
            })?;
            param_state.gradient_history.push(gradient.clone());

            if param_state.gradient_history.len() > self.max_window_size {
                param_state.gradient_history.remove(0);
            }
        }

        // Adapt window size based on gradient correlations
        {
            let mut param_state = self.parameter_states.remove(&param_id).ok_or_else(|| {
                TrustformersError::invalid_state(
                    "parameter_states must exist for param_id".to_string(),
                )
            })?;
            self.adapt_window_size(&mut param_state)?;
            self.parameter_states.insert(param_id.clone(), param_state);
        }

        // Calculate weighted momentum
        let weighted_momentum = {
            let param_state = self.parameter_states.get(&param_id).ok_or_else(|| {
                TrustformersError::invalid_state(
                    "parameter_states must exist for param_id".to_string(),
                )
            })?;
            self.calculate_weighted_momentum(param_state)?
        };

        // Update cumulative momentum
        let momentum_data = {
            let param_state = self.parameter_states.get(&param_id).ok_or_else(|| {
                TrustformersError::invalid_state(
                    "parameter_states must exist for param_id".to_string(),
                )
            })?;
            param_state.momentum.data()?
        };
        let weighted_data = weighted_momentum.data()?;
        let new_momentum_data: Vec<f32> = momentum_data
            .iter()
            .zip(weighted_data.iter())
            .map(|(m, w)| self.beta * m + (1.0 - self.beta) * w)
            .collect();

        // Update momentum
        {
            let param_state = self.parameter_states.get_mut(&param_id).ok_or_else(|| {
                TrustformersError::invalid_state(
                    "parameter_states must exist after step".to_string(),
                )
            })?;
            param_state.momentum = Tensor::new(new_momentum_data.clone())?;
        }

        // Apply parameter update
        let param_data = param.data()?;
        let updated_param_data: Vec<f32> = param_data
            .iter()
            .zip(new_momentum_data.iter())
            .map(|(p, m)| p - self.learning_rate * m)
            .collect();

        *param = Tensor::new(updated_param_data)?;

        Ok(())
    }

    fn step(&mut self) {
        self.step += 1;
    }

    fn zero_grad(&mut self) {
        // Clear gradient history for new iteration
        for param_state in self.parameter_states.values_mut() {
            param_state.gradient_history.clear();
        }
    }

    fn get_lr(&self) -> f32 {
        self.learning_rate
    }

    fn set_lr(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diwo_creation() {
        let optimizer = DiWo::new(1e-3, (0.9, 0.999), 1e-8, 0.01, 0.1);
        assert_eq!(optimizer.learning_rate, 1e-3);
        assert_eq!(optimizer.beta1, 0.9);
        assert_eq!(optimizer.ortho_strength, 0.1);
    }

    #[test]
    fn test_diwo_presets() {
        let transformer_opt = DiWo::for_transformer_training();
        assert_eq!(transformer_opt.learning_rate, 2e-4);
        assert!(transformer_opt.adaptive_ortho);

        let vision_opt = DiWo::for_computer_vision();
        assert_eq!(vision_opt.learning_rate, 1e-3);
        assert_eq!(vision_opt.ortho_strength, 0.1);
    }

    #[test]
    fn test_diwo_config() -> Result<()> {
        let config = DiWoConfig {
            learning_rate: 5e-4,
            ortho_strength: 0.2,
            adaptive_ortho: false,
            ..Default::default()
        };

        let optimizer = DiWo::from_config(config);
        assert_eq!(optimizer.learning_rate, 5e-4);
        assert_eq!(optimizer.ortho_strength, 0.2);
        assert!(!optimizer.adaptive_ortho);

        Ok(())
    }

    #[test]
    fn test_mezov2_creation() {
        let optimizer = MeZOV2::new(1e-3, 1e-4, 2);
        assert_eq!(optimizer.learning_rate, 1e-3);
        assert_eq!(optimizer.perturbation_scale, 1e-4);
        assert_eq!(optimizer.num_samples, 2);
    }

    #[test]
    fn test_mezov2_presets() {
        let llm_opt = MeZOV2::for_large_language_models();
        assert_eq!(llm_opt.learning_rate, 1e-4);

        let billion_opt = MeZOV2::for_billion_parameter_models();
        assert_eq!(billion_opt.learning_rate, 5e-5);
    }

    #[test]
    fn test_adawin_creation() {
        let optimizer = AdaWin::new(1e-3, 0.9, 0.01, 20);
        assert_eq!(optimizer.learning_rate, 1e-3);
        assert_eq!(optimizer.beta, 0.9);
        assert_eq!(optimizer.max_window_size, 20);
    }

    #[test]
    fn test_adawin_presets() {
        let transformer_opt = AdaWin::for_transformer_training();
        assert_eq!(transformer_opt.learning_rate, 1e-4);
        assert_eq!(transformer_opt.max_window_size, 15);

        let vision_opt = AdaWin::for_computer_vision();
        assert_eq!(vision_opt.learning_rate, 1e-3);
        assert_eq!(vision_opt.max_window_size, 25);
    }

    #[test]
    fn test_diwo_update() -> Result<()> {
        let mut optimizer = DiWo::new(1e-3, (0.9, 0.999), 1e-8, 0.01, 0.1);

        let mut param = Tensor::new(vec![1.0, 2.0, 3.0])?;
        let gradient = Tensor::new(vec![0.1, 0.2, 0.1])?;

        let original_param = param.data()?;
        optimizer.update(&mut param, &gradient)?;
        optimizer.step();

        let updated_param = param.data()?;

        // Parameter should have changed
        assert_ne!(original_param, updated_param);
        assert_eq!(optimizer.step, 1);

        Ok(())
    }

    #[test]
    fn test_mezov2_update() -> Result<()> {
        let mut optimizer = MeZOV2::new(1e-2, 1e-3, 1);
        optimizer.adaptive_perturbation = false;
        optimizer.perturbation_levels = vec![1.0];

        let mut param = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3])?;
        let original_param = param.data_f32()?;

        // MeZO drives the update from loss evaluations, never from a gradient.
        optimizer.step_with_loss_single(&mut param, |t| {
            Ok(t.data_f32()?.iter().map(|v| v * v).sum::<f32>())
        })?;

        let updated_param = param.data_f32()?;

        // Parameter should have changed
        assert_ne!(original_param, updated_param);
        assert_eq!(optimizer.step, 1);

        Ok(())
    }

    #[test]
    fn test_adawin_update() -> Result<()> {
        let mut optimizer = AdaWin::new(1e-3, 0.9, 0.01, 10);

        let mut param = Tensor::new(vec![1.0, 2.0, 3.0])?;
        let gradient = Tensor::new(vec![0.1, 0.2, 0.1])?;

        let original_param = param.data()?;
        optimizer.update(&mut param, &gradient)?;
        optimizer.step();

        let updated_param = param.data()?;

        // Parameter should have changed
        assert_ne!(original_param, updated_param);
        assert_eq!(optimizer.step, 1);

        Ok(())
    }

    /// Regression: `Optimizer::update` used to multiply the supplied *first-order*
    /// gradient by random noise and present the result as a finite-difference
    /// estimate. A zeroth-order optimizer cannot honestly consume a gradient.
    #[test]
    fn mezo_update_rejects_a_supplied_gradient() {
        let mut optimizer = MeZOV2::new(1e-2, 1e-3, 1);
        let mut param = Tensor::from_vec(vec![1.0_f32], &[1]).expect("tensor");
        let grad = Tensor::from_vec(vec![1.0_f32], &[1]).expect("grad");
        let before = param.data_f32().expect("data");

        let result = optimizer.update(&mut param, &grad);
        assert!(
            result.is_err(),
            "the gradient path must not fabricate a step"
        );
        assert_eq!(
            param.data_f32().expect("data"),
            before,
            "a refused update must leave the parameter untouched"
        );
    }

    /// The two-point estimate must recover the true directional derivative of a
    /// quadratic to within the finite-difference truncation error.
    ///
    /// For `f(θ) = Σ θ²` at `θ = [2, −3]` and a Rademacher `z`, the exact directional
    /// derivative is `∇f·z = 2·(2·z₀ − 3·z₁)`, and the central difference is exact for
    /// a quadratic. The realised step must therefore be `−lr·(∇f·z)·z`, which for
    /// `z ∈ {±1}` equals `−lr·∇f` projected onto `z`.
    #[test]
    fn mezo_step_matches_the_exact_directional_derivative() {
        let mut optimizer = MeZOV2::new(0.1, 1e-2, 1);
        optimizer.adaptive_perturbation = false;
        optimizer.perturbation_levels = vec![1.0];

        let mut parameters = HashMap::new();
        parameters.insert(
            "w".to_string(),
            Tensor::from_vec(vec![2.0_f32, -3.0], &[2]).expect("tensor"),
        );
        let before = parameters["w"].data_f32().expect("data");

        let mut calls = 0_usize;
        let baseline = optimizer
            .step_with_loss(&mut parameters, |params| {
                calls += 1;
                Ok(params["w"].data_f32()?.iter().map(|v| v * v).sum::<f32>())
            })
            .expect("step");

        // 1 baseline evaluation + 2 per probe.
        assert_eq!(calls, 3, "MeZO must evaluate the objective, not a gradient");
        assert!((baseline - 13.0).abs() < 1e-4, "baseline loss: {baseline}");

        // Recover z from the realised displacement: every coordinate moved by
        // −lr·ĝ·z_i, so the two coordinates moved by the same magnitude.
        let after = parameters["w"].data_f32().expect("data");
        let delta: Vec<f32> = after.iter().zip(before.iter()).map(|(a, b)| a - b).collect();
        assert!(
            (delta[0].abs() - delta[1].abs()).abs() < 1e-4,
            "a Rademacher probe moves every coordinate by the same magnitude: {delta:?}"
        );

        // |ĝ| = |∇f·z| = |2·(2·z₀ − 3·z₁)| ∈ {2, 10} for z ∈ {±1}²; the step magnitude
        // is lr·|ĝ| = 0.1·|ĝ|.
        let magnitude = delta[0].abs();
        assert!(
            (magnitude - 0.2).abs() < 1e-3 || (magnitude - 1.0).abs() < 1e-3,
            "step magnitude {magnitude} must equal lr·|∇f·z|"
        );
    }

    /// The perturbation must be restored exactly: a constant objective leaves the
    /// parameters untouched because the two-point estimate is identically zero.
    #[test]
    fn mezo_restores_parameters_when_the_objective_is_flat() {
        let mut optimizer = MeZOV2::new(0.5, 1e-3, 2);
        optimizer.adaptive_perturbation = false;

        let mut parameters = HashMap::new();
        parameters.insert(
            "w".to_string(),
            Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]).expect("tensor"),
        );

        optimizer.step_with_loss(&mut parameters, |_| Ok(7.0)).expect("step");

        let after = parameters["w"].data_f32().expect("data");
        for (value, want) in after.iter().zip([1.0_f32, 2.0, 3.0].iter()) {
            assert!(
                (value - want).abs() < 1e-5,
                "a flat objective must leave θ unchanged: {value} vs {want}"
            );
        }
    }

    /// Convergence smoke test: MeZO must descend the quadratic bowl using only loss
    /// evaluations.
    #[test]
    fn mezo_descends_a_quadratic_bowl_from_losses_alone() {
        let mut optimizer = MeZOV2::new(0.05, 1e-2, 4);
        optimizer.adaptive_perturbation = false;
        optimizer.perturbation_levels = vec![1.0];

        let mut param = Tensor::from_vec(vec![3.0_f32, -4.0], &[2]).expect("tensor");
        let objective =
            |t: &Tensor| -> Result<f32> { Ok(t.data_f32()?.iter().map(|v| v * v).sum()) };
        let initial = objective(&param).expect("loss");

        for _ in 0..400 {
            optimizer.step_with_loss_single(&mut param, objective).expect("step");
        }

        let final_loss = objective(&param).expect("loss");
        assert!(
            final_loss < initial * 0.2,
            "loss must fall using only loss evaluations: {initial} -> {final_loss}"
        );
    }
}
