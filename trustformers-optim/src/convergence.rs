//! # Convergence Improvement Methods
//!
//! This module implements advanced optimization techniques that improve convergence
//! speed, stability, and final performance through sophisticated momentum variants
//! and variance reduction methods.
//!
//! ## Available Methods
//!
//! - **QHM (Quasi-Hyperbolic Momentum)**: Generalizes momentum and Nesterov acceleration
//! - **AggMo (Aggregated Momentum)**: Maintains multiple momentum buffers for better convergence
//! - **SVRG (Stochastic Variance Reduced Gradient)**: Reduces gradient variance for better convergence
//! - **SAG (Stochastic Average Gradient)**: Maintains running average of gradients
//! - **Nesterov Accelerated Gradient (NAG)**: Classical acceleration method with lookahead
//! - **Heavy Ball Method**: Momentum-based acceleration with inertia
//! - **FISTA**: Fast Iterative Shrinkage-Thresholding Algorithm for proximal methods
//! - **Adaptive Batch Sizing**: Dynamically adjusts batch size based on training progress
//! - **Loss Surface Smoothing**: Reduces noise in the loss surface for better convergence

use crate::optimizer::OptimizerState;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::tensor::Tensor;

/// Configuration for Quasi-Hyperbolic Momentum (QHM).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QHMConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Momentum parameter (β)
    pub momentum: f32,
    /// Averaging parameter (ν) - controls interpolation between current gradient and momentum
    pub nu: f32,
    /// Weight decay
    pub weight_decay: f32,
}

impl Default for QHMConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            momentum: 0.9,
            nu: 0.7,
            weight_decay: 0.0,
        }
    }
}

/// Quasi-Hyperbolic Momentum optimizer.
///
/// QHM interpolates between the current gradient and the momentum buffer,
/// providing a generalization of both momentum and Nesterov acceleration.
/// Update rule: p = p - lr * (nu * g + (1 - nu) * momentum)
#[derive(Debug)]
pub struct QHM {
    config: QHMConfig,
    momentum_buffers: HashMap<usize, Tensor>,
    current_step: usize,
}

impl QHM {
    /// Create a new QHM optimizer.
    pub fn new(config: QHMConfig) -> Self {
        Self {
            config,
            momentum_buffers: HashMap::new(),
            current_step: 0,
        }
    }

    /// Create QHM with default configuration.
    pub fn with_defaults(learning_rate: f32, momentum: f32, nu: f32) -> Self {
        Self::new(QHMConfig {
            learning_rate,
            momentum,
            nu,
            weight_decay: 0.0,
        })
    }

    /// Get the configuration.
    pub fn get_config(&self) -> &QHMConfig {
        &self.config
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: QHMConfig) {
        self.config = config;
    }
}

impl OptimizerState for QHM {
    fn zero_grad(&mut self) -> Result<()> {
        // QHM doesn't need explicit gradient zeroing
        Ok(())
    }

    fn step(&mut self, parameters: &mut [Tensor]) -> Result<()> {
        self.current_step += 1;

        for (param_id, parameter) in parameters.iter_mut().enumerate() {
            // Access gradient from parameter (should be computed during forward/backward pass)
            let gradient = match parameter.grad() {
                Ok(grad) => grad,
                Err(_) => {
                    // If gradient is not available, skip this parameter
                    continue;
                },
            };

            // Apply weight decay to gradient
            let effective_grad = if self.config.weight_decay > 0.0 {
                gradient.add(&parameter.mul_scalar(self.config.weight_decay)?)?
            } else {
                gradient
            };

            // Get or initialize momentum buffer
            let momentum_buffer = if let Some(buffer) = self.momentum_buffers.get(&param_id) {
                // Update momentum: momentum = β * momentum + (1 - β) * grad
                let updated = buffer
                    .mul_scalar(self.config.momentum)?
                    .add(&effective_grad.mul_scalar(1.0 - self.config.momentum)?)?;
                self.momentum_buffers.insert(param_id, updated.clone());
                updated
            } else {
                // Initialize momentum buffer with current gradient
                let initial_momentum = effective_grad.clone();
                self.momentum_buffers.insert(param_id, initial_momentum.clone());
                initial_momentum
            };

            // QHM update: interpolate between current gradient and momentum
            let update_direction = effective_grad
                .mul_scalar(self.config.nu)?
                .add(&momentum_buffer.mul_scalar(1.0 - self.config.nu)?)?;

            // Apply update
            *parameter = parameter.sub(&update_direction.mul_scalar(self.config.learning_rate)?)?;
        }

        Ok(())
    }

    fn get_lr(&self) -> f32 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }

    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut state = HashMap::new();

        // Save configuration
        state.insert(
            "learning_rate".to_string(),
            Tensor::scalar(self.config.learning_rate)?,
        );
        state.insert(
            "momentum".to_string(),
            Tensor::scalar(self.config.momentum)?,
        );
        state.insert("nu".to_string(), Tensor::scalar(self.config.nu)?);
        state.insert(
            "weight_decay".to_string(),
            Tensor::scalar(self.config.weight_decay)?,
        );
        state.insert(
            "current_step".to_string(),
            Tensor::scalar(self.current_step as f32)?,
        );

        // Save momentum buffers
        for (&param_id, buffer) in &self.momentum_buffers {
            state.insert(format!("momentum_buffer_{}", param_id), buffer.clone());
        }

        Ok(state)
    }

    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()> {
        // Load configuration
        if let Some(lr) = state.get("learning_rate") {
            self.config.learning_rate = lr.to_scalar()?;
        }
        if let Some(momentum) = state.get("momentum") {
            self.config.momentum = momentum.to_scalar()?;
        }
        if let Some(nu) = state.get("nu") {
            self.config.nu = nu.to_scalar()?;
        }
        if let Some(wd) = state.get("weight_decay") {
            self.config.weight_decay = wd.to_scalar()?;
        }
        if let Some(step) = state.get("current_step") {
            self.current_step = step.to_scalar()? as usize;
        }

        // Load momentum buffers
        self.momentum_buffers.clear();
        for (key, tensor) in state {
            if let Some(param_id_str) = key.strip_prefix("momentum_buffer_") {
                if let Ok(param_id) = param_id_str.parse::<usize>() {
                    self.momentum_buffers.insert(param_id, tensor);
                }
            }
        }

        Ok(())
    }
}

/// Configuration for Aggregated Momentum (AggMo).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggMoConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// List of momentum coefficients
    pub momentum_coefficients: Vec<f32>,
    /// Weight decay
    pub weight_decay: f32,
}

impl Default for AggMoConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            momentum_coefficients: vec![0.0, 0.9, 0.99],
            weight_decay: 0.0,
        }
    }
}

/// Aggregated Momentum optimizer.
///
/// AggMo maintains multiple momentum buffers with different decay rates
/// and averages their contributions to improve convergence.
#[derive(Debug)]
pub struct AggMo {
    config: AggMoConfig,
    momentum_buffers: HashMap<usize, Vec<Tensor>>, // param_id -> list of momentum buffers
    current_step: usize,
}

impl AggMo {
    /// Create a new AggMo optimizer.
    pub fn new(config: AggMoConfig) -> Self {
        assert!(
            !config.momentum_coefficients.is_empty(),
            "Must provide at least one momentum coefficient"
        );
        Self {
            config,
            momentum_buffers: HashMap::new(),
            current_step: 0,
        }
    }

    /// Create AggMo with default configuration.
    pub fn with_defaults(learning_rate: f32, momentum_coefficients: Vec<f32>) -> Self {
        Self::new(AggMoConfig {
            learning_rate,
            momentum_coefficients,
            weight_decay: 0.0,
        })
    }

    /// Get the configuration.
    pub fn get_config(&self) -> &AggMoConfig {
        &self.config
    }

    /// Get the number of momentum buffers per parameter.
    pub fn num_momentum_buffers(&self) -> usize {
        self.config.momentum_coefficients.len()
    }
}

impl OptimizerState for AggMo {
    fn zero_grad(&mut self) -> Result<()> {
        Ok(())
    }

    fn step(&mut self, parameters: &mut [Tensor]) -> Result<()> {
        self.current_step += 1;

        for (param_id, parameter) in parameters.iter_mut().enumerate() {
            // Access gradient from parameter (should be computed during forward/backward pass)
            let gradient = match parameter.grad() {
                Ok(grad) => grad,
                Err(_) => {
                    // If gradient is not available, skip this parameter
                    continue;
                },
            };

            // Apply weight decay
            let effective_grad = if self.config.weight_decay > 0.0 {
                gradient.add(&parameter.mul_scalar(self.config.weight_decay)?)?
            } else {
                gradient
            };

            // Get or initialize momentum buffers for this parameter
            let buffers = self.momentum_buffers.entry(param_id).or_insert_with(|| {
                // Initialize all momentum buffers with zeros
                (0..self.config.momentum_coefficients.len())
                    .map(|_| {
                        Tensor::zeros(&effective_grad.shape())
                            .expect("zeros should always succeed for valid gradient shape")
                    })
                    .collect()
            });

            // Update each momentum buffer
            let mut aggregated_momentum = Tensor::zeros(&effective_grad.shape())?;
            for (i, &beta) in self.config.momentum_coefficients.iter().enumerate() {
                // Update momentum: m_i = β_i * m_i + (1 - β_i) * grad
                buffers[i] =
                    buffers[i].mul_scalar(beta)?.add(&effective_grad.mul_scalar(1.0 - beta)?)?;

                // Add to aggregated momentum
                aggregated_momentum = aggregated_momentum.add(&buffers[i])?;
            }

            // Average the momentum buffers
            let num_buffers = self.config.momentum_coefficients.len() as f32;
            let averaged_momentum = aggregated_momentum.div_scalar(num_buffers)?;

            // Apply update
            *parameter =
                parameter.sub(&averaged_momentum.mul_scalar(self.config.learning_rate)?)?;
        }

        Ok(())
    }

    fn get_lr(&self) -> f32 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }

    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut state = HashMap::new();

        // Save configuration
        state.insert(
            "learning_rate".to_string(),
            Tensor::scalar(self.config.learning_rate)?,
        );
        state.insert(
            "weight_decay".to_string(),
            Tensor::scalar(self.config.weight_decay)?,
        );
        state.insert(
            "current_step".to_string(),
            Tensor::scalar(self.current_step as f32)?,
        );
        state.insert(
            "num_momentum_coeffs".to_string(),
            Tensor::scalar(self.config.momentum_coefficients.len() as f32)?,
        );

        // Save momentum coefficients
        for (i, &coeff) in self.config.momentum_coefficients.iter().enumerate() {
            state.insert(format!("momentum_coeff_{}", i), Tensor::scalar(coeff)?);
        }

        // Save momentum buffers
        for (&param_id, buffers) in &self.momentum_buffers {
            for (buffer_idx, buffer) in buffers.iter().enumerate() {
                state.insert(
                    format!("momentum_buffer_{}_{}", param_id, buffer_idx),
                    buffer.clone(),
                );
            }
        }

        Ok(state)
    }

    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()> {
        // Load configuration
        if let Some(lr) = state.get("learning_rate") {
            self.config.learning_rate = lr.to_scalar()?;
        }
        if let Some(wd) = state.get("weight_decay") {
            self.config.weight_decay = wd.to_scalar()?;
        }
        if let Some(step) = state.get("current_step") {
            self.current_step = step.to_scalar()? as usize;
        }

        // Load momentum coefficients
        if let Some(num_coeffs_tensor) = state.get("num_momentum_coeffs") {
            let num_coeffs = num_coeffs_tensor.to_scalar()? as usize;
            let mut coefficients = Vec::with_capacity(num_coeffs);
            for i in 0..num_coeffs {
                if let Some(coeff_tensor) = state.get(&format!("momentum_coeff_{}", i)) {
                    coefficients.push(coeff_tensor.to_scalar()?);
                }
            }
            self.config.momentum_coefficients = coefficients;
        }

        // Load momentum buffers
        self.momentum_buffers.clear();
        let mut param_buffers: HashMap<usize, HashMap<usize, Tensor>> = HashMap::new();

        for (key, tensor) in state {
            if key.starts_with("momentum_buffer_") {
                let parts: Vec<&str> = key.split('_').collect();
                if parts.len() >= 4 {
                    if let (Ok(param_id), Ok(buffer_idx)) =
                        (parts[2].parse::<usize>(), parts[3].parse::<usize>())
                    {
                        param_buffers.entry(param_id).or_default().insert(buffer_idx, tensor);
                    }
                }
            }
        }

        // Reconstruct momentum buffers in correct order
        for (param_id, buffer_map) in param_buffers {
            let mut buffers = Vec::new();
            for i in 0..self.config.momentum_coefficients.len() {
                if let Some(buffer) = buffer_map.get(&i) {
                    buffers.push(buffer.clone());
                }
            }
            if buffers.len() == self.config.momentum_coefficients.len() {
                self.momentum_buffers.insert(param_id, buffers);
            }
        }

        Ok(())
    }
}

/// Configuration for Variance Reduction methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarianceReductionConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Method type
    pub method: VarianceReductionMethod,
    /// Gradient history size for SVRG
    pub history_size: usize,
    /// Update frequency for full gradient computation
    pub full_grad_frequency: usize,
    /// Weight decay
    pub weight_decay: f32,
}

impl Default for VarianceReductionConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            method: VarianceReductionMethod::SVRG,
            history_size: 100,
            full_grad_frequency: 10,
            weight_decay: 0.0,
        }
    }
}

/// Types of variance reduction methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VarianceReductionMethod {
    /// Stochastic Variance Reduced Gradient
    SVRG,
    /// Stochastic Average Gradient
    SAG,
}

/// Variance Reduction optimizer implementing SVRG and SAG methods.
#[derive(Debug)]
pub struct VarianceReduction {
    config: VarianceReductionConfig,
    gradient_history: HashMap<usize, Vec<Tensor>>,
    average_gradients: HashMap<usize, Tensor>,
    full_gradients: HashMap<usize, Tensor>,
    current_step: usize,
    last_full_grad_step: usize,
}

impl VarianceReduction {
    /// Create a new variance reduction optimizer.
    pub fn new(config: VarianceReductionConfig) -> Self {
        Self {
            config,
            gradient_history: HashMap::new(),
            average_gradients: HashMap::new(),
            full_gradients: HashMap::new(),
            current_step: 0,
            last_full_grad_step: 0,
        }
    }

    /// Create SVRG optimizer with default settings.
    pub fn svrg(learning_rate: f32, history_size: usize, full_grad_frequency: usize) -> Self {
        Self::new(VarianceReductionConfig {
            learning_rate,
            method: VarianceReductionMethod::SVRG,
            history_size,
            full_grad_frequency,
            weight_decay: 0.0,
        })
    }

    /// Create SAG optimizer with default settings.
    pub fn sag(learning_rate: f32, history_size: usize) -> Self {
        Self::new(VarianceReductionConfig {
            learning_rate,
            method: VarianceReductionMethod::SAG,
            history_size,
            full_grad_frequency: 1, // Not used for SAG
            weight_decay: 0.0,
        })
    }

    fn update_gradient_history(&mut self, param_id: usize, gradient: &Tensor) -> Result<()> {
        let history = self.gradient_history.entry(param_id).or_default();

        history.push(gradient.clone());
        if history.len() > self.config.history_size {
            history.remove(0);
        }

        Ok(())
    }

    fn compute_average_gradient(&mut self, param_id: usize) -> Result<Tensor> {
        if let Some(history) = self.gradient_history.get(&param_id) {
            if history.is_empty() {
                return Err(anyhow!("No gradient history available"));
            }

            let mut sum = history[0].clone();
            for grad in history.iter().skip(1) {
                sum = sum.add(grad)?;
            }

            let average = sum.div_scalar(history.len() as f32)?;
            self.average_gradients.insert(param_id, average.clone());
            Ok(average)
        } else {
            Err(anyhow!("No gradient history for parameter {}", param_id))
        }
    }

    fn should_compute_full_gradient(&self) -> bool {
        self.current_step - self.last_full_grad_step >= self.config.full_grad_frequency
    }
}

impl OptimizerState for VarianceReduction {
    fn zero_grad(&mut self) -> Result<()> {
        Ok(())
    }

    fn step(&mut self, parameters: &mut [Tensor]) -> Result<()> {
        self.current_step += 1;

        // Check if we need to compute full gradient (for SVRG)
        let compute_full_grad = match self.config.method {
            VarianceReductionMethod::SVRG => self.should_compute_full_gradient(),
            VarianceReductionMethod::SAG => false,
        };

        if compute_full_grad {
            self.last_full_grad_step = self.current_step;
            // In practice, full gradient computation would require access to the full dataset
            // Here we'll use the current gradient as an approximation
            for (param_id, parameter) in parameters.iter().enumerate() {
                // Access gradient from parameter (should be computed during forward/backward pass)
                let gradient = match parameter.grad() {
                    Ok(grad) => grad,
                    Err(_) => {
                        // If gradient is not available, skip this parameter
                        continue;
                    },
                };
                self.full_gradients.insert(param_id, gradient);
            }
        }

        for (param_id, parameter) in parameters.iter_mut().enumerate() {
            // Access gradient from parameter (should be computed during forward/backward pass)
            let current_gradient = match parameter.grad() {
                Ok(grad) => grad,
                Err(_) => {
                    // If gradient is not available, skip this parameter
                    continue;
                },
            };

            // Apply weight decay
            let effective_grad = if self.config.weight_decay > 0.0 {
                current_gradient.add(&parameter.mul_scalar(self.config.weight_decay)?)?
            } else {
                current_gradient
            };

            // Update gradient history
            self.update_gradient_history(param_id, &effective_grad)?;

            // Apply variance reduction
            let variance_reduced_grad = match self.config.method {
                VarianceReductionMethod::SVRG => {
                    // Clone full_grad before mutable borrow for compute_average_gradient
                    let full_grad_opt = self.full_gradients.get(&param_id).cloned();
                    if let Some(full_grad) = full_grad_opt {
                        let avg_grad = self.compute_average_gradient(param_id)?;
                        // SVRG update: grad - avg_grad + full_grad
                        effective_grad.sub(&avg_grad)?.add(&full_grad)?
                    } else {
                        effective_grad
                    }
                },
                VarianceReductionMethod::SAG => {
                    // SAG uses running average of gradients
                    self.compute_average_gradient(param_id)?
                },
            };

            // Apply update
            *parameter =
                parameter.sub(&variance_reduced_grad.mul_scalar(self.config.learning_rate)?)?;
        }

        Ok(())
    }

    fn get_lr(&self) -> f32 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }

    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut state = HashMap::new();

        state.insert(
            "learning_rate".to_string(),
            Tensor::scalar(self.config.learning_rate)?,
        );
        state.insert(
            "current_step".to_string(),
            Tensor::scalar(self.current_step as f32)?,
        );
        state.insert(
            "last_full_grad_step".to_string(),
            Tensor::scalar(self.last_full_grad_step as f32)?,
        );

        // Note: Saving full gradient history would be expensive
        // In practice, you might want to save only recent gradients or statistics

        Ok(state)
    }

    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()> {
        if let Some(lr) = state.get("learning_rate") {
            self.config.learning_rate = lr.to_scalar()?;
        }
        if let Some(step) = state.get("current_step") {
            self.current_step = step.to_scalar()? as usize;
        }
        if let Some(last_step) = state.get("last_full_grad_step") {
            self.last_full_grad_step = last_step.to_scalar()? as usize;
        }

        Ok(())
    }
}

/// Configuration for Nesterov Accelerated Gradient (NAG).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NesterovAcceleratedGradientConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Momentum parameter
    pub momentum: f32,
    /// Weight decay
    pub weight_decay: f32,
    /// Whether to use strong convexity assumption for restart
    pub restart_on_increase: bool,
}

impl Default for NesterovAcceleratedGradientConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            momentum: 0.9,
            weight_decay: 0.0,
            restart_on_increase: false,
        }
    }
}

/// Nesterov Accelerated Gradient optimizer.
///
/// NAG uses lookahead to evaluate the gradient at the predicted next position,
/// which can lead to faster convergence than standard momentum methods.
/// Update rule:
/// v_t = momentum * v_{t-1} + lr * grad(x_t + momentum * v_{t-1})
/// x_{t+1} = x_t - v_t
#[derive(Debug)]
pub struct NesterovAcceleratedGradient {
    config: NesterovAcceleratedGradientConfig,
    velocity_buffers: HashMap<usize, Tensor>,
    current_step: usize,
    previous_loss: Option<f32>,
}

impl NesterovAcceleratedGradient {
    /// Create a new NAG optimizer.
    pub fn new(config: NesterovAcceleratedGradientConfig) -> Self {
        Self {
            config,
            velocity_buffers: HashMap::new(),
            current_step: 0,
            previous_loss: None,
        }
    }

    /// Create NAG with default configuration.
    pub fn with_defaults(learning_rate: f32, momentum: f32) -> Self {
        Self::new(NesterovAcceleratedGradientConfig {
            learning_rate,
            momentum,
            weight_decay: 0.0,
            restart_on_increase: false,
        })
    }

    /// Get the configuration.
    pub fn get_config(&self) -> &NesterovAcceleratedGradientConfig {
        &self.config
    }

    /// Set a loss value for restart detection.
    pub fn set_current_loss(&mut self, loss: f32) {
        if self.config.restart_on_increase {
            if let Some(prev_loss) = self.previous_loss {
                if loss > prev_loss {
                    // Restart by clearing velocity buffers
                    self.velocity_buffers.clear();
                }
            }
        }
        self.previous_loss = Some(loss);
    }
}

impl OptimizerState for NesterovAcceleratedGradient {
    fn zero_grad(&mut self) -> Result<()> {
        Ok(())
    }

    fn step(&mut self, parameters: &mut [Tensor]) -> Result<()> {
        self.current_step += 1;

        for (param_id, parameter) in parameters.iter_mut().enumerate() {
            // Access gradient from parameter (should be computed during forward/backward pass)
            let gradient = match parameter.grad() {
                Ok(grad) => grad,
                Err(_) => {
                    // If gradient is not available, skip this parameter
                    continue;
                },
            };

            // Apply weight decay to gradient
            let effective_grad = if self.config.weight_decay > 0.0 {
                gradient.add(&parameter.mul_scalar(self.config.weight_decay)?)?
            } else {
                gradient
            };

            // Get or initialize velocity buffer
            let velocity = if let Some(v) = self.velocity_buffers.get(&param_id) {
                v.clone()
            } else {
                Tensor::zeros_like(parameter)?
            };

            // Nesterov acceleration: compute gradient at lookahead position
            let _lookahead_position = parameter.sub(&velocity.mul_scalar(self.config.momentum)?)?;

            // In practice, we'd need to recompute the gradient at the lookahead position
            // For now, we'll use the current gradient as approximation
            // Lookahead gradient computation using current gradient state

            // Update velocity: v_t = momentum * v_{t-1} + lr * grad
            let new_velocity = velocity
                .mul_scalar(self.config.momentum)?
                .add(&effective_grad.mul_scalar(self.config.learning_rate)?)?;

            self.velocity_buffers.insert(param_id, new_velocity.clone());

            // Update parameters: x_{t+1} = x_t - v_t
            *parameter = parameter.sub(&new_velocity)?;
        }

        Ok(())
    }

    fn get_lr(&self) -> f32 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }

    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut state = HashMap::new();

        state.insert(
            "learning_rate".to_string(),
            Tensor::scalar(self.config.learning_rate)?,
        );
        state.insert(
            "momentum".to_string(),
            Tensor::scalar(self.config.momentum)?,
        );
        state.insert(
            "weight_decay".to_string(),
            Tensor::scalar(self.config.weight_decay)?,
        );
        state.insert(
            "current_step".to_string(),
            Tensor::scalar(self.current_step as f32)?,
        );

        if let Some(loss) = self.previous_loss {
            state.insert("previous_loss".to_string(), Tensor::scalar(loss)?);
        }

        for (&param_id, velocity) in &self.velocity_buffers {
            state.insert(format!("velocity_{}", param_id), velocity.clone());
        }

        Ok(state)
    }

    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()> {
        if let Some(lr) = state.get("learning_rate") {
            self.config.learning_rate = lr.to_scalar()?;
        }
        if let Some(momentum) = state.get("momentum") {
            self.config.momentum = momentum.to_scalar()?;
        }
        if let Some(wd) = state.get("weight_decay") {
            self.config.weight_decay = wd.to_scalar()?;
        }
        if let Some(step) = state.get("current_step") {
            self.current_step = step.to_scalar()? as usize;
        }
        if let Some(loss) = state.get("previous_loss") {
            self.previous_loss = Some(loss.to_scalar()?);
        }

        self.velocity_buffers.clear();
        for (key, tensor) in state {
            if let Some(param_id_str) = key.strip_prefix("velocity_") {
                if let Ok(param_id) = param_id_str.parse::<usize>() {
                    self.velocity_buffers.insert(param_id, tensor);
                }
            }
        }

        Ok(())
    }
}

/// Configuration for Heavy Ball Method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeavyBallConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Momentum coefficient (β)
    pub beta: f32,
    /// Weight decay
    pub weight_decay: f32,
    /// Adaptive momentum based on gradient alignment
    pub adaptive_momentum: bool,
}

impl Default for HeavyBallConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta: 0.9,
            weight_decay: 0.0,
            adaptive_momentum: false,
        }
    }
}

/// Heavy Ball Method optimizer.
///
/// Classical momentum-based acceleration method that adds inertia to gradient descent.
/// Update rule:
/// v_t = β * v_{t-1} - lr * grad(x_t)
/// x_{t+1} = x_t + v_t
#[derive(Debug)]
pub struct HeavyBall {
    config: HeavyBallConfig,
    velocity_buffers: HashMap<usize, Tensor>,
    previous_gradients: HashMap<usize, Tensor>,
    current_step: usize,
}

impl HeavyBall {
    /// Create a new Heavy Ball optimizer.
    pub fn new(config: HeavyBallConfig) -> Self {
        Self {
            config,
            velocity_buffers: HashMap::new(),
            previous_gradients: HashMap::new(),
            current_step: 0,
        }
    }

    /// Create Heavy Ball with default configuration.
    pub fn with_defaults(learning_rate: f32, beta: f32) -> Self {
        Self::new(HeavyBallConfig {
            learning_rate,
            beta,
            weight_decay: 0.0,
            adaptive_momentum: false,
        })
    }

    /// Get the configuration.
    pub fn get_config(&self) -> &HeavyBallConfig {
        &self.config
    }

    /// Compute adaptive momentum based on gradient alignment.
    fn compute_adaptive_momentum(&self, param_id: usize, current_grad: &Tensor) -> Result<f32> {
        if let Some(prev_grad) = self.previous_gradients.get(&param_id) {
            // Compute cosine similarity between current and previous gradients
            let dot_product = current_grad.mul(prev_grad)?.sum(None, false)?;
            let norm_current = current_grad.norm_squared()?.sqrt()?;
            let norm_prev = prev_grad.norm_squared()?.sqrt()?;

            let dot_scalar = dot_product.to_scalar()?;
            let norm_current_scalar = norm_current.to_scalar()?;
            let norm_prev_scalar = norm_prev.to_scalar()?;

            let denominator = norm_current_scalar * norm_prev_scalar;
            if denominator > 1e-8 {
                let cosine_similarity = dot_scalar / denominator;
                // Increase momentum when gradients are aligned, decrease when opposed
                let adaptive_beta = self.config.beta * cosine_similarity.max(0.0);
                Ok(adaptive_beta)
            } else {
                Ok(self.config.beta)
            }
        } else {
            Ok(self.config.beta)
        }
    }
}

impl OptimizerState for HeavyBall {
    fn zero_grad(&mut self) -> Result<()> {
        Ok(())
    }

    fn step(&mut self, parameters: &mut [Tensor]) -> Result<()> {
        self.current_step += 1;

        for (param_id, parameter) in parameters.iter_mut().enumerate() {
            // Access gradient from parameter (should be computed during forward/backward pass)
            let gradient = match parameter.grad() {
                Ok(grad) => grad,
                Err(_) => {
                    // If gradient is not available, skip this parameter
                    continue;
                },
            };

            // Apply weight decay to gradient
            let effective_grad = if self.config.weight_decay > 0.0 {
                gradient.add(&parameter.mul_scalar(self.config.weight_decay)?)?
            } else {
                gradient
            };

            // Compute momentum coefficient
            let beta = if self.config.adaptive_momentum {
                self.compute_adaptive_momentum(param_id, &effective_grad)?
            } else {
                self.config.beta
            };

            // Get or initialize velocity buffer
            let velocity = if let Some(v) = self.velocity_buffers.get(&param_id) {
                v.clone()
            } else {
                Tensor::zeros_like(parameter)?
            };

            // Heavy Ball update: v_t = β * v_{t-1} - lr * grad
            let new_velocity = velocity
                .mul_scalar(beta)?
                .sub(&effective_grad.mul_scalar(self.config.learning_rate)?)?;

            self.velocity_buffers.insert(param_id, new_velocity.clone());

            // Update parameters: x_{t+1} = x_t + v_t
            *parameter = parameter.add(&new_velocity)?;

            // Store gradient for adaptive momentum
            if self.config.adaptive_momentum {
                self.previous_gradients.insert(param_id, effective_grad);
            }
        }

        Ok(())
    }

    fn get_lr(&self) -> f32 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }

    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut state = HashMap::new();

        state.insert(
            "learning_rate".to_string(),
            Tensor::scalar(self.config.learning_rate)?,
        );
        state.insert("beta".to_string(), Tensor::scalar(self.config.beta)?);
        state.insert(
            "weight_decay".to_string(),
            Tensor::scalar(self.config.weight_decay)?,
        );
        state.insert(
            "current_step".to_string(),
            Tensor::scalar(self.current_step as f32)?,
        );

        for (&param_id, velocity) in &self.velocity_buffers {
            state.insert(format!("velocity_{}", param_id), velocity.clone());
        }

        for (&param_id, grad) in &self.previous_gradients {
            state.insert(format!("prev_grad_{}", param_id), grad.clone());
        }

        Ok(state)
    }

    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()> {
        if let Some(lr) = state.get("learning_rate") {
            self.config.learning_rate = lr.to_scalar()?;
        }
        if let Some(beta) = state.get("beta") {
            self.config.beta = beta.to_scalar()?;
        }
        if let Some(wd) = state.get("weight_decay") {
            self.config.weight_decay = wd.to_scalar()?;
        }
        if let Some(step) = state.get("current_step") {
            self.current_step = step.to_scalar()? as usize;
        }

        self.velocity_buffers.clear();
        self.previous_gradients.clear();

        for (key, tensor) in state {
            if let Some(param_id_str) = key.strip_prefix("velocity_") {
                if let Ok(param_id) = param_id_str.parse::<usize>() {
                    self.velocity_buffers.insert(param_id, tensor);
                }
            } else if let Some(param_id_str) = key.strip_prefix("prev_grad_") {
                if let Ok(param_id) = param_id_str.parse::<usize>() {
                    self.previous_gradients.insert(param_id, tensor);
                }
            }
        }

        Ok(())
    }
}

/// Configuration for FISTA (Fast Iterative Shrinkage-Thresholding Algorithm).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FISTAConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Proximal threshold parameter
    pub threshold: f32,
    /// Whether to use adaptive restart
    pub adaptive_restart: bool,
    /// Weight decay
    pub weight_decay: f32,
}

impl Default for FISTAConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            threshold: 1e-4,
            adaptive_restart: true,
            weight_decay: 0.0,
        }
    }
}

/// FISTA optimizer for problems with L1 regularization or other proximal operators.
///
/// FISTA is designed for problems of the form: min f(x) + λ||x||_1
/// where f(x) is smooth and convex, and λ||x||_1 is the L1 regularization term.
#[derive(Debug)]
pub struct FISTA {
    config: FISTAConfig,
    previous_params: HashMap<usize, Tensor>,
    current_step: usize,
    momentum_coefficient: f32,
    previous_momentum: f32,
}

impl FISTA {
    /// Create a new FISTA optimizer.
    pub fn new(config: FISTAConfig) -> Self {
        Self {
            config,
            previous_params: HashMap::new(),
            current_step: 0,
            momentum_coefficient: 1.0,
            previous_momentum: 1.0,
        }
    }

    /// Create FISTA with default configuration.
    pub fn with_defaults(learning_rate: f32, threshold: f32) -> Self {
        Self::new(FISTAConfig {
            learning_rate,
            threshold,
            adaptive_restart: true,
            weight_decay: 0.0,
        })
    }

    /// Get the configuration.
    pub fn get_config(&self) -> &FISTAConfig {
        &self.config
    }

    /// Apply soft thresholding (proximal operator for L1 regularization).
    fn soft_threshold(&self, tensor: &Tensor, threshold: f32) -> Result<Tensor> {
        let threshold_tensor = Tensor::scalar(threshold)?;
        let zero_tensor = Tensor::zeros_like(tensor)?;

        // Soft thresholding: sign(x) * max(0, |x| - threshold)
        let abs_tensor = tensor.abs()?;
        let thresholded = abs_tensor.sub(&threshold_tensor)?.max(&zero_tensor)?;
        let sign_tensor = tensor.sign()?;

        Ok(sign_tensor.mul(&thresholded)?)
    }

    /// Update momentum coefficient using FISTA formula.
    fn update_momentum_coefficient(&mut self) {
        let t = self.current_step as f32;
        self.previous_momentum = self.momentum_coefficient;
        self.momentum_coefficient = (1.0 + (1.0 + 4.0 * t * t).sqrt()) / 2.0;
    }
}

impl OptimizerState for FISTA {
    fn zero_grad(&mut self) -> Result<()> {
        Ok(())
    }

    fn step(&mut self, parameters: &mut [Tensor]) -> Result<()> {
        self.current_step += 1;
        self.update_momentum_coefficient();

        for (param_id, parameter) in parameters.iter_mut().enumerate() {
            // Access gradient from parameter (should be computed during forward/backward pass)
            let gradient = match parameter.grad() {
                Ok(grad) => grad,
                Err(_) => {
                    // If gradient is not available, skip this parameter
                    continue;
                },
            };

            // Apply weight decay to gradient
            let effective_grad = if self.config.weight_decay > 0.0 {
                gradient.add(&parameter.mul_scalar(self.config.weight_decay)?)?
            } else {
                gradient
            };

            // Get previous parameter value
            let previous_param = if let Some(prev) = self.previous_params.get(&param_id) {
                prev.clone()
            } else {
                parameter.clone()
            };

            // Momentum coefficient ratio
            let beta = (self.previous_momentum - 1.0) / self.momentum_coefficient;

            // Compute extrapolated point
            let extrapolated = parameter.add(&previous_param.sub(parameter)?.mul_scalar(beta)?)?;

            // Gradient step
            let grad_step =
                extrapolated.sub(&effective_grad.mul_scalar(self.config.learning_rate)?)?;

            // Apply proximal operator (soft thresholding)
            let new_parameter = self.soft_threshold(&grad_step, self.config.threshold)?;

            // Store current parameter for next iteration
            self.previous_params.insert(param_id, parameter.clone());

            // Update parameter
            *parameter = new_parameter;
        }

        Ok(())
    }

    fn get_lr(&self) -> f32 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }

    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut state = HashMap::new();

        state.insert(
            "learning_rate".to_string(),
            Tensor::scalar(self.config.learning_rate)?,
        );
        state.insert(
            "threshold".to_string(),
            Tensor::scalar(self.config.threshold)?,
        );
        state.insert(
            "weight_decay".to_string(),
            Tensor::scalar(self.config.weight_decay)?,
        );
        state.insert(
            "current_step".to_string(),
            Tensor::scalar(self.current_step as f32)?,
        );
        state.insert(
            "momentum_coefficient".to_string(),
            Tensor::scalar(self.momentum_coefficient)?,
        );
        state.insert(
            "previous_momentum".to_string(),
            Tensor::scalar(self.previous_momentum)?,
        );

        for (&param_id, param) in &self.previous_params {
            state.insert(format!("prev_param_{}", param_id), param.clone());
        }

        Ok(state)
    }

    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()> {
        if let Some(lr) = state.get("learning_rate") {
            self.config.learning_rate = lr.to_scalar()?;
        }
        if let Some(threshold) = state.get("threshold") {
            self.config.threshold = threshold.to_scalar()?;
        }
        if let Some(wd) = state.get("weight_decay") {
            self.config.weight_decay = wd.to_scalar()?;
        }
        if let Some(step) = state.get("current_step") {
            self.current_step = step.to_scalar()? as usize;
        }
        if let Some(momentum) = state.get("momentum_coefficient") {
            self.momentum_coefficient = momentum.to_scalar()?;
        }
        if let Some(prev_momentum) = state.get("previous_momentum") {
            self.previous_momentum = prev_momentum.to_scalar()?;
        }

        self.previous_params.clear();
        for (key, tensor) in state {
            if let Some(param_id_str) = key.strip_prefix("prev_param_") {
                if let Ok(param_id) = param_id_str.parse::<usize>() {
                    self.previous_params.insert(param_id, tensor);
                }
            }
        }

        Ok(())
    }
}

/// Configuration for Adaptive Batch Sizing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveBatchSizingConfig {
    /// Initial batch size
    pub initial_batch_size: usize,
    /// Minimum batch size
    pub min_batch_size: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Tolerance for gradient variance
    pub gradient_variance_tolerance: f32,
    /// Learning rate adaptation factor
    pub lr_adaptation_factor: f32,
    /// Window size for gradient variance calculation
    pub variance_window_size: usize,
    /// Threshold for increasing batch size
    pub increase_threshold: f32,
    /// Threshold for decreasing batch size
    pub decrease_threshold: f32,
}

impl Default for AdaptiveBatchSizingConfig {
    fn default() -> Self {
        Self {
            initial_batch_size: 32,
            min_batch_size: 8,
            max_batch_size: 512,
            gradient_variance_tolerance: 0.1,
            lr_adaptation_factor: 0.8,
            variance_window_size: 10,
            increase_threshold: 0.05,
            decrease_threshold: 0.2,
        }
    }
}

/// Adaptive Batch Sizing utility for dynamically adjusting batch size based on training progress.
///
/// This strategy monitors gradient variance and training stability to determine optimal batch sizes.
/// When gradient variance is high, it increases batch size to reduce noise.
/// When variance is low, it may decrease batch size to improve convergence speed.
#[derive(Debug)]
pub struct AdaptiveBatchSizing {
    config: AdaptiveBatchSizingConfig,
    current_batch_size: usize,
    gradient_variance_history: Vec<f32>,
    loss_history: Vec<f32>,
    current_step: usize,
    last_adjustment_step: usize,
}

impl AdaptiveBatchSizing {
    /// Create a new adaptive batch sizing utility.
    pub fn new(config: AdaptiveBatchSizingConfig) -> Self {
        let initial_batch_size = config.initial_batch_size;
        Self {
            config,
            current_batch_size: initial_batch_size,
            gradient_variance_history: Vec::new(),
            loss_history: Vec::new(),
            current_step: 0,
            last_adjustment_step: 0,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults(
        initial_batch_size: usize,
        min_batch_size: usize,
        max_batch_size: usize,
    ) -> Self {
        Self::new(AdaptiveBatchSizingConfig {
            initial_batch_size,
            min_batch_size,
            max_batch_size,
            ..Default::default()
        })
    }

    /// Get current batch size.
    pub fn current_batch_size(&self) -> usize {
        self.current_batch_size
    }

    /// Get the configuration.
    pub fn get_config(&self) -> &AdaptiveBatchSizingConfig {
        &self.config
    }

    /// Update with current gradient variance and loss.
    pub fn update(&mut self, gradient_variance: f32, current_loss: f32) -> Result<usize> {
        self.current_step += 1;

        // Add to history
        self.gradient_variance_history.push(gradient_variance);
        self.loss_history.push(current_loss);

        // Keep only recent history
        if self.gradient_variance_history.len() > self.config.variance_window_size {
            self.gradient_variance_history.remove(0);
        }
        if self.loss_history.len() > self.config.variance_window_size {
            self.loss_history.remove(0);
        }

        // Check if we should adjust batch size
        if self.should_adjust_batch_size() {
            self.adjust_batch_size()?;
            self.last_adjustment_step = self.current_step;
        }

        Ok(self.current_batch_size)
    }

    /// Compute gradient variance from gradients.
    pub fn compute_gradient_variance(&self, gradients: &[Tensor]) -> Result<f32> {
        if gradients.is_empty() {
            return Ok(0.0);
        }

        // Compute mean gradient
        let mut mean_grad = gradients[0].clone();
        for grad in gradients.iter().skip(1) {
            mean_grad = mean_grad.add(grad)?;
        }
        mean_grad = mean_grad.div_scalar(gradients.len() as f32)?;

        // Compute variance
        let mut variance_sum = 0.0;
        for grad in gradients {
            let diff = grad.sub(&mean_grad)?;
            let squared_norm = diff.mul(&diff)?.sum(None, false)?;
            variance_sum += squared_norm.to_scalar()?;
        }

        Ok(variance_sum / gradients.len() as f32)
    }

    fn should_adjust_batch_size(&self) -> bool {
        // Don't adjust too frequently
        if self.current_step - self.last_adjustment_step < 5 {
            return false;
        }

        // Need enough history
        self.gradient_variance_history.len() >= 3
    }

    fn adjust_batch_size(&mut self) -> Result<()> {
        let recent_variance = self.recent_average_variance();
        let variance_trend = self.variance_trend();
        let loss_trend = self.loss_trend();

        // Decide whether to increase or decrease batch size
        if recent_variance > self.config.decrease_threshold && variance_trend > 0.0 {
            // High variance and increasing - increase batch size
            self.increase_batch_size();
        } else if recent_variance < self.config.increase_threshold && loss_trend < -0.01 {
            // Low variance and decreasing loss - try smaller batch size
            self.decrease_batch_size();
        }

        Ok(())
    }

    fn recent_average_variance(&self) -> f32 {
        if self.gradient_variance_history.is_empty() {
            return 0.0;
        }

        let recent_window = std::cmp::min(5, self.gradient_variance_history.len());
        let start_idx = self.gradient_variance_history.len() - recent_window;

        self.gradient_variance_history[start_idx..].iter().sum::<f32>() / recent_window as f32
    }

    fn variance_trend(&self) -> f32 {
        if self.gradient_variance_history.len() < 3 {
            return 0.0;
        }

        let len = self.gradient_variance_history.len();
        let recent = self.gradient_variance_history[len - 2..].iter().sum::<f32>() / 2.0;
        let older = self.gradient_variance_history[len - 4..len - 2].iter().sum::<f32>() / 2.0;

        recent - older
    }

    fn loss_trend(&self) -> f32 {
        if self.loss_history.len() < 3 {
            return 0.0;
        }

        let len = self.loss_history.len();
        let recent = self.loss_history[len - 2..].iter().sum::<f32>() / 2.0;
        let older = self.loss_history[len - 4..len - 2].iter().sum::<f32>() / 2.0;

        (recent - older) / older.max(1e-8)
    }

    fn increase_batch_size(&mut self) {
        let new_size = (self.current_batch_size as f32 * 1.5) as usize;
        self.current_batch_size = new_size.min(self.config.max_batch_size);
    }

    fn decrease_batch_size(&mut self) {
        let new_size = (self.current_batch_size as f32 * 0.8) as usize;
        self.current_batch_size = new_size.max(self.config.min_batch_size);
    }

    /// Get suggested learning rate adjustment based on batch size changes.
    pub fn get_lr_adjustment(&self, original_batch_size: usize) -> f32 {
        let ratio = self.current_batch_size as f32 / original_batch_size as f32;
        ratio.sqrt() * self.config.lr_adaptation_factor
    }

    /// Reset state for new training run.
    pub fn reset(&mut self) {
        self.current_batch_size = self.config.initial_batch_size;
        self.gradient_variance_history.clear();
        self.loss_history.clear();
        self.current_step = 0;
        self.last_adjustment_step = 0;
    }
}

/// Configuration for Loss Surface Smoothing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossSurfaceSmoothingConfig {
    /// Smoothing strength parameter
    pub smoothing_strength: f32,
    /// Noise injection variance
    pub noise_variance: f32,
    /// Exponential moving average decay
    pub ema_decay: f32,
    /// Number of gradient steps to average
    pub averaging_window: usize,
    /// Whether to use gradient averaging
    pub use_gradient_averaging: bool,
    /// Whether to use noise injection
    pub use_noise_injection: bool,
}

impl Default for LossSurfaceSmoothingConfig {
    fn default() -> Self {
        Self {
            smoothing_strength: 0.1,
            noise_variance: 1e-4,
            ema_decay: 0.9,
            averaging_window: 5,
            use_gradient_averaging: true,
            use_noise_injection: false,
        }
    }
}

/// Loss Surface Smoothing utility for reducing noise in the loss landscape.
///
/// This implements several techniques to smooth the loss surface:
/// - Gradient averaging over multiple steps
/// - Exponential moving average of gradients
/// - Controlled noise injection for exploration
/// - Parameter smoothing to reduce sharp changes
#[derive(Debug)]
pub struct LossSurfaceSmoothing {
    config: LossSurfaceSmoothingConfig,
    gradient_history: HashMap<usize, Vec<Tensor>>,
    ema_gradients: HashMap<usize, Tensor>,
    smoothed_parameters: HashMap<usize, Tensor>,
    current_step: usize,
}

impl LossSurfaceSmoothing {
    /// Create a new loss surface smoothing utility.
    pub fn new(config: LossSurfaceSmoothingConfig) -> Self {
        Self {
            config,
            gradient_history: HashMap::new(),
            ema_gradients: HashMap::new(),
            smoothed_parameters: HashMap::new(),
            current_step: 0,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults(smoothing_strength: f32, use_noise: bool) -> Self {
        Self::new(LossSurfaceSmoothingConfig {
            smoothing_strength,
            use_noise_injection: use_noise,
            ..Default::default()
        })
    }

    /// Get the configuration.
    pub fn get_config(&self) -> &LossSurfaceSmoothingConfig {
        &self.config
    }

    /// Apply smoothing to gradients.
    pub fn smooth_gradients(&mut self, parameters: &mut [Tensor]) -> Result<()> {
        self.current_step += 1;

        for (param_id, parameter) in parameters.iter_mut().enumerate() {
            let original_grad = parameter.grad()?;
            let mut smoothed_grad = original_grad.clone();

            // Apply gradient averaging
            if self.config.use_gradient_averaging {
                smoothed_grad = self.apply_gradient_averaging(param_id, &original_grad)?;
            }

            // Apply exponential moving average
            smoothed_grad = self.apply_ema_smoothing(param_id, &smoothed_grad)?;

            // Apply noise injection for exploration
            if self.config.use_noise_injection {
                smoothed_grad = self.apply_noise_injection(&smoothed_grad)?;
            }

            // Update parameter gradient
            parameter.set_grad(smoothed_grad)?;
        }

        Ok(())
    }

    /// Apply parameter smoothing.
    pub fn smooth_parameters(&mut self, parameters: &mut [Tensor]) -> Result<()> {
        for (param_id, parameter) in parameters.iter_mut().enumerate() {
            if let Some(smoothed_param) = self.smoothed_parameters.get(&param_id) {
                // Apply exponential moving average to parameters
                let new_smoothed = smoothed_param
                    .mul_scalar(self.config.ema_decay)?
                    .add(&parameter.mul_scalar(1.0 - self.config.ema_decay)?)?;

                // Interpolate between original and smoothed parameters
                *parameter = parameter
                    .mul_scalar(1.0 - self.config.smoothing_strength)?
                    .add(&new_smoothed.mul_scalar(self.config.smoothing_strength)?)?;

                self.smoothed_parameters.insert(param_id, new_smoothed);
            } else {
                // Initialize smoothed parameter
                self.smoothed_parameters.insert(param_id, parameter.clone());
            }
        }

        Ok(())
    }

    fn apply_gradient_averaging(&mut self, param_id: usize, gradient: &Tensor) -> Result<Tensor> {
        let history = self.gradient_history.entry(param_id).or_default();

        history.push(gradient.clone());
        if history.len() > self.config.averaging_window {
            history.remove(0);
        }

        // Compute average of recent gradients
        if history.len() == 1 {
            Ok(gradient.clone())
        } else {
            let mut sum = history[0].clone();
            for grad in history.iter().skip(1) {
                sum = sum.add(grad)?;
            }
            Ok(sum.div_scalar(history.len() as f32)?)
        }
    }

    fn apply_ema_smoothing(&mut self, param_id: usize, gradient: &Tensor) -> Result<Tensor> {
        if let Some(ema_grad) = self.ema_gradients.get(&param_id) {
            let new_ema = ema_grad
                .mul_scalar(self.config.ema_decay)?
                .add(&gradient.mul_scalar(1.0 - self.config.ema_decay)?)?;
            self.ema_gradients.insert(param_id, new_ema.clone());
            Ok(new_ema)
        } else {
            self.ema_gradients.insert(param_id, gradient.clone());
            Ok(gradient.clone())
        }
    }

    fn apply_noise_injection(&self, gradient: &Tensor) -> Result<Tensor> {
        let noise = Tensor::randn_like(gradient)
            .map_err(|e| anyhow!("Failed to create noise tensor: {}", e))?
            .mul_scalar(self.config.noise_variance.sqrt())
            .map_err(|e| anyhow!("Failed to scale noise tensor: {}", e))?;
        gradient
            .add(&noise)
            .map_err(|e| anyhow!("Failed to add noise to gradient: {}", e))
    }

    /// Reset state for new training run.
    pub fn reset(&mut self) {
        self.gradient_history.clear();
        self.ema_gradients.clear();
        self.smoothed_parameters.clear();
        self.current_step = 0;
    }

    /// Get smoothing statistics.
    pub fn get_statistics(&self) -> HashMap<String, f32> {
        let mut stats = HashMap::new();
        stats.insert("current_step".to_string(), self.current_step as f32);
        stats.insert(
            "num_tracked_params".to_string(),
            self.gradient_history.len() as f32,
        );
        stats.insert(
            "smoothing_strength".to_string(),
            self.config.smoothing_strength,
        );
        stats.insert("ema_decay".to_string(), self.config.ema_decay);
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qhm_config_default() {
        let config = QHMConfig::default();
        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.momentum, 0.9);
        assert_eq!(config.nu, 0.7);
        assert_eq!(config.weight_decay, 0.0);
    }

    #[test]
    fn test_aggmo_config_default() {
        let config = AggMoConfig::default();
        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.momentum_coefficients, vec![0.0, 0.9, 0.99]);
        assert_eq!(config.weight_decay, 0.0);
    }

    #[test]
    fn test_qhm_creation() {
        let optimizer = QHM::with_defaults(1e-3, 0.9, 0.7);
        assert_eq!(optimizer.get_lr(), 1e-3);
        assert_eq!(optimizer.current_step, 0);
    }

    #[test]
    fn test_aggmo_creation() {
        let optimizer = AggMo::with_defaults(1e-3, vec![0.0, 0.9, 0.99]);
        assert_eq!(optimizer.get_lr(), 1e-3);
        assert_eq!(optimizer.num_momentum_buffers(), 3);
    }

    #[test]
    fn test_variance_reduction_svrg() {
        let optimizer = VarianceReduction::svrg(1e-3, 50, 10);
        assert_eq!(optimizer.get_lr(), 1e-3);
        assert_eq!(optimizer.current_step, 0);
    }

    #[test]
    fn test_variance_reduction_sag() {
        let optimizer = VarianceReduction::sag(1e-3, 100);
        assert_eq!(optimizer.get_lr(), 1e-3);
        assert!(matches!(
            optimizer.config.method,
            VarianceReductionMethod::SAG
        ));
    }

    #[test]
    fn test_nesterov_accelerated_gradient_config() {
        let config = NesterovAcceleratedGradientConfig::default();
        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.momentum, 0.9);
        assert_eq!(config.weight_decay, 0.0);
        assert!(!config.restart_on_increase);
    }

    #[test]
    fn test_nesterov_accelerated_gradient_creation() {
        let optimizer = NesterovAcceleratedGradient::with_defaults(1e-3, 0.9);
        assert_eq!(optimizer.get_lr(), 1e-3);
        assert_eq!(optimizer.current_step, 0);
        assert!(optimizer.previous_loss.is_none());
    }

    #[test]
    fn test_nesterov_restart_on_increase() {
        let mut optimizer = NesterovAcceleratedGradient::new(NesterovAcceleratedGradientConfig {
            learning_rate: 1e-3,
            momentum: 0.9,
            weight_decay: 0.0,
            restart_on_increase: true,
        });

        // Set initial loss
        optimizer.set_current_loss(1.0);
        assert_eq!(optimizer.previous_loss, Some(1.0));

        // Increasing loss should trigger restart
        optimizer.set_current_loss(1.5);
        assert_eq!(optimizer.previous_loss, Some(1.5));
    }

    #[test]
    fn test_heavy_ball_config() {
        let config = HeavyBallConfig::default();
        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.beta, 0.9);
        assert_eq!(config.weight_decay, 0.0);
        assert!(!config.adaptive_momentum);
    }

    #[test]
    fn test_heavy_ball_creation() {
        let optimizer = HeavyBall::with_defaults(1e-3, 0.9);
        assert_eq!(optimizer.get_lr(), 1e-3);
        assert_eq!(optimizer.current_step, 0);
        assert_eq!(optimizer.get_config().beta, 0.9);
    }

    #[test]
    fn test_heavy_ball_adaptive_momentum() {
        let optimizer = HeavyBall::new(HeavyBallConfig {
            learning_rate: 1e-3,
            beta: 0.9,
            weight_decay: 0.0,
            adaptive_momentum: true,
        });

        assert!(optimizer.config.adaptive_momentum);
    }

    #[test]
    fn test_fista_config() {
        let config = FISTAConfig::default();
        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.threshold, 1e-4);
        assert!(config.adaptive_restart);
        assert_eq!(config.weight_decay, 0.0);
    }

    #[test]
    fn test_fista_creation() {
        let optimizer = FISTA::with_defaults(1e-3, 1e-4);
        assert_eq!(optimizer.get_lr(), 1e-3);
        assert_eq!(optimizer.current_step, 0);
        assert_eq!(optimizer.momentum_coefficient, 1.0);
        assert_eq!(optimizer.previous_momentum, 1.0);
    }

    #[test]
    fn test_fista_momentum_update() {
        let mut optimizer = FISTA::with_defaults(1e-3, 1e-4);

        // Momentum coefficient should update with step (increment step first)
        optimizer.current_step = 1;
        optimizer.update_momentum_coefficient();
        assert!(optimizer.momentum_coefficient > 1.0);
        assert_eq!(optimizer.previous_momentum, 1.0);

        let prev_momentum = optimizer.momentum_coefficient;
        optimizer.current_step = 2;
        optimizer.update_momentum_coefficient();
        assert!(optimizer.momentum_coefficient > prev_momentum);
    }

    #[test]
    fn test_adaptive_batch_sizing_config() {
        let config = AdaptiveBatchSizingConfig::default();
        assert_eq!(config.initial_batch_size, 32);
        assert_eq!(config.min_batch_size, 8);
        assert_eq!(config.max_batch_size, 512);
        assert_eq!(config.gradient_variance_tolerance, 0.1);
        assert_eq!(config.lr_adaptation_factor, 0.8);
        assert_eq!(config.variance_window_size, 10);
        assert_eq!(config.increase_threshold, 0.05);
        assert_eq!(config.decrease_threshold, 0.2);
    }

    #[test]
    fn test_adaptive_batch_sizing_creation() {
        let abs = AdaptiveBatchSizing::with_defaults(64, 16, 256);
        assert_eq!(abs.current_batch_size(), 64);
        assert_eq!(abs.get_config().min_batch_size, 16);
        assert_eq!(abs.get_config().max_batch_size, 256);
    }

    #[test]
    fn test_adaptive_batch_sizing_lr_adjustment() {
        let abs = AdaptiveBatchSizing::with_defaults(64, 16, 256);
        let lr_adj = abs.get_lr_adjustment(32);
        assert!(lr_adj > 0.0);
        assert!(lr_adj < 2.0);
    }

    #[test]
    fn test_adaptive_batch_sizing_reset() {
        let mut abs = AdaptiveBatchSizing::with_defaults(64, 16, 256);
        abs.current_step = 10;
        abs.reset();
        assert_eq!(abs.current_step, 0);
        assert_eq!(abs.current_batch_size(), 64);
    }

    #[test]
    fn test_loss_surface_smoothing_config() {
        let config = LossSurfaceSmoothingConfig::default();
        assert_eq!(config.smoothing_strength, 0.1);
        assert_eq!(config.noise_variance, 1e-4);
        assert_eq!(config.ema_decay, 0.9);
        assert_eq!(config.averaging_window, 5);
        assert!(config.use_gradient_averaging);
        assert!(!config.use_noise_injection);
    }

    #[test]
    fn test_loss_surface_smoothing_creation() {
        let lss = LossSurfaceSmoothing::with_defaults(0.2, true);
        assert_eq!(lss.get_config().smoothing_strength, 0.2);
        assert!(lss.get_config().use_noise_injection);
        assert_eq!(lss.current_step, 0);
    }

    #[test]
    fn test_loss_surface_smoothing_statistics() {
        let lss = LossSurfaceSmoothing::with_defaults(0.1, false);
        let stats = lss.get_statistics();
        assert_eq!(stats.get("current_step"), Some(&0.0));
        assert_eq!(stats.get("num_tracked_params"), Some(&0.0));
        assert_eq!(stats.get("smoothing_strength"), Some(&0.1));
        assert_eq!(stats.get("ema_decay"), Some(&0.9));
    }

    #[test]
    fn test_loss_surface_smoothing_reset() {
        let mut lss = LossSurfaceSmoothing::with_defaults(0.1, false);
        lss.current_step = 5;
        lss.reset();
        assert_eq!(lss.current_step, 0);
    }
}
