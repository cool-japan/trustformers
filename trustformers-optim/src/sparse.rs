//! # Sparse Momentum Methods
//!
//! This module provides optimizers specifically designed for sparse gradients,
//! commonly encountered in embedding layers, large language models, and recommendation systems.
//! These optimizers are memory-efficient and only update parameters that receive non-zero gradients.
//!
//! ## Benefits
//! - Memory efficient for sparse models (embeddings, transformers)
//! - Faster convergence on sparse data
//! - Reduced computation for inactive parameters
//! - Better handling of rare features

use crate::optimizer::OptimizerState;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::tensor::Tensor;

/// Configuration for sparse optimization methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseConfig {
    /// Sparsity threshold below which gradients are considered zero
    pub sparsity_threshold: f32,
    /// Maximum number of active parameters to track
    pub max_active_params: Option<usize>,
    /// Whether to use lazy updates for momentum
    pub lazy_updates: bool,
    /// Frequency of momentum state cleanup (steps)
    pub cleanup_frequency: usize,
    /// Whether to compress inactive momentum states
    pub compress_inactive: bool,
}

impl Default for SparseConfig {
    fn default() -> Self {
        Self {
            sparsity_threshold: 1e-8,
            max_active_params: None,
            lazy_updates: true,
            cleanup_frequency: 1000,
            compress_inactive: false,
        }
    }
}

/// Sparse momentum state for a parameter.
#[derive(Debug, Clone)]
pub struct SparseMomentumState {
    /// Momentum buffer (only for active indices)
    pub momentum: HashMap<usize, f32>,
    /// Last update step for each active index
    pub last_update: HashMap<usize, usize>,
    /// Accumulated gradient norm (for adaptive methods)
    pub grad_norm_acc: HashMap<usize, f32>,
    /// Whether state is compressed
    pub is_compressed: bool,
}

impl Default for SparseMomentumState {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseMomentumState {
    pub fn new() -> Self {
        Self {
            momentum: HashMap::new(),
            last_update: HashMap::new(),
            grad_norm_acc: HashMap::new(),
            is_compressed: false,
        }
    }

    /// Get the number of active parameters.
    pub fn num_active(&self) -> usize {
        self.momentum.len()
    }

    /// Apply lazy momentum updates.
    pub fn apply_lazy_update(&mut self, current_step: usize, decay: f32) {
        for (idx, momentum) in self.momentum.iter_mut() {
            if let Some(&last_step) = self.last_update.get(idx) {
                let steps_skipped = current_step - last_step - 1;
                if steps_skipped > 0 {
                    // Apply exponential decay for skipped steps
                    *momentum *= decay.powi(steps_skipped as i32);
                }
            }
        }
    }

    /// Clean up old momentum states.
    pub fn cleanup(&mut self, max_age_steps: usize, current_step: usize) {
        let mut to_remove = Vec::new();

        for (idx, &last_step) in &self.last_update {
            if current_step - last_step > max_age_steps {
                to_remove.push(*idx);
            }
        }

        for idx in to_remove {
            self.momentum.remove(&idx);
            self.last_update.remove(&idx);
            self.grad_norm_acc.remove(&idx);
        }
    }

    /// Compress inactive momentum states.
    pub fn compress(&mut self) {
        if self.is_compressed {
            return;
        }

        // Remove very small momentum values
        let threshold = 1e-10;
        self.momentum.retain(|_, &mut v| v.abs() > threshold);
        self.grad_norm_acc.retain(|_, &mut v| v > threshold);

        self.is_compressed = true;
    }

    /// Decompress momentum states.
    pub fn decompress(&mut self) {
        self.is_compressed = false;
    }
}

/// Sparse SGD with momentum optimizer.
#[derive(Debug)]
pub struct SparseSGD {
    learning_rate: f32,
    momentum: f32,
    dampening: f32,
    weight_decay: f32,
    nesterov: bool,
    config: SparseConfig,
    momentum_states: HashMap<usize, SparseMomentumState>,
    current_step: usize,
}

impl SparseSGD {
    pub fn new(
        learning_rate: f32,
        momentum: f32,
        dampening: f32,
        weight_decay: f32,
        nesterov: bool,
        config: SparseConfig,
    ) -> Self {
        Self {
            learning_rate,
            momentum,
            dampening,
            weight_decay,
            nesterov,
            config,
            momentum_states: HashMap::new(),
            current_step: 0,
        }
    }

    /// Create with default sparse configuration.
    pub fn with_default_config(
        learning_rate: f32,
        momentum: f32,
        dampening: f32,
        weight_decay: f32,
        nesterov: bool,
    ) -> Self {
        Self::new(
            learning_rate,
            momentum,
            dampening,
            weight_decay,
            nesterov,
            SparseConfig::default(),
        )
    }

    /// Get sparse indices from gradient tensor.
    fn get_sparse_indices(&self, gradient: &Tensor) -> Result<Vec<usize>> {
        let grad_data = gradient.data()?;
        let indices: Vec<usize> = grad_data
            .iter()
            .enumerate()
            .filter_map(
                |(i, &val)| {
                    if val.abs() > self.config.sparsity_threshold {
                        Some(i)
                    } else {
                        None
                    }
                },
            )
            .collect();

        // Limit active parameters if configured
        if let Some(max_active) = self.config.max_active_params {
            if indices.len() > max_active {
                // Keep indices with largest gradients
                let mut indexed_grads: Vec<(usize, f32)> =
                    indices.iter().map(|&i| (i, grad_data[i].abs())).collect();
                indexed_grads.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                return Ok(indexed_grads.into_iter().take(max_active).map(|(i, _)| i).collect());
            }
        }

        Ok(indices)
    }

    /// Update sparse momentum for a parameter.
    fn update_sparse_momentum(
        &mut self,
        param_id: usize,
        gradient: &Tensor,
        parameter: &mut Tensor,
    ) -> Result<()> {
        let sparse_indices = self.get_sparse_indices(gradient)?;
        if sparse_indices.is_empty() {
            return Ok(());
        }

        let grad_data = gradient.data()?;
        let mut param_data = parameter.data()?;

        // Get or create momentum state
        let momentum_state = self.momentum_states.entry(param_id).or_default();

        // Apply lazy updates if enabled
        if self.config.lazy_updates {
            momentum_state.apply_lazy_update(self.current_step, self.momentum);
        }

        // Update momentum for each sparse index
        for &idx in &sparse_indices {
            let mut grad_val = grad_data[idx];

            // Apply weight decay
            if self.weight_decay != 0.0 {
                grad_val += self.weight_decay * param_data[idx];
            }

            // Update momentum
            let momentum_val = momentum_state.momentum.get(&idx).copied().unwrap_or(0.0);
            let new_momentum = self.momentum * momentum_val + (1.0 - self.dampening) * grad_val;
            momentum_state.momentum.insert(idx, new_momentum);
            momentum_state.last_update.insert(idx, self.current_step);

            // Apply update
            let update = if self.nesterov {
                grad_val + self.momentum * new_momentum
            } else {
                new_momentum
            };

            param_data[idx] -= self.learning_rate * update;
        }

        // Update parameter tensor
        *parameter = Tensor::from_vec(param_data, &parameter.shape())?;

        Ok(())
    }

    /// Get momentum statistics.
    pub fn get_momentum_stats(&self) -> HashMap<usize, usize> {
        self.momentum_states
            .iter()
            .map(|(&param_id, state)| (param_id, state.num_active()))
            .collect()
    }

    /// Total number of active momentum states across all parameters.
    pub fn total_active_states(&self) -> usize {
        self.momentum_states.values().map(|s| s.num_active()).sum()
    }

    /// Cleanup old momentum states for all parameters.
    pub fn cleanup_momentum_states(&mut self) {
        if self.current_step % self.config.cleanup_frequency == 0 {
            let max_age = self.config.cleanup_frequency * 2;
            for state in self.momentum_states.values_mut() {
                state.cleanup(max_age, self.current_step);
                if self.config.compress_inactive {
                    state.compress();
                }
            }
        }
    }
}

impl OptimizerState for SparseSGD {
    fn zero_grad(&mut self) -> Result<()> {
        // For sparse optimizers, we don't need to explicitly zero gradients
        // since we only process non-zero gradients
        Ok(())
    }

    fn step(&mut self, parameters: &mut [Tensor]) -> Result<()> {
        self.current_step += 1;

        for (param_id, parameter) in parameters.iter_mut().enumerate() {
            // For sparse optimizers, we assume gradients are already computed
            // and available in the parameter's grad field
            if let Ok(gradient) = parameter.grad() {
                self.update_sparse_momentum(param_id, &gradient, parameter)?;
            }
        }

        // Periodic cleanup
        self.cleanup_momentum_states();

        Ok(())
    }

    fn get_lr(&self) -> f32 {
        self.learning_rate
    }

    fn set_lr(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut state = HashMap::new();

        // Save hyperparameters
        state.insert(
            "learning_rate".to_string(),
            Tensor::scalar(self.learning_rate)?,
        );
        state.insert("momentum".to_string(), Tensor::scalar(self.momentum)?);
        state.insert("dampening".to_string(), Tensor::scalar(self.dampening)?);
        state.insert(
            "weight_decay".to_string(),
            Tensor::scalar(self.weight_decay)?,
        );
        state.insert(
            "nesterov".to_string(),
            Tensor::scalar(self.nesterov as i32 as f32)?,
        );
        state.insert(
            "current_step".to_string(),
            Tensor::scalar(self.current_step as f32)?,
        );

        // Save momentum states (this is simplified - real implementation would be more complex)
        for (&param_id, momentum_state) in &self.momentum_states {
            let num_active = momentum_state.num_active();
            state.insert(
                format!("momentum_state_{}_active_count", param_id),
                Tensor::scalar(num_active as f32)?,
            );
        }

        Ok(state)
    }

    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()> {
        // Load hyperparameters
        if let Some(lr_tensor) = state.get("learning_rate") {
            self.learning_rate = lr_tensor.to_scalar()?;
        }
        if let Some(momentum_tensor) = state.get("momentum") {
            self.momentum = momentum_tensor.to_scalar()?;
        }
        if let Some(dampening_tensor) = state.get("dampening") {
            self.dampening = dampening_tensor.to_scalar()?;
        }
        if let Some(wd_tensor) = state.get("weight_decay") {
            self.weight_decay = wd_tensor.to_scalar()?;
        }
        if let Some(nesterov_tensor) = state.get("nesterov") {
            self.nesterov = nesterov_tensor.to_scalar()? > 0.5;
        }
        if let Some(step_tensor) = state.get("current_step") {
            self.current_step = step_tensor.to_scalar()? as usize;
        }

        // Note: Loading full momentum states would require more complex serialization
        // This is a simplified implementation

        Ok(())
    }
}

/// Sparse Adam optimizer for handling sparse gradients efficiently.
#[derive(Debug)]
pub struct SparseAdam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    config: SparseConfig,
    momentum_states: HashMap<usize, SparseMomentumState>,
    variance_states: HashMap<usize, HashMap<usize, f32>>,
    current_step: usize,
}

impl SparseAdam {
    pub fn new(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        config: SparseConfig,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            config,
            momentum_states: HashMap::new(),
            variance_states: HashMap::new(),
            current_step: 0,
        }
    }

    pub fn with_default_config(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Self {
        Self::new(
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            SparseConfig::default(),
        )
    }

    fn get_sparse_indices(&self, gradient: &Tensor) -> Result<Vec<usize>> {
        let grad_data = gradient.data()?;
        Ok(grad_data
            .iter()
            .enumerate()
            .filter_map(
                |(i, &val)| {
                    if val.abs() > self.config.sparsity_threshold {
                        Some(i)
                    } else {
                        None
                    }
                },
            )
            .collect())
    }

    fn update_sparse_adam(
        &mut self,
        param_id: usize,
        gradient: &Tensor,
        parameter: &mut Tensor,
    ) -> Result<()> {
        let sparse_indices = self.get_sparse_indices(gradient)?;
        if sparse_indices.is_empty() {
            return Ok(());
        }

        let grad_data = gradient.data()?;
        let mut param_data = parameter.data()?;

        // Get or create states
        let momentum_state = self.momentum_states.entry(param_id).or_default();
        let variance_state = self.variance_states.entry(param_id).or_default();

        // Bias correction terms
        let bias_correction1 = 1.0 - self.beta1.powi(self.current_step as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.current_step as i32);

        // Update sparse parameters
        for &idx in &sparse_indices {
            let mut grad_val = grad_data[idx];

            // Apply weight decay
            if self.weight_decay != 0.0 {
                grad_val += self.weight_decay * param_data[idx];
            }

            // Update biased first moment estimate
            let momentum_val = momentum_state.momentum.get(&idx).copied().unwrap_or(0.0);
            let new_momentum = self.beta1 * momentum_val + (1.0 - self.beta1) * grad_val;
            momentum_state.momentum.insert(idx, new_momentum);

            // Update biased second raw moment estimate
            let variance_val = variance_state.get(&idx).copied().unwrap_or(0.0);
            let new_variance = self.beta2 * variance_val + (1.0 - self.beta2) * grad_val * grad_val;
            variance_state.insert(idx, new_variance);

            // Compute bias-corrected first and second moment estimates
            let momentum_corrected = new_momentum / bias_correction1;
            let variance_corrected = new_variance / bias_correction2;

            // Update parameter
            let denom = variance_corrected.sqrt() + self.epsilon;
            param_data[idx] -= self.learning_rate * momentum_corrected / denom;

            momentum_state.last_update.insert(idx, self.current_step);
        }

        // Update parameter tensor
        *parameter = Tensor::from_vec(param_data, &parameter.shape())?;

        Ok(())
    }
}

impl OptimizerState for SparseAdam {
    fn zero_grad(&mut self) -> Result<()> {
        Ok(())
    }

    fn step(&mut self, parameters: &mut [Tensor]) -> Result<()> {
        self.current_step += 1;

        for (param_id, parameter) in parameters.iter_mut().enumerate() {
            if let Ok(gradient) = parameter.grad() {
                self.update_sparse_adam(param_id, &gradient, parameter)?;
            }
        }

        Ok(())
    }

    fn get_lr(&self) -> f32 {
        self.learning_rate
    }

    fn set_lr(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut state = HashMap::new();
        state.insert(
            "learning_rate".to_string(),
            Tensor::scalar(self.learning_rate)?,
        );
        state.insert("beta1".to_string(), Tensor::scalar(self.beta1)?);
        state.insert("beta2".to_string(), Tensor::scalar(self.beta2)?);
        state.insert("epsilon".to_string(), Tensor::scalar(self.epsilon)?);
        state.insert(
            "weight_decay".to_string(),
            Tensor::scalar(self.weight_decay)?,
        );
        state.insert(
            "current_step".to_string(),
            Tensor::scalar(self.current_step as f32)?,
        );
        Ok(state)
    }

    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()> {
        if let Some(lr) = state.get("learning_rate") {
            self.learning_rate = lr.to_scalar()?;
        }
        if let Some(beta1) = state.get("beta1") {
            self.beta1 = beta1.to_scalar()?;
        }
        if let Some(beta2) = state.get("beta2") {
            self.beta2 = beta2.to_scalar()?;
        }
        if let Some(eps) = state.get("epsilon") {
            self.epsilon = eps.to_scalar()?;
        }
        if let Some(wd) = state.get("weight_decay") {
            self.weight_decay = wd.to_scalar()?;
        }
        if let Some(step) = state.get("current_step") {
            self.current_step = step.to_scalar()? as usize;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_config_default() {
        let config = SparseConfig::default();
        assert_eq!(config.sparsity_threshold, 1e-8);
        assert!(config.max_active_params.is_none());
        assert!(config.lazy_updates);
        assert_eq!(config.cleanup_frequency, 1000);
        assert!(!config.compress_inactive);
    }

    #[test]
    fn test_sparse_momentum_state() {
        let mut state = SparseMomentumState::new();
        assert_eq!(state.num_active(), 0);

        state.momentum.insert(0, 1.0);
        state.momentum.insert(5, 2.0);
        assert_eq!(state.num_active(), 2);

        state.cleanup(0, 100);
        assert_eq!(state.num_active(), 2); // No cleanup without last_update entries
    }

    #[test]
    fn test_sparse_sgd_creation() {
        let optimizer = SparseSGD::with_default_config(0.01, 0.9, 0.0, 1e-4, false);
        assert_eq!(optimizer.get_lr(), 0.01);
        assert_eq!(optimizer.total_active_states(), 0);
    }

    #[test]
    fn test_sparse_adam_creation() {
        let optimizer = SparseAdam::with_default_config(1e-3, 0.9, 0.999, 1e-8, 0.01);
        assert_eq!(optimizer.get_lr(), 1e-3);
        assert_eq!(optimizer.current_step, 0);
    }

    #[test]
    fn test_sparse_sgd_lr_update() {
        let mut optimizer = SparseSGD::with_default_config(0.01, 0.9, 0.0, 1e-4, false);
        assert_eq!(optimizer.get_lr(), 0.01);

        optimizer.set_lr(0.001);
        assert_eq!(optimizer.get_lr(), 0.001);
    }
}
