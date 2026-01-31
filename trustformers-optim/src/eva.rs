//! # EVA Optimizer
//!
//! EVA (Exponential Moving Average with Variance Adaptation) is a state-of-the-art optimizer
//! that adapts the learning rate based on the variance of gradient estimates.
//!
//! ## Key Features
//!
//! - **Adaptive Learning Rate**: Uses gradient variance to adapt learning rate
//! - **Exponential Moving Averages**: Maintains momentum and variance estimates
//! - **Robustness**: More stable than Adam in certain scenarios
//! - **Computational Efficiency**: Low overhead compared to second-order methods
//!
//! ## Algorithm
//!
//! EVA updates parameters using:
//! 1. Exponential moving average of gradients (momentum)
//! 2. Exponential moving average of squared gradients (variance)
//! 3. Variance-adapted learning rate scaling
//! 4. Optional bias correction
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use trustformers_optim::EVA;
//!
//! let mut optimizer = EVA::new(
//!     1e-3,   // learning_rate
//!     0.9,    // beta1
//!     0.999,  // beta2
//!     1e-8,   // epsilon
//!     0.01,   // weight_decay
//!     true,   // variance_adaptation
//! );
//! ```

use crate::common::{OptimizerState, StateMemoryStats};
use crate::traits::StatefulOptimizer;
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

/// Configuration for EVA optimizer.
#[derive(Debug, Clone)]
pub struct EVAConfig {
    /// Learning rate
    pub lr: f32,
    /// First moment coefficient
    pub beta1: f32,
    /// Second moment coefficient
    pub beta2: f32,
    /// Term added for numerical stability
    pub eps: f32,
    /// Weight decay (L2 penalty)
    pub weight_decay: f32,
    /// Whether to use variance adaptation
    pub variance_adaptation: bool,
    /// Whether to use bias correction
    pub bias_correction: bool,
    /// Variance adaptation strength
    pub adaptation_strength: f32,
}

impl Default for EVAConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            variance_adaptation: true,
            bias_correction: true,
            adaptation_strength: 1.0,
        }
    }
}

/// EVA (Exponential Moving Average with Variance Adaptation) optimizer.
#[derive(Debug)]
pub struct EVA {
    config: EVAConfig,
    state: OptimizerState,
    exp_avg: HashMap<String, Vec<f32>>,
    exp_avg_sq: HashMap<String, Vec<f32>>,
    var_adaptation: HashMap<String, Vec<f32>>,
    step_count: usize,
}

impl EVA {
    /// Creates a new EVA optimizer with default configuration.
    pub fn new(
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        variance_adaptation: bool,
    ) -> Self {
        let config = EVAConfig {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            variance_adaptation,
            bias_correction: true,
            adaptation_strength: 1.0,
        };

        Self::with_config(config)
    }

    /// Creates a new EVA optimizer with custom configuration.
    pub fn with_config(config: EVAConfig) -> Self {
        Self {
            config,
            state: OptimizerState::new(),
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
            var_adaptation: HashMap::new(),
            step_count: 0,
        }
    }

    /// Convenience constructor for EVA with AdamW-like settings.
    pub fn adamw_like(lr: f32, weight_decay: f32) -> Self {
        Self::new(lr, 0.9, 0.999, 1e-8, weight_decay, true)
    }

    /// Convenience constructor for EVA with variance adaptation disabled.
    pub fn no_variance_adaptation(lr: f32, beta1: f32, beta2: f32, eps: f32) -> Self {
        Self::new(lr, beta1, beta2, eps, 0.0, false)
    }

    /// Gets the current learning rate.
    pub fn get_lr(&self) -> f32 {
        self.config.lr
    }

    /// Sets the learning rate.
    pub fn set_lr(&mut self, lr: f32) {
        self.config.lr = lr;
    }

    /// Gets current optimizer configuration.
    pub fn config(&self) -> &EVAConfig {
        &self.config
    }

    /// Gets memory statistics for the optimizer state.
    pub fn memory_stats(&self) -> StateMemoryStats {
        let mut total_parameters = 0;
        #[allow(dead_code)]
        let mut _total_buffers = 0;
        #[allow(unused_assignments)]
        for buffer in self.exp_avg.values() {
            total_parameters += buffer.len();
            _total_buffers += 1;
        }

        for buffer in self.exp_avg_sq.values() {
            total_parameters += buffer.len();
            _total_buffers += 1;
        }

        if self.config.variance_adaptation {
            for buffer in self.var_adaptation.values() {
                total_parameters += buffer.len();
                _total_buffers += 1;
            }
        }

        StateMemoryStats {
            momentum_elements: total_parameters,
            variance_elements: total_parameters,
            third_moment_elements: if self.config.variance_adaptation {
                total_parameters
            } else {
                0
            },
            total_bytes: total_parameters * 4, // f32 = 4 bytes
            num_parameters: total_parameters,
        }
    }

    /// Computes variance adaptation factor.
    #[allow(dead_code)]
    fn compute_variance_adaptation(&self, grad_var: f32, step: usize) -> f32 {
        if !self.config.variance_adaptation || step == 0 {
            return 1.0;
        }

        let adaptation = (grad_var + self.config.eps).sqrt();
        let strength = self.config.adaptation_strength;

        // Apply strength and clamp to reasonable range
        let factor = 1.0 / (1.0 + strength * adaptation);
        factor.clamp(0.1, 2.0)
    }
}

impl Optimizer for EVA {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        self.step_count += 1;

        match (parameter, grad) {
            (Tensor::F32(param), Tensor::F32(grad_data)) => {
                let param_id = format!("{:p}", param.as_ptr());
                let size = grad_data.len();

                // Initialize state if needed
                let exp_avg =
                    self.exp_avg.entry(param_id.clone()).or_insert_with(|| vec![0.0; size]);
                let exp_avg_sq =
                    self.exp_avg_sq.entry(param_id.clone()).or_insert_with(|| vec![0.0; size]);
                let mut var_adapt = if self.config.variance_adaptation {
                    Some(
                        self.var_adaptation
                            .entry(param_id.clone())
                            .or_insert_with(|| vec![0.0; size]),
                    )
                } else {
                    None
                };

                // Check buffer sizes
                if exp_avg.len() != size || exp_avg_sq.len() != size {
                    return Err(TrustformersError::tensor_op_error(
                        "EVA buffer size mismatch",
                        "EVA::update",
                    ));
                }

                if let Some(ref va) = var_adapt {
                    if va.len() != size {
                        return Err(TrustformersError::tensor_op_error(
                            "EVA variance adaptation buffer size mismatch",
                            "EVA::update",
                        ));
                    }
                }

                // Compute bias correction factors
                let bias_correction1 = if self.config.bias_correction {
                    1.0 - self.config.beta1.powi(self.step_count as i32)
                } else {
                    1.0
                };

                let bias_correction2 = if self.config.bias_correction {
                    1.0 - self.config.beta2.powi(self.step_count as i32)
                } else {
                    1.0
                };

                // Compute gradient variance for adaptation
                let grad_var = if self.config.variance_adaptation {
                    let mean_grad = grad_data.iter().sum::<f32>() / size as f32;
                    grad_data.iter().map(|&g| (g - mean_grad).powi(2)).sum::<f32>() / size as f32
                } else {
                    0.0
                };

                let variance_factor = if self.config.variance_adaptation && self.step_count > 0 {
                    let adaptation = (grad_var + self.config.eps).sqrt();
                    let strength = self.config.adaptation_strength;
                    let factor = 1.0 / (1.0 + strength * adaptation);
                    factor.clamp(0.1, 2.0)
                } else {
                    1.0
                };

                // Update parameters
                for (i, ((&g, p), (m, v))) in grad_data
                    .iter()
                    .zip(param.iter_mut())
                    .zip(exp_avg.iter_mut().zip(exp_avg_sq.iter_mut()))
                    .enumerate()
                {
                    // Apply weight decay
                    let grad_with_decay = if self.config.weight_decay > 0.0 {
                        g + self.config.weight_decay * (*p)
                    } else {
                        g
                    };

                    // Update biased first moment estimate
                    *m = self.config.beta1 * (*m) + (1.0 - self.config.beta1) * grad_with_decay;

                    // Update biased second moment estimate
                    *v = self.config.beta2 * (*v)
                        + (1.0 - self.config.beta2) * grad_with_decay * grad_with_decay;

                    // Update variance adaptation if enabled
                    if let Some(ref mut va) = var_adapt {
                        va[i] = 0.9 * va[i] + 0.1 * grad_with_decay.abs();
                    }

                    // Compute bias-corrected estimates
                    let m_hat = *m / bias_correction1;
                    let v_hat = *v / bias_correction2;

                    // Apply variance adaptation
                    let adapted_lr = self.config.lr * variance_factor;

                    // Update parameter
                    *p -= adapted_lr * m_hat / (v_hat.sqrt() + self.config.eps);
                }

                Ok(())
            },
            _ => Err(TrustformersError::tensor_op_error(
                "EVA optimizer only supports F32 tensors",
                "EVA::update",
            )),
        }
    }

    fn zero_grad(&mut self) {
        // EVA doesn't accumulate gradients, so this is a no-op
    }

    fn step(&mut self) {
        // Update is called per parameter, so step is a no-op
    }

    fn get_lr(&self) -> f32 {
        self.config.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.lr = lr;
    }
}

impl StatefulOptimizer for EVA {
    type Config = EVAConfig;
    type State = OptimizerState;

    fn state(&self) -> &OptimizerState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut OptimizerState {
        &mut self.state
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn memory_usage(&self) -> StateMemoryStats {
        self.memory_stats()
    }

    fn reset_state(&mut self) {
        self.exp_avg.clear();
        self.exp_avg_sq.clear();
        self.var_adaptation.clear();
        self.step_count = 0;
        self.state = OptimizerState::new();
    }

    fn num_parameters(&self) -> usize {
        self.exp_avg.values().map(|v| v.len()).sum()
    }

    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut dict = HashMap::new();

        for (key, value) in &self.exp_avg {
            dict.insert(format!("exp_avg_{}", key), Tensor::new(value.clone())?);
        }

        for (key, value) in &self.exp_avg_sq {
            dict.insert(format!("exp_avg_sq_{}", key), Tensor::new(value.clone())?);
        }

        if self.config.variance_adaptation {
            for (key, value) in &self.var_adaptation {
                dict.insert(
                    format!("var_adaptation_{}", key),
                    Tensor::new(value.clone())?,
                );
            }
        }

        dict.insert(
            "step_count".to_string(),
            Tensor::new(vec![self.step_count as f32])?,
        );

        Ok(dict)
    }

    fn load_state_dict(&mut self, state_dict: HashMap<String, Tensor>) -> Result<()> {
        // Load step count
        if let Some(Tensor::F32(data)) = state_dict.get("step_count") {
            if !data.is_empty() {
                self.step_count = data[0] as usize;
            }
        }

        // Load exp_avg
        for (key, value) in &state_dict {
            if let Some(param_key) = key.strip_prefix("exp_avg_") {
                if let Tensor::F32(data) = value {
                    self.exp_avg.insert(
                        param_key.to_string(),
                        data.as_slice().expect("F32 tensor should have valid slice").to_vec(),
                    );
                }
            }
        }

        // Load exp_avg_sq
        for (key, value) in &state_dict {
            if let Some(param_key) = key.strip_prefix("exp_avg_sq_") {
                if let Tensor::F32(data) = value {
                    self.exp_avg_sq.insert(
                        param_key.to_string(),
                        data.as_slice().expect("F32 tensor should have valid slice").to_vec(),
                    );
                }
            }
        }

        // Load variance adaptation
        if self.config.variance_adaptation {
            for (key, value) in &state_dict {
                if let Some(param_key) = key.strip_prefix("var_adaptation_") {
                    if let Tensor::F32(data) = value {
                        self.var_adaptation.insert(
                            param_key.to_string(),
                            data.as_slice().expect("F32 tensor should have valid slice").to_vec(),
                        );
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::tensor::Tensor;

    #[test]
    fn test_eva_creation() {
        let optimizer = EVA::new(1e-3, 0.9, 0.999, 1e-8, 0.01, true);
        assert_eq!(optimizer.get_lr(), 1e-3);
        assert_eq!(optimizer.config().beta1, 0.9);
        assert_eq!(optimizer.config().beta2, 0.999);
        assert_eq!(optimizer.config().eps, 1e-8);
        assert_eq!(optimizer.config().weight_decay, 0.01);
        assert!(optimizer.config().variance_adaptation);
    }

    #[test]
    fn test_eva_adamw_like() {
        let optimizer = EVA::adamw_like(1e-3, 0.01);
        assert_eq!(optimizer.get_lr(), 1e-3);
        assert_eq!(optimizer.config().weight_decay, 0.01);
        assert!(optimizer.config().variance_adaptation);
    }

    #[test]
    fn test_eva_no_variance_adaptation() {
        let optimizer = EVA::no_variance_adaptation(1e-3, 0.9, 0.999, 1e-8);
        assert_eq!(optimizer.get_lr(), 1e-3);
        assert!(!optimizer.config().variance_adaptation);
    }

    #[test]
    fn test_eva_lr_setter() {
        let mut optimizer = EVA::new(1e-3, 0.9, 0.999, 1e-8, 0.01, true);
        optimizer.set_lr(2e-3);
        assert_eq!(optimizer.get_lr(), 2e-3);
    }

    #[test]
    fn test_eva_memory_stats() {
        let optimizer = EVA::new(1e-3, 0.9, 0.999, 1e-8, 0.01, true);
        let stats = optimizer.memory_stats();
        assert_eq!(stats.num_parameters, 0);
        assert_eq!(stats.total_bytes, 0);
    }

    #[test]
    fn test_eva_variance_adaptation() {
        let optimizer = EVA::new(1e-3, 0.9, 0.999, 1e-8, 0.01, true);
        let factor = optimizer.compute_variance_adaptation(0.1, 1);
        assert!(factor > 0.1 && factor < 2.0);
    }

    #[test]
    fn test_eva_state_dict() {
        let optimizer = EVA::new(1e-3, 0.9, 0.999, 1e-8, 0.01, true);
        let state_dict = optimizer.state_dict();
        assert!(state_dict.unwrap().contains_key("step_count"));
    }

    #[test]
    fn test_eva_load_state_dict() {
        let mut optimizer = EVA::new(1e-3, 0.9, 0.999, 1e-8, 0.01, true);
        let mut state_dict = HashMap::new();
        state_dict.insert("step_count".to_string(), Tensor::new(vec![10.0]).unwrap());

        optimizer.load_state_dict(state_dict).unwrap();
        assert_eq!(optimizer.step_count, 10);
    }
}
