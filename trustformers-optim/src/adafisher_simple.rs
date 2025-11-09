//! # AdaFisher: Adaptive Second Order Optimization via Fisher Information (Simplified)
//!
//! This is a simplified implementation of AdaFisher that uses basic tensor operations
//! available in the TrustformeRS core. The full implementation would require more
//! advanced tensor operations for true Fisher information computation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::{errors::Result, tensor::Tensor, traits::Optimizer};

use crate::{common::StateMemoryStats, traits::StatefulOptimizer};

/// Configuration for AdaFisher optimizer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaFisherConfig {
    pub learning_rate: f32,
    pub fisher_decay: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
}

impl Default for AdaFisherConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            fisher_decay: 0.95,
            epsilon: 1e-6,
            weight_decay: 0.01,
        }
    }
}

/// Simplified AdaFisher optimizer state.
#[derive(Clone, Debug)]
pub struct AdaFisherState {
    pub momentum: Tensor,
    pub variance: Tensor,
    pub step: usize,
}

/// Simplified AdaFisher optimizer.
#[derive(Clone, Debug)]
pub struct AdaFisher {
    config: AdaFisherConfig,
    states: HashMap<String, AdaFisherState>,
    step: usize,
    memory_stats: StateMemoryStats,
}

impl AdaFisher {
    pub fn new(learning_rate: f32, fisher_decay: f32, epsilon: f32, _block_size: usize) -> Self {
        Self {
            config: AdaFisherConfig {
                learning_rate,
                fisher_decay,
                epsilon,
                weight_decay: 0.01,
            },
            states: HashMap::new(),
            step: 0,
            memory_stats: StateMemoryStats {
                momentum_elements: 0,
                variance_elements: 0,
                third_moment_elements: 0,
                total_bytes: 0,
                num_parameters: 0,
            },
        }
    }

    pub fn for_language_models() -> Self {
        Self::new(3e-4, 0.99, 1e-8, 128)
    }

    pub fn for_image_classification() -> Self {
        Self::new(1e-3, 0.95, 1e-6, 64)
    }

    pub fn fisher_stats(&self) -> HashMap<String, (usize, usize, f32)> {
        self.states
            .iter()
            .map(|(name, state)| (name.clone(), (1, state.step, 64.0)))
            .collect()
    }

    pub fn fisher_memory_usage(&self) -> usize {
        self.states.len() * 1024 // Simplified estimate
    }
}

impl Optimizer for AdaFisher {
    fn update(&mut self, parameter: &mut Tensor, gradient: &Tensor) -> Result<()> {
        // Create a unique parameter ID based on shape and hash of first few elements
        let param_id = format!(
            "param_{}_{:?}_{}",
            self.states.len(),
            parameter.shape(),
            parameter
                .data_f32()
                .unwrap_or_default()
                .get(0..5)
                .unwrap_or(&[])
                .iter()
                .fold(0u64, |acc, &x| acc.wrapping_add(x.to_bits() as u64))
        );

        let state = self.states.entry(param_id).or_insert_with(|| AdaFisherState {
            momentum: Tensor::zeros_like(parameter).unwrap(),
            variance: Tensor::zeros_like(parameter).unwrap(),
            step: 0,
        });

        state.step += 1;

        // Adam-like update with Fisher information approximation
        state.momentum = state
            .momentum
            .mul_scalar(self.config.fisher_decay)?
            .add(&gradient.mul_scalar(1.0 - self.config.fisher_decay)?)?;

        state.variance = state
            .variance
            .mul_scalar(self.config.fisher_decay)?
            .add(&gradient.pow_scalar(2.0)?.mul_scalar(1.0 - self.config.fisher_decay)?)?;

        // Bias correction
        let bias_correction1 = 1.0 - self.config.fisher_decay.powi(state.step as i32);
        let bias_correction2 = 1.0 - self.config.fisher_decay.powi(state.step as i32);

        let corrected_momentum = state.momentum.div_scalar(bias_correction1)?;
        let corrected_variance = state.variance.div_scalar(bias_correction2)?;

        // Update parameter
        let denominator = corrected_variance.sqrt()?.add_scalar(self.config.epsilon)?;
        let update = corrected_momentum.div(&denominator)?.mul_scalar(self.config.learning_rate)?;

        *parameter = parameter.sub(&update)?;

        Ok(())
    }

    fn zero_grad(&mut self) {
        // Nothing to clear for AdaFisher
    }

    fn step(&mut self) {
        self.step += 1;
    }

    fn get_lr(&self) -> f32 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }
}

impl StatefulOptimizer for AdaFisher {
    type Config = AdaFisherConfig;
    type State = StateMemoryStats;

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn state(&self) -> &Self::State {
        &self.memory_stats
    }

    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.memory_stats
    }

    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut state_dict = HashMap::new();
        state_dict.insert("step".to_string(), Tensor::scalar(self.step as f32)?);
        Ok(state_dict)
    }

    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()> {
        if let Some(step_tensor) = state.get("step") {
            self.step = step_tensor.to_scalar()? as usize;
        }
        Ok(())
    }

    fn memory_usage(&self) -> StateMemoryStats {
        self.memory_stats.clone()
    }

    fn reset_state(&mut self) {
        self.states.clear();
        self.step = 0;
    }

    fn num_parameters(&self) -> usize {
        self.states.len()
    }
}
