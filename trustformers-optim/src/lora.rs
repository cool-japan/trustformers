//! # Low-Rank Adaptation (LoRA) Optimizers
//!
//! This module provides optimizers specifically designed for Low-Rank Adaptation (LoRA),
//! a parameter-efficient fine-tuning technique that reduces the number of trainable parameters
//! by decomposing weight updates into low-rank matrices.
//!
//! ## Overview
//!
//! LoRA works by freezing the original model weights and introducing trainable low-rank
//! decomposition matrices A and B such that ∆W = B × A, where the rank r << min(input_dim, output_dim).
//!
//! ## Benefits
//! - Reduces trainable parameters by 10,000x or more
//! - Enables efficient fine-tuning on consumer hardware
//! - Maintains model quality comparable to full fine-tuning
//! - Allows efficient storage and sharing of multiple adaptations

use crate::optimizer::OptimizerState;
use anyhow::{anyhow, Result as AnyhowResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::errors::Result;
use trustformers_core::tensor::Tensor;

/// Configuration for LoRA optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAConfig {
    /// Rank of the low-rank decomposition
    pub rank: usize,
    /// Alpha parameter for scaling (typically rank * 2)
    pub alpha: f32,
    /// Dropout probability for LoRA layers
    pub dropout: f32,
    /// Whether to enable bias training
    pub bias: bool,
    /// Modules to apply LoRA to (e.g., ["query", "key", "value"])
    pub target_modules: Vec<String>,
    /// Whether to merge adapter weights into base model
    pub merge_weights: bool,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.1,
            bias: false,
            target_modules: vec!["query".to_string(), "key".to_string(), "value".to_string()],
            merge_weights: false,
        }
    }
}

/// LoRA adapter containing the low-rank matrices.
#[derive(Debug, Clone)]
pub struct LoRAAdapter {
    /// Low-rank matrix A (input_dim x rank)
    pub lora_a: Tensor,
    /// Low-rank matrix B (rank x output_dim)
    pub lora_b: Tensor,
    /// Scaling factor
    pub scaling: f32,
    /// Whether this adapter is active
    pub active: bool,
}

impl LoRAAdapter {
    /// Creates a new LoRA adapter with random initialization.
    pub fn new(input_dim: usize, output_dim: usize, rank: usize, alpha: f32) -> Result<Self> {
        // Initialize A with small random values, B with zeros (common practice)
        let lora_a = Tensor::randn(&[input_dim, rank])?;
        let lora_b = Tensor::zeros(&[rank, output_dim])?;
        let scaling = alpha / rank as f32;

        Ok(Self {
            lora_a,
            lora_b,
            scaling,
            active: true,
        })
    }

    /// Compute the LoRA update: scaling * B @ A
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.active {
            return Tensor::zeros_like(input);
        }

        // Compute x @ A @ B * scaling
        let intermediate = input.matmul(&self.lora_a)?;
        let output = intermediate.matmul(&self.lora_b)?;
        output.mul_scalar(self.scaling)
    }

    /// Get the effective weight matrix ∆W = scaling * B @ A
    pub fn get_delta_weight(&self) -> Result<Tensor> {
        if !self.active {
            return Err(
                trustformers_core::errors::TrustformersError::tensor_op_error(
                    "Adapter is not active",
                    "get_delta_weight",
                ),
            );
        }

        let delta_w = self.lora_b.matmul(&self.lora_a)?;
        delta_w.mul_scalar(self.scaling)
    }

    /// Merge adapter weights into the base weight matrix
    pub fn merge_into_weight(&self, base_weight: &mut Tensor) -> Result<()> {
        if !self.active {
            return Ok(());
        }

        let delta_w = self.get_delta_weight()?;
        *base_weight = base_weight.add(&delta_w)?;
        Ok(())
    }

    /// Enable or disable this adapter
    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    /// Get the number of trainable parameters
    pub fn num_parameters(&self) -> usize {
        self.lora_a.len() + self.lora_b.len()
    }
}

/// LoRA-specific optimizer that only updates adapter parameters.
pub struct LoRAOptimizer {
    /// Base optimizer for LoRA parameters
    base_optimizer: Box<dyn OptimizerState>,
    /// LoRA adapters by module name
    adapters: HashMap<String, LoRAAdapter>,
    /// LoRA configuration
    config: LoRAConfig,
    /// Frozen base model parameters
    frozen_parameters: HashMap<String, Tensor>,
    /// Learning rate
    learning_rate: f32,
}

impl std::fmt::Debug for LoRAOptimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoRAOptimizer")
            .field("adapters", &self.adapters)
            .field("config", &self.config)
            .field("frozen_parameters", &self.frozen_parameters)
            .field("learning_rate", &self.learning_rate)
            .finish()
    }
}

impl LoRAOptimizer {
    /// Creates a new LoRA optimizer.
    pub fn new(
        base_optimizer: Box<dyn OptimizerState>,
        config: LoRAConfig,
        learning_rate: f32,
    ) -> Self {
        Self {
            base_optimizer,
            adapters: HashMap::new(),
            config,
            frozen_parameters: HashMap::new(),
            learning_rate,
        }
    }

    /// Add a LoRA adapter for a specific module.
    pub fn add_adapter(
        &mut self,
        module_name: &str,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<()> {
        let adapter = LoRAAdapter::new(input_dim, output_dim, self.config.rank, self.config.alpha)?;
        self.adapters.insert(module_name.to_string(), adapter);
        Ok(())
    }

    /// Remove a LoRA adapter.
    pub fn remove_adapter(&mut self, module_name: &str) -> Option<LoRAAdapter> {
        self.adapters.remove(module_name)
    }

    /// Get a reference to an adapter.
    pub fn get_adapter(&self, module_name: &str) -> Option<&LoRAAdapter> {
        self.adapters.get(module_name)
    }

    /// Get a mutable reference to an adapter.
    pub fn get_adapter_mut(&mut self, module_name: &str) -> Option<&mut LoRAAdapter> {
        self.adapters.get_mut(module_name)
    }

    /// Enable or disable an adapter.
    pub fn set_adapter_active(&mut self, module_name: &str, active: bool) -> Result<()> {
        if let Some(adapter) = self.adapters.get_mut(module_name) {
            adapter.set_active(active);
            Ok(())
        } else {
            Err(
                trustformers_core::errors::TrustformersError::tensor_op_error(
                    &format!("Adapter {} not found", module_name),
                    "set_adapter_active",
                ),
            )
        }
    }

    /// Enable or disable all adapters.
    pub fn set_all_adapters_active(&mut self, active: bool) {
        for adapter in self.adapters.values_mut() {
            adapter.set_active(active);
        }
    }

    /// Get the total number of trainable parameters.
    pub fn num_trainable_parameters(&self) -> usize {
        self.adapters.values().map(|a| a.num_parameters()).sum()
    }

    /// Freeze base model parameters.
    pub fn freeze_base_parameters(&mut self, parameters: HashMap<String, Tensor>) {
        self.frozen_parameters = parameters;
    }

    /// Merge all active adapters into their respective base weights.
    pub fn merge_adapters_into_base(&mut self) -> Result<()> {
        for (module_name, adapter) in &self.adapters {
            if adapter.active {
                if let Some(base_weight) = self.frozen_parameters.get_mut(module_name) {
                    adapter.merge_into_weight(base_weight)?;
                }
            }
        }
        Ok(())
    }

    /// Save adapter weights (for efficient storage/sharing).
    pub fn save_adapters(&self) -> HashMap<String, (Tensor, Tensor, f32)> {
        self.adapters
            .iter()
            .map(|(name, adapter)| {
                (
                    name.clone(),
                    (
                        adapter.lora_a.clone(),
                        adapter.lora_b.clone(),
                        adapter.scaling,
                    ),
                )
            })
            .collect()
    }

    /// Load adapter weights.
    pub fn load_adapters(
        &mut self,
        adapters: HashMap<String, (Tensor, Tensor, f32)>,
    ) -> Result<()> {
        for (name, (lora_a, lora_b, scaling)) in adapters {
            let adapter = LoRAAdapter {
                lora_a,
                lora_b,
                scaling,
                active: true,
            };
            self.adapters.insert(name, adapter);
        }
        Ok(())
    }

    /// Get configuration.
    pub fn get_config(&self) -> &LoRAConfig {
        &self.config
    }

    /// Extract trainable parameters (adapter parameters only).
    fn get_trainable_parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for adapter in self.adapters.values() {
            if adapter.active {
                params.push(adapter.lora_a.clone());
                params.push(adapter.lora_b.clone());
            }
        }
        params
    }

    /// Update adapter parameters from optimized tensors.
    fn update_adapters_from_parameters(&mut self, parameters: &[Tensor]) -> AnyhowResult<()> {
        let mut param_idx = 0;
        for adapter in self.adapters.values_mut() {
            if adapter.active {
                if param_idx + 1 >= parameters.len() {
                    return Err(anyhow!("Not enough parameters provided"));
                }
                adapter.lora_a = parameters[param_idx].clone();
                adapter.lora_b = parameters[param_idx + 1].clone();
                param_idx += 2;
            }
        }
        Ok(())
    }
}

impl OptimizerState for LoRAOptimizer {
    fn zero_grad(&mut self) -> AnyhowResult<()> {
        self.base_optimizer.zero_grad()
    }

    fn step(&mut self, _parameters: &mut [Tensor]) -> AnyhowResult<()> {
        // For LoRA, we only optimize adapter parameters, not the full model parameters
        let mut trainable_params = self.get_trainable_parameters();

        // Step the base optimizer on adapter parameters
        self.base_optimizer.step(&mut trainable_params)?;

        // Update our adapters with the optimized parameters
        self.update_adapters_from_parameters(&trainable_params)?;

        Ok(())
    }

    fn get_lr(&self) -> f32 {
        self.learning_rate
    }

    fn set_lr(&mut self, lr: f32) {
        self.learning_rate = lr;
        self.base_optimizer.set_lr(lr);
    }

    fn state_dict(&self) -> AnyhowResult<HashMap<String, Tensor>> {
        let mut state = HashMap::new();

        // Save adapter states
        for (name, adapter) in &self.adapters {
            state.insert(format!("adapter_{}_lora_a", name), adapter.lora_a.clone());
            state.insert(format!("adapter_{}_lora_b", name), adapter.lora_b.clone());
            state.insert(
                format!("adapter_{}_scaling", name),
                Tensor::scalar(adapter.scaling)?,
            );
            state.insert(
                format!("adapter_{}_active", name),
                Tensor::scalar(adapter.active as i32 as f32)?,
            );
        }

        // Save base optimizer state
        let base_state = self.base_optimizer.state_dict()?;
        for (key, value) in base_state {
            state.insert(format!("base_{}", key), value);
        }

        Ok(state)
    }

    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> AnyhowResult<()> {
        let mut base_state = HashMap::new();
        let mut adapter_states: HashMap<
            String,
            (Option<Tensor>, Option<Tensor>, Option<f32>, Option<bool>),
        > = HashMap::new();

        // Separate adapter and base optimizer states
        for (key, value) in state {
            if key.starts_with("adapter_") {
                let parts: Vec<&str> = key.split('_').collect();
                if parts.len() >= 3 {
                    let adapter_name = parts[1];
                    let field = parts[2..].join("_");

                    let entry = adapter_states
                        .entry(adapter_name.to_string())
                        .or_insert((None, None, None, None));

                    match field.as_str() {
                        "lora_a" => entry.0 = Some(value),
                        "lora_b" => entry.1 = Some(value),
                        "scaling" => entry.2 = Some(value.to_scalar()?),
                        "active" => entry.3 = Some(value.to_scalar()? > 0.5),
                        _ => {},
                    }
                }
            } else if key.starts_with("base_") {
                base_state.insert(key[5..].to_string(), value);
            }
        }

        // Reconstruct adapters
        for (name, (lora_a_opt, lora_b_opt, scaling_opt, active_opt)) in adapter_states {
            if let (Some(lora_a), Some(lora_b), Some(scaling), Some(active)) =
                (lora_a_opt, lora_b_opt, scaling_opt, active_opt)
            {
                let adapter = LoRAAdapter {
                    lora_a,
                    lora_b,
                    scaling,
                    active,
                };
                self.adapters.insert(name, adapter);
            }
        }

        // Load base optimizer state
        self.base_optimizer.load_state_dict(base_state)?;

        Ok(())
    }
}

/// Convenience function to create a LoRA-enabled Adam optimizer.
pub fn create_lora_adam(
    learning_rate: f32,
    config: LoRAConfig,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
) -> LoRAOptimizer {
    let adam = Box::new(crate::sparse::SparseAdam::with_default_config(
        learning_rate,
        beta1,
        beta2,
        epsilon,
        weight_decay,
    ));
    LoRAOptimizer::new(adam, config, learning_rate)
}

/// Convenience function to create a LoRA-enabled AdamW optimizer.
pub fn create_lora_adamw(
    learning_rate: f32,
    config: LoRAConfig,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
) -> LoRAOptimizer {
    let adamw = Box::new(crate::sparse::SparseAdam::with_default_config(
        learning_rate,
        beta1,
        beta2,
        epsilon,
        weight_decay,
    ));
    LoRAOptimizer::new(adamw, config, learning_rate)
}

/// Convenience function to create a LoRA-enabled SGD optimizer.
pub fn create_lora_sgd(
    learning_rate: f32,
    config: LoRAConfig,
    momentum: f32,
    _dampening: f32,
    _weight_decay: f32,
    _nesterov: bool,
) -> LoRAOptimizer {
    let sgd = Box::new(crate::convergence::QHM::with_defaults(
        learning_rate,
        momentum,
        0.999,
    ));
    LoRAOptimizer::new(sgd, config, learning_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_config_default() {
        let config = LoRAConfig::default();
        assert_eq!(config.rank, 8);
        assert_eq!(config.alpha, 16.0);
        assert_eq!(config.dropout, 0.1);
        assert!(!config.bias);
        assert_eq!(config.target_modules.len(), 3);
        assert!(!config.merge_weights);
    }

    #[test]
    fn test_lora_adapter_creation() {
        let adapter = LoRAAdapter::new(512, 256, 8, 16.0).unwrap();

        assert_eq!(adapter.lora_a.shape(), &[512, 8]);
        assert_eq!(adapter.lora_b.shape(), &[8, 256]);
        assert_eq!(adapter.scaling, 2.0); // 16.0 / 8
        assert!(adapter.active);
    }

    #[test]
    fn test_lora_adapter_parameters() {
        let adapter = LoRAAdapter::new(512, 256, 8, 16.0).unwrap();
        let expected_params = 512 * 8 + 8 * 256; // A + B parameters
        assert_eq!(adapter.num_parameters(), expected_params);
    }

    #[test]
    fn test_lora_optimizer_creation() {
        let config = LoRAConfig::default();
        let optimizer = create_lora_adam(1e-3, config, 0.9, 0.999, 1e-8, 0.01);

        assert_eq!(optimizer.get_lr(), 1e-3);
        assert_eq!(optimizer.num_trainable_parameters(), 0); // No adapters added yet
    }

    #[test]
    fn test_adapter_management() {
        let config = LoRAConfig::default();
        let mut optimizer = create_lora_adam(1e-3, config, 0.9, 0.999, 1e-8, 0.01);

        // Add adapter
        optimizer.add_adapter("query", 512, 512).unwrap();
        assert_eq!(optimizer.num_trainable_parameters(), 512 * 8 + 8 * 512);

        // Check adapter exists
        assert!(optimizer.get_adapter("query").is_some());

        // Remove adapter
        let removed = optimizer.remove_adapter("query");
        assert!(removed.is_some());
        assert_eq!(optimizer.num_trainable_parameters(), 0);
    }

    #[test]
    fn test_adapter_activation() {
        let config = LoRAConfig::default();
        let mut optimizer = create_lora_adam(1e-3, config, 0.9, 0.999, 1e-8, 0.01);

        optimizer.add_adapter("query", 512, 512).unwrap();

        // Initially active
        assert!(optimizer.get_adapter("query").unwrap().active);

        // Deactivate
        optimizer.set_adapter_active("query", false).unwrap();
        assert!(!optimizer.get_adapter("query").unwrap().active);

        // Activate all
        optimizer.set_all_adapters_active(true);
        assert!(optimizer.get_adapter("query").unwrap().active);
    }
}
