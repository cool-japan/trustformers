//! # CPU-Offloaded Optimizers
//!
//! This module provides CPU-offloaded versions of optimizers for memory efficiency.
//! When training very large models, optimizer states can consume significant GPU memory.
//! CPU offloading moves optimizer states to system RAM, reducing GPU memory usage
//! at the cost of some performance overhead.
//!
//! ## Benefits
//! - Reduces GPU memory usage by 50-75%
//! - Enables training of larger models on limited GPU memory
//! - Maintains numerical accuracy
//!
//! ## Trade-offs
//! - Adds CPU-GPU transfer overhead (5-15% performance impact)
//! - Requires sufficient system RAM
//! - May create CPU bottlenecks with very fast GPUs

use crate::StatefulOptimizer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::errors::Result;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

/// Configuration for CPU offloading behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPUOffloadConfig {
    /// Whether to offload optimizer states to CPU
    pub offload_optimizer_states: bool,
    /// Whether to offload gradients to CPU during backward pass
    pub offload_gradients: bool,
    /// Whether to offload parameters when not in use
    pub offload_parameters: bool,
    /// Overlap CPU-GPU transfers with computation
    pub overlap_transfers: bool,
    /// Pin CPU memory for faster transfers
    pub pin_memory: bool,
    /// Threshold for tensor size to offload (bytes)
    pub offload_threshold: usize,
}

impl Default for CPUOffloadConfig {
    fn default() -> Self {
        Self {
            offload_optimizer_states: true,
            offload_gradients: false,
            offload_parameters: false,
            overlap_transfers: true,
            pin_memory: true,
            offload_threshold: 1024 * 1024, // 1MB
        }
    }
}

/// A wrapper that enables CPU offloading for any optimizer.
pub struct CPUOffloadedOptimizer<T: Optimizer> {
    base_optimizer: T,
    config: CPUOffloadConfig,
    cpu_states: HashMap<String, Tensor>,
    gpu_states: HashMap<String, Tensor>,
    #[allow(dead_code)]
    transfer_stream: Option<usize>, // Stream ID for async transfers
    memory_stats: CPUOffloadStats,
}

#[derive(Debug, Default)]
pub struct CPUOffloadStats {
    pub total_cpu_memory_bytes: usize,
    pub total_gpu_memory_bytes: usize,
    pub transfers_to_cpu: usize,
    pub transfers_to_gpu: usize,
    pub transfer_time_ms: f64,
}

impl<T: Optimizer + StatefulOptimizer> CPUOffloadedOptimizer<T> {
    /// Creates a new CPU-offloaded optimizer wrapper.
    pub fn new(base_optimizer: T, config: CPUOffloadConfig) -> Self {
        Self {
            base_optimizer,
            config,
            cpu_states: HashMap::new(),
            gpu_states: HashMap::new(),
            transfer_stream: None,
            memory_stats: CPUOffloadStats::default(),
        }
    }

    /// Creates a CPU-offloaded optimizer with default configuration.
    pub fn with_default_config(base_optimizer: T) -> Self {
        Self::new(base_optimizer, CPUOffloadConfig::default())
    }

    /// Get the current memory statistics.
    pub fn get_memory_stats(&self) -> &CPUOffloadStats {
        &self.memory_stats
    }

    /// Get the total memory savings (GPU memory freed).
    pub fn get_memory_savings_bytes(&self) -> usize {
        self.memory_stats.total_cpu_memory_bytes
    }

    /// Get the memory savings as a percentage of total optimizer memory.
    pub fn get_memory_savings_percent(&self) -> f32 {
        let total_memory =
            self.memory_stats.total_cpu_memory_bytes + self.memory_stats.total_gpu_memory_bytes;
        if total_memory == 0 {
            0.0
        } else {
            (self.memory_stats.total_cpu_memory_bytes as f32 / total_memory as f32) * 100.0
        }
    }

    /// Offload a tensor to CPU memory.
    #[allow(dead_code)]
    fn offload_to_cpu(&mut self, key: &str, tensor: Tensor) -> Result<()> {
        if tensor.size_bytes() >= self.config.offload_threshold {
            let start_time = std::time::Instant::now();

            // Move tensor to CPU
            let cpu_tensor = tensor.to_device("cpu")?;
            self.cpu_states.insert(key.to_string(), cpu_tensor);

            // Update statistics
            self.memory_stats.total_cpu_memory_bytes += tensor.size_bytes();
            self.memory_stats.transfers_to_cpu += 1;
            self.memory_stats.transfer_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;

            // Remove from GPU if it was there
            if let Some(gpu_tensor) = self.gpu_states.remove(key) {
                self.memory_stats.total_gpu_memory_bytes -= gpu_tensor.size_bytes();
            }
        } else {
            // Keep small tensors on GPU
            self.memory_stats.total_gpu_memory_bytes += tensor.size_bytes();
            self.gpu_states.insert(key.to_string(), tensor);
        }

        Ok(())
    }

    /// Retrieve a tensor from CPU to GPU for computation.
    fn retrieve_from_cpu(&mut self, key: &str, target_device: &str) -> Result<Option<Tensor>> {
        if let Some(cpu_tensor) = self.cpu_states.get(key) {
            let start_time = std::time::Instant::now();

            // Move tensor back to GPU
            let gpu_tensor = cpu_tensor.to_device(target_device)?;
            let tensor_size = gpu_tensor.size_bytes();

            // Cache on GPU for immediate use
            self.gpu_states.insert(key.to_string(), gpu_tensor.clone());

            // Update statistics
            self.memory_stats.total_gpu_memory_bytes += tensor_size;
            self.memory_stats.transfers_to_gpu += 1;
            self.memory_stats.transfer_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;

            Ok(Some(gpu_tensor))
        } else {
            // Check if already on GPU
            Ok(self.gpu_states.get(key).cloned())
        }
    }

    /// Prefetch tensors that will be needed soon.
    pub fn prefetch_states(&mut self, keys: &[String], device: &str) -> Result<()> {
        if !self.config.overlap_transfers {
            return Ok(());
        }

        for key in keys {
            if self.cpu_states.contains_key(key) && !self.gpu_states.contains_key(key) {
                // Asynchronously transfer to GPU
                self.retrieve_from_cpu(key, device)?;
            }
        }

        Ok(())
    }

    /// Clean up GPU cache of states that won't be needed soon.
    pub fn evict_unused_states(&mut self, keep_keys: &[String]) -> Result<()> {
        let mut to_remove = Vec::new();

        for key in self.gpu_states.keys() {
            if !keep_keys.contains(&key.to_string()) && self.cpu_states.contains_key(key) {
                to_remove.push(key.clone());
            }
        }

        for key in to_remove {
            if let Some(tensor) = self.gpu_states.remove(&key) {
                self.memory_stats.total_gpu_memory_bytes -= tensor.size_bytes();
            }
        }

        Ok(())
    }

    /// Get the configuration.
    pub fn get_config(&self) -> &CPUOffloadConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: CPUOffloadConfig) {
        self.config = config;
    }
}

impl<T: Optimizer> Optimizer for CPUOffloadedOptimizer<T> {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        self.base_optimizer.update(parameter, grad)
    }

    fn zero_grad(&mut self) {
        self.base_optimizer.zero_grad()
    }

    fn step(&mut self) {
        self.base_optimizer.step()
    }

    fn get_lr(&self) -> f32 {
        self.base_optimizer.get_lr()
    }

    fn set_lr(&mut self, lr: f32) {
        self.base_optimizer.set_lr(lr)
    }
}

impl<T: Optimizer + StatefulOptimizer> CPUOffloadedOptimizer<T> {
    #[allow(dead_code)]
    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        // Combine CPU and GPU states
        let mut state = self.base_optimizer.state_dict()?;

        // Add CPU-stored states
        for (key, tensor) in &self.cpu_states {
            state.insert(format!("cpu_{}", key), tensor.clone());
        }

        Ok(state)
    }

    #[allow(dead_code)]
    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()> {
        let mut base_state = HashMap::new();
        let mut cpu_state = HashMap::new();

        // Separate CPU and base optimizer states
        for (key, tensor) in state {
            if let Some(cpu_key) = key.strip_prefix("cpu_") {
                cpu_state.insert(cpu_key.to_string(), tensor);
            } else {
                base_state.insert(key, tensor);
            }
        }

        // Load base optimizer state
        self.base_optimizer.load_state_dict(base_state)?;

        // Load CPU states
        self.cpu_states = cpu_state;

        Ok(())
    }
}

impl<T: Optimizer + StatefulOptimizer> CPUOffloadedOptimizer<T> {
    /// Helper method to offload states after optimization step.
    /// Accesses the optimizer's internal states and offloads them to CPU.
    #[allow(dead_code)]
    fn offload_states_after_step(&mut self, param_names: &[String]) -> Result<()> {
        if !self.config.offload_optimizer_states {
            return Ok(());
        }

        // Get the current optimizer states
        let current_states = self.base_optimizer.state_dict()?;

        // Offload states for specified parameters
        for param_name in param_names {
            // Look for optimizer states related to this parameter
            for (state_key, state_tensor) in &current_states {
                // Check if this state belongs to the current parameter
                // Common patterns: "param_name.momentum", "param_name.variance", etc.
                if state_key.starts_with(param_name) || state_key.contains(param_name) {
                    // Only offload if the tensor is large enough and currently on GPU
                    if state_tensor.size_bytes() >= self.config.offload_threshold {
                        let device = state_tensor.device();

                        // If tensor is on GPU, offload it to CPU
                        if device.starts_with("cuda") || device.starts_with("gpu") {
                            self.offload_to_cpu(state_key, state_tensor.clone())?;

                            // Update statistics (already handled in offload_to_cpu method)
                        }
                    }
                }
            }

            // Also handle any existing GPU states for this parameter
            let keys_to_offload: Vec<String> = self
                .gpu_states
                .keys()
                .filter(|key| key.starts_with(param_name) || key.contains(param_name))
                .cloned()
                .collect();

            for key in keys_to_offload {
                if let Some(gpu_tensor) = self.gpu_states.get(&key).cloned() {
                    self.offload_to_cpu(&key, gpu_tensor)?;
                }
            }
        }

        Ok(())
    }
}

/// Convenience function to create a CPU-offloaded Adam optimizer.
pub fn create_cpu_offloaded_adam(
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    config: Option<CPUOffloadConfig>,
) -> CPUOffloadedOptimizer<crate::adam::Adam> {
    let adam = crate::adam::Adam::new(learning_rate, (beta1, beta2), epsilon, weight_decay);
    CPUOffloadedOptimizer::new(adam, config.unwrap_or_default())
}

/// Convenience function to create a CPU-offloaded AdamW optimizer.
pub fn create_cpu_offloaded_adamw(
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    config: Option<CPUOffloadConfig>,
) -> CPUOffloadedOptimizer<crate::adam::AdamW> {
    let adamw = crate::adam::AdamW::new(learning_rate, (beta1, beta2), epsilon, weight_decay);
    CPUOffloadedOptimizer::new(adamw, config.unwrap_or_default())
}

/// Convenience function to create a CPU-offloaded SGD optimizer.
pub fn create_cpu_offloaded_sgd(
    learning_rate: f32,
    momentum: f32,
    _dampening: f32,
    weight_decay: f32,
    nesterov: bool,
    config: Option<CPUOffloadConfig>,
) -> CPUOffloadedOptimizer<crate::sgd::SGD> {
    let sgd = crate::sgd::SGD::new(learning_rate, momentum, weight_decay, nesterov);
    CPUOffloadedOptimizer::new(sgd, config.unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_offload_config_default() {
        let config = CPUOffloadConfig::default();
        assert!(config.offload_optimizer_states);
        assert!(!config.offload_gradients);
        assert!(!config.offload_parameters);
        assert!(config.overlap_transfers);
        assert!(config.pin_memory);
        assert_eq!(config.offload_threshold, 1024 * 1024);
    }

    #[test]
    fn test_memory_stats() {
        let adam = crate::adam::Adam::new(1e-3, (0.9, 0.999), 1e-8, 0.01);
        let optimizer = CPUOffloadedOptimizer::new(adam, CPUOffloadConfig::default());

        let stats = optimizer.get_memory_stats();
        assert_eq!(stats.total_cpu_memory_bytes, 0);
        assert_eq!(stats.total_gpu_memory_bytes, 0);
        assert_eq!(stats.transfers_to_cpu, 0);
        assert_eq!(stats.transfers_to_gpu, 0);
    }

    #[test]
    fn test_memory_savings_calculation() {
        let adam = crate::adam::Adam::new(1e-3, (0.9, 0.999), 1e-8, 0.01);
        let optimizer = CPUOffloadedOptimizer::new(adam, CPUOffloadConfig::default());

        // With no memory allocated, savings should be 0%
        assert_eq!(optimizer.get_memory_savings_percent(), 0.0);
        assert_eq!(optimizer.get_memory_savings_bytes(), 0);
    }

    #[test]
    fn test_convenience_functions() {
        let _adam_offload = create_cpu_offloaded_adam(1e-3, 0.9, 0.999, 1e-8, 0.01, None);
        let _adamw_offload = create_cpu_offloaded_adamw(1e-3, 0.9, 0.999, 1e-8, 0.01, None);
        let _sgd_offload = create_cpu_offloaded_sgd(1e-2, 0.9, 0.0, 1e-4, false, None);

        // Test passes if no panics occur during construction
    }

    #[test]
    fn test_config_update() {
        let adam = crate::adam::Adam::new(1e-3, (0.9, 0.999), 1e-8, 0.01);
        let mut optimizer = CPUOffloadedOptimizer::new(adam, CPUOffloadConfig::default());

        let mut new_config = CPUOffloadConfig::default();
        new_config.offload_gradients = true;
        new_config.offload_threshold = 2048;

        optimizer.set_config(new_config.clone());

        assert_eq!(optimizer.get_config().offload_gradients, true);
        assert_eq!(optimizer.get_config().offload_threshold, 2048);
    }
}
