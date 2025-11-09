//! Main ZeRO Optimizer Implementation
//!
//! This module provides the main ZeRO optimizer wrapper that coordinates
//! between different ZeRO stages and manages the optimization process.

use std::collections::HashMap;
use std::sync::Arc;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::parallel::ModelParallelContext;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

use super::{
    ZeROImplementationStage, ZeROMemoryStats, ZeROStage1, ZeROStage2, ZeROStage3, ZeROState,
};

/// Configuration for ZeRO optimizer
#[derive(Debug, Clone)]
pub struct ZeROConfig {
    /// ZeRO stage to use
    pub stage: ZeROStage,
    /// Target bucket size for gradient communication (in MB)
    pub bucket_size_mb: usize,
    /// Whether to overlap communication with computation
    pub overlap_comm: bool,
    /// Reduce bucket size (number of elements to reduce at once)
    pub reduce_bucket_size: usize,
    /// Prefetch depth for parameter gathering
    pub prefetch_depth: usize,
    /// Maximum memory usage threshold before releasing parameters
    pub max_memory_usage_mb: usize,
    /// Enable gradient compression
    pub gradient_compression: bool,
    /// Pin memory for faster GPU transfers
    pub pin_memory: bool,
}

impl Default for ZeROConfig {
    fn default() -> Self {
        Self {
            stage: ZeROStage::Stage1,
            bucket_size_mb: 25,
            overlap_comm: true,
            reduce_bucket_size: 500_000_000, // 500M elements
            prefetch_depth: 2,
            max_memory_usage_mb: 1024, // 1GB
            gradient_compression: false,
            pin_memory: true,
        }
    }
}

/// ZeRO optimization stages
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeROStage {
    /// Stage 1: Partition optimizer states only
    Stage1,
    /// Stage 2: Partition optimizer states + gradients
    Stage2,
    /// Stage 3: Partition optimizer states + gradients + parameters
    Stage3,
}

impl From<ZeROStage> for ZeROImplementationStage {
    fn from(stage: ZeROStage) -> Self {
        match stage {
            ZeROStage::Stage1 => ZeROImplementationStage::Stage1,
            ZeROStage::Stage2 => ZeROImplementationStage::Stage2,
            ZeROStage::Stage3 => ZeROImplementationStage::Stage3,
        }
    }
}

/// Main ZeRO optimizer that wraps an underlying optimizer
pub struct ZeROOptimizer<T: Optimizer> {
    /// Underlying base optimizer
    base_optimizer: T,
    /// ZeRO configuration
    config: ZeROConfig,
    /// Model parallel context for communication
    mp_context: Arc<ModelParallelContext>,
    /// ZeRO-specific state
    zero_state: ZeROState,
    /// Stage 1 implementation
    stage1: Option<ZeROStage1<T>>,
    /// Stage 2 implementation
    stage2: Option<ZeROStage2<T>>,
    /// Stage 3 implementation
    stage3: Option<ZeROStage3<T>>,
    /// Memory statistics
    memory_stats: ZeROMemoryStats,
    /// Parameter names for tracking
    parameter_names: Vec<String>,
}

impl<T: Optimizer> ZeROOptimizer<T> {
    /// Create a new ZeRO optimizer
    pub fn new(
        base_optimizer: T,
        config: ZeROConfig,
        mp_context: Arc<ModelParallelContext>,
    ) -> Result<Self> {
        let mut optimizer = Self {
            base_optimizer,
            config: config.clone(),
            mp_context: mp_context.clone(),
            zero_state: ZeROState::new(),
            stage1: None,
            stage2: None,
            stage3: None,
            memory_stats: ZeROMemoryStats::new(),
            parameter_names: Vec::new(),
        };

        // Initialize the appropriate stage
        optimizer.initialize_stage(config.stage)?;

        Ok(optimizer)
    }

    /// Initialize the specified ZeRO stage
    fn initialize_stage(&mut self, stage: ZeROStage) -> Result<()> {
        match stage {
            ZeROStage::Stage1 => {
                self.stage1 = Some(ZeROStage1::new(
                    self.mp_context.clone(),
                    self.config.clone(),
                )?);
            },
            ZeROStage::Stage2 => {
                self.stage2 = Some(ZeROStage2::new(
                    self.mp_context.clone(),
                    self.config.clone(),
                )?);
            },
            ZeROStage::Stage3 => {
                self.stage3 = Some(ZeROStage3::new(
                    self.mp_context.clone(),
                    self.config.clone(),
                )?);
            },
        }
        Ok(())
    }

    /// Register parameters with ZeRO optimizer
    pub fn register_parameters(&mut self, parameters: HashMap<String, Tensor>) -> Result<()> {
        self.parameter_names = parameters.keys().cloned().collect();

        match self.config.stage {
            ZeROStage::Stage1 => {
                if let Some(stage1) = &mut self.stage1 {
                    stage1.register_parameters(parameters)?;
                }
            },
            ZeROStage::Stage2 => {
                if let Some(stage2) = &mut self.stage2 {
                    stage2.register_parameters(parameters)?;
                }
            },
            ZeROStage::Stage3 => {
                if let Some(stage3) = &mut self.stage3 {
                    stage3.register_parameters(parameters)?;
                }
            },
        }

        self.update_memory_stats();
        Ok(())
    }

    /// Update gradients for ZeRO optimization
    pub fn update_gradients(&mut self, gradients: HashMap<String, Tensor>) -> Result<()> {
        match self.config.stage {
            ZeROStage::Stage1 => {
                // Stage 1 doesn't partition gradients, use regular optimizer
                for (name, grad) in gradients {
                    if let Some(stage1) = &mut self.stage1 {
                        stage1.accumulate_gradient(&name, &grad)?;
                    }
                }
            },
            ZeROStage::Stage2 => {
                if let Some(stage2) = &mut self.stage2 {
                    stage2.update_gradients(gradients)?;
                }
            },
            ZeROStage::Stage3 => {
                if let Some(stage3) = &mut self.stage3 {
                    stage3.update_gradients(gradients)?;
                }
            },
        }
        Ok(())
    }

    /// Gather parameters for forward pass (Stage 3 only)
    pub fn gather_parameters(
        &mut self,
        parameter_names: &[String],
    ) -> Result<HashMap<String, Tensor>> {
        match self.config.stage {
            ZeROStage::Stage3 => {
                if let Some(stage3) = &mut self.stage3 {
                    stage3.gather_parameters(parameter_names)
                } else {
                    Err(TrustformersError::runtime_error(
                        "Stage 3 not initialized".into(),
                    ))
                }
            },
            _ => {
                // For Stage 1 and 2, parameters are not partitioned
                Err(TrustformersError::runtime_error(
                    "Parameter gathering only available in Stage 3".into(),
                ))
            },
        }
    }

    /// Release gathered parameters to save memory (Stage 3 only)
    pub fn release_parameters(&mut self, parameter_names: &[String]) -> Result<()> {
        match self.config.stage {
            ZeROStage::Stage3 => {
                if let Some(stage3) = &mut self.stage3 {
                    stage3.release_parameters(parameter_names)
                } else {
                    Err(TrustformersError::runtime_error(
                        "Stage 3 not initialized".into(),
                    ))
                }
            },
            _ => Ok(()), // No-op for other stages
        }
    }

    /// Get memory statistics
    pub fn get_memory_stats(&self) -> &ZeROMemoryStats {
        &self.memory_stats
    }

    /// Update memory statistics
    fn update_memory_stats(&mut self) {
        let memory_usage = self.zero_state.memory_usage();

        self.memory_stats.optimizer_memory_saved =
            memory_usage.get("optimizer_states").copied().unwrap_or(0);
        self.memory_stats.gradient_memory_saved =
            memory_usage.get("gradient_partitions").copied().unwrap_or(0);
        self.memory_stats.parameter_memory_saved =
            memory_usage.get("parameter_partitions").copied().unwrap_or(0);
        self.memory_stats.communication_overhead =
            memory_usage.get("communication_buffers").copied().unwrap_or(0);

        self.memory_stats.update_totals();
    }

    /// Check if memory usage exceeds threshold
    pub fn check_memory_usage(&self) -> bool {
        let total_memory_mb = self.memory_stats.total_memory_saved / (1024 * 1024);
        total_memory_mb > self.config.max_memory_usage_mb
    }

    /// Get current ZeRO stage
    pub fn get_stage(&self) -> ZeROStage {
        self.config.stage
    }

    /// Get the underlying base optimizer
    pub fn base_optimizer(&self) -> &T {
        &self.base_optimizer
    }

    /// Get mutable reference to base optimizer
    pub fn base_optimizer_mut(&mut self) -> &mut T {
        &mut self.base_optimizer
    }

    /// Get model parallel context
    pub fn mp_context(&self) -> &Arc<ModelParallelContext> {
        &self.mp_context
    }

    /// Perform optimizer step with ZeRO optimizations
    pub fn optimizer_step(&mut self) -> Result<()> {
        match self.config.stage {
            ZeROStage::Stage1 => {
                if let Some(stage1) = &mut self.stage1 {
                    stage1.optimizer_step(&mut self.base_optimizer)?;
                }
            },
            ZeROStage::Stage2 => {
                if let Some(stage2) = &mut self.stage2 {
                    stage2.optimizer_step(&mut self.base_optimizer)?;
                }
            },
            ZeROStage::Stage3 => {
                if let Some(stage3) = &mut self.stage3 {
                    stage3.optimizer_step(&mut self.base_optimizer)?;
                }
            },
        }

        self.zero_state.step();
        self.update_memory_stats();
        Ok(())
    }
}

impl<T: Optimizer> Optimizer for ZeROOptimizer<T> {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        // ZeRO optimizer handles updates through its stage implementations
        // This method is called for individual parameter updates
        match self.config.stage {
            ZeROStage::Stage1 => {
                if let Some(stage1) = &mut self.stage1 {
                    stage1.update_parameter(parameter, grad, &mut self.base_optimizer)
                } else {
                    self.base_optimizer.update(parameter, grad)
                }
            },
            ZeROStage::Stage2 | ZeROStage::Stage3 => {
                // For Stage 2 and 3, gradients are handled in batches
                // Individual updates are not recommended
                Err(TrustformersError::runtime_error(
                    "Individual parameter updates not supported in ZeRO Stage 2/3. Use batch updates."
                        .into()
                ))
            },
        }
    }

    fn zero_grad(&mut self) {
        self.zero_state.zero_grad();
        self.base_optimizer.zero_grad();
    }

    fn step(&mut self) {
        self.base_optimizer.step();
        self.zero_state.step();
    }

    fn get_lr(&self) -> f32 {
        self.base_optimizer.get_lr()
    }

    fn set_lr(&mut self, lr: f32) {
        self.base_optimizer.set_lr(lr);
    }

    fn accumulate_grad(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        // Handle gradient accumulation through ZeRO stages
        match self.config.stage {
            ZeROStage::Stage1 => {
                if let Some(stage1) = &mut self.stage1 {
                    // Stage 1 can use regular gradient accumulation
                    stage1.accumulate_gradient_for_parameter(parameter, grad)
                } else {
                    self.base_optimizer.accumulate_grad(parameter, grad)
                }
            },
            ZeROStage::Stage2 | ZeROStage::Stage3 => {
                // For Stage 2/3, accumulation is handled in the stage implementation
                Err(TrustformersError::runtime_error(
                    "Gradient accumulation in ZeRO Stage 2/3 should be handled through update_gradients"
                        .into()
                ))
            },
        }
    }

    fn apply_accumulated_grads(&mut self, accumulation_steps: usize) -> Result<()> {
        match self.config.stage {
            ZeROStage::Stage1 => {
                if let Some(stage1) = &mut self.stage1 {
                    stage1.apply_accumulated_gradients(&mut self.base_optimizer, accumulation_steps)
                } else {
                    self.base_optimizer.apply_accumulated_grads(accumulation_steps)
                }
            },
            ZeROStage::Stage2 => {
                if let Some(stage2) = &mut self.stage2 {
                    stage2.apply_accumulated_gradients(&mut self.base_optimizer, accumulation_steps)
                } else {
                    Err(TrustformersError::runtime_error(
                        "Stage 2 not initialized".into(),
                    ))
                }
            },
            ZeROStage::Stage3 => {
                if let Some(stage3) = &mut self.stage3 {
                    stage3.apply_accumulated_gradients(&mut self.base_optimizer, accumulation_steps)
                } else {
                    Err(TrustformersError::runtime_error(
                        "Stage 3 not initialized".into(),
                    ))
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adam::Adam;
    use trustformers_core::parallel::{
        CommunicationBackend, ModelParallelConfig, ModelParallelStrategy,
    };

    #[test]
    fn test_zero_optimizer_creation() {
        let config = ModelParallelConfig {
            num_devices: 2,
            device_ids: vec![0, 1],
            strategy: ModelParallelStrategy::Pipeline,
            comm_backend: CommunicationBackend::Custom,
            ..Default::default()
        };
        let mp_context = Arc::new(ModelParallelContext::new(config).unwrap());

        let adam = Adam::new(0.001, (0.9, 0.999), 1e-8, 0.01);
        let zero_config = ZeROConfig::default();

        let zero_optimizer = ZeROOptimizer::new(adam, zero_config, mp_context);
        assert!(zero_optimizer.is_ok());

        let optimizer = zero_optimizer.unwrap();
        assert_eq!(optimizer.get_stage(), ZeROStage::Stage1);
    }

    #[test]
    fn test_zero_stage_initialization() {
        let config = ModelParallelConfig {
            num_devices: 4,
            device_ids: vec![0, 1, 2, 3],
            strategy: ModelParallelStrategy::Pipeline,
            comm_backend: CommunicationBackend::Custom,
            ..Default::default()
        };
        let mp_context = Arc::new(ModelParallelContext::new(config).unwrap());

        // Test Stage 2
        let adam = Adam::new(0.001, (0.9, 0.999), 1e-8, 0.01);
        let zero_config = ZeROConfig {
            stage: ZeROStage::Stage2,
            ..Default::default()
        };

        let zero_optimizer = ZeROOptimizer::new(adam, zero_config, mp_context.clone());
        assert!(zero_optimizer.is_ok());

        // Test Stage 3
        let adam = Adam::new(0.001, (0.9, 0.999), 1e-8, 0.01);
        let zero_config = ZeROConfig {
            stage: ZeROStage::Stage3,
            ..Default::default()
        };

        let zero_optimizer = ZeROOptimizer::new(adam, zero_config, mp_context);
        assert!(zero_optimizer.is_ok());
    }

    #[test]
    fn test_parameter_registration() {
        let config = ModelParallelConfig {
            num_devices: 2,
            device_ids: vec![0, 1],
            strategy: ModelParallelStrategy::Pipeline,
            comm_backend: CommunicationBackend::Custom,
            ..Default::default()
        };
        let mp_context = Arc::new(ModelParallelContext::new(config).unwrap());

        let adam = Adam::new(0.001, (0.9, 0.999), 1e-8, 0.01);
        let zero_config = ZeROConfig::default();
        let mut zero_optimizer = ZeROOptimizer::new(adam, zero_config, mp_context).unwrap();

        let mut parameters = HashMap::new();
        parameters.insert("weight1".to_string(), Tensor::ones(&[4, 4]).unwrap());
        parameters.insert("bias1".to_string(), Tensor::ones(&[4]).unwrap());

        let result = zero_optimizer.register_parameters(parameters);
        assert!(result.is_ok());
        assert_eq!(zero_optimizer.parameter_names.len(), 2);
    }

    #[test]
    fn test_memory_stats() {
        let config = ModelParallelConfig {
            num_devices: 2,
            device_ids: vec![0, 1],
            strategy: ModelParallelStrategy::Pipeline,
            comm_backend: CommunicationBackend::Custom,
            ..Default::default()
        };
        let mp_context = Arc::new(ModelParallelContext::new(config).unwrap());

        let adam = Adam::new(0.001, (0.9, 0.999), 1e-8, 0.01);
        let zero_config = ZeROConfig::default();
        let zero_optimizer = ZeROOptimizer::new(adam, zero_config, mp_context).unwrap();

        let stats = zero_optimizer.get_memory_stats();
        assert_eq!(stats.optimizer_memory_saved, 0); // No parameters registered yet
        assert_eq!(stats.gradient_memory_saved, 0);
        assert_eq!(stats.parameter_memory_saved, 0);
    }
}
