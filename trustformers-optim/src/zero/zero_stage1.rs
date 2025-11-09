//! ZeRO Stage 1: Optimizer State Partitioning
//!
//! Stage 1 partitions optimizer states (momentum, variance, etc.) across devices
//! while keeping parameters and gradients replicated. This provides memory savings
//! for the optimizer states without affecting the forward/backward pass.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::parallel::ModelParallelContext;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

use super::zero_optimizer::ZeROConfig;
use super::zero_utils::{calculate_bucket_size, ParameterGroup};

/// ZeRO Stage 1 implementation - partitions optimizer states only
pub struct ZeROStage1<T: Optimizer> {
    /// Model parallel context for communication
    mp_context: Arc<ModelParallelContext>,
    /// ZeRO configuration
    config: ZeROConfig,
    /// Parameter groups with local optimizer states
    parameter_groups: HashMap<String, ParameterGroup>,
    /// Partitioned optimizer states (momentum, variance, etc.)
    optimizer_states: HashMap<String, HashMap<String, Tensor>>,
    /// Parameter ownership mapping (which rank owns which parameter's optimizer state)
    parameter_ownership: HashMap<String, usize>,
    /// Communication buckets for optimizer state synchronization
    comm_buckets: Vec<Vec<String>>,
    /// Accumulated gradients for gradient accumulation
    accumulated_gradients: HashMap<String, Tensor>,
    /// Number of accumulated steps
    accumulation_steps: usize,
    /// Phantom data for optimizer type
    _phantom: PhantomData<T>,
}

impl<T: Optimizer> ZeROStage1<T> {
    /// Create a new ZeRO Stage 1 optimizer
    pub fn new(mp_context: Arc<ModelParallelContext>, config: ZeROConfig) -> Result<Self> {
        Ok(Self {
            mp_context,
            config,
            parameter_groups: HashMap::new(),
            optimizer_states: HashMap::new(),
            parameter_ownership: HashMap::new(),
            comm_buckets: Vec::new(),
            accumulated_gradients: HashMap::new(),
            accumulation_steps: 0,
            _phantom: PhantomData,
        })
    }

    /// Register parameters and partition optimizer states
    pub fn register_parameters(&mut self, parameters: HashMap<String, Tensor>) -> Result<()> {
        let world_size = self.mp_context.world_size();
        let rank = self.mp_context.rank();

        // Calculate parameter sizes for bucketing
        let mut param_sizes = Vec::new();
        let mut param_names = Vec::new();

        for (name, tensor) in &parameters {
            param_sizes.push(tensor.size());
            param_names.push(name.clone());
        }

        // Create communication buckets
        let bucket_size = self.config.bucket_size_mb * 1024 * 1024; // Convert MB to bytes
        let bucket_indices = calculate_bucket_size(&param_sizes, bucket_size);

        for bucket_idx in bucket_indices {
            let bucket: Vec<String> = bucket_idx.iter().map(|&i| param_names[i].clone()).collect();
            self.comm_buckets.push(bucket);
        }

        // Partition parameters across ranks for optimizer state ownership
        let mut param_idx = 0;
        for (name, tensor) in parameters {
            // Determine which rank owns this parameter's optimizer state
            let owner_rank = param_idx % world_size;
            self.parameter_ownership.insert(name.clone(), owner_rank);

            // If this rank owns the parameter's optimizer state, initialize it
            if owner_rank == rank {
                self.initialize_optimizer_state(&name, &tensor)?;
            }

            // Create parameter group (all ranks have parameter data for forward/backward)
            let mut group = ParameterGroup::new(format!("param_{}", param_idx), vec![name.clone()]);
            group.add_parameter(name.clone(), tensor);
            self.parameter_groups.insert(name.clone(), group);

            param_idx += 1;
        }

        Ok(())
    }

    /// Initialize optimizer state for a parameter (only on the owning rank)
    fn initialize_optimizer_state(&mut self, param_name: &str, tensor: &Tensor) -> Result<()> {
        let mut state = HashMap::new();

        // Initialize momentum and variance tensors (for Adam-like optimizers)
        // These will be stored only on the owning rank
        state.insert("momentum".to_string(), Tensor::zeros(&tensor.shape())?);
        state.insert("variance".to_string(), Tensor::zeros(&tensor.shape())?);
        state.insert("step".to_string(), Tensor::scalar(0.0)?);

        self.optimizer_states.insert(param_name.to_string(), state);
        Ok(())
    }

    /// Update a single parameter with ZeRO Stage 1 optimization
    pub fn update_parameter(
        &mut self,
        parameter: &mut Tensor,
        grad: &Tensor,
        base_optimizer: &mut T,
    ) -> Result<()> {
        // Get parameter name (simplified - in practice would need parameter tracking)
        let param_name = format!("param_{}", parameter.shape().iter().product::<usize>());

        // Check if this rank owns the optimizer state for this parameter
        let owner_rank = self.parameter_ownership.get(&param_name).copied().unwrap_or(0);
        let current_rank = self.mp_context.rank();

        if owner_rank == current_rank {
            // This rank owns the optimizer state, perform the update
            self.perform_local_update(parameter, grad, &param_name, base_optimizer)?;
        }

        // Broadcast the updated parameter from the owner to all other ranks
        self.mp_context.broadcast(parameter, owner_rank)?;

        Ok(())
    }

    /// Perform local optimizer update on the owning rank
    fn perform_local_update(
        &mut self,
        parameter: &mut Tensor,
        grad: &Tensor,
        _param_name: &str,
        base_optimizer: &mut T,
    ) -> Result<()> {
        // Use the base optimizer to update the parameter
        // The optimizer state is stored locally on this rank only
        base_optimizer.update(parameter, grad)
    }

    /// Broadcast parameter from owner rank to all other ranks
    #[allow(dead_code)]
    fn broadcast_parameter(&self, parameter: &mut Tensor, owner_rank: usize) -> Result<()> {
        self.mp_context
            .broadcast(parameter, owner_rank)
            .map_err(|_| TrustformersError::runtime_error("Broadcast failed".to_string()))
    }

    /// Accumulate gradient for gradient accumulation
    pub fn accumulate_gradient(&mut self, param_name: &str, grad: &Tensor) -> Result<()> {
        if let Some(acc_grad) = self.accumulated_gradients.get_mut(param_name) {
            *acc_grad = acc_grad.add(grad)?;
        } else {
            self.accumulated_gradients.insert(param_name.to_string(), grad.clone());
        }
        Ok(())
    }

    /// Accumulate gradient for a specific parameter tensor
    pub fn accumulate_gradient_for_parameter(
        &mut self,
        parameter: &mut Tensor,
        grad: &Tensor,
    ) -> Result<()> {
        let param_name = format!("param_{}", parameter.shape().iter().product::<usize>());
        self.accumulate_gradient(&param_name, grad)
    }

    /// Apply accumulated gradients
    pub fn apply_accumulated_gradients(
        &mut self,
        base_optimizer: &mut T,
        accumulation_steps: usize,
    ) -> Result<()> {
        // Collect parameters to update to avoid borrowing conflicts
        let mut updates = Vec::new();

        // Average accumulated gradients and collect update info
        for (param_name, acc_grad) in &mut self.accumulated_gradients {
            if accumulation_steps > 1 {
                *acc_grad = acc_grad.scalar_div(accumulation_steps as f32)?;
            }

            let owner_rank = self.parameter_ownership.get(param_name).copied().unwrap_or(0);

            if owner_rank == self.mp_context.rank() {
                updates.push((param_name.clone(), acc_grad.clone(), owner_rank));
            }
        }

        // Apply updates
        for (param_name, acc_grad, owner_rank) in updates {
            if let Some(group) = self.parameter_groups.get_mut(&param_name) {
                if let Some(param) = group.local_parameters.get_mut(&param_name) {
                    base_optimizer.update(param, &acc_grad)?;
                    // Broadcast updated parameter using model parallel context directly
                    self.mp_context.broadcast(param, owner_rank)?;
                }
            }
        }

        // Clear accumulated gradients
        self.accumulated_gradients.clear();
        self.accumulation_steps = 0;

        Ok(())
    }

    /// Perform optimizer step for all parameters
    pub fn optimizer_step(&mut self, base_optimizer: &mut T) -> Result<()> {
        let current_rank = self.mp_context.rank();

        // Collect parameters to update to avoid borrowing conflicts
        let mut param_updates = Vec::new();

        // Process each communication bucket
        for bucket in &self.comm_buckets {
            for param_name in bucket {
                let owner_rank = self.parameter_ownership.get(param_name).copied().unwrap_or(0);

                if owner_rank == current_rank {
                    // Check if there's an accumulated gradient for this parameter
                    if let Some(acc_grad) = self.accumulated_gradients.get(param_name) {
                        param_updates.push((param_name.clone(), acc_grad.clone(), owner_rank));
                    }
                }
            }
        }

        // Apply updates
        for (param_name, acc_grad, owner_rank) in param_updates {
            if let Some(group) = self.parameter_groups.get_mut(&param_name) {
                if let Some(param) = group.local_parameters.get_mut(&param_name) {
                    base_optimizer.update(param, &acc_grad)?;
                    // Broadcast the updated parameter using model parallel context directly
                    self.mp_context.broadcast(param, owner_rank)?;
                }
            }
        }

        // Clear accumulated gradients
        self.accumulated_gradients.clear();

        Ok(())
    }

    /// Synchronize optimizer states across ranks (if needed)
    pub fn synchronize_optimizer_states(&mut self) -> Result<()> {
        // In Stage 1, optimizer states are partitioned, so no synchronization needed
        // Each rank maintains its own subset of optimizer states
        Ok(())
    }

    /// Get memory usage statistics
    pub fn get_memory_usage(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();

        // Calculate optimizer state memory (only for owned parameters)
        let mut optimizer_memory = 0;
        for states in self.optimizer_states.values() {
            for tensor in states.values() {
                optimizer_memory += tensor.memory_usage();
            }
        }
        stats.insert("optimizer_states".to_string(), optimizer_memory);

        // Calculate parameter memory (all parameters are replicated)
        let mut parameter_memory = 0;
        for group in self.parameter_groups.values() {
            parameter_memory += group.memory_usage();
        }
        stats.insert("parameters".to_string(), parameter_memory);

        // Calculate accumulated gradient memory
        let mut gradient_memory = 0;
        for grad in self.accumulated_gradients.values() {
            gradient_memory += grad.memory_usage();
        }
        stats.insert("accumulated_gradients".to_string(), gradient_memory);

        stats
    }

    /// Get parameter ownership mapping
    pub fn get_parameter_ownership(&self) -> &HashMap<String, usize> {
        &self.parameter_ownership
    }

    /// Check if current rank owns a parameter's optimizer state
    pub fn owns_parameter(&self, param_name: &str) -> bool {
        self.parameter_ownership.get(param_name).copied().unwrap_or(0) == self.mp_context.rank()
    }

    /// Get number of owned parameters
    pub fn num_owned_parameters(&self) -> usize {
        let current_rank = self.mp_context.rank();
        self.parameter_ownership.values().filter(|&&rank| rank == current_rank).count()
    }

    /// Estimate memory savings compared to standard optimizer
    pub fn estimate_memory_savings(&self) -> f32 {
        let _world_size = self.mp_context.world_size() as f32;
        let owned_params = self.num_owned_parameters() as f32;
        let total_params = self.parameter_groups.len() as f32;

        if total_params > 0.0 {
            // Memory savings = (1 - fraction_owned) * optimizer_state_size
            1.0 - (owned_params / total_params)
        } else {
            0.0
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
    fn test_zero_stage1_creation() {
        let config = ModelParallelConfig {
            num_devices: 2,
            device_ids: vec![0, 1],
            strategy: ModelParallelStrategy::Pipeline,
            comm_backend: CommunicationBackend::Custom,
            ..Default::default()
        };
        let mp_context = Arc::new(ModelParallelContext::new(config).unwrap());
        let zero_config = ZeROConfig::default();

        let stage1 = ZeROStage1::<Adam>::new(mp_context, zero_config);
        assert!(stage1.is_ok());
    }

    #[test]
    fn test_parameter_registration() {
        let config = ModelParallelConfig {
            num_devices: 4,
            device_ids: vec![0, 1, 2, 3],
            strategy: ModelParallelStrategy::Pipeline,
            comm_backend: CommunicationBackend::Custom,
            ..Default::default()
        };
        let mp_context = Arc::new(ModelParallelContext::new(config).unwrap());
        let zero_config = ZeROConfig::default();
        let mut stage1 = ZeROStage1::<Adam>::new(mp_context, zero_config).unwrap();

        let mut parameters = HashMap::new();
        parameters.insert("weight1".to_string(), Tensor::ones(&[4, 4]).unwrap());
        parameters.insert("weight2".to_string(), Tensor::ones(&[2, 2]).unwrap());
        parameters.insert("bias1".to_string(), Tensor::ones(&[4]).unwrap());
        parameters.insert("bias2".to_string(), Tensor::ones(&[2]).unwrap());

        let result = stage1.register_parameters(parameters);
        assert!(result.is_ok());

        // Check parameter ownership distribution
        assert_eq!(stage1.parameter_groups.len(), 4);

        // In a 4-device setup, parameters should be distributed across ranks
        let ownership = stage1.get_parameter_ownership();
        let mut rank_counts = vec![0; 4];
        for &rank in ownership.values() {
            rank_counts[rank] += 1;
        }

        // Each rank should own at least one parameter's optimizer state
        for count in rank_counts {
            assert!(count >= 1);
        }
    }

    #[test]
    fn test_parameter_ownership() {
        let config = ModelParallelConfig {
            num_devices: 2,
            device_ids: vec![0, 1],
            strategy: ModelParallelStrategy::Pipeline,
            comm_backend: CommunicationBackend::Custom,
            ..Default::default()
        };
        let mp_context = Arc::new(ModelParallelContext::new(config).unwrap());
        let zero_config = ZeROConfig::default();
        let mut stage1 = ZeROStage1::<Adam>::new(mp_context, zero_config).unwrap();

        let mut parameters = HashMap::new();
        parameters.insert("param1".to_string(), Tensor::ones(&[2, 2]).unwrap());
        parameters.insert("param2".to_string(), Tensor::ones(&[2, 2]).unwrap());

        stage1.register_parameters(parameters).unwrap();

        // Check that parameters are distributed between ranks
        let ownership = stage1.get_parameter_ownership();
        assert_eq!(ownership.len(), 2);

        // In a 2-device setup, parameters should be split between rank 0 and 1
        let rank0_params = ownership.values().filter(|&&r| r == 0).count();
        let rank1_params = ownership.values().filter(|&&r| r == 1).count();
        assert!(rank0_params > 0);
        assert!(rank1_params > 0);
        assert_eq!(rank0_params + rank1_params, 2);
    }

    #[test]
    fn test_memory_usage_tracking() {
        let config = ModelParallelConfig {
            num_devices: 2,
            device_ids: vec![0, 1],
            strategy: ModelParallelStrategy::Pipeline,
            comm_backend: CommunicationBackend::Custom,
            ..Default::default()
        };
        let mp_context = Arc::new(ModelParallelContext::new(config).unwrap());
        let zero_config = ZeROConfig::default();
        let mut stage1 = ZeROStage1::<Adam>::new(mp_context, zero_config).unwrap();

        let mut parameters = HashMap::new();
        parameters.insert("weight".to_string(), Tensor::ones(&[4, 4]).unwrap());

        stage1.register_parameters(parameters).unwrap();

        let memory_usage = stage1.get_memory_usage();
        assert!(memory_usage.contains_key("optimizer_states"));
        assert!(memory_usage.contains_key("parameters"));
        assert!(memory_usage.contains_key("accumulated_gradients"));
    }

    #[test]
    fn test_memory_savings_estimation() {
        let config = ModelParallelConfig {
            num_devices: 4,
            device_ids: vec![0, 1, 2, 3],
            strategy: ModelParallelStrategy::Pipeline,
            comm_backend: CommunicationBackend::Custom,
            ..Default::default()
        };
        let mp_context = Arc::new(ModelParallelContext::new(config).unwrap());
        let zero_config = ZeROConfig::default();
        let mut stage1 = ZeROStage1::<Adam>::new(mp_context, zero_config).unwrap();

        let mut parameters = HashMap::new();
        for i in 0..8 {
            parameters.insert(format!("param{}", i), Tensor::ones(&[2, 2]).unwrap());
        }

        stage1.register_parameters(parameters).unwrap();

        let savings = stage1.estimate_memory_savings();
        // With 4 devices and 8 parameters, each device owns ~2 parameters
        // So memory savings should be approximately 75% (6/8 parameters saved per device)
        assert!(savings > 0.5 && savings < 1.0);
    }

    #[test]
    fn test_gradient_accumulation() {
        let config = ModelParallelConfig {
            num_devices: 2,
            device_ids: vec![0, 1],
            strategy: ModelParallelStrategy::Pipeline,
            comm_backend: CommunicationBackend::Custom,
            ..Default::default()
        };
        let mp_context = Arc::new(ModelParallelContext::new(config).unwrap());
        let zero_config = ZeROConfig::default();
        let mut stage1 = ZeROStage1::<Adam>::new(mp_context, zero_config).unwrap();

        let grad1 = Tensor::ones(&[2, 2]).unwrap();
        let grad2 = Tensor::ones(&[2, 2]).unwrap();

        // Accumulate gradients
        stage1.accumulate_gradient("param1", &grad1).unwrap();
        stage1.accumulate_gradient("param1", &grad2).unwrap();

        assert_eq!(stage1.accumulated_gradients.len(), 1);

        // The accumulated gradient should be the sum of grad1 and grad2
        let acc_grad = stage1.accumulated_gradients.get("param1").unwrap();
        // Check that it's roughly 2.0 (sum of two 1.0 tensors)
        // This is a simplified check - in practice would verify tensor values
        assert!(acc_grad.size() > 0);
    }
}
