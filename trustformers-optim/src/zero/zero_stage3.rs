//! ZeRO Stage 3: Full Parameter Partitioning
//!
//! Stage 3 partitions optimizer states, gradients, AND parameters across devices.
//! This provides maximum memory savings by distributing all model components,
//! but requires parameter gathering for forward passes and additional communication.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
use trustformers_core::errors::Result;
use trustformers_core::parallel::ModelParallelContext;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

use super::zero_optimizer::ZeROConfig;
use super::zero_stage2::ZeROStage2;
use super::zero_utils::{partition_parameters, ParameterPartition};

/// ZeRO Stage 3 implementation - partitions optimizer states + gradients + parameters
pub struct ZeROStage3<T: Optimizer> {
    /// Stage 2 functionality (optimizer + gradient partitioning)
    stage2: ZeROStage2<T>,
    /// Model parallel context for communication
    mp_context: Arc<ModelParallelContext>,
    /// ZeRO configuration
    config: ZeROConfig,
    /// Partitioned parameters
    parameter_partitions: HashMap<String, ParameterPartition>,
    /// Currently gathered parameters (for forward pass)
    gathered_parameters: HashMap<String, Tensor>,
    /// Parameter access schedule for optimal gathering
    access_schedule: Vec<Vec<String>>,
    /// Memory-efficient parameter management
    param_memory_manager: ParameterMemoryManager,
    /// Communication scheduler for overlapping
    comm_scheduler: CommunicationScheduler,
    /// Phantom data for optimizer type
    _phantom: PhantomData<T>,
}

/// Memory management for parameters in Stage 3
#[derive(Debug)]
struct ParameterMemoryManager {
    /// Maximum memory threshold for gathered parameters
    max_memory_bytes: usize,
    /// Currently used memory
    current_memory_bytes: usize,
    /// LRU cache for parameter gathering
    lru_order: Vec<String>,
    /// Parameter memory usage tracking
    param_memory_usage: HashMap<String, usize>,
}

impl ParameterMemoryManager {
    fn new(max_memory_mb: usize) -> Self {
        Self {
            max_memory_bytes: max_memory_mb * 1024 * 1024,
            current_memory_bytes: 0,
            lru_order: Vec::new(),
            param_memory_usage: HashMap::new(),
        }
    }

    /// Check if we can gather a parameter without exceeding memory limit
    fn can_gather(&self, param_size: usize) -> bool {
        self.current_memory_bytes + param_size <= self.max_memory_bytes
    }

    /// Add a gathered parameter to memory tracking
    fn add_parameter(&mut self, name: String, size: usize) {
        self.current_memory_bytes += size;
        self.param_memory_usage.insert(name.clone(), size);

        // Update LRU order
        if let Some(pos) = self.lru_order.iter().position(|x| x == &name) {
            self.lru_order.remove(pos);
        }
        self.lru_order.push(name);
    }

    /// Remove a parameter from memory tracking
    fn remove_parameter(&mut self, name: &str) -> usize {
        if let Some(size) = self.param_memory_usage.remove(name) {
            self.current_memory_bytes = self.current_memory_bytes.saturating_sub(size);
            if let Some(pos) = self.lru_order.iter().position(|x| x == name) {
                self.lru_order.remove(pos);
            }
            size
        } else {
            0
        }
    }

    /// Get least recently used parameters to evict
    fn get_lru_parameters(&self, target_size: usize) -> Vec<String> {
        let mut evict_list = Vec::new();
        let mut freed_size = 0;

        for name in &self.lru_order {
            if freed_size >= target_size {
                break;
            }
            if let Some(&size) = self.param_memory_usage.get(name) {
                evict_list.push(name.clone());
                freed_size += size;
            }
        }

        evict_list
    }

    /// Get current memory usage percentage
    fn memory_usage_percent(&self) -> f32 {
        (self.current_memory_bytes as f32 / self.max_memory_bytes as f32) * 100.0
    }
}

/// Communication scheduler for overlapping computation and communication
#[derive(Debug)]
struct CommunicationScheduler {
    /// Pending gather operations
    pending_gathers: Vec<String>,
    /// Pending release operations
    pending_releases: Vec<String>,
    /// Communication queue for batching
    #[allow(dead_code)]
    comm_queue: Vec<CommOp>,
}

#[derive(Debug, Clone)]
enum CommOp {
    #[allow(dead_code)]
    Gather(String),
    Release(String),
    AllGather(Vec<String>),
    #[allow(dead_code)]
    ReduceScatter(Vec<String>),
}

impl CommunicationScheduler {
    fn new() -> Self {
        Self {
            pending_gathers: Vec::new(),
            pending_releases: Vec::new(),
            comm_queue: Vec::new(),
        }
    }

    /// Schedule a parameter gather operation
    fn schedule_gather(&mut self, param_name: String) {
        if !self.pending_gathers.contains(&param_name) {
            self.pending_gathers.push(param_name);
        }
    }

    /// Schedule a parameter release operation
    #[allow(dead_code)]
    fn schedule_release(&mut self, param_name: String) {
        if !self.pending_releases.contains(&param_name) {
            self.pending_releases.push(param_name);
        }
    }

    /// Batch operations for efficient communication
    fn flush_operations(&mut self) -> Vec<CommOp> {
        let mut ops = Vec::new();

        // Batch gather operations
        if !self.pending_gathers.is_empty() {
            ops.push(CommOp::AllGather(self.pending_gathers.drain(..).collect()));
        }

        // Process releases
        for param_name in self.pending_releases.drain(..) {
            ops.push(CommOp::Release(param_name));
        }

        ops
    }
}

impl<T: Optimizer> ZeROStage3<T> {
    /// Create a new ZeRO Stage 3 optimizer
    pub fn new(mp_context: Arc<ModelParallelContext>, config: ZeROConfig) -> Result<Self> {
        let stage2 = ZeROStage2::new(mp_context.clone(), config.clone())?;

        let param_memory_manager = ParameterMemoryManager::new(config.max_memory_usage_mb);
        let comm_scheduler = CommunicationScheduler::new();

        Ok(Self {
            stage2,
            mp_context: mp_context.clone(),
            config,
            parameter_partitions: HashMap::new(),
            gathered_parameters: HashMap::new(),
            access_schedule: Vec::new(),
            param_memory_manager,
            comm_scheduler,
            _phantom: PhantomData,
        })
    }

    /// Register parameters and set up full partitioning
    pub fn register_parameters(&mut self, parameters: HashMap<String, Tensor>) -> Result<()> {
        // First register with Stage 2 for optimizer and gradient partitioning
        self.stage2.register_parameters(parameters.clone())?;

        let world_size = self.mp_context.world_size();
        let rank = self.mp_context.rank();

        // Partition parameters across devices
        self.parameter_partitions = partition_parameters(&parameters, world_size, rank)?;

        // Create parameter access schedule
        self.create_access_schedule(&parameters)?;

        println!("ZeRO Stage 3: Registered {} parameters", parameters.len());
        println!(
            "  Parameter partitions: {}",
            self.parameter_partitions.len()
        );
        println!("  Access schedule layers: {}", self.access_schedule.len());

        Ok(())
    }

    /// Create optimal parameter access schedule
    fn create_access_schedule(&mut self, parameters: &HashMap<String, Tensor>) -> Result<()> {
        // Group parameters by likely access patterns (e.g., by layer)
        // This is simplified - in practice would analyze model structure

        let param_names: Vec<String> = parameters.keys().cloned().collect();
        let chunk_size = self.config.prefetch_depth.max(1);

        for chunk in param_names.chunks(chunk_size) {
            self.access_schedule.push(chunk.to_vec());
        }

        Ok(())
    }

    /// Gather parameters for forward pass
    pub fn gather_parameters(
        &mut self,
        parameter_names: &[String],
    ) -> Result<HashMap<String, Tensor>> {
        let mut gathered = HashMap::new();

        // Check memory constraints and evict if necessary
        self.manage_memory_for_gathering(parameter_names)?;

        // Schedule gather operations
        for name in parameter_names {
            if !self.gathered_parameters.contains_key(name) {
                self.comm_scheduler.schedule_gather(name.clone());
            }
        }

        // Execute scheduled gather operations
        self.execute_gather_operations()?;

        // Return gathered parameters
        for name in parameter_names {
            if let Some(param) = self.gathered_parameters.get(name) {
                gathered.insert(name.clone(), param.clone());
            }
        }

        Ok(gathered)
    }

    /// Manage memory by evicting LRU parameters if needed
    fn manage_memory_for_gathering(&mut self, parameter_names: &[String]) -> Result<()> {
        // Calculate memory needed for new parameters
        let mut memory_needed = 0;
        for name in parameter_names {
            if !self.gathered_parameters.contains_key(name) {
                if let Some(partition) = self.parameter_partitions.get(name) {
                    memory_needed +=
                        partition.partition_info.global_shape.iter().product::<usize>() * 4;
                    // Assume f32
                }
            }
        }

        // Check if we need to evict parameters
        if !self.param_memory_manager.can_gather(memory_needed) {
            let evict_list = self.param_memory_manager.get_lru_parameters(memory_needed);
            for param_name in evict_list {
                self.release_parameter(&param_name)?;
            }
        }

        Ok(())
    }

    /// Execute gathered gather operations
    fn execute_gather_operations(&mut self) -> Result<()> {
        let ops = self.comm_scheduler.flush_operations();

        for op in ops {
            match op {
                CommOp::AllGather(param_names) => {
                    self.batch_gather_parameters(&param_names)?;
                },
                CommOp::Gather(param_name) => {
                    self.gather_single_parameter(&param_name)?;
                },
                CommOp::Release(param_name) => {
                    self.release_parameter(&param_name)?;
                },
                _ => {}, // Other operations handled elsewhere
            }
        }

        Ok(())
    }

    /// Gather multiple parameters in a batch for efficiency
    fn batch_gather_parameters(&mut self, param_names: &[String]) -> Result<()> {
        // Collect partitions that need gathering
        let mut partition_names = Vec::new();
        for name in param_names {
            if let Some(partition) = self.parameter_partitions.get(name) {
                if !partition.is_gathered {
                    partition_names.push(name.clone());
                }
            }
        }

        // Gather each parameter individually to avoid borrow checker issues
        for name in partition_names {
            if let Some(partition) = self.parameter_partitions.get_mut(&name) {
                partition.gather(&self.mp_context)?;

                if let Some(full_param) = &partition.full_parameter {
                    let param_size = full_param.memory_usage();
                    self.gathered_parameters.insert(name.clone(), full_param.clone());
                    self.param_memory_manager.add_parameter(name, param_size);
                }
            }
        }

        Ok(())
    }

    /// Gather a single parameter
    fn gather_single_parameter(&mut self, param_name: &str) -> Result<()> {
        if let Some(partition) = self.parameter_partitions.get_mut(param_name) {
            if !partition.is_gathered {
                partition.gather(&self.mp_context)?;

                if let Some(full_param) = &partition.full_parameter {
                    let param_size = full_param.memory_usage();
                    self.gathered_parameters.insert(param_name.to_string(), full_param.clone());
                    self.param_memory_manager.add_parameter(param_name.to_string(), param_size);
                }
            }
        }
        Ok(())
    }

    /// Release gathered parameters to save memory
    pub fn release_parameters(&mut self, parameter_names: &[String]) -> Result<()> {
        for name in parameter_names {
            self.release_parameter(name)?;
        }
        Ok(())
    }

    /// Release a single parameter
    fn release_parameter(&mut self, param_name: &str) -> Result<()> {
        // Remove from gathered parameters
        self.gathered_parameters.remove(param_name);

        // Update memory tracking
        self.param_memory_manager.remove_parameter(param_name);

        // Release in partition
        if let Some(partition) = self.parameter_partitions.get_mut(param_name) {
            partition.release();
        }

        Ok(())
    }

    /// Update gradients with ZeRO Stage 3 optimization
    pub fn update_gradients(&mut self, gradients: HashMap<String, Tensor>) -> Result<()> {
        // Use Stage 2 for gradient handling
        self.stage2.update_gradients(gradients)
    }

    /// Apply accumulated gradients with ZeRO Stage 3
    pub fn apply_accumulated_gradients(
        &mut self,
        base_optimizer: &mut T,
        accumulation_steps: usize,
    ) -> Result<()> {
        // First gather parameters needed for optimizer updates
        let param_names_for_update: Vec<String> =
            self.parameter_partitions.keys().cloned().collect();
        self.gather_parameters(&param_names_for_update)?;

        // Use Stage 2 for gradient application
        self.stage2.apply_accumulated_gradients(base_optimizer, accumulation_steps)?;

        // Optionally release parameters after update to save memory
        if self.config.max_memory_usage_mb > 0 {
            let memory_usage_percent = self.param_memory_manager.memory_usage_percent();
            if memory_usage_percent > 80.0 {
                // Release if using > 80% of memory
                let lru_params = self.param_memory_manager.get_lru_parameters(
                    self.param_memory_manager.current_memory_bytes / 4, // Release 25%
                );
                for param_name in lru_params {
                    self.release_parameter(&param_name)?;
                }
            }
        }

        Ok(())
    }

    /// Perform optimizer step with ZeRO Stage 3
    pub fn optimizer_step(&mut self, base_optimizer: &mut T) -> Result<()> {
        // Gather parameters needed for optimizer step
        let param_names: Vec<String> = self.parameter_partitions.keys().cloned().collect();
        self.gather_parameters(&param_names)?;

        // Use Stage 2 for optimizer step
        self.stage2.optimizer_step(base_optimizer)?;

        // Redistribute updated parameters back to partitions
        self.redistribute_updated_parameters()?;

        Ok(())
    }

    /// Redistribute updated parameters back to their partitions
    fn redistribute_updated_parameters(&mut self) -> Result<()> {
        for (name, gathered_param) in &self.gathered_parameters {
            if let Some(partition) = self.parameter_partitions.get_mut(name) {
                // In a real implementation, would scatter the updated parameter
                // back to all devices. For now, just update the local shard.
                // Simplified parameter scattering - just scale the parameter for demonstration
                let scale_factor = 1.0 / (partition.partition_info.world_size as f32);
                partition.local_shard = gathered_param.mul_scalar(scale_factor)?;
            }
        }
        Ok(())
    }

    /// Prefetch parameters for next forward pass
    pub fn prefetch_parameters(&mut self, layer_idx: usize) -> Result<()> {
        if layer_idx < self.access_schedule.len() {
            let param_names = &self.access_schedule[layer_idx];

            // Schedule prefetch operations
            for name in param_names {
                self.comm_scheduler.schedule_gather(name.clone());
            }

            // Execute prefetch in background (simplified)
            self.execute_gather_operations()?;
        }
        Ok(())
    }

    /// Get comprehensive memory usage statistics for Stage 3
    pub fn get_memory_usage(&self) -> HashMap<String, usize> {
        let mut stats = self.stage2.get_memory_usage();

        // Add parameter partition memory
        let mut partition_memory = 0;
        for partition in self.parameter_partitions.values() {
            partition_memory += partition.memory_usage();
        }
        stats.insert("parameter_partitions".to_string(), partition_memory);

        // Add gathered parameter memory
        let mut gathered_memory = 0;
        for param in self.gathered_parameters.values() {
            gathered_memory += param.memory_usage();
        }
        stats.insert("gathered_parameters".to_string(), gathered_memory);

        // Add memory manager overhead
        stats.insert(
            "memory_manager_overhead".to_string(),
            self.param_memory_manager.current_memory_bytes,
        );

        stats
    }

    /// Estimate memory savings compared to standard optimizer
    pub fn estimate_memory_savings(&self) -> HashMap<String, f32> {
        let mut savings = self.stage2.estimate_memory_savings();

        // Parameter memory savings
        let world_size = self.mp_context.world_size() as f32;
        let param_savings = 1.0 - (1.0 / world_size); // Each rank stores 1/world_size of parameters
        savings.insert("parameters".to_string(), param_savings);

        // Update total savings (weighted average of all components)
        let total_savings =
            (savings["optimizer_states"] + savings["gradients"] + param_savings) / 3.0;
        savings.insert("total".to_string(), total_savings);

        savings
    }

    /// Get parameter partition information
    pub fn get_parameter_partition(&self, param_name: &str) -> Option<&ParameterPartition> {
        self.parameter_partitions.get(param_name)
    }

    /// Check if a parameter is currently gathered
    pub fn is_parameter_gathered(&self, param_name: &str) -> bool {
        self.gathered_parameters.contains_key(param_name)
    }

    /// Get current memory usage percentage
    pub fn memory_usage_percent(&self) -> f32 {
        self.param_memory_manager.memory_usage_percent()
    }

    /// Force garbage collection of unused parameters
    pub fn garbage_collect(&mut self) -> Result<()> {
        // Release all gathered parameters that haven't been accessed recently
        let all_params: Vec<String> = self.gathered_parameters.keys().cloned().collect();
        for param_name in all_params {
            self.release_parameter(&param_name)?;
        }

        println!(
            "ZeRO Stage 3: Garbage collection completed. Memory usage: {:.1}%",
            self.memory_usage_percent()
        );

        Ok(())
    }

    /// Get stage 2 reference for advanced operations
    pub fn stage2(&self) -> &ZeROStage2<T> {
        &self.stage2
    }

    /// Get mutable stage 2 reference
    pub fn stage2_mut(&mut self) -> &mut ZeROStage2<T> {
        &mut self.stage2
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
    fn test_zero_stage3_creation() {
        let config = ModelParallelConfig {
            num_devices: 2,
            device_ids: vec![0, 1],
            strategy: ModelParallelStrategy::Pipeline,
            comm_backend: CommunicationBackend::Custom,
            ..Default::default()
        };
        let mp_context = Arc::new(ModelParallelContext::new(config).unwrap());
        let zero_config = ZeROConfig::default();

        let stage3 = ZeROStage3::<Adam>::new(mp_context, zero_config);
        assert!(stage3.is_ok());
    }

    #[test]
    fn test_parameter_partitioning() {
        let config = ModelParallelConfig {
            num_devices: 4,
            device_ids: vec![0, 1, 2, 3],
            strategy: ModelParallelStrategy::Pipeline,
            comm_backend: CommunicationBackend::Custom,
            ..Default::default()
        };
        let mp_context = Arc::new(ModelParallelContext::new(config).unwrap());
        let zero_config = ZeROConfig::default();
        let mut stage3 = ZeROStage3::<Adam>::new(mp_context, zero_config).unwrap();

        let mut parameters = HashMap::new();
        parameters.insert("weight1".to_string(), Tensor::ones(&[16, 16]).unwrap());
        parameters.insert("weight2".to_string(), Tensor::ones(&[8, 8]).unwrap());
        parameters.insert("bias1".to_string(), Tensor::ones(&[16]).unwrap());

        let result = stage3.register_parameters(parameters);
        assert!(result.is_ok());

        // Check that parameter partitions were created
        assert_eq!(stage3.parameter_partitions.len(), 3);
        assert!(stage3.get_parameter_partition("weight1").is_some());
        assert!(stage3.get_parameter_partition("weight2").is_some());
        assert!(stage3.get_parameter_partition("bias1").is_some());

        // Check partition metadata
        for partition in stage3.parameter_partitions.values() {
            assert_eq!(partition.partition_info.world_size, 4);
            assert!(!partition.is_gathered); // Initially not gathered
        }
    }

    #[test]
    fn test_parameter_gathering_and_release() {
        let config = ModelParallelConfig {
            num_devices: 2,
            device_ids: vec![0, 1],
            strategy: ModelParallelStrategy::Pipeline,
            comm_backend: CommunicationBackend::Custom,
            ..Default::default()
        };
        let mp_context = Arc::new(ModelParallelContext::new(config).unwrap());
        let zero_config = ZeROConfig {
            max_memory_usage_mb: 100, // Small memory limit for testing
            ..Default::default()
        };
        let mut stage3 = ZeROStage3::<Adam>::new(mp_context, zero_config).unwrap();

        let mut parameters = HashMap::new();
        parameters.insert("weight1".to_string(), Tensor::ones(&[4, 4]).unwrap());
        parameters.insert("weight2".to_string(), Tensor::ones(&[4, 4]).unwrap());

        stage3.register_parameters(parameters).unwrap();

        // Test parameter gathering
        let param_names = vec!["weight1".to_string()];
        let gathered = stage3.gather_parameters(&param_names).unwrap();
        assert_eq!(gathered.len(), 1);
        assert!(stage3.is_parameter_gathered("weight1"));
        assert!(!stage3.is_parameter_gathered("weight2"));

        // Test parameter release
        stage3.release_parameters(&param_names).unwrap();
        assert!(!stage3.is_parameter_gathered("weight1"));
    }

    #[test]
    fn test_memory_management() {
        let mut memory_manager = ParameterMemoryManager::new(10); // 10 MB limit

        // Test adding parameters
        memory_manager.add_parameter("param1".to_string(), 5 * 1024 * 1024); // 5 MB
        memory_manager.add_parameter("param2".to_string(), 3 * 1024 * 1024); // 3 MB

        assert_eq!(memory_manager.current_memory_bytes, 8 * 1024 * 1024);
        assert!(memory_manager.can_gather(2 * 1024 * 1024)); // Can fit 2 MB more
        assert!(!memory_manager.can_gather(3 * 1024 * 1024)); // Cannot fit 3 MB more

        // Test LRU eviction
        let lru_params = memory_manager.get_lru_parameters(4 * 1024 * 1024);
        assert_eq!(lru_params.len(), 1);
        assert_eq!(lru_params[0], "param1"); // First added, first to evict

        // Test parameter removal
        memory_manager.remove_parameter("param1");
        assert_eq!(memory_manager.current_memory_bytes, 3 * 1024 * 1024);
    }

    #[test]
    fn test_communication_scheduler() {
        let mut scheduler = CommunicationScheduler::new();

        // Schedule operations
        scheduler.schedule_gather("param1".to_string());
        scheduler.schedule_gather("param2".to_string());
        scheduler.schedule_release("param3".to_string());

        // Flush operations
        let ops = scheduler.flush_operations();
        assert_eq!(ops.len(), 2); // One batch gather + one release

        // Check operation types
        assert!(matches!(ops[0], CommOp::AllGather(_)));
        assert!(matches!(ops[1], CommOp::Release(_)));
    }

    #[test]
    fn test_access_schedule_creation() {
        let config = ModelParallelConfig {
            num_devices: 2,
            device_ids: vec![0, 1],
            strategy: ModelParallelStrategy::Pipeline,
            comm_backend: CommunicationBackend::Custom,
            ..Default::default()
        };
        let mp_context = Arc::new(ModelParallelContext::new(config).unwrap());
        let zero_config = ZeROConfig {
            prefetch_depth: 2,
            ..Default::default()
        };
        let mut stage3 = ZeROStage3::<Adam>::new(mp_context, zero_config).unwrap();

        let mut parameters = HashMap::new();
        for i in 0..6 {
            parameters.insert(format!("param{}", i), Tensor::ones(&[2, 2]).unwrap());
        }

        stage3.register_parameters(parameters).unwrap();

        // Check that access schedule was created
        assert!(!stage3.access_schedule.is_empty());

        // With prefetch_depth=2, should have chunks of size 2
        for chunk in &stage3.access_schedule {
            assert!(chunk.len() <= 2);
        }
    }

    #[test]
    fn test_memory_savings_estimation() {
        let config = ModelParallelConfig {
            num_devices: 8,
            device_ids: vec![0, 1, 2, 3, 4, 5, 6, 7],
            strategy: ModelParallelStrategy::Pipeline,
            comm_backend: CommunicationBackend::Custom,
            ..Default::default()
        };
        let mp_context = Arc::new(ModelParallelContext::new(config).unwrap());
        let zero_config = ZeROConfig::default();
        let mut stage3 = ZeROStage3::<Adam>::new(mp_context, zero_config).unwrap();

        let mut parameters = HashMap::new();
        for i in 0..16 {
            parameters.insert(format!("param{}", i), Tensor::ones(&[4, 4]).unwrap());
        }

        stage3.register_parameters(parameters).unwrap();

        let savings = stage3.estimate_memory_savings();

        // Should have savings for optimizer states, gradients, AND parameters
        assert!(savings.contains_key("optimizer_states"));
        assert!(savings.contains_key("gradients"));
        assert!(savings.contains_key("parameters"));
        assert!(savings.contains_key("total"));

        // With 8 devices, each component should save ~87.5% (7/8)
        assert!(savings["parameters"] > 0.8);
        assert!(savings["gradients"] > 0.8);

        // Total savings should be very high
        assert!(savings["total"] > 0.8);
    }

    #[test]
    fn test_garbage_collection() {
        let config = ModelParallelConfig {
            num_devices: 2,
            device_ids: vec![0, 1],
            strategy: ModelParallelStrategy::Pipeline,
            comm_backend: CommunicationBackend::Custom,
            ..Default::default()
        };
        let mp_context = Arc::new(ModelParallelContext::new(config).unwrap());
        let zero_config = ZeROConfig::default();
        let mut stage3 = ZeROStage3::<Adam>::new(mp_context, zero_config).unwrap();

        let mut parameters = HashMap::new();
        parameters.insert("weight".to_string(), Tensor::ones(&[4, 4]).unwrap());

        stage3.register_parameters(parameters).unwrap();

        // Gather parameter
        let param_names = vec!["weight".to_string()];
        stage3.gather_parameters(&param_names).unwrap();
        assert!(stage3.is_parameter_gathered("weight"));

        // Garbage collect
        stage3.garbage_collect().unwrap();
        assert!(!stage3.is_parameter_gathered("weight"));
        assert_eq!(stage3.memory_usage_percent(), 0.0);
    }
}
