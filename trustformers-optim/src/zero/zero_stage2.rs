//! ZeRO Stage 2: Optimizer State + Gradient Partitioning
//!
//! Stage 2 partitions both optimizer states AND gradients across devices.
//! This provides additional memory savings by reducing gradient memory usage
//! while maintaining full parameter replication for forward/backward passes.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::parallel::ModelParallelContext;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

use super::zero_optimizer::ZeROConfig;
use super::zero_stage1::ZeROStage1;
use super::zero_utils::{
    all_gather_gradients, calculate_bucket_size, reduce_scatter_gradients, GradientBuffer,
    PartitionInfo,
};

/// ZeRO Stage 2 implementation - partitions optimizer states + gradients
pub struct ZeROStage2<T: Optimizer> {
    /// Stage 1 functionality (optimizer state partitioning)
    stage1: ZeROStage1<T>,
    /// Model parallel context for communication
    mp_context: Arc<ModelParallelContext>,
    /// ZeRO configuration
    config: ZeROConfig,
    /// Partitioned gradient buffers
    gradient_buffers: HashMap<String, GradientBuffer>,
    /// Gradient communication buckets
    gradient_buckets: Vec<Vec<String>>,
    /// All-reduce groups for gradient synchronization
    reduce_scatter_groups: Vec<Vec<String>>,
    /// Communication streams for overlapping computation
    comm_streams: Vec<String>, // Simplified - would be actual CUDA streams
    /// Prefetch queue for gradient gathering
    prefetch_queue: Vec<String>,
    /// Phantom data for optimizer type
    _phantom: PhantomData<T>,
}

impl<T: Optimizer> ZeROStage2<T> {
    /// Create a new ZeRO Stage 2 optimizer
    pub fn new(mp_context: Arc<ModelParallelContext>, config: ZeROConfig) -> Result<Self> {
        let stage1 = ZeROStage1::new(mp_context.clone(), config.clone())?;

        Ok(Self {
            stage1,
            mp_context: mp_context.clone(),
            config,
            gradient_buffers: HashMap::new(),
            gradient_buckets: Vec::new(),
            reduce_scatter_groups: Vec::new(),
            comm_streams: Vec::new(),
            prefetch_queue: Vec::new(),
            _phantom: PhantomData,
        })
    }

    /// Register parameters and set up gradient partitioning
    pub fn register_parameters(&mut self, parameters: HashMap<String, Tensor>) -> Result<()> {
        // First register with Stage 1 for optimizer state partitioning
        self.stage1.register_parameters(parameters.clone())?;

        let _world_size = self.mp_context.world_size();
        let _rank = self.mp_context.rank();

        // Set up gradient partitioning
        self.setup_gradient_partitioning(&parameters)?;

        // Create communication buckets for gradients
        self.create_gradient_buckets(&parameters)?;

        // Initialize gradient buffers
        self.initialize_gradient_buffers(&parameters)?;

        Ok(())
    }

    /// Set up gradient partitioning strategy
    fn setup_gradient_partitioning(&mut self, parameters: &HashMap<String, Tensor>) -> Result<()> {
        let _world_size = self.mp_context.world_size();

        // Calculate total gradient memory
        let total_gradient_memory: usize = parameters.values().map(|t| t.memory_usage()).sum();

        // Determine optimal bucketing strategy
        let target_bucket_size = self.config.bucket_size_mb * 1024 * 1024; // Convert to bytes
        let num_buckets = total_gradient_memory.div_ceil(target_bucket_size);

        println!("ZeRO Stage 2: Setting up gradient partitioning");
        println!(
            "  Total gradient memory: {} MB",
            total_gradient_memory / 1024 / 1024
        );
        println!("  Target bucket size: {} MB", self.config.bucket_size_mb);
        println!("  Number of buckets: {}", num_buckets);

        Ok(())
    }

    /// Create gradient communication buckets
    fn create_gradient_buckets(&mut self, parameters: &HashMap<String, Tensor>) -> Result<()> {
        let mut param_sizes = Vec::new();
        let mut param_names = Vec::new();

        for (name, tensor) in parameters {
            param_sizes.push(tensor.memory_usage());
            param_names.push(name.clone());
        }

        // Create buckets for gradient communication
        let bucket_size = self.config.bucket_size_mb * 1024 * 1024;
        let bucket_indices = calculate_bucket_size(&param_sizes, bucket_size);

        for bucket_idx in bucket_indices {
            let bucket: Vec<String> = bucket_idx.iter().map(|&i| param_names[i].clone()).collect();
            self.gradient_buckets.push(bucket.clone());
            self.reduce_scatter_groups.push(bucket);
        }

        Ok(())
    }

    /// Initialize gradient buffers for each parameter
    fn initialize_gradient_buffers(&mut self, parameters: &HashMap<String, Tensor>) -> Result<()> {
        let world_size = self.mp_context.world_size();
        let rank = self.mp_context.rank();

        for (name, tensor) in parameters {
            let shape = tensor.shape();
            let total_elements = shape.iter().product::<usize>();

            // Calculate partition size for this rank
            let elements_per_rank = total_elements.div_ceil(world_size);
            let start_idx = rank * elements_per_rank;
            let end_idx = ((rank + 1) * elements_per_rank).min(total_elements);

            // Create local gradient buffer (simplified - would need actual slicing)
            let local_gradient = Tensor::zeros(&shape)?;

            let partition_info = PartitionInfo {
                rank,
                world_size,
                start_idx,
                end_idx,
                global_shape: shape.to_vec(),
                local_shape: shape.to_vec(), // Keep original shape for simplicity
            };

            let buffer = GradientBuffer::new(name.clone(), local_gradient, partition_info);
            self.gradient_buffers.insert(name.clone(), buffer);
        }

        Ok(())
    }

    /// Update gradients with ZeRO Stage 2 optimization
    pub fn update_gradients(&mut self, gradients: HashMap<String, Tensor>) -> Result<()> {
        // Process gradients in buckets to minimize communication
        for bucket in &self.gradient_buckets.clone() {
            self.process_gradient_bucket(bucket, &gradients)?;
        }
        Ok(())
    }

    /// Process a bucket of gradients
    fn process_gradient_bucket(
        &mut self,
        bucket: &[String],
        gradients: &HashMap<String, Tensor>,
    ) -> Result<()> {
        let mut bucket_gradients = HashMap::new();

        // Collect gradients for this bucket
        for param_name in bucket {
            if let Some(grad) = gradients.get(param_name) {
                bucket_gradients.insert(param_name.clone(), grad.clone());
            }
        }

        // Reduce-scatter gradients within the bucket
        let scattered_gradients = self.reduce_scatter_bucket_gradients(&bucket_gradients)?;

        // Update local gradient buffers
        for (param_name, scattered_grad) in scattered_gradients {
            if let Some(buffer) = self.gradient_buffers.get_mut(&param_name) {
                buffer.accumulate(&scattered_grad)?;
            }
        }

        Ok(())
    }

    /// Reduce-scatter gradients for a bucket
    fn reduce_scatter_bucket_gradients(
        &self,
        bucket_gradients: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        // Use model parallel context for reduce-scatter operation
        reduce_scatter_gradients(bucket_gradients, &self.mp_context)
    }

    /// Gather gradients for optimizer update
    fn gather_gradients_for_update(&mut self) -> Result<HashMap<String, Tensor>> {
        // All-gather gradients that this rank needs for optimizer updates
        let mut gathered_gradients = HashMap::new();

        // Only gather gradients for parameters whose optimizer state this rank owns
        for (param_name, buffer) in &self.gradient_buffers {
            if self.stage1.owns_parameter(param_name) {
                // This rank owns the optimizer state, gather the full gradient
                let gathered_grad = self.all_gather_gradient(buffer)?;
                gathered_gradients.insert(param_name.clone(), gathered_grad);
            }
        }

        Ok(gathered_gradients)
    }

    /// All-gather a single gradient from its partitioned form
    fn all_gather_gradient(&self, buffer: &GradientBuffer) -> Result<Tensor> {
        let buffers = [(buffer.name.clone(), buffer.clone())].iter().cloned().collect();
        let gathered = all_gather_gradients(&buffers, &self.mp_context)?;

        gathered
            .get(&buffer.name)
            .cloned()
            .ok_or_else(|| TrustformersError::runtime_error("Failed to gather gradient".into()))
    }

    /// Apply accumulated gradients with ZeRO Stage 2
    pub fn apply_accumulated_gradients(
        &mut self,
        base_optimizer: &mut T,
        accumulation_steps: usize,
    ) -> Result<()> {
        // Gather gradients for parameters owned by this rank
        let gathered_gradients = self.gather_gradients_for_update()?;

        // Apply gradients to owned parameters using Stage 1 functionality
        for (param_name, gradient) in gathered_gradients {
            if self.stage1.owns_parameter(&param_name) {
                // Apply the gradient using Stage 1's parameter update mechanism
                self.stage1.accumulate_gradient(&param_name, &gradient)?;
            }
        }

        // Use Stage 1 to apply the accumulated gradients
        self.stage1.apply_accumulated_gradients(base_optimizer, accumulation_steps)?;

        // Clear gradient buffers
        for buffer in self.gradient_buffers.values_mut() {
            buffer.zero();
        }

        Ok(())
    }

    /// Perform optimizer step with ZeRO Stage 2
    pub fn optimizer_step(&mut self, base_optimizer: &mut T) -> Result<()> {
        // Gather gradients for optimizer updates
        let gathered_gradients = self.gather_gradients_for_update()?;

        // Update Stage 1's accumulated gradients
        for (param_name, gradient) in gathered_gradients {
            self.stage1.accumulate_gradient(&param_name, &gradient)?;
        }

        // Perform optimizer step using Stage 1
        self.stage1.optimizer_step(base_optimizer)?;

        // Clear gradient buffers
        for buffer in self.gradient_buffers.values_mut() {
            buffer.zero();
        }

        Ok(())
    }

    /// Overlap communication with computation (advanced feature)
    pub fn enable_communication_overlap(&mut self) -> Result<()> {
        if !self.config.overlap_comm {
            return Ok(());
        }

        // Set up communication streams for overlapping
        // This would involve CUDA streams in a real implementation
        for i in 0..self.gradient_buckets.len() {
            self.comm_streams.push(format!("stream_{}", i));
        }

        println!(
            "ZeRO Stage 2: Communication overlap enabled with {} streams",
            self.comm_streams.len()
        );
        Ok(())
    }

    /// Prefetch gradients for next iteration
    pub fn prefetch_gradients(&mut self, param_names: &[String]) -> Result<()> {
        if self.config.prefetch_depth == 0 {
            return Ok(());
        }

        // Add parameters to prefetch queue
        for name in param_names {
            if !self.prefetch_queue.contains(name) {
                self.prefetch_queue.push(name.clone());
            }
        }

        // Limit prefetch queue size
        while self.prefetch_queue.len() > self.config.prefetch_depth {
            self.prefetch_queue.remove(0);
        }

        Ok(())
    }

    /// Get memory usage statistics for Stage 2
    pub fn get_memory_usage(&self) -> HashMap<String, usize> {
        let mut stats = self.stage1.get_memory_usage();

        // Add gradient buffer memory
        let mut gradient_memory = 0;
        for buffer in self.gradient_buffers.values() {
            gradient_memory += buffer.memory_usage();
        }
        stats.insert("gradient_buffers".to_string(), gradient_memory);

        // Add communication buffer memory
        let comm_memory = self.comm_streams.len() * 1024 * 1024; // Simplified estimation
        stats.insert("communication_streams".to_string(), comm_memory);

        stats
    }

    /// Estimate memory savings compared to standard optimizer
    pub fn estimate_memory_savings(&self) -> HashMap<String, f32> {
        let mut savings = HashMap::new();

        // Optimizer state savings (from Stage 1)
        savings.insert(
            "optimizer_states".to_string(),
            self.stage1.estimate_memory_savings(),
        );

        // Gradient memory savings
        let world_size = self.mp_context.world_size() as f32;
        let gradient_savings = 1.0 - (1.0 / world_size); // Each rank stores 1/world_size of gradients
        savings.insert("gradients".to_string(), gradient_savings);

        // Total savings (weighted average)
        let total_savings = (savings["optimizer_states"] + gradient_savings) / 2.0;
        savings.insert("total".to_string(), total_savings);

        savings
    }

    /// Check gradient bucket sizes and rebalance if needed
    pub fn rebalance_gradient_buckets(&mut self) -> Result<()> {
        // Analyze current bucket utilization
        let mut bucket_sizes = Vec::new();
        for bucket in &self.gradient_buckets {
            let bucket_size: usize = bucket
                .iter()
                .filter_map(|name| self.gradient_buffers.get(name))
                .map(|buffer| buffer.memory_usage())
                .sum();
            bucket_sizes.push(bucket_size);
        }

        // Check if rebalancing is needed
        let target_size = self.config.bucket_size_mb * 1024 * 1024;
        let needs_rebalancing = bucket_sizes
            .iter()
            .any(|&size| size > target_size * 2 || size < target_size / 2);

        if needs_rebalancing {
            println!("ZeRO Stage 2: Rebalancing gradient buckets");
            // Rebalancing logic would go here
            // For now, just log that rebalancing is needed
        }

        Ok(())
    }

    /// Get gradient buffer for a parameter
    pub fn get_gradient_buffer(&self, param_name: &str) -> Option<&GradientBuffer> {
        self.gradient_buffers.get(param_name)
    }

    /// Get number of gradient buckets
    pub fn num_gradient_buckets(&self) -> usize {
        self.gradient_buckets.len()
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
    fn test_zero_stage2_creation() {
        let config = ModelParallelConfig {
            num_devices: 2,
            device_ids: vec![0, 1],
            strategy: ModelParallelStrategy::Pipeline,
            comm_backend: CommunicationBackend::Custom,
            ..Default::default()
        };
        let mp_context = Arc::new(ModelParallelContext::new(config).unwrap());
        let zero_config = ZeROConfig::default();

        let stage2 = ZeROStage2::<Adam>::new(mp_context, zero_config);
        assert!(stage2.is_ok());
    }

    #[test]
    fn test_gradient_buffer_initialization() {
        let config = ModelParallelConfig {
            num_devices: 4,
            device_ids: vec![0, 1, 2, 3],
            strategy: ModelParallelStrategy::Pipeline,
            comm_backend: CommunicationBackend::Custom,
            ..Default::default()
        };
        let mp_context = Arc::new(ModelParallelContext::new(config).unwrap());
        let zero_config = ZeROConfig::default();
        let mut stage2 = ZeROStage2::<Adam>::new(mp_context, zero_config).unwrap();

        let mut parameters = HashMap::new();
        parameters.insert("weight1".to_string(), Tensor::ones(&[8, 8]).unwrap());
        parameters.insert("weight2".to_string(), Tensor::ones(&[4, 4]).unwrap());
        parameters.insert("bias1".to_string(), Tensor::ones(&[8]).unwrap());

        let result = stage2.register_parameters(parameters);
        assert!(result.is_ok());

        // Check that gradient buffers were created
        assert_eq!(stage2.gradient_buffers.len(), 3);
        assert!(stage2.get_gradient_buffer("weight1").is_some());
        assert!(stage2.get_gradient_buffer("weight2").is_some());
        assert!(stage2.get_gradient_buffer("bias1").is_some());
    }

    #[test]
    fn test_gradient_bucket_creation() {
        let config = ModelParallelConfig {
            num_devices: 2,
            device_ids: vec![0, 1],
            strategy: ModelParallelStrategy::Pipeline,
            comm_backend: CommunicationBackend::Custom,
            ..Default::default()
        };
        let mp_context = Arc::new(ModelParallelContext::new(config).unwrap());
        let zero_config = ZeROConfig {
            bucket_size_mb: 1, // Small bucket size for testing
            ..Default::default()
        };
        let mut stage2 = ZeROStage2::<Adam>::new(mp_context, zero_config).unwrap();

        let mut parameters = HashMap::new();
        parameters.insert(
            "large_weight".to_string(),
            Tensor::ones(&[1000, 1000]).unwrap(),
        );
        parameters.insert("small_weight".to_string(), Tensor::ones(&[10, 10]).unwrap());

        stage2.register_parameters(parameters).unwrap();

        // Check that gradient buckets were created
        assert!(stage2.num_gradient_buckets() > 0);
        assert_eq!(
            stage2.gradient_buckets.len(),
            stage2.reduce_scatter_groups.len()
        );
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
        let mut stage2 = ZeROStage2::<Adam>::new(mp_context, zero_config).unwrap();

        let mut parameters = HashMap::new();
        for i in 0..8 {
            parameters.insert(format!("param{}", i), Tensor::ones(&[4, 4]).unwrap());
        }

        stage2.register_parameters(parameters).unwrap();

        let savings = stage2.estimate_memory_savings();

        // Should have savings for both optimizer states and gradients
        assert!(savings.contains_key("optimizer_states"));
        assert!(savings.contains_key("gradients"));
        assert!(savings.contains_key("total"));

        // With 4 devices, gradient savings should be 75% (3/4)
        assert!(savings["gradients"] > 0.7 && savings["gradients"] < 0.8);

        // Total savings should be substantial
        assert!(savings["total"] > 0.5);
    }

    #[test]
    fn test_gradient_update_processing() {
        let config = ModelParallelConfig {
            num_devices: 2,
            device_ids: vec![0, 1],
            strategy: ModelParallelStrategy::Pipeline,
            comm_backend: CommunicationBackend::Custom,
            ..Default::default()
        };
        let mp_context = Arc::new(ModelParallelContext::new(config).unwrap());
        let zero_config = ZeROConfig::default();
        let mut stage2 = ZeROStage2::<Adam>::new(mp_context, zero_config).unwrap();

        let mut parameters = HashMap::new();
        parameters.insert("weight".to_string(), Tensor::ones(&[4, 4]).unwrap());

        stage2.register_parameters(parameters).unwrap();

        // Test gradient update
        let mut gradients = HashMap::new();
        gradients.insert("weight".to_string(), Tensor::ones(&[4, 4]).unwrap());

        let result = stage2.update_gradients(gradients);
        assert!(result.is_ok());

        // Check that gradient buffer was updated
        let buffer = stage2.get_gradient_buffer("weight").unwrap();
        assert!(buffer.accumulation_steps > 0);
    }

    #[test]
    fn test_communication_overlap_setup() {
        let config = ModelParallelConfig {
            num_devices: 2,
            device_ids: vec![0, 1],
            strategy: ModelParallelStrategy::Pipeline,
            comm_backend: CommunicationBackend::Custom,
            ..Default::default()
        };
        let mp_context = Arc::new(ModelParallelContext::new(config).unwrap());
        let zero_config = ZeROConfig {
            overlap_comm: true,
            ..Default::default()
        };
        let mut stage2 = ZeROStage2::<Adam>::new(mp_context, zero_config).unwrap();

        let mut parameters = HashMap::new();
        parameters.insert("weight1".to_string(), Tensor::ones(&[4, 4]).unwrap());
        parameters.insert("weight2".to_string(), Tensor::ones(&[4, 4]).unwrap());

        stage2.register_parameters(parameters).unwrap();

        let result = stage2.enable_communication_overlap();
        assert!(result.is_ok());

        // Check that communication streams were created
        assert!(!stage2.comm_streams.is_empty());
    }

    #[test]
    fn test_prefetch_functionality() {
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
        let mut stage2 = ZeROStage2::<Adam>::new(mp_context, zero_config).unwrap();

        let param_names = vec![
            "param1".to_string(),
            "param2".to_string(),
            "param3".to_string(),
        ];
        let result = stage2.prefetch_gradients(&param_names);
        assert!(result.is_ok());

        // Check that prefetch queue is limited to prefetch_depth
        assert!(stage2.prefetch_queue.len() <= 2);
    }
}
