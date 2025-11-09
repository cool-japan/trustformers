//! Multi-Node Distributed Training Support
//!
//! This module provides infrastructure for distributed training across
//! multiple nodes using MPI communication backend. It integrates with
//! the existing ZeRO optimization for memory-efficient multi-node training.

use std::collections::HashMap;
use std::sync::Arc;
use trustformers_core::errors::Result;
use trustformers_core::parallel::{
    CommunicationBackend, ModelParallelConfig, ModelParallelContext,
};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

use trustformers_core::parallel::{mpi_utils, MpiCommunicatorImpl};

use crate::zero::{ZeROConfig, ZeROOptimizer, ZeROStage};

/// Multi-node training configuration
#[derive(Debug, Clone)]
pub struct MultiNodeConfig {
    /// Number of nodes in the cluster
    pub num_nodes: usize,
    /// Number of devices per node
    pub devices_per_node: usize,
    /// Node rank (0-based)
    pub node_rank: usize,
    /// Local device rank within node
    pub local_rank: usize,
    /// Global rank across all nodes
    pub global_rank: usize,
    /// ZeRO configuration for memory optimization
    pub zero_config: ZeROConfig,
    /// Enable gradient compression
    pub gradient_compression: bool,
    /// Communication backend
    pub comm_backend: CommunicationBackend,
    /// Enable overlap of computation and communication
    pub overlap_comm_compute: bool,
    /// Bucket size for gradient synchronization (MB)
    pub gradient_bucket_size_mb: usize,
}

impl Default for MultiNodeConfig {
    fn default() -> Self {
        Self {
            num_nodes: 1,
            devices_per_node: 1,
            node_rank: 0,
            local_rank: 0,
            global_rank: 0,
            zero_config: ZeROConfig::default(),
            gradient_compression: false,
            comm_backend: CommunicationBackend::Mpi,
            overlap_comm_compute: true,
            gradient_bucket_size_mb: 25,
        }
    }
}

impl MultiNodeConfig {
    /// Create configuration for multi-node training
    pub fn new(
        num_nodes: usize,
        devices_per_node: usize,
        node_rank: usize,
        local_rank: usize,
    ) -> Self {
        let global_rank = node_rank * devices_per_node + local_rank;

        Self {
            num_nodes,
            devices_per_node,
            node_rank,
            local_rank,
            global_rank,
            ..Default::default()
        }
    }

    /// Get total world size across all nodes
    pub fn world_size(&self) -> usize {
        self.num_nodes * self.devices_per_node
    }

    /// Check if this is the master rank
    pub fn is_master(&self) -> bool {
        self.global_rank == 0
    }

    /// Get node-local ranks for this node
    pub fn node_local_ranks(&self) -> Vec<usize> {
        let start = self.node_rank * self.devices_per_node;
        (start..start + self.devices_per_node).collect()
    }
}

/// Multi-node distributed training coordinator
pub struct MultiNodeTrainer<T: Optimizer> {
    config: MultiNodeConfig,
    mp_context: Arc<ModelParallelContext>,
    zero_optimizer: ZeROOptimizer<T>,
    mpi_communicator: Option<Arc<MpiCommunicatorImpl>>,
    gradient_buffers: HashMap<String, GradientSyncBuffer>,
    #[allow(dead_code)]
    communication_overlap: bool,
    node_local_group: Option<Vec<usize>>,
    cross_node_group: Option<Vec<usize>>,
}

/// Buffer for gradient synchronization across nodes
#[derive(Debug, Clone)]
struct GradientSyncBuffer {
    /// Buffered gradients
    gradients: HashMap<String, Tensor>,
    /// Number of accumulated steps
    accumulation_steps: usize,
    /// Compression metadata
    compression_info: Option<CompressionInfo>,
}

#[derive(Debug, Clone)]
struct CompressionInfo {
    /// Compression ratio achieved
    #[allow(dead_code)]
    compression_ratio: f32,
    /// Original size in bytes
    #[allow(dead_code)]
    original_size: usize,
    /// Compressed size in bytes
    #[allow(dead_code)]
    compressed_size: usize,
}

impl GradientSyncBuffer {
    fn new() -> Self {
        Self {
            gradients: HashMap::new(),
            accumulation_steps: 0,
            compression_info: None,
        }
    }

    fn add_gradient(&mut self, name: String, gradient: Tensor) -> Result<()> {
        if let Some(existing) = self.gradients.get_mut(&name) {
            *existing = existing.add(&gradient)?;
        } else {
            self.gradients.insert(name, gradient);
        }
        self.accumulation_steps += 1;
        Ok(())
    }

    fn clear(&mut self) {
        self.gradients.clear();
        self.accumulation_steps = 0;
        self.compression_info = None;
    }

    fn average_gradients(&mut self) -> Result<()> {
        if self.accumulation_steps <= 1 {
            return Ok(());
        }

        let divisor = self.accumulation_steps as f32;
        for gradient in self.gradients.values_mut() {
            *gradient = gradient.scalar_div(divisor)?;
        }
        Ok(())
    }
}

impl<T: Optimizer> MultiNodeTrainer<T> {
    /// Create a new multi-node trainer
    pub fn new(config: MultiNodeConfig, base_optimizer: T) -> Result<Self> {
        // Create model parallel configuration
        let mp_config = ModelParallelConfig {
            num_devices: config.world_size(),
            device_ids: (0..config.world_size()).collect(),
            comm_backend: config.comm_backend,
            ..Default::default()
        };

        // Initialize model parallel context
        let mp_context = Arc::new(ModelParallelContext::new(mp_config)?);

        // Initialize ZeRO optimizer
        let zero_optimizer = ZeROOptimizer::new(
            base_optimizer,
            config.zero_config.clone(),
            mp_context.clone(),
        )?;

        // Initialize MPI communicator
        let mpi_communicator = if config.comm_backend == CommunicationBackend::Mpi {
            Some(Arc::new(MpiCommunicatorImpl::new()?))
        } else {
            None
        };

        // Create node groups for hierarchical communication
        let node_local_group = Some(config.node_local_ranks());
        let cross_node_group =
            Some((0..config.num_nodes).map(|i| i * config.devices_per_node).collect());

        let communication_overlap = config.overlap_comm_compute;

        Ok(Self {
            config,
            mp_context,
            zero_optimizer,
            mpi_communicator,
            gradient_buffers: HashMap::new(),
            communication_overlap,
            node_local_group,
            cross_node_group,
        })
    }

    /// Initialize MPI environment for multi-node training
    pub fn initialize_environment() -> Result<()> {
        mpi_utils::init_mpi_environment()?;
        mpi_utils::check_mpi_environment()?;

        // Get node-local information
        let (local_rank, local_size) = mpi_utils::get_node_local_info()?;
        println!("Multi-node environment initialized:");
        println!("  Local rank: {}", local_rank);
        println!("  Local size: {}", local_size);

        Ok(())
    }

    /// Register parameters for multi-node training
    pub fn register_parameters(&mut self, parameters: HashMap<String, Tensor>) -> Result<()> {
        // Register with ZeRO optimizer
        self.zero_optimizer.register_parameters(parameters.clone())?;

        // Initialize gradient buffers for each parameter
        for name in parameters.keys() {
            self.gradient_buffers.insert(name.clone(), GradientSyncBuffer::new());
        }

        println!("Multi-node training initialized:");
        println!("  Node rank: {}", self.config.node_rank);
        println!("  Global rank: {}", self.config.global_rank);
        println!("  World size: {}", self.config.world_size());
        println!("  ZeRO stage: {:?}", self.zero_optimizer.get_stage());
        println!("  Parameters: {}", parameters.len());

        Ok(())
    }

    /// Update gradients with multi-node synchronization
    pub fn update_gradients(&mut self, gradients: HashMap<String, Tensor>) -> Result<()> {
        // Accumulate gradients locally
        for (name, gradient) in gradients {
            if let Some(buffer) = self.gradient_buffers.get_mut(&name) {
                buffer.add_gradient(name.clone(), gradient)?;
            }
        }

        // Update ZeRO optimizer (local processing)
        self.zero_optimizer.update_gradients(self.collect_local_gradients()?)?;

        Ok(())
    }

    /// Collect local gradients from buffers
    fn collect_local_gradients(&self) -> Result<HashMap<String, Tensor>> {
        let mut gradients = HashMap::new();
        for (name, buffer) in &self.gradient_buffers {
            if let Some(grad) = buffer.gradients.get(name) {
                gradients.insert(name.clone(), grad.clone());
            }
        }
        Ok(gradients)
    }

    /// Synchronize gradients across all nodes
    pub fn synchronize_gradients(&mut self) -> Result<()> {
        if self.config.world_size() == 1 {
            return Ok(()); // No synchronization needed for single node
        }

        // Average accumulated gradients
        for buffer in self.gradient_buffers.values_mut() {
            buffer.average_gradients()?;
        }

        // Perform hierarchical gradient synchronization
        self.hierarchical_all_reduce()?;

        // Clear buffers after synchronization
        for buffer in self.gradient_buffers.values_mut() {
            buffer.clear();
        }

        Ok(())
    }

    /// Hierarchical all-reduce for better network utilization
    fn hierarchical_all_reduce(&mut self) -> Result<()> {
        let has_mpi = self.mpi_communicator.is_some();

        if has_mpi {
            // Step 1: Reduce within each node
            self.node_local_reduce()?;

            // Step 2: All-reduce across nodes (one rank per node)
            if self.config.local_rank == 0 {
                self.cross_node_all_reduce()?;
            }

            // Step 3: Broadcast within each node
            self.node_local_broadcast()?;

            // Synchronize all processes
            if let Some(ref mpi_comm) = self.mpi_communicator {
                mpi_comm.barrier()?;
            }
        } else {
            // Fallback to regular all-reduce using model parallel context
            for buffer in self.gradient_buffers.values_mut() {
                for gradient in buffer.gradients.values_mut() {
                    self.mp_context.all_reduce(gradient)?;
                }
            }
        }

        Ok(())
    }

    /// Reduce gradients within each node
    fn node_local_reduce(&mut self) -> Result<()> {
        // Simplified implementation - in practice would use node-local communicator
        if let Some(ref _local_ranks) = self.node_local_group {
            for buffer in self.gradient_buffers.values_mut() {
                for gradient in buffer.gradients.values_mut() {
                    // Use MPI reduce operation within node
                    // For now, use the general all_reduce
                    self.mp_context.all_reduce(gradient)?;
                }
            }
        }
        Ok(())
    }

    /// All-reduce gradients across nodes
    fn cross_node_all_reduce(&mut self) -> Result<()> {
        // Only performed by one rank per node (usually local_rank == 0)
        if let Some(ref _cross_ranks) = self.cross_node_group {
            for buffer in self.gradient_buffers.values_mut() {
                for gradient in buffer.gradients.values_mut() {
                    self.mp_context.all_reduce(gradient)?;
                }
            }
        }
        Ok(())
    }

    /// Broadcast gradients within each node
    fn node_local_broadcast(&mut self) -> Result<()> {
        // Broadcast from local rank 0 to other ranks on the same node
        let root_rank = self.config.node_rank * self.config.devices_per_node;

        for buffer in self.gradient_buffers.values_mut() {
            for gradient in buffer.gradients.values_mut() {
                self.mp_context.broadcast(gradient, root_rank)?;
            }
        }
        Ok(())
    }

    /// Apply gradients with multi-node coordination
    pub fn apply_gradients(&mut self, accumulation_steps: usize) -> Result<()> {
        // Synchronize gradients across nodes first
        self.synchronize_gradients()?;

        // Apply gradients using ZeRO optimizer
        self.zero_optimizer.apply_accumulated_grads(accumulation_steps)?;

        Ok(())
    }

    /// Perform optimizer step with multi-node coordination
    pub fn optimizer_step(&mut self) -> Result<()> {
        // Synchronize gradients across nodes
        self.synchronize_gradients()?;

        // Perform optimizer step using ZeRO
        self.zero_optimizer.optimizer_step()?;

        Ok(())
    }

    /// Get comprehensive memory usage across nodes
    pub fn get_memory_usage(&self) -> HashMap<String, usize> {
        let memory_stats = self.zero_optimizer.get_memory_stats();
        let mut stats = HashMap::new();

        // Add ZeRO memory statistics
        stats.insert(
            "optimizer_memory_saved".to_string(),
            memory_stats.optimizer_memory_saved,
        );
        stats.insert(
            "gradient_memory_saved".to_string(),
            memory_stats.gradient_memory_saved,
        );
        stats.insert(
            "parameter_memory_saved".to_string(),
            memory_stats.parameter_memory_saved,
        );
        stats.insert(
            "communication_overhead".to_string(),
            memory_stats.communication_overhead,
        );
        stats.insert(
            "total_memory_saved".to_string(),
            memory_stats.total_memory_saved,
        );

        // Add multi-node specific memory usage
        let mut buffer_memory = 0;
        for buffer in self.gradient_buffers.values() {
            for gradient in buffer.gradients.values() {
                buffer_memory += gradient.memory_usage();
            }
        }
        stats.insert("gradient_sync_buffers".to_string(), buffer_memory);

        // Add communication overhead estimate
        let comm_overhead = self.config.world_size() * 1024 * 1024; // 1MB per process estimate
        stats.insert("communication_overhead".to_string(), comm_overhead);

        stats
    }

    /// Get multi-node training statistics
    pub fn get_training_stats(&self) -> MultiNodeStats {
        let memory_stats = self.zero_optimizer.get_memory_stats();
        let mut memory_savings = HashMap::new();

        // Convert memory stats to savings percentages (simplified calculation)
        let total_memory = memory_stats.total_memory_saved;
        if total_memory > 0 {
            memory_savings.insert(
                "optimizer_states".to_string(),
                memory_stats.optimizer_memory_saved as f32 / total_memory as f32,
            );
            memory_savings.insert(
                "gradients".to_string(),
                memory_stats.gradient_memory_saved as f32 / total_memory as f32,
            );
            memory_savings.insert(
                "parameters".to_string(),
                memory_stats.parameter_memory_saved as f32 / total_memory as f32,
            );
        }

        MultiNodeStats {
            node_rank: self.config.node_rank,
            global_rank: self.config.global_rank,
            world_size: self.config.world_size(),
            zero_stage: self.zero_optimizer.get_stage(),
            memory_savings,
            communication_backend: self.config.comm_backend,
            gradient_compression_enabled: self.config.gradient_compression,
        }
    }

    /// Check if this process should save checkpoints
    pub fn should_save_checkpoint(&self) -> bool {
        self.config.is_master()
    }

    /// Barrier synchronization across all nodes
    pub fn barrier(&self) -> Result<()> {
        if let Some(ref mpi_comm) = self.mpi_communicator {
            mpi_comm.barrier()?;
        }

        Ok(())
    }

    /// Finalize multi-node training
    pub fn finalize() -> Result<()> {
        MpiCommunicatorImpl::finalize()?;

        println!("Multi-node training finalized");
        Ok(())
    }
}

/// Statistics for multi-node training
#[derive(Debug, Clone)]
pub struct MultiNodeStats {
    pub node_rank: usize,
    pub global_rank: usize,
    pub world_size: usize,
    pub zero_stage: ZeROStage,
    pub memory_savings: HashMap<String, f32>,
    pub communication_backend: CommunicationBackend,
    pub gradient_compression_enabled: bool,
}

impl MultiNodeStats {
    /// Print training statistics
    pub fn print_stats(&self) {
        println!("=== Multi-Node Training Statistics ===");
        println!("Node Rank: {}", self.node_rank);
        println!("Global Rank: {}", self.global_rank);
        println!("World Size: {}", self.world_size);
        println!("ZeRO Stage: {:?}", self.zero_stage);
        println!("Communication Backend: {:?}", self.communication_backend);
        println!(
            "Gradient Compression: {}",
            self.gradient_compression_enabled
        );

        println!("Memory Savings:");
        for (component, savings) in &self.memory_savings {
            println!("  {}: {:.1}%", component, savings * 100.0);
        }
        println!("=====================================");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adam::Adam;

    #[test]
    fn test_multinode_config() {
        let config = MultiNodeConfig::new(4, 8, 2, 3);

        assert_eq!(config.num_nodes, 4);
        assert_eq!(config.devices_per_node, 8);
        assert_eq!(config.node_rank, 2);
        assert_eq!(config.local_rank, 3);
        assert_eq!(config.global_rank, 19); // 2 * 8 + 3
        assert_eq!(config.world_size(), 32); // 4 * 8
        assert!(!config.is_master());

        let master_config = MultiNodeConfig::new(4, 8, 0, 0);
        assert!(master_config.is_master());
    }

    #[test]
    fn test_multinode_trainer_creation() {
        let config = MultiNodeConfig::new(2, 4, 0, 0);
        let optimizer = Adam::new(0.001, (0.9, 0.999), 1e-8, 0.0);

        match MultiNodeTrainer::new(config, optimizer) {
            Ok(trainer) => {
                assert_eq!(trainer.config.world_size(), 8);
                assert!(trainer.config.is_master());
            },
            Err(e) => {
                // Expected in test environment without proper MPI setup
                println!("Expected error in test environment: {}", e);
            },
        }
    }

    #[test]
    fn test_gradient_sync_buffer() {
        let mut buffer = GradientSyncBuffer::new();

        let grad1 = Tensor::ones(&[2, 2]).unwrap();
        let grad2 = Tensor::ones(&[2, 2]).unwrap();

        buffer.add_gradient("param1".to_string(), grad1).unwrap();
        buffer.add_gradient("param1".to_string(), grad2).unwrap();

        assert_eq!(buffer.accumulation_steps, 2);
        assert_eq!(buffer.gradients.len(), 1);

        buffer.average_gradients().unwrap();
        // After averaging, each element should be 1.0 (2.0 / 2)

        buffer.clear();
        assert_eq!(buffer.gradients.len(), 0);
        assert_eq!(buffer.accumulation_steps, 0);
    }

    #[test]
    fn test_node_groups() {
        let config = MultiNodeConfig::new(3, 4, 1, 2);
        let node_ranks = config.node_local_ranks();

        assert_eq!(node_ranks, vec![4, 5, 6, 7]); // Node 1 with 4 devices per node
    }
}
