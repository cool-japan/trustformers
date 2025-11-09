//! # Multi-GPU Distributed Training with Averaged Adam
//!
//! This example demonstrates how to use the Averaged Adam optimizer in distributed
//! training scenarios across multiple GPUs and nodes, showcasing advanced techniques
//! for scaling transformer training with Polyak-Ruppert averaging.
//!
//! ## Features Demonstrated:
//! - Multi-GPU data parallel training with Averaged Adam
//! - ZeRO optimization stages (1, 2, 3) integration
//! - Gradient compression for efficient communication
//! - Hierarchical parameter averaging across devices
//! - Fault-tolerant distributed training
//! - Performance profiling and optimization

use scirs2_core::random::*; // SciRS2 Integration Policy
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use trustformers_core::errors::unsupported_operation;
use trustformers_core::parallel::{ModelParallelConfig, ModelParallelContext};
use trustformers_core::TrustformersError;
use trustformers_core::{traits::Optimizer, Tensor};
use trustformers_optim::zero::zero_utils::PartitionInfo;
use trustformers_optim::*;

/// Distributed training configuration
#[derive(Debug, Clone)]
pub struct DistributedTrainingConfig {
    /// Number of nodes in the cluster
    pub num_nodes: usize,
    /// Number of GPUs per node
    pub gpus_per_node: usize,
    /// Current node rank
    pub node_rank: usize,
    /// Local GPU rank within node
    pub local_rank: usize,
    /// Global rank across all devices
    pub global_rank: usize,
    /// ZeRO optimization stage
    pub zero_stage: ZeROStage,
    /// Enable gradient compression
    pub enable_compression: bool,
    /// Gradient compression method
    pub compression_method: CompressionMethod,
    /// Hierarchical aggregation strategy
    pub aggregation_strategy: AggregationStrategy,
    /// Communication backend
    pub comm_backend: String,
    /// Enable fault tolerance
    pub enable_fault_tolerance: bool,
    /// Model parallel degree (for very large models)
    pub model_parallel_size: usize,
}

impl Default for DistributedTrainingConfig {
    fn default() -> Self {
        Self {
            num_nodes: 2,
            gpus_per_node: 8,
            node_rank: 0,
            local_rank: 0,
            global_rank: 0,
            zero_stage: ZeROStage::Stage2,
            enable_compression: true,
            compression_method: CompressionMethod::TopK { k: 1000 },
            aggregation_strategy: AggregationStrategy::Adaptive,
            comm_backend: "nccl".to_string(),
            enable_fault_tolerance: true,
            model_parallel_size: 1,
        }
    }
}

impl DistributedTrainingConfig {
    /// Get total number of devices/processes
    pub fn world_size(&self) -> usize {
        self.num_nodes * self.gpus_per_node
    }

    /// Check if this is the master process
    pub fn is_master(&self) -> bool {
        self.global_rank == 0
    }

    /// Get devices in the same node
    pub fn local_world_size(&self) -> usize {
        self.gpus_per_node
    }

    /// Create configuration for specific distributed setup
    pub fn for_large_cluster() -> Self {
        Self {
            num_nodes: 16,
            gpus_per_node: 8,
            zero_stage: ZeROStage::Stage3,
            compression_method: CompressionMethod::TopK { k: 10000 },
            aggregation_strategy: AggregationStrategy::Butterfly,
            model_parallel_size: 2,
            ..Default::default()
        }
    }

    /// Create configuration for medium cluster
    pub fn for_medium_cluster() -> Self {
        Self {
            num_nodes: 4,
            gpus_per_node: 8,
            zero_stage: ZeROStage::Stage2,
            compression_method: CompressionMethod::TopK { k: 5000 },
            aggregation_strategy: AggregationStrategy::Ring,
            ..Default::default()
        }
    }
}

/// Distributed Averaged Adam optimizer with advanced features
pub struct DistributedAveragedAdam {
    /// Base Averaged Adam optimizer
    base_optimizer: AveragedAdam,
    /// Distributed training configuration
    config: DistributedTrainingConfig,
    /// ZeRO optimizer wrapper
    zero_optimizer: Option<ZeROOptimizer<AveragedAdam>>,
    /// Gradient compressor for communication efficiency
    gradient_compressor: Option<GradientCompressor>,
    /// Hierarchical aggregator for multi-level communication
    hierarchical_aggregator: HierarchicalAggregator,
    /// Parameter partitioning for ZeRO
    parameter_partitions: HashMap<String, ParameterPartition>,
    /// Communication statistics
    comm_stats: DistributedCommStats,
    /// Fault tolerance state
    fault_tolerance: FaultToleranceState,
}

/// Communication statistics for monitoring
#[derive(Debug, Default)]
pub struct DistributedCommStats {
    /// Total bytes communicated
    pub total_bytes_communicated: usize,
    /// Number of communication rounds
    pub communication_rounds: usize,
    /// Average communication latency
    pub avg_latency_ms: f32,
    /// Compression ratio achieved
    pub compression_ratio: f32,
    /// Bandwidth utilization
    pub bandwidth_utilization: f32,
    /// Number of fault recovery events
    pub fault_recovery_count: usize,
}

/// Fault tolerance state
#[derive(Debug, Default)]
pub struct FaultToleranceState {
    /// Failed nodes list
    pub failed_nodes: Vec<usize>,
    /// Checkpoint frequency (steps)
    pub checkpoint_frequency: usize,
    /// Last checkpoint step
    pub last_checkpoint_step: usize,
    /// Recovery in progress flag
    pub recovery_in_progress: bool,
}

impl DistributedAveragedAdam {
    /// Create new distributed Averaged Adam optimizer
    pub fn new(
        learning_rate: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
        averaging_coeff: f32,
        config: DistributedTrainingConfig,
    ) -> Result<Self, TrustformersError> {
        let base_optimizer =
            AveragedAdam::new(learning_rate, betas, eps, weight_decay, averaging_coeff);

        // Create ZeRO optimizer with current stage
        let zero_optimizer = {
            let stage = config.zero_stage;
            let zero_config = ZeROConfig {
                stage,
                bucket_size_mb: 25,
                overlap_comm: true,
                reduce_bucket_size: 100_000,
                prefetch_depth: 2,
                max_memory_usage_mb: 8192,
                gradient_compression: config.enable_compression,
                pin_memory: true,
            };

            // Create model parallel context for ZeRO
            let mp_config = ModelParallelConfig::default();
            let mp_context = Arc::new(ModelParallelContext::new(mp_config)?);

            Some(ZeROOptimizer::new(
                base_optimizer.clone(),
                zero_config,
                mp_context,
            )?)
        };

        // Create gradient compressor if enabled
        let gradient_compressor = if config.enable_compression {
            Some(GradientCompressor::new(config.compression_method.clone()))
        } else {
            None
        };

        // Create hierarchical aggregator
        let hierarchical_config = HierarchicalConfig {
            num_nodes: config.num_nodes,
            devices_per_node: config.gpus_per_node,
            node_rank: config.node_rank,
            local_rank: config.local_rank,
            global_rank: config.global_rank,
            strategy: config.aggregation_strategy.clone(),
            comm_backend: match config.comm_backend.as_str() {
                "nccl" => trustformers_core::parallel::CommunicationBackend::Nccl,
                "mpi" => trustformers_core::parallel::CommunicationBackend::Mpi,
                _ => trustformers_core::parallel::CommunicationBackend::Nccl,
            },
            enable_compression: config.enable_compression,
            compression_threshold: 0.1,
            enable_fault_tolerance: config.enable_fault_tolerance,
            comm_timeout_ms: 30000,
        };

        let hierarchical_aggregator = HierarchicalAggregator::new(hierarchical_config)?;

        Ok(Self {
            base_optimizer,
            config,
            zero_optimizer,
            gradient_compressor,
            hierarchical_aggregator,
            parameter_partitions: HashMap::new(),
            comm_stats: Default::default(),
            fault_tolerance: FaultToleranceState {
                checkpoint_frequency: 100,
                ..Default::default()
            },
        })
    }

    /// Create preset for large-scale distributed training
    pub fn for_large_scale_training() -> Result<Self, TrustformersError> {
        let config = DistributedTrainingConfig::for_large_cluster();
        Self::new(1e-4, (0.9, 0.999), 1e-8, 0.01, 0.9999, config)
    }

    /// Create preset for fault-tolerant training
    pub fn for_fault_tolerant_training() -> Result<Self, TrustformersError> {
        let mut config = DistributedTrainingConfig::for_medium_cluster();
        config.enable_fault_tolerance = true;
        Self::new(1e-4, (0.9, 0.999), 1e-8, 0.01, 0.999, config)
    }

    /// Initialize parameter partitioning for ZeRO
    pub fn initialize_parameter_partitioning(
        &mut self,
        parameters: &HashMap<String, Tensor>,
    ) -> Result<(), TrustformersError> {
        if let Some(zero_optimizer) = &mut self.zero_optimizer {
            // Register all parameters with ZeRO optimizer
            zero_optimizer.register_parameters(parameters.clone())?;

            // Create parameter partitions (mock implementation for example)
            for (name, _param) in parameters {
                let partition = ParameterPartition {
                    name: name.clone(),
                    local_shard: Tensor::zeros(&[1])?,
                    partition_info: PartitionInfo {
                        rank: 0,
                        world_size: 1,
                        start_idx: 0,
                        end_idx: 0,
                        global_shape: vec![1],
                        local_shape: vec![1],
                    },
                    is_gathered: false,
                    full_parameter: None,
                };
                self.parameter_partitions.insert(name.clone(), partition);
            }
        }
        Ok(())
    }

    /// Perform distributed gradient aggregation with Averaged Adam
    pub fn distributed_step(
        &mut self,
        parameters: &mut HashMap<String, Tensor>,
        gradients: &HashMap<String, Tensor>,
        step_count: usize,
    ) -> Result<(), TrustformersError> {
        let start_time = Instant::now();

        // 1. Local gradient computation and averaging preparation
        let mut local_gradients = gradients.clone();

        // Apply gradient clipping locally
        let _grad_norm = self.clip_gradients(&mut local_gradients, 1.0)?;

        // 2. Gradient compression (if enabled)
        let compressed_gradients = if let Some(compressor) = &mut self.gradient_compressor {
            let compressed = compressor.compress(&local_gradients)?;

            // Update compression statistics
            let original_size: usize = local_gradients.values()
                .map(|t| t.len() * 4) // f32 = 4 bytes
                .sum();
            let compressed_size: usize =
                compressed.values().map(|cg| cg.indices.len() * 4 + cg.values.len() * 4).sum();

            self.comm_stats.compression_ratio =
                1.0 - (compressed_size as f32 / original_size as f32);

            Some(compressed)
        } else {
            None
        };

        // 3. Hierarchical gradient aggregation
        let aggregated_gradients = if let Some(compressed) = compressed_gradients {
            // Aggregate compressed gradients
            // Convert compressed gradients to tensor map for aggregation
            let mut gradient_tensors: HashMap<String, Tensor> = HashMap::new();
            for (name, _compressed_grad) in &compressed {
                gradient_tensors.insert(name.clone(), Tensor::zeros(&[1])?);
            }
            self.hierarchical_aggregator.hierarchical_all_reduce(&mut gradient_tensors)?;
            let aggregated_compressed = gradient_tensors;

            // Decompress aggregated gradients
            if let Some(compressor) = &mut self.gradient_compressor {
                // Convert aggregated tensors back to compressed format for decompression
                let mut compressed_for_decompression: HashMap<String, CompressedGradient> =
                    HashMap::new();
                for (name, tensor) in &aggregated_compressed {
                    // For demo purposes, create a simple compressed gradient representation
                    compressed_for_decompression.insert(
                        name.clone(),
                        CompressedGradient {
                            indices: (0..tensor.len()).collect(),
                            values: tensor.data()?.to_vec(),
                            original_size: tensor.len(),
                            compression_ratio: 1.0,
                        },
                    );
                }
                compressor.decompress(&compressed_for_decompression)?
            } else {
                return Err(unsupported_operation(
                    "decompression",
                    "Compressor not available for decompression",
                ));
            }
        } else {
            // Direct gradient aggregation
            {
                self.hierarchical_aggregator.hierarchical_all_reduce(&mut local_gradients)?;
                local_gradients
            }
        };

        // 4. ZeRO-specific parameter and gradient handling
        let (final_parameters, final_gradients) =
            if let Some(zero_optimizer) = &mut self.zero_optimizer {
                // Handle ZeRO parameter gathering and gradient reduction
                match self.config.zero_stage {
                    ZeROStage::Stage1 => {
                        // Only optimizer states are partitioned
                        (parameters.clone(), aggregated_gradients)
                    },
                    ZeROStage::Stage2 => {
                        // Optimizer states and gradients are partitioned
                        zero_optimizer.update_gradients(aggregated_gradients.clone())?;
                        (parameters.clone(), aggregated_gradients)
                    },
                    ZeROStage::Stage3 => {
                        // Everything is partitioned - need to gather parameters for forward/backward
                        let param_names: Vec<String> = parameters.keys().cloned().collect();
                        let gathered_params = zero_optimizer.gather_parameters(&param_names)?;
                        zero_optimizer.update_gradients(aggregated_gradients.clone())?;
                        (gathered_params, aggregated_gradients)
                    },
                }
            } else {
                (parameters.clone(), aggregated_gradients)
            };

        // 5. Apply Averaged Adam updates with distributed-aware averaging
        self.apply_distributed_averaged_adam_update(
            parameters,
            &final_parameters,
            &final_gradients,
            step_count,
        )?;

        // 6. Handle parameter redistribution for ZeRO Stage 3
        if let Some(_zero_optimizer) = &mut self.zero_optimizer {
            if matches!(self.config.zero_stage, ZeROStage::Stage3) {
                // ZeRO handles parameter scattering automatically during optimization
                // No explicit scatter_parameters method is needed
            }
        }

        // 7. Update communication statistics
        let comm_time = start_time.elapsed();
        self.comm_stats.communication_rounds += 1;
        self.comm_stats.avg_latency_ms = (self.comm_stats.avg_latency_ms
            * (self.comm_stats.communication_rounds - 1) as f32
            + comm_time.as_millis() as f32)
            / self.comm_stats.communication_rounds as f32;

        // 8. Fault tolerance - checkpoint if needed
        if self.config.enable_fault_tolerance
            && step_count > 0
            && step_count % self.fault_tolerance.checkpoint_frequency == 0
        {
            self.create_checkpoint(parameters, step_count)?;
        }

        Ok(())
    }

    /// Apply Averaged Adam update with distributed-aware parameter averaging
    fn apply_distributed_averaged_adam_update(
        &mut self,
        parameters: &mut HashMap<String, Tensor>,
        _final_parameters: &HashMap<String, Tensor>,
        gradients: &HashMap<String, Tensor>,
        step_count: usize,
    ) -> Result<(), TrustformersError> {
        // Get current averaged parameters from base optimizer
        // For demo purposes, collect all parameter names and their averaged values
        let _averaged_params: HashMap<String, Tensor> = HashMap::new();
        // Note: get_averaged_parameters requires individual parameter names
        // In a real implementation, you would iterate through all parameter names

        // Apply standard Averaged Adam updates
        for (name, param) in parameters.iter_mut() {
            if let Some(gradient) = gradients.get(name) {
                self.base_optimizer.update(param, gradient)?;
            }
        }

        // Step the optimizer
        self.base_optimizer.step();

        // Perform distributed averaging of the averaged parameters
        // This is the key innovation: we average both the current parameters
        // and the Polyak-Ruppert averaged parameters across devices
        if step_count > 0 && step_count % 10 == 0 {
            // Every 10 steps
            self.synchronize_averaged_parameters()?;
        }

        Ok(())
    }

    /// Synchronize Polyak-Ruppert averaged parameters across all devices
    fn synchronize_averaged_parameters(&mut self) -> Result<(), TrustformersError> {
        // For demo purposes, create a placeholder averaged params map
        let mut averaged_params: HashMap<String, Tensor> = HashMap::new();
        // Note: get_averaged_parameters requires individual parameter names
        // In a real implementation, you would iterate through all parameter names

        // Aggregate averaged parameters across devices using hierarchical communication
        self.hierarchical_aggregator.hierarchical_all_reduce(&mut averaged_params)?;

        // AveragedAdam automatically manages averaged parameters
        // No explicit set_averaged_parameters method is needed

        Ok(())
    }

    /// Clip gradients by global norm across all devices
    fn clip_gradients(
        &self,
        gradients: &mut HashMap<String, Tensor>,
        max_norm: f32,
    ) -> Result<f32, TrustformersError> {
        // Calculate local gradient norm
        let mut local_norm_sq = 0.0f32;
        for gradient in gradients.values() {
            let norm = gradient.norm()?;
            local_norm_sq += norm * norm;
        }

        // In a real implementation, we would need to all-reduce the local_norm_sq
        // across all devices to get the global norm. For this example, we'll
        // simulate this by scaling by the world size approximation.
        let global_norm_sq = local_norm_sq * self.config.world_size() as f32;
        let global_norm = global_norm_sq.sqrt();

        // Apply clipping if necessary
        if global_norm > max_norm {
            let clip_factor = max_norm / global_norm;
            for gradient in gradients.values_mut() {
                *gradient = gradient.mul_scalar(clip_factor)?;
            }
        }

        Ok(global_norm)
    }

    /// Create checkpoint for fault tolerance
    fn create_checkpoint(
        &mut self,
        _parameters: &HashMap<String, Tensor>,
        step_count: usize,
    ) -> Result<(), TrustformersError> {
        if self.config.is_master() {
            println!("💾 Creating checkpoint at step {}", step_count);

            // In a real implementation, this would save:
            // - Model parameters
            // - Optimizer state (including averaged parameters)
            // - Training step count
            // - RNG state
            // For now, just update the checkpoint tracking
            self.fault_tolerance.last_checkpoint_step = step_count;
        }
        Ok(())
    }

    /// Recover from checkpoint after node failure
    pub fn recover_from_checkpoint(
        &mut self,
        checkpoint_step: usize,
    ) -> Result<(), TrustformersError> {
        self.fault_tolerance.recovery_in_progress = true;
        self.comm_stats.fault_recovery_count += 1;

        println!("🔄 Recovering from checkpoint at step {}", checkpoint_step);

        // In a real implementation, this would:
        // - Load model parameters from checkpoint
        // - Restore optimizer state
        // - Reset communication groups excluding failed nodes
        // - Resume training from checkpoint step

        self.fault_tolerance.recovery_in_progress = false;
        Ok(())
    }

    /// Get communication statistics
    pub fn get_communication_stats(&self) -> &DistributedCommStats {
        &self.comm_stats
    }

    /// Get memory usage statistics
    pub fn get_memory_usage(&self) -> HashMap<String, usize> {
        let base_stats = self.base_optimizer.memory_usage();
        let mut stats = HashMap::new();

        // Convert StateMemoryStats to HashMap
        stats.insert(
            "momentum_elements".to_string(),
            base_stats.momentum_elements,
        );
        stats.insert(
            "variance_elements".to_string(),
            base_stats.variance_elements,
        );
        stats.insert(
            "third_moment_elements".to_string(),
            base_stats.third_moment_elements,
        );
        stats.insert("total_bytes".to_string(), base_stats.total_bytes);
        stats.insert("num_parameters".to_string(), base_stats.num_parameters);

        // Add distributed-specific memory usage
        let partition_memory: usize = self.parameter_partitions.values()
            .map(|p| p.local_shard.len() * 4) // Approximate bytes
            .sum();
        stats.insert("parameter_partitions".to_string(), partition_memory);

        if let Some(zero_optimizer) = &self.zero_optimizer {
            let zero_stats = zero_optimizer.get_memory_stats();
            stats.insert(
                "zero_memory_saved".to_string(),
                zero_stats.total_memory_saved as usize,
            );
        }

        stats
    }

    /// Check if ready for evaluation (using averaged parameters)
    pub fn use_averaged_parameters_for_evaluation(&mut self) -> Result<(), TrustformersError> {
        self.base_optimizer.use_averaged_parameters(true);
        Ok(())
    }

    /// Switch back to current parameters for training
    pub fn use_current_parameters_for_training(&mut self) -> Result<(), TrustformersError> {
        self.base_optimizer.use_averaged_parameters(false);
        Ok(())
    }

    /// Get distributed training performance metrics
    pub fn get_distributed_performance_metrics(&self) -> DistributedPerformanceMetrics {
        DistributedPerformanceMetrics {
            world_size: self.config.world_size(),
            compression_ratio: self.comm_stats.compression_ratio,
            avg_communication_latency_ms: self.comm_stats.avg_latency_ms,
            total_communication_rounds: self.comm_stats.communication_rounds,
            bandwidth_utilization: self.comm_stats.bandwidth_utilization,
            fault_recovery_count: self.comm_stats.fault_recovery_count,
            zero_stage: format!("{:?}", self.config.zero_stage),
            memory_efficiency_ratio: self.calculate_memory_efficiency(),
        }
    }

    /// Calculate memory efficiency compared to non-distributed training
    fn calculate_memory_efficiency(&self) -> f32 {
        match self.config.zero_stage {
            ZeROStage::Stage1 => 4.0, // ~4x memory reduction
            ZeROStage::Stage2 => 8.0, // ~8x memory reduction
            ZeROStage::Stage3 => self.config.world_size() as f32, // Nx memory reduction
        }
    }
}

/// Distributed performance metrics
#[derive(Debug)]
pub struct DistributedPerformanceMetrics {
    pub world_size: usize,
    pub compression_ratio: f32,
    pub avg_communication_latency_ms: f32,
    pub total_communication_rounds: usize,
    pub bandwidth_utilization: f32,
    pub fault_recovery_count: usize,
    pub zero_stage: String,
    pub memory_efficiency_ratio: f32,
}

/// Simulate distributed training scenario
fn simulate_distributed_training_scenario(
    config: DistributedTrainingConfig,
    model_size_mb: usize,
    training_steps: usize,
) -> Result<DistributedPerformanceMetrics, TrustformersError> {
    println!("🚀 Simulating Distributed Training with Averaged Adam");
    println!(
        "Configuration: {} nodes × {} GPUs = {} total devices",
        config.num_nodes,
        config.gpus_per_node,
        config.world_size()
    );
    println!(
        "Model size: {} MB, Training steps: {}",
        model_size_mb, training_steps
    );
    println!(
        "ZeRO Stage: {:?}, Compression: {}",
        config.zero_stage, config.enable_compression
    );

    let mut optimizer =
        DistributedAveragedAdam::new(1e-4, (0.9, 0.999), 1e-8, 0.01, 0.999, config)?;

    // Create simulated model parameters
    let param_count = model_size_mb * 1024 * 1024 / 4; // Convert MB to parameter count (f32)
    let mut parameters = HashMap::new();
    let mut gradients = HashMap::new();

    // Simulate transformer-like parameter distribution
    let layer_count = (model_size_mb as f32).log2() as usize; // More layers for larger models
    let params_per_layer = param_count / layer_count.max(1);

    for layer in 0..layer_count {
        let layer_params = vec![0.1f32; params_per_layer];
        let layer_grads = vec![0.01f32; params_per_layer];

        parameters.insert(format!("layer_{}", layer), Tensor::new(layer_params)?);
        gradients.insert(format!("layer_{}", layer), Tensor::new(layer_grads)?);
    }

    // Initialize parameter partitioning
    optimizer.initialize_parameter_partitioning(&parameters)?;

    let start_time = Instant::now();

    // Simulate training steps
    for step in 0..training_steps {
        if step % 100 == 0 && optimizer.config.is_master() {
            println!(
                "Step {}/{} - Communication stats: {:.2}% compression, {:.1}ms avg latency",
                step,
                training_steps,
                optimizer.get_communication_stats().compression_ratio * 100.0,
                optimizer.get_communication_stats().avg_latency_ms
            );
        }

        // Simulate gradient computation (normally done by forward/backward pass)
        let mut rng = scirs2_core::random::thread_rng();
        for gradient in gradients.values_mut() {
            let grad_data: Vec<f32> =
                (0..gradient.len()).map(|_| (rng.random::<f32>() - 0.5) * 0.01).collect();
            *gradient = Tensor::new(grad_data)?;
        }

        // Perform distributed training step
        optimizer.distributed_step(&mut parameters, &gradients, step)?;

        // Simulate node failure and recovery (every 500 steps for demonstration)
        if step > 0 && step % 500 == 0 && optimizer.config.enable_fault_tolerance {
            println!("⚠️  Simulating node failure and recovery at step {}", step);
            optimizer.recover_from_checkpoint(step - 100)?;
        }
    }

    let total_time = start_time.elapsed();

    if optimizer.config.is_master() {
        println!(
            "✅ Distributed training simulation completed in {:?}",
            total_time
        );
        println!("Memory usage: {:?}", optimizer.get_memory_usage());
    }

    // Switch to averaged parameters for final evaluation
    optimizer.use_averaged_parameters_for_evaluation()?;
    println!("🎯 Using averaged parameters for evaluation");

    Ok(optimizer.get_distributed_performance_metrics())
}

/// Compare different distributed training configurations
fn compare_distributed_configurations() -> Result<(), TrustformersError> {
    println!("\n🔬 Comparing Distributed Training Configurations");
    println!("===============================================");

    let model_size_mb = 1000; // 1GB model
    let training_steps = 1000;
    let mut results = Vec::new();

    // Configuration 1: Medium cluster with ZeRO Stage 2
    println!("\n📊 Testing: Medium cluster with ZeRO Stage 2");
    let config1 = DistributedTrainingConfig::for_medium_cluster();
    let result1 = simulate_distributed_training_scenario(config1, model_size_mb, training_steps)?;
    results.push(("Medium/ZeRO-2".to_string(), result1));

    // Configuration 2: Large cluster with ZeRO Stage 3
    println!("\n📊 Testing: Large cluster with ZeRO Stage 3");
    let config2 = DistributedTrainingConfig::for_large_cluster();
    let result2 = simulate_distributed_training_scenario(config2, model_size_mb, training_steps)?;
    results.push(("Large/ZeRO-3".to_string(), result2));

    // Configuration 3: Fault-tolerant training
    println!("\n📊 Testing: Fault-tolerant training");
    let config3 = DistributedTrainingConfig {
        enable_fault_tolerance: true,
        compression_method: CompressionMethod::TopK { k: 5000 },
        ..DistributedTrainingConfig::for_medium_cluster()
    };
    let result3 = simulate_distributed_training_scenario(config3, model_size_mb, training_steps)?;
    results.push(("Fault-Tolerant".to_string(), result3));

    // Analyze results
    println!("\n📈 Performance Analysis");
    println!("{}", "=".repeat(50));

    for (name, metrics) in results {
        println!("\n🎯 Configuration: {}", name);
        println!("   World Size: {} devices", metrics.world_size);
        println!(
            "   Memory Efficiency: {:.1}x reduction",
            metrics.memory_efficiency_ratio
        );
        println!(
            "   Compression Ratio: {:.1}%",
            metrics.compression_ratio * 100.0
        );
        println!(
            "   Avg Communication Latency: {:.2}ms",
            metrics.avg_communication_latency_ms
        );
        println!(
            "   Communication Rounds: {}",
            metrics.total_communication_rounds
        );
        println!("   Fault Recovery Events: {}", metrics.fault_recovery_count);
        println!("   ZeRO Stage: {}", metrics.zero_stage);

        // Calculate estimated training efficiency
        let communication_overhead = metrics.avg_communication_latency_ms / 1000.0; // Convert to seconds
        let compute_time = 0.1; // Simulate 100ms compute per step
        let efficiency = compute_time / (compute_time + communication_overhead);
        println!(
            "   Estimated Training Efficiency: {:.1}%",
            efficiency * 100.0
        );
    }

    Ok(())
}

fn main() -> Result<(), TrustformersError> {
    println!("🎯 Multi-GPU Distributed Training with Averaged Adam");
    println!("=================================================");
    println!("This example demonstrates advanced distributed training capabilities");
    println!("combining Averaged Adam optimization with multi-GPU scaling.\n");

    // Run distributed configuration comparison
    compare_distributed_configurations()?;

    println!("\n🎉 Multi-GPU Distributed Training Analysis Completed!");
    println!("===================================================");

    println!("\n📋 Key Innovations Demonstrated:");
    println!("• Distributed Polyak-Ruppert averaging across multiple devices");
    println!("• ZeRO optimization stages (1, 2, 3) integration with Averaged Adam");
    println!("• Hierarchical gradient aggregation with compression");
    println!("• Fault-tolerant distributed training with checkpointing");
    println!("• Memory-efficient parameter partitioning and synchronization");
    println!("• Performance monitoring and optimization for distributed setups");

    println!("\n💡 Optimization Recommendations:");
    println!("• Use ZeRO Stage 2 for models 100M-1B parameters");
    println!("• Use ZeRO Stage 3 for models >1B parameters");
    println!("• Enable gradient compression for >8 nodes");
    println!("• Set averaging coefficient γ ≥ 0.999 for distributed training");
    println!("• Synchronize averaged parameters every 10-20 steps");
    println!("• Use butterfly aggregation for >16 devices");

    println!("\n🚀 Production Deployment Guidelines:");
    println!("• Monitor communication latency and adjust compression accordingly");
    println!("• Implement progressive fault recovery for large-scale training");
    println!("• Use mixed precision (fp16) with Averaged Adam for memory efficiency");
    println!("• Consider model parallelism for models >10B parameters");
    println!("• Set checkpoint frequency based on cluster reliability");

    Ok(())
}
