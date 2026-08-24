//! Runtime strategy/result types: a chosen parallelism strategy, its performance metrics, ML features and genetic-algorithm individuals.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::distributed::DistributedConfig;
use crate::expert_parallelism::ExpertParallelismConfig;
use crate::parallelism_3d::ParallelismConfig;
use crate::sequence_parallelism::SequenceParallelismConfig;
use crate::tensor_parallelism::TensorParallelismConfig;
use std::time::Duration;

/// Individual in genetic algorithm population for strategy optimization
#[derive(Debug, Clone)]
pub struct GeneticIndividual {
    /// Parallelism strategy
    pub strategy: ParallelismStrategy,
    /// Fitness score (higher is better)
    pub fitness: f32,
    /// Data parallelism size
    pub dp_size: usize,
    /// Model parallelism size
    pub mp_size: usize,
    /// Pipeline parallelism size
    pub pp_size: usize,
}
/// Features extracted for ML-based strategy prediction
#[derive(Debug, Clone)]
pub struct MLFeatures {
    pub log_num_parameters: f64,
    pub num_layers: f64,
    pub log_hidden_size: f64,
    pub num_attention_heads: f64,
    pub log_sequence_length: f64,
    pub log_vocab_size: f64,
    pub has_moe: f64,
    pub log_num_devices: f64,
    pub log_memory_per_device: f64,
    pub log_compute_per_device: f64,
    pub log_bandwidth: f64,
    pub network_latency: f64,
    pub memory_to_compute_ratio: f64,
    pub parameters_per_device: f64,
    pub communication_intensity: f64,
}
/// Parallelism strategy recommendation
#[derive(Debug, Clone)]
pub struct ParallelismStrategy {
    /// Strategy identifier
    pub strategy_id: String,
    /// Data parallelism configuration
    pub data_parallel: Option<DistributedConfig>,
    /// 3D parallelism configuration
    pub parallelism_3d: Option<ParallelismConfig>,
    /// Expert parallelism configuration
    pub expert_parallel: Option<ExpertParallelismConfig>,
    /// Sequence parallelism configuration
    pub sequence_parallel: Option<SequenceParallelismConfig>,
    /// Tensor parallelism configuration
    pub tensor_parallel: Option<TensorParallelismConfig>,
    /// Expected performance metrics
    pub expected_performance: PerformanceMetrics,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Rationale for this strategy
    pub rationale: String,
}
/// Performance metrics for evaluation
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Expected training time per step
    pub time_per_step: Duration,
    /// Expected memory usage per device
    pub memory_per_device: u64,
    /// Expected communication overhead
    pub communication_overhead: f32,
    /// Expected throughput (samples/second)
    pub throughput: f64,
    /// Expected efficiency score
    pub efficiency: f32,
    /// Expected scalability factor
    pub scalability: f32,
}
