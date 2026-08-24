//! Configuration types for automatic parallelism selection: the top-level config, hardware/model constraints, and performance requirements.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Architecture types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArchitectureType {
    Transformer,
    GPT,
    BERT,
    T5,
    MoE,
    ConvNet,
    RNN,
    Custom(String),
}
/// Automatic Parallelism Selection Configuration
///
/// This system automatically chooses the optimal parallelism strategy based on:
/// - Model architecture and size
/// - Hardware configuration
/// - Memory constraints
/// - Communication bandwidth
/// - Performance requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoParallelismConfig {
    /// Enable automatic parallelism selection
    pub enabled: bool,
    /// Strategy selection algorithm
    pub selection_algorithm: SelectionAlgorithm,
    /// Performance optimization objective
    pub optimization_objective: OptimizationObjective,
    /// Hardware constraints
    pub hardware_constraints: HardwareConstraints,
    /// Model constraints
    pub model_constraints: ModelConstraints,
    /// Performance requirements
    pub performance_requirements: PerformanceRequirements,
    /// Strategy evaluation method
    pub evaluation_method: EvaluationMethod,
    /// Whether to use dynamic adaptation during training
    pub dynamic_adaptation: bool,
    /// Adaptation frequency (number of steps)
    pub adaptation_frequency: usize,
}
impl Default for AutoParallelismConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            selection_algorithm: SelectionAlgorithm::CostBasedOptimization,
            optimization_objective: OptimizationObjective::MinimizeTime,
            hardware_constraints: HardwareConstraints::default(),
            model_constraints: ModelConstraints::default(),
            performance_requirements: PerformanceRequirements::default(),
            evaluation_method: EvaluationMethod::ModelBased,
            dynamic_adaptation: false,
            adaptation_frequency: 1000,
        }
    }
}
/// Device types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceType {
    GPU,
    TPU,
    CPU,
    Custom(String),
}
/// Evaluation methods for parallelism strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvaluationMethod {
    /// Model-based evaluation using analytical models
    ModelBased,
    /// Simulation-based evaluation
    SimulationBased,
    /// Profiling-based evaluation (run small experiments)
    ProfilingBased,
    /// Hybrid approach
    Hybrid,
}
/// Hardware constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConstraints {
    /// Number of available devices
    pub num_devices: usize,
    /// Memory per device (in bytes)
    pub memory_per_device: u64,
    /// Compute capability per device (FLOPS)
    pub compute_per_device: f64,
    /// Inter-device bandwidth (bytes/second)
    pub inter_device_bandwidth: u64,
    /// Intra-node bandwidth (bytes/second)
    pub intra_node_bandwidth: u64,
    /// Network latency (microseconds)
    pub network_latency: f64,
    /// Device types (GPU, TPU, CPU)
    pub device_types: Vec<DeviceType>,
    /// Topology information
    pub topology: NetworkTopology,
}
impl Default for HardwareConstraints {
    fn default() -> Self {
        Self {
            num_devices: 8,
            memory_per_device: 80 * 1024 * 1024 * 1024,
            compute_per_device: 312e12,
            inter_device_bandwidth: 600 * 1024 * 1024 * 1024,
            intra_node_bandwidth: 900 * 1024 * 1024 * 1024,
            network_latency: 5.0,
            device_types: vec![DeviceType::GPU; 8],
            topology: NetworkTopology::FullyConnected,
        }
    }
}
/// Model constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConstraints {
    /// Total number of parameters
    pub num_parameters: u64,
    /// Number of layers
    pub num_layers: usize,
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Model architecture type
    pub architecture_type: ArchitectureType,
    /// Whether model uses MoE
    pub has_mixture_of_experts: bool,
    /// Number of experts (if MoE)
    pub num_experts: Option<usize>,
}
impl Default for ModelConstraints {
    fn default() -> Self {
        Self {
            num_parameters: 7_000_000_000,
            num_layers: 32,
            hidden_size: 4096,
            num_attention_heads: 32,
            max_sequence_length: 2048,
            vocab_size: 50257,
            architecture_type: ArchitectureType::Transformer,
            has_mixture_of_experts: false,
            num_experts: None,
        }
    }
}
/// Network topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkTopology {
    FullyConnected,
    Ring,
    Tree,
    Mesh2D,
    Mesh3D,
    Torus,
    Custom(String),
}
/// Optimization objectives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationObjective {
    /// Minimize training time
    MinimizeTime,
    /// Minimize memory usage
    MinimizeMemory,
    /// Minimize communication overhead
    MinimizeCommunication,
    /// Maximize throughput
    MaximizeThroughput,
    /// Maximize efficiency (throughput/resources)
    MaximizeEfficiency,
    /// Multi-objective optimization
    MultiObjective(Vec<OptimizationObjective>),
}
/// Performance requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRequirements {
    /// Maximum acceptable training time
    pub max_training_time: Option<Duration>,
    /// Minimum required throughput (samples/second)
    pub min_throughput: Option<f64>,
    /// Maximum memory usage per device
    pub max_memory_per_device: Option<u64>,
    /// Maximum communication overhead percentage
    pub max_communication_overhead: Option<f32>,
    /// Minimum efficiency requirement
    pub min_efficiency: Option<f32>,
}
impl Default for PerformanceRequirements {
    fn default() -> Self {
        Self {
            max_training_time: None,
            min_throughput: None,
            max_memory_per_device: None,
            max_communication_overhead: Some(0.3),
            min_efficiency: Some(0.7),
        }
    }
}
/// Selection algorithms for parallelism strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionAlgorithm {
    /// Rule-based selection using heuristics
    RuleBased,
    /// Cost-based optimization
    CostBasedOptimization,
    /// Machine learning-based selection
    MLBased,
    /// Genetic algorithm optimization
    GeneticAlgorithm,
    /// Simulated annealing
    SimulatedAnnealing,
    /// Multi-objective optimization
    MultiObjective,
}
