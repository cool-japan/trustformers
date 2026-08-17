//! MLX-*style* graph execution API for Apple Silicon - implemented on Metal and CPU.
//!
//! # This does NOT link Apple's MLX
//!
//! Read this before using anything named `Mlx*` here. This module implements an API
//! *shaped like* Apple's MLX (a declared operation graph, unified-memory allocation,
//! a compile step, per-op compute-unit assignment). It has **no dependency on, and no
//! linkage to, Apple's MLX framework or `mlx-rs`** - there is no `mlx` crate in
//! `Cargo.toml`, no FFI and no `unsafe` graph interop. The `Mlx` prefix is kept only
//! so existing callers keep compiling; read it as "MLX-style", never as "MLX-backed".
//!
//! The module docs used to claim "native MLX (Machine Learning eXchange) framework
//! integration ... seamless CPU/GPU/Neural Engine coordination" while every operation
//! ran as a scalar Rust loop on the CPU. That claim is withdrawn.
//!
//! # What actually executes
//!
//! * With the crate's `metal` feature on macOS, `MatMul`, `Attention` and `LayerNorm`
//!   dispatch to the real Metal kernels in `trustformers_core::gpu_ops::metal`.
//! * Otherwise every op runs on the CPU through `trustformers_core` tensor kernels.
//! * **Nothing ever runs on the Neural Engine.** Neither Metal nor this crate can
//!   dispatch to the ANE (that needs CoreML), so `AssignedComputeUnit` has no
//!   `NeuralEngine` variant any more - it was a label no code path acted on.
//!
//! # Hardware figures
//!
//! Capabilities come from [`super::device_probe::probe_hardware`] (`sysctlbyname`) and,
//! when the `metal` feature is on, from the live `MTLDevice`. Quantities Apple exposes
//! no API for (GPU core count, ANE TOPS, memory bandwidth) are `None`, never a spec-sheet
//! constant.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::error::{CoreError, Result};
use trustformers_core::Tensor;
use trustformers_core::TrustformersError;

/// MLX framework configuration for Apple Silicon optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlxConfig {
    /// Target Apple Silicon device
    pub device: AppleSiliconDevice,
    /// Memory configuration for unified memory architecture
    pub memory_config: UnifiedMemoryConfig,
    /// Compute unit preferences
    pub compute_units: ComputeUnitConfig,
    /// Graph optimization settings
    pub graph_optimization: GraphOptimizationConfig,
    /// Precision configuration
    pub precision_config: PrecisionConfig,
    /// Performance profiling settings
    pub profiling_config: ProfilingConfig,
    /// Compilation strategy
    pub compilation_strategy: CompilationStrategy,
}

/// Apple Silicon device types supported by MLX
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AppleSiliconDevice {
    /// M1 chip
    M1,
    /// M1 Pro chip
    M1Pro,
    /// M1 Max chip
    M1Max,
    /// M1 Ultra chip
    M1Ultra,
    /// M2 chip
    M2,
    /// M2 Pro chip
    M2Pro,
    /// M2 Max chip
    M2Max,
    /// M2 Ultra chip
    M2Ultra,
    /// M3 chip
    M3,
    /// M3 Pro chip
    M3Pro,
    /// M3 Max chip
    M3Max,
    /// M4 chip (latest)
    M4,
    /// M4 Pro chip
    M4Pro,
    /// M4 Max chip
    M4Max,
    /// A17 Pro (iPhone 15 Pro)
    A17Pro,
    /// A18 (iPhone 16)
    A18,
    /// A18 Pro (iPhone 16 Pro)
    A18Pro,
}

/// Unified memory architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedMemoryConfig {
    /// Total system memory to use (GB)
    pub max_memory_gb: f32,
    /// Memory bandwidth optimization
    pub bandwidth_optimization: BandwidthOptimization,
    /// Memory pool strategy
    pub pool_strategy: MemoryPoolStrategy,
    /// Zero-copy operations enablement
    pub zero_copy_enabled: bool,
    /// Memory pressure handling
    pub pressure_handling: MemoryPressureHandling,
}

/// Memory bandwidth optimization strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BandwidthOptimization {
    /// Conservative memory usage
    Conservative,
    /// Balanced memory and performance
    Balanced,
    /// Aggressive memory utilization
    Aggressive,
    /// Adaptive based on workload
    Adaptive,
}

/// Memory pool management strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryPoolStrategy {
    /// Linear allocation
    Linear,
    /// Buddy allocation system
    Buddy,
    /// Slab allocation
    Slab,
    /// Custom MLX optimized
    MlxOptimized,
}

/// Memory pressure handling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryPressureHandling {
    /// Graceful degradation
    GracefulDegradation,
    /// Aggressive cleanup
    AggressiveCleanup,
    /// Model offloading
    ModelOffloading,
    /// Dynamic quantization
    DynamicQuantization,
}

/// Compute unit configuration for heterogeneous execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeUnitConfig {
    /// CPU cores utilization
    pub cpu_config: CpuConfig,
    /// GPU cores utilization
    pub gpu_config: GpuConfig,
    /// Neural Engine utilization
    pub neural_engine_config: NeuralEngineConfig,
    /// Automatic load balancing
    pub auto_load_balancing: bool,
    /// Workload distribution strategy
    pub distribution_strategy: WorkloadDistributionStrategy,
}

/// CPU configuration for Apple Silicon
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuConfig {
    /// Performance cores usage
    pub performance_cores: u8,
    /// Efficiency cores usage
    pub efficiency_cores: u8,
    /// AMX (Apple Matrix) utilization
    pub amx_enabled: bool,
    /// SIMD optimization
    pub simd_optimization: bool,
}

/// GPU configuration for Apple Silicon
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// GPU core count to use
    pub gpu_cores: u16,
    /// Metal Performance Shaders integration
    pub mps_integration: bool,
    /// Tile-based deferred rendering optimization
    pub tbdr_optimization: bool,
    /// Compute pipeline preference
    pub compute_pipeline_preference: ComputePipelinePreference,
}

/// Neural Engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralEngineConfig {
    /// Neural Engine utilization percentage
    pub utilization_percentage: f32,
    /// INT8 quantization preference
    pub int8_quantization: bool,
    /// Batch processing optimization
    pub batch_optimization: bool,
    /// Model caching strategy
    pub model_caching: ModelCachingStrategy,
}

/// Compute pipeline preferences
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputePipelinePreference {
    /// Favor throughput
    Throughput,
    /// Favor latency
    Latency,
    /// Balanced approach
    Balanced,
    /// Power efficient
    PowerEfficient,
}

/// Model caching strategies for Neural Engine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelCachingStrategy {
    /// Persistent caching
    Persistent,
    /// LRU-based caching
    LRU,
    /// Frequency-based caching
    FrequencyBased,
    /// Predictive caching
    Predictive,
}

/// Workload distribution strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkloadDistributionStrategy {
    /// CPU-first approach
    CpuFirst,
    /// GPU-first approach
    GpuFirst,
    /// Neural Engine first
    NeuralEngineFirst,
    /// Heterogeneous distribution
    Heterogeneous,
    /// Adaptive based on workload
    Adaptive,
}

/// Graph optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphOptimizationConfig {
    /// Enable operator fusion
    pub operator_fusion: bool,
    /// Memory layout optimization
    pub memory_layout_optimization: bool,
    /// Constant folding
    pub constant_folding: bool,
    /// Dead code elimination
    pub dead_code_elimination: bool,
    /// Loop optimization
    pub loop_optimization: bool,
    /// Vectorization
    pub vectorization: bool,
    /// Graph-level quantization
    pub graph_quantization: bool,
}

/// Precision configuration for different operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionConfig {
    /// Default precision
    pub default_precision: MlxPrecision,
    /// Per-operation precision overrides
    pub operation_overrides: HashMap<String, MlxPrecision>,
    /// Mixed precision training
    pub mixed_precision_training: bool,
    /// Automatic mixed precision
    pub auto_mixed_precision: bool,
    /// Loss scaling for training
    pub loss_scaling: bool,
}

/// MLX precision types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MlxPrecision {
    /// 32-bit floating point
    Float32,
    /// 16-bit floating point
    Float16,
    /// Brain floating point
    BFloat16,
    /// 32-bit integer
    Int32,
    /// 16-bit integer
    Int16,
    /// 8-bit integer
    Int8,
    /// 4-bit integer
    Int4,
    /// Complex 64-bit
    Complex64,
    /// Complex 128-bit
    Complex128,
}

/// Performance profiling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingConfig {
    /// Enable performance profiling
    pub enabled: bool,
    /// Detailed memory tracking
    pub memory_tracking: bool,
    /// Compute unit utilization tracking
    pub compute_utilization_tracking: bool,
    /// Graph execution profiling
    pub graph_execution_profiling: bool,
    /// Export format for profiling data
    pub export_format: ProfilingExportFormat,
}

/// Profiling data export formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProfilingExportFormat {
    /// JSON format
    Json,
    /// Chrome tracing format
    ChromeTracing,
    /// MLX native format
    MlxNative,
    /// Apple Instruments format
    Instruments,
}

/// Compilation strategies for MLX models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompilationStrategy {
    /// Just-in-time compilation
    JIT,
    /// Ahead-of-time compilation
    AOT,
    /// Hybrid compilation
    Hybrid,
    /// Lazy compilation
    Lazy,
}

/// MLX operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MlxOperation {
    /// Matrix multiplication
    MatMul,
    /// Convolution
    Convolution,
    /// Attention mechanism
    Attention,
    /// Layer normalization
    LayerNorm,
    /// Batch normalization
    BatchNorm,
    /// Activation functions
    Activation,
    /// Embedding lookup
    Embedding,
    /// Softmax
    Softmax,
    /// Reduction operations
    Reduction,
    /// Element-wise operations
    ElementWise,
}

/// MLX performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
/// Measured execution metrics.
///
/// Only quantities this process can actually observe are present. The previous
/// version hardcoded `cpu_utilization = 75.0`, `gpu_utilization = 85.0`,
/// `neural_engine_utilization = 90.0`, a constant 15 W and a constant thermal state
/// under a `// Simulate MLX performance characteristics` comment, and printed them as
/// telemetry. Those fields are either measured now or gone.
pub struct MlxPerformanceMetrics {
    /// Graph nodes executed per second, from `Instant` timing of the last run.
    pub ops_per_second: f64,
    /// Wall-clock duration of the last `execute_model` call, in milliseconds.
    pub last_execution_ms: f64,
    /// Number of graph nodes executed in the last run.
    pub last_execution_nodes: usize,
    /// Process CPU utilisation (percent of one core) sampled via `sysinfo`.
    /// `None` when the platform does not report it.
    pub cpu_utilization: Option<f32>,
    /// Resident set size of this process in GiB, sampled via `sysinfo`.
    pub process_memory_gb: Option<f32>,
    /// Bytes the unified memory pool has handed out, in GiB.
    pub pool_memory_gb: f32,
    /// Compilation time of the last `compile_model` call, in milliseconds.
    pub compilation_time_ms: f64,
}

/// MLX unified compute engine
pub struct MlxEngine {
    pub(super) config: MlxConfig,
    pub(super) device_capabilities: DeviceCapabilities,
    pub(super) performance_metrics: MlxPerformanceMetrics,
    pub(super) compiled_models: HashMap<String, CompiledMlxModel>,
    pub(super) memory_pool: UnifiedMemoryPool,
}

/// Capabilities of the machine this process is running on.
///
/// Every populated field is measured: CPU/memory figures come from `sysctlbyname`
/// and the Metal fields from the live `MTLDevice`. Fields Apple exposes no query for
/// are `Option::None` - the previous version reported a chip-spec lookup table plus a
/// fabricated `mlx_version: "0.15.0"` for a framework that is not linked.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// `machdep.cpu.brand_string`, e.g. "Apple M4 Pro".
    pub cpu_brand: String,
    /// `hw.perflevel0.logicalcpu`. `None` on hardware without core clusters.
    pub performance_cores: Option<u32>,
    /// `hw.perflevel1.logicalcpu`. `None` on hardware without core clusters.
    pub efficiency_cores: Option<u32>,
    /// `hw.logicalcpu` - total logical CPUs.
    pub logical_cores: u32,
    /// `hw.memsize` converted to GiB.
    pub unified_memory_gb: f32,
    /// `hw.optional.amx_version` when present. `None` means "not reported", not "absent".
    pub amx_version: Option<u32>,
    /// `MTLDevice.name` when the `metal` feature is on.
    pub metal_device_name: Option<String>,
    /// Highest supported `MTLGPUFamily.Apple*` generation, when queryable.
    pub apple_gpu_family: Option<u32>,
    /// `MTLDevice.maxBufferLength` in bytes, when queryable.
    pub metal_max_buffer_bytes: Option<usize>,
    /// `MTLDevice.recommendedMaxWorkingSetSize` in bytes, when queryable.
    pub metal_recommended_working_set_bytes: Option<u64>,
    /// `MTLDevice.hasUnifiedMemory`, when queryable.
    pub metal_unified_memory: Option<bool>,
    /// GPU core count. Always `None`: neither Metal nor sysctl exposes it.
    pub gpu_cores: Option<u16>,
    /// Neural Engine throughput. Always `None`: Apple publishes no query API.
    pub neural_engine_tops: Option<f32>,
    /// Memory bandwidth. Always `None`: Apple publishes no query API.
    pub memory_bandwidth_gbps: Option<f32>,
}

/// Apple's **published marketing specifications** for a chip model.
///
/// A transcription of Apple's product pages, returned by
/// [`MlxEngine::published_specs_for`](super::MlxEngine::published_specs_for). It is
/// explicitly *not* a measurement of the running machine - use
/// [`DeviceCapabilities`] for that. The distinction exists because the old
/// `detect_device_capabilities` returned this table under a name that promised
/// detection.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PublishedChipSpecs {
    /// Chip these specs describe.
    pub device: AppleSiliconDevice,
    /// Published performance-core count.
    pub performance_cores: u8,
    /// Published efficiency-core count.
    pub efficiency_cores: u8,
    /// Published GPU core count.
    pub gpu_cores: u16,
    /// Published Neural Engine throughput in TOPS.
    pub neural_engine_tops: f32,
    /// Memory in the base configuration, in GB. Individual machines vary.
    pub base_configuration_memory_gb: f32,
    /// Published memory bandwidth in GB/s.
    pub memory_bandwidth_gbps: f32,
    /// Whether the chip ships Apple's AMX matrix coprocessor.
    pub amx_support: bool,
}

/// Compiled MLX model representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledMlxModel {
    /// Model identifier
    pub model_id: String,
    /// Compilation metadata
    pub compilation_metadata: CompilationMetadata,
    /// Optimized graph representation
    pub optimized_graph: OptimizedGraph,
    /// Memory requirements
    pub memory_requirements: MemoryRequirements,
    /// Performance characteristics
    pub performance_profile: ModelPerformanceProfile,
}

/// Compilation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationMetadata {
    /// Compilation timestamp
    pub compilation_time: std::time::SystemTime,
    /// Which execution backend the graph was compiled for: `"metal"` or `"cpu"`.
    ///
    /// Replaces `mlx_version`, which reported the version of a framework this crate
    /// never linked.
    pub engine_backend: String,
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    /// Target device
    pub target_device: AppleSiliconDevice,
    /// Compilation options
    pub compilation_options: HashMap<String, String>,
}

/// Optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// No optimization
    O0,
    /// Basic optimization
    O1,
    /// Standard optimization
    O2,
    /// Aggressive optimization
    O3,
    /// Size optimization
    Os,
    /// Fast compilation
    Ofast,
}

/// Optimized graph representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedGraph {
    /// Graph nodes
    pub nodes: Vec<GraphNode>,
    /// Graph edges
    pub edges: Vec<GraphEdge>,
    /// Execution order
    pub execution_order: Vec<usize>,
    /// Memory layout
    pub memory_layout: MemoryLayout,
}

/// Graph node representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Node ID
    pub id: usize,
    /// Operation type
    pub operation: MlxOperation,
    /// Input tensors
    pub inputs: Vec<TensorId>,
    /// Output tensors
    pub outputs: Vec<TensorId>,
    /// Node-specific parameters
    pub parameters: HashMap<String, f32>,
    /// Assigned compute unit
    pub compute_unit: AssignedComputeUnit,
}

/// Graph edge representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Source node ID
    pub source: usize,
    /// Destination node ID
    pub destination: usize,
    /// Tensor ID being passed
    pub tensor_id: TensorId,
    /// Data type
    pub data_type: MlxPrecision,
}

/// Tensor identifier
pub type TensorId = u64;

/// Assigned compute unit for graph nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssignedComputeUnit {
    /// Executed by CPU kernels.
    CPU,
    /// Executed by the Metal kernels in `trustformers_core::gpu_ops::metal`.
    ///
    /// Only reachable when the crate is built with the `metal` feature on macOS; the
    /// planner assigns `CPU` otherwise rather than labelling work it cannot dispatch.
    Gpu,
}

/// Memory layout for optimized execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLayout {
    /// Tensor memory allocations
    pub tensor_allocations: HashMap<TensorId, MemoryAllocation>,
    /// Total memory required
    pub total_memory_bytes: usize,
    /// Memory alignment requirements
    pub alignment_requirements: HashMap<TensorId, usize>,
}

/// Memory allocation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAllocation {
    /// Memory offset
    pub offset: usize,
    /// Size in bytes
    pub size_bytes: usize,
    /// Memory type
    pub memory_type: MemoryType,
    /// Lifetime
    pub lifetime: MemoryLifetime,
}

/// Memory types in unified memory architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryType {
    /// System RAM
    SystemRAM,
    /// GPU memory
    GPUMemory,
    /// Neural Engine memory
    NeuralEngineMemory,
    /// Shared memory
    SharedMemory,
}

/// Memory lifetime management
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryLifetime {
    /// Persistent across calls
    Persistent,
    /// Temporary for single inference
    Temporary,
    /// Cached for multiple calls
    Cached,
    /// Streaming memory
    Streaming,
}

/// Memory requirements for compiled models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRequirements {
    /// Minimum memory required (GB)
    pub minimum_memory_gb: f32,
    /// Recommended memory (GB)
    pub recommended_memory_gb: f32,
    /// Peak memory usage (GB)
    pub peak_memory_gb: f32,
    /// Memory fragmentation factor
    pub fragmentation_factor: f32,
}

/// Model performance profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceProfile {
    /// Expected latency (milliseconds)
    pub expected_latency_ms: f64,
    /// Expected throughput (inferences/second)
    pub expected_throughput: f64,
    /// Power consumption estimate (watts)
    pub power_consumption_watts: f32,
    /// Thermal impact (0.0-1.0)
    pub thermal_impact: f32,
    /// Accuracy metrics
    pub accuracy_metrics: HashMap<String, f32>,
}

/// Unified memory pool for Apple Silicon
pub struct UnifiedMemoryPool {
    pub(super) config: UnifiedMemoryConfig,
    pub(super) allocated_memory: HashMap<TensorId, MemoryAllocation>,
    pub(super) total_allocated_bytes: usize,
    pub(super) peak_allocated_bytes: usize,
    pub(super) allocation_count: usize,
}

impl Default for MlxConfig {
    fn default() -> Self {
        Self {
            device: AppleSiliconDevice::M4,
            memory_config: UnifiedMemoryConfig::default(),
            compute_units: ComputeUnitConfig::default(),
            graph_optimization: GraphOptimizationConfig::default(),
            precision_config: PrecisionConfig::default(),
            profiling_config: ProfilingConfig::default(),
            compilation_strategy: CompilationStrategy::Hybrid,
        }
    }
}

impl Default for UnifiedMemoryConfig {
    /// `max_memory_gb: 0.0` means "no explicit cap"; a fixed 16 GiB default made the
    /// default config unusable on 8 GiB machines.
    fn default() -> Self {
        Self {
            max_memory_gb: 0.0,
            bandwidth_optimization: BandwidthOptimization::Balanced,
            pool_strategy: MemoryPoolStrategy::MlxOptimized,
            zero_copy_enabled: true,
            pressure_handling: MemoryPressureHandling::GracefulDegradation,
        }
    }
}

impl Default for ComputeUnitConfig {
    /// Auto-sizing defaults.
    ///
    /// Zero means "use whatever the probed device has". The previous defaults asked
    /// for 8 performance cores and 20 GPU cores unconditionally, which exceeded the
    /// capability table for the default device (M4: 4 performance cores, 10 GPU
    /// cores) - so `MlxEngine::new(MlxConfig::default())` could never succeed.
    fn default() -> Self {
        Self {
            cpu_config: CpuConfig {
                performance_cores: 0,
                efficiency_cores: 0,
                amx_enabled: true,
                simd_optimization: true,
            },
            gpu_config: GpuConfig {
                gpu_cores: 0,
                mps_integration: true,
                tbdr_optimization: true,
                compute_pipeline_preference: ComputePipelinePreference::Balanced,
            },
            neural_engine_config: NeuralEngineConfig {
                utilization_percentage: 0.0,
                int8_quantization: true,
                batch_optimization: true,
                model_caching: ModelCachingStrategy::Predictive,
            },
            auto_load_balancing: true,
            distribution_strategy: WorkloadDistributionStrategy::Adaptive,
        }
    }
}

impl Default for GraphOptimizationConfig {
    fn default() -> Self {
        Self {
            operator_fusion: true,
            memory_layout_optimization: true,
            constant_folding: true,
            dead_code_elimination: true,
            loop_optimization: true,
            vectorization: true,
            graph_quantization: true,
        }
    }
}

impl Default for PrecisionConfig {
    fn default() -> Self {
        Self {
            default_precision: MlxPrecision::Float16,
            operation_overrides: HashMap::new(),
            mixed_precision_training: true,
            auto_mixed_precision: true,
            loss_scaling: true,
        }
    }
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            memory_tracking: true,
            compute_utilization_tracking: true,
            graph_execution_profiling: true,
            export_format: ProfilingExportFormat::Json,
        }
    }
}
