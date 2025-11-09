//! Comprehensive Types Module for Performance Optimizer
//!
//! This module contains all 87 types extracted from the performance optimizer system,
//! organized into logical categories for optimal maintainability and comprehension.
//! Each type includes comprehensive documentation and appropriate Default implementations.

use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicBool, AtomicUsize},
        Arc,
    },
    time::Duration,
};
use tokio::task::JoinHandle;

use crate::test_parallelization::PerformanceOptimizationConfig;
use crate::test_performance_monitoring::TrendDirection;

// =============================================================================
// CORE OPTIMIZATION TYPES (8 types)
// =============================================================================

/// Performance optimization engine for test parallelization
///
/// This is the main coordinator that orchestrates all performance optimization
/// activities, including adaptive parallelism, resource optimization, and
/// intelligent scaling for the TrustformeRS test parallelization framework.
pub struct PerformanceOptimizer {
    /// Configuration
    pub config: Arc<RwLock<PerformanceOptimizationConfig>>,
    /// Adaptive parallelism controller
    pub adaptive_controller: Arc<AdaptiveParallelismController>,
    /// Optimization history
    pub optimization_history: Arc<Mutex<OptimizationHistory>>,
    /// Real-time metrics
    pub real_time_metrics: Arc<RwLock<RealTimeMetrics>>,
    /// Background optimization tasks
    pub background_tasks: Vec<JoinHandle<()>>,
    /// Shutdown signal
    pub shutdown: Arc<AtomicBool>,
}

/// Adaptive parallelism controller for dynamic optimization
///
/// Manages dynamic adjustment of parallelism levels based on real-time
/// performance feedback, system resource availability, and machine learning
/// predictions to maintain optimal test execution performance.
pub struct AdaptiveParallelismController {
    /// Current parallelism level
    pub current_parallelism: Arc<AtomicUsize>,
    /// Optimal parallelism estimator
    pub optimal_estimator: Arc<OptimalParallelismEstimator>,
    /// Parallelism adjustment history
    pub adjustment_history: Arc<Mutex<Vec<ParallelismAdjustment>>>,
    /// Performance feedback system
    pub feedback_system: Arc<PerformanceFeedbackSystem>,
    /// Adaptive learning model
    pub learning_model: Arc<AdaptiveLearningModel>,
    /// Controller configuration
    pub config: Arc<RwLock<AdaptiveParallelismConfig>>,
}

/// Configuration for adaptive parallelism control
///
/// Defines parameters for dynamic parallelism adjustment including learning
/// rates, stability thresholds, and exploration parameters for optimal
/// performance optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveParallelismConfig {
    /// Enable adaptive parallelism
    pub enabled: bool,
    /// Minimum parallelism level
    pub min_parallelism: usize,
    /// Maximum parallelism level
    pub max_parallelism: usize,
    /// Adjustment interval
    pub adjustment_interval: Duration,
    /// Performance measurement window
    pub measurement_window: Duration,
    /// Learning rate for adaptation
    pub learning_rate: f32,
    /// Stability threshold
    pub stability_threshold: f32,
    /// Exploration rate
    pub exploration_rate: f32,
    /// Conservative mode
    pub conservative_mode: bool,
}

/// Optimal parallelism estimator using multiple algorithms
///
/// Employs various estimation algorithms and machine learning models to
/// predict the optimal parallelism level for given test characteristics
/// and system conditions.
pub struct OptimalParallelismEstimator {
    /// Performance model
    pub performance_model: Arc<Mutex<PerformanceModel>>,
    /// Historical performance data
    pub historical_data: Arc<Mutex<Vec<PerformanceDataPoint>>>,
    /// System resource model
    pub resource_model: Arc<Mutex<SystemResourceModel>>,
    /// Estimation algorithms
    pub algorithms: Arc<Mutex<Vec<Box<dyn EstimationAlgorithm + Send + Sync>>>>,
    /// Estimation accuracy tracker
    pub accuracy_tracker: Arc<Mutex<EstimationAccuracyTracker>>,
}

/// Comprehensive optimization recommendations
///
/// Contains all optimization recommendations including parallelism adjustments,
/// resource optimizations, and batching strategies with priority and impact
/// assessments.
#[derive(Debug, Clone)]
pub struct OptimizationRecommendations {
    /// Parallelism recommendation
    pub parallelism: ParallelismEstimate,
    /// Resource optimization recommendations
    pub resource_optimization: Vec<ResourceOptimizationRecommendation>,
    /// Batching recommendations
    pub batching: BatchingRecommendation,
    /// Overall priority
    pub priority: f32,
    /// Expected improvement
    pub expected_improvement: f32,
}

/// Resource optimization recommendation
///
/// Specific recommendation for optimizing a particular system resource
/// including expected impact and implementation complexity assessment.
#[derive(Debug, Clone)]
pub struct ResourceOptimizationRecommendation {
    /// Resource type
    pub resource_type: String,
    /// Recommended action
    pub action: String,
    /// Expected impact
    pub expected_impact: f32,
    /// Implementation complexity
    pub complexity: String,
}

/// Batching recommendation for test execution
///
/// Recommendation for optimal test batching strategy including batch size,
/// strategy type, and expected performance improvement.
#[derive(Debug, Clone)]
pub struct BatchingRecommendation {
    /// Recommended batch size
    pub batch_size: usize,
    /// Batching strategy
    pub strategy: String,
    /// Expected improvement
    pub expected_improvement: f32,
}

/// Record of parallelism adjustment
///
/// Complete record of a parallelism level adjustment including before/after
/// performance measurements, adjustment reason, and effectiveness assessment.
#[derive(Debug, Clone)]
pub struct ParallelismAdjustment {
    /// Adjustment timestamp
    pub timestamp: DateTime<Utc>,
    /// Previous parallelism level
    pub previous_level: usize,
    /// New parallelism level
    pub new_level: usize,
    /// Adjustment reason
    pub reason: AdjustmentReason,
    /// Performance before adjustment
    pub performance_before: PerformanceMeasurement,
    /// Performance after adjustment
    pub performance_after: Option<PerformanceMeasurement>,
    /// Adjustment effectiveness
    pub effectiveness: Option<f32>,
}

// =============================================================================
// PERFORMANCE MODELING TYPES (12 types)
// =============================================================================

/// Performance model for parallelism estimation
///
/// Mathematical model used to predict performance characteristics based on
/// parallelism levels, test characteristics, and system state.
#[derive(Debug, Clone)]
pub struct PerformanceModel {
    /// Model type
    pub model_type: PerformanceModelType,
    /// Model parameters
    pub parameters: HashMap<String, f64>,
    /// Model accuracy
    pub accuracy: f32,
    /// Model last updated
    pub last_updated: DateTime<Utc>,
    /// Model training data size
    pub training_data_size: usize,
    /// Model validation results
    pub validation_results: ModelValidationResults,
}

/// Types of performance models available
///
/// Enumeration of different mathematical and machine learning models
/// that can be used for performance prediction and optimization.
#[derive(Debug, Clone)]
pub enum PerformanceModelType {
    /// Linear regression model
    LinearRegression,
    /// Polynomial regression model
    PolynomialRegression { degree: usize },
    /// Exponential model
    Exponential,
    /// Neural network model
    NeuralNetwork { hidden_layers: Vec<usize> },
    /// Ensemble model
    Ensemble { models: Vec<PerformanceModelType> },
    /// Custom model
    Custom(String),
}

/// Results from model validation process
///
/// Comprehensive validation metrics including statistical measures,
/// cross-validation scores, and validation timestamp.
#[derive(Debug, Clone)]
pub struct ModelValidationResults {
    /// R-squared score
    pub r_squared: f32,
    /// Mean absolute error
    pub mean_absolute_error: f32,
    /// Root mean squared error
    pub root_mean_squared_error: f32,
    /// Cross-validation scores
    pub cross_validation_scores: Vec<f32>,
    /// Validation timestamp
    pub validated_at: DateTime<Utc>,
}

/// Performance data point for model training
///
/// Single measurement point containing parallelism level, performance metrics,
/// test characteristics, and system state for training performance models.
#[derive(Debug, Clone)]
pub struct PerformanceDataPoint {
    /// Parallelism level
    pub parallelism: usize,
    /// Throughput (tests per second)
    pub throughput: f64,
    /// Latency (average test execution time)
    pub latency: Duration,
    /// CPU utilization
    pub cpu_utilization: f32,
    /// Memory utilization
    pub memory_utilization: f32,
    /// Resource efficiency
    pub resource_efficiency: f32,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Test characteristics
    pub test_characteristics: TestCharacteristics,
    /// System state
    pub system_state: SystemState,
}

/// Comprehensive performance measurement
///
/// Detailed performance measurement including throughput, latency, resource
/// utilization, and efficiency metrics for a specific time period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMeasurement {
    /// Throughput (tests per second)
    pub throughput: f64,
    /// Average latency
    pub average_latency: Duration,
    /// CPU utilization
    pub cpu_utilization: f32,
    /// Memory utilization
    pub memory_utilization: f32,
    /// Resource efficiency
    pub resource_efficiency: f32,
    /// Measurement timestamp
    pub timestamp: DateTime<Utc>,
    /// Measurement duration
    pub measurement_duration: Duration,
    /// CPU usage (alias for cpu_utilization)
    pub cpu_usage: f32,
    /// Memory usage (alias for memory_utilization)
    pub memory_usage: f32,
    /// Latency (alias for average_latency)
    pub latency: Duration,
}

/// Performance feedback from system components
///
/// Feedback information from various system components including source,
/// type, value, and context for performance optimization decisions.
#[derive(Debug, Clone)]
pub struct PerformanceFeedback {
    /// Feedback source
    pub source: FeedbackSource,
    /// Feedback type
    pub feedback_type: FeedbackType,
    /// Feedback value
    pub value: f64,
    /// Feedback timestamp
    pub timestamp: DateTime<Utc>,
    /// Associated parallelism level
    pub parallelism_level: usize,
    /// Feedback context
    pub context: FeedbackContext,
}

/// Processed feedback with recommendations
///
/// Feedback that has been processed through analysis algorithms with
/// confidence scores and recommended actions.
#[derive(Debug, Clone)]
pub struct ProcessedFeedback {
    /// Original feedback
    pub original_feedback: PerformanceFeedback,
    /// Processed value
    pub processed_value: f64,
    /// Processing method
    pub processing_method: String,
    /// Processing confidence
    pub confidence: f32,
    /// Recommended action
    pub recommended_action: Option<RecommendedAction>,
}

/// Performance trend analysis
///
/// Analysis of performance trends over time including direction, strength,
/// confidence, and supporting data points.
#[derive(Debug, Clone)]
pub struct PerformanceTrend {
    /// Trend direction
    pub direction: TrendDirection,
    /// Trend strength
    pub strength: f32,
    /// Trend confidence
    pub confidence: f32,
    /// Trend period
    pub period: Duration,
    /// Trend data points
    pub data_points: Vec<PerformanceDataPoint>,
}

/// Snapshot of performance at a specific time
///
/// Complete performance snapshot including metrics, model version,
/// and comparative analysis against baseline performance.
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    /// Snapshot timestamp
    pub timestamp: DateTime<Utc>,
    /// Model version
    pub model_version: u64,
    /// Performance metrics
    pub metrics: ModelPerformanceMetrics,
    /// Comparative performance
    pub comparative_performance: Option<ComparativePerformance>,
}

/// Comparative performance analysis
///
/// Comparison between baseline and current performance including
/// improvement metrics and statistical significance.
#[derive(Debug, Clone)]
pub struct ComparativePerformance {
    /// Baseline performance
    pub baseline: ModelPerformanceMetrics,
    /// Current performance
    pub current: ModelPerformanceMetrics,
    /// Performance improvement
    pub improvement: f32,
    /// Statistical significance
    pub significance: f32,
}

/// Machine learning model performance metrics
///
/// Comprehensive metrics for evaluating machine learning model performance
/// including accuracy, loss, and convergence status.
#[derive(Debug, Clone)]
pub struct ModelPerformanceMetrics {
    /// Training accuracy
    pub training_accuracy: f32,
    /// Validation accuracy
    pub validation_accuracy: f32,
    /// Test accuracy
    pub test_accuracy: f32,
    /// Loss function value
    pub loss: f64,
    /// Convergence status
    pub convergence_status: ConvergenceStatus,
    /// Overall accuracy
    pub accuracy: f32,
    /// Precision metric
    pub precision: f32,
    /// Recall metric
    pub recall: f32,
    /// F1 score
    pub f1_score: f32,
    /// Training examples count
    pub training_examples: usize,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

/// Model convergence status enumeration
///
/// Indicates the current state of model convergence during training
/// and optimization processes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConvergenceStatus {
    /// Model converged
    Converged,
    /// Model converging
    Converging,
    /// Model not converged
    NotConverged,
    /// Model diverging
    Diverging,
    /// Convergence unknown
    Unknown,
}

// =============================================================================
// SYSTEM RESOURCE MODELING TYPES (18 types)
// =============================================================================

/// Complete system state snapshot
///
/// Comprehensive snapshot of system state including resource availability,
/// utilization, and environmental conditions at measurement time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    /// Available CPU cores
    pub available_cores: usize,
    /// Available memory (MB)
    pub available_memory_mb: u64,
    /// System load average
    pub load_average: f32,
    /// Active processes
    pub active_processes: usize,
    /// I/O wait percentage
    pub io_wait_percent: f32,
    /// Network utilization
    pub network_utilization: f32,
    /// Temperature metrics
    pub temperature_metrics: Option<TemperatureMetrics>,
}

/// System temperature monitoring
///
/// Temperature metrics for various system components including CPU, GPU,
/// and thermal throttling status for performance optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureMetrics {
    /// CPU temperature (Celsius)
    pub cpu_temperature: f32,
    /// GPU temperature (Celsius)
    pub gpu_temperature: Option<f32>,
    /// System temperature (Celsius)
    pub system_temperature: f32,
    /// Thermal throttling active
    pub thermal_throttling: bool,
}

/// Comprehensive system resource model
///
/// Complete model of system resources including CPU, memory, I/O, network,
/// and GPU characteristics for performance optimization decisions.
#[derive(Debug, Clone)]
pub struct SystemResourceModel {
    /// CPU model
    pub cpu_model: CpuModel,
    /// Memory model
    pub memory_model: MemoryModel,
    /// I/O model
    pub io_model: IoModel,
    /// Network model
    pub network_model: NetworkModel,
    /// GPU model
    pub gpu_model: Option<GpuModel>,
    /// Model last updated
    pub last_updated: DateTime<Utc>,
}

/// CPU characteristics model
///
/// Detailed CPU model including core count, frequency, cache hierarchy,
/// and performance characteristics for optimization calculations.
#[derive(Debug, Clone)]
pub struct CpuModel {
    /// Number of cores
    pub core_count: usize,
    /// Number of threads
    pub thread_count: usize,
    /// Base frequency (MHz)
    pub base_frequency_mhz: u32,
    /// Max frequency (MHz)
    pub max_frequency_mhz: u32,
    /// Cache hierarchy
    pub cache_hierarchy: CacheHierarchy,
    /// Performance characteristics
    pub performance_characteristics: CpuPerformanceCharacteristics,
}

/// Memory system model
///
/// Memory subsystem characteristics including capacity, type, speed,
/// bandwidth, and access patterns for performance optimization.
#[derive(Debug, Clone)]
pub struct MemoryModel {
    /// Total memory (MB)
    pub total_memory_mb: u64,
    /// Memory type
    pub memory_type: MemoryType,
    /// Memory speed (MHz)
    pub memory_speed_mhz: u32,
    /// Memory bandwidth (GB/s)
    pub bandwidth_gbps: f32,
    /// Memory latency
    pub latency: Duration,
    /// Page size (KB)
    pub page_size_kb: u32,
}

/// I/O subsystem model
///
/// Storage and I/O characteristics including device types, bandwidth,
/// latency, and queue depth for I/O optimization strategies.
#[derive(Debug, Clone)]
pub struct IoModel {
    /// Storage devices
    pub storage_devices: Vec<StorageDevice>,
    /// Total I/O bandwidth (MB/s)
    pub total_bandwidth_mbps: f32,
    /// Average I/O latency
    pub average_latency: Duration,
    /// I/O queue depth
    pub queue_depth: usize,
}

/// Network subsystem model
///
/// Network infrastructure characteristics including interfaces, bandwidth,
/// latency, and reliability metrics for network-aware optimizations.
#[derive(Debug, Clone)]
pub struct NetworkModel {
    /// Network interfaces
    pub interfaces: Vec<NetworkInterface>,
    /// Total bandwidth (Mbps)
    pub total_bandwidth_mbps: f32,
    /// Network latency
    pub latency: Duration,
    /// Packet loss rate
    pub packet_loss_rate: f32,
}

/// GPU subsystem model
///
/// GPU characteristics including devices, memory, compute capability,
/// and utilization patterns for GPU-accelerated optimizations.
#[derive(Debug, Clone)]
pub struct GpuModel {
    /// GPU devices
    pub devices: Vec<GpuDeviceModel>,
    /// Total GPU memory (MB)
    pub total_memory_mb: u64,
    /// GPU compute capability
    pub compute_capability: f32,
    /// GPU utilization characteristics
    pub utilization_characteristics: GpuUtilizationCharacteristics,
}

/// CPU cache hierarchy model
///
/// Detailed cache hierarchy including L1, L2, L3 cache sizes and
/// cache line characteristics for memory optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheHierarchy {
    /// L1 cache size (KB)
    pub l1_cache_kb: u32,
    /// L2 cache size (KB)
    pub l2_cache_kb: u32,
    /// L3 cache size (KB)
    pub l3_cache_kb: Option<u32>,
    /// Cache line size (bytes)
    pub cache_line_size: u32,
}

/// CPU performance characteristics
///
/// Detailed CPU performance metrics including IPC, context switch overhead,
/// thread creation costs, and NUMA topology information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuPerformanceCharacteristics {
    /// Instructions per clock
    pub instructions_per_clock: f32,
    /// Context switch overhead
    pub context_switch_overhead: Duration,
    /// Thread creation overhead
    pub thread_creation_overhead: Duration,
    /// NUMA topology
    pub numa_topology: Option<NumaTopology>,
}

/// NUMA topology information
///
/// NUMA (Non-Uniform Memory Access) topology details including node count,
/// core distribution, and inter-node latency characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaTopology {
    /// Number of NUMA nodes
    pub node_count: usize,
    /// Cores per NUMA node
    pub cores_per_node: Vec<usize>,
    /// Inter-node latency
    pub inter_node_latency: Duration,
    /// Intra-node latency
    pub intra_node_latency: Duration,
}

/// Memory technology types
///
/// Enumeration of different memory technologies with their specific
/// characteristics and performance profiles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryType {
    /// DDR3 memory
    Ddr3,
    /// DDR4 memory
    Ddr4,
    /// DDR5 memory
    Ddr5,
    /// LPDDR4 memory
    Lpddr4,
    /// LPDDR5 memory
    Lpddr5,
    /// High bandwidth memory
    Hbm,
    /// Custom memory type
    Custom(String),
}

/// Storage device characteristics
///
/// Individual storage device model including type, capacity, bandwidth,
/// IOPS, and access latency for storage optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageDevice {
    /// Device type
    pub device_type: StorageDeviceType,
    /// Capacity (GB)
    pub capacity_gb: u64,
    /// Read bandwidth (MB/s)
    pub read_bandwidth_mbps: f32,
    /// Write bandwidth (MB/s)
    pub write_bandwidth_mbps: f32,
    /// Random read IOPS
    pub random_read_iops: u32,
    /// Random write IOPS
    pub random_write_iops: u32,
    /// Access latency
    pub access_latency: Duration,
}

/// Storage device technology types
///
/// Classification of storage technologies with different performance
/// characteristics and optimization strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageDeviceType {
    /// Solid state drive
    Ssd,
    /// NVMe SSD
    NvmeSsd,
    /// Hard disk drive
    Hdd,
    /// Network attached storage
    Nas,
    /// RAM disk
    RamDisk,
    /// Custom storage
    Custom(String),
}

/// Network interface characteristics
///
/// Individual network interface model including type, bandwidth,
/// MTU, and operational status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInterface {
    /// Interface type
    pub interface_type: NetworkInterfaceType,
    /// Bandwidth (Mbps)
    pub bandwidth_mbps: f32,
    /// MTU size
    pub mtu_size: u32,
    /// Interface status
    pub status: NetworkInterfaceStatus,
}

/// Network interface technology types
///
/// Classification of network interface types with different
/// performance and reliability characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkInterfaceType {
    /// Ethernet
    Ethernet,
    /// WiFi
    Wifi,
    /// Loopback
    Loopback,
    /// InfiniBand
    InfiniBand,
    /// Custom interface
    Custom(String),
}

/// Network interface operational status
///
/// Current operational state of network interfaces for
/// network-aware optimization decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkInterfaceStatus {
    /// Interface up
    Up,
    /// Interface down
    Down,
    /// Interface degraded
    Degraded,
    /// Interface unknown
    Unknown,
}

/// Individual GPU device model
///
/// Detailed characteristics of a single GPU device including
/// memory, compute units, clock speeds, and bandwidth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceModel {
    /// Device ID
    pub device_id: usize,
    /// Device name
    pub device_name: String,
    /// Memory (MB)
    pub memory_mb: u64,
    /// Compute units
    pub compute_units: u32,
    /// Base clock (MHz)
    pub base_clock_mhz: u32,
    /// Boost clock (MHz)
    pub boost_clock_mhz: u32,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth_gbps: f32,
}

/// GPU utilization characteristics
///
/// GPU-specific performance characteristics including context switching,
/// memory transfer, and kernel launch overheads.
#[derive(Debug, Clone)]
pub struct GpuUtilizationCharacteristics {
    /// Context switch overhead
    pub context_switch_overhead: Duration,
    /// Memory transfer overhead
    pub memory_transfer_overhead: Duration,
    /// Kernel launch overhead
    pub kernel_launch_overhead: Duration,
    /// Maximum concurrent kernels
    pub max_concurrent_kernels: usize,
}

// =============================================================================
// TEST CHARACTERISTICS TYPES (10 types)
// =============================================================================

/// Comprehensive test characteristics
///
/// Complete characterization of test properties including resource intensity,
/// concurrency requirements, and dependencies for optimization decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCharacteristics {
    /// Test categories distribution
    pub category_distribution: HashMap<String, f32>,
    /// Average test duration
    #[serde(skip)]
    pub average_duration: Duration,
    /// Resource intensity
    pub resource_intensity: ResourceIntensity,
    /// Concurrency requirements
    pub concurrency_requirements: ConcurrencyRequirements,
    /// Dependency complexity
    pub dependency_complexity: f32,
}

/// Resource intensity profile
///
/// Quantified intensity of resource usage across different system
/// components for resource-aware optimization strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceIntensity {
    /// CPU intensity (0.0 to 1.0)
    pub cpu_intensity: f32,
    /// Memory intensity (0.0 to 1.0)
    pub memory_intensity: f32,
    /// I/O intensity (0.0 to 1.0)
    pub io_intensity: f32,
    /// Network intensity (0.0 to 1.0)
    pub network_intensity: f32,
    /// GPU intensity (0.0 to 1.0)
    pub gpu_intensity: Option<f32>,
}

// Re-export the canonical ConcurrencyRequirements from test_characterization module
// to avoid duplication and ensure consistency across the codebase
pub use crate::performance_optimizer::test_characterization::types::patterns::ConcurrencyRequirements;

/// Resource sharing capabilities
///
/// Defines which system resources can be safely shared between
/// concurrent test executions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSharingCapabilities {
    /// CPU sharing supported
    pub cpu_sharing: bool,
    /// Memory sharing supported
    pub memory_sharing: bool,
    /// I/O sharing supported
    pub io_sharing: bool,
    /// Network sharing supported
    pub network_sharing: bool,
    /// Custom resource sharing
    pub custom_sharing: HashMap<String, bool>,
}

/// Test synchronization requirements
///
/// Synchronization needs including exclusive access requirements,
/// ordering constraints, and coordination points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynchronizationRequirements {
    /// Requires exclusive access
    pub exclusive_access: Vec<String>,
    /// Requires ordered execution
    pub ordered_execution: bool,
    /// Synchronization points
    pub synchronization_points: Vec<SynchronizationPoint>,
    /// Lock dependencies
    pub lock_dependencies: Vec<LockDependency>,
}

/// Synchronization coordination point
///
/// Specific synchronization point requiring coordination between
/// multiple concurrent test executions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynchronizationPoint {
    /// Point identifier
    pub id: String,
    /// Point type
    pub point_type: SynchronizationPointType,
    /// Required participants
    pub required_participants: usize,
    /// Timeout for synchronization
    pub timeout: Duration,
}

/// Types of synchronization points
///
/// Classification of different synchronization mechanisms
/// available for test coordination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SynchronizationPointType {
    /// Barrier synchronization
    Barrier,
    /// Rendezvous synchronization
    Rendezvous,
    /// Producer-consumer synchronization
    ProducerConsumer,
    /// Custom synchronization
    Custom(String),
}

/// Lock dependency specification
///
/// Specification of lock dependencies including type, duration,
/// and priority for deadlock avoidance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockDependency {
    /// Lock identifier
    pub lock_id: String,
    /// Lock type
    pub lock_type: LockType,
    /// Lock duration estimate
    pub duration_estimate: Duration,
    /// Lock priority
    pub priority: f32,
}

/// Lock mechanism types
///
/// Different types of locking mechanisms with varying
/// performance and concurrency characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LockType {
    /// Shared lock (read)
    Shared,
    /// Exclusive lock (write)
    Exclusive,
    /// Upgradeable lock
    Upgradeable,
    /// Custom lock
    Custom(String),
}

/// Feedback context information
///
/// Contextual information accompanying performance feedback including
/// test characteristics and system state during feedback generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackContext {
    /// Test characteristics during feedback
    pub test_characteristics: TestCharacteristics,
    /// System state during feedback
    pub system_state: SystemState,
    /// Additional context
    pub additional_context: HashMap<String, String>,
}

// =============================================================================
// PARALLELISM ESTIMATION TYPES (8 types)
// =============================================================================

/// Estimation algorithm trait for parallelism optimization
///
/// Interface for algorithms that estimate optimal parallelism levels
/// based on historical data, test characteristics, and system state.
pub trait EstimationAlgorithm {
    /// Estimate optimal parallelism level
    fn estimate_optimal_parallelism(
        &self,
        historical_data: &[PerformanceDataPoint],
        current_characteristics: &TestCharacteristics,
        system_state: &SystemState,
    ) -> Result<ParallelismEstimate>;

    /// Get algorithm name
    fn name(&self) -> &str;

    /// Get algorithm confidence for current data
    fn confidence(&self, data_points: usize) -> f32;
}

/// Parallelism estimation result
///
/// Result of parallelism estimation including optimal level, confidence,
/// expected improvement, and method metadata.
#[derive(Debug, Clone)]
pub struct ParallelismEstimate {
    /// Estimated optimal parallelism
    pub optimal_parallelism: usize,
    /// Confidence in estimate (0.0 to 1.0)
    pub confidence: f32,
    /// Expected performance improvement
    pub expected_improvement: f32,
    /// Estimation method used
    pub method: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Estimation accuracy tracking
///
/// Tracks the accuracy of parallelism estimation algorithms over time
/// for continuous improvement and algorithm selection.
#[derive(Debug, Default, Clone)]
pub struct EstimationAccuracyTracker {
    /// Estimation history
    pub estimation_history: Vec<EstimationRecord>,
    /// Accuracy by algorithm
    pub accuracy_by_algorithm: HashMap<String, f32>,
    /// Total estimations made
    pub total_estimations: u64,
    /// Correct estimations
    pub correct_estimations: u64,
    /// Average estimation error
    pub average_error: f32,
}

/// Individual estimation record
///
/// Record of a single estimation including predicted and actual values,
/// algorithm used, confidence, and accuracy metrics.
#[derive(Debug, Clone)]
pub struct EstimationRecord {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Estimated parallelism
    pub estimated_parallelism: usize,
    /// Actual optimal parallelism
    pub actual_optimal_parallelism: Option<usize>,
    /// Estimation algorithm
    pub algorithm: String,
    /// Estimation confidence
    pub confidence: f32,
    /// Estimation error
    pub error: Option<f32>,
}

/// Reasons for parallelism adjustments
///
/// Enumeration of different reasons why parallelism levels
/// might be adjusted during optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdjustmentReason {
    /// Performance improvement opportunity
    PerformanceImprovement,
    /// Resource constraint
    ResourceConstraint,
    /// System overload
    SystemOverload,
    /// System load changes
    SystemLoad,
    /// Algorithm recommendation
    AlgorithmRecommendation,
    /// Manual adjustment
    Manual,
    /// Experiment
    Experiment,
}

/// Performance feedback system
///
/// System for collecting, processing, and aggregating performance feedback
/// from various sources for continuous optimization improvement.
pub struct PerformanceFeedbackSystem {
    /// Feedback queue
    pub feedback_queue: Arc<Mutex<VecDeque<PerformanceFeedback>>>,
    /// Feedback processors
    pub feedback_processors: Arc<Mutex<Vec<Box<dyn FeedbackProcessor + Send + Sync>>>>,
    /// Feedback aggregator
    pub feedback_aggregator: Arc<FeedbackAggregator>,
    /// Real-time feedback enabled
    pub real_time_feedback: Arc<AtomicBool>,
}

/// Sources of performance feedback
///
/// Classification of different sources that can provide
/// performance feedback for optimization decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeedbackSource {
    /// Performance monitor
    PerformanceMonitor,
    /// Resource monitor
    ResourceMonitor,
    /// Test execution engine
    TestExecutionEngine,
    /// External system
    ExternalSystem,
    /// User input
    UserInput,
}

/// Types of performance feedback
///
/// Different categories of performance feedback that can
/// influence optimization decisions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeedbackType {
    /// Throughput feedback
    Throughput,
    /// Latency feedback
    Latency,
    /// Resource utilization feedback
    ResourceUtilization,
    /// Quality feedback
    Quality,
    /// Error rate feedback
    ErrorRate,
    /// Custom feedback
    Custom(String),
}

// =============================================================================
// FEEDBACK SYSTEMS TYPES (9 types)
// =============================================================================

/// Feedback processor trait
///
/// Interface for processing raw performance feedback into actionable
/// insights and recommendations for optimization decisions.
pub trait FeedbackProcessor {
    /// Process feedback
    fn process_feedback(&self, feedback: &PerformanceFeedback) -> Result<ProcessedFeedback>;

    /// Get processor name
    fn name(&self) -> &str;

    /// Check if processor can handle feedback type
    fn can_process(&self, feedback_type: &FeedbackType) -> bool;
}

/// Recommended optimization action
///
/// Specific action recommendation including type, parameters,
/// priority, and expected impact assessment.
#[derive(Debug, Clone)]
pub struct RecommendedAction {
    /// Action type
    pub action_type: ActionType,
    /// Action parameters
    pub parameters: HashMap<String, String>,
    /// Action priority
    pub priority: f32,
    /// Expected impact
    pub expected_impact: f32,
    /// Whether action is reversible
    pub reversible: bool,
    /// Estimated duration to complete
    pub estimated_duration: Duration,
}

/// Types of optimization actions
///
/// Classification of different optimization actions that can
/// be recommended and executed by the system.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionType {
    /// Increase parallelism
    IncreaseParallelism,
    /// Decrease parallelism
    DecreaseParallelism,
    /// Adjust resource allocation
    AdjustResourceAllocation,
    /// Change scheduling strategy
    ChangeSchedulingStrategy,
    /// Optimize test batching
    OptimizeTestBatching,
    /// Tune system parameters
    TuneParameters,
    /// Optimize resources
    OptimizeResources,
    /// Custom action
    Custom(String),
}

/// Feedback aggregation system
///
/// System for aggregating multiple feedback sources into coherent
/// optimization recommendations with confidence metrics.
pub struct FeedbackAggregator {
    /// Aggregation strategies
    pub strategies: Arc<Mutex<Vec<Box<dyn AggregationStrategy + Send + Sync>>>>,
    /// Aggregated feedback cache
    pub aggregated_cache: Arc<Mutex<HashMap<String, AggregatedFeedback>>>,
    /// Aggregation history
    pub aggregation_history: Arc<Mutex<Vec<AggregationRecord>>>,
}

/// Aggregation strategy trait
///
/// Interface for different strategies to aggregate multiple
/// feedback sources into unified recommendations.
pub trait AggregationStrategy {
    /// Aggregate feedback
    fn aggregate(&self, feedback: &[ProcessedFeedback]) -> Result<AggregatedFeedback>;

    /// Get strategy name
    fn name(&self) -> &str;

    /// Check if strategy is applicable
    fn is_applicable(&self, feedback: &[ProcessedFeedback]) -> bool;
}

/// Aggregated feedback result
///
/// Result of feedback aggregation including aggregated value,
/// confidence, contributing feedback count, and recommendations.
#[derive(Debug, Clone)]
pub struct AggregatedFeedback {
    /// Aggregated value
    pub aggregated_value: f64,
    /// Aggregation confidence
    pub confidence: f32,
    /// Contributing feedback count
    pub contributing_count: usize,
    /// Contributing feedback count (alias)
    pub contributing_feedback_count: usize,
    /// Aggregation method
    pub aggregation_method: String,
    /// Aggregation timestamp
    pub timestamp: DateTime<Utc>,
    /// Recommended actions
    pub recommended_actions: Vec<RecommendedAction>,
}

/// Aggregation process record
///
/// Record of a feedback aggregation process including input count,
/// strategy used, result, and processing duration.
#[derive(Debug, Clone)]
pub struct AggregationRecord {
    /// Aggregation timestamp
    pub timestamp: DateTime<Utc>,
    /// Input feedback count
    pub input_count: usize,
    /// Aggregation strategy used
    pub strategy: String,
    /// Aggregation result
    pub result: AggregatedFeedback,
    /// Aggregation duration
    pub duration: Duration,
    /// Input feedback count (alias)
    pub input_feedback_count: usize,
    /// Strategies used in aggregation
    pub strategies_used: Vec<String>,
    /// Aggregated results
    pub aggregated_results: Vec<AggregatedFeedback>,
}

// =============================================================================
// MACHINE LEARNING TYPES (12 types)
// =============================================================================

/// Adaptive learning model for continuous optimization
///
/// Machine learning model that continuously learns from performance
/// data to improve optimization decisions over time.
pub struct AdaptiveLearningModel {
    /// Model state
    pub model_state: Arc<RwLock<ModelState>>,
    /// Learning algorithm
    pub learning_algorithm: Arc<Mutex<Box<dyn LearningAlgorithm + Send + Sync>>>,
    /// Training data
    pub training_data: Arc<Mutex<TrainingDataset>>,
    /// Model validation
    pub model_validation: Arc<ModelValidation>,
    /// Learning history
    pub learning_history: Arc<Mutex<LearningHistory>>,
    /// Training dataset (alias)
    pub training_dataset: Arc<Mutex<TrainingDataset>>,
}

/// Current state of the learning model
///
/// Complete state of the machine learning model including parameters,
/// weights, version, and performance metrics.
#[derive(Debug, Clone)]
pub struct ModelState {
    /// Model parameters
    pub parameters: HashMap<String, f64>,
    /// Model weights
    pub weights: Vec<f64>,
    /// Model bias
    pub bias: f64,
    /// Model version
    pub version: u64,
    /// Last training timestamp
    pub last_training: DateTime<Utc>,
    /// Model performance metrics
    pub performance_metrics: ModelPerformanceMetrics,
    /// Learning rate
    pub learning_rate: f64,
    /// Model accuracy
    pub accuracy: f64,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
    /// Training examples count
    pub training_examples_count: usize,
}

/// Learning algorithm interface
///
/// Interface for different machine learning algorithms that can
/// be used for performance optimization.
pub trait LearningAlgorithm {
    /// Train the model
    fn train(&mut self, training_data: &TrainingDataset) -> Result<ModelState>;

    /// Predict with the model
    fn predict(&self, input: &[f64]) -> Result<f64>;

    /// Update model with new data
    fn update(&mut self, new_data: &[TrainingExample]) -> Result<ModelState>;

    /// Get algorithm name
    fn name(&self) -> &str;
}

/// Training dataset for machine learning
///
/// Complete training dataset including examples, split ratios,
/// statistics, and quality metrics for model training.
#[derive(Debug, Clone)]
pub struct TrainingDataset {
    /// Training examples
    pub examples: Vec<TrainingExample>,
    /// Dataset split ratios
    pub split_ratios: DatasetSplitRatios,
    /// Dataset statistics
    pub statistics: DatasetStatistics,
    /// Dataset version
    pub version: u64,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
    /// Validation split ratio
    pub validation_split: f32,
}

/// Individual training example
///
/// Single training example with input features, target value,
/// weight, and metadata for machine learning.
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// Input features
    pub features: Vec<f64>,
    /// Target value
    pub target: f64,
    /// Example weight
    pub weight: f64,
    /// Example timestamp
    pub timestamp: DateTime<Utc>,
    /// Example metadata
    pub metadata: HashMap<String, String>,
}

/// Dataset split configuration
///
/// Configuration for splitting datasets into training, validation,
/// and test sets for proper model evaluation.
#[derive(Debug, Clone)]
pub struct DatasetSplitRatios {
    /// Training set ratio
    pub training: f32,
    /// Validation set ratio
    pub validation: f32,
    /// Test set ratio
    pub test: f32,
}

/// Dataset statistical analysis
///
/// Comprehensive statistical analysis of the training dataset
/// including feature statistics, target statistics, and quality metrics.
#[derive(Debug, Clone)]
pub struct DatasetStatistics {
    /// Number of examples
    pub example_count: usize,
    /// Feature statistics
    pub feature_stats: Vec<FeatureStatistics>,
    /// Target statistics
    pub target_stats: TargetStatistics,
    /// Data quality metrics
    pub quality_metrics: DataQualityMetrics,
    /// Total examples
    pub total_examples: usize,
    /// Feature statistics (detailed)
    pub feature_statistics: Vec<FeatureStatistics>,
    /// Target statistics (detailed)
    pub target_statistics: TargetStatistics,
}

/// Statistics for individual features
///
/// Statistical analysis of individual features including distribution
/// statistics, missing values, and data quality indicators.
#[derive(Debug, Clone)]
pub struct FeatureStatistics {
    /// Feature index
    pub feature_index: usize,
    /// Feature name
    pub feature_name: String,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Missing value count
    pub missing_count: usize,
}

/// Target variable statistics
///
/// Statistical analysis of the target variable including distribution
/// characteristics and statistical properties.
#[derive(Debug, Clone)]
pub struct TargetStatistics {
    /// Mean target value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Target distribution
    pub distribution: TargetDistribution,
}

/// Target variable distribution analysis
///
/// Analysis of target variable distribution including type,
/// parameters, and goodness of fit measures.
#[derive(Debug, Clone)]
pub struct TargetDistribution {
    /// Distribution type
    pub distribution_type: DistributionType,
    /// Distribution parameters
    pub parameters: HashMap<String, f64>,
    /// Goodness of fit
    pub goodness_of_fit: f32,
}

/// Statistical distribution types
///
/// Classification of different statistical distributions
/// for modeling target variable behavior.
#[derive(Debug, Clone)]
pub enum DistributionType {
    /// Normal distribution
    Normal,
    /// Uniform distribution
    Uniform,
    /// Exponential distribution
    Exponential,
    /// Custom distribution
    Custom(String),
}

impl Default for DistributionType {
    fn default() -> Self {
        DistributionType::Normal
    }
}

/// Data quality assessment metrics
///
/// Comprehensive assessment of data quality including completeness,
/// consistency, accuracy, and outlier detection.
#[derive(Debug, Clone)]
pub struct DataQualityMetrics {
    /// Completeness (0.0 to 1.0)
    pub completeness: f32,
    /// Consistency (0.0 to 1.0)
    pub consistency: f32,
    /// Accuracy (0.0 to 1.0)
    pub accuracy: f32,
    /// Validity (0.0 to 1.0)
    pub validity: f32,
    /// Outlier percentage
    pub outlier_percentage: f32,
    /// Timeliness (0.0 to 1.0)
    pub timeliness: f32,
}

/// Model validation system
///
/// System for validating machine learning models using various
/// strategies and maintaining validation history.
pub struct ModelValidation {
    /// Validation strategies
    pub strategies: Arc<Mutex<Vec<Box<dyn ValidationStrategy + Send + Sync>>>>,
    /// Validation results cache
    pub results_cache: Arc<Mutex<HashMap<String, ValidationResult>>>,
    /// Validation history
    pub validation_history: Arc<Mutex<Vec<ValidationRecord>>>,
}

/// Validation strategy interface
///
/// Interface for different model validation strategies including
/// cross-validation, holdout validation, and custom methods.
pub trait ValidationStrategy {
    /// Validate the model
    fn validate(
        &self,
        model: &ModelState,
        validation_data: &[TrainingExample],
    ) -> Result<ValidationResult>;

    /// Get strategy name
    fn name(&self) -> &str;

    /// Check if strategy is applicable
    fn is_applicable(&self, model: &ModelState) -> bool;
}

/// Model validation result
///
/// Result of model validation including scores, metrics,
/// validation status, and detailed analysis.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Validation score
    pub score: f32,
    /// Validation metrics
    pub metrics: HashMap<String, f64>,
    /// Validation passed
    pub passed: bool,
    /// Validation timestamp
    pub timestamp: DateTime<Utc>,
    /// Validation method
    pub method: String,
    /// Validation details
    pub details: ValidationDetails,
    /// Strategy name used for validation
    pub strategy_name: String,
    /// Confidence level of validation result
    pub confidence: f32,
}

/// Detailed validation analysis
///
/// Detailed validation results including confusion matrix,
/// classification metrics, and ROC curve analysis.
#[derive(Debug, Clone, Default)]
pub struct ValidationDetails {
    /// True positives
    pub true_positives: usize,
    /// False positives
    pub false_positives: usize,
    /// True negatives
    pub true_negatives: usize,
    /// False negatives
    pub false_negatives: usize,
    /// Confusion matrix
    pub confusion_matrix: Vec<Vec<usize>>,
    /// ROC curve points
    pub roc_curve: Vec<(f32, f32)>,
    /// R-squared score for regression models
    pub r_squared: f32,
    /// Mean absolute error
    pub mean_absolute_error: f32,
    /// Root mean squared error
    pub root_mean_squared_error: f32,
    /// Cross-validation scores from k-fold validation
    pub cross_validation_scores: Vec<f32>,
}

/// Validation process record
///
/// Record of a model validation process including model version,
/// strategy used, results, and processing duration.
#[derive(Debug, Clone)]
pub struct ValidationRecord {
    /// Validation timestamp
    pub timestamp: DateTime<Utc>,
    /// Model version validated
    pub model_version: u64,
    /// Validation strategy used
    pub strategy: String,
    /// Validation result
    pub result: ValidationResult,
    /// Validation duration
    pub duration: Duration,
    /// Model name
    pub model_name: String,
    /// Dataset size
    pub dataset_size: usize,
    /// Strategies used
    pub strategies_used: Vec<String>,
    /// Results (plural)
    pub results: Vec<ValidationResult>,
}

/// Learning process history
///
/// Complete history of the learning process including training epochs,
/// model updates, and performance evolution over time.
#[derive(Debug, Default)]
pub struct LearningHistory {
    /// Training epochs
    pub training_epochs: Vec<TrainingEpoch>,
    /// Model updates
    pub model_updates: Vec<ModelUpdate>,
    /// Performance evolution
    pub performance_evolution: Vec<PerformanceSnapshot>,
    /// Learning rate history
    pub learning_rate_history: Vec<(DateTime<Utc>, f32)>,
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Event type
    pub event_type: String,
    /// Parameters before update
    pub parameters_before: HashMap<String, f64>,
    /// Parameters after update
    pub parameters_after: HashMap<String, f64>,
    /// Performance impact
    pub performance_impact: f64,
}

/// Individual training epoch record
///
/// Record of a single training epoch including loss, accuracy,
/// duration, and timestamp information.
#[derive(Debug, Clone)]
pub struct TrainingEpoch {
    /// Epoch number
    pub epoch: u64,
    /// Training loss
    pub training_loss: f64,
    /// Validation loss
    pub validation_loss: f64,
    /// Training accuracy
    pub training_accuracy: f32,
    /// Validation accuracy
    pub validation_accuracy: f32,
    /// Epoch duration
    pub duration: Duration,
    /// Epoch timestamp
    pub timestamp: DateTime<Utc>,
}

/// Model update record
///
/// Record of a model update including type, version changes,
/// reason, and performance impact assessment.
#[derive(Debug, Clone)]
pub struct ModelUpdate {
    /// Update timestamp
    pub timestamp: DateTime<Utc>,
    /// Update type
    pub update_type: ModelUpdateType,
    /// Previous model version
    pub previous_version: u64,
    /// New model version
    pub new_version: u64,
    /// Update reason
    pub reason: String,
    /// Performance impact
    pub performance_impact: Option<f32>,
}

/// Types of model updates
///
/// Classification of different types of model updates
/// and their characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelUpdateType {
    /// Incremental update
    Incremental,
    /// Full retrain
    FullRetrain,
    /// Parameter adjustment
    ParameterAdjustment,
    /// Architecture change
    ArchitectureChange,
}

// =============================================================================
// OPTIMIZATION HISTORY TYPES (7 types)
// =============================================================================

/// Complete optimization history tracking
///
/// Comprehensive tracking of optimization events, trends, effectiveness,
/// and statistical analysis for continuous improvement.
#[derive(Debug, Default)]
pub struct OptimizationHistory {
    /// Optimization events
    pub events: Vec<OptimizationEvent>,
    /// Performance trends
    pub trends: HashMap<String, PerformanceTrend>,
    /// Optimization effectiveness
    pub effectiveness: OptimizationEffectiveness,
    /// History statistics
    pub statistics: OptimizationStatistics,
}

/// Individual optimization event
///
/// Record of a single optimization event including type, description,
/// before/after performance, and metadata.
#[derive(Debug, Clone)]
pub struct OptimizationEvent {
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Event type
    pub event_type: OptimizationEventType,
    /// Event description
    pub description: String,
    /// Performance before
    pub performance_before: Option<PerformanceMeasurement>,
    /// Performance after
    pub performance_after: Option<PerformanceMeasurement>,
    /// Optimization parameters
    pub parameters: HashMap<String, String>,
    /// Event metadata
    pub metadata: HashMap<String, String>,
}

/// Types of optimization events
///
/// Classification of different optimization events that can
/// occur during system operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptimizationEventType {
    /// Parallelism adjustment
    ParallelismAdjustment,
    /// Resource reallocation
    ResourceReallocation,
    /// Algorithm change
    AlgorithmChange,
    /// Configuration update
    ConfigurationUpdate,
    /// Performance regression
    PerformanceRegression,
    /// Custom event
    Custom(String),
}

/// Optimization effectiveness tracking
///
/// Analysis of optimization effectiveness including success rates,
/// improvements, and best/worst optimization records.
#[derive(Debug, Default)]
pub struct OptimizationEffectiveness {
    /// Total optimizations applied
    pub total_optimizations: u64,
    /// Successful optimizations
    pub successful_optimizations: u64,
    /// Average performance improvement
    pub average_improvement: f32,
    /// Best optimization
    pub best_optimization: Option<OptimizationRecord>,
    /// Worst optimization
    pub worst_optimization: Option<OptimizationRecord>,
}

/// Individual optimization record
///
/// Record of a specific optimization including type, improvement,
/// timestamp, and duration for effectiveness analysis.
#[derive(Debug, Clone)]
pub struct OptimizationRecord {
    /// Optimization ID
    pub id: String,
    /// Optimization type
    pub optimization_type: OptimizationEventType,
    /// Performance improvement
    pub improvement: f32,
    /// Optimization timestamp
    pub timestamp: DateTime<Utc>,
    /// Optimization duration
    pub duration: Duration,
}

/// Optimization statistical analysis
///
/// Statistical analysis of optimization performance including frequency,
/// success rates, improvements, and type distributions.
#[derive(Debug, Default)]
pub struct OptimizationStatistics {
    /// Optimization frequency
    pub frequency: f32,
    /// Success rate
    pub success_rate: f32,
    /// Average improvement
    pub average_improvement: f32,
    /// Standard deviation of improvements
    pub improvement_std_dev: f32,
    /// Optimization types distribution
    pub type_distribution: HashMap<String, u64>,
}

// =============================================================================
// REAL-TIME METRICS TYPES (3 types)
// =============================================================================

/// Real-time system metrics
///
/// Current real-time metrics including parallelism level, throughput,
/// resource utilization, and collection metadata.
#[derive(Debug, Clone, Default)]
pub struct RealTimeMetrics {
    /// Current parallelism level
    pub current_parallelism: usize,
    /// Current throughput
    pub current_throughput: f64,
    /// Current latency
    pub current_latency: Duration,
    /// Current CPU utilization
    pub current_cpu_utilization: f32,
    /// Current memory utilization
    pub current_memory_utilization: f32,
    /// Current resource efficiency
    pub current_resource_efficiency: f32,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
    /// Metrics collection interval
    pub collection_interval: Duration,
    /// Alias for current_throughput
    pub throughput: f64,
    /// Alias for current_latency
    pub latency: Duration,
    /// Error rate
    pub error_rate: f32,
    /// Generic metric value
    pub value: f64,
    /// Metric type identifier
    pub metric_type: String,
    /// Resource usage information
    pub resource_usage:
        crate::performance_optimizer::test_characterization::types::resources::ResourceUsage,
    /// CPU utilization (alias without 'current_' prefix)
    pub cpu_utilization: f32,
    /// Memory utilization (alias without 'current_' prefix)
    pub memory_utilization: f32,
}

impl RealTimeMetrics {
    /// Get a metric value by key
    pub fn get(&self, key: &str) -> Option<f64> {
        match key {
            "throughput" | "current_throughput" => Some(self.current_throughput),
            "latency" | "current_latency" => Some(self.current_latency.as_secs_f64()),
            "cpu_utilization" | "current_cpu_utilization" => {
                Some(self.current_cpu_utilization as f64)
            },
            "memory_utilization" | "current_memory_utilization" => {
                Some(self.current_memory_utilization as f64)
            },
            "resource_efficiency" | "current_resource_efficiency" => {
                Some(self.current_resource_efficiency as f64)
            },
            "error_rate" => Some(self.error_rate as f64),
            "value" => Some(self.value),
            "parallelism" | "current_parallelism" => Some(self.current_parallelism as f64),
            _ => None,
        }
    }

    /// Get all metric values as a vector
    pub fn values(&self) -> Vec<f64> {
        vec![
            self.current_throughput,
            self.current_latency.as_secs_f64(),
            self.current_cpu_utilization as f64,
            self.current_memory_utilization as f64,
            self.current_resource_efficiency as f64,
            self.error_rate as f64,
            self.value,
            self.current_parallelism as f64,
        ]
    }
}

// =============================================================================
// DEFAULT IMPLEMENTATIONS
// =============================================================================

impl Default for AdaptiveParallelismConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_parallelism: 1,
            max_parallelism: num_cpus::get() * 2,
            adjustment_interval: Duration::from_secs(30),
            measurement_window: Duration::from_secs(60),
            learning_rate: 0.1,
            stability_threshold: 0.05,
            exploration_rate: 0.1,
            conservative_mode: false,
        }
    }
}

impl Default for DatasetSplitRatios {
    fn default() -> Self {
        Self {
            training: 0.7,
            validation: 0.2,
            test: 0.1,
        }
    }
}

impl Default for ResourceIntensity {
    fn default() -> Self {
        Self {
            cpu_intensity: 0.5,
            memory_intensity: 0.3,
            io_intensity: 0.2,
            network_intensity: 0.1,
            gpu_intensity: None,
        }
    }
}

impl Default for ResourceSharingCapabilities {
    fn default() -> Self {
        Self {
            cpu_sharing: true,
            memory_sharing: false,
            io_sharing: true,
            network_sharing: true,
            custom_sharing: HashMap::new(),
        }
    }
}

impl Default for ConvergenceStatus {
    fn default() -> Self {
        ConvergenceStatus::Unknown
    }
}

impl Default for ModelPerformanceMetrics {
    fn default() -> Self {
        Self {
            training_accuracy: 0.0,
            validation_accuracy: 0.0,
            test_accuracy: 0.0,
            loss: 0.0,
            convergence_status: ConvergenceStatus::NotConverged,
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            training_examples: 0,
            last_updated: Utc::now(),
        }
    }
}

impl Default for ModelState {
    fn default() -> Self {
        Self {
            parameters: HashMap::new(),
            weights: Vec::new(),
            bias: 0.0,
            version: 1,
            last_training: Utc::now(),
            performance_metrics: ModelPerformanceMetrics::default(),
            learning_rate: 0.01,
            accuracy: 0.5,
            last_updated: Utc::now(),
            training_examples_count: 0,
        }
    }
}

impl Default for TargetDistribution {
    fn default() -> Self {
        Self {
            distribution_type: DistributionType::Normal,
            parameters: HashMap::new(),
            goodness_of_fit: 0.0,
        }
    }
}

impl Default for TargetStatistics {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            distribution: TargetDistribution::default(),
        }
    }
}

impl Default for DataQualityMetrics {
    fn default() -> Self {
        Self {
            completeness: 1.0,
            consistency: 1.0,
            accuracy: 1.0,
            validity: 1.0,
            outlier_percentage: 0.0,
            timeliness: 1.0,
        }
    }
}

impl Default for DatasetStatistics {
    fn default() -> Self {
        Self {
            example_count: 0,
            feature_stats: Vec::new(),
            target_stats: TargetStatistics::default(),
            quality_metrics: DataQualityMetrics::default(),
            total_examples: 0,
            feature_statistics: Vec::new(),
            target_statistics: TargetStatistics::default(),
        }
    }
}

impl Default for TrainingDataset {
    fn default() -> Self {
        Self {
            examples: Vec::new(),
            split_ratios: DatasetSplitRatios::default(),
            statistics: DatasetStatistics::default(),
            version: 1,
            last_updated: Utc::now(),
            validation_split: 0.2,
        }
    }
}

impl Default for SynchronizationRequirements {
    fn default() -> Self {
        Self {
            exclusive_access: Vec::new(),
            ordered_execution: false,
            synchronization_points: Vec::new(),
            lock_dependencies: Vec::new(),
        }
    }
}

// Default for ConcurrencyRequirements is implemented in
// test_characterization::types::patterns module where the type is defined

impl Default for TestCharacteristics {
    fn default() -> Self {
        Self {
            category_distribution: HashMap::new(),
            average_duration: Duration::from_secs(1),
            resource_intensity: ResourceIntensity::default(),
            concurrency_requirements: ConcurrencyRequirements::default(),
            dependency_complexity: 0.0,
        }
    }
}

impl Default for SystemState {
    fn default() -> Self {
        Self {
            available_cores: num_cpus::get(),
            available_memory_mb: 8192,
            load_average: 0.0,
            active_processes: 0,
            io_wait_percent: 0.0,
            network_utilization: 0.0,
            temperature_metrics: None,
        }
    }
}

impl Default for PerformanceMeasurement {
    fn default() -> Self {
        Self {
            throughput: 0.0,
            average_latency: Duration::from_millis(100),
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            resource_efficiency: 0.0,
            timestamp: Utc::now(),
            measurement_duration: Duration::from_secs(1),
            cpu_usage: 0.0,
            memory_usage: 0.0,
            latency: Duration::from_millis(100),
        }
    }
}

impl Default for PerformanceDataPoint {
    fn default() -> Self {
        Self {
            parallelism: 1,
            throughput: 0.0,
            latency: Duration::from_millis(100),
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            resource_efficiency: 0.0,
            timestamp: Utc::now(),
            test_characteristics: TestCharacteristics::default(),
            system_state: SystemState::default(),
        }
    }
}

impl Default for ParallelismEstimate {
    fn default() -> Self {
        Self {
            optimal_parallelism: 1,
            confidence: 0.5,
            expected_improvement: 0.0,
            method: "default".to_string(),
            metadata: HashMap::new(),
        }
    }
}

impl Default for AggregatedFeedback {
    fn default() -> Self {
        Self {
            aggregated_value: 0.0,
            confidence: 0.0,
            contributing_count: 0,
            contributing_feedback_count: 0,
            aggregation_method: String::new(),
            timestamp: Utc::now(),
            recommended_actions: Vec::new(),
        }
    }
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self {
            score: 0.0,
            metrics: HashMap::new(),
            passed: false,
            timestamp: Utc::now(),
            method: String::new(),
            details: ValidationDetails::default(),
            strategy_name: String::new(),
            confidence: 0.5,
        }
    }
}

// Re-export commonly needed types from submodules for easy access
pub use crate::performance_optimizer::real_time_metrics::types::{
    CleanupPriority, DataPoint, ImpactArea, RiskType,
};
pub use crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel;
