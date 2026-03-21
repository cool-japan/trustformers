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

// Re-export types moved to types_ml module for backward compatibility
pub use super::types_ml::*;

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
