//! Mobile Performance Profiler Types
//!
//! This module contains all the core types, configuration structures, metrics definitions,
//! and enums used throughout the mobile performance profiler system. It provides a comprehensive
//! foundation for mobile ML inference performance monitoring, debugging, and optimization.
//!
//! # Organization
//!
//! The types are organized into several logical groups:
//!
//! - **Configuration Types**: All configuration structures for profiler setup
//! - **Profiling Enums**: Core enums for modes, types, and classifications
//! - **Metrics Types**: All metrics data structures for performance measurement
//! - **Core Data Types**: Main operational data structures
//! - **Platform-specific Types**: iOS and Android specific structures
//!
//! # Usage
//!
//! ```rust
//! use trustformers_mobile::mobile_performance_profiler::types::*;
//!
//! // Create a profiler configuration
//! let config = MobileProfilerConfig::default();
//!
//! // Create a metrics snapshot
//! let snapshot = MobileMetricsSnapshot::default();
//! ```

use crate::device_info::ThermalState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// =============================================================================
// CONFIGURATION TYPES
// =============================================================================

/// Configuration for data sampling behavior
///
/// Controls how frequently metrics are collected and how much historical
/// data is retained for analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Sampling interval in milliseconds
    pub interval_ms: u64,
    /// Maximum number of samples to keep in memory
    pub max_samples: usize,
    /// Enable adaptive sampling based on system load
    pub adaptive_sampling: bool,
    /// High frequency sampling threshold in milliseconds
    pub high_freq_threshold_ms: u64,
    /// Low frequency sampling threshold in milliseconds
    pub low_freq_threshold_ms: u64,
}

/// Configuration for memory profiling features
///
/// Controls which memory-related metrics and analysis features are enabled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProfilingConfig {
    /// Enable memory profiling
    pub enabled: bool,
    /// Track memory allocations
    pub track_allocations: bool,
    /// Track memory deallocations
    pub track_deallocations: bool,
    /// Enable memory leak detection
    pub leak_detection: bool,
    /// Monitor memory pressure events
    pub pressure_monitoring: bool,
    /// Enable detailed heap analysis (expensive)
    pub heap_analysis: bool,
    /// Stack trace depth for allocation tracking
    pub stack_trace_depth: usize,
}

/// Configuration for CPU profiling features
///
/// Controls CPU performance monitoring, thermal tracking, and core utilization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuProfilingConfig {
    /// Enable CPU profiling
    pub enabled: bool,
    /// Track CPU usage per thread
    pub per_thread_tracking: bool,
    /// Monitor thermal throttling
    pub thermal_monitoring: bool,
    /// Monitor CPU frequency changes
    pub frequency_monitoring: bool,
    /// Track individual core utilization
    pub core_utilization: bool,
    /// Estimate power consumption
    pub power_estimation: bool,
}

/// Configuration for GPU profiling features
///
/// Controls GPU performance monitoring, memory tracking, and thermal management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuProfilingConfig {
    /// Enable GPU profiling
    pub enabled: bool,
    /// Track GPU memory usage
    pub memory_tracking: bool,
    /// Monitor GPU utilization
    pub utilization_monitoring: bool,
    /// Track shader performance (expensive)
    pub shader_tracking: bool,
    /// Monitor GPU thermal status
    pub thermal_monitoring: bool,
    /// Track GPU power consumption
    pub power_tracking: bool,
}

/// Configuration for network profiling features
///
/// Controls network performance monitoring, bandwidth tracking, and latency analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkProfilingConfig {
    /// Enable network profiling
    pub enabled: bool,
    /// Track bandwidth usage
    pub bandwidth_tracking: bool,
    /// Monitor network latency
    pub latency_monitoring: bool,
    /// Monitor connection pool status
    pub connection_monitoring: bool,
    /// Analyze request/response patterns
    pub request_analysis: bool,
    /// Track network error rates
    pub error_tracking: bool,
}

/// Configuration for real-time monitoring
///
/// Controls real-time updates, alerts, and live performance tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeConfig {
    /// Enable real-time monitoring
    pub enabled: bool,
    /// Update interval for real-time data in milliseconds
    pub update_interval_ms: u64,
    /// Enable performance alerts
    pub performance_alerts: bool,
    /// Alert threshold configuration
    pub alert_thresholds: AlertThresholds,
    /// Maximum history points for real-time display
    pub max_history_points: usize,
}

/// Alert thresholds for real-time monitoring
///
/// Defines the threshold values that trigger performance alerts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Memory usage threshold percentage (0-100)
    pub memory_threshold_percent: f32,
    /// CPU usage threshold percentage (0-100)
    pub cpu_threshold_percent: f32,
    /// GPU usage threshold percentage (0-100)
    pub gpu_threshold_percent: f32,
    /// Temperature threshold in Celsius
    pub temperature_threshold_c: f32,
    /// Inference latency threshold in milliseconds
    pub latency_threshold_ms: f32,
}

/// Configuration for data export
///
/// Controls how profiling data is exported and saved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Enable automatic export of profiling data
    pub auto_export: bool,
    /// Export format (JSON, CSV, HTML, etc.)
    pub format: ExportFormat,
    /// Directory for exported files
    pub export_directory: String,
    /// Include raw metrics data in export
    pub include_raw_data: bool,
    /// Include visualization charts in export
    pub include_visualizations: bool,
    /// Compression level for exported data (0-9)
    pub compression_level: u8,
}

// =============================================================================
// PROFILING ENUMS
// =============================================================================

/// Profiling modes that determine the level of detail and performance impact
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProfilingMode {
    /// Development mode with detailed profiling for debugging
    Development,
    /// Production mode with lightweight profiling
    Production,
    /// Debug mode with maximum detail and overhead
    Debug,
    /// Benchmark mode focused on performance measurement
    Benchmark,
    /// Custom mode with user-defined configuration
    Custom,
}

/// Types of performance bottlenecks that can be detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BottleneckType {
    /// Memory-related bottlenecks
    Memory,
    /// CPU-related bottlenecks
    CPU,
    /// GPU-related bottlenecks
    GPU,
    /// Network-related bottlenecks
    Network,
    /// Latency-related bottlenecks
    Latency,
    /// Thermal throttling bottlenecks
    Thermal,
    /// Power consumption bottlenecks
    Power,
    /// Cache performance bottlenecks
    Cache,
}

/// Severity levels for performance bottlenecks
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BottleneckSeverity {
    /// Low severity issues
    Low,
    /// Medium severity issues
    Medium,
    /// High severity issues requiring attention
    High,
    /// Critical issues requiring immediate action
    Critical,
}

/// Types of optimization suggestions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuggestionType {
    /// Model architecture optimizations
    ModelOptimization,
    /// Cache strategy optimizations
    CacheOptimization,
    /// General performance optimizations
    PerformanceOptimization,
    /// Hardware utilization optimizations
    HardwareOptimization,
    /// Network optimization suggestions
    NetworkOptimization,
    /// Power efficiency optimizations
    PowerOptimization,
}

/// Implementation difficulty levels for optimization suggestions
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DifficultyLevel {
    /// Easy to implement
    Low,
    /// Moderate implementation effort
    Medium,
    /// Complex implementation required
    High,
}

/// Priority levels for optimization suggestions
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PriorityLevel {
    /// Low priority optimization
    Low,
    /// Medium priority optimization
    Medium,
    /// High priority optimization
    High,
    /// Critical priority requiring immediate attention
    Critical,
}

/// Export formats for profiling data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// JSON format
    JSON,
    /// CSV format
    CSV,
    /// HTML report format
    HTML,
    /// SQLite database format
    SQLite,
    /// Parquet columnar format
    Parquet,
}

/// Types of profiling events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    /// ML inference started
    InferenceStart,
    /// ML inference completed
    InferenceEnd,
    /// Model loading started
    ModelLoad,
    /// Model unloading started
    ModelUnload,
    /// Memory allocation event
    MemoryAllocation,
    /// Memory deallocation event
    MemoryDeallocation,
    /// Thermal throttling event
    ThermalThrottle,
    /// Power management event
    PowerEvent,
    /// Network request initiated
    NetworkRequest,
    /// Network response received
    NetworkResponse,
    /// GPU operation executed
    GPUOperation,
    /// Cache hit event
    CacheHit,
    /// Cache miss event
    CacheMiss,
}

/// Temperature trend directions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemperatureTrend {
    /// Temperature is stable
    Stable,
    /// Temperature is rising
    Rising,
    /// Temperature is falling
    Falling,
    /// Temperature is critically high
    Critical,
}

/// Impact levels for performance issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImpactLevel {
    /// Low impact on performance
    Low,
    /// Medium impact on performance
    Medium,
    /// High impact on performance
    High,
    /// Critical impact on performance
    Critical,
}

/// Chart types for data visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChartType {
    /// Line chart for time series data
    LineChart,
    /// Bar chart for categorical data
    BarChart,
    /// Heat map for correlation data
    HeatMap,
    /// Histogram for distribution data
    Histogram,
    /// Scatter plot for relationship data
    ScatterPlot,
    /// Timeline for event data
    Timeline,
}

/// Types of performance alerts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertType {
    /// Memory pressure alert
    MemoryPressure,
    /// Thermal throttling alert
    ThermalThrottling,
    /// Battery drain alert
    BatteryDrain,
    /// Performance degradation alert
    PerformanceDegradation,
    /// Network issue alert
    NetworkIssue,
    /// Resource contention alert
    ResourceContention,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning alert
    Warning,
    /// Error alert
    Error,
    /// Critical alert requiring immediate attention
    Critical,
}

/// Trend directions for metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Performance is improving
    Improving,
    /// Performance is stable
    Stable,
    /// Performance is degrading
    Degrading,
}

/// System health status levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Excellent system health
    Excellent,
    /// Good system health
    Good,
    /// Healthy system with no significant issues
    Healthy,
    /// Fair system health with minor issues
    Fair,
    /// Poor system health with significant issues
    Poor,
    /// Critical system health requiring immediate attention
    Critical,
}

// =============================================================================
// METRICS TYPES
// =============================================================================

/// ML inference performance metrics
///
/// Tracks the performance of machine learning inference operations including
/// latency, throughput, success rates, and caching effectiveness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetrics {
    /// Total number of inferences performed
    pub total_inferences: u64,
    /// Number of successful inferences
    pub successful_inferences: u64,
    /// Number of failed inferences
    pub failed_inferences: u64,
    /// Average inference latency in milliseconds
    pub avg_latency_ms: f64,
    /// Minimum observed latency in milliseconds
    pub min_latency_ms: f64,
    /// Maximum observed latency in milliseconds
    pub max_latency_ms: f64,
    /// Throughput in inferences per second
    pub throughput_per_sec: f64,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,
    /// Model loading time in milliseconds
    pub model_load_time_ms: f64,
}

/// Memory usage metrics
///
/// Comprehensive memory usage tracking across different memory types
/// including heap, native, graphics, and system memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Heap memory used in MB
    pub heap_used_mb: f32,
    /// Heap memory free in MB
    pub heap_free_mb: f32,
    /// Total heap memory in MB
    pub heap_total_mb: f32,
    /// Native memory used in MB
    pub native_used_mb: f32,
    /// Graphics memory used in MB
    pub graphics_used_mb: f32,
    /// Code memory used in MB
    pub code_used_mb: f32,
    /// Stack memory used in MB
    pub stack_used_mb: f32,
    /// Other memory used in MB
    pub other_used_mb: f32,
    /// Available system memory in MB
    pub available_mb: f32,
}

/// CPU performance metrics
///
/// Tracks CPU utilization, frequency, temperature, and throttling status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuMetrics {
    /// Overall CPU usage percentage (0-100)
    pub usage_percent: f32,
    /// User space CPU usage percentage
    pub user_percent: f32,
    /// System/kernel CPU usage percentage
    pub system_percent: f32,
    /// CPU idle percentage
    pub idle_percent: f32,
    /// Current CPU frequency in MHz
    pub frequency_mhz: u32,
    /// CPU temperature in Celsius
    pub temperature_c: f32,
    /// Thermal throttling level (0.0 to 1.0)
    pub throttling_level: f32,
}

/// GPU performance metrics
///
/// Tracks GPU utilization, memory usage, frequency, temperature, and power consumption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    /// GPU usage percentage (0-100)
    pub usage_percent: f32,
    /// GPU memory used in MB
    pub memory_used_mb: f32,
    /// Total GPU memory in MB
    pub memory_total_mb: f32,
    /// Current GPU frequency in MHz
    pub frequency_mhz: u32,
    /// GPU temperature in Celsius
    pub temperature_c: f32,
    /// GPU power consumption in milliwatts
    pub power_mw: f32,
}

/// Network performance metrics
///
/// Tracks network bandwidth, latency, connections, and error rates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Total bytes received
    pub bytes_received: u64,
    /// Total packets sent
    pub packets_sent: u64,
    /// Total packets received
    pub packets_received: u64,
    /// Number of active connections
    pub connection_count: u32,
    /// Network latency in milliseconds
    pub latency_ms: f64,
    /// Bandwidth in Mbps
    pub bandwidth_mbps: f64,
    /// Network error rate (0.0 to 1.0)
    pub error_rate: f64,
}

/// Thermal performance metrics
///
/// Tracks device temperature, thermal state, throttling, and temperature trends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalMetrics {
    /// Current device temperature in Celsius
    pub temperature_c: f32,
    /// Thermal state from the system
    pub thermal_state: ThermalState,
    /// Current throttling level (0.0 to 1.0)
    pub throttling_level: f32,
    /// Temperature trend direction
    pub temperature_trend: TemperatureTrend,
}

/// Battery performance metrics
///
/// Tracks battery level, charging status, power consumption, and estimated battery life.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatteryMetrics {
    /// Battery level percentage (0-100)
    pub level_percent: u8,
    /// Whether the device is currently charging
    pub is_charging: bool,
    /// Current power consumption in milliwatts
    pub power_consumption_mw: f32,
    /// Estimated battery life in minutes
    pub estimated_life_minutes: u32,
}

/// Platform-specific metrics container
///
/// Contains metrics that are specific to iOS or Android platforms.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PlatformMetrics {
    /// iOS-specific metrics (only available on iOS)
    #[cfg(target_os = "ios")]
    pub ios: Option<IOSMetrics>,
    /// Android-specific metrics (only available on Android)
    #[cfg(target_os = "android")]
    pub android: Option<AndroidMetrics>,
}

/// iOS-specific performance metrics
///
/// Contains metrics specific to iOS devices including Metal and Core ML performance.
#[cfg(target_os = "ios")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IOSMetrics {
    /// Metal GPU performance statistics
    pub metal_stats: MetalPerformanceStats,
    /// Core ML inference performance statistics
    pub coreml_stats: CoreMLPerformanceStats,
    /// iOS memory pressure information
    pub memory_pressure: IOSMemoryPressure,
}

/// Android-specific performance metrics
///
/// Contains metrics specific to Android devices including NNAPI and GPU delegate performance.
#[cfg(target_os = "android")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AndroidMetrics {
    /// NNAPI performance statistics
    pub nnapi_stats: NNAPIPerformanceStats,
    /// GPU delegate performance statistics
    pub gpu_delegate_stats: GPUDelegateStats,
    /// Android memory management statistics
    pub memory_stats: AndroidMemoryStats,
    /// Android Doze mode status
    pub doze_status: DozeStatus,
}

/// Complete snapshot of all metrics at a specific point in time
///
/// This is the primary metrics container that includes all performance data
/// collected during a single sampling interval.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MobileMetricsSnapshot {
    /// Timestamp when metrics were collected (Unix timestamp in milliseconds)
    pub timestamp: u64,
    /// Memory usage metrics
    pub memory: MemoryMetrics,
    /// CPU performance metrics
    pub cpu: CpuMetrics,
    /// GPU performance metrics
    pub gpu: GpuMetrics,
    /// Network performance metrics
    pub network: NetworkMetrics,
    /// ML inference metrics
    pub inference: InferenceMetrics,
    /// Thermal performance metrics
    pub thermal: ThermalMetrics,
    /// Battery performance metrics
    pub battery: BatteryMetrics,
    /// Platform-specific metrics
    pub platform: PlatformMetrics,
}

/// Event-specific metrics
///
/// Metrics captured at the time of a specific profiling event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMetrics {
    /// Memory usage at event time in MB
    pub memory_usage_mb: f32,
    /// CPU usage at event time (percentage)
    pub cpu_usage_percent: f32,
    /// GPU usage at event time (percentage)
    pub gpu_usage_percent: f32,
    /// Temperature at event time in Celsius
    pub temperature_c: f32,
    /// Battery level at event time (percentage)
    pub battery_percent: u8,
}

// =============================================================================
// CORE DATA TYPES
// =============================================================================

/// Performance bottleneck description
///
/// Represents a detected performance issue with details about its impact
/// and suggested remediation steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    /// Type of bottleneck detected
    pub bottleneck_type: BottleneckType,
    /// Severity level of the bottleneck
    pub severity: BottleneckSeverity,
    /// Human-readable description of the issue
    pub description: String,
    /// Component or system affected by the bottleneck
    pub affected_component: String,
    /// Impact score (0.0 to 100.0)
    pub impact_score: f32,
    /// List of suggested remediation steps
    pub suggestions: Vec<String>,
    /// Timestamp when bottleneck was detected
    pub timestamp: u64,
}

/// Optimization suggestion
///
/// Detailed recommendation for improving performance with implementation
/// guidance and expected impact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    /// Type of optimization suggested
    pub suggestion_type: SuggestionType,
    /// Brief title of the suggestion
    pub title: String,
    /// Detailed description of the optimization
    pub description: String,
    /// Step-by-step implementation guide
    pub implementation_steps: Vec<String>,
    /// Estimated performance improvement
    pub estimated_improvement: String,
    /// Implementation difficulty level
    pub difficulty: DifficultyLevel,
    /// Priority level for implementation
    pub priority: PriorityLevel,
}

/// Session metadata
///
/// Contains information about the profiling session context including
/// application and device details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// Unique session identifier
    pub session_id: String,
    /// Session start time
    pub start_time: u64,
    /// Session end time
    pub end_time: Option<u64>,
    /// Application version being profiled
    pub app_version: String,
    /// Build configuration (debug/release)
    pub build_config: String,
    /// Device model identifier
    pub device_model: String,
    /// Operating system version
    pub os_version: String,
    /// Available memory at session start in MB
    pub initial_memory_mb: u32,
    /// Battery level at session start (percentage)
    pub initial_battery_percent: u8,
    /// Thermal state at session start
    pub initial_thermal_state: ThermalState,
}

/// Individual profiling event
///
/// Represents a specific event that occurred during profiling with
/// associated metadata and metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingEvent {
    /// Unique event identifier
    pub event_id: String,
    /// Event timestamp (Unix timestamp in milliseconds)
    pub timestamp: u64,
    /// Type of event that occurred
    pub event_type: EventType,
    /// Event category for organization
    pub category: String,
    /// Human-readable event description
    pub description: String,
    /// Event-specific data payload
    pub data: EventData,
    /// Event metadata
    pub metadata: HashMap<String, String>,
    /// Event tags for filtering and searching
    pub tags: Vec<String>,
    /// Thread ID where event occurred
    pub thread_id: u64,
    /// Event duration in milliseconds (if applicable)
    pub duration_ms: Option<f64>,
}

/// Event data payload
///
/// Contains the specific data and metrics associated with a profiling event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventData {
    /// Event-specific key-value data
    pub payload: HashMap<String, String>,
    /// Associated performance metrics (if available)
    pub metrics: Option<EventMetrics>,
}

/// Real-time monitoring state
///
/// Current state of the real-time performance monitoring system including
/// alerts, trends, and health status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeState {
    /// Current overall performance score (0.0 to 100.0)
    pub performance_score: f32,
    /// List of currently active performance alerts
    pub active_alerts: Vec<PerformanceAlert>,
    /// Trending metrics data
    pub trending_metrics: TrendingMetrics,
    /// Overall system health assessment
    pub system_health: SystemHealth,
}

/// Performance alert
///
/// Represents an active or historical performance alert with details
/// about the issue and recommended actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    /// Unique alert identifier
    pub id: String,
    /// Type of alert
    pub alert_type: AlertType,
    /// Severity level of the alert
    pub severity: AlertSeverity,
    /// Human-readable alert message
    pub message: String,
    /// Timestamp when alert was triggered
    pub timestamp: u64,
    /// Suggested action to resolve the alert
    pub suggested_action: String,
}

/// Trending metrics container
///
/// Contains trend information for key performance metrics over time.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrendingMetrics {
    /// Memory usage trend
    pub memory_trend: MetricTrend,
    /// CPU usage trend
    pub cpu_trend: MetricTrend,
    /// Inference latency trend
    pub latency_trend: MetricTrend,
    /// Temperature trend
    pub temperature_trend: MetricTrend,
}

/// Individual metric trend
///
/// Represents the trend data for a specific performance metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricTrend {
    /// Current metric value
    pub current: f32,
    /// Previous metric value for comparison
    pub previous: f32,
    /// Trend direction (improving, stable, degrading)
    pub direction: TrendDirection,
    /// Magnitude of change
    pub magnitude: f32,
}

/// System health assessment
///
/// Overall assessment of system health with component-level details
/// and recommendations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    /// Overall health score (0.0 to 100.0)
    pub overall_score: f32,
    /// Health scores for individual components
    pub component_scores: HashMap<String, f32>,
    /// Overall health status classification
    pub status: HealthStatus,
    /// Health-related recommendations
    pub recommendations: Vec<String>,
}

/// Complete profiling data container
///
/// Contains all profiling data collected during a session including
/// metrics, events, analysis results, and summaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingData {
    /// Session information and metadata
    pub session_info: SessionInfo,
    /// All collected metrics snapshots
    pub metrics: Vec<MobileMetricsSnapshot>,
    /// All profiling events that occurred
    pub events: Vec<ProfilingEvent>,
    /// Detected performance bottlenecks
    pub bottlenecks: Vec<PerformanceBottleneck>,
    /// Generated optimization suggestions
    pub suggestions: Vec<OptimizationSuggestion>,
    /// Statistical summary of the session
    pub summary: ProfilingSummary,
    /// Overall system health assessment
    pub system_health: SystemHealth,
    /// Timestamp when data was exported
    pub export_timestamp: u64,
    /// Version of the profiler that collected the data
    pub profiler_version: String,
}

/// Session information
///
/// High-level information about a profiling session including timing
/// and device context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    /// Unique session identifier
    pub id: String,
    /// Session start time (Unix timestamp in milliseconds)
    pub start_time: u64,
    /// Session end time (Unix timestamp in milliseconds)
    pub end_time: Option<u64>,
    /// Total session duration in milliseconds
    pub duration_ms: Option<u64>,
    /// Device information for the session
    pub device_info: crate::device_info::MobileDeviceInfo,
    /// Additional session metadata
    pub metadata: SessionMetadata,
}

/// Profiling summary statistics
///
/// Statistical summary of a profiling session with key performance indicators
/// and aggregate metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingSummary {
    /// Total number of inferences performed
    pub total_inferences: u64,
    /// Total number of profiling events recorded
    pub total_events: u64,
    /// Total number of bottlenecks detected
    pub total_bottlenecks: u64,
    /// Total session duration in milliseconds
    pub session_duration_ms: u64,
    /// Average inference time in milliseconds
    pub avg_inference_time_ms: f64,
    /// Peak memory usage in MB
    pub peak_memory_mb: f32,
    /// Average CPU usage percentage
    pub avg_cpu_usage_percent: f32,
    /// Average GPU usage percentage
    pub avg_gpu_usage_percent: f32,
    /// Total battery consumed in mAh
    pub battery_consumed_mah: f32,
    /// Number of thermal events detected
    pub thermal_events: u32,
    /// Overall performance score (0.0 to 100.0)
    pub performance_score: f32,
}

// =============================================================================
// PLATFORM-SPECIFIC TYPES
// =============================================================================

/// iOS Metal performance statistics
///
/// Performance metrics specific to Metal GPU operations on iOS.
#[cfg(target_os = "ios")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalPerformanceStats {
    /// Metal GPU utilization percentage
    pub gpu_utilization: f32,
    /// Metal memory usage in MB
    pub memory_usage_mb: f32,
    /// Number of Metal command buffers processed
    pub command_buffers: u64,
    /// Metal shader compilation time in milliseconds
    pub shader_compile_time_ms: f64,
}

/// iOS Core ML performance statistics
///
/// Performance metrics specific to Core ML inference on iOS.
#[cfg(target_os = "ios")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMLPerformanceStats {
    /// Core ML inference count
    pub inference_count: u64,
    /// Average Core ML inference time in milliseconds
    pub avg_inference_time_ms: f64,
    /// Core ML model loading time in milliseconds
    pub model_load_time_ms: f64,
    /// Core ML compute unit utilization
    pub compute_unit_utilization: f32,
}

/// iOS memory pressure information
///
/// iOS-specific memory pressure and management data.
#[cfg(target_os = "ios")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IOSMemoryPressure {
    /// Memory pressure level (0-3)
    pub pressure_level: u8,
    /// Available memory in MB
    pub available_memory_mb: f32,
    /// Memory warning count
    pub memory_warnings: u32,
    /// Jetsam event count
    pub jetsam_events: u32,
}

/// Android NNAPI performance statistics
///
/// Performance metrics specific to NNAPI operations on Android.
#[cfg(target_os = "android")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NNAPIPerformanceStats {
    /// NNAPI execution count
    pub execution_count: u64,
    /// Average NNAPI execution time in milliseconds
    pub avg_execution_time_ms: f64,
    /// NNAPI compilation time in milliseconds
    pub compilation_time_ms: f64,
    /// NNAPI driver version
    pub driver_version: String,
}

/// Android GPU delegate performance statistics
///
/// Performance metrics for GPU delegate operations on Android.
#[cfg(target_os = "android")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUDelegateStats {
    /// GPU delegate initialization time in milliseconds
    pub init_time_ms: f64,
    /// Number of operations delegated to GPU
    pub delegated_operations: u32,
    /// GPU delegate inference time in milliseconds
    pub inference_time_ms: f64,
    /// GPU memory allocation in MB
    pub gpu_memory_mb: f32,
}

/// Android memory management statistics
///
/// Android-specific memory management and garbage collection data.
#[cfg(target_os = "android")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AndroidMemoryStats {
    /// Heap size in MB
    pub heap_size_mb: f32,
    /// Heap allocation rate in MB/s
    pub heap_alloc_rate_mbs: f32,
    /// Garbage collection count
    pub gc_count: u32,
    /// Total garbage collection time in milliseconds
    pub gc_time_ms: u64,
}

/// Android Doze mode status
///
/// Information about Android's Doze mode and app standby status.
#[cfg(target_os = "android")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DozeStatus {
    /// Whether the device is in Doze mode
    pub in_doze_mode: bool,
    /// Whether the app is in standby mode
    pub app_standby: bool,
    /// Battery optimization enabled
    pub battery_optimization: bool,
    /// Background activity restricted
    pub background_restricted: bool,
}

// =============================================================================
// DEFAULT IMPLEMENTATIONS
// =============================================================================

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            interval_ms: 100,
            max_samples: 10000,
            adaptive_sampling: true,
            high_freq_threshold_ms: 50,
            low_freq_threshold_ms: 1000,
        }
    }
}

impl Default for MemoryProfilingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            track_allocations: true,
            track_deallocations: true,
            leak_detection: true,
            pressure_monitoring: true,
            heap_analysis: false, // Expensive operation
            stack_trace_depth: 10,
        }
    }
}

impl Default for CpuProfilingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            per_thread_tracking: true,
            thermal_monitoring: true,
            frequency_monitoring: false, // Platform dependent
            core_utilization: true,
            power_estimation: true,
        }
    }
}

impl Default for GpuProfilingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            memory_tracking: true,
            utilization_monitoring: true,
            shader_tracking: false, // Expensive operation
            thermal_monitoring: true,
            power_tracking: true,
        }
    }
}

impl Default for NetworkProfilingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            bandwidth_tracking: true,
            latency_monitoring: true,
            connection_monitoring: true,
            request_analysis: true,
            error_tracking: true,
        }
    }
}

impl Default for RealTimeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            update_interval_ms: 1000,
            performance_alerts: true,
            alert_thresholds: AlertThresholds::default(),
            max_history_points: 100,
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            memory_threshold_percent: 80.0,
            cpu_threshold_percent: 85.0,
            gpu_threshold_percent: 90.0,
            temperature_threshold_c: 40.0,
            latency_threshold_ms: 100.0,
        }
    }
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            auto_export: false,
            format: ExportFormat::JSON,
            export_directory: std::env::temp_dir()
                .join("trustformers_profiling")
                .to_string_lossy()
                .to_string(),
            include_raw_data: true,
            include_visualizations: false,
            compression_level: 6,
        }
    }
}

impl Default for MemoryMetrics {
    fn default() -> Self {
        Self {
            heap_used_mb: 0.0,
            heap_free_mb: 0.0,
            heap_total_mb: 0.0,
            native_used_mb: 0.0,
            graphics_used_mb: 0.0,
            code_used_mb: 0.0,
            stack_used_mb: 0.0,
            other_used_mb: 0.0,
            available_mb: 0.0,
        }
    }
}

impl Default for CpuMetrics {
    fn default() -> Self {
        Self {
            usage_percent: 0.0,
            user_percent: 0.0,
            system_percent: 0.0,
            idle_percent: 100.0,
            frequency_mhz: 0,
            temperature_c: 0.0,
            throttling_level: 0.0,
        }
    }
}

impl Default for GpuMetrics {
    fn default() -> Self {
        Self {
            usage_percent: 0.0,
            memory_used_mb: 0.0,
            memory_total_mb: 0.0,
            frequency_mhz: 0,
            temperature_c: 0.0,
            power_mw: 0.0,
        }
    }
}

impl Default for NetworkMetrics {
    fn default() -> Self {
        Self {
            bytes_sent: 0,
            bytes_received: 0,
            packets_sent: 0,
            packets_received: 0,
            connection_count: 0,
            latency_ms: 0.0,
            bandwidth_mbps: 0.0,
            error_rate: 0.0,
        }
    }
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self {
            total_inferences: 0,
            successful_inferences: 0,
            failed_inferences: 0,
            avg_latency_ms: 0.0,
            min_latency_ms: 0.0,
            max_latency_ms: 0.0,
            throughput_per_sec: 0.0,
            cache_hit_rate: 0.0,
            model_load_time_ms: 0.0,
        }
    }
}

impl Default for ThermalMetrics {
    fn default() -> Self {
        Self {
            temperature_c: 0.0,
            thermal_state: ThermalState::Nominal,
            throttling_level: 0.0,
            temperature_trend: TemperatureTrend::Stable,
        }
    }
}

impl Default for BatteryMetrics {
    fn default() -> Self {
        Self {
            level_percent: 100,
            is_charging: false,
            power_consumption_mw: 0.0,
            estimated_life_minutes: 0,
        }
    }
}

#[cfg(target_os = "ios")]
impl Default for MetalPerformanceStats {
    fn default() -> Self {
        Self {
            gpu_utilization: 0.0,
            memory_usage_mb: 0.0,
            command_buffers: 0,
            shader_compile_time_ms: 0.0,
        }
    }
}

#[cfg(target_os = "ios")]
impl Default for CoreMLPerformanceStats {
    fn default() -> Self {
        Self {
            inference_count: 0,
            avg_inference_time_ms: 0.0,
            model_load_time_ms: 0.0,
            compute_unit_utilization: 0.0,
        }
    }
}

#[cfg(target_os = "ios")]
impl Default for IOSMemoryPressure {
    fn default() -> Self {
        Self {
            pressure_level: 0,
            available_memory_mb: 0.0,
            memory_warnings: 0,
            jetsam_events: 0,
        }
    }
}

#[cfg(target_os = "android")]
impl Default for NNAPIPerformanceStats {
    fn default() -> Self {
        Self {
            execution_count: 0,
            avg_execution_time_ms: 0.0,
            compilation_time_ms: 0.0,
            driver_version: String::new(),
        }
    }
}

#[cfg(target_os = "android")]
impl Default for GPUDelegateStats {
    fn default() -> Self {
        Self {
            init_time_ms: 0.0,
            delegated_operations: 0,
            inference_time_ms: 0.0,
            gpu_memory_mb: 0.0,
        }
    }
}

#[cfg(target_os = "android")]
impl Default for AndroidMemoryStats {
    fn default() -> Self {
        Self {
            heap_size_mb: 0.0,
            heap_alloc_rate_mbs: 0.0,
            gc_count: 0,
            gc_time_ms: 0,
        }
    }
}

#[cfg(target_os = "android")]
impl Default for DozeStatus {
    fn default() -> Self {
        Self {
            in_doze_mode: false,
            app_standby: false,
            battery_optimization: false,
            background_restricted: false,
        }
    }
}

impl Default for SessionMetadata {
    fn default() -> Self {
        Self {
            session_id: String::new(),
            start_time: 0,
            end_time: None,
            app_version: String::new(),
            build_config: "release".to_string(),
            device_model: String::new(),
            os_version: String::new(),
            initial_memory_mb: 0,
            initial_battery_percent: 100,
            initial_thermal_state: ThermalState::Nominal,
        }
    }
}

impl Default for ProfilingSummary {
    fn default() -> Self {
        Self {
            total_inferences: 0,
            total_events: 0,
            total_bottlenecks: 0,
            session_duration_ms: 0,
            avg_inference_time_ms: 0.0,
            peak_memory_mb: 0.0,
            avg_cpu_usage_percent: 0.0,
            avg_gpu_usage_percent: 0.0,
            battery_consumed_mah: 0.0,
            thermal_events: 0,
            performance_score: 0.0,
        }
    }
}

impl Default for MetricTrend {
    fn default() -> Self {
        Self {
            current: 0.0,
            previous: 0.0,
            direction: TrendDirection::Stable,
            magnitude: 0.0,
        }
    }
}

impl Default for SystemHealth {
    fn default() -> Self {
        Self {
            overall_score: 0.0,
            component_scores: HashMap::new(),
            status: HealthStatus::Good,
            recommendations: Vec::new(),
        }
    }
}

// =============================================================================
// PROFILER COMPONENT CONFIGURATION TYPES
// =============================================================================

/// Bottleneck detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckDetectionConfig {
    /// Enable detection
    pub enabled: bool,
    /// Detection sensitivity
    pub sensitivity: f32,
    /// Detection interval
    pub detection_interval_ms: u64,
    /// Minimum impact threshold
    pub min_impact_threshold: f32,
}

impl Default for BottleneckDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sensitivity: 0.7,
            detection_interval_ms: 5000,
            min_impact_threshold: 0.1,
        }
    }
}

/// Optimization engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationEngineConfig {
    /// Enable optimization suggestions
    pub enabled: bool,
    /// Suggestion generation interval
    pub generation_interval_ms: u64,
    /// Maximum suggestions to keep
    pub max_suggestions: usize,
    /// Minimum improvement threshold
    pub min_improvement_threshold: f32,
}

impl Default for OptimizationEngineConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            generation_interval_ms: 30000,
            max_suggestions: 10,
            min_improvement_threshold: 0.05,
        }
    }
}

/// Real-time monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeMonitoringConfig {
    /// Enable real-time monitoring
    pub enabled: bool,
    /// Update frequency
    pub update_frequency_ms: u64,
    /// Alert evaluation interval
    pub alert_interval_ms: u64,
    /// Maximum alerts to keep
    pub max_alerts: usize,
    /// Enable background monitoring thread
    pub enable_background_monitoring: bool,
    /// Monitoring interval for background thread
    pub monitoring_interval: Duration,
    /// Maximum metrics buffer size
    pub max_metrics_buffer_size: usize,
}

impl Default for RealTimeMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            update_frequency_ms: 1000,
            alert_interval_ms: 5000,
            max_alerts: 100,
            enable_background_monitoring: false,
            monitoring_interval: Duration::from_millis(100),
            max_metrics_buffer_size: 1000,
        }
    }
}

/// Export manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportManagerConfig {
    /// Auto-export enabled
    pub auto_export: bool,
    /// Export interval
    pub export_interval_ms: u64,
    /// Maximum export size
    pub max_export_size_mb: u32,
    /// Compression enabled
    pub compression_enabled: bool,
}

impl Default for ExportManagerConfig {
    fn default() -> Self {
        Self {
            auto_export: false,
            export_interval_ms: 300000,
            max_export_size_mb: 100,
            compression_enabled: true,
        }
    }
}

/// Alert manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertManagerConfig {
    /// Alert evaluation frequency
    pub evaluation_frequency_ms: u64,
    /// Maximum alert history
    pub max_alert_history: usize,
    /// Alert cooldown period
    pub alert_cooldown_ms: u64,
}

impl Default for AlertManagerConfig {
    fn default() -> Self {
        Self {
            evaluation_frequency_ms: 1000,
            max_alert_history: 1000,
            alert_cooldown_ms: 60000,
        }
    }
}

/// Analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Analysis interval
    pub analysis_interval_ms: u64,
    /// Trend analysis window
    pub trend_window_ms: u64,
    /// Health analysis enabled
    pub health_analysis: bool,
    /// Performance regression detection
    pub regression_detection: bool,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            analysis_interval_ms: 60000,
            trend_window_ms: 300000,
            health_analysis: true,
            regression_detection: true,
        }
    }
}

/// Profiler capabilities and supported features
///
/// Describes what features and monitoring capabilities are available
/// on the current platform and configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerCapabilities {
    /// Memory profiling support
    pub memory_profiling: bool,
    /// CPU profiling support
    pub cpu_profiling: bool,
    /// GPU profiling support
    pub gpu_profiling: bool,
    /// Network profiling support
    pub network_profiling: bool,
    /// Thermal monitoring support
    pub thermal_monitoring: bool,
    /// Battery monitoring support
    pub battery_monitoring: bool,
    /// Real-time monitoring support
    pub real_time_monitoring: bool,
    /// Platform-specific capabilities
    pub platform_specific: PlatformCapabilities,
}

/// Platform-specific profiler capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformCapabilities {
    /// iOS-specific features available
    pub ios_features: Vec<String>,
    /// Android-specific features available
    pub android_features: Vec<String>,
    /// Generic features available
    pub generic_features: Vec<String>,
}

impl Default for ProfilerCapabilities {
    fn default() -> Self {
        Self {
            memory_profiling: true,
            cpu_profiling: true,
            gpu_profiling: true,
            network_profiling: true,
            thermal_monitoring: true,
            battery_monitoring: true,
            real_time_monitoring: true,
            platform_specific: PlatformCapabilities::default(),
        }
    }
}

impl Default for PlatformCapabilities {
    fn default() -> Self {
        Self {
            ios_features: vec![],
            android_features: vec![],
            generic_features: vec![
                "basic_metrics".to_string(),
                "performance_analysis".to_string(),
            ],
        }
    }
}
