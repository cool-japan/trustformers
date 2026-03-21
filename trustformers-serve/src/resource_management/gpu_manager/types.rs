//! GPU Manager Types
//!
//! This module contains all types, configurations, and data structures used by the GPU manager
//! system for comprehensive GPU resource management, monitoring, and optimization.
//!
//! # Organization
//!
//! The types are organized into functional categories:
//! - **Error Types**: Error handling and result types
//! - **Configuration Types**: System and component configuration structures
//! - **Core Data Types**: Fundamental GPU device and allocation types
//! - **Metrics Types**: Real-time and historical monitoring data
//! - **Health and Status Types**: Device health monitoring and status tracking
//! - **Alert Types**: Alert system configuration and event types
//! - **Performance Types**: Benchmarking and performance analysis
//! - **Load Balancing Types**: Load distribution strategies and data
//! - **Enums**: All enumeration types for various categorizations
//! - **Traits**: Interface definitions for extensibility
//! - **Type Aliases**: Convenience type aliases
//!
//! All types maintain backward compatibility and include comprehensive documentation
//! with derive macros for serialization, debugging, and other common operations.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt, time::Duration};
use thiserror::Error;

// ================================================================================================
// ERROR TYPES
// ================================================================================================

/// Comprehensive error types for GPU operations
///
/// This enum covers all possible error conditions that can occur during GPU resource management,
/// from device discovery and allocation to monitoring and performance tracking.
///
/// Note: Cannot derive Clone due to anyhow::Error field in MonitoringError variant
/// Note: Cannot derive Serialize due to anyhow::Error field in MonitoringError variant
#[derive(Debug, Error)]
pub enum GpuManagerError {
    /// GPU device not found
    #[error("GPU device {device_id} not found")]
    DeviceNotFound { device_id: usize },

    /// Insufficient GPU memory for allocation
    #[error("Insufficient GPU memory: required {required_mb}MB, available {available_mb}MB")]
    InsufficientMemory { required_mb: u64, available_mb: u64 },

    /// GPU device is not available for allocation
    #[error("GPU device {device_id} is not available (status: {status:?})")]
    DeviceUnavailable { device_id: usize, status: String },

    /// Required framework is not supported by the device
    #[error("Framework requirement not met: {framework} not supported")]
    FrameworkNotSupported { framework: String },

    /// Performance constraint violated during allocation
    #[error("Performance constraint violated: {constraint}")]
    ConstraintViolated { constraint: String },

    /// GPU driver error occurred
    #[error("GPU driver error: {message}")]
    DriverError { message: String },

    /// Hardware failure detected on GPU device
    #[error("Hardware failure detected on device {device_id}: {details}")]
    HardwareFailure { device_id: usize, details: String },

    /// Monitoring system error
    #[error("Monitoring system error: {source}")]
    MonitoringError {
        #[from]
        source: anyhow::Error,
    },

    /// Alert system error
    #[error("Alert system error: {message}")]
    AlertError { message: String },

    /// Configuration error
    #[error("Configuration error: {field} - {message}")]
    ConfigurationError { field: String, message: String },

    /// Invalid configuration detected
    #[error("Invalid configuration: {message}")]
    InvalidConfiguration { message: String },
}

/// Result type for GPU operations with GpuManagerError
pub type GpuResult<T> = Result<T, GpuManagerError>;

// ================================================================================================
// CONFIGURATION TYPES
// ================================================================================================

/// Configuration for GPU device pool management
///
/// This configuration controls how GPU devices are discovered, allocated, and managed
/// within the resource pool system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuPoolConfig {
    /// Maximum number of GPU devices that can be allocated
    pub max_devices: usize,

    /// Enable GPU monitoring
    pub enable_monitoring: bool,

    /// Monitoring interval in seconds
    pub monitoring_interval_secs: u64,

    /// Memory allocation threshold (0.0 to 1.0)
    pub memory_threshold: f32,

    /// Temperature threshold in Celsius
    pub temperature_threshold: f32,

    /// Enable performance tracking
    pub enable_performance_tracking: bool,

    /// Enable health monitoring for GPU devices
    pub enable_health_monitoring: bool,

    /// Minimum memory required per device in MB
    pub min_memory_mb: u64,

    /// Enable GPU alerting system
    pub enable_alerts: bool,

    /// Enable load balancing across GPU devices
    pub enable_load_balancing: bool,

    /// Allocation timeout in seconds
    pub allocation_timeout_secs: u64,

    /// Alert threshold configuration
    pub alert_thresholds: GpuAlertThresholds,

    /// Memory allocation threshold (0.0 to 1.0) for allocation decisions
    pub memory_allocation_threshold: f32,
}

impl Default for GpuPoolConfig {
    fn default() -> Self {
        Self {
            max_devices: 8,
            enable_monitoring: true,
            monitoring_interval_secs: 5,
            memory_threshold: 0.9,
            temperature_threshold: 85.0,
            enable_performance_tracking: true,
            enable_health_monitoring: true,
            min_memory_mb: 1024, // 1GB minimum
            enable_alerts: false,
            enable_load_balancing: false,
            allocation_timeout_secs: 30,
            alert_thresholds: GpuAlertThresholds::default(),
            memory_allocation_threshold: 0.8, // 80% allocation threshold
        }
    }
}

/// Configuration for GPU monitoring system
///
/// Controls real-time monitoring, metrics collection, and alert integration
/// for GPU devices in the resource pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMonitoringConfig {
    /// Enable real-time monitoring
    pub enable_realtime: bool,

    /// Monitoring interval
    pub monitoring_interval: Duration,

    /// Enable performance tracking during monitoring
    pub enable_performance_tracking: bool,

    /// Enable alerts integration
    pub enable_alerts: bool,

    /// Alert configuration
    pub alert_config: GpuAlertConfig,
}

impl Default for GpuMonitoringConfig {
    fn default() -> Self {
        Self {
            enable_realtime: true,
            monitoring_interval: Duration::from_secs(5),
            enable_performance_tracking: true,
            enable_alerts: true,
            alert_config: GpuAlertConfig::default(),
        }
    }
}

/// Configuration for GPU alert system
///
/// Defines alert thresholds, escalation rules, and notification settings
/// for proactive GPU health monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAlertConfig {
    /// Enable temperature alerts
    pub enable_temperature_alerts: bool,

    /// Enable utilization alerts
    pub enable_utilization_alerts: bool,

    /// Enable memory usage alerts
    pub enable_memory_alerts: bool,

    /// Alert threshold configuration
    pub thresholds: GpuAlertThresholds,

    /// Alert escalation rules
    pub escalation_rules: Vec<GpuAlertEscalationRule>,

    /// Alert cooldown period
    pub cooldown_period: Duration,

    /// Enable automatic alert escalation
    pub escalation_enabled: bool,

    /// Escalation delay in seconds
    pub escalation_delay_seconds: u64,
}

impl Default for GpuAlertConfig {
    fn default() -> Self {
        Self {
            enable_temperature_alerts: true,
            enable_utilization_alerts: true,
            enable_memory_alerts: true,
            thresholds: GpuAlertThresholds::default(),
            escalation_rules: Vec::new(),
            cooldown_period: Duration::from_secs(300),
            escalation_enabled: false,
            escalation_delay_seconds: 600,
        }
    }
}

/// Alert threshold configuration for GPU monitoring
///
/// Defines specific threshold values for different types of alerts
/// including temperature, utilization, memory usage, and power consumption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAlertThresholds {
    /// Temperature warning threshold in Celsius
    pub temperature_warning: f32,

    /// Temperature critical threshold in Celsius
    pub temperature_critical: f32,

    /// Utilization critical threshold percentage
    pub utilization_critical_percent: f32,

    /// Memory critical threshold percentage
    pub memory_critical_percent: f32,

    /// Power consumption warning threshold in watts
    pub power_warning_watts: f32,

    /// Power consumption critical threshold in watts
    pub power_critical_watts: f32,

    /// Utilization warning threshold percentage
    pub utilization_warning_percent: f32,

    /// Memory warning threshold percentage
    pub memory_warning_percent: f32,
}

impl Default for GpuAlertThresholds {
    fn default() -> Self {
        Self {
            temperature_warning: 80.0,
            temperature_critical: 90.0,
            utilization_critical_percent: 95.0,
            memory_critical_percent: 90.0,
            power_warning_watts: 250.0,
            power_critical_watts: 300.0,
            utilization_warning_percent: 85.0,
            memory_warning_percent: 80.0,
        }
    }
}

// ================================================================================================
// CORE DATA TYPES
// ================================================================================================

/// Information about a GPU device
///
/// Contains comprehensive information about a GPU device including
/// hardware specifications, current status, and capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    /// Unique device identifier
    pub device_id: usize,

    /// Human-readable device name
    pub device_name: String,

    /// Total memory available in MB
    pub total_memory_mb: u64,

    /// Currently available memory in MB
    pub available_memory_mb: u64,

    /// Current utilization percentage (0.0 to 100.0)
    pub utilization_percent: f32,

    /// List of device capabilities
    pub capabilities: Vec<GpuCapability>,

    /// Current device status
    pub status: GpuDeviceStatus,

    /// Timestamp of last update
    pub last_updated: DateTime<Utc>,
}

/// Information about an allocated GPU device
///
/// Tracks allocation details including the device, test association,
/// memory allocation, and performance requirements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAllocation {
    /// Information about the allocated device
    pub device: GpuDeviceInfo,

    /// ID of the test that allocated this GPU
    pub test_id: String,

    /// Amount of memory allocated in MB
    pub memory_allocated_mb: u64,

    /// Timestamp when the GPU was allocated
    pub allocated_at: DateTime<Utc>,

    /// Expected time when the GPU will be released
    pub expected_release: Option<DateTime<Utc>>,

    /// Type of usage for this GPU
    pub usage_type: GpuUsageType,

    /// Performance requirements for the allocation
    pub performance_requirements: GpuPerformanceRequirements,
}

/// Performance requirements for GPU allocation
///
/// Specifies the minimum requirements and constraints that must be
/// satisfied for a successful GPU allocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuPerformanceRequirements {
    /// Minimum memory required in MB
    pub min_memory_mb: u64,

    /// Minimum compute capability version
    pub min_compute_capability: f32,

    /// Required framework support
    pub required_frameworks: Vec<String>,

    /// Performance constraints to satisfy
    pub constraints: Vec<GpuConstraint>,
}

/// A performance constraint for GPU usage
///
/// Defines specific constraints that must be met during GPU allocation
/// and usage, with importance levels for prioritization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConstraint {
    /// Type of constraint
    pub constraint_type: GpuConstraintType,

    /// Constraint value
    pub value: f64,

    /// Importance level of the constraint
    pub importance: ConstraintImportance,
}

// ================================================================================================
// METRICS TYPES
// ================================================================================================

/// Real-time metrics for a GPU device
///
/// Contains current performance and utilization metrics collected
/// from GPU monitoring systems for immediate analysis and alerting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuRealTimeMetrics {
    /// Device identifier
    pub device_id: usize,

    /// Timestamp of the metrics
    pub timestamp: DateTime<Utc>,

    /// Memory usage in MB
    pub memory_usage_mb: u64,

    /// Utilization percentage
    pub utilization_percent: f32,

    /// Temperature in Celsius
    pub temperature_celsius: f32,

    /// Power consumption in watts
    pub power_consumption_watts: f32,

    /// Current clock speeds
    pub clock_speeds: GpuClockSpeeds,

    /// Fan speeds (percentage for each fan)
    pub fan_speeds: Vec<f32>,
}

/// GPU clock speed information
///
/// Tracks the current operating frequencies of different GPU components
/// for performance monitoring and optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuClockSpeeds {
    /// Core clock speed in MHz
    pub core_clock_mhz: u32,

    /// Memory clock speed in MHz
    pub memory_clock_mhz: u32,

    /// Shader clock speed in MHz (if available)
    pub shader_clock_mhz: Option<u32>,
}

/// Historical metric entry for GPU monitoring
///
/// Stores historical GPU metrics for trend analysis and
/// long-term performance tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuHistoricalMetric {
    /// Device identifier
    pub device_id: usize,

    /// Timestamp of the metric
    pub timestamp: DateTime<Utc>,

    /// Type of metric recorded
    pub metric_type: GpuMetricType,

    /// Metric value
    pub value: f64,
}

// ================================================================================================
// HEALTH AND STATUS TYPES
// ================================================================================================

/// GPU health status information
///
/// Comprehensive health assessment for a GPU device including
/// health score, status checks, and identified issues.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuHealthStatus {
    /// Device identifier
    pub device_id: usize,

    /// Overall health indicator
    pub is_healthy: bool,

    /// Health score (0.0 to 1.0)
    pub health_score: f32,

    /// Timestamp of last health check
    pub last_check: DateTime<Utc>,

    /// List of identified issues
    pub issues: Vec<String>,

    /// Temperature status check
    pub temperature_ok: bool,

    /// Memory status check
    pub memory_ok: bool,

    /// Performance status check
    pub performance_ok: bool,
}

// ================================================================================================
// ALERT TYPES
// ================================================================================================

/// GPU alert information
///
/// Represents an active or historical GPU alert with full context
/// including device, alert type, severity, and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAlert {
    /// Unique alert identifier
    pub alert_id: String,

    /// Device that triggered the alert
    pub device_id: usize,

    /// Type of alert
    pub alert_type: GpuAlertType,

    /// Severity level
    pub severity: AlertSeverity,

    /// Human-readable alert message
    pub message: String,

    /// Current metric value that triggered the alert
    pub current_value: f64,

    /// Threshold value that was exceeded
    pub threshold_value: f64,

    /// Timestamp when the alert was triggered
    pub timestamp: DateTime<Utc>,

    /// Whether the alert has been acknowledged
    pub acknowledged: bool,
}

/// Event record for GPU alerts
///
/// Tracks the lifecycle of GPU alerts including trigger events,
/// acknowledgments, and resolutions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAlertEvent {
    /// Timestamp of the event
    pub timestamp: DateTime<Utc>,

    /// Type of alert event
    pub event_type: GpuAlertEventType,

    /// Associated alert information
    pub alert: GpuAlert,

    /// Additional event details
    pub details: HashMap<String, String>,
}

/// Rule for escalating GPU alerts
///
/// Defines conditions and actions for automatic alert escalation
/// based on duration, severity, or other criteria.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAlertEscalationRule {
    /// Name of the escalation rule
    pub name: String,

    /// Alert types that trigger this rule
    pub alert_types: Vec<GpuAlertType>,

    /// Conditions that must be met for escalation
    pub conditions: Vec<EscalationCondition>,

    /// Actions to take when escalating
    pub actions: Vec<EscalationAction>,

    /// Delay before escalation
    pub delay: Duration,
}

/// Condition for alert escalation
///
/// Specifies the criteria that must be met for an alert
/// to be escalated to a higher severity level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationCondition {
    /// Type of condition to check
    pub condition_type: EscalationConditionType,

    /// Value to compare against
    pub value: String,

    /// Comparison operator to use
    pub operator: ComparisonOperator,
}

/// Action to take during alert escalation
///
/// Defines the specific action and its parameters to execute
/// when an alert escalation condition is met.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationAction {
    /// Type of action to perform
    pub action_type: EscalationActionType,

    /// Parameters for the action
    pub parameters: HashMap<String, String>,

    /// Priority of the action (0.0 to 1.0)
    pub priority: f32,
}

// ================================================================================================
// PERFORMANCE TYPES
// ================================================================================================

/// Performance benchmark result for a GPU
///
/// Contains the results of a performance benchmark including
/// score, execution time, and metadata for analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuPerformanceBenchmark {
    /// Name of the benchmark
    pub name: String,

    /// Device identifier
    pub device_id: usize,

    /// Type of benchmark performed
    pub benchmark_type: GpuBenchmarkType,

    /// Benchmark score
    pub score: f64,

    /// Time taken to execute the benchmark
    pub execution_time: Duration,

    /// Timestamp when the benchmark was run
    pub timestamp: DateTime<Utc>,

    /// Additional benchmark parameters and metadata
    pub parameters: HashMap<String, String>,
}

/// Performance record for GPU usage
///
/// Records performance metrics during actual GPU usage
/// for historical analysis and optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuPerformanceRecord {
    /// Device identifier
    pub device_id: usize,

    /// Test identifier
    pub test_id: String,

    /// Performance metrics collected
    pub metrics: HashMap<String, f64>,

    /// Timestamp of the record
    pub timestamp: DateTime<Utc>,

    /// Duration of the test
    pub duration: Duration,
}

/// Performance baseline for a GPU device
///
/// Establishes baseline performance metrics for a device
/// used for regression detection and performance analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuPerformanceBaseline {
    /// Device identifier
    pub device_id: usize,

    /// Baseline metric values
    pub baseline_metrics: HashMap<String, f64>,

    /// Timestamp when the baseline was established
    pub established_at: DateTime<Utc>,

    /// Number of samples used to establish the baseline
    pub sample_count: usize,

    /// Confidence level of the baseline (0.0 to 1.0)
    pub confidence_level: f32,
}

/// Performance analysis results
///
/// Comprehensive analysis of GPU performance including trends,
/// regressions, and optimization recommendations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuPerformanceAnalysis {
    /// Performance trends by device
    pub trends: HashMap<usize, PerformanceTrend>,

    /// Detected performance regressions
    pub regressions: Vec<PerformanceRegression>,

    /// Performance improvement recommendations
    pub recommendations: Vec<PerformanceRecommendation>,

    /// Timestamp when the analysis was performed
    pub analyzed_at: DateTime<Utc>,
}

impl Default for GpuPerformanceAnalysis {
    fn default() -> Self {
        Self {
            trends: HashMap::new(),
            regressions: Vec::new(),
            recommendations: Vec::new(),
            analyzed_at: Utc::now(),
        }
    }
}

/// Performance trend information
///
/// Describes the direction and strength of performance trends
/// over time with confidence metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    /// Direction of the trend
    pub direction: TrendDirection,

    /// Strength of the trend (0.0 to 100.0)
    pub strength: f32,

    /// Confidence in the trend (0.0 to 1.0)
    pub confidence: f32,

    /// Time period of the trend
    pub period: Duration,
}

/// Performance regression detection
///
/// Records detected performance regressions with baseline
/// comparison and severity assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRegression {
    /// Device identifier
    pub device_id: usize,

    /// Name of the affected metric
    pub metric_name: String,

    /// Baseline value for comparison
    pub baseline_value: f64,

    /// Current value showing regression
    pub current_value: f64,

    /// Percentage of regression
    pub regression_percent: f32,

    /// Timestamp when the regression was detected
    pub detected_at: DateTime<Utc>,

    /// Severity level of the regression
    pub severity: RegressionSeverity,
}

/// Performance improvement recommendation
///
/// Provides actionable recommendations for improving GPU performance
/// with impact assessment and implementation guidance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecommendation {
    /// Device identifier
    pub device_id: usize,

    /// Type of recommendation
    pub recommendation_type: PerformanceRecommendationType,

    /// Description of the recommendation
    pub description: String,

    /// Expected performance impact (0.0 to 1.0)
    pub expected_impact: f32,

    /// Implementation difficulty
    pub difficulty: RecommendationDifficulty,

    /// Priority level
    pub priority: RecommendationPriority,
}

// ================================================================================================
// LOAD BALANCING TYPES
// ================================================================================================

/// Load balancing strategies for GPU allocation
///
/// Defines different strategies for distributing GPU workloads
/// across available devices for optimal performance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin allocation across devices
    RoundRobin,

    /// Allocate to the least loaded device first
    LeastLoaded,

    /// Best fit allocation based on requirements
    BestFit,

    /// Random allocation for load distribution
    Random,
}

impl Default for LoadBalancingStrategy {
    fn default() -> Self {
        Self::LeastLoaded
    }
}

// ================================================================================================
// ENUMERATION TYPES
// ================================================================================================

/// GPU device capabilities
///
/// Represents different capabilities and features supported
/// by GPU devices for compatibility checking.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuCapability {
    /// CUDA support with version
    Cuda(String),

    /// OpenCL support with version
    OpenCl(String),

    /// Vulkan support with version
    Vulkan(String),

    /// Machine learning framework support
    MachineLearning(Vec<String>),

    /// Custom capability with name and description
    Custom(String, String),
}

/// Status of a GPU device
///
/// Tracks the current operational status of GPU devices
/// for allocation and monitoring purposes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuDeviceStatus {
    /// Device is available for allocation
    Available,

    /// Device is currently busy
    Busy,

    /// Device has an error condition
    Error(String),

    /// Device is in maintenance mode
    Maintenance,

    /// Device is offline
    Offline,
}

/// Types of GPU usage
///
/// Categorizes different types of GPU workloads for
/// resource allocation and performance optimization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuUsageType {
    /// Model training workload
    Training,

    /// Model inference workload
    Inference,

    /// Data processing workload
    DataProcessing,

    /// Benchmarking workload
    Benchmarking,

    /// Custom usage type
    Custom(String),
}

/// Types of GPU constraints
///
/// Defines different types of performance and resource
/// constraints for GPU allocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuConstraintType {
    /// Maximum memory usage limit
    MaxMemoryUsage,

    /// Maximum utilization percentage
    MaxUtilization,

    /// Minimum performance requirement
    MinPerformance,

    /// Power consumption limit in watts
    PowerLimit,

    /// Temperature limit in Celsius
    TemperatureLimit,

    /// Custom constraint type
    Custom(String),
}

/// Importance levels for constraints
///
/// Prioritizes constraints for allocation decisions
/// and conflict resolution.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConstraintImportance {
    /// Low importance - nice to have
    Low,

    /// Medium importance - should be satisfied
    Medium,

    /// High importance - must be satisfied
    High,

    /// Critical importance - failure to satisfy stops execution
    Critical,
}

/// Types of GPU metrics
///
/// Categorizes different metrics collected from GPU devices
/// for monitoring and analysis.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpuMetricType {
    /// Memory usage in MB or percentage
    MemoryUsage,

    /// GPU utilization percentage
    Utilization,

    /// Temperature in Celsius
    Temperature,

    /// Power consumption in watts
    PowerConsumption,

    /// Clock speeds in MHz
    ClockSpeeds,

    /// Throughput in operations per second
    Throughput,

    /// Error rate percentage
    ErrorRate,

    /// Custom metric type
    Custom(String),
}

/// Types of GPU alerts
///
/// Categorizes different alert conditions that can be
/// triggered by GPU monitoring systems.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuAlertType {
    /// High memory usage alert
    HighMemoryUsage,

    /// High utilization alert
    HighUtilization,

    /// High temperature alert
    HighTemperature,

    /// High power consumption alert
    HighPowerConsumption,

    /// Low performance alert
    LowPerformance,

    /// Device error alert
    DeviceError,

    /// Device offline alert
    DeviceOffline,

    /// Custom alert type
    Custom(String),
}

/// Severity levels for alerts
///
/// Defines the urgency and importance of alerts for
/// appropriate response and escalation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational message
    Info,

    /// Warning condition
    Warning,

    /// Error condition
    Error,

    /// Critical condition requiring immediate attention
    Critical,
}

/// Types of GPU alert events
///
/// Tracks the lifecycle events of GPU alerts for
/// audit trails and analysis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuAlertEventType {
    /// Alert was triggered
    Triggered,

    /// Alert condition was resolved
    Resolved,

    /// Alert was escalated to higher severity
    Escalated,

    /// Alert was acknowledged by operator
    Acknowledged,

    /// Alert was suppressed
    Suppressed,

    /// Alert was archived
    Archived,
}

/// Types of escalation conditions
///
/// Defines the criteria for automatic alert escalation
/// based on various factors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EscalationConditionType {
    /// Alert duration threshold
    Duration,

    /// Alert count threshold
    Count,

    /// Severity level threshold
    Severity,

    /// Custom escalation condition
    Custom(String),
}

/// Comparison operators for conditions
///
/// Provides operators for evaluating escalation conditions
/// and constraint checking.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOperator {
    /// Equal to
    Equal,

    /// Greater than
    GreaterThan,

    /// Less than
    LessThan,

    /// Greater than or equal to
    GreaterThanOrEqual,

    /// Less than or equal to
    LessThanOrEqual,

    /// Not equal to
    NotEqual,
}

/// Types of escalation actions
///
/// Defines the actions that can be taken when alerts
/// are escalated beyond normal thresholds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EscalationActionType {
    /// Send notification to administrators
    Notify,

    /// Throttle GPU usage
    Throttle,

    /// Stop test execution
    StopTest,

    /// Restart GPU device
    RestartDevice,

    /// Custom escalation action
    Custom(String),
}

/// Types of GPU benchmarks
///
/// Categorizes different types of performance benchmarks
/// for comprehensive GPU evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuBenchmarkType {
    /// Compute performance benchmark
    Compute,

    /// Memory bandwidth benchmark
    MemoryBandwidth,

    /// Matrix operations benchmark
    MatrixOperations,

    /// Machine learning inference benchmark
    MLInference,

    /// Machine learning training benchmark
    MLTraining,

    /// Custom benchmark type
    Custom(String),
}

/// Performance trend directions
///
/// Indicates the direction of performance changes over time
/// for trend analysis and alerting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Performance is improving over time
    Improving,

    /// Performance is stable
    Stable,

    /// Performance is degrading over time
    Degrading,

    /// Insufficient data to determine trend
    Unknown,
}

/// Severity levels for performance regressions
///
/// Categorizes the impact of detected performance regressions
/// for appropriate response prioritization.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RegressionSeverity {
    /// Minor performance regression
    Minor,

    /// Moderate performance regression
    Moderate,

    /// Major performance regression
    Major,

    /// Critical performance regression
    Critical,
}

/// Types of performance recommendations
///
/// Categorizes different types of performance optimization
/// recommendations for actionable guidance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceRecommendationType {
    /// Optimize GPU utilization
    OptimizeUtilization,

    /// Reduce memory usage
    ReduceMemoryUsage,

    /// Adjust clock speeds
    AdjustClockSpeeds,

    /// Update GPU drivers
    UpdateDrivers,

    /// Redistribute workload across devices
    RedistributeWorkload,

    /// Custom recommendation
    Custom(String),
}

/// Difficulty levels for implementing recommendations
///
/// Assesses the implementation complexity of performance
/// recommendations for prioritization.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RecommendationDifficulty {
    /// Easy to implement
    Easy,

    /// Medium difficulty to implement
    Medium,

    /// Hard to implement
    Hard,

    /// Very hard to implement
    VeryHard,
}

/// Priority levels for recommendations
///
/// Prioritizes performance recommendations for
/// implementation planning and resource allocation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RecommendationPriority {
    /// Low priority
    Low,

    /// Medium priority
    Medium,

    /// High priority
    High,

    /// Critical priority
    Critical,
}

// ================================================================================================
// STATISTICS TYPES
// ================================================================================================

/// Statistics for GPU usage across the resource pool
///
/// Tracks comprehensive usage statistics for GPU resources
/// including allocation patterns, efficiency metrics, and utilization.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct GpuUsageStatistics {
    /// Total number of GPU allocations over time
    pub total_allocations: u64,

    /// Number of currently allocated GPUs
    pub currently_allocated: usize,

    /// Peak number of GPUs allocated simultaneously
    pub peak_usage: usize,

    /// Total GPU hours consumed
    pub total_gpu_hours: f64,

    /// Average memory allocated per allocation in MB
    pub average_memory_allocated_mb: f64,

    /// Peak memory usage percentage across all devices
    pub peak_memory_usage_percent: f32,

    /// Allocation efficiency (successful allocations / total requests)
    pub efficiency: f32,
}

// ================================================================================================
// TRAIT DEFINITIONS
// ================================================================================================

/// Trait for handling GPU alerts
///
/// Defines the interface for alert handlers that can process
/// and respond to GPU alerts in the monitoring system.
pub trait GpuAlertHandler: Send + Sync {
    /// Handle a GPU alert
    ///
    /// Processes the given alert and takes appropriate action
    /// based on the alert type and severity.
    ///
    /// # Arguments
    ///
    /// * `alert` - The GPU alert to handle
    ///
    /// # Returns
    ///
    /// Result indicating success or failure of alert handling
    fn handle_alert(&self, alert: &GpuAlert) -> Result<()>;

    /// Get the name of the handler
    ///
    /// Returns a human-readable name for this alert handler
    /// for identification and logging purposes.
    fn name(&self) -> &str;

    /// Check if the handler can handle a specific alert type
    ///
    /// Determines whether this handler is capable of processing
    /// alerts of the given type.
    ///
    /// # Arguments
    ///
    /// * `alert_type` - The type of alert to check
    ///
    /// # Returns
    ///
    /// True if the handler can process this alert type
    fn can_handle(&self, alert_type: &GpuAlertType) -> bool;
}

// ================================================================================================
// TYPE ALIASES
// ================================================================================================

/// Type alias for GPU device identifier
pub type GpuDeviceId = usize;

/// Type alias for test identifier
pub type TestId = String;

/// Type alias for allocation identifier
pub type AllocationId = String;

/// Type alias for alert identifier
pub type AlertId = String;

/// Type alias for metric value
pub type MetricValue = f64;

/// Type alias for memory size in megabytes
pub type MemoryMB = u64;

/// Type alias for utilization percentage
pub type UtilizationPercent = f32;

/// Type alias for temperature in Celsius
pub type TemperatureCelsius = f32;

/// Type alias for power consumption in watts
pub type PowerWatts = f32;

/// Type alias for clock speed in MHz
pub type ClockSpeedMHz = u32;

// ================================================================================================
// UTILITY IMPLEMENTATIONS
// ================================================================================================

impl fmt::Display for GpuDeviceStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Available => write!(f, "Available"),
            Self::Busy => write!(f, "Busy"),
            Self::Error(msg) => write!(f, "Error: {}", msg),
            Self::Maintenance => write!(f, "Maintenance"),
            Self::Offline => write!(f, "Offline"),
        }
    }
}

impl fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "Info"),
            Self::Warning => write!(f, "Warning"),
            Self::Error => write!(f, "Error"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

impl fmt::Display for LoadBalancingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RoundRobin => write!(f, "Round Robin"),
            Self::LeastLoaded => write!(f, "Least Loaded"),
            Self::BestFit => write!(f, "Best Fit"),
            Self::Random => write!(f, "Random"),
        }
    }
}

impl fmt::Display for GpuAlertType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HighMemoryUsage => write!(f, "High Memory Usage"),
            Self::HighUtilization => write!(f, "High Utilization"),
            Self::HighTemperature => write!(f, "High Temperature"),
            Self::HighPowerConsumption => write!(f, "High Power Consumption"),
            Self::LowPerformance => write!(f, "Low Performance"),
            Self::DeviceError => write!(f, "Device Error"),
            Self::DeviceOffline => write!(f, "Device Offline"),
            Self::Custom(msg) => write!(f, "Custom: {}", msg),
        }
    }
}

impl AlertSeverity {
    /// Get the numeric priority of the severity level
    pub fn priority(&self) -> u8 {
        match self {
            Self::Info => 0,
            Self::Warning => 1,
            Self::Error => 2,
            Self::Critical => 3,
        }
    }

    /// Check if this severity requires immediate attention
    pub fn requires_immediate_attention(&self) -> bool {
        matches!(self, Self::Error | Self::Critical)
    }
}

impl GpuCapability {
    /// Check if this capability supports a specific framework
    pub fn supports_framework(&self, framework: &str) -> bool {
        match self {
            Self::Cuda(_) => framework.eq_ignore_ascii_case("cuda"),
            Self::OpenCl(_) => framework.eq_ignore_ascii_case("opencl"),
            Self::Vulkan(_) => framework.eq_ignore_ascii_case("vulkan"),
            Self::MachineLearning(frameworks) => frameworks.contains(&framework.to_string()),
            Self::Custom(name, _) => name.eq_ignore_ascii_case(framework),
        }
    }
}

impl GpuConstraintType {
    /// Check if this constraint type is resource-related
    pub fn is_resource_constraint(&self) -> bool {
        matches!(
            self,
            Self::MaxMemoryUsage | Self::MaxUtilization | Self::PowerLimit
        )
    }

    /// Check if this constraint type is performance-related
    pub fn is_performance_constraint(&self) -> bool {
        matches!(self, Self::MinPerformance | Self::TemperatureLimit)
    }
}

impl TrendDirection {
    /// Check if the trend indicates performance issues
    pub fn indicates_problems(&self) -> bool {
        matches!(self, Self::Degrading)
    }

    /// Check if the trend is positive
    pub fn is_positive(&self) -> bool {
        matches!(self, Self::Improving | Self::Stable)
    }
}
