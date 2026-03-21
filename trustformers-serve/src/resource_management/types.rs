//! Core Types for TrustformeRS Resource Management System
//!
//! This module contains all the fundamental types, enums, and configuration structures
//! used throughout the resource management system. The types are organized into logical
//! groups for easy navigation and maintenance.

use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::PathBuf, time::Duration};

// Re-export types moved to types_data module for backward compatibility
pub use super::types_data::*;

// Re-export common types from test modules

// ================================
// Configuration Types
// ================================

/// Configuration for network port pool management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortPoolConfig {
    /// Range of ports available for allocation
    pub port_range: (u16, u16),

    /// Maximum number of ports that can be allocated simultaneously
    pub max_allocation: usize,

    /// Allocation timeout in seconds
    pub allocation_timeout_secs: u64,

    /// Enable port reservation system
    pub enable_reservation: bool,

    /// Reserved port ranges that should not be allocated
    pub reserved_ranges: Vec<(u16, u16)>,
}

impl Default for PortPoolConfig {
    fn default() -> Self {
        Self {
            port_range: (8000, 9000),
            max_allocation: 100,
            allocation_timeout_secs: 300,
            enable_reservation: true,
            reserved_ranges: vec![(22, 22), (80, 80), (443, 443)],
        }
    }
}

/// Configuration for temporary directory pool management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TempDirPoolConfig {
    /// Base directory for temporary directories
    pub base_path: PathBuf,

    /// Maximum number of directories that can be allocated
    pub max_directories: usize,

    /// Maximum size per directory in bytes
    pub max_directory_size_bytes: u64,

    /// Default cleanup policy
    pub default_cleanup_policy: TempDirectoryCleanupPolicy,

    /// Enable automatic cleanup
    pub enable_auto_cleanup: bool,

    /// Cleanup interval in seconds
    pub cleanup_interval_secs: u64,
}

impl Default for TempDirPoolConfig {
    fn default() -> Self {
        Self {
            base_path: PathBuf::from("/tmp/trustformers"),
            max_directories: 50,
            max_directory_size_bytes: 1024 * 1024 * 1024, // 1GB
            default_cleanup_policy: TempDirectoryCleanupPolicy::SessionEnd,
            enable_auto_cleanup: true,
            cleanup_interval_secs: 300,
        }
    }
}

/// Configuration for GPU device pool management
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
        }
    }
}

/// Configuration for database connection pool management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabasePoolConfig {
    /// Maximum number of connections in the pool
    pub max_connections: usize,

    /// Minimum number of connections to maintain
    pub min_connections: usize,

    /// Connection timeout in seconds
    pub connection_timeout_secs: u64,

    /// Connection idle timeout in seconds
    pub idle_timeout_secs: u64,

    /// Maximum lifetime of a connection in seconds
    pub max_lifetime_secs: u64,

    /// Database URL for connections
    pub database_url: String,
}

impl Default for DatabasePoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 20,
            min_connections: 5,
            connection_timeout_secs: 30,
            idle_timeout_secs: 300,
            max_lifetime_secs: 3600,
            database_url: "sqlite::memory:".to_string(),
        }
    }
}

/// Configuration for resource monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMonitoringConfig {
    /// Enable real-time monitoring
    pub enable_real_time: bool,

    /// Monitoring interval in seconds
    pub monitoring_interval_secs: u64,

    /// Metrics retention period in seconds
    pub retention_period_secs: u64,

    /// Enable alerts
    pub enable_alerts: bool,

    /// Alert thresholds
    pub alert_thresholds: HashMap<String, f64>,
}

impl Default for ResourceMonitoringConfig {
    fn default() -> Self {
        let mut alert_thresholds = HashMap::new();
        alert_thresholds.insert("cpu_usage".to_string(), 0.8);
        alert_thresholds.insert("memory_usage".to_string(), 0.9);
        alert_thresholds.insert("disk_usage".to_string(), 0.85);

        Self {
            enable_real_time: true,
            monitoring_interval_secs: 10,
            retention_period_secs: 86400, // 24 hours
            enable_alerts: true,
            alert_thresholds,
        }
    }
}

/// Configuration for conflict resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictResolutionConfig {
    /// Enable automatic conflict resolution
    pub enable_auto_resolution: bool,

    /// Conflict detection sensitivity (0.0 to 1.0)
    pub detection_sensitivity: f32,

    /// Resolution timeout in seconds
    pub resolution_timeout_secs: u64,

    /// Maximum retry attempts for resolution
    pub max_retry_attempts: usize,

    /// Backoff strategy for retries
    pub backoff_strategy: BackoffStrategy,
}

impl Default for ConflictResolutionConfig {
    fn default() -> Self {
        Self {
            enable_auto_resolution: true,
            detection_sensitivity: 0.7,
            resolution_timeout_secs: 60,
            max_retry_attempts: 3,
            backoff_strategy: BackoffStrategy::Exponential,
        }
    }
}

/// Configuration for resource cleanup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceCleanupConfig {
    /// Enable automatic cleanup
    pub enable_auto_cleanup: bool,

    /// Cleanup interval in seconds
    pub cleanup_interval_secs: u64,

    /// Grace period before forceful cleanup in seconds
    pub grace_period_secs: u64,

    /// Maximum cleanup retries
    pub max_cleanup_retries: usize,

    /// Enable cleanup verification
    pub enable_cleanup_verification: bool,
}

impl Default for ResourceCleanupConfig {
    fn default() -> Self {
        Self {
            enable_auto_cleanup: true,
            cleanup_interval_secs: 300,
            grace_period_secs: 30,
            max_cleanup_retries: 3,
            enable_cleanup_verification: true,
        }
    }
}

/// GPU monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMonitoringConfig {
    /// Monitoring interval
    pub monitoring_interval: Duration,

    /// Metrics retention period
    pub retention_period: Duration,

    /// Enable real-time monitoring
    pub real_time_monitoring: bool,

    /// Alert thresholds
    pub alert_thresholds: GpuAlertThresholds,

    /// Monitored metrics
    pub monitored_metrics: Vec<GpuMetricType>,

    /// Alert configuration
    pub alert_config: GpuAlertConfig,

    /// Enable alerts
    pub enable_alerts: bool,

    /// Enable performance tracking
    pub enable_performance_tracking: bool,
}

impl Default for GpuMonitoringConfig {
    fn default() -> Self {
        Self {
            monitoring_interval: Duration::from_secs(1),
            retention_period: Duration::from_secs(3600),
            real_time_monitoring: false,
            alert_thresholds: GpuAlertThresholds::default(),
            monitored_metrics: Vec::new(),
            alert_config: GpuAlertConfig::default(),
            enable_alerts: true,
            enable_performance_tracking: true,
        }
    }
}

/// GPU alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAlertConfig {
    /// Enable alerts
    pub enabled: bool,

    /// Alert cooldown period
    pub cooldown_period: Duration,

    /// Alert thresholds
    pub thresholds: GpuAlertThresholds,

    /// Alert escalation rules
    pub escalation_rules: Vec<GpuAlertEscalationRule>,

    /// Maximum escalation level allowed
    pub max_escalation_level: u8,

    /// Alert retention time in hours
    pub alert_retention_hours: u64,

    /// Enable power consumption alerts
    pub enable_power_alerts: bool,
}

impl Default for GpuAlertConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cooldown_period: Duration::from_secs(300), // 5 minutes
            thresholds: GpuAlertThresholds::default(),
            escalation_rules: Vec::new(),
            max_escalation_level: 3,
            alert_retention_hours: 24,
            enable_power_alerts: true,
        }
    }
}

/// Resource pool configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePoolConfigs {
    /// Network port pool configuration
    pub network_port_pool: PortPoolConfig,

    /// Temporary directory pool configuration
    pub temp_directory_pool: TempDirPoolConfig,

    /// GPU device pool configuration
    pub gpu_device_pool: GpuPoolConfig,

    /// Database pool configuration
    pub database_pool: DatabasePoolConfig,
}

impl Default for ResourcePoolConfigs {
    fn default() -> Self {
        Self {
            network_port_pool: PortPoolConfig::default(),
            temp_directory_pool: TempDirPoolConfig::default(),
            gpu_device_pool: GpuPoolConfig::default(),
            database_pool: DatabasePoolConfig::default(),
        }
    }
}

/// Master resource management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceManagementConfig {
    /// Resource pool configurations
    pub resource_pools: ResourcePoolConfigs,

    /// Resource monitoring configuration
    pub resource_monitoring: ResourceMonitoringConfig,

    /// Conflict resolution configuration
    pub conflict_resolution: ConflictResolutionConfig,

    /// Resource cleanup configuration
    pub resource_cleanup: ResourceCleanupConfig,

    /// Enable parallel execution
    pub enable_parallel_execution: bool,

    /// Maximum concurrent tests
    pub max_concurrent_tests: usize,

    /// Test execution timeout in seconds
    pub test_execution_timeout_secs: u64,
}

impl Default for ResourceManagementConfig {
    fn default() -> Self {
        Self {
            resource_pools: ResourcePoolConfigs::default(),
            resource_monitoring: ResourceMonitoringConfig::default(),
            conflict_resolution: ConflictResolutionConfig::default(),
            resource_cleanup: ResourceCleanupConfig::default(),
            enable_parallel_execution: true,
            max_concurrent_tests: 4,
            test_execution_timeout_secs: 300,
        }
    }
}

// ================================
// Enum Definitions
// ================================

/// Types of port usage for network allocation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PortUsageType {
    /// HTTP server
    HttpServer,

    /// HTTPS server
    HttpsServer,

    /// TCP socket
    TcpSocket,

    /// UDP socket
    UdpSocket,

    /// Database connection
    Database,

    /// Custom usage with description
    Custom(String),
}

/// Status of a temporary directory
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DirectoryStatus {
    /// Available for allocation
    Available,

    /// Currently allocated to a test
    Allocated,

    /// Under cleanup process
    Cleaning,

    /// Failed or corrupted
    Failed,

    /// In maintenance mode
    Maintenance,
}

/// Cleanup policy for temporary directories
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TempDirectoryCleanupPolicy {
    /// Immediate cleanup on test completion
    Immediate,

    /// Cleanup after specified delay
    Delayed(Duration),

    /// Cleanup at end of test session
    SessionEnd,

    /// Manual cleanup required
    Manual,

    /// Keep for debugging purposes
    Debug,
}

/// Types of cleanup events
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CleanupEventType {
    /// Cleanup operation started
    Started,

    /// Cleanup completed successfully
    Completed,

    /// Cleanup operation failed
    Failed,

    /// Cleanup was skipped
    Skipped,

    /// Cleanup was deferred to later
    Deferred,
}

/// Result of a cleanup operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CleanupResult {
    /// Successful cleanup with statistics
    Success {
        files_removed: usize,
        bytes_freed: u64,
    },

    /// Partial cleanup with errors
    Partial {
        files_removed: usize,
        files_failed: usize,
        bytes_freed: u64,
        errors: Vec<String>,
    },

    /// Failed cleanup with error information
    Failed {
        error: String,
        files_attempted: usize,
    },

    /// Skipped cleanup with reason
    Skipped { reason: String },
}

/// Result of a cleanup task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupTaskResult {
    pub task_id: String,
    pub success: bool,
    pub duration: Duration,
    pub resources_cleaned: usize,
    pub errors: Vec<String>,
    pub details: HashMap<String, String>,
}

/// GPU device capabilities
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
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
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
}

/// Types of escalation conditions
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuBenchmarkType {
    /// Compute performance benchmark
    Compute,

    /// Memory bandwidth benchmark
    MemoryBandwidth,

    /// Machine learning performance benchmark
    MachineLearning,

    /// Graphics performance benchmark
    Graphics,

    /// Custom benchmark type
    Custom(String),
}

/// Performance trend directions
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

/// Status of worker processes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerStatus {
    /// Worker is idle and available
    Idle,

    /// Worker is busy executing a task
    Busy,

    /// Worker is offline
    Offline,
}

/// Execution status for tasks
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStatus {
    /// Task is queued for execution
    Queued,

    /// Task is currently running
    Running,

    /// Task completed successfully
    Completed,

    /// Task failed to complete
    Failed,

    /// Task was cancelled
    Cancelled,
}

/// Health status for system components
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Component is healthy
    Healthy,

    /// Component has warnings
    Warning,

    /// Component is in critical state
    Critical,
}

/// Backoff strategies for retry mechanisms
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackoffStrategy {
    /// Linear backoff (constant delay)
    Linear,

    /// Exponential backoff (doubling delay)
    Exponential,

    /// Custom backoff with specific delays
    Custom(Vec<Duration>),
}
