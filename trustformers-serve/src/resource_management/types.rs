//! Core Types for TrustformeRS Resource Management System
//!
//! This module contains all the fundamental types, enums, and configuration structures
//! used throughout the resource management system. The types are organized into logical
//! groups for easy navigation and maintenance.

use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};
use tokio::task::JoinHandle;

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

// ================================
// Core Data Structures
// ================================

/// Information about an allocated network port
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortAllocation {
    /// The allocated port number
    pub port: u16,

    /// ID of the test that allocated this port
    pub test_id: String,

    /// Timestamp when the port was allocated
    pub allocated_at: DateTime<Utc>,

    /// Expected time when the port will be released
    pub expected_release: Option<DateTime<Utc>>,

    /// Type of usage for this port
    pub usage_type: PortUsageType,

    /// Additional metadata about the allocation
    pub metadata: HashMap<String, String>,
}

/// Request for port reservation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortReservationRequest {
    /// ID of the test requesting the reservation
    pub test_id: String,

    /// Number of ports requested
    pub port_count: usize,

    /// Preferred port range for allocation
    pub preferred_range: Option<(u16, u16)>,

    /// Intended usage type for the ports
    pub usage_type: PortUsageType,

    /// Timestamp when the request was made
    pub requested_at: DateTime<Utc>,

    /// Priority of the request (0.0 to 1.0)
    pub priority: f32,

    /// Timeout for the reservation
    pub timeout: Duration,

    /// Specific port requested (alternative to port_count)
    pub port: Option<u16>,

    /// Duration for the reservation
    pub duration: Option<Duration>,
}

/// Statistics for port usage
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PortUsageStatistics {
    /// Total number of ports allocated over time
    pub total_allocated: u64,

    /// Number of currently allocated ports
    pub currently_allocated: usize,

    /// Peak number of ports allocated simultaneously
    pub peak_usage: usize,

    /// Number of allocation failures
    pub allocation_failures: u64,

    /// Average duration of port allocations
    pub average_allocation_duration: Duration,

    /// Port utilization by range (range -> utilization percentage)
    pub utilization_by_range: HashMap<(u16, u16), f32>,
}

/// Information about a temporary directory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TempDirectoryInfo {
    /// Path to the directory
    pub path: PathBuf,

    /// Size limit for the directory in bytes
    pub size_limit: u64,

    /// Current usage in bytes
    pub current_usage: u64,

    /// Access permissions for the directory
    pub permissions: DirectoryPermissions,

    /// Timestamp when the directory was created
    pub created_at: DateTime<Utc>,

    /// Timestamp when the directory was last accessed
    pub last_accessed: DateTime<Utc>,

    /// Current status of the directory
    pub status: DirectoryStatus,

    /// Size in bytes
    pub size_bytes: u64,

    /// Usage information
    pub usage_info: Option<DirectoryUsageInfo>,
}

/// Directory access permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectoryPermissions {
    /// Owner has read permission
    pub owner_read: bool,

    /// Owner has write permission
    pub owner_write: bool,

    /// Owner has execute permission
    pub owner_execute: bool,

    /// Group permissions (octal notation)
    pub group_permissions: u8,

    /// Other permissions (octal notation)
    pub other_permissions: u8,
}

impl Default for DirectoryPermissions {
    fn default() -> Self {
        Self {
            owner_read: true,
            owner_write: true,
            owner_execute: true,
            group_permissions: 0o77,
            other_permissions: 0o77,
        }
    }
}

/// Information about an allocated temporary directory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TempDirectoryAllocation {
    /// Information about the allocated directory
    pub directory: TempDirectoryInfo,

    /// ID of the test that allocated this directory
    pub test_id: String,

    /// Timestamp when the directory was allocated
    pub allocated_at: DateTime<Utc>,

    /// Expected time for cleanup
    pub expected_cleanup: Option<DateTime<Utc>>,

    /// Cleanup policy for this directory
    pub cleanup_policy: TempDirectoryCleanupPolicy,

    /// Usage tracking information
    pub usage_tracking: DirectoryUsageTracking,

    /// Cleanup timestamp
    pub cleanup_at: Option<DateTime<Utc>>,

    /// Directory purpose
    pub purpose: String,
}

/// Tracking information for directory usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectoryUsageTracking {
    /// Number of files created in the directory
    pub files_created: usize,

    /// Total bytes written to the directory
    pub bytes_written: u64,

    /// Total bytes read from the directory
    pub bytes_read: u64,

    /// Peak usage in bytes
    pub peak_usage: u64,

    /// Timeline of usage over time
    pub usage_timeline: Vec<(DateTime<Utc>, u64)>,
}

/// Task for cleanup operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupTask {
    /// Unique identifier for the task
    pub id: String,

    /// Path to the directory to clean
    pub directory_path: PathBuf,

    /// Cleanup policy to apply
    pub policy: TempDirectoryCleanupPolicy,

    /// Scheduled time for cleanup
    pub scheduled_time: DateTime<Utc>,

    /// Priority of the task (0.0 to 1.0)
    pub priority: f32,

    /// ID of the associated test
    pub test_id: String,

    /// Number of retry attempts made
    pub retry_count: usize,

    /// Task identifier
    pub task_id: String,

    /// Target path
    pub target_path: PathBuf,

    /// Cleanup type
    pub cleanup_type: CleanupType,

    /// Scheduled at timestamp
    pub scheduled_at: DateTime<Utc>,

    /// Task type (alternative representation)
    pub task_type: String,

    /// Additional task details
    pub details: HashMap<String, String>,
}

/// Event record for cleanup operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupEvent {
    /// Timestamp of the event
    pub timestamp: DateTime<Utc>,

    /// Type of cleanup event
    pub event_type: CleanupEventType,

    /// Path to the directory involved
    pub directory_path: PathBuf,

    /// ID of the associated test
    pub test_id: String,

    /// Duration of the cleanup operation
    pub duration: Duration,

    /// Result of the cleanup operation
    pub result: CleanupResult,

    /// Additional details about the event
    pub details: HashMap<String, String>,

    /// Event identifier
    pub event_id: String,

    /// Task identifier
    pub task_id: String,

    /// Target path (alternative to directory_path)
    pub target_path: PathBuf,

    /// Bytes cleaned
    pub bytes_cleaned: u64,

    /// Files cleaned
    pub files_cleaned: usize,

    /// Success status
    pub success: bool,

    /// Error message if failed
    pub error: Option<String>,
}

/// Statistics for cleanup operations
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CleanupStatistics {
    /// Total number of cleanups performed
    pub total_cleanups: u64,

    /// Number of successful cleanups
    pub successful_cleanups: u64,

    /// Number of failed cleanups
    pub failed_cleanups: u64,

    /// Total number of files removed
    pub total_files_removed: u64,

    /// Total bytes freed by cleanup operations
    pub total_bytes_freed: u64,

    /// Average time taken for cleanup operations
    pub average_cleanup_time: Duration,

    /// Cleanup efficiency (successful / total)
    pub cleanup_efficiency: f32,

    /// Total tasks executed (alias for total_cleanups)
    pub total_tasks: u64,

    /// Total files cleaned (alias for total_files_removed)
    pub total_files_cleaned: u64,

    /// Total bytes cleaned (alias for total_bytes_freed)
    pub total_bytes_cleaned: u64,

    /// Average duration of cleanup tasks (alias for average_cleanup_time)
    pub average_duration: Duration,
}

/// Statistics for directory usage
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct DirectoryUsageStatistics {
    /// Total number of directories created
    pub total_created: u64,

    /// Number of currently allocated directories
    pub currently_allocated: usize,

    /// Peak number of directories allocated simultaneously
    pub peak_usage: usize,

    /// Total bytes used across all directories
    pub total_bytes_used: u64,

    /// Average lifetime of directories
    pub average_lifetime: Duration,

    /// Overall directory utilization percentage
    pub utilization: f32,
}

/// Information about a GPU device
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConstraint {
    /// Type of constraint
    pub constraint_type: GpuConstraintType,

    /// Constraint value
    pub value: f64,

    /// Importance level of the constraint
    pub importance: ConstraintImportance,
}

/// Real-time metrics for a GPU device
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

    /// Associated test ID (if any)
    pub test_id: Option<String>,

    /// Additional metadata for the metric
    pub metadata: HashMap<String, String>,
}

/// Alert threshold configuration for GPU monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAlertThresholds {
    /// High memory usage threshold (0.0 to 1.0)
    pub high_memory_usage: f32,

    /// High utilization threshold (0.0 to 1.0)
    pub high_utilization: f32,

    /// High temperature threshold in Celsius
    pub high_temperature: f32,

    /// High power consumption threshold in watts
    pub high_power_consumption: f32,

    /// Low performance threshold (0.0 to 1.0)
    pub low_performance: f32,

    /// Error rate threshold (0.0 to 1.0)
    pub error_rate: f32,
}

impl Default for GpuAlertThresholds {
    fn default() -> Self {
        Self {
            high_memory_usage: 0.9,
            high_utilization: 0.95,
            high_temperature: 85.0,
            high_power_consumption: 300.0,
            low_performance: 0.5,
            error_rate: 0.05,
        }
    }
}

/// GPU alert information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAlert {
    /// Unique alert identifier
    pub id: String,

    /// Device that triggered the alert
    pub device_id: usize,

    /// Type of alert
    pub alert_type: GpuAlertType,

    /// Severity level
    pub severity: AlertSeverity,

    /// Human-readable alert message
    pub message: String,

    /// Timestamp when the alert was triggered
    pub triggered_at: DateTime<Utc>,

    /// Associated test ID (if any)
    pub test_id: Option<String>,

    /// Additional alert metadata
    pub metadata: HashMap<String, String>,
}

/// Event record for GPU alerts
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationAction {
    /// Type of action to perform
    pub action_type: EscalationActionType,

    /// Parameters for the action
    pub parameters: HashMap<String, String>,

    /// Priority of the action (0.0 to 1.0)
    pub priority: f32,
}

/// Performance benchmark result for a GPU
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuPerformanceBenchmark {
    /// Device identifier
    pub device_id: usize,

    /// Name of the benchmark
    pub name: String,

    /// Type of benchmark performed
    pub benchmark_type: GpuBenchmarkType,

    /// Benchmark score
    pub score: f64,

    /// Timestamp when the benchmark was run
    pub timestamp: DateTime<Utc>,

    /// Additional benchmark metadata
    pub metadata: HashMap<String, String>,
}

/// Performance record for GPU usage
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuPerformanceBaseline {
    /// Device identifier
    pub device_id: usize,

    /// Baseline metric values
    pub baseline_metrics: HashMap<String, f64>,

    /// Timestamp when the baseline was established
    pub established_at: DateTime<Utc>,

    /// Confidence level of the baseline (0.0 to 1.0)
    pub confidence: f32,

    /// Number of samples used to establish the baseline
    pub sample_count: usize,
}

/// Performance analysis results
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    /// Direction of the trend
    pub direction: TrendDirection,

    /// Strength of the trend (0.0 to 1.0)
    pub strength: f32,

    /// Confidence in the trend (0.0 to 1.0)
    pub confidence: f32,

    /// Time period of the trend
    pub period: Duration,
}

/// Performance regression detection
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

/// Statistics for GPU usage
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct GpuUsageStatistics {
    /// Total number of GPU allocations
    pub total_allocations: u64,

    /// Number of currently allocated GPUs
    pub currently_allocated: usize,

    /// Peak number of GPUs allocated simultaneously
    pub peak_usage: usize,

    /// Average utilization percentage across all GPUs
    pub average_utilization: f32,

    /// Total memory allocated across all GPUs in MB
    pub total_memory_allocated_mb: u64,

    /// Allocation efficiency percentage
    pub allocation_efficiency: f32,

    /// Overall GPU performance index
    pub performance_index: f32,
}

/// Statistics for database usage
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct DatabaseUsageStatistics {
    /// Total number of connections allocated
    pub total_allocated: u64,

    /// Number of currently active connections
    pub currently_active: usize,

    /// Peak number of connections used simultaneously
    pub peak_usage: usize,

    /// Average lifetime of connections
    pub average_lifetime: Duration,

    /// Connection pool efficiency percentage
    pub pool_efficiency: f32,

    /// Total connections established
    pub total_connections: u64,

    /// Number of active connections
    pub active_connections: usize,

    /// Peak connections reached
    pub peak_connections: usize,

    /// Average connection duration
    pub average_duration: Duration,

    /// Query throughput (queries per second)
    pub query_throughput: f64,
}

/// Overall system resource statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SystemResourceStatistics {
    /// Network port usage statistics
    pub port_stats: PortUsageStatistics,

    /// Directory usage statistics
    pub directory_stats: DirectoryUsageStatistics,

    /// GPU usage statistics
    pub gpu_stats: GpuUsageStatistics,

    /// Database usage statistics
    pub database_stats: DatabaseUsageStatistics,

    /// Overall system efficiency percentage
    pub overall_efficiency: f32,
}

/// System performance snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPerformanceSnapshot {
    /// Timestamp of the snapshot
    pub timestamp: DateTime<Utc>,

    /// CPU utilization percentage
    pub cpu_utilization: f32,

    /// Memory utilization percentage
    pub memory_utilization: f32,

    /// GPU utilization percentage (if available)
    pub gpu_utilization: Option<f32>,

    /// Network utilization percentage
    pub network_utilization: f32,

    /// Disk utilization percentage
    pub disk_utilization: f32,

    /// Overall system efficiency
    pub overall_efficiency: f32,

    /// System resource statistics
    pub system_stats: SystemResourceStatistics,

    /// GPU usage statistics
    pub gpu_stats: GpuUsageStatistics,

    /// Database usage statistics
    pub database_stats: DatabaseUsageStatistics,

    /// Port usage statistics
    pub port_stats: PortUsageStatistics,

    /// Directory usage statistics
    pub directory_stats: DirectoryUsageStatistics,
}

// ================================
// Event and Allocation Types
// ================================

/// Event record for resource allocations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationEvent {
    /// Timestamp of the event
    pub timestamp: DateTime<Utc>,

    /// Resource identifier
    pub resource_id: String,

    /// Test identifier
    pub test_id: String,

    /// Type of event
    pub event_type: String,

    /// Additional event details
    pub details: HashMap<String, String>,

    /// Type of resource being allocated
    pub resource_type: String,
}

/// Load metrics for system monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadMetrics {
    /// CPU usage percentage
    pub cpu_usage: f64,

    /// Memory usage percentage
    pub memory_usage: f64,

    /// Number of active tasks
    pub active_tasks: usize,

    /// Length of the task queue
    pub queue_length: usize,

    /// Timestamp of the metrics
    pub timestamp: DateTime<Utc>,
}

impl Default for LoadMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0.0,
            active_tasks: 0,
            queue_length: 0,
            timestamp: Utc::now(),
        }
    }
}

/// Event for load distribution decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionEvent {
    /// Timestamp of the distribution decision
    pub timestamp: DateTime<Utc>,

    /// Task identifier
    pub task_id: String,

    /// Worker identifier assigned to the task
    pub worker_id: String,

    /// Strategy used for distribution
    pub distribution_strategy: String,

    /// Load score used in the decision
    pub load_score: f64,
}

/// Execution state for tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionState {
    /// Current execution status
    pub status: ExecutionStatus,

    /// Timestamp when execution started
    pub start_time: DateTime<Utc>,

    /// Timestamp when execution ended (if completed)
    pub end_time: Option<DateTime<Utc>>,

    /// Progress percentage (0.0 to 1.0)
    pub progress: f64,

    /// Additional execution metadata
    pub metadata: HashMap<String, String>,

    /// Worker identifier
    pub worker_id: String,

    /// Current state description
    pub state: String,

    /// Current task identifier
    pub current_task: Option<String>,

    /// Time since state was set
    pub state_since: DateTime<Utc>,

    /// Last heartbeat timestamp
    pub last_heartbeat: DateTime<Utc>,
}

/// Performance metrics for execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPerformanceMetrics {
    /// Total number of executions
    pub total_executions: u64,

    /// Number of successful executions
    pub successful_executions: u64,

    /// Number of failed executions
    pub failed_executions: u64,

    /// Average execution duration
    pub average_duration: Duration,

    /// Average CPU usage during execution
    pub cpu_usage_avg: f64,

    /// Average memory usage during execution
    pub memory_usage_avg: f64,
}

impl Default for ExecutionPerformanceMetrics {
    fn default() -> Self {
        Self {
            total_executions: 0,
            successful_executions: 0,
            failed_executions: 0,
            average_duration: Duration::from_secs(0),
            cpu_usage_avg: 0.0,
            memory_usage_avg: 0.0,
        }
    }
}

// ================================
// Manager Structures
// ================================

/// Worker pool for parallel task execution
#[derive(Debug)]
pub struct WorkerPool {
    /// List of available workers
    pub workers: Arc<Mutex<Vec<Worker>>>,

    /// Maximum capacity of the worker pool
    pub capacity: usize,
}

impl Default for WorkerPool {
    fn default() -> Self {
        Self {
            workers: Arc::new(Mutex::new(Vec::new())),
            capacity: 4, // Default to 4 workers
        }
    }
}

/// Individual worker information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Worker {
    /// Unique worker identifier
    pub id: String,

    /// Current status of the worker
    pub status: WorkerStatus,

    /// Current task being executed (if any)
    pub current_task: Option<String>,
}

/// Health check information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    /// Name of the health check
    pub name: String,

    /// Current health status
    pub status: HealthStatus,

    /// Timestamp of the last check
    pub last_check: DateTime<Utc>,
}

/// Alert information for system monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Unique alert identifier
    pub id: String,

    /// Severity level of the alert
    pub severity: AlertSeverity,

    /// Human-readable alert message
    pub message: String,

    /// Timestamp when the alert was created
    pub timestamp: DateTime<Utc>,
}

// ================================
// Trait Definitions
// ================================

/// Trait for handling GPU alerts
pub trait GpuAlertHandler: Send + Sync {
    /// Handle a GPU alert
    fn handle_alert(&self, alert: &GpuAlert) -> Result<()>;

    /// Get the name of the handler
    fn name(&self) -> &str;

    /// Check if the handler can handle a specific alert type
    fn can_handle(&self, alert_type: &GpuAlertType) -> bool;
}

// ================================
// Type Aliases
// ================================

/// Type alias for background task handles
pub type BackgroundTask = JoinHandle<()>;

/// Type alias for resource monitoring results
pub type MonitoringResult<T> = Result<T>;

/// Type alias for allocation results
pub type AllocationResult<T> = Result<T>;

/// Type alias for cleanup operation results
pub type CleanupOpResult<T> = Result<T>;

// ================================
// Utility Functions
// ================================

impl PortUsageType {
    /// Check if the usage type requires exclusive access
    pub fn requires_exclusive_access(&self) -> bool {
        matches!(self, Self::HttpServer | Self::HttpsServer | Self::Database)
    }

    /// Get the default port range for this usage type
    pub fn default_port_range(&self) -> Option<(u16, u16)> {
        match self {
            Self::HttpServer => Some((8000, 8999)),
            Self::HttpsServer => Some((8443, 8499)),
            Self::Database => Some((5432, 5499)),
            _ => None,
        }
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

impl ExecutionStatus {
    /// Check if the execution is in a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }

    /// Check if the execution is active
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Running)
    }
}

// ================================
// Allocation Event Types
// ================================

/// Allocation event types for tracking resource allocation lifecycle
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationEventType {
    /// Resource allocated
    Allocated,
    /// Resource deallocated
    Deallocated,
    /// Allocation failed
    Failed,
}

/// Worker states for resource allocation tracking
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerState {
    /// Worker idle
    Idle,
    /// Worker busy
    Busy,
    /// Worker starting up
    Starting,
    /// Worker shutting down
    Stopping,
}

/// Cleanup type for directory management
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CleanupType {
    /// Delete directory
    DeleteDirectory,
    /// Delete files older than duration
    DeleteOldFiles(Duration),
    /// Empty directory
    EmptyDirectory,
    /// Compress old files
    CompressFiles,
    /// Custom cleanup
    Custom(String),
}

// ============================================================================
// Additional Resource Management Types
// ============================================================================

/// Resource lifecycle stage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceLifecycleStage {
    /// Resource is being created
    Creating,
    /// Resource has been created
    Created,
    /// Resource is active
    Active,
    /// Resource is idle
    Idle,
    /// Resource is being cleaned up
    Cleaning,
    /// Resource is being destroyed
    Destroying,
    /// Resource is destroyed
    Destroyed,
}

/// GPU alert statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAlertStatistics {
    /// Total alerts generated
    pub total_alerts: u64,
    /// Active alerts
    pub active_alerts: u32,
    /// Alerts by severity
    pub alerts_by_severity: std::collections::HashMap<String, u64>,
    /// Last alert time
    pub last_alert_time: Option<DateTime<Utc>>,
    /// Total alerts generated (alternate field name)
    pub total_alerts_generated: u64,
    /// Alerts categorized by type
    pub alerts_by_type: std::collections::HashMap<String, u64>,
    /// Number of handler failures
    pub handler_failures: u64,
    /// Total alerts that have been acknowledged
    pub total_alerts_acknowledged: u64,
}

impl Default for GpuAlertStatistics {
    fn default() -> Self {
        Self {
            total_alerts: 0,
            active_alerts: 0,
            alerts_by_severity: std::collections::HashMap::new(),
            last_alert_time: None,
            total_alerts_generated: 0,
            alerts_by_type: std::collections::HashMap::new(),
            handler_failures: 0,
            total_alerts_acknowledged: 0,
        }
    }
}

/// GPU alert system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAlertSystem {
    /// Alert configuration
    pub config: GpuAlertConfig,
    /// Alert statistics
    pub statistics: GpuAlertStatistics,
    /// Active alerts
    pub active_alerts: Vec<GpuAlert>,
}

/// Resource statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceStatistics {
    /// Total resources allocated
    pub total_allocated: u64,
    /// Active resources
    pub active_resources: u32,
    /// Peak resource usage
    pub peak_usage: u64,
    /// Average resource lifetime
    pub avg_lifetime: Duration,
    /// Resource utilization rate
    pub utilization_rate: f64,
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// Memory utilization percentage
    pub memory_utilization: f64,
    /// Number of allocations
    pub allocation_count: u64,
    /// Total duration of all allocations
    pub total_duration: Duration,
    /// Efficiency score (0.0 to 1.0)
    pub efficiency_score: f64,
}

impl Default for ResourceStatistics {
    fn default() -> Self {
        Self {
            total_allocated: 0,
            active_resources: 0,
            peak_usage: 0,
            avg_lifetime: Duration::from_secs(0),
            utilization_rate: 0.0,
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            allocation_count: 0,
            total_duration: Duration::from_secs(0),
            efficiency_score: 0.0,
        }
    }
}

/// Directory purpose classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DirectoryPurpose {
    /// Test data storage
    TestData,
    /// Temporary files
    Temporary,
    /// Cache storage
    Cache,
    /// Log files
    Logs,
    /// Configuration files
    Config,
    /// Build artifacts
    Build,
    /// General purpose
    General,
}

impl Default for DirectoryPurpose {
    fn default() -> Self {
        Self::General
    }
}

// ============================================================================
// Resource Management Types
// ============================================================================

/// System resource model for resource predictions
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SystemResourceModel {
    pub cpu_model: String,
    pub memory_model: String,
    pub storage_model: String,
    pub network_model: String,
}

/// Cleanup schedule configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupSchedule {
    pub enabled: bool,
    pub interval: Duration,
    pub cleanup_types: Vec<CleanupType>,
    pub priority: String,
}

/// Directory usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectoryUsageInfo {
    pub path: String,
    pub size_bytes: u64,
    pub file_count: usize,
    pub last_accessed: DateTime<Utc>,
    pub subdirectory_count: usize,
    pub total_size_bytes: u64,
    pub last_modified: DateTime<Utc>,
}

/// GPU alert escalation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAlertEscalation {
    pub escalation_level: u8,
    pub notification_channels: Vec<String>,
    pub escalation_delay: Duration,
    pub alert_id: String,
    pub initial_severity: AlertSeverity,
    pub current_level: u8,
    pub escalated_at: DateTime<Utc>,
    pub escalation_history: Vec<String>,
}

impl Default for GpuAlertEscalation {
    fn default() -> Self {
        Self {
            escalation_level: 0,
            notification_channels: Vec::new(),
            escalation_delay: Duration::from_secs(300),
            alert_id: String::new(),
            initial_severity: AlertSeverity::Info,
            current_level: 0,
            escalated_at: Utc::now(),
            escalation_history: Vec::new(),
        }
    }
}

/// Performance baseline for metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceBaseline {
    pub baseline_cpu: f64,
    pub baseline_memory: f64,
    pub baseline_io: f64,
    pub baseline_network: f64,
    pub timestamp: DateTime<Utc>,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceUtilizationMetrics {
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub io_utilization: f64,
    pub network_utilization: f64,
    pub timestamp: DateTime<Utc>,
}

/// Alert threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThreshold {
    pub metric_name: String,
    pub warning_threshold: f64,
    pub critical_threshold: f64,
    pub evaluation_window: Duration,
}

impl Default for CleanupSchedule {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(3600),
            cleanup_types: Vec::new(),
            priority: "medium".to_string(),
        }
    }
}

impl Default for DirectoryUsageInfo {
    fn default() -> Self {
        Self {
            path: String::new(),
            size_bytes: 0,
            file_count: 0,
            last_accessed: Utc::now(),
            subdirectory_count: 0,
            total_size_bytes: 0,
            last_modified: Utc::now(),
        }
    }
}

impl Default for AlertThreshold {
    fn default() -> Self {
        Self {
            metric_name: String::new(),
            warning_threshold: 0.8,
            critical_threshold: 0.95,
            evaluation_window: Duration::from_secs(300),
        }
    }
}
