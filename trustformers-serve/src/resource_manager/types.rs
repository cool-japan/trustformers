//! Core types and data structures for resource management system.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

/// Port usage types
#[derive(Debug, Clone)]
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
    /// Custom usage
    Custom(String),
}

/// Directory status
#[derive(Debug, Clone)]
pub enum DirectoryStatus {
    /// Available for allocation
    Available,
    /// Currently allocated
    Allocated,
    /// Under cleanup
    Cleaning,
    /// Failed/corrupted
    Failed,
    /// Maintenance mode
    Maintenance,
}

/// Directory permissions
#[derive(Debug, Clone)]
pub struct DirectoryPermissions {
    /// Owner read permission
    pub owner_read: bool,
    /// Owner write permission
    pub owner_write: bool,
    /// Owner execute permission
    pub owner_execute: bool,
    /// Group permissions
    pub group_permissions: u8,
    /// Other permissions
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

/// Temporary directory cleanup policy
#[derive(Debug, Clone)]
pub enum TempDirectoryCleanupPolicy {
    /// Immediate cleanup on test completion
    Immediate,
    /// Cleanup after specified delay
    Delayed(Duration),
    /// Cleanup at end of test session
    SessionEnd,
    /// Manual cleanup required
    Manual,
    /// Keep for debugging
    Debug,
}

/// Directory usage tracking
#[derive(Debug, Clone)]
pub struct DirectoryUsageTracking {
    /// Files created
    pub files_created: usize,
    /// Total bytes written
    pub bytes_written: u64,
    /// Total bytes read
    pub bytes_read: u64,
    /// Peak usage
    pub peak_usage: u64,
    /// Usage timeline
    pub usage_timeline: Vec<(DateTime<Utc>, u64)>,
}

/// GPU capabilities
#[derive(Debug, Clone)]
pub enum GpuCapability {
    /// CUDA support
    Cuda(String),
    /// OpenCL support
    OpenCl(String),
    /// Vulkan support
    Vulkan(String),
    /// Machine learning frameworks
    MachineLearning(Vec<String>),
    /// Custom capability
    Custom(String, String),
}

/// GPU device status
#[derive(Debug, Clone)]
pub enum GpuDeviceStatus {
    /// Device available
    Available,
    /// Device busy
    Busy,
    /// Device error
    Error(String),
    /// Device maintenance
    Maintenance,
    /// Device offline
    Offline,
}

/// GPU usage types
#[derive(Debug, Clone)]
pub enum GpuUsageType {
    /// Model training
    Training,
    /// Model inference
    Inference,
    /// Data processing
    DataProcessing,
    /// Benchmarking
    Benchmarking,
    /// Custom usage
    Custom(String),
}

/// GPU performance requirements
#[derive(Debug, Clone)]
pub struct GpuPerformanceRequirements {
    /// Minimum memory required (MB)
    pub min_memory_mb: u64,
    /// Minimum compute capability
    pub min_compute_capability: f32,
    /// Required frameworks
    pub required_frameworks: Vec<String>,
    /// Performance constraints
    pub constraints: Vec<GpuConstraint>,
}

/// GPU constraints
#[derive(Debug, Clone)]
pub struct GpuConstraint {
    /// Constraint type
    pub constraint_type: GpuConstraintType,
    /// Constraint value
    pub value: f64,
    /// Constraint importance
    pub importance: ConstraintImportance,
}

/// GPU constraint types
#[derive(Debug, Clone)]
pub enum GpuConstraintType {
    /// Maximum memory usage
    MaxMemoryUsage,
    /// Maximum utilization
    MaxUtilization,
    /// Minimum performance
    MinPerformance,
    /// Power consumption limit
    PowerLimit,
    /// Temperature limit
    TemperatureLimit,
    /// Custom constraint
    Custom(String),
}

/// Constraint importance levels
#[derive(Debug, Clone)]
pub enum ConstraintImportance {
    /// Low importance
    Low,
    /// Medium importance
    Medium,
    /// High importance
    High,
    /// Critical importance
    Critical,
}

/// GPU clock speeds
#[derive(Debug, Clone)]
pub struct GpuClockSpeeds {
    /// Core clock (MHz)
    pub core_clock_mhz: u32,
    /// Memory clock (MHz)
    pub memory_clock_mhz: u32,
    /// Shader clock (MHz)
    pub shader_clock_mhz: Option<u32>,
}

/// GPU metric types
#[derive(Debug, Clone)]
pub enum GpuMetricType {
    /// Memory usage
    MemoryUsage,
    /// GPU utilization
    Utilization,
    /// Temperature
    Temperature,
    /// Power consumption
    PowerConsumption,
    /// Clock speeds
    ClockSpeeds,
    /// Throughput
    Throughput,
    /// Error rate
    ErrorRate,
    /// Custom metric
    Custom(String),
}

/// GPU alert types
#[derive(Debug, Clone)]
pub enum GpuAlertType {
    /// High memory usage
    HighMemoryUsage,
    /// High utilization
    HighUtilization,
    /// High temperature
    HighTemperature,
    /// High power consumption
    HighPowerConsumption,
    /// Low performance
    LowPerformance,
    /// Device error
    DeviceError,
    /// Device offline
    DeviceOffline,
    /// Custom alert
    Custom(String),
}

/// Alert severity levels
#[derive(Debug, Clone)]
pub enum AlertSeverity {
    /// Informational
    Info,
    /// Warning
    Warning,
    /// Error
    Error,
    /// Critical
    Critical,
}

/// GPU alert event types
#[derive(Debug, Clone)]
pub enum GpuAlertEventType {
    /// Alert triggered
    Triggered,
    /// Alert resolved
    Resolved,
    /// Alert escalated
    Escalated,
    /// Alert acknowledged
    Acknowledged,
    /// Alert suppressed
    Suppressed,
}

/// GPU benchmark types
#[derive(Debug, Clone)]
pub enum GpuBenchmarkType {
    /// Compute performance
    Compute,
    /// Memory bandwidth
    MemoryBandwidth,
    /// Machine learning performance
    MachineLearning,
    /// Graphics performance
    Graphics,
    /// Custom benchmark
    Custom(String),
}

/// Trend direction
#[derive(Debug, Clone)]
pub enum TrendDirection {
    /// Performance improving
    Improving,
    /// Performance stable
    Stable,
    /// Performance degrading
    Degrading,
    /// Insufficient data
    Unknown,
}

/// Regression severity levels
#[derive(Debug, Clone)]
pub enum RegressionSeverity {
    /// Minor regression
    Minor,
    /// Moderate regression
    Moderate,
    /// Major regression
    Major,
    /// Critical regression
    Critical,
}

/// Performance recommendation types
#[derive(Debug, Clone)]
pub enum PerformanceRecommendationType {
    /// Optimize GPU utilization
    OptimizeUtilization,
    /// Reduce memory usage
    ReduceMemoryUsage,
    /// Adjust clock speeds
    AdjustClockSpeeds,
    /// Update drivers
    UpdateDrivers,
    /// Redistribute workload
    RedistributeWorkload,
    /// Custom recommendation
    Custom(String),
}

/// Recommendation difficulty levels
#[derive(Debug, Clone)]
pub enum RecommendationDifficulty {
    /// Easy to implement
    Easy,
    /// Medium difficulty
    Medium,
    /// Hard to implement
    Hard,
    /// Very hard to implement
    VeryHard,
}

/// Recommendation priority levels
#[derive(Debug, Clone)]
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

/// Cleanup event types
#[derive(Debug, Clone)]
pub enum CleanupEventType {
    /// Cleanup started
    Started,
    /// Cleanup completed successfully
    Completed,
    /// Cleanup failed
    Failed,
    /// Cleanup skipped
    Skipped,
    /// Cleanup deferred
    Deferred,
}

/// Cleanup result
#[derive(Debug, Clone)]
pub enum CleanupResult {
    /// Successful cleanup
    Success {
        files_removed: usize,
        bytes_freed: u64,
    },
    /// Partial cleanup
    Partial {
        files_removed: usize,
        files_failed: usize,
        bytes_freed: u64,
        errors: Vec<String>,
    },
    /// Failed cleanup
    Failed {
        error: String,
        files_attempted: usize,
    },
    /// Skipped cleanup
    Skipped { reason: String },
}

/// Escalation condition types
#[derive(Debug, Clone)]
pub enum EscalationConditionType {
    /// Alert duration
    Duration,
    /// Alert count
    Count,
    /// Severity level
    Severity,
    /// Custom condition
    Custom(String),
}

/// Comparison operators
#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    /// Equal to
    Equal,
    /// Greater than
    GreaterThan,
    /// Less than
    LessThan,
    /// Greater than or equal
    GreaterThanOrEqual,
    /// Less than or equal
    LessThanOrEqual,
    /// Not equal
    NotEqual,
}

/// Escalation action types
#[derive(Debug, Clone)]
pub enum EscalationActionType {
    /// Send notification
    Notify,
    /// Throttle GPU usage
    Throttle,
    /// Stop test execution
    StopTest,
    /// Restart device
    RestartDevice,
    /// Custom action
    Custom(String),
}

/// Worker status types
#[derive(Debug, Clone)]
pub enum WorkerStatus {
    Idle,
    Busy,
    Offline,
}

/// Execution status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Health status
#[derive(Debug, Clone)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
}

/// System resource statistics
#[derive(Debug)]
pub struct SystemResourceStatistics {
    /// Port usage statistics
    pub port_stats: PortUsageStatistics,
    /// Directory usage statistics
    pub directory_stats: DirectoryUsageStatistics,
    /// GPU usage statistics
    pub gpu_stats: GpuUsageStatistics,
    /// Database usage statistics
    pub database_stats: DatabaseUsageStatistics,
    /// Overall system efficiency
    pub overall_efficiency: f32,
}

/// Port usage statistics
#[derive(Debug, Default, Clone)]
pub struct PortUsageStatistics {
    /// Total ports allocated
    pub total_allocated: u64,
    /// Currently allocated ports
    pub currently_allocated: usize,
    /// Peak port usage
    pub peak_usage: usize,
    /// Allocation failures
    pub allocation_failures: u64,
    /// Average allocation duration
    pub average_allocation_duration: Duration,
    /// Port utilization by range
    pub utilization_by_range: HashMap<(u16, u16), f32>,
}

/// Directory usage statistics
#[derive(Debug, Default, Clone)]
pub struct DirectoryUsageStatistics {
    /// Total directories created
    pub total_created: u64,
    /// Currently allocated directories
    pub currently_allocated: usize,
    /// Peak directory usage
    pub peak_usage: usize,
    /// Total bytes used
    pub total_bytes_used: u64,
    /// Average directory lifetime
    pub average_lifetime: Duration,
    /// Directory utilization
    pub utilization: f32,
}

/// GPU usage statistics
#[derive(Debug, Default, Clone)]
pub struct GpuUsageStatistics {
    /// Total GPU allocations
    pub total_allocations: u64,
    /// Currently allocated GPUs
    pub currently_allocated: usize,
    /// Peak GPU usage
    pub peak_usage: usize,
    /// Average utilization
    pub average_utilization: f32,
    /// Total memory allocated (MB)
    pub total_memory_allocated_mb: u64,
    /// Allocation efficiency
    pub allocation_efficiency: f32,
    /// GPU performance index
    pub performance_index: f32,
}

/// Database usage statistics
#[derive(Debug, Default, Clone)]
pub struct DatabaseUsageStatistics {
    /// Total connections allocated
    pub total_allocated: u64,
    /// Currently active connections
    pub currently_active: usize,
    /// Peak connection usage
    pub peak_usage: usize,
    /// Average connection lifetime
    pub average_lifetime: Duration,
    /// Connection pool efficiency
    pub pool_efficiency: f32,
}

/// System performance snapshot
#[derive(Debug, Clone)]
pub struct SystemPerformanceSnapshot {
    /// Snapshot timestamp
    pub timestamp: DateTime<Utc>,
    /// CPU utilization
    pub cpu_utilization: f32,
    /// Memory utilization
    pub memory_utilization: f32,
    /// GPU utilization
    pub gpu_utilization: Option<f32>,
    /// Network utilization
    pub network_utilization: f32,
    /// Disk utilization
    pub disk_utilization: f32,
    /// Overall system efficiency
    pub overall_efficiency: f32,
}

/// Allocation event for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationEvent {
    pub timestamp: DateTime<Utc>,
    pub resource_id: String,
    pub test_id: String,
    pub event_type: String,
    pub details: HashMap<String, String>,
}

/// Load metrics for workers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub active_tasks: usize,
    pub queue_length: usize,
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

/// Distribution event for task scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionEvent {
    pub timestamp: DateTime<Utc>,
    pub task_id: String,
    pub worker_id: String,
    pub distribution_strategy: String,
    pub load_score: f64,
}

/// Execution state for tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionState {
    pub status: ExecutionStatus,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub progress: f64,
    pub metadata: HashMap<String, String>,
}

/// Execution performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPerformanceMetrics {
    pub total_executions: u64,
    pub successful_executions: u64,
    pub failed_executions: u64,
    pub average_duration: Duration,
    pub cpu_usage_avg: f64,
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
