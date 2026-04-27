//! Core data structures for resource management
//!
//! Port allocation info, GPU device info, worker pools, resource events,
//! manager structures, trait definitions, and utility functions.

use super::types::*;
use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};
use tokio::task::JoinHandle;

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // ---- LCG helper for deterministic pseudo-random values ----
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0
        }
        fn next_f64(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
        fn next_u16(&mut self) -> u16 {
            (self.next_u64() & 0xFFFF) as u16
        }
        fn next_usize(&mut self, bound: usize) -> usize {
            (self.next_u64() as usize) % bound.max(1)
        }
    }

    // ---- PortUsageType tests ----
    #[test]
    fn test_port_usage_type_http_server_exclusive() {
        assert!(PortUsageType::HttpServer.requires_exclusive_access());
    }

    #[test]
    fn test_port_usage_type_https_server_exclusive() {
        assert!(PortUsageType::HttpsServer.requires_exclusive_access());
    }

    #[test]
    fn test_port_usage_type_database_exclusive() {
        assert!(PortUsageType::Database.requires_exclusive_access());
    }

    #[test]
    fn test_port_usage_type_tcp_not_exclusive() {
        assert!(!PortUsageType::TcpSocket.requires_exclusive_access());
    }

    #[test]
    fn test_port_usage_type_custom_not_exclusive() {
        assert!(!PortUsageType::Custom("test".to_string()).requires_exclusive_access());
    }

    #[test]
    fn test_port_usage_type_http_default_range() {
        let range = PortUsageType::HttpServer.default_port_range();
        assert_eq!(range, Some((8000, 8999)));
    }

    #[test]
    fn test_port_usage_type_https_default_range() {
        let range = PortUsageType::HttpsServer.default_port_range();
        assert_eq!(range, Some((8443, 8499)));
    }

    #[test]
    fn test_port_usage_type_database_default_range() {
        let range = PortUsageType::Database.default_port_range();
        assert_eq!(range, Some((5432, 5499)));
    }

    #[test]
    fn test_port_usage_type_tcp_no_default_range() {
        assert!(PortUsageType::TcpSocket.default_port_range().is_none());
    }

    // ---- GpuCapability tests ----
    #[test]
    fn test_gpu_capability_cuda_supports_cuda() {
        let cap = GpuCapability::Cuda("11.0".to_string());
        assert!(cap.supports_framework("cuda"));
        assert!(cap.supports_framework("CUDA"));
    }

    #[test]
    fn test_gpu_capability_cuda_rejects_opencl() {
        let cap = GpuCapability::Cuda("11.0".to_string());
        assert!(!cap.supports_framework("opencl"));
    }

    #[test]
    fn test_gpu_capability_opencl_supports_opencl() {
        let cap = GpuCapability::OpenCl("3.0".to_string());
        assert!(cap.supports_framework("opencl"));
        assert!(cap.supports_framework("OpenCL"));
    }

    #[test]
    fn test_gpu_capability_vulkan_supports_vulkan() {
        let cap = GpuCapability::Vulkan("1.3".to_string());
        assert!(cap.supports_framework("vulkan"));
    }

    #[test]
    fn test_gpu_capability_ml_supports_listed_framework() {
        let cap =
            GpuCapability::MachineLearning(vec!["pytorch".to_string(), "tensorflow".to_string()]);
        assert!(cap.supports_framework("pytorch"));
        assert!(!cap.supports_framework("jax"));
    }

    #[test]
    fn test_gpu_capability_custom_supports_matching_name() {
        let cap = GpuCapability::Custom("metal".to_string(), "2.0".to_string());
        assert!(cap.supports_framework("Metal"));
        assert!(!cap.supports_framework("vulkan"));
    }

    // ---- AlertSeverity tests ----
    #[test]
    fn test_alert_severity_priority_ordering() {
        assert_eq!(AlertSeverity::Info.priority(), 0);
        assert_eq!(AlertSeverity::Warning.priority(), 1);
        assert_eq!(AlertSeverity::Error.priority(), 2);
        assert_eq!(AlertSeverity::Critical.priority(), 3);
    }

    #[test]
    fn test_alert_severity_priority_increases() {
        assert!(AlertSeverity::Critical.priority() > AlertSeverity::Error.priority());
        assert!(AlertSeverity::Error.priority() > AlertSeverity::Warning.priority());
        assert!(AlertSeverity::Warning.priority() > AlertSeverity::Info.priority());
    }

    #[test]
    fn test_alert_severity_immediate_attention_error() {
        assert!(AlertSeverity::Error.requires_immediate_attention());
    }

    #[test]
    fn test_alert_severity_immediate_attention_critical() {
        assert!(AlertSeverity::Critical.requires_immediate_attention());
    }

    #[test]
    fn test_alert_severity_no_immediate_attention_info() {
        assert!(!AlertSeverity::Info.requires_immediate_attention());
    }

    #[test]
    fn test_alert_severity_no_immediate_attention_warning() {
        assert!(!AlertSeverity::Warning.requires_immediate_attention());
    }

    // ---- ExecutionStatus tests ----
    #[test]
    fn test_execution_status_terminal_completed() {
        assert!(ExecutionStatus::Completed.is_terminal());
    }

    #[test]
    fn test_execution_status_terminal_failed() {
        assert!(ExecutionStatus::Failed.is_terminal());
    }

    #[test]
    fn test_execution_status_terminal_cancelled() {
        assert!(ExecutionStatus::Cancelled.is_terminal());
    }

    #[test]
    fn test_execution_status_not_terminal_running() {
        assert!(!ExecutionStatus::Running.is_terminal());
    }

    #[test]
    fn test_execution_status_active_running() {
        assert!(ExecutionStatus::Running.is_active());
    }

    #[test]
    fn test_execution_status_not_active_completed() {
        assert!(!ExecutionStatus::Completed.is_active());
    }

    // ---- Default impls tests ----
    #[test]
    fn test_directory_permissions_default() {
        let perms = DirectoryPermissions::default();
        assert!(perms.owner_read);
        assert!(perms.owner_write);
        assert!(perms.owner_execute);
    }

    #[test]
    fn test_gpu_alert_thresholds_default() {
        let t = GpuAlertThresholds::default();
        assert!((t.high_memory_usage - 0.9).abs() < f32::EPSILON);
        assert!((t.high_utilization - 0.95).abs() < f32::EPSILON);
        assert!((t.high_temperature - 85.0).abs() < f32::EPSILON);
        assert!((t.error_rate - 0.05).abs() < f32::EPSILON);
    }

    #[test]
    fn test_load_metrics_default() {
        let m = LoadMetrics::default();
        assert!((m.cpu_usage - 0.0).abs() < f64::EPSILON);
        assert!((m.memory_usage - 0.0).abs() < f64::EPSILON);
        assert_eq!(m.active_tasks, 0);
        assert_eq!(m.queue_length, 0);
    }

    #[test]
    fn test_execution_performance_metrics_default() {
        let m = ExecutionPerformanceMetrics::default();
        assert_eq!(m.total_executions, 0);
        assert_eq!(m.successful_executions, 0);
        assert_eq!(m.failed_executions, 0);
        assert_eq!(m.average_duration, Duration::from_secs(0));
    }

    #[test]
    fn test_worker_pool_default() {
        let pool = WorkerPool::default();
        assert_eq!(pool.capacity, 4);
        assert_eq!(pool.workers.lock().len(), 0);
    }

    #[test]
    fn test_gpu_alert_statistics_default() {
        let stats = GpuAlertStatistics::default();
        assert_eq!(stats.total_alerts, 0);
        assert_eq!(stats.active_alerts, 0);
        assert!(stats.alerts_by_severity.is_empty());
        assert!(stats.last_alert_time.is_none());
    }

    #[test]
    fn test_resource_statistics_default() {
        let stats = ResourceStatistics::default();
        assert_eq!(stats.allocation_count, 0);
        assert_eq!(stats.total_duration, Duration::from_secs(0));
    }

    #[test]
    fn test_directory_purpose_default() {
        assert_eq!(DirectoryPurpose::default(), DirectoryPurpose::General);
    }

    #[test]
    fn test_cleanup_schedule_default() {
        let s = CleanupSchedule::default();
        assert!(s.enabled);
        assert_eq!(s.interval, Duration::from_secs(3600));
        assert!(s.cleanup_types.is_empty());
        assert_eq!(s.priority, "medium");
    }

    #[test]
    fn test_alert_threshold_default() {
        let t = AlertThreshold::default();
        assert!(t.metric_name.is_empty());
        assert!((t.warning_threshold - 0.8).abs() < f64::EPSILON);
        assert!((t.critical_threshold - 0.95).abs() < f64::EPSILON);
        assert_eq!(t.evaluation_window, Duration::from_secs(300));
    }

    #[test]
    fn test_gpu_alert_escalation_default() {
        let e = GpuAlertEscalation::default();
        assert_eq!(e.escalation_level, 0);
        assert!(e.notification_channels.is_empty());
        assert_eq!(e.escalation_delay, Duration::from_secs(300));
    }

    #[test]
    fn test_gpu_performance_analysis_default() {
        let a = GpuPerformanceAnalysis::default();
        assert!(a.trends.is_empty());
    }

    #[test]
    fn test_system_resource_model_default() {
        let m = SystemResourceModel::default();
        assert!(m.cpu_model.is_empty());
        assert!(m.memory_model.is_empty());
    }

    #[test]
    fn test_performance_baseline_default() {
        let b = PerformanceBaseline::default();
        assert!((b.baseline_cpu - 0.0).abs() < f64::EPSILON);
    }

    // ---- Enum variant tests ----
    #[test]
    fn test_allocation_event_type_variants() {
        let variants = [
            AllocationEventType::Allocated,
            AllocationEventType::Deallocated,
            AllocationEventType::Failed,
        ];
        for v in &variants {
            let formatted = format!("{:?}", v);
            assert!(!formatted.is_empty());
        }
    }

    #[test]
    fn test_worker_state_variants() {
        let variants = [
            WorkerState::Idle,
            WorkerState::Busy,
            WorkerState::Starting,
            WorkerState::Stopping,
        ];
        assert_eq!(variants.len(), 4);
    }

    #[test]
    fn test_cleanup_type_variants() {
        let ct = CleanupType::DeleteOldFiles(Duration::from_secs(3600));
        let formatted = format!("{:?}", ct);
        assert!(formatted.contains("DeleteOldFiles"));
    }

    #[test]
    fn test_resource_lifecycle_stage_variants() {
        let stages = [
            ResourceLifecycleStage::Creating,
            ResourceLifecycleStage::Created,
            ResourceLifecycleStage::Active,
            ResourceLifecycleStage::Idle,
            ResourceLifecycleStage::Cleaning,
            ResourceLifecycleStage::Destroying,
            ResourceLifecycleStage::Destroyed,
        ];
        assert_eq!(stages.len(), 7);
    }

    #[test]
    fn test_port_usage_statistics_default_values() {
        let stats = PortUsageStatistics::default();
        assert_eq!(stats.total_allocated, 0);
        assert_eq!(stats.currently_allocated, 0);
        assert_eq!(stats.peak_usage, 0);
        assert_eq!(stats.allocation_failures, 0);
    }

    #[test]
    fn test_directory_usage_info_default() {
        let info = DirectoryUsageInfo::default();
        assert!(info.path.is_empty());
        assert_eq!(info.size_bytes, 0);
        assert_eq!(info.file_count, 0);
    }

    #[test]
    fn test_lcg_deterministic_sequence() {
        let mut rng1 = Lcg::new(42);
        let mut rng2 = Lcg::new(42);
        for _ in 0..10 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_lcg_f64_range() {
        let mut rng = Lcg::new(12345);
        for _ in 0..100 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_lcg_u16_range() {
        let mut rng = Lcg::new(99);
        for _ in 0..100 {
            let _v = rng.next_u16(); // just verify it produces values without panic
        }
    }

    #[test]
    fn test_lcg_next_usize_bound() {
        let mut rng = Lcg::new(777);
        for _ in 0..100 {
            let v = rng.next_usize(10);
            assert!(v < 10);
        }
    }
}
