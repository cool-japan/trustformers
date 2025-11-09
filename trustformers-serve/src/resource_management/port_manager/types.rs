//! Port Manager Types for TrustformeRS
//!
//! This module contains all types, configurations, and data structures specifically
//! used by the NetworkPortManager system. These types provide the foundation for
//! advanced port allocation, reservation, conflict detection, and monitoring capabilities.
//!
//! # Type Organization
//!
//! The types are organized into logical categories:
//! - **Error Types**: All error handling and result types
//! - **Configuration Types**: Configuration structures for various subsystems
//! - **Manager Structures**: Core manager and system structures
//! - **Event Types**: Event tracking and history types
//! - **Conflict Management**: Conflict detection and resolution types
//! - **Health Monitoring**: Health status and monitoring types
//! - **Performance Metrics**: Performance tracking and analysis types
//! - **Type Aliases**: Convenient type aliases for common patterns

use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{
        atomic::{AtomicBool, AtomicU64},
        Arc,
    },
    time::Duration,
};
use thiserror::Error;

// Re-import types from parent module to avoid duplication
use super::super::types::{PortPoolConfig, PortAllocation, PortReservationRequest, PortUsageStatistics, PortUsageType, HealthStatus};

// ================================
// Error Types
// ================================

/// Custom error types for port management operations
///
/// This enum covers all possible failure scenarios in the port management system,
/// from resource exhaustion to configuration errors and internal failures.
#[derive(Error, Debug)]
pub enum PortManagementError {
    /// Not enough ports available for the requested allocation
    #[error("Insufficient available ports: requested {requested}, available {available}")]
    InsufficientPorts {
        /// Number of ports requested
        requested: usize,
        /// Number of ports actually available
        available: usize
    },

    /// Attempt to deallocate a port that is not currently allocated
    #[error("Port {port} is not allocated")]
    PortNotAllocated {
        /// Port number that was not allocated
        port: u16
    },

    /// Attempt to allocate a port that is already in use
    #[error("Port {port} is already allocated to test {test_id}")]
    PortAlreadyAllocated {
        /// Port number that is already allocated
        port: u16,
        /// Test ID that currently owns the port
        test_id: String
    },

    /// Attempt to allocate a port that is currently reserved
    #[error("Port {port} is reserved by test {test_id}")]
    PortReserved {
        /// Port number that is reserved
        port: u16,
        /// Test ID that has the reservation
        test_id: String
    },

    /// Attempt to allocate a port in an excluded range
    #[error("Port {port} is in excluded range")]
    PortExcluded {
        /// Port number that is excluded
        port: u16
    },

    /// Invalid port range specified in configuration
    #[error("Invalid port range: {start} to {end}")]
    InvalidPortRange {
        /// Start of the invalid range
        start: u16,
        /// End of the invalid range
        end: u16
    },

    /// Configuration validation or parsing error
    #[error("Configuration error: {message}")]
    ConfigurationError {
        /// Description of the configuration error
        message: String
    },

    /// Resource conflict detected during operation
    #[error("Resource conflict detected: {details}")]
    ResourceConflict {
        /// Details about the conflict
        details: String
    },

    /// Operation timed out before completion
    #[error("Operation timeout after {timeout_secs} seconds")]
    OperationTimeout {
        /// Number of seconds before timeout
        timeout_secs: u64
    },

    /// Internal system error that shouldn't normally occur
    #[error("Internal error: {message}")]
    InternalError {
        /// Description of the internal error
        message: String
    },
}

/// Result type alias for port management operations
///
/// This provides a convenient shorthand for Results that may return PortManagementError.
pub type PortManagementResult<T> = Result<T, PortManagementError>;

// ================================
// Configuration Types
// ================================

/// Configuration for port reservation behavior and limits
///
/// This configuration controls how the reservation system operates, including
/// limits on reservations per test, duration constraints, and queue management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortReservationConfig {
    /// Maximum number of reservations allowed per test
    ///
    /// This prevents any single test from monopolizing the reservation system.
    pub max_reservations_per_test: usize,

    /// Maximum reservation duration allowed
    ///
    /// Reservations longer than this will be capped to this duration.
    pub max_reservation_duration: Duration,

    /// Default reservation duration when none is specified
    ///
    /// Used when a reservation request doesn't specify a duration.
    pub default_reservation_duration: Duration,

    /// Enable automatic queue processing for unfulfilled reservations
    ///
    /// When enabled, reservations that can't be immediately fulfilled are queued
    /// and processed automatically when resources become available.
    pub enable_queue_processing: bool,

    /// Interval between queue processing attempts
    ///
    /// How often the system checks the queue for reservations that can now be fulfilled.
    pub queue_processing_interval: Duration,

    /// Maximum number of reservations that can be queued
    ///
    /// Prevents the queue from growing unbounded during high-demand periods.
    pub max_queue_size: usize,
}

impl Default for PortReservationConfig {
    fn default() -> Self {
        Self {
            max_reservations_per_test: 10,
            max_reservation_duration: Duration::from_secs(3600), // 1 hour
            default_reservation_duration: Duration::from_secs(300), // 5 minutes
            enable_queue_processing: true,
            queue_processing_interval: Duration::from_secs(30),
            max_queue_size: 1000,
        }
    }
}

/// Configuration for port health monitoring system
///
/// Controls how the health monitoring system operates, including check intervals,
/// data retention, and alert generation.
#[derive(Debug, Clone)]
pub struct PortHealthConfig {
    /// Enable health monitoring
    ///
    /// When disabled, health checks are not performed and status remains unknown.
    pub enabled: bool,

    /// Interval between health checks
    ///
    /// How often the system performs comprehensive health assessments.
    pub check_interval: Duration,

    /// Number of health events to keep in history
    ///
    /// Older events beyond this limit are automatically purged.
    pub history_size: usize,

    /// Enable automatic alert generation
    ///
    /// When enabled, alerts are automatically generated based on threshold violations.
    pub enable_alerts: bool,
}

impl Default for PortHealthConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            check_interval: Duration::from_secs(60),
            history_size: 1000,
            enable_alerts: true,
        }
    }
}

/// Alert thresholds for port health monitoring
///
/// Defines the thresholds that trigger warnings and critical alerts in the
/// port management system.
#[derive(Debug, Clone)]
pub struct PortHealthThresholds {
    /// Warning threshold for port utilization percentage (0.0 to 100.0)
    ///
    /// Alerts are generated when utilization exceeds this threshold.
    pub utilization_warning: f32,

    /// Critical threshold for port utilization percentage (0.0 to 100.0)
    ///
    /// Critical alerts are generated when utilization exceeds this threshold.
    pub utilization_critical: f32,

    /// Warning threshold for conflicts per minute
    ///
    /// Alerts are generated when conflict rate exceeds this threshold.
    pub conflicts_per_minute_warning: f32,

    /// Critical threshold for conflicts per minute
    ///
    /// Critical alerts are generated when conflict rate exceeds this threshold.
    pub conflicts_per_minute_critical: f32,

    /// Warning threshold for average allocation time in milliseconds
    ///
    /// Alerts are generated when allocation time exceeds this threshold.
    pub allocation_time_warning_ms: f64,

    /// Critical threshold for average allocation time in milliseconds
    ///
    /// Critical alerts are generated when allocation time exceeds this threshold.
    pub allocation_time_critical_ms: f64,
}

impl Default for PortHealthThresholds {
    fn default() -> Self {
        Self {
            utilization_warning: 80.0,
            utilization_critical: 95.0,
            conflicts_per_minute_warning: 10.0,
            conflicts_per_minute_critical: 50.0,
            allocation_time_warning_ms: 100.0,
            allocation_time_critical_ms: 500.0,
        }
    }
}

/// Configuration for performance monitoring and metrics collection
///
/// Controls how performance metrics are collected, stored, and analyzed
/// within the port management system.
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    /// Enable performance monitoring
    ///
    /// When disabled, performance metrics are not collected.
    pub enabled: bool,

    /// Interval between performance snapshots
    ///
    /// How often comprehensive performance data is captured.
    pub snapshot_interval: Duration,

    /// Number of performance snapshots to keep in history
    ///
    /// Older snapshots beyond this limit are automatically purged.
    pub history_size: usize,

    /// Enable detailed timing measurements
    ///
    /// When enabled, additional timing metrics are collected at the cost of some overhead.
    pub enable_detailed_timing: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            snapshot_interval: Duration::from_secs(300), // 5 minutes
            history_size: 288, // 24 hours worth of 5-minute snapshots
            enable_detailed_timing: false,
        }
    }
}

// ================================
// Manager Structures
// ================================

/// Comprehensive network port management system
///
/// This is the main entry point for all port management operations. It provides
/// thread-safe port allocation, reservation, and monitoring capabilities designed
/// for high-concurrency test execution environments.
///
/// The manager coordinates between several subsystems:
/// - Port allocation and deallocation
/// - Advanced reservation system
/// - Conflict detection and resolution
/// - Health monitoring and alerting
/// - Performance metrics collection
#[derive(Debug, Clone)]
pub struct NetworkPortManager {
    /// Configuration for port management
    pub config: Arc<RwLock<PortPoolConfig>>,

    /// Available ports for allocation
    pub available_ports: Arc<Mutex<HashSet<u16>>>,

    /// Currently allocated ports with their allocation details
    pub allocated_ports: Arc<Mutex<HashMap<u16, PortAllocation>>>,

    /// Advanced port reservation system
    pub reservation_system: Arc<PortReservationSystem>,

    /// Comprehensive usage statistics
    pub usage_stats: Arc<Mutex<PortUsageStatistics>>,

    /// Port conflict detector
    pub conflict_detector: Arc<PortConflictDetector>,

    /// Health monitoring system
    pub health_monitor: Arc<PortHealthMonitor>,

    /// Performance metrics
    pub performance_metrics: Arc<PortPerformanceMetrics>,

    /// Shutdown signal for background tasks
    pub shutdown_signal: Arc<AtomicBool>,
}

/// Advanced port reservation system with conflict detection and queuing
///
/// This system handles port reservations for future use, including:
/// - Immediate reservations when ports are available
/// - Queuing system for delayed fulfillment
/// - Automatic expiry and cleanup
/// - Comprehensive event tracking
#[derive(Debug)]
pub struct PortReservationSystem {
    /// Active reservations by port number
    pub reservations: Arc<Mutex<HashMap<u16, PortReservationRequest>>>,

    /// Reservations by test ID for quick lookup
    pub reservations_by_test: Arc<Mutex<HashMap<String, Vec<u16>>>>,

    /// Reservation queue for when ports are not immediately available
    pub reservation_queue: Arc<Mutex<VecDeque<PortReservationRequest>>>,

    /// Reservation expiry times for automatic cleanup
    pub expiry_times: Arc<Mutex<HashMap<u16, DateTime<Utc>>>>,

    /// Reservation history for analytics
    pub reservation_history: Arc<Mutex<Vec<PortReservationEvent>>>,

    /// Configuration for reservation behavior
    pub config: Arc<RwLock<PortReservationConfig>>,
}

/// Port conflict detection and resolution system
///
/// This system automatically detects various types of port conflicts and
/// can apply resolution strategies based on configurable rules.
#[derive(Debug)]
pub struct PortConflictDetector {
    /// Known conflicts and their resolution strategies
    pub conflict_rules: Arc<RwLock<Vec<ConflictRule>>>,

    /// Conflict detection enabled flag
    pub enabled: AtomicBool,

    /// Conflict history for analysis
    pub conflict_history: Arc<Mutex<Vec<PortConflictEvent>>>,
}

/// Health monitoring system for port management
///
/// Continuously monitors the health of the port management system and
/// generates alerts when issues are detected.
#[derive(Debug)]
pub struct PortHealthMonitor {
    /// Current health check results
    pub health_status: Arc<Mutex<PortHealthStatus>>,

    /// Health check configuration
    pub config: Arc<RwLock<PortHealthConfig>>,

    /// Health check history
    pub health_history: Arc<Mutex<Vec<PortHealthEvent>>>,

    /// Alert thresholds
    pub alert_thresholds: Arc<RwLock<PortHealthThresholds>>,
}

/// Performance metrics collection for port management
///
/// Tracks detailed performance metrics for the port management system,
/// including allocation times, throughput, and resource utilization.
#[derive(Debug)]
pub struct PortPerformanceMetrics {
    /// Total allocations performed
    pub total_allocations: AtomicU64,

    /// Total deallocations performed
    pub total_deallocations: AtomicU64,

    /// Total reservations made
    pub total_reservations: AtomicU64,

    /// Total conflicts detected
    pub total_conflicts: AtomicU64,

    /// Cumulative allocation time in nanoseconds
    pub cumulative_allocation_time_ns: AtomicU64,

    /// Performance history
    pub performance_history: Arc<Mutex<Vec<PerformanceSnapshot>>>,

    /// Performance configuration
    pub config: Arc<RwLock<PerformanceConfig>>,
}

// ================================
// Event Types
// ================================

/// Port reservation event for history tracking
///
/// Records significant events in the reservation system for analytics
/// and debugging purposes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortReservationEvent {
    /// Timestamp when the event occurred
    pub timestamp: DateTime<Utc>,

    /// Type of reservation event
    pub event_type: ReservationEventType,

    /// Port number involved in the event
    pub port: u16,

    /// Test ID that triggered the event
    pub test_id: String,

    /// Additional event details and metadata
    pub details: HashMap<String, String>,
}

/// Types of reservation events that can occur
///
/// Each variant represents a significant state change or action
/// within the reservation system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReservationEventType {
    /// Reservation was successfully created
    Created,

    /// Reservation was fulfilled and converted to an allocation
    Fulfilled,

    /// Reservation was cancelled by the requesting test
    Cancelled,

    /// Reservation expired due to timeout
    Expired,

    /// Reservation was queued because resources were not immediately available
    Queued,

    /// Reservation encountered a conflict during processing
    Conflict,
}

/// Port conflict event for tracking and analysis
///
/// Records detailed information about port conflicts for system analysis
/// and conflict resolution improvement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortConflictEvent {
    /// Timestamp when the conflict was detected
    pub timestamp: DateTime<Utc>,

    /// Port number that had the conflict
    pub port: u16,

    /// Test ID that requested the conflicting port
    pub requesting_test_id: String,

    /// Test ID that currently owns the port (if applicable)
    pub current_owner_test_id: Option<String>,

    /// Classification of the conflict type
    pub conflict_type: ConflictType,

    /// Action taken to resolve the conflict
    pub resolution_action: ConflictAction,

    /// Whether the conflict was successfully resolved
    pub resolved: bool,

    /// Additional conflict details and context
    pub details: HashMap<String, String>,
}

/// Health event for tracking system health over time
///
/// Records health check results and system status changes for
/// trend analysis and issue diagnosis.
#[derive(Debug, Clone)]
pub struct PortHealthEvent {
    /// Timestamp when the health event was recorded
    pub timestamp: DateTime<Utc>,

    /// Overall health status at the time of the event
    pub status: HealthStatus,

    /// Detailed metrics captured during this event
    pub metrics: HashMap<String, f64>,

    /// Any alerts that were generated
    pub alerts: Vec<String>,

    /// Additional event details and context
    pub details: HashMap<String, String>,
}

// ================================
// Conflict Management Types
// ================================

/// Port conflict rule for automatic resolution
///
/// Defines a rule that can automatically detect and resolve specific
/// types of port conflicts based on configured conditions and actions.
#[derive(Debug, Clone)]
pub struct ConflictRule {
    /// Human-readable name for the rule
    pub name: String,

    /// Condition that triggers this rule
    pub condition: ConflictCondition,

    /// Action to take when the condition is met
    pub action: ConflictAction,

    /// Priority of this rule (higher values = higher priority)
    pub priority: u32,

    /// Whether this rule is currently enabled
    pub enabled: bool,
}

/// Conditions that can trigger conflict resolution rules
///
/// Each variant represents a specific type of conflict condition
/// that the system can automatically detect.
#[derive(Debug, Clone)]
pub enum ConflictCondition {
    /// Port is already allocated to another test
    PortAlreadyAllocated,

    /// Port is reserved by a different test
    PortReserved,

    /// Port is in a configured excluded range
    PortExcluded,

    /// Port is a well-known system port (< 1024 or common services)
    WellKnownPort,

    /// Custom condition with a string identifier
    Custom(String),
}

/// Actions that can be taken to resolve port conflicts
///
/// Each variant represents a specific strategy for handling
/// conflicts when they are detected.
#[derive(Debug, Clone)]
pub enum ConflictAction {
    /// Deny the allocation or reservation request
    Deny,

    /// Automatically find an alternative available port
    FindAlternative,

    /// Queue the request for later processing when resources are available
    Queue,

    /// Force the allocation, overriding the existing allocation
    ForceAllocate,

    /// Custom action with a string identifier
    Custom(String),
}

/// Types of port conflicts that can be detected
///
/// Classifies the different categories of conflicts that can occur
/// during port allocation and reservation operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictType {
    /// Port is already allocated to another test
    AlreadyAllocated,

    /// Port is reserved by a different test
    Reserved,

    /// Port is in a configured excluded range
    Excluded,

    /// Port is a well-known system port
    WellKnown,

    /// Port is outside the configured allocation range
    OutOfRange,

    /// Custom conflict type with description
    Custom(String),
}

// ================================
// Health Monitoring Types
// ================================

/// Current health status of the port management system
///
/// Provides a comprehensive view of the system's current state,
/// including utilization metrics, performance indicators, and active alerts.
#[derive(Debug, Clone)]
pub struct PortHealthStatus {
    /// Overall system health status
    pub overall_status: HealthStatus,

    /// Timestamp of the last health check
    pub last_check: DateTime<Utc>,

    /// Number of ports currently available for allocation
    pub available_ports: usize,

    /// Number of ports currently allocated to tests
    pub allocated_ports: usize,

    /// Number of ports currently reserved
    pub reserved_ports: usize,

    /// Current port utilization as a percentage (0.0 to 100.0)
    pub utilization_percent: f32,

    /// Number of conflicts detected in the recent check period
    pub recent_conflicts: usize,

    /// Average time taken for port allocations in milliseconds
    pub avg_allocation_time_ms: f64,

    /// List of currently active alerts
    pub active_alerts: Vec<String>,
}

// ================================
// Performance Metrics Types
// ================================

/// Performance snapshot for historical analysis
///
/// Captures a point-in-time view of system performance metrics
/// for trend analysis and capacity planning.
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    /// Timestamp when the snapshot was taken
    pub timestamp: DateTime<Utc>,

    /// Operations performed per second during the measurement period
    pub ops_per_second: f64,

    /// Average time taken for port allocations in milliseconds
    pub avg_allocation_time_ms: f64,

    /// Percentage of successful operations (0.0 to 100.0)
    pub success_rate_percent: f64,

    /// Percentage of operations that resulted in conflicts (0.0 to 100.0)
    pub conflict_rate_percent: f64,

    /// System utilization percentage (0.0 to 100.0)
    pub utilization_percent: f32,

    /// Additional custom metrics collected during the snapshot
    pub metrics: HashMap<String, f64>,
}

// ================================
// Default Implementations
// ================================

impl Default for PortReservationSystem {
    fn default() -> Self {
        Self {
            reservations: Arc::new(Mutex::new(HashMap::new())),
            reservations_by_test: Arc::new(Mutex::new(HashMap::new())),
            reservation_queue: Arc::new(Mutex::new(VecDeque::new())),
            expiry_times: Arc::new(Mutex::new(HashMap::new())),
            reservation_history: Arc::new(Mutex::new(Vec::new())),
            config: Arc::new(RwLock::new(PortReservationConfig::default())),
        }
    }
}

// ================================
// Type Aliases
// ================================

/// Type alias for port management operation results
pub type PortResult<T> = Result<T, PortManagementError>;

/// Type alias for health monitoring results
pub type HealthResult<T> = Result<T, PortManagementError>;

/// Type alias for performance metrics results
pub type MetricsResult<T> = Result<T, PortManagementError>;

/// Type alias for conflict resolution results
pub type ConflictResult<T> = Result<T, PortManagementError>;

/// Type alias for reservation system results
pub type ReservationResult<T> = Result<T, PortManagementError>;