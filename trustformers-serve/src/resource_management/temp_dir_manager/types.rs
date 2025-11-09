//! Core Types and Error Definitions for Temporary Directory Manager
//!
//! This module contains all the fundamental types, error definitions, and configuration
//! structures used throughout the temporary directory management system.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};
use uuid::Uuid;

use crate::resource_management::types::*;

// ================================================================================================
// Error Types
// ================================================================================================

/// Comprehensive error types for temporary directory operations
#[derive(Debug, thiserror::Error)]
pub enum TempDirError {
    /// Directory allocation failed
    #[error("Failed to allocate directory: {message}")]
    AllocationFailed { message: String },

    /// Directory deallocation failed
    #[error("Failed to deallocate directory '{path}': {message}")]
    DeallocationFailed { path: String, message: String },

    /// Cleanup operation failed
    #[error("Cleanup failed for directory '{path}': {message}")]
    CleanupFailed { path: String, message: String },

    /// Quota exceeded
    #[error("Quota exceeded: requested {requested} bytes, available {available} bytes")]
    QuotaExceeded { requested: u64, available: u64 },

    /// Directory access conflict
    #[error("Directory access conflict for '{path}': {message}")]
    AccessConflict { path: String, message: String },

    /// Permission denied
    #[error("Permission denied for directory '{path}': {message}")]
    PermissionDenied { path: String, message: String },

    /// Directory not found
    #[error("Directory not found: '{path}'")]
    DirectoryNotFound { path: String },

    /// Directory already exists
    #[error("Directory already exists: '{path}'")]
    DirectoryAlreadyExists { path: String },

    /// Disk space insufficient
    #[error("Insufficient disk space: requested {requested} bytes, available {available} bytes")]
    InsufficientDiskSpace { requested: u64, available: u64 },

    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },

    /// Internal system error
    #[error("Internal system error: {message}")]
    InternalError { message: String },

    /// IO operation failed
    #[error("IO operation failed for '{path}': {source}")]
    IoError {
        path: String,
        source: std::io::Error,
    },
}

impl From<std::io::Error> for TempDirError {
    fn from(error: std::io::Error) -> Self {
        TempDirError::IoError {
            path: "unknown".to_string(),
            source: error,
        }
    }
}

// ================================================================================================
// Enhanced Data Types
// ================================================================================================

/// Directory quota management information
#[derive(Debug, Clone)]
pub struct DirectoryQuota {
    /// Maximum total size in bytes
    pub max_total_size: u64,
    /// Maximum number of files
    pub max_file_count: usize,
    /// Maximum subdirectory depth
    pub max_depth: usize,
    /// Current size usage
    pub current_size: Arc<AtomicU64>,
    /// Current file count
    pub current_file_count: Arc<AtomicU64>,
}

impl Default for DirectoryQuota {
    fn default() -> Self {
        Self {
            max_total_size: 1024 * 1024 * 1024, // 1GB default
            max_file_count: 10000,
            max_depth: 10,
            current_size: Arc::new(AtomicU64::new(0)),
            current_file_count: Arc::new(AtomicU64::new(0)),
        }
    }
}

/// Enhanced directory status with detailed state
#[derive(Debug, Clone, PartialEq)]
pub enum EnhancedDirectoryStatus {
    /// Available for allocation
    Available,
    /// Currently allocated to a test
    Allocated {
        test_id: String,
        allocated_at: DateTime<Utc>,
    },
    /// Under cleanup process
    Cleaning { cleanup_started_at: DateTime<Utc> },
    /// Failed or corrupted
    Failed {
        error: String,
        failed_at: DateTime<Utc>,
    },
    /// In maintenance mode
    Maintenance {
        reason: String,
        started_at: DateTime<Utc>,
    },
    /// Quarantined due to issues
    Quarantined {
        reason: String,
        quarantined_at: DateTime<Utc>,
    },
}

/// Directory conflict information
#[derive(Debug, Clone)]
pub struct DirectoryConflict {
    /// Conflicting directory path
    pub path: PathBuf,
    /// Conflict type
    pub conflict_type: ConflictType,
    /// Detected timestamp
    pub detected_at: DateTime<Utc>,
    /// Resolution strategy
    pub resolution_strategy: ConflictResolutionStrategy,
    /// Unique conflict ID
    pub conflict_id: String,
}

impl DirectoryConflict {
    /// Create a new directory conflict
    pub fn new(
        path: PathBuf,
        conflict_type: ConflictType,
        resolution_strategy: ConflictResolutionStrategy,
    ) -> Self {
        Self {
            path,
            conflict_type,
            detected_at: Utc::now(),
            resolution_strategy,
            conflict_id: Uuid::new_v4().to_string(),
        }
    }
}

/// Types of directory conflicts
#[derive(Debug, Clone)]
pub enum ConflictType {
    /// Multiple tests trying to use the same directory
    MultipleTestAccess { test_ids: Vec<String> },
    /// Directory locked by another process
    DirectoryLocked { process_id: Option<u32> },
    /// Permission conflict
    PermissionConflict {
        required_perms: String,
        actual_perms: String,
    },
    /// Disk space conflict
    DiskSpaceConflict { required: u64, available: u64 },
}

/// Conflict resolution strategies
#[derive(Debug, Clone)]
pub enum ConflictResolutionStrategy {
    /// Wait for conflict to resolve
    Wait { timeout: Duration },
    /// Force resolution by terminating conflicting process
    ForceTerminate,
    /// Create alternative directory
    CreateAlternative,
    /// Fail the operation
    Fail,
}

/// Enhanced cleanup task type
#[derive(Debug, Clone)]
pub enum CleanupTaskType {
    /// Delete entire directory
    DeleteDirectory,
    /// Delete files older than specified age
    DeleteOldFiles(Duration),
    /// Empty directory contents but keep the directory
    EmptyDirectory,
    /// Compress old files
    CompressFiles,
    /// Custom cleanup operation
    Custom(String),
    /// Directory cleanup task
    DirectoryCleanup,
    /// Port release task
    PortRelease,
    /// GPU resource release task
    GpuRelease,
    /// Database cleanup task
    DatabaseCleanup,
    /// Custom resource cleanup task
    CustomResourceCleanup(String),
    /// Garbage collection task
    GarbageCollection,
}

/// Enhanced cleanup priority
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub enum CleanupPriority {
    /// Low priority (can be deferred)
    Low = 0,
    /// Normal priority
    Normal = 1,
    /// High priority
    High = 2,
    /// Critical priority (immediate)
    Critical = 3,
}

/// Directory space usage breakdown
#[derive(Debug, Clone)]
pub struct DirectorySpaceUsage {
    /// Total size in bytes
    pub total_size: u64,
    /// Number of files
    pub file_count: usize,
    /// Number of subdirectories
    pub subdirectory_count: usize,
    /// Largest file size
    pub largest_file_size: u64,
    /// Average file size
    pub average_file_size: u64,
    /// Space usage by file type
    pub usage_by_type: HashMap<String, u64>,
}

impl DirectorySpaceUsage {
    /// Create a new empty usage breakdown
    pub fn new() -> Self {
        Self {
            total_size: 0,
            file_count: 0,
            subdirectory_count: 0,
            largest_file_size: 0,
            average_file_size: 0,
            usage_by_type: HashMap::new(),
        }
    }

    /// Update average file size based on current metrics
    pub fn update_average_file_size(&mut self) {
        if self.file_count > 0 {
            self.average_file_size = self.total_size / self.file_count as u64;
        } else {
            self.average_file_size = 0;
        }
    }

    /// Add usage from another breakdown
    pub fn merge(&mut self, other: &DirectorySpaceUsage) {
        self.total_size += other.total_size;
        self.file_count += other.file_count;
        self.subdirectory_count += other.subdirectory_count;
        self.largest_file_size = self.largest_file_size.max(other.largest_file_size);

        for (file_type, size) in &other.usage_by_type {
            *self.usage_by_type.entry(file_type.clone()).or_insert(0) += size;
        }

        self.update_average_file_size();
    }
}

impl Default for DirectorySpaceUsage {
    fn default() -> Self {
        Self::new()
    }
}

// ================================================================================================
// Cleanup System Types
// ================================================================================================

/// Cleanup task for directory management
#[derive(Debug, Clone)]
pub struct CleanupTask {
    /// Unique task identifier
    pub id: String,
    /// Directory path to clean
    pub directory_path: PathBuf,
    /// Test ID that owns the directory
    pub test_id: String,
    /// Cleanup policy to apply
    pub policy: TempDirectoryCleanupPolicy,
    /// Priority level for execution
    pub priority: CleanupPriority,
    /// When the task was scheduled
    pub scheduled_time: DateTime<Utc>,
    /// Expected execution time
    pub expected_execution_time: Option<DateTime<Utc>>,
    /// Task type
    pub task_type: CleanupTaskType,
}

impl CleanupTask {
    /// Create a new cleanup task
    pub fn new(
        directory_path: PathBuf,
        test_id: String,
        policy: TempDirectoryCleanupPolicy,
        priority: CleanupPriority,
        task_type: CleanupTaskType,
    ) -> Self {
        let scheduled_time = Utc::now();
        let expected_execution_time = match &policy {
            TempDirectoryCleanupPolicy::Immediate => Some(scheduled_time),
            TempDirectoryCleanupPolicy::Delayed(delay) => {
                Some(scheduled_time + chrono::Duration::from_std(*delay).unwrap_or_default())
            },
            TempDirectoryCleanupPolicy::SessionEnd => None,
            TempDirectoryCleanupPolicy::Manual => None,
            TempDirectoryCleanupPolicy::Debug => None,
        };

        Self {
            id: Uuid::new_v4().to_string(),
            directory_path,
            test_id,
            policy,
            priority,
            scheduled_time,
            expected_execution_time,
            task_type,
        }
    }

    /// Check if the task is ready for execution
    pub fn is_ready(&self, current_time: DateTime<Utc>) -> bool {
        match &self.expected_execution_time {
            Some(exec_time) => current_time >= *exec_time,
            None => false, // Manual or debug tasks are not automatically ready
        }
    }
}

/// Cleanup event for tracking cleanup operations
#[derive(Debug, Clone)]
pub struct CleanupEvent {
    /// Timestamp of the event
    pub timestamp: DateTime<Utc>,
    /// Type of cleanup event
    pub event_type: CleanupEventType,
    /// Directory path that was cleaned
    pub directory_path: PathBuf,
    /// Test ID associated with the directory
    pub test_id: String,
    /// Duration of the cleanup operation
    pub duration: Duration,
    /// Result of the cleanup operation
    pub result: CleanupResult,
    /// Additional event details
    pub details: HashMap<String, String>,
}

/// Types of cleanup events
#[derive(Debug, Clone, PartialEq)]
pub enum CleanupEventType {
    /// Cleanup started
    Started,
    /// Cleanup completed successfully
    Completed,
    /// Cleanup failed
    Failed,
    /// Cleanup was cancelled
    Cancelled,
    /// Cleanup was deferred
    Deferred,
}

/// Result of a cleanup operation
#[derive(Debug, Clone)]
pub enum CleanupResult {
    /// Successful cleanup
    Success {
        files_removed: usize,
        bytes_freed: u64,
    },
    /// Failed cleanup
    Failed {
        error: String,
        files_attempted: usize,
    },
    /// Cleanup was skipped
    Skipped { reason: String },
}

/// Statistics for cleanup operations
#[derive(Debug, Clone, Default)]
pub struct CleanupStatistics {
    /// Total number of cleanups attempted
    pub total_cleanups: u64,
    /// Number of successful cleanups
    pub successful_cleanups: u64,
    /// Number of failed cleanups
    pub failed_cleanups: u64,
    /// Total bytes freed through cleanup
    pub total_bytes_freed: u64,
    /// Total files removed
    pub total_files_removed: usize,
    /// Average cleanup time
    pub average_cleanup_time: Duration,
    /// Cleanup efficiency (success rate)
    pub cleanup_efficiency: f32,
    /// Last cleanup timestamp
    pub last_cleanup_time: Option<DateTime<Utc>>,
}

impl CleanupStatistics {
    /// Update statistics after a cleanup operation
    pub fn update_after_cleanup(&mut self, result: &CleanupResult, duration: Duration) {
        self.total_cleanups += 1;
        self.last_cleanup_time = Some(Utc::now());

        match result {
            CleanupResult::Success {
                files_removed,
                bytes_freed,
            } => {
                self.successful_cleanups += 1;
                self.total_files_removed += files_removed;
                self.total_bytes_freed += bytes_freed;
            },
            CleanupResult::Failed { .. } => {
                self.failed_cleanups += 1;
            },
            CleanupResult::Skipped { .. } => {
                // Skipped cleanups are not counted as successes or failures
            },
        }

        // Update average cleanup time
        let total_duration_secs =
            self.average_cleanup_time.as_secs() as f64 * (self.total_cleanups - 1) as f64;
        let new_average_secs =
            (total_duration_secs + duration.as_secs() as f64) / self.total_cleanups as f64;
        self.average_cleanup_time = Duration::from_secs(new_average_secs as u64);

        // Update efficiency
        if self.total_cleanups > 0 {
            self.cleanup_efficiency = self.successful_cleanups as f32 / self.total_cleanups as f32;
        }
    }

    /// Generate a summary report of cleanup statistics
    pub fn generate_report(&self) -> String {
        format!(
            "Cleanup Statistics:\n\
             Total cleanups: {}\n\
             Successful: {} ({:.1}%)\n\
             Failed: {} ({:.1}%)\n\
             Total bytes freed: {} bytes\n\
             Total files removed: {}\n\
             Average cleanup time: {}s\n\
             Last cleanup: {}",
            self.total_cleanups,
            self.successful_cleanups,
            if self.total_cleanups > 0 {
                (self.successful_cleanups as f32 / self.total_cleanups as f32) * 100.0
            } else {
                0.0
            },
            self.failed_cleanups,
            if self.total_cleanups > 0 {
                (self.failed_cleanups as f32 / self.total_cleanups as f32) * 100.0
            } else {
                0.0
            },
            self.total_bytes_freed,
            self.total_files_removed,
            self.average_cleanup_time.as_secs(),
            self.last_cleanup_time
                .as_ref()
                .map(|t| t.to_rfc3339())
                .unwrap_or_else(|| "Never".to_string())
        )
    }
}

// ================================================================================================
// Configuration and Management Types
// ================================================================================================

/// Configuration for the temp directory manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TempDirectoryManagerConfig {
    /// Base configuration from the pool config
    pub pool_config: TempDirPoolConfig,
    /// Enable detailed logging
    pub enable_detailed_logging: bool,
    /// Conflict resolution timeout
    pub conflict_resolution_timeout: Duration,
    /// Maximum concurrent cleanup operations
    pub max_concurrent_cleanups: usize,
    /// Statistics retention period
    pub stats_retention_period: Duration,
    /// Enable performance monitoring
    pub enable_performance_monitoring: bool,
}

impl Default for TempDirectoryManagerConfig {
    fn default() -> Self {
        Self {
            pool_config: TempDirPoolConfig::default(),
            enable_detailed_logging: true,
            conflict_resolution_timeout: Duration::from_secs(30),
            max_concurrent_cleanups: 4,
            stats_retention_period: Duration::from_secs(24 * 60 * 60), // 24 hours
            enable_performance_monitoring: true,
        }
    }
}

/// Manager instance information
#[derive(Debug, Clone)]
pub struct ManagerInstanceInfo {
    /// Unique instance identifier
    pub instance_id: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Configuration used
    pub config: TempDirectoryManagerConfig,
    /// Current operational status
    pub status: ManagerStatus,
}

/// Status of the directory manager
#[derive(Debug, Clone, PartialEq)]
pub enum ManagerStatus {
    /// Manager is initializing
    Initializing,
    /// Manager is operational
    Active,
    /// Manager is shutting down
    ShuttingDown,
    /// Manager is shut down
    Shutdown,
    /// Manager encountered an error
    Error { message: String },
}

// ================================================================================================
// Utility Types and Traits
// ================================================================================================

/// Trait for objects that can provide disk space usage information
pub trait DiskSpaceProvider {
    /// Get available disk space in bytes
    fn get_available_space(&self) -> Result<u64, TempDirError>;

    /// Get total disk space in bytes
    fn get_total_space(&self) -> Result<u64, TempDirError>;

    /// Get used disk space in bytes
    fn get_used_space(&self) -> Result<u64, TempDirError>;
}

/// Trait for conflict detection and resolution
pub trait ConflictDetector {
    /// Check for conflicts before allocation
    fn check_conflicts(&self, test_id: &str, count: usize) -> Option<DirectoryConflict>;

    /// Resolve a detected conflict
    fn resolve_conflict(&self, conflict: &DirectoryConflict) -> Result<(), TempDirError>;
}

/// Trait for cleanup scheduling and execution
pub trait CleanupScheduler {
    /// Schedule a cleanup task
    fn schedule_cleanup(&self, task: CleanupTask) -> Result<(), TempDirError>;

    /// Execute pending cleanup tasks
    fn execute_pending_cleanups(&self) -> Result<usize, TempDirError>;

    /// Get cleanup statistics
    fn get_cleanup_statistics(&self) -> CleanupStatistics;
}

/// Result type alias for temporary directory operations
pub type TempDirResult<T> = std::result::Result<T, TempDirError>;

/// Type alias for shared atomic counters
pub type SharedCounter = Arc<AtomicU64>;

/// Create a new shared counter with initial value
pub fn new_shared_counter(initial: u64) -> SharedCounter {
    Arc::new(AtomicU64::new(initial))
}

/// Increment a shared counter and return the new value
pub fn increment_counter(counter: &SharedCounter) -> u64 {
    counter.fetch_add(1, Ordering::SeqCst) + 1
}

/// Decrement a shared counter and return the new value
pub fn decrement_counter(counter: &SharedCounter) -> u64 {
    counter.fetch_sub(1, Ordering::SeqCst).saturating_sub(1)
}

/// Get the current value of a shared counter
pub fn get_counter_value(counter: &SharedCounter) -> u64 {
    counter.load(Ordering::SeqCst)
}
