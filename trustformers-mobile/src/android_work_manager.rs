//! Android Work Manager Integration for TrustformeRS Mobile
//!
//! This module provides comprehensive Android WorkManager integration, enabling
//! TrustformeRS to schedule and execute background tasks, deferred inference,
//! model updates, and federated learning operations that survive app restarts
//! and device reboots.

use crate::{
    device_info::{MobileDeviceDetector, MobileDeviceInfo},
    inference::MobileInferenceEngine,
    model_management::ModelManager,
    MemoryOptimization, MobileConfig,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use trustformers_core::errors::Result;
use trustformers_core::{Tensor, TrustformersError};

/// Android Work Manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AndroidWorkManagerConfig {
    /// Enable periodic work scheduling
    pub enable_periodic_work: bool,
    /// Enable one-time work requests
    pub enable_one_time_work: bool,
    /// Enable expedited work for urgent tasks
    pub enable_expedited_work: bool,
    /// Work constraints configuration
    pub constraints: WorkConstraintsConfig,
    /// Retry policy configuration
    pub retry_policy: WorkRetryPolicyConfig,
    /// Background execution configuration
    pub background_execution: BackgroundExecutionConfig,
    /// Data synchronization configuration
    pub data_sync: DataSyncConfig,
}

/// Work constraints configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkConstraintsConfig {
    /// Require unmetered network (WiFi)
    pub require_unmetered_network: bool,
    /// Require charging
    pub require_charging: bool,
    /// Require device idle
    pub require_device_idle: bool,
    /// Require battery not low
    pub require_battery_not_low: bool,
    /// Required network type
    pub required_network_type: WorkNetworkType,
    /// Storage constraints
    pub storage_constraints: StorageConstraints,
}

/// Required network types for work
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkNetworkType {
    /// No network required
    NotRequired,
    /// Any network connection
    Connected,
    /// Unmetered network only (WiFi)
    Unmetered,
    /// Not roaming
    NotRoaming,
    /// Metered network allowed
    Metered,
}

/// Storage constraints for work
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConstraints {
    /// Minimum free storage in MB
    pub min_free_storage_mb: usize,
    /// Maximum storage usage for cache in MB
    pub max_cache_storage_mb: usize,
    /// Enable storage cleanup
    pub enable_storage_cleanup: bool,
}

/// Work retry policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkRetryPolicyConfig {
    /// Retry policy type
    pub retry_policy: WorkRetryPolicy,
    /// Maximum retry attempts
    pub max_retry_attempts: usize,
    /// Initial retry delay in milliseconds
    pub initial_retry_delay_ms: f64,
    /// Maximum retry delay in milliseconds
    pub max_retry_delay_ms: f64,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
}

/// Work retry policies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkRetryPolicy {
    /// Linear backoff
    Linear,
    /// Exponential backoff
    Exponential,
    /// No retry
    None,
}

/// Background execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundExecutionConfig {
    /// Maximum execution time in seconds
    pub max_execution_time_seconds: f64,
    /// Enable foreground service for long-running tasks
    pub enable_foreground_service: bool,
    /// Foreground service notification configuration
    pub foreground_notification: ForegroundNotificationConfig,
    /// Background task prioritization
    pub task_prioritization: TaskPrioritizationConfig,
}

/// Foreground notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForegroundNotificationConfig {
    /// Notification channel ID
    pub channel_id: String,
    /// Notification title
    pub title: String,
    /// Notification content text
    pub content_text: String,
    /// Enable progress indicator
    pub show_progress: bool,
    /// Enable cancel action
    pub enable_cancel_action: bool,
}

/// Task prioritization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskPrioritizationConfig {
    /// High priority task types
    pub high_priority_tasks: Vec<WorkTaskType>,
    /// Enable adaptive prioritization
    pub enable_adaptive_prioritization: bool,
    /// Priority adjustment based on device state
    pub device_state_priority_adjustment: bool,
    /// Task execution order strategy
    pub execution_order: TaskExecutionOrder,
}

/// Task execution order strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskExecutionOrder {
    /// First In, First Out
    FIFO,
    /// Last In, First Out
    LIFO,
    /// Priority-based
    Priority,
    /// Deadline-based
    Deadline,
    /// Resource-aware
    ResourceAware,
}

/// Data synchronization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSyncConfig {
    /// Enable automatic model updates
    pub enable_model_updates: bool,
    /// Enable federated learning sync
    pub enable_federated_sync: bool,
    /// Sync frequency for periodic tasks
    pub sync_frequency: WorkFrequency,
    /// Conflict resolution strategy
    pub conflict_resolution: ConflictResolutionStrategy,
    /// Data compression settings
    pub compression_settings: DataCompressionConfig,
}

/// Work frequency settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkFrequency {
    /// Interval in minutes
    pub interval_minutes: usize,
    /// Flex interval in minutes
    pub flex_interval_minutes: usize,
    /// Initial delay in minutes
    pub initial_delay_minutes: usize,
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictResolutionStrategy {
    /// Server wins
    ServerWins,
    /// Client wins
    ClientWins,
    /// Last modified wins
    LastModifiedWins,
    /// Merge conflicts
    Merge,
    /// Manual resolution required
    Manual,
}

/// Data compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCompressionConfig {
    /// Enable compression
    pub enable_compression: bool,
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level (1-9)
    pub compression_level: u8,
    /// Minimum size threshold for compression (bytes)
    pub min_size_threshold_bytes: usize,
}

/// Compression algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// GZIP compression
    Gzip,
    /// LZ4 compression
    LZ4,
    /// Brotli compression
    Brotli,
    /// Snappy compression
    Snappy,
}

/// Work task types
// Not `Copy`: the `Custom(String)` variant holds a `String`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkTaskType {
    /// Model inference task
    Inference,
    /// Model download task
    ModelDownload,
    /// Model update task
    ModelUpdate,
    /// Federated learning task
    FederatedLearning,
    /// Data preprocessing task
    DataPreprocessing,
    /// Cache cleanup task
    CacheCleanup,
    /// Performance profiling task
    PerformanceProfiling,
    /// Health check task
    HealthCheck,
    /// Custom task
    Custom(String),
}

/// Work request specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkRequest {
    /// Unique work ID
    pub work_id: String,
    /// Work type
    pub work_type: WorkRequestType,
    /// Task type
    pub task_type: WorkTaskType,
    /// Work priority
    pub priority: WorkPriority,
    /// Input data
    pub input_data: WorkInputData,
    /// Work constraints
    pub constraints: Option<WorkConstraintsConfig>,
    /// Custom tags
    pub tags: Vec<String>,
    /// Work deadline
    pub deadline: Option<std::time::SystemTime>,
}

/// Work request types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkRequestType {
    /// One-time work
    OneTime {
        /// Initial delay in minutes
        initial_delay_minutes: usize,
        /// Enable expedited execution
        expedited: bool,
    },
    /// Periodic work
    Periodic {
        /// Repeat interval
        frequency: WorkFrequency,
        /// Keep existing work policy
        existing_work_policy: ExistingWorkPolicy,
    },
}

/// Existing work policies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExistingWorkPolicy {
    /// Replace existing work
    Replace,
    /// Keep existing work
    Keep,
    /// Append to existing work
    Append,
    /// Replace if running
    ReplaceIfRunning,
}

/// Work priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum WorkPriority {
    /// Minimum priority
    Min = 0,
    /// Low priority
    Low = 1,
    /// Default priority
    Default = 2,
    /// High priority
    High = 3,
    /// Maximum priority
    Max = 4,
}

/// Work input data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkInputData {
    /// Model ID (if applicable)
    pub model_id: Option<String>,
    /// Input tensor data
    pub tensor_data: Option<Vec<f32>>,
    /// Input shape
    pub tensor_shape: Option<Vec<usize>>,
    /// Configuration overrides
    pub config_overrides: Option<HashMap<String, serde_json::Value>>,
    /// Custom parameters
    pub custom_params: HashMap<String, serde_json::Value>,
}

/// Work execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkResult {
    /// Work ID
    pub work_id: String,
    /// Success flag
    pub success: bool,
    /// Result data
    pub result_data: Option<WorkResultData>,
    /// Error information
    pub error: Option<WorkError>,
    /// Execution metrics
    pub metrics: WorkExecutionMetrics,
    /// Retry information
    pub retry_info: WorkRetryInfo,
}

/// Work result data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkResultData {
    /// Output tensor data
    pub output_data: Option<Vec<f32>>,
    /// Output shape
    pub output_shape: Option<Vec<usize>>,
    /// Status message
    pub status_message: String,
    /// Additional result metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Work error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkError {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// Error category
    pub category: WorkErrorCategory,
    /// Is recoverable
    pub recoverable: bool,
    /// Suggested retry delay
    pub suggested_retry_delay_ms: Option<f64>,
}

/// Work error categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkErrorCategory {
    /// Network error
    Network,
    /// Storage error
    Storage,
    /// Memory error
    Memory,
    /// Model error
    Model,
    /// Configuration error
    Configuration,
    /// Timeout error
    Timeout,
    /// Permission error
    Permission,
    /// System error
    System,
    /// Unknown error
    Unknown,
}

/// Work execution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkExecutionMetrics {
    /// Start time
    pub start_time: std::time::SystemTime,
    /// End time
    pub end_time: std::time::SystemTime,
    /// Execution duration in milliseconds
    pub duration_ms: f64,
    /// Memory used in MB
    pub memory_used_mb: usize,
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    /// Network bytes transferred
    pub network_bytes: usize,
    /// Storage bytes used
    pub storage_bytes: usize,
}

/// Work retry information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkRetryInfo {
    /// Current retry attempt
    pub attempt_count: usize,
    /// Maximum retry attempts
    pub max_attempts: usize,
    /// Next retry time
    pub next_retry_time: Option<std::time::SystemTime>,
    /// Backoff delay in milliseconds
    pub backoff_delay_ms: f64,
}

/// Android Work Manager
pub struct AndroidWorkManager {
    config: AndroidWorkManagerConfig,
    inference_engine: Arc<Mutex<MobileInferenceEngine>>,
    // `tokio::sync::Mutex`, not `std::sync::Mutex`: `execute_model_download_work` /
    // `execute_model_update_work` hold this guard across an `.await` (the download/update
    // call itself is async), which makes the enclosing future non-`Send` under a
    // `std::sync::MutexGuard` and fails to compile under `tokio::spawn`'s `Send` bound.
    // `tokio::sync::MutexGuard` is `Send`, so it does not have that restriction.
    model_manager: Arc<tokio::sync::Mutex<ModelManager>>,
    work_queue: Arc<Mutex<WorkQueue>>,
    work_executor: Arc<Mutex<WorkExecutor>>,
    work_statistics: Arc<Mutex<WorkStatistics>>,
}

/// Work queue management
#[derive(Debug)]
struct WorkQueue {
    pending_work: HashMap<String, WorkRequest>,
    running_work: HashMap<String, WorkExecution>,
    completed_work: HashMap<String, WorkResult>,
    work_priorities: std::collections::BinaryHeap<PriorityWorkItem>,
}

/// Priority work item for queue ordering
#[derive(Debug, Clone, PartialEq, Eq)]
struct PriorityWorkItem {
    work_id: String,
    priority: WorkPriority,
    deadline: Option<std::time::SystemTime>,
    created_at: std::time::SystemTime,
}

impl Ord for PriorityWorkItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority first, then by deadline, then by creation time
        self.priority
            .cmp(&other.priority)
            .then_with(|| match (&self.deadline, &other.deadline) {
                (Some(a), Some(b)) => a.cmp(b),
                (Some(_), None) => std::cmp::Ordering::Greater,
                (None, Some(_)) => std::cmp::Ordering::Less,
                (None, None) => std::cmp::Ordering::Equal,
            })
            .then_with(|| self.created_at.cmp(&other.created_at))
    }
}

impl PartialOrd for PriorityWorkItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Work execution tracking
#[derive(Debug, Clone)]
struct WorkExecution {
    work_request: WorkRequest,
    start_time: std::time::SystemTime,
    executor_thread: Option<String>,
    progress: f64,
    can_cancel: bool,
}

/// Work executor
#[derive(Debug)]
struct WorkExecutor {
    max_concurrent_workers: usize,
    active_workers: usize,
    worker_threads: HashMap<String, WorkerThread>,
    execution_context: ExecutionContext,
}

/// Worker thread information
#[derive(Debug, Clone)]
struct WorkerThread {
    thread_id: String,
    current_work_id: Option<String>,
    start_time: std::time::SystemTime,
    cpu_affinity: Option<Vec<usize>>,
}

/// Execution context
#[derive(Debug, Clone)]
struct ExecutionContext {
    device_info: MobileDeviceInfo,
    available_memory_mb: usize,
    battery_level: f64,
    is_charging: bool,
    network_type: WorkNetworkType,
    thermal_state: f64,
}

/// Work statistics
#[derive(Debug, Clone)]
struct WorkStatistics {
    total_work_requests: usize,
    completed_work_requests: usize,
    failed_work_requests: usize,
    retried_work_requests: usize,
    average_execution_time_ms: f64,
    success_rate_by_type: HashMap<WorkTaskType, f64>,
    resource_usage_stats: ResourceUsageStats,
}

/// Resource usage statistics
#[derive(Debug, Clone)]
struct ResourceUsageStats {
    average_memory_usage_mb: f64,
    peak_memory_usage_mb: usize,
    average_cpu_usage_percent: f64,
    total_network_bytes: usize,
    total_storage_bytes: usize,
}

impl AndroidWorkManager {
    /// Create new Android Work Manager
    pub fn new(config: AndroidWorkManagerConfig, mobile_config: MobileConfig) -> Result<Self> {
        config.validate()?;

        let inference_engine = Arc::new(Mutex::new(MobileInferenceEngine::new(mobile_config)?));
        let model_manager = Arc::new(tokio::sync::Mutex::new(ModelManager::new_default()?));
        let work_queue = Arc::new(Mutex::new(WorkQueue::new()));
        let work_executor = Arc::new(Mutex::new(WorkExecutor::new(&config)));
        let work_statistics = Arc::new(Mutex::new(WorkStatistics::new()));

        Ok(Self {
            config,
            inference_engine,
            model_manager,
            work_queue,
            work_executor,
            work_statistics,
        })
    }

    /// Enqueue work request
    pub async fn enqueue_work(&self, work_request: WorkRequest) -> Result<String> {
        tracing::info!(
            "Enqueuing work: {} (type: {:?})",
            work_request.work_id,
            work_request.task_type
        );

        // Validate work request
        self.validate_work_request(&work_request)?;

        // Check constraints
        if !self.check_work_constraints(&work_request).await? {
            return Err(TrustformersError::runtime_error(
                "Work constraints not met".to_string(),
            ));
        }

        // Add to queue
        {
            let mut queue = self.work_queue.lock().unwrap_or_else(|p| p.into_inner());
            queue.enqueue_work(work_request.clone());
        }

        // Update statistics
        {
            let mut stats = self.work_statistics.lock().unwrap_or_else(|p| p.into_inner());
            stats.total_work_requests += 1;
        }

        // Try to schedule immediate execution if possible
        self.try_schedule_work()?;

        Ok(work_request.work_id)
    }

    /// Cancel work by ID
    pub async fn cancel_work(&self, work_id: &str) -> Result<bool> {
        tracing::info!("Cancelling work: {}", work_id);

        let mut queue = self.work_queue.lock().unwrap_or_else(|p| p.into_inner());

        // Remove from pending queue
        if queue.pending_work.remove(work_id).is_some() {
            return Ok(true);
        }

        // Try to cancel running work
        if let Some(execution) = queue.running_work.get(work_id) {
            if execution.can_cancel {
                // Signal cancellation to executor
                let mut executor = self.work_executor.lock().unwrap_or_else(|p| p.into_inner());
                executor.cancel_work(work_id);
                queue.running_work.remove(work_id);
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Get work status
    pub fn get_work_status(&self, work_id: &str) -> Result<WorkStatus> {
        let queue = self.work_queue.lock().unwrap_or_else(|p| p.into_inner());

        if queue.pending_work.contains_key(work_id) {
            Ok(WorkStatus::Pending)
        } else if queue.running_work.contains_key(work_id) {
            Ok(WorkStatus::Running)
        } else if let Some(result) = queue.completed_work.get(work_id) {
            if result.success {
                Ok(WorkStatus::Succeeded)
            } else {
                Ok(WorkStatus::Failed)
            }
        } else {
            Ok(WorkStatus::Unknown)
        }
    }

    /// Get work result
    pub fn get_work_result(&self, work_id: &str) -> Result<Option<WorkResult>> {
        let queue = self.work_queue.lock().unwrap_or_else(|p| p.into_inner());
        Ok(queue.completed_work.get(work_id).cloned())
    }

    /// Get work statistics
    pub fn get_work_statistics(&self) -> Result<String> {
        let stats = self.work_statistics.lock().unwrap_or_else(|p| p.into_inner());

        let stats_json = serde_json::json!({
            "total_work_requests": stats.total_work_requests,
            "completed_work_requests": stats.completed_work_requests,
            "failed_work_requests": stats.failed_work_requests,
            "retried_work_requests": stats.retried_work_requests,
            "success_rate": if stats.total_work_requests > 0 {
                stats.completed_work_requests as f64 / stats.total_work_requests as f64
            } else { 0.0 },
            "average_execution_time_ms": stats.average_execution_time_ms,
            "success_rate_by_type": stats.success_rate_by_type,
            "resource_usage": {
                "average_memory_usage_mb": stats.resource_usage_stats.average_memory_usage_mb,
                "peak_memory_usage_mb": stats.resource_usage_stats.peak_memory_usage_mb,
                "average_cpu_usage_percent": stats.resource_usage_stats.average_cpu_usage_percent,
                "total_network_bytes": stats.resource_usage_stats.total_network_bytes,
                "total_storage_bytes": stats.resource_usage_stats.total_storage_bytes
            }
        });

        Ok(stats_json.to_string())
    }

    /// List all pending work
    pub fn list_pending_work(&self) -> Result<Vec<String>> {
        let queue = self.work_queue.lock().unwrap_or_else(|p| p.into_inner());
        Ok(queue.pending_work.keys().cloned().collect())
    }

    /// List all running work
    pub fn list_running_work(&self) -> Result<Vec<String>> {
        let queue = self.work_queue.lock().unwrap_or_else(|p| p.into_inner());
        Ok(queue.running_work.keys().cloned().collect())
    }

    /// Clean up completed work older than specified duration
    pub fn cleanup_completed_work(&self, older_than_hours: f64) -> Result<usize> {
        let mut queue = self.work_queue.lock().unwrap_or_else(|p| p.into_inner());
        let cutoff_time = std::time::SystemTime::now()
            - std::time::Duration::from_secs_f64(older_than_hours * 3600.0);

        let initial_count = queue.completed_work.len();
        queue.completed_work.retain(|_, result| result.metrics.end_time > cutoff_time);

        let cleaned_count = initial_count - queue.completed_work.len();
        Ok(cleaned_count)
    }

    // Private helper methods

    // Deliberately NOT `async fn`: its body has no `.await` of its own -- it only
    // *constructs* a future and hands it to `tokio::spawn` without polling it inline.
    // Declaring it `async fn` anyway made its opaque return-position-`impl Future` type
    // mutually recursive with `complete_work`'s (this fn spawns a task that awaits
    // `complete_work`, which itself `.await`s this fn), which rustc's Send auto-trait
    // solver cannot resolve through a `tokio::spawn` boundary -- "future cannot be sent
    // between threads safely" even though no `MutexGuard` is ever actually held across a
    // real suspension point. Keeping this synchronous removes the opaque `Future` type
    // (and the cycle) entirely rather than papering over it with a `Box::pin` erasure.
    fn try_schedule_work(&self) -> Result<()> {
        let executor = self.work_executor.lock().unwrap_or_else(|p| p.into_inner());
        let mut queue = self.work_queue.lock().unwrap_or_else(|p| p.into_inner());

        // Check if we can schedule more work
        if executor.can_accept_more_work() {
            if let Some(work_item) = queue.get_next_work() {
                if let Some(work_request) = queue.pending_work.remove(&work_item.work_id) {
                    let execution = WorkExecution {
                        work_request: work_request.clone(),
                        start_time: std::time::SystemTime::now(),
                        executor_thread: None,
                        progress: 0.0,
                        can_cancel: true,
                    };

                    queue.running_work.insert(work_request.work_id.clone(), execution);

                    // Execute work asynchronously
                    let work_manager = self.clone_for_execution();
                    let work_id = work_request.work_id.clone();

                    tokio::spawn(async move {
                        let result = work_manager.execute_work(work_request).await;
                        work_manager.complete_work(&work_id, result).await;
                    });
                }
            }
        }

        Ok(())
    }

    async fn execute_work(&self, work_request: WorkRequest) -> WorkResult {
        let start_time = std::time::SystemTime::now();

        let result = match work_request.task_type {
            WorkTaskType::Inference => self.execute_inference_work(&work_request).await,
            WorkTaskType::ModelDownload => self.execute_model_download_work(&work_request).await,
            WorkTaskType::ModelUpdate => self.execute_model_update_work(&work_request).await,
            WorkTaskType::FederatedLearning => {
                self.execute_federated_learning_work(&work_request).await
            },
            WorkTaskType::DataPreprocessing => {
                self.execute_data_preprocessing_work(&work_request).await
            },
            WorkTaskType::CacheCleanup => self.execute_cache_cleanup_work(&work_request).await,
            WorkTaskType::PerformanceProfiling => {
                self.execute_performance_profiling_work(&work_request).await
            },
            WorkTaskType::HealthCheck => self.execute_health_check_work(&work_request).await,
            WorkTaskType::Custom(_) => self.execute_custom_work(&work_request).await,
        };

        let end_time = std::time::SystemTime::now();
        let duration = end_time.duration_since(start_time).unwrap_or_default();

        let metrics = WorkExecutionMetrics {
            start_time,
            end_time,
            duration_ms: duration.as_millis() as f64,
            memory_used_mb: self.get_current_memory_usage(),
            cpu_usage_percent: self.get_current_cpu_usage(),
            network_bytes: 0, // Would be tracked during execution
            storage_bytes: 0, // Would be tracked during execution
        };

        match result {
            Ok(result_data) => WorkResult {
                work_id: work_request.work_id,
                success: true,
                result_data: Some(result_data),
                error: None,
                metrics,
                retry_info: WorkRetryInfo {
                    attempt_count: 1,
                    max_attempts: self.config.retry_policy.max_retry_attempts,
                    next_retry_time: None,
                    backoff_delay_ms: 0.0,
                },
            },
            Err(error) => WorkResult {
                work_id: work_request.work_id,
                success: false,
                result_data: None,
                error: Some(WorkError {
                    code: "EXECUTION_ERROR".to_string(),
                    message: error.to_string(),
                    category: WorkErrorCategory::System,
                    recoverable: true,
                    suggested_retry_delay_ms: Some(self.config.retry_policy.initial_retry_delay_ms),
                }),
                metrics,
                retry_info: WorkRetryInfo {
                    attempt_count: 1,
                    max_attempts: self.config.retry_policy.max_retry_attempts,
                    next_retry_time: Some(
                        std::time::SystemTime::now()
                            + std::time::Duration::from_millis(
                                self.config.retry_policy.initial_retry_delay_ms as u64,
                            ),
                    ),
                    backoff_delay_ms: self.config.retry_policy.initial_retry_delay_ms,
                },
            },
        }
    }

    async fn complete_work(&self, work_id: &str, result: WorkResult) {
        // Move work from running to completed
        {
            let mut queue = self.work_queue.lock().unwrap_or_else(|p| p.into_inner());
            queue.running_work.remove(work_id);
            queue.completed_work.insert(work_id.to_string(), result.clone());
        }

        // Update statistics
        {
            let mut stats = self.work_statistics.lock().unwrap_or_else(|p| p.into_inner());
            if result.success {
                stats.completed_work_requests += 1;
            } else {
                stats.failed_work_requests += 1;
            }

            // Update running average execution time
            let alpha = 0.1;
            if stats.completed_work_requests + stats.failed_work_requests == 1 {
                stats.average_execution_time_ms = result.metrics.duration_ms;
            } else {
                stats.average_execution_time_ms = alpha * result.metrics.duration_ms
                    + (1.0 - alpha) * stats.average_execution_time_ms;
            }
        }

        // Try to schedule next work
        let _ = self.try_schedule_work();
    }

    fn clone_for_execution(&self) -> Self {
        Self {
            config: self.config.clone(),
            inference_engine: self.inference_engine.clone(),
            model_manager: self.model_manager.clone(),
            work_queue: self.work_queue.clone(),
            work_executor: self.work_executor.clone(),
            work_statistics: self.work_statistics.clone(),
        }
    }

    // Work execution implementations for different task types
    async fn execute_inference_work(&self, work_request: &WorkRequest) -> Result<WorkResultData> {
        if let Some(ref model_id) = work_request.input_data.model_id {
            if let (Some(ref tensor_data), Some(ref tensor_shape)) = (
                &work_request.input_data.tensor_data,
                &work_request.input_data.tensor_shape,
            ) {
                let input_tensor = Tensor::from_vec(tensor_data.clone(), tensor_shape)?;

                let result = {
                    let mut engine =
                        self.inference_engine.lock().unwrap_or_else(|p| p.into_inner());
                    // `MobileInferenceEngine` holds one loaded model at a
                    // time and has no per-`model_id` dispatch (see its own
                    // doc comment in `inference.rs`) -- it runs whichever
                    // checkpoint was most recently loaded into this
                    // `AndroidWorkManager`'s shared engine, regardless of
                    // which `model_id` this work item names. This is an
                    // existing single-model-engine limitation of
                    // `AndroidWorkManager`, not something this fix
                    // introduces; the previous call here
                    // (`engine.inference(model_id, &input_tensor)`) did not
                    // even match `MobileInferenceEngine::inference`'s real
                    // one-argument signature and could not have compiled.
                    if !engine.has_loaded_model() {
                        return Err(TrustformersError::runtime_error(format!(
                            "no model is loaded in the shared inference engine (requested \
                             model_id '{model_id}')"
                        )));
                    }
                    engine.inference(&input_tensor)?
                };

                let output_data = result.data_f32()?.to_vec();
                let output_shape = result.shape().to_vec();

                Ok(WorkResultData {
                    output_data: Some(output_data),
                    output_shape: Some(output_shape),
                    status_message: "Inference completed successfully".to_string(),
                    metadata: HashMap::new(),
                })
            } else {
                Err(TrustformersError::runtime_error(
                    "Missing tensor data for inference".into(),
                ))
            }
        } else {
            Err(TrustformersError::runtime_error(
                "Missing model ID for inference".into(),
            ))
        }
    }

    async fn execute_model_download_work(
        &self,
        work_request: &WorkRequest,
    ) -> Result<WorkResultData> {
        if let Some(ref model_id) = work_request.input_data.model_id {
            let mut model_manager = self.model_manager.lock().await;
            // Simulate model download
            model_manager.download_model(model_id, None).await?;

            Ok(WorkResultData {
                output_data: None,
                output_shape: None,
                status_message: format!("Model {} downloaded successfully", model_id),
                metadata: HashMap::new(),
            })
        } else {
            Err(TrustformersError::runtime_error(
                "Missing model ID for download".into(),
            ))
        }
    }

    async fn execute_model_update_work(
        &self,
        work_request: &WorkRequest,
    ) -> Result<WorkResultData> {
        if let Some(ref model_id) = work_request.input_data.model_id {
            let mut model_manager = self.model_manager.lock().await;
            // Simulate model update
            model_manager.update_model(model_id).await?;

            Ok(WorkResultData {
                output_data: None,
                output_shape: None,
                status_message: format!("Model {} updated successfully", model_id),
                metadata: HashMap::new(),
            })
        } else {
            Err(TrustformersError::runtime_error(
                "Missing model ID for update".into(),
            ))
        }
    }

    async fn execute_federated_learning_work(
        &self,
        _work_request: &WorkRequest,
    ) -> Result<WorkResultData> {
        // Federated learning implementation would go here
        Ok(WorkResultData {
            output_data: None,
            output_shape: None,
            status_message: "Federated learning round completed".to_string(),
            metadata: HashMap::new(),
        })
    }

    async fn execute_data_preprocessing_work(
        &self,
        _work_request: &WorkRequest,
    ) -> Result<WorkResultData> {
        // Data preprocessing implementation would go here
        Ok(WorkResultData {
            output_data: None,
            output_shape: None,
            status_message: "Data preprocessing completed".to_string(),
            metadata: HashMap::new(),
        })
    }

    async fn execute_cache_cleanup_work(
        &self,
        _work_request: &WorkRequest,
    ) -> Result<WorkResultData> {
        // Cache cleanup implementation would go here
        Ok(WorkResultData {
            output_data: None,
            output_shape: None,
            status_message: "Cache cleanup completed".to_string(),
            metadata: HashMap::new(),
        })
    }

    async fn execute_performance_profiling_work(
        &self,
        _work_request: &WorkRequest,
    ) -> Result<WorkResultData> {
        // Performance profiling implementation would go here
        Ok(WorkResultData {
            output_data: None,
            output_shape: None,
            status_message: "Performance profiling completed".to_string(),
            metadata: HashMap::new(),
        })
    }

    async fn execute_health_check_work(
        &self,
        _work_request: &WorkRequest,
    ) -> Result<WorkResultData> {
        // Health check implementation would go here
        Ok(WorkResultData {
            output_data: None,
            output_shape: None,
            status_message: "Health check completed - all systems operational".to_string(),
            metadata: HashMap::new(),
        })
    }

    async fn execute_custom_work(&self, _work_request: &WorkRequest) -> Result<WorkResultData> {
        // Custom work implementation would go here
        Ok(WorkResultData {
            output_data: None,
            output_shape: None,
            status_message: "Custom work completed".to_string(),
            metadata: HashMap::new(),
        })
    }

    fn validate_work_request(&self, work_request: &WorkRequest) -> Result<()> {
        if work_request.work_id.is_empty() {
            return Err(TrustformersError::config_error(
                "Work ID cannot be empty",
                "validate_work_request",
            ));
        }

        // Validate input data based on task type
        match work_request.task_type {
            WorkTaskType::Inference => {
                if work_request.input_data.model_id.is_none() {
                    return Err(TrustformersError::config_error(
                        "Model ID required for inference task",
                        "validate_work_request",
                    ));
                }
                if work_request.input_data.tensor_data.is_none() {
                    return Err(TrustformersError::config_error(
                        "Tensor data required for inference task",
                        "validate_work_request",
                    ));
                }
            },
            WorkTaskType::ModelDownload | WorkTaskType::ModelUpdate
                if work_request.input_data.model_id.is_none() =>
            {
                return Err(TrustformersError::config_error(
                    "Model ID required for model task",
                    "validate_work_request",
                ));
            },
            _ => {
                // Other task types may have different validation requirements
            },
        }

        Ok(())
    }

    async fn check_work_constraints(&self, work_request: &WorkRequest) -> Result<bool> {
        let constraints = work_request.constraints.as_ref().unwrap_or(&self.config.constraints);

        // Check network constraints
        if constraints.require_unmetered_network && !self.is_unmetered_network_available() {
            return Ok(false);
        }

        // Check battery constraints
        if constraints.require_charging && !self.is_device_charging() {
            return Ok(false);
        }

        if constraints.require_battery_not_low && self.is_battery_low() {
            return Ok(false);
        }

        // Check device idle constraint
        if constraints.require_device_idle && !self.is_device_idle() {
            return Ok(false);
        }

        // Check storage constraints
        if !self.check_storage_constraints(&constraints.storage_constraints) {
            return Ok(false);
        }

        Ok(true)
    }

    /// Real resident-memory usage of this process, in MB, via `sysinfo`
    /// (the same crate/pattern `crash_reporter::collect_memory_usage`
    /// already uses). Previously a hardcoded `64`, regardless of actual
    /// memory pressure.
    fn get_current_memory_usage(&self) -> usize {
        use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

        let pid = Pid::from_u32(std::process::id());
        let mut system = System::new();
        system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            true,
            ProcessRefreshKind::nothing().with_memory(),
        );
        system.process(pid).map(|p| (p.memory() / (1024 * 1024)) as usize).unwrap_or(0)
    }

    /// Real global CPU usage percentage via `sysinfo`. `sysinfo` requires
    /// two samples separated by `MINIMUM_CPU_UPDATE_INTERVAL` for an
    /// accurate reading (the same pattern `crash_reporter::collect_cpu_info`
    /// uses); this method is called from constraint checks ahead of
    /// scheduling a work item, not in a per-frame hot path, so the short
    /// blocking sleep is acceptable. Previously a hardcoded `25.0`,
    /// regardless of actual load.
    fn get_current_cpu_usage(&self) -> f64 {
        use sysinfo::System;

        let mut system = System::new();
        system.refresh_cpu_usage();
        std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
        system.refresh_cpu_usage();
        system.global_cpu_usage() as f64
    }

    // Real per-platform detection of network metering, charging state,
    // battery level and device-idle state requires querying Android's
    // `ConnectivityManager`/`BatteryManager`/`PowerManager` system services
    // through JNI, which in turn requires a `Context` (or at minimum a
    // `JavaVM`, as `crate::android::engine::AndroidInferenceEngine::init_jvm`
    // is given). `AndroidWorkManager` is never handed one -- its
    // constructor takes only `AndroidWorkManagerConfig` and `MobileConfig` --
    // so there is nothing to honestly query from any of the four methods
    // below. This is the same class of gap already documented on
    // `AndroidInferenceEngine::nnapi_inference`/`gpu_inference`: real,
    // substantial work (threading a `Context`/`JavaVM` through this
    // manager's construction and every caller) that this pass does not
    // fabricate a shortcut for.
    //
    // What changes here is the *direction* of the fallback. The previous
    // constants were each wrong in the specific way that made constraints
    // easiest to (incorrectly) satisfy: unmetered network "available"
    // (`true`), battery "not low" (`is_battery_low` => `false`), and device
    // "idle" (`true`) were all assumed by default, so a job declared
    // "requires charging + unmetered network + battery not low" could run
    // on a low battery over metered cellular the moment `require_charging`
    // (the one constraint that *was* conservative) happened not to be set.
    // Every fallback below now fails *closed* instead: a condition is
    // reported satisfied only when it has actually been verified, so an
    // unverifiable precondition blocks the work that declared it required
    // rather than silently waving it through.

    /// Conservative fallback: never assume an unmetered connection that has
    /// not been verified. Previously `true` unconditionally.
    fn is_unmetered_network_available(&self) -> bool {
        false
    }

    /// Conservative fallback: never assume charging that has not been
    /// verified. Unchanged in *value* from the previous constant (already
    /// `false`) but now documented as a deliberate fail-closed default
    /// rather than an unexplained placeholder.
    fn is_device_charging(&self) -> bool {
        false
    }

    /// Conservative fallback: assume the battery *may* be low until
    /// verified otherwise. Previously `false` unconditionally -- the
    /// direction that let a `require_battery_not_low` job run on a real
    /// low battery.
    fn is_battery_low(&self) -> bool {
        true
    }

    /// Conservative fallback: never assume the device is idle without
    /// verification. Previously `true` unconditionally -- the direction
    /// that let a `require_device_idle` job run while the device was
    /// actively in use.
    fn is_device_idle(&self) -> bool {
        false
    }

    /// Real, cross-platform available-storage check via `sysinfo::Disks`
    /// (no JNI/Context needed -- disk space is exposed the same way on
    /// every desktop/mobile OS `sysinfo` supports). Compares against the
    /// disk backing the current working directory when that can be
    /// determined (the longest matching mount-point prefix, the standard
    /// "which filesystem owns this path" rule), falling back to the
    /// largest available-space figure across all detected disks otherwise.
    /// Previously a hardcoded `true`, regardless of real free space --
    /// `enable_storage_cleanup`/`min_free_storage_mb` were configurable but
    /// never actually enforced.
    fn check_storage_constraints(&self, constraints: &StorageConstraints) -> bool {
        use sysinfo::Disks;

        let disks = Disks::new_with_refreshed_list();
        let cwd = std::env::current_dir().ok();

        let available_bytes = cwd
            .as_deref()
            .and_then(|cwd| {
                disks
                    .list()
                    .iter()
                    .filter(|d| cwd.starts_with(d.mount_point()))
                    .max_by_key(|d| d.mount_point().as_os_str().len())
                    .map(|d| d.available_space())
            })
            .or_else(|| disks.list().iter().map(|d| d.available_space()).max())
            .unwrap_or(0);

        let available_mb = available_bytes / (1024 * 1024);
        available_mb >= constraints.min_free_storage_mb as u64
    }
}

/// Work status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkStatus {
    /// Work is pending execution
    Pending,
    /// Work is currently running
    Running,
    /// Work completed successfully
    Succeeded,
    /// Work failed
    Failed,
    /// Work was cancelled
    Cancelled,
    /// Work status is unknown
    Unknown,
}

// Implementation details for helper structs

impl WorkQueue {
    fn new() -> Self {
        Self {
            pending_work: HashMap::new(),
            running_work: HashMap::new(),
            completed_work: HashMap::new(),
            work_priorities: std::collections::BinaryHeap::new(),
        }
    }

    fn enqueue_work(&mut self, work_request: WorkRequest) {
        let priority_item = PriorityWorkItem {
            work_id: work_request.work_id.clone(),
            priority: work_request.priority,
            deadline: work_request.deadline,
            created_at: std::time::SystemTime::now(),
        };

        self.pending_work.insert(work_request.work_id.clone(), work_request);
        self.work_priorities.push(priority_item);
    }

    fn get_next_work(&mut self) -> Option<PriorityWorkItem> {
        self.work_priorities.pop()
    }
}

impl WorkExecutor {
    fn new(config: &AndroidWorkManagerConfig) -> Self {
        Self {
            max_concurrent_workers: 4, // Default
            active_workers: 0,
            worker_threads: HashMap::new(),
            execution_context: ExecutionContext {
                // Real detection, falling back to `MobileDeviceInfo::default()`
                // (not a panic) if it fails -- this constructor is
                // infallible (`WorkExecutor::new` returns `Self`, not
                // `Result<Self>`), so a detection error cannot propagate.
                device_info: MobileDeviceDetector::detect().unwrap_or_default(),
                available_memory_mb: 512,
                battery_level: 100.0,
                is_charging: false,
                network_type: WorkNetworkType::Connected,
                thermal_state: 0.0,
            },
        }
    }

    fn can_accept_more_work(&self) -> bool {
        self.active_workers < self.max_concurrent_workers
    }

    fn cancel_work(&mut self, _work_id: &str) {
        // Implementation for cancelling work
    }
}

impl WorkStatistics {
    fn new() -> Self {
        Self {
            total_work_requests: 0,
            completed_work_requests: 0,
            failed_work_requests: 0,
            retried_work_requests: 0,
            average_execution_time_ms: 0.0,
            success_rate_by_type: HashMap::new(),
            resource_usage_stats: ResourceUsageStats {
                average_memory_usage_mb: 0.0,
                peak_memory_usage_mb: 0,
                average_cpu_usage_percent: 0.0,
                total_network_bytes: 0,
                total_storage_bytes: 0,
            },
        }
    }
}

impl Default for AndroidWorkManagerConfig {
    fn default() -> Self {
        Self {
            enable_periodic_work: true,
            enable_one_time_work: true,
            enable_expedited_work: true,
            constraints: WorkConstraintsConfig {
                require_unmetered_network: false,
                require_charging: false,
                require_device_idle: false,
                require_battery_not_low: true,
                required_network_type: WorkNetworkType::Connected,
                storage_constraints: StorageConstraints {
                    min_free_storage_mb: 100,
                    max_cache_storage_mb: 500,
                    enable_storage_cleanup: true,
                },
            },
            retry_policy: WorkRetryPolicyConfig {
                retry_policy: WorkRetryPolicy::Exponential,
                max_retry_attempts: 3,
                initial_retry_delay_ms: 1000.0,
                max_retry_delay_ms: 300000.0, // 5 minutes
                backoff_multiplier: 2.0,
            },
            background_execution: BackgroundExecutionConfig {
                max_execution_time_seconds: 600.0, // 10 minutes
                enable_foreground_service: true,
                foreground_notification: ForegroundNotificationConfig {
                    channel_id: "trustformers_work".to_string(),
                    title: "TrustformeRS Background Processing".to_string(),
                    content_text: "Processing machine learning tasks".to_string(),
                    show_progress: true,
                    enable_cancel_action: true,
                },
                task_prioritization: TaskPrioritizationConfig {
                    high_priority_tasks: vec![
                        WorkTaskType::Inference,
                        WorkTaskType::FederatedLearning,
                    ],
                    enable_adaptive_prioritization: true,
                    device_state_priority_adjustment: true,
                    execution_order: TaskExecutionOrder::Priority,
                },
            },
            data_sync: DataSyncConfig {
                enable_model_updates: true,
                enable_federated_sync: true,
                sync_frequency: WorkFrequency {
                    interval_minutes: 60,
                    flex_interval_minutes: 15,
                    initial_delay_minutes: 5,
                },
                conflict_resolution: ConflictResolutionStrategy::LastModifiedWins,
                compression_settings: DataCompressionConfig {
                    enable_compression: true,
                    algorithm: CompressionAlgorithm::Gzip,
                    compression_level: 6,
                    min_size_threshold_bytes: 1024,
                },
            },
        }
    }
}

impl AndroidWorkManagerConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.retry_policy.max_retry_attempts > 10 {
            return Err(TrustformersError::config_error(
                "Too many retry attempts",
                "validate",
            ));
        }

        if self.background_execution.max_execution_time_seconds > 3600.0 {
            return Err(TrustformersError::config_error(
                "Execution time too long",
                "validate",
            ));
        }

        if self.constraints.storage_constraints.min_free_storage_mb < 50 {
            return Err(TrustformersError::config_error(
                "Minimum storage too low",
                "validate",
            ));
        }

        Ok(())
    }
}

// Mock implementations for ModelManager methods
impl ModelManager {
    fn new_default() -> Result<Self> {
        // Default model manager creation
        Ok(Self::new(
            crate::model_management::ModelManagerConfig::default(),
        )?)
    }

    async fn update_model(&mut self, _model_id: &str) -> Result<()> {
        // Model update implementation
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_work_manager_config_default() {
        let config = AndroidWorkManagerConfig::default();
        assert!(config.enable_periodic_work);
        assert!(config.enable_one_time_work);
        assert_eq!(
            config.retry_policy.retry_policy,
            WorkRetryPolicy::Exponential
        );
    }

    #[test]
    fn test_work_manager_config_validation() {
        let mut config = AndroidWorkManagerConfig::default();
        assert!(config.validate().is_ok());

        config.retry_policy.max_retry_attempts = 15;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_work_priority_ordering() {
        assert!(WorkPriority::Max > WorkPriority::High);
        assert!(WorkPriority::High > WorkPriority::Default);
        assert!(WorkPriority::Default > WorkPriority::Low);
        assert!(WorkPriority::Low > WorkPriority::Min);
    }

    #[tokio::test]
    async fn test_work_manager_creation() {
        let work_config = AndroidWorkManagerConfig::default();
        let mobile_config = MobileConfig::android_optimized();

        let result = AndroidWorkManager::new(work_config, mobile_config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_work_request_validation() {
        let work_config = AndroidWorkManagerConfig::default();
        let mobile_config = MobileConfig::android_optimized();
        let manager =
            AndroidWorkManager::new(work_config, mobile_config).expect("Operation failed");

        let valid_request = WorkRequest {
            work_id: "test_inference".to_string(),
            work_type: WorkRequestType::OneTime {
                initial_delay_minutes: 0,
                expedited: false,
            },
            task_type: WorkTaskType::Inference,
            priority: WorkPriority::Default,
            input_data: WorkInputData {
                model_id: Some("test_model".to_string()),
                tensor_data: Some(vec![1.0, 2.0, 3.0]),
                tensor_shape: Some(vec![1, 3]),
                config_overrides: None,
                custom_params: HashMap::new(),
            },
            constraints: None,
            tags: vec!["test".to_string()],
            deadline: None,
        };

        assert!(manager.validate_work_request(&valid_request).is_ok());

        let invalid_request = WorkRequest {
            work_id: "".to_string(), // Invalid: empty work ID
            ..valid_request
        };

        assert!(manager.validate_work_request(&invalid_request).is_err());
    }
}
