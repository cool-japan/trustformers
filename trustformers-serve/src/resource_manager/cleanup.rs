//! Resource cleanup and lifecycle management.

use super::types::{CleanupEventType, CleanupResult};
use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::Duration,
};
use tracing::{debug, info, warn};

use crate::test_parallelization::ResourceCleanupConfig;

/// Cleanup manager for coordinating resource cleanup
pub struct CleanupManager {
    /// Configuration
    config: Arc<Mutex<ResourceCleanupConfig>>,
    /// Cleanup queue
    cleanup_queue: Arc<Mutex<VecDeque<CleanupTask>>>,
    /// Active cleanup operations
    active_cleanups: Arc<Mutex<HashMap<String, CleanupOperation>>>,
    /// Cleanup history
    cleanup_history: Arc<Mutex<Vec<CleanupEvent>>>,
    /// Cleanup statistics
    cleanup_stats: Arc<Mutex<CleanupStatistics>>,
}

/// Cleanup task
#[derive(Debug, Clone)]
pub struct CleanupTask {
    /// Task ID
    pub task_id: String,
    /// Test ID associated with cleanup
    pub test_id: String,
    /// Resource type to cleanup
    pub resource_type: ResourceType,
    /// Resource identifier
    pub resource_id: String,
    /// Cleanup priority
    pub priority: CleanupPriority,
    /// Scheduled cleanup time
    pub scheduled_time: DateTime<Utc>,
    /// Cleanup strategy
    pub strategy: CleanupStrategy,
    /// Retry count
    pub retry_count: usize,
    /// Task metadata
    pub metadata: HashMap<String, String>,
}

/// Resource types for cleanup
#[derive(Debug, Clone)]
pub enum ResourceType {
    NetworkPort,
    TempDirectory,
    GpuDevice,
    DatabaseConnection,
    CustomResource(String),
    Mixed,
}

/// Cleanup priorities
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum CleanupPriority {
    Low,
    Medium,
    High,
    Critical,
    Emergency,
}

/// Cleanup strategies
#[derive(Debug, Clone)]
pub enum CleanupStrategy {
    /// Immediate cleanup
    Immediate,
    /// Graceful cleanup with timeout
    Graceful { timeout: Duration },
    /// Delayed cleanup
    Delayed { delay: Duration },
    /// Background cleanup
    Background,
    /// Force cleanup (ignore errors)
    Force,
}

/// Active cleanup operation
#[derive(Debug, Clone)]
pub struct CleanupOperation {
    /// Operation ID
    pub operation_id: String,
    /// Associated cleanup task
    pub task: CleanupTask,
    /// Operation start time
    pub started_at: DateTime<Utc>,
    /// Operation status
    pub status: CleanupOperationStatus,
    /// Progress percentage (0.0 - 1.0)
    pub progress: f32,
    /// Operation details
    pub details: HashMap<String, String>,
}

/// Cleanup operation status
#[derive(Debug, Clone)]
pub enum CleanupOperationStatus {
    /// Operation is starting
    Starting,
    /// Operation is in progress
    InProgress,
    /// Operation completed successfully
    Completed,
    /// Operation failed
    Failed(String),
    /// Operation was cancelled
    Cancelled,
    /// Operation timed out
    TimedOut,
}

/// Cleanup event record
#[derive(Debug, Clone)]
pub struct CleanupEvent {
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Event type
    pub event_type: CleanupEventType,
    /// Test ID
    pub test_id: String,
    /// Resource type
    pub resource_type: ResourceType,
    /// Resource ID
    pub resource_id: String,
    /// Cleanup duration
    pub duration: Duration,
    /// Cleanup result
    pub result: CleanupResult,
    /// Additional details
    pub details: HashMap<String, String>,
}

/// Cleanup statistics
#[derive(Debug, Default, Clone)]
pub struct CleanupStatistics {
    /// Total cleanup tasks processed
    pub total_tasks: u64,
    /// Successful cleanups
    pub successful_cleanups: u64,
    /// Failed cleanups
    pub failed_cleanups: u64,
    /// Average cleanup time
    pub average_cleanup_time: Duration,
    /// Cleanup efficiency percentage
    pub efficiency_percentage: f32,
    /// Cleanups by resource type
    pub by_resource_type: HashMap<String, u64>,
    /// Cleanups by priority
    pub by_priority: HashMap<String, u64>,
}

/// Cleanup scheduler for managing cleanup timing
#[derive(Debug)]
pub struct CleanupScheduler {
    /// Scheduled tasks
    scheduled_tasks: Arc<Mutex<VecDeque<CleanupTask>>>,
    /// Recurring cleanup jobs
    recurring_jobs: Arc<Mutex<Vec<RecurringCleanupJob>>>,
}

/// Recurring cleanup job
#[derive(Debug, Clone)]
pub struct RecurringCleanupJob {
    /// Job ID
    pub job_id: String,
    /// Job name
    pub name: String,
    /// Cleanup interval
    pub interval: Duration,
    /// Last execution time
    pub last_execution: DateTime<Utc>,
    /// Next scheduled execution
    pub next_execution: DateTime<Utc>,
    /// Job configuration
    pub config: RecurringJobConfig,
}

/// Recurring job configuration
#[derive(Debug, Clone)]
pub struct RecurringJobConfig {
    /// Resource types to cleanup
    pub resource_types: Vec<ResourceType>,
    /// Cleanup criteria
    pub criteria: CleanupCriteria,
    /// Maximum items per run
    pub max_items_per_run: usize,
    /// Job enabled
    pub enabled: bool,
}

/// Cleanup criteria
#[derive(Debug, Clone)]
pub struct CleanupCriteria {
    /// Age threshold
    pub age_threshold: Duration,
    /// Unused threshold
    pub unused_threshold: Duration,
    /// Status filters
    pub status_filters: Vec<String>,
    /// Custom filters
    pub custom_filters: HashMap<String, String>,
}

impl CleanupManager {
    /// Create new cleanup manager
    pub async fn new(config: ResourceCleanupConfig) -> Result<Self> {
        Ok(Self {
            config: Arc::new(Mutex::new(config)),
            cleanup_queue: Arc::new(Mutex::new(VecDeque::new())),
            active_cleanups: Arc::new(Mutex::new(HashMap::new())),
            cleanup_history: Arc::new(Mutex::new(Vec::new())),
            cleanup_stats: Arc::new(Mutex::new(CleanupStatistics::default())),
        })
    }

    /// Schedule a cleanup task
    pub async fn schedule_cleanup(&self, task: CleanupTask) -> Result<()> {
        info!(
            "Scheduling cleanup task: {} for test: {}",
            task.task_id, task.test_id
        );

        let mut cleanup_queue = self.cleanup_queue.lock();
        cleanup_queue.push_back(task);

        // Sort by priority and scheduled time
        let mut tasks: Vec<_> = cleanup_queue.drain(..).collect();
        tasks.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.scheduled_time.cmp(&b.scheduled_time))
        });

        cleanup_queue.extend(tasks);
        Ok(())
    }

    /// Process cleanup queue
    pub async fn process_cleanup_queue(&self) -> Result<usize> {
        let mut processed_count = 0;
        let now = Utc::now();

        // Get tasks ready for cleanup
        let mut cleanup_queue = self.cleanup_queue.lock();
        let mut ready_tasks = Vec::new();

        while let Some(task) = cleanup_queue.front() {
            if task.scheduled_time <= now {
                ready_tasks.push(cleanup_queue.pop_front().unwrap());
            } else {
                break; // Tasks are ordered by time
            }
        }
        drop(cleanup_queue);

        // Process ready tasks
        for task in ready_tasks {
            match self.execute_cleanup_task(&task).await {
                Ok(_) => {
                    processed_count += 1;
                    self.record_cleanup_success(&task).await?;
                },
                Err(e) => {
                    warn!("Cleanup task failed: {} - {}", task.task_id, e);
                    self.record_cleanup_failure(&task, e.to_string()).await?;

                    // Retry logic
                    if task.retry_count < 3 {
                        self.schedule_retry(&task).await?;
                    }
                },
            }
        }

        Ok(processed_count)
    }

    /// Execute a cleanup task
    async fn execute_cleanup_task(&self, task: &CleanupTask) -> Result<()> {
        let operation_id = format!("op-{}-{}", task.task_id, Utc::now().timestamp());

        debug!(
            "Executing cleanup task: {} (operation: {})",
            task.task_id, operation_id
        );

        // Create cleanup operation
        let operation = CleanupOperation {
            operation_id: operation_id.clone(),
            task: task.clone(),
            started_at: Utc::now(),
            status: CleanupOperationStatus::Starting,
            progress: 0.0,
            details: HashMap::new(),
        };

        // Track operation
        let mut active_cleanups = self.active_cleanups.lock();
        active_cleanups.insert(operation_id.clone(), operation);
        drop(active_cleanups);

        // Execute cleanup based on resource type and strategy
        match &task.resource_type {
            ResourceType::NetworkPort => {
                self.cleanup_network_port(&task.resource_id).await?;
            },
            ResourceType::TempDirectory => {
                self.cleanup_temp_directory(&task.resource_id).await?;
            },
            ResourceType::GpuDevice => {
                self.cleanup_gpu_device(&task.resource_id).await?;
            },
            ResourceType::DatabaseConnection => {
                self.cleanup_database_connection(&task.resource_id).await?;
            },
            ResourceType::CustomResource(resource_type) => {
                self.cleanup_custom_resource(resource_type, &task.resource_id).await?;
            },
            ResourceType::Mixed => {
                self.cleanup_mixed_resources(&task.test_id).await?;
            },
        }

        // Mark operation as completed
        let mut active_cleanups = self.active_cleanups.lock();
        if let Some(operation) = active_cleanups.get_mut(&operation_id) {
            operation.status = CleanupOperationStatus::Completed;
            operation.progress = 1.0;
        }

        info!("Successfully completed cleanup task: {}", task.task_id);
        Ok(())
    }

    /// Cleanup network port resource
    async fn cleanup_network_port(&self, resource_id: &str) -> Result<()> {
        debug!("Cleaning up network port: {}", resource_id);
        // In a real implementation, this would interact with the NetworkPortManager
        Ok(())
    }

    /// Cleanup temporary directory resource
    async fn cleanup_temp_directory(&self, resource_id: &str) -> Result<()> {
        debug!("Cleaning up temporary directory: {}", resource_id);
        // In a real implementation, this would interact with the TempDirectoryManager
        Ok(())
    }

    /// Cleanup GPU device resource
    async fn cleanup_gpu_device(&self, resource_id: &str) -> Result<()> {
        debug!("Cleaning up GPU device: {}", resource_id);
        // In a real implementation, this would interact with the GpuResourceManager
        Ok(())
    }

    /// Cleanup database connection resource
    async fn cleanup_database_connection(&self, resource_id: &str) -> Result<()> {
        debug!("Cleaning up database connection: {}", resource_id);
        // In a real implementation, this would interact with the DatabaseConnectionManager
        Ok(())
    }

    /// Cleanup custom resource
    async fn cleanup_custom_resource(&self, resource_type: &str, resource_id: &str) -> Result<()> {
        debug!(
            "Cleaning up custom resource: {} ({})",
            resource_type, resource_id
        );
        // In a real implementation, this would interact with the CustomResourceManager
        Ok(())
    }

    /// Cleanup mixed resources for a test
    async fn cleanup_mixed_resources(&self, test_id: &str) -> Result<()> {
        debug!("Cleaning up all resources for test: {}", test_id);
        // In a real implementation, this would coordinate cleanup across all managers
        Ok(())
    }

    /// Schedule a retry for a failed task
    async fn schedule_retry(&self, task: &CleanupTask) -> Result<()> {
        let mut retry_task = task.clone();
        retry_task.retry_count += 1;
        retry_task.scheduled_time =
            Utc::now() + chrono::Duration::minutes(5 * retry_task.retry_count as i64);

        self.schedule_cleanup(retry_task).await
    }

    /// Record successful cleanup
    async fn record_cleanup_success(&self, task: &CleanupTask) -> Result<()> {
        let event = CleanupEvent {
            timestamp: Utc::now(),
            event_type: CleanupEventType::Completed,
            test_id: task.test_id.clone(),
            resource_type: task.resource_type.clone(),
            resource_id: task.resource_id.clone(),
            duration: Duration::from_secs(1), // Placeholder
            result: CleanupResult::Success {
                files_removed: 0,
                bytes_freed: 0,
            },
            details: HashMap::new(),
        };

        let mut cleanup_history = self.cleanup_history.lock();
        cleanup_history.push(event);

        // Update statistics
        let mut stats = self.cleanup_stats.lock();
        stats.total_tasks += 1;
        stats.successful_cleanups += 1;

        Ok(())
    }

    /// Record failed cleanup
    async fn record_cleanup_failure(&self, task: &CleanupTask, error: String) -> Result<()> {
        let event = CleanupEvent {
            timestamp: Utc::now(),
            event_type: CleanupEventType::Failed,
            test_id: task.test_id.clone(),
            resource_type: task.resource_type.clone(),
            resource_id: task.resource_id.clone(),
            duration: Duration::from_secs(1), // Placeholder
            result: CleanupResult::Failed {
                error,
                files_attempted: 0,
            },
            details: HashMap::new(),
        };

        let mut cleanup_history = self.cleanup_history.lock();
        cleanup_history.push(event);

        // Update statistics
        let mut stats = self.cleanup_stats.lock();
        stats.total_tasks += 1;
        stats.failed_cleanups += 1;

        Ok(())
    }

    /// Get cleanup statistics
    pub async fn get_statistics(&self) -> Result<CleanupStatistics> {
        let stats = self.cleanup_stats.lock();
        // MutexGuard doesn't implement Clone, dereference to clone the inner value
        Ok((*stats).clone())
    }

    /// Get active cleanup operations
    pub async fn get_active_operations(&self) -> Result<Vec<CleanupOperation>> {
        let active_cleanups = self.active_cleanups.lock();
        Ok(active_cleanups.values().cloned().collect())
    }

    /// Get cleanup history
    pub async fn get_cleanup_history(&self) -> Result<Vec<CleanupEvent>> {
        let cleanup_history = self.cleanup_history.lock();
        Ok(cleanup_history.clone())
    }

    /// Force cleanup all resources
    pub async fn force_cleanup_all(&self) -> Result<usize> {
        info!("Initiating force cleanup of all resources");

        // Create emergency cleanup task
        let emergency_task = CleanupTask {
            task_id: format!("emergency-{}", Utc::now().timestamp()),
            test_id: "all".to_string(),
            resource_type: ResourceType::Mixed,
            resource_id: "all".to_string(),
            priority: CleanupPriority::Emergency,
            scheduled_time: Utc::now(),
            strategy: CleanupStrategy::Force,
            retry_count: 0,
            metadata: HashMap::new(),
        };

        self.execute_cleanup_task(&emergency_task).await?;
        Ok(1)
    }

    /// Update cleanup configuration
    pub async fn update_config(&self, config: ResourceCleanupConfig) -> Result<()> {
        let mut current_config = self.config.lock();
        *current_config = config;
        info!("Updated cleanup manager configuration");
        Ok(())
    }
}
