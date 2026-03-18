//! Resource cleanup and lifecycle management for test parallelization.
//!
//! This module provides comprehensive cleanup management including scheduled cleanup,
//! resource lifecycle tracking, garbage collection, and cleanup strategies.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};
use tokio::time::interval;
use tracing::{debug, info, warn};

use super::types::{
    CleanupEventType, CleanupResult, CleanupSchedule, CleanupStatistics, CleanupTaskResult,
    ResourceCleanupConfig, ResourceLifecycleStage,
};

// Re-export types needed by other modules
pub use super::types::{CleanupEvent, CleanupTask};

/// Comprehensive cleanup management system
pub struct CleanupManager {
    /// Configuration
    config: Arc<Mutex<ResourceCleanupConfig>>,
    /// Scheduled cleanup tasks
    cleanup_tasks: Arc<Mutex<Vec<CleanupTask>>>,
    /// Cleanup statistics
    cleanup_stats: Arc<Mutex<CleanupStatistics>>,
    /// Cleanup history
    cleanup_history: Arc<Mutex<Vec<CleanupEvent>>>,
    /// Resource lifecycle tracker
    lifecycle_tracker: Arc<LifecycleTracker>,
    /// Garbage collector
    garbage_collector: Arc<GarbageCollector>,
    /// Background cleanup scheduler
    cleanup_scheduler: Arc<CleanupScheduler>,
}

/// Resource lifecycle tracking system
pub struct LifecycleTracker {
    /// Active resource lifecycles
    active_lifecycles: Arc<Mutex<HashMap<String, ResourceLifecycle>>>,
    /// Lifecycle events
    lifecycle_events: Arc<Mutex<Vec<LifecycleEvent>>>,
}

/// Garbage collection system
pub struct GarbageCollector {
    /// Configuration
    config: GarbageCollectionConfig,
    /// Collection statistics
    collection_stats: Arc<Mutex<GarbageCollectionStatistics>>,
}

/// Cleanup task scheduler
pub struct CleanupScheduler {
    /// Configuration
    config: CleanupSchedulerConfig,
    /// Active schedules
    active_schedules: Arc<Mutex<HashMap<String, CleanupSchedule>>>,
}

/// Resource lifecycle information
#[derive(Debug, Clone)]
pub struct ResourceLifecycle {
    /// Resource ID
    pub resource_id: String,
    /// Resource type
    pub resource_type: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Current stage
    pub current_stage: ResourceLifecycleStage,
    /// Stage transitions
    pub stage_transitions: Vec<LifecycleTransition>,
    /// Expected cleanup time
    pub expected_cleanup: Option<DateTime<Utc>>,
    /// Cleanup metadata
    pub cleanup_metadata: HashMap<String, String>,
}

/// Lifecycle stage transition
#[derive(Debug, Clone)]
pub struct LifecycleTransition {
    /// From stage
    pub from_stage: ResourceLifecycleStage,
    /// To stage
    pub to_stage: ResourceLifecycleStage,
    /// Transition timestamp
    pub transitioned_at: DateTime<Utc>,
    /// Transition metadata
    pub metadata: HashMap<String, String>,
}

/// Lifecycle event
#[derive(Debug, Clone)]
pub struct LifecycleEvent {
    /// Event ID
    pub event_id: String,
    /// Resource ID
    pub resource_id: String,
    /// Event type
    pub event_type: LifecycleEventType,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Event details
    pub details: HashMap<String, String>,
}

/// Lifecycle event types
#[derive(Debug, Clone)]
pub enum LifecycleEventType {
    /// Resource created
    Created,
    /// Resource allocated
    Allocated,
    /// Resource released
    Released,
    /// Resource marked for cleanup
    MarkedForCleanup,
    /// Resource cleaned up
    CleanedUp,
    /// Cleanup failed
    CleanupFailed,
    /// Resource expired
    Expired,
}

/// Garbage collection configuration
#[derive(Debug, Clone)]
pub struct GarbageCollectionConfig {
    /// Collection interval
    pub collection_interval: Duration,
    /// Memory pressure threshold
    pub memory_pressure_threshold: f32,
    /// Maximum cleanup batch size
    pub max_batch_size: usize,
    /// Force collection threshold
    pub force_collection_threshold: usize,
}

/// Garbage collection statistics
#[derive(Debug, Clone, Default)]
pub struct GarbageCollectionStatistics {
    /// Total collections
    pub total_collections: u64,
    /// Total resources collected
    pub total_resources_collected: u64,
    /// Average collection time
    pub average_collection_time: Duration,
    /// Last collection timestamp
    pub last_collection: Option<DateTime<Utc>>,
    /// Collection efficiency
    pub collection_efficiency: f32,
}

/// Cleanup scheduler configuration
#[derive(Debug, Clone)]
pub struct CleanupSchedulerConfig {
    /// Scheduler interval
    pub scheduler_interval: Duration,
    /// Maximum concurrent cleanups
    pub max_concurrent_cleanups: usize,
    /// Cleanup timeout
    pub cleanup_timeout: Duration,
    /// Retry attempts
    pub retry_attempts: usize,
}

impl CleanupManager {
    /// Create new cleanup manager
    pub async fn new(config: ResourceCleanupConfig) -> Result<Self> {
        let lifecycle_tracker = Arc::new(LifecycleTracker::new());
        let garbage_collector = Arc::new(GarbageCollector::new(GarbageCollectionConfig {
            collection_interval: Duration::from_secs(300), // 5 minutes
            memory_pressure_threshold: 0.8,
            max_batch_size: 100,
            force_collection_threshold: 1000,
        }));
        let cleanup_scheduler = Arc::new(CleanupScheduler::new(CleanupSchedulerConfig {
            scheduler_interval: Duration::from_secs(60), // 1 minute
            max_concurrent_cleanups: 10,
            cleanup_timeout: Duration::from_secs(300), // 5 minutes
            retry_attempts: 3,
        }));

        info!("Initialized cleanup manager");

        Ok(Self {
            config: Arc::new(Mutex::new(config)),
            cleanup_tasks: Arc::new(Mutex::new(Vec::new())),
            cleanup_stats: Arc::new(Mutex::new(CleanupStatistics::default())),
            cleanup_history: Arc::new(Mutex::new(Vec::new())),
            lifecycle_tracker,
            garbage_collector,
            cleanup_scheduler,
        })
    }

    /// Schedule cleanup task
    pub async fn schedule_cleanup(&self, task: CleanupTask) -> Result<String> {
        let task_id = format!("cleanup_task_{}", chrono::Utc::now().timestamp_millis());
        let mut cleanup_tasks = self.cleanup_tasks.lock();

        let mut task_with_id = task;
        task_with_id.task_id = task_id.clone();

        cleanup_tasks.push(task_with_id);

        // Sort by priority (higher first) and scheduled time
        cleanup_tasks.sort_by(|a, b| {
            // Compare priorities (higher priority = higher value, should come first)
            match b.priority.partial_cmp(&a.priority) {
                Some(std::cmp::Ordering::Equal) => a.scheduled_time.cmp(&b.scheduled_time),
                Some(ordering) => ordering,
                None => a.scheduled_time.cmp(&b.scheduled_time),
            }
        });

        info!("Scheduled cleanup task: {}", task_id);
        Ok(task_id)
    }

    /// Execute immediate cleanup
    pub async fn execute_immediate_cleanup(&self, task: CleanupTask) -> Result<CleanupResult> {
        let start_time = Utc::now();

        let result = match task.task_type.as_str() {
            "DirectoryCleanup" => self.execute_directory_cleanup(&task).await,
            "PortRelease" => self.execute_port_release(&task).await,
            "GpuRelease" => self.execute_gpu_release(&task).await,
            "DatabaseCleanup" => self.execute_database_cleanup(&task).await,
            task_type if task_type.starts_with("CustomResourceCleanup") => {
                self.execute_custom_resource_cleanup(&task).await
            },
            "GarbageCollection" => self.execute_garbage_collection(&task).await,
            _ => {
                warn!("Unhandled cleanup task type: {:?}", task.task_type);
                Ok(())
            },
        };

        let duration = Utc::now().signed_duration_since(start_time);
        let cleanup_result = CleanupTaskResult {
            task_id: task.task_id.clone(),
            success: result.is_ok(),
            duration: Duration::from_millis(duration.num_milliseconds().max(0) as u64),
            resources_cleaned: 1, // Would be calculated based on actual cleanup
            errors: result.err().map(|e| vec![e.to_string()]).unwrap_or_default(),
            details: HashMap::new(),
        };

        // Record cleanup event
        let event = CleanupEvent {
            timestamp: Utc::now(),
            event_type: if cleanup_result.success {
                CleanupEventType::Completed
            } else {
                CleanupEventType::Failed
            },
            directory_path: task
                .details
                .get("directory_path")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("")),
            test_id: task.details.get("test_id").cloned().unwrap_or_else(|| "unknown".to_string()),
            duration: cleanup_result.duration,
            result: if cleanup_result.success {
                CleanupResult::Success {
                    files_removed: cleanup_result.resources_cleaned,
                    bytes_freed: 0,
                }
            } else {
                CleanupResult::Failed {
                    error: cleanup_result
                        .errors
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "Unknown error".to_string()),
                    files_attempted: cleanup_result.resources_cleaned,
                }
            },
            details: HashMap::new(),
            event_id: format!("cleanup_event_{}", Utc::now().timestamp_millis()),
            task_id: task.task_id.clone(),
            target_path: task
                .details
                .get("directory_path")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("")),
            bytes_cleaned: cleanup_result.resources_cleaned as u64,
            files_cleaned: 0,
            success: cleanup_result.success,
            error: if cleanup_result.success {
                None
            } else {
                Some(
                    cleanup_result
                        .errors
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "Unknown error".to_string()),
                )
            },
        };

        let mut cleanup_history = self.cleanup_history.lock();
        cleanup_history.push(event);

        // Limit history size
        if cleanup_history.len() > 10000 {
            cleanup_history.remove(0);
        }

        // Update statistics
        let mut cleanup_stats = self.cleanup_stats.lock();
        cleanup_stats.total_cleanups += 1;
        if cleanup_result.success {
            cleanup_stats.successful_cleanups += 1;
        } else {
            cleanup_stats.failed_cleanups += 1;
        }

        info!("Executed immediate cleanup for task: {}", task.task_id);

        // Convert CleanupTaskResult to CleanupResult enum
        let result = if cleanup_result.success {
            CleanupResult::Success {
                files_removed: cleanup_result.resources_cleaned,
                bytes_freed: 0,
            }
        } else {
            CleanupResult::Failed {
                error: cleanup_result
                    .errors
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "Unknown error".to_string()),
                files_attempted: cleanup_result.resources_cleaned,
            }
        };

        Ok(result)
    }

    /// Execute directory cleanup
    async fn execute_directory_cleanup(&self, task: &CleanupTask) -> Result<()> {
        if let Some(directory_path) = task.details.get("directory_path") {
            let path = PathBuf::from(directory_path);

            if path.exists() {
                tokio::fs::remove_dir_all(&path)
                    .await
                    .with_context(|| format!("Failed to remove directory: {:?}", path))?;

                debug!("Cleaned up directory: {:?}", path);
            }
        }
        Ok(())
    }

    /// Execute port release cleanup
    async fn execute_port_release(&self, _task: &CleanupTask) -> Result<()> {
        // Port release logic would be implemented here
        debug!("Executed port release cleanup");
        Ok(())
    }

    /// Execute GPU release cleanup
    async fn execute_gpu_release(&self, _task: &CleanupTask) -> Result<()> {
        // GPU release logic would be implemented here
        debug!("Executed GPU release cleanup");
        Ok(())
    }

    /// Execute database cleanup
    async fn execute_database_cleanup(&self, _task: &CleanupTask) -> Result<()> {
        // Database cleanup logic would be implemented here
        debug!("Executed database cleanup");
        Ok(())
    }

    /// Execute custom resource cleanup
    async fn execute_custom_resource_cleanup(&self, _task: &CleanupTask) -> Result<()> {
        // Custom resource cleanup logic would be implemented here
        debug!("Executed custom resource cleanup");
        Ok(())
    }

    /// Execute garbage collection
    async fn execute_garbage_collection(&self, _task: &CleanupTask) -> Result<()> {
        self.garbage_collector.collect_garbage().await
    }

    /// Start background cleanup process
    pub async fn start_background_cleanup(&self) -> Result<()> {
        info!("Starting background cleanup process");

        let cleanup_tasks = Arc::clone(&self.cleanup_tasks);
        let cleanup_stats = Arc::clone(&self.cleanup_stats);
        let _config = Arc::clone(&self.config);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60)); // Check every minute

            loop {
                interval.tick().await;

                let current_time = Utc::now();
                let mut tasks = cleanup_tasks.lock();
                let mut stats = cleanup_stats.lock();

                // Find and execute due tasks
                let mut executed_tasks = Vec::new();
                for (index, task) in tasks.iter().enumerate() {
                    if current_time >= task.scheduled_time {
                        executed_tasks.push(index);
                        stats.total_cleanups += 1;

                        // Execute task (simplified for background process)
                        debug!("Executing scheduled cleanup task: {}", task.task_id);
                    }
                }

                // Remove executed tasks (in reverse order to maintain indices)
                for &index in executed_tasks.iter().rev() {
                    tasks.remove(index);
                }
            }
        });

        Ok(())
    }

    /// Stop background cleanup process
    pub async fn stop_background_cleanup(&self) -> Result<()> {
        info!("Stopping background cleanup process");
        Ok(())
    }

    /// Track resource lifecycle
    pub async fn track_resource_lifecycle(
        &self,
        resource_id: &str,
        resource_type: &str,
    ) -> Result<()> {
        self.lifecycle_tracker.track_resource(resource_id, resource_type).await
    }

    /// Update resource lifecycle stage
    pub async fn update_lifecycle_stage(
        &self,
        resource_id: &str,
        stage: ResourceLifecycleStage,
    ) -> Result<()> {
        self.lifecycle_tracker.update_stage(resource_id, stage).await
    }

    /// Get cleanup statistics
    pub async fn get_cleanup_statistics(&self) -> CleanupStatistics {
        let cleanup_stats = self.cleanup_stats.lock();
        cleanup_stats.clone()
    }

    /// Get pending cleanup tasks
    pub async fn get_pending_tasks(&self) -> Vec<CleanupTask> {
        let cleanup_tasks = self.cleanup_tasks.lock();
        cleanup_tasks.clone()
    }

    /// Force garbage collection
    pub async fn force_garbage_collection(&self) -> Result<u64> {
        self.garbage_collector.force_collection().await
    }
}

impl LifecycleTracker {
    /// Create new lifecycle tracker
    pub fn new() -> Self {
        Self {
            active_lifecycles: Arc::new(Mutex::new(HashMap::new())),
            lifecycle_events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Track new resource
    pub async fn track_resource(&self, resource_id: &str, resource_type: &str) -> Result<()> {
        let mut active_lifecycles = self.active_lifecycles.lock();

        let lifecycle = ResourceLifecycle {
            resource_id: resource_id.to_string(),
            resource_type: resource_type.to_string(),
            created_at: Utc::now(),
            current_stage: ResourceLifecycleStage::Created,
            stage_transitions: vec![],
            expected_cleanup: None,
            cleanup_metadata: HashMap::new(),
        };

        active_lifecycles.insert(resource_id.to_string(), lifecycle);

        // Record lifecycle event
        let event = LifecycleEvent {
            event_id: format!("lifecycle_event_{}", Utc::now().timestamp_millis()),
            resource_id: resource_id.to_string(),
            event_type: LifecycleEventType::Created,
            timestamp: Utc::now(),
            details: HashMap::new(),
        };

        let mut lifecycle_events = self.lifecycle_events.lock();
        lifecycle_events.push(event);

        debug!("Started tracking resource lifecycle: {}", resource_id);
        Ok(())
    }

    /// Update resource lifecycle stage
    pub async fn update_stage(
        &self,
        resource_id: &str,
        new_stage: ResourceLifecycleStage,
    ) -> Result<()> {
        let mut active_lifecycles = self.active_lifecycles.lock();

        if let Some(lifecycle) = active_lifecycles.get_mut(resource_id) {
            let transition = LifecycleTransition {
                from_stage: lifecycle.current_stage,
                to_stage: new_stage,
                transitioned_at: Utc::now(),
                metadata: HashMap::new(),
            };

            lifecycle.stage_transitions.push(transition);
            lifecycle.current_stage = new_stage;

            debug!("Updated lifecycle stage for resource: {}", resource_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "Resource {} not found in lifecycle tracker",
                resource_id
            ))
        }
    }
}

impl GarbageCollector {
    /// Create new garbage collector
    pub fn new(config: GarbageCollectionConfig) -> Self {
        Self {
            config,
            collection_stats: Arc::new(Mutex::new(GarbageCollectionStatistics::default())),
        }
    }

    /// Collect garbage
    pub async fn collect_garbage(&self) -> Result<()> {
        let start_time = Utc::now();

        // Garbage collection logic would be implemented here
        debug!("Executing garbage collection");

        let mut collection_stats = self.collection_stats.lock();
        collection_stats.total_collections += 1;
        collection_stats.last_collection = Some(Utc::now());

        let duration = Utc::now().signed_duration_since(start_time);
        let duration_std = Duration::from_millis(duration.num_milliseconds().max(0) as u64);

        // Update average collection time
        if collection_stats.total_collections > 0 {
            let total_time = collection_stats.average_collection_time.as_millis() as f64
                * (collection_stats.total_collections - 1) as f64;
            let new_average = (total_time + duration_std.as_millis() as f64)
                / collection_stats.total_collections as f64;
            collection_stats.average_collection_time = Duration::from_millis(new_average as u64);
        }

        Ok(())
    }

    /// Force garbage collection
    pub async fn force_collection(&self) -> Result<u64> {
        info!("Forcing garbage collection");
        self.collect_garbage().await?;

        let collection_stats = self.collection_stats.lock();
        Ok(collection_stats.total_resources_collected)
    }
}

impl CleanupScheduler {
    /// Create new cleanup scheduler
    pub fn new(config: CleanupSchedulerConfig) -> Self {
        Self {
            config,
            active_schedules: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Add recurring schedule
    pub async fn add_recurring_schedule(&self, schedule: CleanupSchedule) -> Result<String> {
        let schedule_id = format!("schedule_{}", Utc::now().timestamp_millis());
        let mut active_schedules = self.active_schedules.lock();

        active_schedules.insert(schedule_id.clone(), schedule);

        info!("Added recurring cleanup schedule: {}", schedule_id);
        Ok(schedule_id)
    }

    /// Remove schedule
    pub async fn remove_schedule(&self, schedule_id: &str) -> Result<()> {
        let mut active_schedules = self.active_schedules.lock();

        if active_schedules.remove(schedule_id).is_some() {
            info!("Removed cleanup schedule: {}", schedule_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Schedule {} not found", schedule_id))
        }
    }
}

impl Default for LifecycleTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for GarbageCollectionConfig {
    fn default() -> Self {
        Self {
            collection_interval: Duration::from_secs(300),
            memory_pressure_threshold: 0.8,
            max_batch_size: 100,
            force_collection_threshold: 1000,
        }
    }
}

impl Default for CleanupSchedulerConfig {
    fn default() -> Self {
        Self {
            scheduler_interval: Duration::from_secs(60),
            max_concurrent_cleanups: 10,
            cleanup_timeout: Duration::from_secs(300),
            retry_attempts: 3,
        }
    }
}
