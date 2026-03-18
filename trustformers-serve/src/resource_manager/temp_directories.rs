//! Temporary directory lifecycle management for test parallelization.

// Re-export types for external access
pub use super::types::{
    CleanupEventType, CleanupResult, DirectoryPermissions, DirectoryStatus,
    DirectoryUsageStatistics, DirectoryUsageTracking, TempDirectoryCleanupPolicy,
};
use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    sync::Arc,
    time::Duration,
};
use tracing::{debug, info};

use crate::test_parallelization::TempDirPoolConfig;

/// Temporary directory management
pub struct TempDirectoryManager {
    /// Configuration
    config: Arc<RwLock<TempDirPoolConfig>>,
    /// Base directory for all temp directories
    base_directory: PathBuf,
    /// Available directories
    available_directories: Arc<Mutex<Vec<TempDirectoryInfo>>>,
    /// Allocated directories
    allocated_directories: Arc<Mutex<HashMap<String, TempDirectoryAllocation>>>,
    /// Directory cleanup scheduler
    cleanup_scheduler: Arc<DirectoryCleanupScheduler>,
    /// Directory usage statistics
    usage_stats: Arc<Mutex<DirectoryUsageStatistics>>,
}

/// Temporary directory information
#[derive(Debug, Clone)]
pub struct TempDirectoryInfo {
    /// Directory path
    pub path: PathBuf,
    /// Directory size limit (bytes)
    pub size_limit: u64,
    /// Current usage (bytes)
    pub current_usage: u64,
    /// Access permissions
    pub permissions: DirectoryPermissions,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last accessed timestamp
    pub last_accessed: DateTime<Utc>,
    /// Directory status
    pub status: DirectoryStatus,
}

/// Temporary directory allocation
#[derive(Debug, Clone)]
pub struct TempDirectoryAllocation {
    /// Allocated directory info
    pub directory: TempDirectoryInfo,
    /// Test ID that allocated the directory
    pub test_id: String,
    /// Allocation timestamp
    pub allocated_at: DateTime<Utc>,
    /// Expected cleanup time
    pub expected_cleanup: Option<DateTime<Utc>>,
    /// Cleanup policy
    pub cleanup_policy: TempDirectoryCleanupPolicy,
    /// Usage tracking
    pub usage_tracking: DirectoryUsageTracking,
}

/// Directory cleanup scheduler
#[derive(Debug)]
pub struct DirectoryCleanupScheduler {
    /// Cleanup queue
    cleanup_queue: Arc<Mutex<VecDeque<CleanupTask>>>,
    /// Scheduled cleanups
    scheduled_cleanups: Arc<Mutex<HashMap<String, DateTime<Utc>>>>,
    /// Cleanup history
    cleanup_history: Arc<Mutex<Vec<CleanupEvent>>>,
    /// Cleanup statistics
    cleanup_stats: Arc<Mutex<CleanupStatistics>>,
}

impl Default for DirectoryCleanupScheduler {
    fn default() -> Self {
        Self {
            cleanup_queue: Arc::new(Mutex::new(VecDeque::new())),
            scheduled_cleanups: Arc::new(Mutex::new(HashMap::new())),
            cleanup_history: Arc::new(Mutex::new(Vec::new())),
            cleanup_stats: Arc::new(Mutex::new(CleanupStatistics::default())),
        }
    }
}

/// Cleanup task
#[derive(Debug, Clone)]
pub struct CleanupTask {
    /// Task ID
    pub id: String,
    /// Directory to clean
    pub directory_path: PathBuf,
    /// Cleanup policy
    pub policy: TempDirectoryCleanupPolicy,
    /// Scheduled time
    pub scheduled_time: DateTime<Utc>,
    /// Task priority
    pub priority: f32,
    /// Associated test ID
    pub test_id: String,
    /// Retry count
    pub retry_count: usize,
}

/// Cleanup event record
#[derive(Debug, Clone)]
pub struct CleanupEvent {
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Event type
    pub event_type: CleanupEventType,
    /// Directory path
    pub directory_path: PathBuf,
    /// Test ID
    pub test_id: String,
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
    /// Total cleanups performed
    pub total_cleanups: u64,
    /// Successful cleanups
    pub successful_cleanups: u64,
    /// Failed cleanups
    pub failed_cleanups: u64,
    /// Total files removed
    pub total_files_removed: u64,
    /// Total bytes freed
    pub total_bytes_freed: u64,
    /// Average cleanup time
    pub average_cleanup_time: Duration,
    /// Cleanup efficiency
    pub cleanup_efficiency: f32,
}

impl TempDirectoryManager {
    /// Create new temporary directory manager
    pub async fn new(config: TempDirPoolConfig) -> Result<Self> {
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            base_directory: PathBuf::from("/tmp"),
            available_directories: Arc::new(Mutex::new(Vec::new())),
            allocated_directories: Arc::new(Mutex::new(HashMap::new())),
            cleanup_scheduler: Arc::new(DirectoryCleanupScheduler::default()),
            usage_stats: Arc::new(Mutex::new(DirectoryUsageStatistics::default())),
        })
    }

    /// Allocate directories for a test
    pub async fn allocate_directories(&self, count: usize, test_id: &str) -> Result<Vec<String>> {
        info!("Allocating {} directories for test: {}", count, test_id);

        // For now, return placeholder directory paths
        // In a real implementation, this would:
        // 1. Create unique temporary directories
        // 2. Set appropriate permissions
        // 3. Track the allocation
        // 4. Update statistics

        let directories: Vec<String> =
            (0..count).map(|i| format!("/tmp/test-{}-dir-{}", test_id, i)).collect();

        // Create allocation records
        for dir_path in &directories {
            let directory_info = TempDirectoryInfo {
                path: PathBuf::from(dir_path),
                size_limit: 1024 * 1024 * 100, // 100MB default
                current_usage: 0,
                permissions: DirectoryPermissions::default(),
                created_at: Utc::now(),
                last_accessed: Utc::now(),
                status: DirectoryStatus::Allocated,
            };

            let allocation = TempDirectoryAllocation {
                directory: directory_info,
                test_id: test_id.to_string(),
                allocated_at: Utc::now(),
                expected_cleanup: None,
                cleanup_policy: TempDirectoryCleanupPolicy::Immediate,
                usage_tracking: DirectoryUsageTracking {
                    files_created: 0,
                    bytes_written: 0,
                    bytes_read: 0,
                    peak_usage: 0,
                    usage_timeline: Vec::new(),
                },
            };

            let mut allocated_directories = self.allocated_directories.lock();
            allocated_directories.insert(dir_path.clone(), allocation);
        }

        // Update statistics
        let mut stats = self.usage_stats.lock();
        stats.total_created += directories.len() as u64;
        stats.currently_allocated += directories.len();

        info!(
            "Allocated directories {:?} for test: {}",
            directories, test_id
        );
        Ok(directories)
    }

    /// Deallocate directories for a test
    pub async fn deallocate_directories_for_test(&self, test_id: &str) -> Result<()> {
        debug!("Deallocating temporary directories for test: {}", test_id);

        let mut allocated_directories = self.allocated_directories.lock();
        let dirs_to_remove: Vec<String> = allocated_directories
            .iter()
            .filter(|(_, allocation)| allocation.test_id == test_id)
            .map(|(path, _)| path.clone())
            .collect();

        for dir_path in &dirs_to_remove {
            if let Some(allocation) = allocated_directories.remove(dir_path) {
                // Schedule cleanup based on policy
                self.schedule_cleanup_internal(&allocation).await?;
            }
        }

        // Update statistics
        let mut stats = self.usage_stats.lock();
        stats.currently_allocated = stats.currently_allocated.saturating_sub(dirs_to_remove.len());

        info!(
            "Released {} temporary directories for test: {}",
            dirs_to_remove.len(),
            test_id
        );
        Ok(())
    }

    /// Schedule cleanup for a directory
    pub async fn schedule_cleanup(&self, dir_path: &str) -> Result<()> {
        debug!("Scheduling cleanup for directory: {}", dir_path);

        let allocated_directories = self.allocated_directories.lock();
        if let Some(allocation) = allocated_directories.get(dir_path) {
            self.schedule_cleanup_internal(allocation).await?;
        }

        Ok(())
    }

    /// Internal cleanup scheduling
    async fn schedule_cleanup_internal(&self, allocation: &TempDirectoryAllocation) -> Result<()> {
        let scheduled_time = match &allocation.cleanup_policy {
            TempDirectoryCleanupPolicy::Immediate => Utc::now(),
            TempDirectoryCleanupPolicy::Delayed(delay) => {
                Utc::now() + chrono::Duration::from_std(*delay)?
            },
            TempDirectoryCleanupPolicy::SessionEnd => {
                // Schedule for later (placeholder)
                Utc::now() + chrono::Duration::hours(1)
            },
            TempDirectoryCleanupPolicy::Manual => {
                // Don't schedule automatic cleanup
                return Ok(());
            },
            TempDirectoryCleanupPolicy::Debug => {
                // Keep for extended time for debugging
                Utc::now() + chrono::Duration::days(1)
            },
        };

        let cleanup_task = CleanupTask {
            id: format!(
                "cleanup-{}-{}",
                allocation.test_id,
                allocation.directory.path.display()
            ),
            directory_path: allocation.directory.path.clone(),
            policy: allocation.cleanup_policy.clone(),
            scheduled_time,
            priority: 1.0,
            test_id: allocation.test_id.clone(),
            retry_count: 0,
        };

        let mut cleanup_queue = self.cleanup_scheduler.cleanup_queue.lock();
        cleanup_queue.push_back(cleanup_task);

        debug!(
            "Scheduled cleanup for directory: {}",
            allocation.directory.path.display()
        );
        Ok(())
    }

    /// Check if requested number of directories are available
    pub async fn check_availability(&self, count: usize) -> Result<bool> {
        // In a real implementation, this would check against available disk space
        // and current allocations
        let allocated_count = self.allocated_directories.lock().len();

        // Simple check assuming a maximum of 1000 directories
        Ok(allocated_count + count <= 1000)
    }

    /// Get directory usage statistics
    pub async fn get_statistics(&self) -> Result<DirectoryUsageStatistics> {
        let stats = self.usage_stats.lock();
        // MutexGuard doesn't implement Clone, dereference to clone the inner value
        Ok((*stats).clone())
    }

    /// Get current directory allocations
    pub async fn get_allocations(&self) -> Result<Vec<TempDirectoryAllocation>> {
        let allocated_directories = self.allocated_directories.lock();
        Ok(allocated_directories.values().cloned().collect())
    }

    /// Get allocations for a specific test
    pub async fn get_allocations_for_test(
        &self,
        test_id: &str,
    ) -> Result<Vec<TempDirectoryAllocation>> {
        let allocated_directories = self.allocated_directories.lock();
        Ok(allocated_directories
            .values()
            .filter(|allocation| allocation.test_id == test_id)
            .cloned()
            .collect())
    }

    /// Process pending cleanup tasks
    pub async fn process_cleanup_queue(&self) -> Result<usize> {
        let mut processed_count = 0;
        let now = Utc::now();

        // Get tasks ready for cleanup
        let mut cleanup_queue = self.cleanup_scheduler.cleanup_queue.lock();
        let mut ready_tasks = Vec::new();

        while let Some(task) = cleanup_queue.front() {
            if task.scheduled_time <= now {
                // Safe: we just checked front() returned Some
                ready_tasks.push(
                    cleanup_queue
                        .pop_front()
                        .expect("queue should not be empty after front() returned Some"),
                );
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
                    self.record_cleanup_event(
                        &task,
                        CleanupResult::Success {
                            files_removed: 0,
                            bytes_freed: 0,
                        },
                    )
                    .await?;
                },
                Err(e) => {
                    self.record_cleanup_event(
                        &task,
                        CleanupResult::Failed {
                            error: e.to_string(),
                            files_attempted: 0,
                        },
                    )
                    .await?;

                    // Retry logic could be implemented here
                    if task.retry_count < 3 {
                        let mut retry_task = task.clone();
                        retry_task.retry_count += 1;
                        retry_task.scheduled_time = now + chrono::Duration::minutes(5);

                        let mut cleanup_queue = self.cleanup_scheduler.cleanup_queue.lock();
                        cleanup_queue.push_back(retry_task);
                    }
                },
            }
        }

        Ok(processed_count)
    }

    /// Execute a cleanup task
    async fn execute_cleanup_task(&self, task: &CleanupTask) -> Result<()> {
        debug!(
            "Executing cleanup task for directory: {}",
            task.directory_path.display()
        );

        // In a real implementation, this would:
        // 1. Remove all files in the directory
        // 2. Remove the directory itself
        // 3. Update filesystem state
        // 4. Calculate cleanup metrics

        info!("Cleaned up directory: {}", task.directory_path.display());
        Ok(())
    }

    /// Record a cleanup event
    async fn record_cleanup_event(&self, task: &CleanupTask, result: CleanupResult) -> Result<()> {
        let event = CleanupEvent {
            timestamp: Utc::now(),
            event_type: match &result {
                CleanupResult::Success { .. } => CleanupEventType::Completed,
                CleanupResult::Failed { .. } => CleanupEventType::Failed,
                CleanupResult::Partial { .. } => CleanupEventType::Completed,
                CleanupResult::Skipped { .. } => CleanupEventType::Skipped,
            },
            directory_path: task.directory_path.clone(),
            test_id: task.test_id.clone(),
            duration: Duration::from_secs(1), // Placeholder
            result,
            details: HashMap::new(),
        };

        let mut cleanup_history = self.cleanup_scheduler.cleanup_history.lock();
        cleanup_history.push(event);

        // Keep only recent history (last 1000 events)
        if cleanup_history.len() > 1000 {
            cleanup_history.remove(0);
        }

        // Update cleanup statistics
        let mut stats = self.cleanup_scheduler.cleanup_stats.lock();
        stats.total_cleanups += 1;

        Ok(())
    }

    /// Get cleanup statistics
    pub async fn get_cleanup_statistics(&self) -> Result<CleanupStatistics> {
        let stats = self.cleanup_scheduler.cleanup_stats.lock();
        // MutexGuard doesn't implement Clone, dereference to clone the inner value
        Ok((*stats).clone())
    }

    /// Force cleanup all directories for emergency situations
    pub async fn force_cleanup_all(&self) -> Result<usize> {
        let mut allocated_directories = self.allocated_directories.lock();
        let count = allocated_directories.len();

        for (_, allocation) in allocated_directories.drain() {
            // Immediate cleanup without scheduling
            let _ = self
                .execute_cleanup_task(&CleanupTask {
                    id: format!("force-cleanup-{}", allocation.test_id),
                    directory_path: allocation.directory.path,
                    policy: TempDirectoryCleanupPolicy::Immediate,
                    scheduled_time: Utc::now(),
                    priority: 10.0,
                    test_id: allocation.test_id,
                    retry_count: 0,
                })
                .await;
        }

        // Reset statistics
        let mut stats = self.usage_stats.lock();
        stats.currently_allocated = 0;

        info!("Force cleaned up all {} temporary directories", count);
        Ok(count)
    }

    /// Update configuration
    pub async fn update_config(&self, config: TempDirPoolConfig) -> Result<()> {
        let mut current_config = self.config.write();
        *current_config = config;
        info!("Updated temporary directory manager configuration");
        Ok(())
    }
}
