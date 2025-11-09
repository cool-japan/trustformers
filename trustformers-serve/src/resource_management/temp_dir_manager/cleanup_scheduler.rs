//! Directory Cleanup Scheduler Implementation
//!
//! This module handles the scheduling and execution of cleanup tasks for temporary directories,
//! including immediate cleanup, delayed cleanup, and bulk cleanup operations.

use super::types::*;
use super::utils::*;
use crate::resource_management::types::TempDirectoryCleanupPolicy;

use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use std::sync::Arc;
use std::{
    collections::{HashMap, VecDeque},
    fs,
    path::Path,
};
use tracing::{debug, error, info, instrument, warn};

// ================================================================================================
// Directory Cleanup Scheduler
// ================================================================================================

/// Manages cleanup operations for temporary directories
///
/// This component handles the scheduling and execution of cleanup tasks,
/// including immediate cleanup, delayed cleanup, and bulk cleanup operations.
#[derive(Debug)]
pub struct DirectoryCleanupScheduler {
    /// Queue of scheduled cleanup tasks
    scheduled_tasks: Arc<Mutex<VecDeque<CleanupTask>>>,

    /// Cleanup execution history
    cleanup_history: Arc<Mutex<Vec<CleanupEvent>>>,

    /// Cleanup operation statistics
    cleanup_stats: Arc<Mutex<CleanupStatistics>>,

    /// Automatic cleanup enabled flag
    auto_cleanup_enabled: bool,

    /// Maximum number of concurrent cleanup operations
    max_concurrent_operations: usize,

    /// Currently executing cleanup operations
    active_operations: Arc<Mutex<HashMap<String, DateTime<Utc>>>>,
}

impl DirectoryCleanupScheduler {
    /// Create a new cleanup scheduler
    pub fn new() -> Self {
        Self {
            scheduled_tasks: Arc::new(Mutex::new(VecDeque::new())),
            cleanup_history: Arc::new(Mutex::new(Vec::new())),
            cleanup_stats: Arc::new(Mutex::new(CleanupStatistics::default())),
            auto_cleanup_enabled: true,
            max_concurrent_operations: 4,
            active_operations: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create a new cleanup scheduler with custom configuration
    pub fn with_config(auto_cleanup_enabled: bool, max_concurrent_operations: usize) -> Self {
        Self {
            scheduled_tasks: Arc::new(Mutex::new(VecDeque::new())),
            cleanup_history: Arc::new(Mutex::new(Vec::new())),
            cleanup_stats: Arc::new(Mutex::new(CleanupStatistics::default())),
            auto_cleanup_enabled,
            max_concurrent_operations,
            active_operations: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Schedule a cleanup task
    ///
    /// # Arguments
    ///
    /// * `task` - The cleanup task to schedule
    ///
    /// # Errors
    ///
    /// This function will return an error if task validation fails
    #[instrument(skip(self))]
    pub async fn schedule_task(&self, task: CleanupTask) -> TempDirResult<()> {
        self.validate_task(&task)?;

        let mut scheduled_tasks = self.scheduled_tasks.lock();

        // Insert task in priority order
        let insert_index = scheduled_tasks
            .iter()
            .position(|t| t.priority < task.priority)
            .unwrap_or(scheduled_tasks.len());

        scheduled_tasks.insert(insert_index, task.clone());

        debug!(
            task_id = %task.id,
            priority = ?task.priority,
            expected_execution = ?task.expected_execution_time,
            "Scheduled cleanup task"
        );

        // If this is a critical priority task, trigger immediate execution
        if task.priority == CleanupPriority::Critical {
            drop(scheduled_tasks); // Release lock before executing
            tokio::spawn({
                let scheduler = self.clone();
                async move {
                    if let Err(e) = scheduler.execute_pending_tasks().await {
                        error!(error = %e, "Failed to execute critical cleanup task");
                    }
                }
            });
        }

        Ok(())
    }

    /// Execute pending cleanup tasks
    ///
    /// # Returns
    ///
    /// Returns the number of tasks executed
    ///
    /// # Errors
    ///
    /// This function will return an error if task execution fails
    #[instrument(skip(self))]
    pub async fn execute_pending_tasks(&self) -> TempDirResult<usize> {
        if !self.auto_cleanup_enabled {
            debug!("Auto cleanup is disabled, skipping execution");
            return Ok(0);
        }

        let now = Utc::now();
        let mut executed_count = 0;

        // Check current active operations
        {
            let active_operations = self.active_operations.lock();
            let max_operations = self.max_concurrent_operations;

            if active_operations.len() >= max_operations {
                debug!(
                    active_count = active_operations.len(),
                    max_operations = max_operations,
                    "Maximum concurrent operations reached, deferring execution"
                );
                return Ok(0);
            }
        }

        let mut tasks_to_execute = Vec::new();

        // Find tasks ready for execution
        {
            let mut scheduled_tasks = self.scheduled_tasks.lock();
            while let Some(task) = scheduled_tasks.front() {
                if task.is_ready(now) && tasks_to_execute.len() < self.max_concurrent_operations {
                    tasks_to_execute.push(scheduled_tasks.pop_front().unwrap());
                } else {
                    break;
                }
            }
        }

        if tasks_to_execute.is_empty() {
            return Ok(0);
        }

        debug!(
            task_count = tasks_to_execute.len(),
            "Executing cleanup tasks"
        );

        // Execute tasks concurrently
        let mut join_handles = Vec::new();

        for task in tasks_to_execute {
            let scheduler_clone = self.clone();
            let task_clone = task.clone();

            let handle =
                tokio::spawn(async move { scheduler_clone.execute_single_task(task_clone).await });

            join_handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in join_handles {
            match handle.await {
                Ok(Ok(())) => executed_count += 1,
                Ok(Err(e)) => {
                    error!(error = %e, "Cleanup task execution failed");
                },
                Err(e) => {
                    error!(error = %e, "Cleanup task join failed");
                },
            }
        }

        if executed_count > 0 {
            info!(executed_count = %executed_count, "Executed cleanup tasks");
        }

        Ok(executed_count)
    }

    /// Execute a single cleanup task
    async fn execute_single_task(&self, task: CleanupTask) -> TempDirResult<()> {
        let execution_start = Utc::now();

        // Register as active operation
        {
            let mut active_operations = self.active_operations.lock();
            active_operations.insert(task.id.clone(), execution_start);
        }

        debug!(
            task_id = %task.id,
            path = %task.directory_path.display(),
            "Starting cleanup task execution"
        );

        let result = self.execute_cleanup_task(&task).await;

        // Remove from active operations
        {
            let mut active_operations = self.active_operations.lock();
            active_operations.remove(&task.id);
        }

        // Record cleanup event
        let cleanup_event = match result {
            Ok((bytes_cleaned, files_cleaned)) => {
                let mut cleanup_stats = self.cleanup_stats.lock();
                cleanup_stats.successful_cleanups += 1;
                cleanup_stats.total_bytes_freed += bytes_cleaned;
                cleanup_stats.total_files_removed += files_cleaned;

                CleanupEvent {
                    timestamp: execution_start,
                    event_type: CleanupEventType::Completed,
                    directory_path: task.directory_path.clone(),
                    test_id: task.test_id.clone(),
                    duration: std::time::Duration::from_millis(
                        (Utc::now().timestamp_millis() - execution_start.timestamp_millis()) as u64,
                    ),
                    result: CleanupResult::Success {
                        files_removed: files_cleaned,
                        bytes_freed: bytes_cleaned,
                    },
                    details: HashMap::new(),
                }
            },
            Err(e) => {
                let mut cleanup_stats = self.cleanup_stats.lock();
                cleanup_stats.failed_cleanups += 1;

                let error_string = e.to_string();
                error!(
                    task_id = %task.id,
                    error = %error_string,
                    "Failed to execute cleanup task"
                );

                CleanupEvent {
                    timestamp: execution_start,
                    event_type: CleanupEventType::Failed,
                    directory_path: task.directory_path.clone(),
                    test_id: task.test_id.clone(),
                    duration: std::time::Duration::from_millis(
                        (Utc::now().timestamp_millis() - execution_start.timestamp_millis()) as u64,
                    ),
                    result: CleanupResult::Failed {
                        error: error_string,
                        files_attempted: 0,
                    },
                    details: HashMap::new(),
                }
            },
        };

        // Update statistics and history
        {
            let mut cleanup_history = self.cleanup_history.lock();
            let mut cleanup_stats = self.cleanup_stats.lock();

            cleanup_history.push(cleanup_event.clone());
            cleanup_stats.update_after_cleanup(&cleanup_event.result, cleanup_event.duration);

            // Limit history size to prevent unbounded growth
            while cleanup_history.len() > 10000 {
                cleanup_history.remove(0);
            }
        }

        // Return result based on cleanup event type
        match cleanup_event.event_type {
            CleanupEventType::Completed => {
                info!(
                    task_id = %task.id,
                    path = %task.directory_path.display(),
                    "Successfully completed cleanup task"
                );
                Ok(())
            },
            CleanupEventType::Failed => Err(TempDirError::CleanupFailed {
                path: task.directory_path.display().to_string(),
                message: "Cleanup task failed".to_string(),
            }),
            _ => Ok(()),
        }
    }

    /// Execute a cleanup task based on its policy
    async fn execute_cleanup_task(&self, task: &CleanupTask) -> TempDirResult<(u64, usize)> {
        match &task.policy {
            TempDirectoryCleanupPolicy::Immediate => {
                self.delete_directory(&task.directory_path).await
            },
            TempDirectoryCleanupPolicy::Delayed(delay) => {
                // Check if enough time has passed
                let elapsed = Utc::now().signed_duration_since(task.scheduled_time);
                if elapsed >= chrono::Duration::from_std(*delay).unwrap_or_default() {
                    self.delete_directory(&task.directory_path).await
                } else {
                    // Reschedule for later
                    let reschedule_task = CleanupTask {
                        id: task.id.clone(),
                        directory_path: task.directory_path.clone(),
                        test_id: task.test_id.clone(),
                        policy: task.policy.clone(),
                        priority: task.priority.clone(),
                        scheduled_time: Utc::now()
                            + chrono::Duration::from_std(*delay).unwrap_or_default(),
                        expected_execution_time: Some(
                            Utc::now() + chrono::Duration::from_std(*delay).unwrap_or_default(),
                        ),
                        task_type: task.task_type.clone(),
                    };

                    self.scheduled_tasks.lock().push_back(reschedule_task);
                    Ok((0, 0)) // No cleanup performed, just rescheduled
                }
            },
            TempDirectoryCleanupPolicy::SessionEnd => {
                // For session end, we always clean up immediately when scheduled
                self.delete_directory(&task.directory_path).await
            },
            TempDirectoryCleanupPolicy::Manual => {
                // Manual cleanup is handled externally
                debug!(
                    path = %task.directory_path.display(),
                    "Skipping manual cleanup task"
                );
                Ok((0, 0))
            },
            TempDirectoryCleanupPolicy::Debug => {
                // Debug mode preserves directories
                debug!(
                    path = %task.directory_path.display(),
                    "Skipping cleanup for debug directory"
                );
                Ok((0, 0))
            },
        }
    }

    /// Delete a directory and calculate cleanup statistics
    async fn delete_directory(&self, path: &Path) -> TempDirResult<(u64, usize)> {
        if !path.exists() {
            debug!(path = %path.display(), "Directory does not exist, skipping cleanup");
            return Ok((0, 0));
        }

        // Calculate size before deletion
        let (bytes_to_clean, files_to_clean) =
            calculate_directory_size(path).map_err(|e| TempDirError::CleanupFailed {
                path: path.display().to_string(),
                message: format!("Failed to calculate directory size: {}", e),
            })?;

        // Perform deletion
        fs::remove_dir_all(path).map_err(|e| TempDirError::CleanupFailed {
            path: path.display().to_string(),
            message: format!("Failed to remove directory: {}", e),
        })?;

        info!(
            path = %path.display(),
            bytes_cleaned = %bytes_to_clean,
            files_cleaned = %files_to_clean,
            "Successfully deleted directory"
        );

        Ok((bytes_to_clean, files_to_clean))
    }

    /// Get cleanup statistics
    pub async fn get_statistics(&self) -> CleanupStatistics {
        self.cleanup_stats.lock().clone()
    }

    /// Generate a cleanup report
    pub async fn generate_report(&self) -> String {
        let stats = self.get_statistics().await;
        let scheduled_count = self.scheduled_tasks.lock().len();
        let active_count = self.active_operations.lock().len();

        format!(
            "== Cleanup Scheduler Report ==\n\
             Scheduled tasks: {}\n\
             Active operations: {}\n\
             Auto cleanup enabled: {}\n\
             Max concurrent operations: {}\n\
             \n\
             {}",
            scheduled_count,
            active_count,
            self.auto_cleanup_enabled,
            self.max_concurrent_operations,
            stats.generate_report()
        )
    }

    /// Get pending tasks count
    pub async fn get_pending_task_count(&self) -> usize {
        self.scheduled_tasks.lock().len()
    }

    /// Get active operations count
    pub async fn get_active_operations_count(&self) -> usize {
        self.active_operations.lock().len()
    }

    /// Cancel all pending tasks for a specific test
    pub async fn cancel_tasks_for_test(&self, test_id: &str) -> usize {
        let mut scheduled_tasks = self.scheduled_tasks.lock();
        let original_len = scheduled_tasks.len();

        scheduled_tasks.retain(|task| task.test_id != test_id);

        let cancelled_count = original_len - scheduled_tasks.len();

        if cancelled_count > 0 {
            info!(
                test_id = %test_id,
                cancelled_count = %cancelled_count,
                "Cancelled cleanup tasks for test"
            );
        }

        cancelled_count
    }

    /// Force cleanup of a specific directory (bypass scheduling)
    pub async fn force_cleanup(&self, path: &std::path::PathBuf) -> TempDirResult<(u64, usize)> {
        warn!(path = %path.display(), "Performing force cleanup");
        self.delete_directory(path).await
    }

    /// Enable or disable automatic cleanup
    pub async fn set_auto_cleanup_enabled(&mut self, enabled: bool) {
        self.auto_cleanup_enabled = enabled;
        info!(enabled = %enabled, "Auto cleanup setting changed");
    }

    /// Set maximum concurrent cleanup operations
    pub async fn set_max_concurrent_operations(&mut self, max_operations: usize) {
        if max_operations == 0 {
            warn!("Attempted to set max concurrent operations to 0, ignoring");
            return;
        }

        self.max_concurrent_operations = max_operations;
        info!(max_operations = %max_operations, "Max concurrent operations updated");
    }

    /// Get cleanup history (last N events)
    pub async fn get_cleanup_history(&self, limit: Option<usize>) -> Vec<CleanupEvent> {
        let history = self.cleanup_history.lock();
        let limit = limit.unwrap_or(100);

        if history.len() <= limit {
            history.clone()
        } else {
            history[history.len() - limit..].to_vec()
        }
    }

    /// Clear cleanup history
    pub async fn clear_cleanup_history(&self) {
        self.cleanup_history.lock().clear();
        info!("Cleanup history cleared");
    }

    /// Validate a cleanup task before scheduling
    fn validate_task(&self, task: &CleanupTask) -> TempDirResult<()> {
        if task.directory_path.as_os_str().is_empty() {
            return Err(TempDirError::ConfigurationError {
                message: "Task directory path cannot be empty".to_string(),
            });
        }

        if task.test_id.is_empty() {
            return Err(TempDirError::ConfigurationError {
                message: "Task test ID cannot be empty".to_string(),
            });
        }

        if !task.directory_path.is_absolute() {
            return Err(TempDirError::ConfigurationError {
                message: "Task directory path must be absolute".to_string(),
            });
        }

        Ok(())
    }
}

impl Clone for DirectoryCleanupScheduler {
    fn clone(&self) -> Self {
        Self {
            scheduled_tasks: self.scheduled_tasks.clone(),
            cleanup_history: self.cleanup_history.clone(),
            cleanup_stats: self.cleanup_stats.clone(),
            auto_cleanup_enabled: self.auto_cleanup_enabled,
            max_concurrent_operations: self.max_concurrent_operations,
            active_operations: self.active_operations.clone(),
        }
    }
}

impl Default for DirectoryCleanupScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ================================================================================================
// Helper Functions
// ================================================================================================

/// Create a cleanup task for immediate directory deletion
pub fn create_immediate_cleanup_task(
    directory_path: std::path::PathBuf,
    test_id: String,
) -> CleanupTask {
    CleanupTask::new(
        directory_path,
        test_id,
        TempDirectoryCleanupPolicy::Immediate,
        CleanupPriority::Normal,
        CleanupTaskType::DeleteDirectory,
    )
}

/// Create a cleanup task for delayed directory deletion
pub fn create_delayed_cleanup_task(
    directory_path: std::path::PathBuf,
    test_id: String,
    delay: std::time::Duration,
) -> CleanupTask {
    CleanupTask::new(
        directory_path,
        test_id,
        TempDirectoryCleanupPolicy::Delayed(delay),
        CleanupPriority::Normal,
        CleanupTaskType::DeleteDirectory,
    )
}

/// Create a high-priority cleanup task
pub fn create_priority_cleanup_task(
    directory_path: std::path::PathBuf,
    test_id: String,
    priority: CleanupPriority,
) -> CleanupTask {
    CleanupTask::new(
        directory_path,
        test_id,
        TempDirectoryCleanupPolicy::Immediate,
        priority,
        CleanupTaskType::DeleteDirectory,
    )
}
