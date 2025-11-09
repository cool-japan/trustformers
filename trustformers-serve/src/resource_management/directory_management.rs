//! Temporary directory management for test resource allocation.
//!
//! This module provides comprehensive temporary directory management capabilities
//! including directory creation, cleanup scheduling, permission management, and
//! usage tracking for parallel test execution.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use std::{collections::HashMap, fs, path::PathBuf, sync::Arc, time::Duration};
use tracing::{debug, error, info, warn};

use super::types::{
    CleanupEvent, CleanupEventType, CleanupResult, CleanupStatistics, CleanupType,
    DirectoryPermissions, DirectoryUsageInfo, DirectoryUsageStatistics, TempDirPoolConfig,
    TempDirectoryAllocation,
};
use crate::resource_management::temp_dir_manager::types::CleanupPriority;

// Re-export types needed by other modules
pub use super::types::{CleanupTask, TempDirectoryInfo};

/// Temporary directory management system
pub struct TempDirectoryManager {
    /// Configuration
    config: Arc<RwLock<TempDirPoolConfig>>,
    /// Base directory for temporary files
    base_directory: PathBuf,
    /// Available directory pool
    available_directories: Arc<Mutex<Vec<TempDirectoryInfo>>>,
    /// Allocated directories
    allocated_directories: Arc<Mutex<HashMap<PathBuf, TempDirectoryAllocation>>>,
    /// Directory cleanup scheduler
    cleanup_scheduler: Arc<DirectoryCleanupScheduler>,
    /// Usage statistics
    usage_stats: Arc<Mutex<DirectoryUsageStatistics>>,
}

/// Directory usage tracking system
pub struct DirectoryUsageTracking {
    /// Usage history
    usage_history: Vec<(DateTime<Utc>, DirectoryUsageInfo)>,
    /// Total files created
    total_files_created: u64,
    /// Total bytes written
    total_bytes_written: u64,
    /// Access patterns
    access_patterns: HashMap<String, u32>,
}

/// Directory cleanup scheduler
pub struct DirectoryCleanupScheduler {
    /// Scheduled cleanup tasks
    scheduled_tasks: Arc<Mutex<Vec<CleanupTask>>>,
    /// Cleanup history
    cleanup_history: Arc<Mutex<Vec<CleanupEvent>>>,
    /// Cleanup statistics
    cleanup_stats: Arc<Mutex<CleanupStatistics>>,
    /// Automatic cleanup enabled
    auto_cleanup_enabled: bool,
}

impl TempDirectoryManager {
    /// Create new temporary directory manager
    pub async fn new(config: TempDirPoolConfig) -> Result<Self> {
        let base_directory = config.base_path.clone();

        // Ensure base directory exists
        if !base_directory.exists() {
            fs::create_dir_all(&base_directory).with_context(|| {
                format!("Failed to create base directory: {:?}", base_directory)
            })?;
            info!("Created base directory: {:?}", base_directory);
        }

        // Initialize available directories pool
        let available_directories = Vec::new();

        info!(
            "Initialized temporary directory manager with base directory: {:?}",
            base_directory
        );

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            base_directory,
            available_directories: Arc::new(Mutex::new(available_directories)),
            allocated_directories: Arc::new(Mutex::new(HashMap::new())),
            cleanup_scheduler: Arc::new(DirectoryCleanupScheduler::new()),
            usage_stats: Arc::new(Mutex::new(DirectoryUsageStatistics::default())),
        })
    }

    /// Allocate temporary directories for a test
    pub async fn allocate_directories(&self, count: usize, test_id: &str) -> Result<Vec<String>> {
        if count == 0 {
            return Ok(vec![]);
        }

        let mut allocated_paths = Vec::new();
        let mut allocated_directories = self.allocated_directories.lock();
        let mut usage_stats = self.usage_stats.lock();

        for i in 0..count {
            let dir_name = format!("test_{}_{}", test_id, i);
            let dir_path = self.base_directory.join(&dir_name);

            // Create directory
            fs::create_dir_all(&dir_path)
                .with_context(|| format!("Failed to create directory: {:?}", dir_path))?;

            // Set appropriate permissions
            let permissions = DirectoryPermissions::default();
            self.set_directory_permissions(&dir_path, &permissions)?;

            // Create directory info
            let directory_info = TempDirectoryInfo {
                path: dir_path.clone(),
                size_limit: 0, // No specific limit for individual directories
                current_usage: 0,
                permissions: permissions.clone(),
                created_at: Utc::now(),
                last_accessed: Utc::now(),
                status: super::types::DirectoryStatus::Allocated,
                size_bytes: 0,
                usage_info: Some(DirectoryUsageInfo {
                    path: dir_path.to_string_lossy().to_string(),
                    size_bytes: 0,
                    file_count: 0,
                    subdirectory_count: 0,
                    total_size_bytes: 0,
                    last_accessed: Utc::now(),
                    last_modified: Utc::now(),
                }),
            };

            // Create allocation record
            let allocation = TempDirectoryAllocation {
                directory: directory_info,
                test_id: test_id.to_string(),
                allocated_at: Utc::now(),
                expected_cleanup: None,
                cleanup_policy: super::types::TempDirectoryCleanupPolicy::Immediate,
                usage_tracking: super::types::DirectoryUsageTracking {
                    files_created: 0,
                    bytes_written: 0,
                    bytes_read: 0,
                    peak_usage: 0,
                    usage_timeline: Vec::new(),
                },
                cleanup_at: None,
                purpose: "test_data".to_string(),
            };

            allocated_directories.insert(dir_path.clone(), allocation);
            allocated_paths.push(dir_path.to_string_lossy().to_string());
        }

        // Update statistics
        usage_stats.total_created += count as u64;
        usage_stats.currently_allocated = allocated_directories.len();
        usage_stats.peak_usage = usage_stats.peak_usage.max(allocated_directories.len());

        info!(
            "Allocated {} temporary directories for test {}: {:?}",
            allocated_paths.len(),
            test_id,
            allocated_paths
        );

        Ok(allocated_paths)
    }

    /// Deallocate a specific directory
    pub async fn deallocate_directory(&self, dir_path: &str) -> Result<()> {
        let path = PathBuf::from(dir_path);
        let mut allocated_directories = self.allocated_directories.lock();
        let mut usage_stats = self.usage_stats.lock();

        if let Some(allocation) = allocated_directories.remove(&path) {
            // Schedule cleanup
            self.schedule_directory_cleanup(&path, CleanupPriority::Normal).await?;

            usage_stats.currently_allocated = allocated_directories.len();

            // Update average lifetime statistics
            let lifetime = allocation.allocated_at.signed_duration_since(Utc::now()).abs();
            let lifetime_std = Duration::from_secs(lifetime.num_seconds().max(0) as u64);

            if usage_stats.total_created > 0 {
                let total_lifetime = usage_stats.average_lifetime.as_secs() as f64
                    * (usage_stats.total_created - 1) as f64;
                let new_average = (total_lifetime + lifetime_std.as_secs() as f64)
                    / usage_stats.total_created as f64;
                usage_stats.average_lifetime = Duration::from_secs(new_average as u64);
            }

            info!(
                "Deallocated directory {} for test {}",
                dir_path, allocation.test_id
            );
            Ok(())
        } else {
            warn!(
                "Attempted to deallocate directory {} that was not allocated",
                dir_path
            );
            Err(anyhow::anyhow!("Directory {} was not allocated", dir_path))
        }
    }

    /// Deallocate all directories for a specific test
    pub async fn deallocate_directories_for_test(&self, test_id: &str) -> Result<()> {
        debug!("Deallocating temporary directories for test: {}", test_id);

        let mut allocated_directories = self.allocated_directories.lock();
        let mut usage_stats = self.usage_stats.lock();
        let mut deallocated_paths = Vec::new();

        // Find and collect directories to deallocate
        allocated_directories.retain(|path, allocation| {
            if allocation.test_id == test_id {
                deallocated_paths.push(path.clone());
                false // Remove from allocated_directories
            } else {
                true // Keep in allocated_directories
            }
        });

        usage_stats.currently_allocated = allocated_directories.len();

        // Schedule cleanup for all deallocated directories
        for path in &deallocated_paths {
            if let Err(e) = self.schedule_directory_cleanup(path, CleanupPriority::Normal).await {
                warn!("Failed to schedule cleanup for directory {:?}: {}", path, e);
            }
        }

        if !deallocated_paths.is_empty() {
            info!(
                "Released {} temporary directories for test {}: {:?}",
                deallocated_paths.len(),
                test_id,
                deallocated_paths
            );
        }

        Ok(())
    }

    /// Check if requested number of directories can be created
    pub async fn check_availability(&self, count: usize) -> Result<bool> {
        let config = self.config.read();
        let allocated_directories = self.allocated_directories.lock();

        let current_count = allocated_directories.len();
        let max_directories = config.max_directories;

        Ok(current_count + count <= max_directories)
    }

    /// Get current directory usage statistics
    pub async fn get_statistics(&self) -> Result<DirectoryUsageStatistics> {
        let stats = self.usage_stats.lock();
        Ok(stats.clone())
    }

    /// Set directory permissions
    fn set_directory_permissions(
        &self,
        path: &PathBuf,
        permissions: &DirectoryPermissions,
    ) -> Result<()> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let mut mode = 0o000;

            if permissions.owner_read {
                mode |= 0o400;
            }
            if permissions.owner_write {
                mode |= 0o200;
            }
            if permissions.owner_execute {
                mode |= 0o100;
            }
            // TODO: DirectoryPermissions uses group_permissions and other_permissions as u8 instead of individual bools
            // Group permissions: read=4, write=2, execute=1
            if permissions.group_permissions & 0o4 != 0 {
                mode |= 0o040;
            }
            if permissions.group_permissions & 0o2 != 0 {
                mode |= 0o020;
            }
            if permissions.group_permissions & 0o1 != 0 {
                mode |= 0o010;
            }
            // Other permissions: read=4, write=2, execute=1
            if permissions.other_permissions & 0o4 != 0 {
                mode |= 0o004;
            }
            if permissions.other_permissions & 0o2 != 0 {
                mode |= 0o002;
            }
            if permissions.other_permissions & 0o1 != 0 {
                mode |= 0o001;
            }

            let metadata = fs::metadata(path)?;
            let mut perms = metadata.permissions();
            perms.set_mode(mode);
            fs::set_permissions(path, perms)?;
        }

        #[cfg(not(unix))]
        {
            // On non-Unix systems, just make the directory writable/readable
            let metadata = fs::metadata(path)?;
            let mut perms = metadata.permissions();
            perms.set_readonly(!permissions.owner_write);
            fs::set_permissions(path, perms)?;
        }

        debug!("Set permissions for directory: {:?}", path);
        Ok(())
    }

    /// Schedule directory cleanup
    async fn schedule_directory_cleanup(
        &self,
        path: &PathBuf,
        priority: CleanupPriority,
    ) -> Result<()> {
        let task_id_value = format!("cleanup_{}", Utc::now().timestamp_millis());
        let scheduled_at_value = Utc::now();
        let test_id_value =
            self.get_test_id_for_path(path).await.unwrap_or_else(|| "unknown".to_string());

        let task = CleanupTask {
            id: task_id_value.clone(),
            directory_path: path.clone(),
            policy: super::types::TempDirectoryCleanupPolicy::Immediate,
            scheduled_time: scheduled_at_value,
            priority: priority as i32 as f32,
            test_id: test_id_value.clone(),
            retry_count: 0,
            task_id: task_id_value.clone(),
            target_path: path.clone(),
            cleanup_type: CleanupType::DeleteDirectory,
            scheduled_at: scheduled_at_value,
            task_type: "directory_cleanup".to_string(),
            details: std::collections::HashMap::new(),
        };

        self.cleanup_scheduler.schedule_task(task).await
    }

    /// Get test ID associated with a path
    async fn get_test_id_for_path(&self, path: &PathBuf) -> Option<String> {
        let allocated_directories = self.allocated_directories.lock();
        allocated_directories.get(path).map(|allocation| allocation.test_id.clone())
    }

    /// Update directory usage information
    pub async fn update_directory_usage(&self, path: &PathBuf) -> Result<()> {
        let mut allocated_directories = self.allocated_directories.lock();

        if let Some(allocation) = allocated_directories.get_mut(path) {
            // Calculate directory size and file count
            let (size_bytes, file_count, subdirectory_count) =
                self.calculate_directory_stats(path)?;

            allocation.directory.usage_info = Some(DirectoryUsageInfo {
                path: path.to_string_lossy().to_string(),
                size_bytes,
                file_count,
                subdirectory_count,
                total_size_bytes: size_bytes,
                last_accessed: Utc::now(),
                last_modified: Utc::now(),
            });

            allocation.directory.size_bytes = size_bytes;

            debug!(
                "Updated usage info for directory {:?}: {} bytes, {} files",
                path, size_bytes, file_count
            );
        }

        Ok(())
    }

    /// Calculate directory statistics
    fn calculate_directory_stats(&self, path: &PathBuf) -> Result<(u64, usize, usize)> {
        let mut total_size = 0u64;
        let mut file_count = 0usize;
        let mut subdirectory_count = 0usize;

        if path.exists() {
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                let metadata = entry.metadata()?;

                if metadata.is_file() {
                    total_size += metadata.len();
                    file_count += 1;
                } else if metadata.is_dir() {
                    subdirectory_count += 1;
                    // Recursively calculate subdirectory stats
                    let (sub_size, sub_files, sub_dirs) =
                        self.calculate_directory_stats(&entry.path())?;
                    total_size += sub_size;
                    file_count += sub_files;
                    subdirectory_count += sub_dirs;
                }
            }
        }

        Ok((total_size, file_count, subdirectory_count))
    }

    /// Get detailed directory allocation information
    pub async fn get_directory_allocations(&self) -> HashMap<PathBuf, TempDirectoryAllocation> {
        let allocated_directories = self.allocated_directories.lock();
        allocated_directories.clone()
    }

    /// Get allocated directory count
    pub async fn get_allocated_directory_count(&self) -> usize {
        let allocated_directories = self.allocated_directories.lock();
        allocated_directories.len()
    }

    /// Check if a specific directory is allocated
    pub async fn is_directory_allocated(&self, path: &PathBuf) -> bool {
        let allocated_directories = self.allocated_directories.lock();
        allocated_directories.contains_key(path)
    }

    /// Get allocation details for a specific directory
    pub async fn get_directory_allocation(
        &self,
        path: &PathBuf,
    ) -> Option<TempDirectoryAllocation> {
        let allocated_directories = self.allocated_directories.lock();
        allocated_directories.get(path).cloned()
    }

    /// Update directory manager configuration
    pub async fn update_config(&self, new_config: TempDirPoolConfig) -> Result<()> {
        let mut config = self.config.write();
        *config = new_config;

        info!("Updated temporary directory manager configuration");
        Ok(())
    }

    /// Get directory utilization percentage
    pub async fn get_utilization(&self) -> f32 {
        let config = self.config.read();
        let allocated_count = self.get_allocated_directory_count().await;
        let max_directories = config.max_directories;

        if max_directories == 0 {
            0.0
        } else {
            allocated_count as f32 / max_directories as f32
        }
    }

    /// Clean up orphaned directories
    pub async fn cleanup_orphaned_directories(&self) -> Result<usize> {
        let allocated_directories = self.allocated_directories.lock();
        let mut cleaned_count = 0;

        // Check for directories in base directory that are not tracked
        if self.base_directory.exists() {
            for entry in fs::read_dir(&self.base_directory)? {
                let entry = entry?;
                let path = entry.path();

                if path.is_dir() && !allocated_directories.contains_key(&path) {
                    // This is an orphaned directory
                    if let Err(e) = fs::remove_dir_all(&path) {
                        warn!("Failed to remove orphaned directory {:?}: {}", path, e);
                    } else {
                        cleaned_count += 1;
                        info!("Removed orphaned directory: {:?}", path);
                    }
                }
            }
        }

        if cleaned_count > 0 {
            info!("Cleaned up {} orphaned directories", cleaned_count);
        }

        Ok(cleaned_count)
    }

    /// Generate directory allocation report
    pub async fn generate_allocation_report(&self) -> String {
        let stats = self.get_statistics().await.unwrap_or_default();
        let allocated_count = self.get_allocated_directory_count().await;
        let utilization = self.get_utilization().await;

        format!(
            "Directory Allocation Report:\n\
             - Allocated directories: {}\n\
             - Total created: {}\n\
             - Peak usage: {}\n\
             - Current utilization: {:.1}%\n\
             - Average lifetime: {}s\n\
             - Total bytes used: {}",
            allocated_count,
            stats.total_created,
            stats.peak_usage,
            utilization * 100.0,
            stats.average_lifetime.as_secs(),
            stats.total_bytes_used
        )
    }

    /// Execute pending cleanup tasks
    pub async fn execute_pending_cleanups(&self) -> Result<usize> {
        self.cleanup_scheduler.execute_pending_tasks().await
    }

    /// Get cleanup statistics
    pub async fn get_cleanup_statistics(&self) -> CleanupStatistics {
        self.cleanup_scheduler.get_statistics().await
    }
}

impl DirectoryCleanupScheduler {
    /// Create new directory cleanup scheduler
    pub fn new() -> Self {
        Self {
            scheduled_tasks: Arc::new(Mutex::new(Vec::new())),
            cleanup_history: Arc::new(Mutex::new(Vec::new())),
            cleanup_stats: Arc::new(Mutex::new(CleanupStatistics::default())),
            auto_cleanup_enabled: true,
        }
    }

    /// Schedule a cleanup task
    pub async fn schedule_task(&self, task: CleanupTask) -> Result<()> {
        let mut scheduled_tasks = self.scheduled_tasks.lock();
        scheduled_tasks.push(task.clone());

        // Sort by priority and scheduled time
        scheduled_tasks.sort_by(|a, b| {
            a.priority
                .partial_cmp(&b.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.scheduled_at.cmp(&b.scheduled_at))
        });

        debug!("Scheduled cleanup task: {:?}", task.task_id);
        Ok(())
    }

    /// Execute pending cleanup tasks
    pub async fn execute_pending_tasks(&self) -> Result<usize> {
        let mut scheduled_tasks = self.scheduled_tasks.lock();
        let mut cleanup_history = self.cleanup_history.lock();
        let mut cleanup_stats = self.cleanup_stats.lock();

        let now = Utc::now();
        let mut executed_count = 0;

        // Find tasks ready for execution
        let mut tasks_to_execute = Vec::new();
        scheduled_tasks.retain(|task| {
            if task.scheduled_at <= now {
                tasks_to_execute.push(task.clone());
                false
            } else {
                true
            }
        });

        // Execute tasks
        for task in tasks_to_execute {
            let execution_start = Utc::now();
            let result = self.execute_cleanup_task(&task).await;

            let cleanup_event = match result {
                Ok((bytes_cleaned, files_cleaned)) => {
                    cleanup_stats.successful_cleanups += 1;
                    cleanup_stats.total_bytes_cleaned += bytes_cleaned;
                    cleanup_stats.total_files_cleaned += files_cleaned as u64;
                    executed_count += 1;

                    CleanupEvent {
                        timestamp: execution_start,
                        event_type: CleanupEventType::Completed,
                        directory_path: task.target_path.clone(),
                        test_id: task.test_id.clone(),
                        duration: Duration::from_secs(0), // Will be updated below
                        result: CleanupResult::Success {
                            files_removed: files_cleaned,
                            bytes_freed: bytes_cleaned,
                        },
                        details: HashMap::new(),
                        event_id: format!("cleanup_event_{}", Utc::now().timestamp_millis()),
                        task_id: task.task_id.clone(),
                        target_path: task.target_path.clone(),
                        bytes_cleaned,
                        files_cleaned,
                        success: true,
                        error: None,
                    }
                },
                Err(e) => {
                    cleanup_stats.failed_cleanups += 1;
                    error!("Failed to execute cleanup task {}: {}", task.task_id, e);

                    CleanupEvent {
                        timestamp: execution_start,
                        event_type: CleanupEventType::Failed,
                        directory_path: task.target_path.clone(),
                        test_id: task.test_id.clone(),
                        duration: Duration::from_secs(0), // Will be updated below
                        result: CleanupResult::Failed {
                            error: e.to_string(),
                            files_attempted: 0,
                        },
                        details: HashMap::new(),
                        event_id: format!("cleanup_event_{}", Utc::now().timestamp_millis()),
                        task_id: task.task_id.clone(),
                        target_path: task.target_path.clone(),
                        bytes_cleaned: 0,
                        files_cleaned: 0,
                        success: false,
                        error: Some(e.to_string()),
                    }
                },
            };

            cleanup_history.push(cleanup_event);
            cleanup_stats.total_tasks += 1;

            // Update average duration
            let execution_duration = Utc::now().signed_duration_since(execution_start);
            let duration_std = Duration::from_secs(execution_duration.num_seconds().max(0) as u64);

            if cleanup_stats.total_tasks > 0 {
                let total_duration = cleanup_stats.average_duration.as_secs() as f64
                    * (cleanup_stats.total_tasks - 1) as f64;
                let new_average = (total_duration + duration_std.as_secs() as f64)
                    / cleanup_stats.total_tasks as f64;
                cleanup_stats.average_duration = Duration::from_secs(new_average as u64);
            }
        }

        // Limit history size
        while cleanup_history.len() > 1000 {
            cleanup_history.remove(0);
        }

        if executed_count > 0 {
            info!("Executed {} cleanup tasks", executed_count);
        }

        Ok(executed_count)
    }

    /// Execute a single cleanup task
    async fn execute_cleanup_task(&self, task: &CleanupTask) -> Result<(u64, usize)> {
        match &task.cleanup_type {
            CleanupType::DeleteDirectory => {
                let (bytes_cleaned, files_cleaned) =
                    self.calculate_directory_size(&task.target_path)?;

                if task.target_path.exists() {
                    fs::remove_dir_all(&task.target_path).with_context(|| {
                        format!("Failed to remove directory: {:?}", task.target_path)
                    })?;

                    info!(
                        "Deleted directory: {:?} ({} bytes, {} files)",
                        task.target_path, bytes_cleaned, files_cleaned
                    );
                }

                Ok((bytes_cleaned, files_cleaned))
            },
            CleanupType::DeleteOldFiles(age) => {
                let (bytes_cleaned, files_cleaned) =
                    self.delete_old_files(&task.target_path, *age)?;
                Ok((bytes_cleaned, files_cleaned))
            },
            CleanupType::EmptyDirectory => {
                let (bytes_cleaned, files_cleaned) = self.empty_directory(&task.target_path)?;
                Ok((bytes_cleaned, files_cleaned))
            },
            CleanupType::CompressFiles => {
                // Placeholder for file compression
                Ok((0, 0))
            },
            CleanupType::Custom(_) => {
                // Placeholder for custom cleanup
                Ok((0, 0))
            },
        }
    }

    /// Calculate directory size
    fn calculate_directory_size(&self, path: &PathBuf) -> Result<(u64, usize)> {
        let mut total_size = 0u64;
        let mut file_count = 0usize;

        if path.exists() && path.is_dir() {
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                let metadata = entry.metadata()?;

                if metadata.is_file() {
                    total_size += metadata.len();
                    file_count += 1;
                } else if metadata.is_dir() {
                    let (sub_size, sub_count) = self.calculate_directory_size(&entry.path())?;
                    total_size += sub_size;
                    file_count += sub_count;
                }
            }
        }

        Ok((total_size, file_count))
    }

    /// Delete old files from directory
    fn delete_old_files(&self, path: &PathBuf, max_age: Duration) -> Result<(u64, usize)> {
        let mut bytes_cleaned = 0u64;
        let mut files_cleaned = 0usize;
        let cutoff_time = Utc::now() - chrono::Duration::from_std(max_age)?;

        if path.exists() && path.is_dir() {
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                let metadata = entry.metadata()?;

                if metadata.is_file() {
                    if let Ok(modified) = metadata.modified() {
                        let modified_chrono = DateTime::<Utc>::from(modified);
                        if modified_chrono < cutoff_time {
                            bytes_cleaned += metadata.len();
                            files_cleaned += 1;
                            fs::remove_file(entry.path())?;
                        }
                    }
                }
            }
        }

        Ok((bytes_cleaned, files_cleaned))
    }

    /// Empty directory contents
    fn empty_directory(&self, path: &PathBuf) -> Result<(u64, usize)> {
        let mut bytes_cleaned = 0u64;
        let mut files_cleaned = 0usize;

        if path.exists() && path.is_dir() {
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                let metadata = entry.metadata()?;

                if metadata.is_file() {
                    bytes_cleaned += metadata.len();
                    files_cleaned += 1;
                    fs::remove_file(entry.path())?;
                } else if metadata.is_dir() {
                    let (sub_bytes, sub_files) = self.calculate_directory_size(&entry.path())?;
                    bytes_cleaned += sub_bytes;
                    files_cleaned += sub_files;
                    fs::remove_dir_all(entry.path())?;
                }
            }
        }

        Ok((bytes_cleaned, files_cleaned))
    }

    /// Get cleanup statistics
    pub async fn get_statistics(&self) -> CleanupStatistics {
        let cleanup_stats = self.cleanup_stats.lock();
        cleanup_stats.clone()
    }

    /// Get pending task count
    pub fn get_pending_task_count(&self) -> usize {
        let scheduled_tasks = self.scheduled_tasks.lock();
        scheduled_tasks.len()
    }

    /// Cancel all tasks for a specific test
    pub async fn cancel_tasks_for_test(&self, test_id: &str) -> usize {
        let mut scheduled_tasks = self.scheduled_tasks.lock();
        let initial_count = scheduled_tasks.len();

        scheduled_tasks.retain(|task| &task.test_id != test_id);

        let cancelled_count = initial_count - scheduled_tasks.len();
        if cancelled_count > 0 {
            info!(
                "Cancelled {} cleanup tasks for test {}",
                cancelled_count, test_id
            );
        }

        cancelled_count
    }
}

impl Default for DirectoryCleanupScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl DirectoryUsageTracking {
    /// Create new directory usage tracking
    pub fn new() -> Self {
        Self {
            usage_history: Vec::new(),
            total_files_created: 0,
            total_bytes_written: 0,
            access_patterns: HashMap::new(),
        }
    }

    /// Record file creation
    pub fn record_file_creation(&mut self, size_bytes: u64) {
        self.total_files_created += 1;
        self.total_bytes_written += size_bytes;
    }

    /// Record access pattern
    pub fn record_access_pattern(&mut self, pattern: &str) {
        *self.access_patterns.entry(pattern.to_string()).or_insert(0) += 1;
    }

    /// Get usage statistics
    pub fn get_usage_stats(&self) -> (u64, u64, &HashMap<String, u32>) {
        (
            self.total_files_created,
            self.total_bytes_written,
            &self.access_patterns,
        )
    }
}

impl Default for DirectoryUsageTracking {
    fn default() -> Self {
        Self::new()
    }
}
