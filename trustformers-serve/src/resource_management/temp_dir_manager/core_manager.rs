//! Core Temporary Directory Manager Implementation
//!
//! This module contains the main TempDirectoryManager struct and its implementation,
//! providing the primary interface for temporary directory management operations.

use super::types::*;
use super::{
    cleanup_scheduler::DirectoryCleanupScheduler, conflict_resolver::DirectoryConflictResolver,
    quota_manager::DirectoryQuotaManager, utils::*,
};

// Explicit imports to disambiguate types from parent resource_management module
// CleanupTask and CleanupStatistics exist in both super::types and crate::resource_management::types
use super::types::{CleanupStatistics, CleanupTask};

use chrono::Utc;
use parking_lot::{Mutex, RwLock};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::{task::JoinHandle, time::interval};
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

use crate::resource_management::types::*;

// ================================================================================================
// Core Manager Implementation
// ================================================================================================

/// Comprehensive temporary directory manager
///
/// This is the main interface for managing temporary directories in the test parallelization
/// system. It provides thread-safe operations for allocation, deallocation, cleanup, and
/// monitoring of temporary directories.
///
/// ## Features
///
/// - **Thread-safe operations** using Arc, Mutex, and RwLock
/// - **Automatic cleanup** with configurable policies
/// - **Quota enforcement** to prevent disk space exhaustion
/// - **Conflict resolution** for concurrent access
/// - **Comprehensive logging** and error handling
/// - **Statistics tracking** for performance monitoring
#[derive(Debug)]
pub struct TempDirectoryManager {
    /// Manager instance information
    instance_info: Arc<RwLock<ManagerInstanceInfo>>,

    /// Configuration settings
    config: Arc<RwLock<TempDirectoryManagerConfig>>,

    /// Base directory for all temporary directories
    base_directory: PathBuf,

    /// Pool of available directories
    available_directories: Arc<Mutex<Vec<PathBuf>>>,

    /// Currently allocated directories
    allocated_directories: Arc<Mutex<HashMap<PathBuf, TempDirectoryAllocation>>>,

    /// Directory cleanup scheduler
    cleanup_scheduler: Arc<DirectoryCleanupScheduler>,

    /// Quota manager for disk space monitoring
    quota_manager: Arc<DirectoryQuotaManager>,

    /// Conflict resolver for handling access conflicts
    conflict_resolver: Arc<DirectoryConflictResolver>,

    /// Usage statistics tracker
    usage_stats: Arc<Mutex<DirectoryUsageStatistics>>,

    /// Background cleanup task handle
    cleanup_task_handle: Arc<Mutex<Option<JoinHandle<()>>>>,

    /// Shutdown signal
    shutdown_signal: Arc<AtomicBool>,
}

impl TempDirectoryManager {
    /// Create a new temporary directory manager
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the directory manager
    ///
    /// # Returns
    ///
    /// Returns a Result containing the configured TempDirectoryManager or an error
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The base directory cannot be created
    /// - Permission issues prevent directory access
    /// - Configuration validation fails
    #[instrument(skip(config))]
    pub async fn new(config: TempDirPoolConfig) -> TempDirResult<Self> {
        let manager_config = TempDirectoryManagerConfig {
            pool_config: config.clone(),
            ..Default::default()
        };
        Self::with_full_config(manager_config).await
    }

    /// Create a new temporary directory manager with full configuration
    #[instrument(skip(config))]
    pub async fn with_full_config(config: TempDirectoryManagerConfig) -> TempDirResult<Self> {
        let base_directory = config.pool_config.base_path.clone();
        let instance_id = format!("temp_dir_manager_{}", Uuid::new_v4());

        info!(
            instance_id = %instance_id,
            base_path = %base_directory.display(),
            "Initializing temporary directory manager"
        );

        // Validate and create base directory
        Self::ensure_base_directory(&base_directory).await?;

        // Create instance info
        let instance_info = ManagerInstanceInfo {
            instance_id: instance_id.clone(),
            created_at: Utc::now(),
            config: config.clone(),
            status: ManagerStatus::Initializing,
        };

        // Initialize components
        let cleanup_scheduler = Arc::new(DirectoryCleanupScheduler::new());
        let quota_manager = Arc::new(DirectoryQuotaManager::new(&config.pool_config));
        let conflict_resolver = Arc::new(DirectoryConflictResolver::new());

        // Initialize directory pool
        let available_directories = Arc::new(Mutex::new(Vec::new()));
        let allocated_directories = Arc::new(Mutex::new(HashMap::new()));

        let manager = Self {
            instance_info: Arc::new(RwLock::new(instance_info)),
            config: Arc::new(RwLock::new(config.clone())),
            base_directory: base_directory.clone(),
            available_directories,
            allocated_directories,
            cleanup_scheduler: cleanup_scheduler.clone(),
            quota_manager: quota_manager.clone(),
            conflict_resolver,
            usage_stats: Arc::new(Mutex::new(DirectoryUsageStatistics::default())),
            cleanup_task_handle: Arc::new(Mutex::new(None)),
            shutdown_signal: Arc::new(AtomicBool::new(false)),
        };

        // Start background cleanup task if auto-cleanup is enabled
        if config.pool_config.enable_auto_cleanup {
            manager.start_background_cleanup().await?;
        }

        // Perform initial cleanup of orphaned directories
        manager.cleanup_orphaned_directories().await?;

        // Update status to active
        manager.instance_info.write().status = ManagerStatus::Active;

        info!(
            instance_id = %instance_id,
            "Temporary directory manager initialized successfully"
        );

        Ok(manager)
    }

    /// Get the manager instance information
    pub fn get_instance_info(&self) -> ManagerInstanceInfo {
        let instance_info = self.instance_info.read();
        instance_info.clone()
    }

    /// Allocate temporary directories for a test
    ///
    /// # Arguments
    ///
    /// * `count` - Number of directories to allocate
    /// * `test_id` - Unique identifier for the test
    ///
    /// # Returns
    ///
    /// Returns a vector of directory paths that have been allocated
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - Insufficient disk space is available
    /// - Maximum directory count would be exceeded
    /// - Directory creation fails
    /// - Permission issues prevent access
    #[instrument(skip(self), fields(instance_id = %self.get_instance_id()))]
    pub async fn allocate_directories(
        &self,
        count: usize,
        test_id: &str,
    ) -> TempDirResult<Vec<String>> {
        if count == 0 {
            return Ok(vec![]);
        }

        info!(
            test_id = %test_id,
            count = %count,
            "Allocating temporary directories"
        );

        // Check availability and quotas
        self.check_allocation_feasibility(count, test_id).await?;

        let mut allocated_paths = Vec::new();
        let mut allocated_directories = self.allocated_directories.lock();
        let mut usage_stats = self.usage_stats.lock();

        for i in 0..count {
            // Generate unique directory name
            let dir_name = format!(
                "test_{}_{}_dir_{}",
                test_id,
                Utc::now().timestamp_millis(),
                i
            );
            let dir_path = self.base_directory.join(&dir_name);

            // Create directory with proper permissions
            let directory_info = self.create_directory(&dir_path, test_id).await?;

            // Create allocation record
            let allocation = TempDirectoryAllocation {
                directory: directory_info,
                test_id: test_id.to_string(),
                allocated_at: Utc::now(),
                expected_cleanup: None,
                cleanup_policy: self.config.read().pool_config.default_cleanup_policy.clone(),
                usage_tracking: DirectoryUsageTracking {
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

            // Update quota usage
            self.quota_manager
                .reserve_space(
                    &dir_path,
                    self.config.read().pool_config.max_directory_size_bytes,
                )
                .await?;
        }

        // Update statistics
        usage_stats.total_created += count as u64;
        usage_stats.currently_allocated = allocated_directories.len();
        usage_stats.peak_usage = usage_stats.peak_usage.max(allocated_directories.len());

        info!(
            test_id = %test_id,
            count = %count,
            allocated_paths = ?allocated_paths,
            "Successfully allocated temporary directories"
        );

        Ok(allocated_paths)
    }

    /// Deallocate directories for a specific test
    ///
    /// # Arguments
    ///
    /// * `test_id` - Test ID to deallocate directories for
    ///
    /// # Returns
    ///
    /// Returns the number of directories that were deallocated
    ///
    /// # Errors
    ///
    /// This function will return an error if cleanup scheduling fails
    #[instrument(skip(self), fields(instance_id = %self.get_instance_id()))]
    pub async fn deallocate_directories_for_test(&self, test_id: &str) -> TempDirResult<usize> {
        info!(test_id = %test_id, "Deallocating directories for test");

        let (directories_to_cleanup, deallocated_count) = {
            let mut allocated_directories = self.allocated_directories.lock();
            let mut collected = Vec::new();
            let mut count = 0;

            allocated_directories.retain(|path, allocation| {
                if allocation.test_id == test_id {
                    collected.push((path.clone(), allocation.clone()));
                    count += 1;
                    false
                } else {
                    true
                }
            });

            (collected, count)
        };

        // Schedule cleanup tasks
        for (path, allocation) in directories_to_cleanup {
            {
                let mut usage_stats = self.usage_stats.lock();
                self.update_average_lifetime(&mut usage_stats, &allocation);
            }

            // Release quota
            self.quota_manager.release_space(&path).await?;

            // Schedule cleanup
            let cleanup_task = CleanupTask::new(
                path.clone(),
                test_id.to_string(),
                allocation.cleanup_policy.clone(),
                CleanupPriority::Normal,
                CleanupTaskType::DeleteDirectory,
            );

            self.cleanup_scheduler.schedule_task(cleanup_task).await?;
        }

        // Update statistics
        {
            let mut usage_stats = self.usage_stats.lock();
            usage_stats.currently_allocated = self.allocated_directories.lock().len();
        }

        info!(
            test_id = %test_id,
            deallocated_count = %deallocated_count,
            "Successfully deallocated directories for test"
        );

        Ok(deallocated_count)
    }

    /// Get current usage statistics
    ///
    /// # Returns
    ///
    /// Returns the current directory usage statistics
    pub async fn get_statistics(&self) -> TempDirResult<DirectoryUsageStatistics> {
        let stats = self.usage_stats.lock().clone();
        Ok(stats)
    }

    /// Get the number of currently allocated directories
    pub async fn get_allocated_directory_count(&self) -> usize {
        self.allocated_directories.lock().len()
    }

    /// Get current utilization as a percentage (0.0 to 1.0)
    pub async fn get_utilization(&self) -> f32 {
        let allocated_count = self.get_allocated_directory_count().await;
        let max_directories = self.config.read().pool_config.max_directories;

        if max_directories == 0 {
            0.0
        } else {
            allocated_count as f32 / max_directories as f32
        }
    }

    /// Update configuration
    #[instrument(skip(self, new_config), fields(instance_id = %self.get_instance_id()))]
    pub async fn update_config(&self, new_config: TempDirectoryManagerConfig) -> TempDirResult<()> {
        info!("Updating temporary directory manager configuration");

        // Validate new configuration
        self.validate_configuration(&new_config)?;

        // Update configuration
        *self.config.write() = new_config.clone();
        self.instance_info.write().config = new_config;

        info!("Configuration updated successfully");
        Ok(())
    }

    /// Clean up orphaned directories
    ///
    /// This method scans the base directory for directories that are not tracked
    /// as allocated and removes them.
    ///
    /// # Returns
    ///
    /// Returns the number of directories cleaned up
    ///
    /// # Errors
    ///
    /// This function will return an error if directory scanning or removal fails
    #[instrument(skip(self), fields(instance_id = %self.get_instance_id()))]
    pub async fn cleanup_orphaned_directories(&self) -> TempDirResult<usize> {
        info!("Starting orphaned directory cleanup");

        let allocated_directories = self.allocated_directories.lock();
        let mut cleaned_count = 0;

        if self.base_directory.exists() {
            let entries =
                fs::read_dir(&self.base_directory).map_err(|e| TempDirError::IoError {
                    path: self.base_directory.display().to_string(),
                    source: e,
                })?;

            for entry in entries {
                let entry = entry.map_err(|e| TempDirError::IoError {
                    path: self.base_directory.display().to_string(),
                    source: e,
                })?;
                let path = entry.path();

                if path.is_dir() && !allocated_directories.contains_key(&path) {
                    // This is an orphaned directory
                    match fs::remove_dir_all(&path) {
                        Ok(()) => {
                            cleaned_count += 1;
                            info!(path = %path.display(), "Removed orphaned directory");
                        },
                        Err(e) => {
                            warn!(
                                path = %path.display(),
                                error = %e,
                                "Failed to remove orphaned directory"
                            );
                        },
                    }
                }
            }
        }

        if cleaned_count > 0 {
            info!(cleaned_count = %cleaned_count, "Orphaned directory cleanup completed");
        }

        Ok(cleaned_count)
    }

    /// Generate a comprehensive allocation report
    ///
    /// # Returns
    ///
    /// Returns a detailed string report of current allocations and statistics
    pub async fn generate_allocation_report(&self) -> String {
        let stats = self.get_statistics().await.unwrap_or_default();
        let allocated_count = self.get_allocated_directory_count().await;
        let utilization = self.get_utilization().await;
        let available_space = self.quota_manager.get_available_space().await.unwrap_or(0);
        let instance_info = self.get_instance_info();

        format!(
            "=== Temporary Directory Manager Report ===\n\
             Instance ID: {}\n\
             Status: {:?}\n\
             Created At: {}\n\
             Base Directory: {}\n\
             \n\
             == Current State ==\n\
             Allocated directories: {}\n\
             Current utilization: {:.1}%\n\
             Available disk space: {} bytes\n\
             \n\
             == Historical Statistics ==\n\
             Total created: {}\n\
             Peak usage: {}\n\
             Average lifetime: {}s\n\
             Total bytes used: {}\n\
             \n\
             == Cleanup Statistics ==\n\
             {}",
            instance_info.instance_id,
            instance_info.status,
            instance_info.created_at.to_rfc3339(),
            self.base_directory.display(),
            allocated_count,
            utilization * 100.0,
            available_space,
            stats.total_created,
            stats.peak_usage,
            stats.average_lifetime.as_secs(),
            stats.total_bytes_used,
            self.cleanup_scheduler.generate_report().await
        )
    }

    /// Execute any pending cleanup tasks
    ///
    /// # Returns
    ///
    /// Returns the number of cleanup tasks executed
    ///
    /// # Errors
    ///
    /// This function will return an error if cleanup execution fails
    pub async fn execute_pending_cleanups(&self) -> TempDirResult<usize> {
        self.cleanup_scheduler.execute_pending_tasks().await
    }

    /// Get cleanup statistics
    ///
    /// # Returns
    ///
    /// Returns the current cleanup statistics
    pub async fn get_cleanup_statistics(&self) -> CleanupStatistics {
        self.cleanup_scheduler.get_statistics().await
    }

    /// Shutdown the manager and clean up resources
    ///
    /// This method should be called when the manager is no longer needed.
    /// It will stop background tasks and perform final cleanup.
    #[instrument(skip(self), fields(instance_id = %self.get_instance_id()))]
    pub async fn shutdown(&self) -> TempDirResult<()> {
        info!("Shutting down temporary directory manager");

        // Update status
        self.instance_info.write().status = ManagerStatus::ShuttingDown;

        // Signal shutdown
        self.shutdown_signal.store(true, Ordering::SeqCst);

        // Stop background cleanup task
        if let Some(handle) = self.cleanup_task_handle.lock().take() {
            handle.abort();
            debug!("Background cleanup task stopped");
        }

        // Execute final cleanup
        let executed = self.execute_pending_cleanups().await?;
        if executed > 0 {
            info!(executed_tasks = %executed, "Executed final cleanup tasks");
        }

        // Update status
        self.instance_info.write().status = ManagerStatus::Shutdown;

        info!("Temporary directory manager shutdown completed");
        Ok(())
    }

    // ============================================================================================
    // Private Implementation Methods
    // ============================================================================================

    /// Get the manager instance ID
    fn get_instance_id(&self) -> String {
        self.instance_info.read().instance_id.clone()
    }

    /// Ensure the base directory exists and is accessible
    async fn ensure_base_directory(base_dir: &Path) -> TempDirResult<()> {
        if !base_dir.exists() {
            fs::create_dir_all(base_dir).map_err(|e| TempDirError::IoError {
                path: base_dir.display().to_string(),
                source: e,
            })?;
            info!(path = %base_dir.display(), "Created base directory");
        }

        // Verify write permissions
        let test_file = base_dir.join(".write_test");
        if let Err(e) = fs::write(&test_file, b"test") {
            return Err(TempDirError::PermissionDenied {
                path: base_dir.display().to_string(),
                message: format!("Cannot write to directory: {}", e),
            });
        }
        let _ = fs::remove_file(&test_file);

        Ok(())
    }

    /// Check if allocation is feasible given current constraints
    async fn check_allocation_feasibility(&self, count: usize, test_id: &str) -> TempDirResult<()> {
        let config = self.config.read();
        let allocated_directories = self.allocated_directories.lock();

        // Check directory count limit
        let current_count = allocated_directories.len();
        if current_count + count > config.pool_config.max_directories {
            return Err(TempDirError::AllocationFailed {
                message: format!(
                    "Would exceed maximum directory count: {} + {} > {}",
                    current_count, count, config.pool_config.max_directories
                ),
            });
        }

        // Check disk space
        let required_space = (count as u64) * config.pool_config.max_directory_size_bytes;
        let available_space = self.quota_manager.get_available_space().await?;

        if required_space > available_space {
            return Err(TempDirError::InsufficientDiskSpace {
                requested: required_space,
                available: available_space,
            });
        }

        // Check for conflicts
        if let Some(conflict) = self.conflict_resolver.check_conflicts(test_id, count).await {
            return Err(TempDirError::AccessConflict {
                path: format!("test_{}", test_id),
                message: format!("Detected conflict: {:?}", conflict.conflict_type),
            });
        }

        Ok(())
    }

    /// Create a directory with proper configuration
    async fn create_directory(
        &self,
        path: &Path,
        test_id: &str,
    ) -> TempDirResult<TempDirectoryInfo> {
        // Create the directory
        fs::create_dir_all(path).map_err(|e| TempDirError::IoError {
            path: path.display().to_string(),
            source: e,
        })?;

        // Set proper permissions
        let permissions = DirectoryPermissions::default();
        set_directory_permissions(path, &permissions)?;

        // Create directory info
        let directory_info = TempDirectoryInfo {
            path: path.to_path_buf(),
            size_limit: self.config.read().pool_config.max_directory_size_bytes,
            current_usage: 0,
            permissions,
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            status: DirectoryStatus::Allocated,
            size_bytes: 0,
            usage_info: None,
        };

        debug!(
            test_id = %test_id,
            path = %path.display(),
            "Created directory"
        );

        Ok(directory_info)
    }

    /// Update average lifetime statistics
    fn update_average_lifetime(
        &self,
        stats: &mut DirectoryUsageStatistics,
        allocation: &TempDirectoryAllocation,
    ) {
        let lifetime = Utc::now().signed_duration_since(allocation.allocated_at);
        let lifetime_std = Duration::from_secs(lifetime.num_seconds().max(0) as u64);

        if stats.total_created > 0 {
            let total_lifetime =
                stats.average_lifetime.as_secs() as f64 * (stats.total_created - 1) as f64;
            let new_average =
                (total_lifetime + lifetime_std.as_secs() as f64) / stats.total_created as f64;
            stats.average_lifetime = Duration::from_secs(new_average as u64);
        }
    }

    /// Validate configuration settings
    fn validate_configuration(&self, config: &TempDirectoryManagerConfig) -> TempDirResult<()> {
        if config.pool_config.max_directories == 0 {
            return Err(TempDirError::ConfigurationError {
                message: "Maximum directories must be greater than 0".to_string(),
            });
        }

        if config.pool_config.max_directory_size_bytes == 0 {
            return Err(TempDirError::ConfigurationError {
                message: "Maximum directory size must be greater than 0".to_string(),
            });
        }

        if !config.pool_config.base_path.is_absolute() {
            return Err(TempDirError::ConfigurationError {
                message: "Base path must be absolute".to_string(),
            });
        }

        Ok(())
    }

    /// Start background cleanup task
    async fn start_background_cleanup(&self) -> TempDirResult<()> {
        let cleanup_scheduler = self.cleanup_scheduler.clone();
        let shutdown_signal = self.shutdown_signal.clone();
        let cleanup_interval = self.config.read().pool_config.cleanup_interval_secs;

        let handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(cleanup_interval));

            while !shutdown_signal.load(Ordering::SeqCst) {
                interval.tick().await;

                if let Err(e) = cleanup_scheduler.execute_pending_tasks().await {
                    error!(error = %e, "Background cleanup task failed");
                }
            }

            debug!("Background cleanup task terminated");
        });

        *self.cleanup_task_handle.lock() = Some(handle);
        debug!("Background cleanup task started");

        Ok(())
    }
}

// Implement Drop to ensure proper cleanup
impl Drop for TempDirectoryManager {
    fn drop(&mut self) {
        if !self.shutdown_signal.load(Ordering::SeqCst) {
            // Manager is being dropped without proper shutdown
            warn!(
                instance_id = %self.get_instance_id(),
                "TempDirectoryManager dropped without proper shutdown"
            );

            // Stop background task
            if let Some(handle) = self.cleanup_task_handle.lock().take() {
                handle.abort();
            }

            // Update status
            self.instance_info.write().status = ManagerStatus::Error {
                message: "Manager dropped without proper shutdown".to_string(),
            };
        }
    }
}
