//! Modular Temporary Directory Manager
//!
//! This module provides a comprehensive temporary directory management system
//! designed for test parallelization. It has been refactored into a modular
//! architecture with specialized components for better maintainability and performance.
//!
//! # Architecture
//!
//! The system is organized into the following specialized modules:
//!
//! - [`types`] - Core type definitions, errors, and configuration structures
//! - [`core_manager`] - Main TempDirectoryManager implementation
//! - [`cleanup_scheduler`] - DirectoryCleanupScheduler for cleanup operations
//! - [`quota_manager`] - DirectoryQuotaManager for disk space management
//! - [`conflict_resolver`] - DirectoryConflictResolver for conflict handling
//! - [`utils`] - Utility functions and helper operations
//!
//! # Key Features
//!
//! - **Thread-safe operations** with proper synchronization
//! - **Automatic cleanup** with configurable policies
//! - **Quota enforcement** to prevent disk space exhaustion
//! - **Conflict resolution** for concurrent access
//! - **Comprehensive logging** and error handling
//! - **Statistics tracking** for performance monitoring
//!
//! # Example Usage
//!
//! ```rust
//! use trustformers_serve::resource_management::temp_dir_manager::TempDirectoryManager;
//! use trustformers_serve::resource_management::types::TempDirPoolConfig;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = TempDirPoolConfig::default();
//! let manager = TempDirectoryManager::new(config).await?;
//!
//! // Allocate directories for a test
//! let test_id = "test_model_training";
//! let dirs = manager.allocate_directories(2, test_id).await?;
//!
//! // Use the directories...
//!
//! // Clean up when test completes
//! manager.deallocate_directories_for_test(test_id).await?;
//! # Ok(())
//! # }
//! ```

pub mod cleanup_scheduler;
pub mod conflict_resolver;
pub mod core_manager;
pub mod quota_manager;
pub mod types;
pub mod utils;

// Re-export all public types for direct access
pub use types::{
    decrement_counter,
    get_counter_value,
    increment_counter,
    new_shared_counter,
    CleanupEvent,
    CleanupEventType,
    CleanupPriority,
    CleanupResult,
    CleanupScheduler,

    CleanupStatistics,

    // Cleanup types
    CleanupTask,
    CleanupTaskType,
    ConflictDetector,
    ConflictResolutionStrategy,
    ConflictType,
    DirectoryConflict,
    DirectoryQuota,
    DirectorySpaceUsage,

    // Traits
    DiskSpaceProvider,
    EnhancedDirectoryStatus,
    ManagerInstanceInfo,
    ManagerStatus,

    // Utility types and functions
    SharedCounter,
    // Core types
    TempDirError,
    TempDirResult,
    // Configuration types
    TempDirectoryManagerConfig,
};

pub use cleanup_scheduler::DirectoryCleanupScheduler;
pub use conflict_resolver::{
    ConflictDetectionSettings, ConflictResolutionRecord, ConflictResolutionStatistics,
    ConflictSensitivity, DirectoryConflictResolver, ResolutionOutcome,
};
pub use core_manager::TempDirectoryManager;
pub use quota_manager::{DirectoryQuotaManager, DirectoryUsageInfo, QuotaEnforcementLevel};
pub use utils::{
    calculate_directory_size, calculate_directory_usage, clean_empty_directories,
    copy_directory_contents, create_directory_recursive, format_bytes, format_duration,
    generate_unique_directory_name, get_available_disk_space, get_canonical_path,
    get_directory_permissions, has_sufficient_space, has_write_permission, is_directory_empty,
    is_path_safe, move_directory, remove_directory_recursive, sanitize_path_component,
    set_directory_permissions, validate_directory_name, validate_test_id,
};

use anyhow::Result;
use std::sync::Arc;

// ================================================================================================
// Main Manager Factory and Convenience Functions
// ================================================================================================

/// Create a new temporary directory manager with default configuration
pub async fn create_temp_directory_manager() -> Result<TempDirectoryManager, TempDirError> {
    let config = crate::resource_management::types::TempDirPoolConfig::default();
    TempDirectoryManager::new(config).await
}

/// Create a temporary directory manager with custom configuration
pub async fn create_temp_directory_manager_with_config(
    config: crate::resource_management::types::TempDirPoolConfig,
) -> Result<TempDirectoryManager, TempDirError> {
    TempDirectoryManager::new(config).await
}

/// Create a temporary directory manager with full custom configuration
pub async fn create_temp_directory_manager_with_full_config(
    config: TempDirectoryManagerConfig,
) -> Result<TempDirectoryManager, TempDirError> {
    TempDirectoryManager::with_full_config(config).await
}

// ================================================================================================
// Specialized Component Factories
// ================================================================================================

/// Create a cleanup scheduler with default settings
pub fn create_cleanup_scheduler() -> DirectoryCleanupScheduler {
    DirectoryCleanupScheduler::new()
}

/// Create a cleanup scheduler with custom settings
pub fn create_cleanup_scheduler_with_config(
    auto_cleanup_enabled: bool,
    max_concurrent_operations: usize,
) -> DirectoryCleanupScheduler {
    DirectoryCleanupScheduler::with_config(auto_cleanup_enabled, max_concurrent_operations)
}

/// Create a quota manager with configuration
pub fn create_quota_manager(
    config: &crate::resource_management::types::TempDirPoolConfig,
) -> DirectoryQuotaManager {
    DirectoryQuotaManager::new(config)
}

/// Create a quota manager with custom enforcement level
pub fn create_quota_manager_with_enforcement(
    config: &crate::resource_management::types::TempDirPoolConfig,
    enforcement_level: QuotaEnforcementLevel,
    monitoring_enabled: bool,
) -> DirectoryQuotaManager {
    DirectoryQuotaManager::with_settings(config, enforcement_level, monitoring_enabled)
}

/// Create a conflict resolver with default settings
pub fn create_conflict_resolver() -> DirectoryConflictResolver {
    DirectoryConflictResolver::new()
}

/// Create a conflict resolver with custom settings
pub fn create_conflict_resolver_with_settings(
    settings: ConflictDetectionSettings,
) -> DirectoryConflictResolver {
    DirectoryConflictResolver::with_settings(settings)
}

// ================================================================================================
// Builder Pattern for Complex Configuration
// ================================================================================================

/// Builder for TempDirectoryManager configuration
pub struct TempDirectoryManagerBuilder {
    config: TempDirectoryManagerConfig,
}

impl TempDirectoryManagerBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: TempDirectoryManagerConfig::default(),
        }
    }

    /// Set the pool configuration
    pub fn with_pool_config(
        mut self,
        pool_config: crate::resource_management::types::TempDirPoolConfig,
    ) -> Self {
        self.config.pool_config = pool_config;
        self
    }

    /// Enable or disable detailed logging
    pub fn with_detailed_logging(mut self, enabled: bool) -> Self {
        self.config.enable_detailed_logging = enabled;
        self
    }

    /// Set conflict resolution timeout
    pub fn with_conflict_resolution_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.config.conflict_resolution_timeout = timeout;
        self
    }

    /// Set maximum concurrent cleanup operations
    pub fn with_max_concurrent_cleanups(mut self, max_operations: usize) -> Self {
        self.config.max_concurrent_cleanups = max_operations;
        self
    }

    /// Set statistics retention period
    pub fn with_stats_retention_period(mut self, retention_period: std::time::Duration) -> Self {
        self.config.stats_retention_period = retention_period;
        self
    }

    /// Enable or disable performance monitoring
    pub fn with_performance_monitoring(mut self, enabled: bool) -> Self {
        self.config.enable_performance_monitoring = enabled;
        self
    }

    /// Build the TempDirectoryManager
    pub async fn build(self) -> Result<TempDirectoryManager, TempDirError> {
        TempDirectoryManager::with_full_config(self.config).await
    }
}

impl Default for TempDirectoryManagerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ================================================================================================
// Integrated System Functions
// ================================================================================================

/// Comprehensive system for managing temporary directories
pub struct TempDirectorySystem {
    manager: Arc<TempDirectoryManager>,
    cleanup_scheduler: Arc<DirectoryCleanupScheduler>,
    quota_manager: Arc<DirectoryQuotaManager>,
    conflict_resolver: Arc<DirectoryConflictResolver>,
}

impl TempDirectorySystem {
    /// Create a new integrated temp directory system
    pub async fn new(config: TempDirectoryManagerConfig) -> Result<Self, TempDirError> {
        let manager = Arc::new(TempDirectoryManager::with_full_config(config.clone()).await?);
        let cleanup_scheduler = Arc::new(DirectoryCleanupScheduler::with_config(
            config.pool_config.enable_auto_cleanup,
            config.max_concurrent_cleanups,
        ));
        let quota_manager = Arc::new(DirectoryQuotaManager::new(&config.pool_config));
        let conflict_resolver = Arc::new(DirectoryConflictResolver::new());

        Ok(Self {
            manager,
            cleanup_scheduler,
            quota_manager,
            conflict_resolver,
        })
    }

    /// Get the main directory manager
    pub fn manager(&self) -> &Arc<TempDirectoryManager> {
        &self.manager
    }

    /// Get the cleanup scheduler
    pub fn cleanup_scheduler(&self) -> &Arc<DirectoryCleanupScheduler> {
        &self.cleanup_scheduler
    }

    /// Get the quota manager
    pub fn quota_manager(&self) -> &Arc<DirectoryQuotaManager> {
        &self.quota_manager
    }

    /// Get the conflict resolver
    pub fn conflict_resolver(&self) -> &Arc<DirectoryConflictResolver> {
        &self.conflict_resolver
    }

    /// Generate a comprehensive system report
    pub async fn generate_system_report(&self) -> String {
        let manager_report = self.manager.generate_allocation_report().await;
        let cleanup_report = self.cleanup_scheduler.generate_report().await;
        let quota_report = self.quota_manager.generate_quota_report().await;
        let resolution_stats = self.conflict_resolver.get_resolution_statistics().await;

        format!(
            "=== Temporary Directory System Report ===\n\
             \n\
             {}\n\
             \n\
             {}\n\
             \n\
             {}\n\
             \n\
             == Conflict Resolution ==\n\
             Total conflicts detected: {}\n\
             Successful resolutions: {}\n\
             Failed resolutions: {}\n\
             Resolution efficiency: {:.1}%\n\
             Average resolution time: {}",
            manager_report,
            cleanup_report,
            quota_report,
            resolution_stats.total_conflicts_detected,
            resolution_stats.successful_resolutions,
            resolution_stats.failed_resolutions,
            resolution_stats.resolution_efficiency * 100.0,
            format_duration(resolution_stats.average_resolution_time)
        )
    }

    /// Perform system-wide cleanup and maintenance
    pub async fn perform_maintenance(&self) -> Result<MaintenanceResult, TempDirError> {
        let mut result = MaintenanceResult::default();

        // Execute pending cleanups
        result.cleanup_tasks_executed = self.cleanup_scheduler.execute_pending_tasks().await?;

        // Perform quota cleanup if needed
        result.quota_cleanups_performed = self.quota_manager.perform_quota_cleanup().await?.len();

        // Resolve active conflicts
        result.conflicts_resolved = self.conflict_resolver.force_resolve_all_conflicts().await?;

        // Clean up orphaned directories
        result.orphaned_directories_cleaned = self.manager.cleanup_orphaned_directories().await?;

        Ok(result)
    }

    /// Shutdown the entire system gracefully
    pub async fn shutdown(&self) -> Result<(), TempDirError> {
        // Perform final maintenance
        let _maintenance_result = self.perform_maintenance().await?;

        // Shutdown the main manager
        self.manager.shutdown().await?;

        Ok(())
    }
}

/// Result of a maintenance operation
#[derive(Debug, Default)]
pub struct MaintenanceResult {
    /// Number of cleanup tasks executed
    pub cleanup_tasks_executed: usize,
    /// Number of quota cleanups performed
    pub quota_cleanups_performed: usize,
    /// Number of conflicts resolved
    pub conflicts_resolved: usize,
    /// Number of orphaned directories cleaned
    pub orphaned_directories_cleaned: usize,
}

impl MaintenanceResult {
    /// Check if any maintenance actions were performed
    pub fn has_actions(&self) -> bool {
        self.cleanup_tasks_executed > 0
            || self.quota_cleanups_performed > 0
            || self.conflicts_resolved > 0
            || self.orphaned_directories_cleaned > 0
    }

    /// Generate a summary of maintenance actions
    pub fn summary(&self) -> String {
        if !self.has_actions() {
            return "No maintenance actions required".to_string();
        }

        let mut actions = Vec::new();

        if self.cleanup_tasks_executed > 0 {
            actions.push(format!(
                "{} cleanup tasks executed",
                self.cleanup_tasks_executed
            ));
        }

        if self.quota_cleanups_performed > 0 {
            actions.push(format!(
                "{} quota cleanups performed",
                self.quota_cleanups_performed
            ));
        }

        if self.conflicts_resolved > 0 {
            actions.push(format!("{} conflicts resolved", self.conflicts_resolved));
        }

        if self.orphaned_directories_cleaned > 0 {
            actions.push(format!(
                "{} orphaned directories cleaned",
                self.orphaned_directories_cleaned
            ));
        }

        format!("Maintenance completed: {}", actions.join(", "))
    }
}

// ================================================================================================
// Convenience Functions for Common Operations
// ================================================================================================

/// Quick allocation of temporary directories with automatic cleanup
pub async fn allocate_temp_directories_with_cleanup(
    count: usize,
    test_id: &str,
    _cleanup_policy: crate::resource_management::types::TempDirectoryCleanupPolicy,
) -> Result<(TempDirectoryManager, Vec<String>), TempDirError> {
    let manager = create_temp_directory_manager().await?;
    let directories = manager.allocate_directories(count, test_id).await?;

    // The cleanup policy is already set during allocation based on the manager's configuration
    // In a more advanced version, you could customize this per allocation

    Ok((manager, directories))
}

/// Allocate directories and return a guard that automatically cleans up when dropped
pub async fn allocate_temp_directories_with_guard(
    count: usize,
    test_id: &str,
) -> Result<TempDirectoryGuard, TempDirError> {
    let manager = Arc::new(create_temp_directory_manager().await?);
    let directories = manager.allocate_directories(count, test_id).await?;

    Ok(TempDirectoryGuard {
        manager,
        test_id: test_id.to_string(),
        directories,
    })
}

/// RAII guard that automatically cleans up directories when dropped
pub struct TempDirectoryGuard {
    manager: Arc<TempDirectoryManager>,
    test_id: String,
    directories: Vec<String>,
}

impl TempDirectoryGuard {
    /// Get the allocated directories
    pub fn directories(&self) -> &[String] {
        &self.directories
    }

    /// Get the test ID
    pub fn test_id(&self) -> &str {
        &self.test_id
    }

    /// Manually trigger cleanup (guard will still clean up on drop)
    pub async fn cleanup(&self) -> Result<usize, TempDirError> {
        self.manager.deallocate_directories_for_test(&self.test_id).await
    }
}

impl Drop for TempDirectoryGuard {
    fn drop(&mut self) {
        let manager = self.manager.clone();
        let test_id = self.test_id.clone();

        // Spawn cleanup task
        tokio::spawn(async move {
            if let Err(e) = manager.deallocate_directories_for_test(&test_id).await {
                tracing::warn!(
                    test_id = %test_id,
                    error = %e,
                    "Failed to cleanup directories in guard drop"
                );
            }
        });
    }
}

// ================================================================================================
// Testing Utilities
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::Duration;

    #[tokio::test]
    async fn test_manager_creation() {
        let manager = create_temp_directory_manager().await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_builder_pattern() {
        let manager = TempDirectoryManagerBuilder::new()
            .with_detailed_logging(true)
            .with_max_concurrent_cleanups(2)
            .with_conflict_resolution_timeout(Duration::from_secs(10))
            .build()
            .await;

        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_system_integration() {
        let config = TempDirectoryManagerConfig::default();
        let system = TempDirectorySystem::new(config).await;
        assert!(system.is_ok());

        if let Ok(system) = system {
            let report = system.generate_system_report().await;
            assert!(!report.is_empty());
        }
    }

    #[tokio::test]
    async fn test_directory_guard() {
        let guard = allocate_temp_directories_with_guard(1, "test_guard").await;
        assert!(guard.is_ok());

        if let Ok(guard) = guard {
            assert_eq!(guard.directories().len(), 1);
            assert_eq!(guard.test_id(), "test_guard");

            // Guard will automatically clean up when dropped
        }
    }
}
