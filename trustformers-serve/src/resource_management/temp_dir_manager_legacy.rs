//! Comprehensive Temporary Directory Manager for TrustformeRS Test Parallelization
//!
//! This module provides a robust and comprehensive temporary directory management system
//! designed for test parallelization. It includes advanced features such as:
//!
//! - **Directory allocation and deallocation** with lifecycle management
//! - **Quota enforcement** and disk space monitoring
//! - **Automated cleanup** with configurable retention policies
//! - **Thread-safe operations** with proper synchronization
//! - **Conflict detection and resolution** for concurrent access
//! - **Comprehensive statistics** and usage tracking
//! - **Error handling** with detailed error types
//! - **Logging and observability** for operations monitoring
//!
//! ## Architecture
//!
//! This module has been refactored into a modular architecture with specialized
//! components for better maintainability and performance. The original monolithic
//! implementation is preserved through re-exports for backward compatibility.
//!
//! ### Modular Components
//!
//! - [`temp_dir_manager::types`] - Core type definitions and error handling
//! - [`temp_dir_manager::core_manager`] - Main TempDirectoryManager implementation
//! - [`temp_dir_manager::cleanup_scheduler`] - Cleanup operations management
//! - [`temp_dir_manager::quota_manager`] - Disk space quota enforcement
//! - [`temp_dir_manager::conflict_resolver`] - Conflict detection and resolution
//! - [`temp_dir_manager::utils`] - Utility functions and helpers
//!
//! ## Key Components
//!
//! - [`TempDirectoryManager`] - Main coordinator for all directory operations
//! - [`DirectoryCleanupScheduler`] - Handles automated cleanup operations
//! - [`DirectoryQuotaManager`] - Manages disk space quotas and monitoring
//! - [`DirectoryConflictResolver`] - Resolves access conflicts
//!
//! ## Error Handling
//!
//! The module defines a comprehensive [`TempDirError`] enum that covers all possible
//! error conditions with detailed context information.
//!
//! ## Thread Safety
//!
//! All operations are thread-safe using Arc, Mutex, and RwLock for appropriate
//! shared state management.
//!
//! ## Usage Examples
//!
//! ### Basic Directory Allocation
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
//! println!("Allocated directories: {:?}", dirs);
//!
//! // Use the directories...
//!
//! // Cleanup when test completes
//! manager.deallocate_directories_for_test(test_id).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Advanced Configuration
//!
//! ```rust
//! use trustformers_serve::resource_management::temp_dir_manager::{
//!     TempDirectoryManagerBuilder,
//!     TempDirectorySystem,
//! };
//! use trustformers_serve::resource_management::types::{
//!     TempDirPoolConfig,
//!     TempDirectoryCleanupPolicy,
//! };
//! use std::path::PathBuf;
//! use std::time::Duration;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Using the builder pattern
//! let manager = TempDirectoryManagerBuilder::new()
//!     .with_detailed_logging(true)
//!     .with_max_concurrent_cleanups(4)
//!     .with_conflict_resolution_timeout(Duration::from_secs(30))
//!     .with_performance_monitoring(true)
//!     .build()
//!     .await?;
//!
//! // Or use the integrated system
//! let system = TempDirectorySystem::new(Default::default()).await?;
//! let report = system.generate_system_report().await;
//! println!("{}", report);
//! # Ok(())
//! # }
//! ```
//!
//! ### Using Directory Guard for Automatic Cleanup
//!
//! ```rust
//! use trustformers_serve::resource_management::temp_dir_manager::allocate_temp_directories_with_guard;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let guard = allocate_temp_directories_with_guard(3, "test_with_guard").await?;
//!
//! // Use the directories
//! for dir in guard.directories() {
//!     println!("Using directory: {}", dir);
//!     // Perform test operations...
//! }
//!
//! // Directories are automatically cleaned up when guard is dropped
//! # Ok(())
//! # }
//! ```

// Re-export the entire modular temp directory management system
pub mod temp_dir_manager {
    pub mod types;
    pub mod core_manager;
    pub mod cleanup_scheduler;
    pub mod quota_manager;
    pub mod conflict_resolver;
    pub mod utils;

    // Re-export the mod.rs contents
    pub use super::temp_dir_manager::*;
}

// Re-export all public components from the modular implementation for direct access
pub use self::temp_dir_manager::{
    // Core types and errors
    TempDirError,
    TempDirResult,
    DirectoryQuota,
    EnhancedDirectoryStatus,
    DirectoryConflict,
    ConflictType,
    ConflictResolutionStrategy,
    DirectorySpaceUsage,

    // Cleanup types
    CleanupTask,
    CleanupTaskType,
    CleanupPriority,
    CleanupEvent,
    CleanupEventType,
    CleanupResult,
    CleanupStatistics,

    // Configuration types
    TempDirectoryManagerConfig,
    ManagerInstanceInfo,
    ManagerStatus,

    // Main components
    TempDirectoryManager,
    DirectoryCleanupScheduler,
    DirectoryQuotaManager,
    DirectoryConflictResolver,

    // Specialized types
    QuotaEnforcementLevel,
    DirectoryUsageInfo,
    ConflictDetectionSettings,
    ConflictSensitivity,
    ConflictResolutionRecord,
    ResolutionOutcome,
    ConflictResolutionStatistics,

    // Traits
    DiskSpaceProvider,
    ConflictDetector,
    CleanupScheduler,

    // Utility functions
    calculate_directory_size,
    calculate_directory_usage,
    get_available_disk_space,
    has_sufficient_space,
    clean_empty_directories,
    is_directory_empty,
    create_directory_recursive,
    remove_directory_recursive,
    set_directory_permissions,
    get_directory_permissions,
    has_write_permission,
    generate_unique_directory_name,
    sanitize_path_component,
    is_path_safe,
    get_canonical_path,
    copy_directory_contents,
    move_directory,
    validate_directory_name,
    validate_test_id,
    format_bytes,
    format_duration,

    // Factory functions
    create_temp_directory_manager,
    create_temp_directory_manager_with_config,
    create_temp_directory_manager_with_full_config,
    create_cleanup_scheduler,
    create_cleanup_scheduler_with_config,
    create_quota_manager,
    create_quota_manager_with_enforcement,
    create_conflict_resolver,
    create_conflict_resolver_with_settings,

    // Builder and system components
    TempDirectoryManagerBuilder,
    TempDirectorySystem,
    MaintenanceResult,

    // Convenience functions
    allocate_temp_directories_with_cleanup,
    allocate_temp_directories_with_guard,
    TempDirectoryGuard,

    // Utility types
    SharedCounter,
    new_shared_counter,
    increment_counter,
    decrement_counter,
    get_counter_value,
};

// Additional imports needed for backward compatibility with existing code
use anyhow::{Context, Result};
use chrono::{DateTime, Utc, Datelike, Timelike};
use parking_lot::{Mutex, RwLock};
use std::{
    collections::{HashMap, VecDeque},
    env,
    fs::{self, Metadata},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, SystemTime},
};
use tokio::{task::JoinHandle, time::interval};
use tracing::{debug, error, info, instrument, warn};

use super::types::*;

// ================================================================================================
// Backward Compatibility Layer for Legacy Code
// ================================================================================================

// Re-export types for backward compatibility with any existing code that imports directly
pub use self::temp_dir_manager::types::TempDirError;

/// Legacy compatibility: Alias for TempDirectoryManager
pub type TempDirectoryManagerLegacy = TempDirectoryManager;

/// Legacy compatibility: Create a temporary directory manager (old interface)
pub async fn create_legacy_temp_directory_manager(
    config: TempDirPoolConfig,
) -> Result<TempDirectoryManager, TempDirError> {
    TempDirectoryManager::new(config).await
}

/// Legacy compatibility: Create directory cleanup scheduler (old interface)
pub fn create_legacy_cleanup_scheduler() -> DirectoryCleanupScheduler {
    DirectoryCleanupScheduler::new()
}

/// Legacy compatibility: Create quota manager (old interface)
pub fn create_legacy_quota_manager(config: &TempDirPoolConfig) -> DirectoryQuotaManager {
    DirectoryQuotaManager::new(config)
}

/// Legacy compatibility: Create conflict resolver (old interface)
pub fn create_legacy_conflict_resolver() -> DirectoryConflictResolver {
    DirectoryConflictResolver::new()
}

// ================================================================================================
// Enhanced Factory Functions for Common Use Cases
// ================================================================================================

/// Create a temp directory manager optimized for testing
pub async fn create_testing_temp_directory_manager() -> Result<TempDirectoryManager, TempDirError> {
    let base_path = env::temp_dir().join("trustformers_test_dirs");

    let config = TempDirPoolConfig {
        base_path,
        max_directories: 50,
        max_directory_size_bytes: 100 * 1024 * 1024, // 100MB per directory
        default_cleanup_policy: TempDirectoryCleanupPolicy::Immediate,
        enable_auto_cleanup: true,
        cleanup_interval_secs: 30,
    };

    TempDirectoryManager::new(config).await
}

/// Create a temp directory manager optimized for production
pub async fn create_production_temp_directory_manager(
    base_path: PathBuf,
) -> Result<TempDirectoryManager, TempDirError> {
    let config = TempDirPoolConfig {
        base_path,
        max_directories: 200,
        max_directory_size_bytes: 1024 * 1024 * 1024, // 1GB per directory
        default_cleanup_policy: TempDirectoryCleanupPolicy::Delayed(Duration::from_secs(3600)),
        enable_auto_cleanup: true,
        cleanup_interval_secs: 300,
    };

    TempDirectoryManagerBuilder::new()
        .with_pool_config(config)
        .with_detailed_logging(false)
        .with_max_concurrent_cleanups(8)
        .with_performance_monitoring(true)
        .build()
        .await
}

/// Create a temp directory manager for development/debugging
pub async fn create_debug_temp_directory_manager() -> Result<TempDirectoryManager, TempDirError> {
    let base_path = env::temp_dir().join("trustformers_debug_dirs");

    let config = TempDirPoolConfig {
        base_path,
        max_directories: 20,
        max_directory_size_bytes: 50 * 1024 * 1024, // 50MB per directory
        default_cleanup_policy: TempDirectoryCleanupPolicy::Debug,
        enable_auto_cleanup: false, // Manual cleanup for debugging
        cleanup_interval_secs: 0,
    };

    TempDirectoryManagerBuilder::new()
        .with_pool_config(config)
        .with_detailed_logging(true)
        .with_max_concurrent_cleanups(1)
        .with_performance_monitoring(true)
        .build()
        .await
}

// ================================================================================================
// Global Manager Instance (Optional Singleton Pattern)
// ================================================================================================

use std::sync::OnceLock;

static GLOBAL_MANAGER: OnceLock<Arc<TempDirectoryManager>> = OnceLock::new();

/// Get or create a global temp directory manager instance
pub async fn get_global_temp_directory_manager() -> Result<Arc<TempDirectoryManager>, TempDirError> {
    match GLOBAL_MANAGER.get() {
        Some(manager) => Ok(manager.clone()),
        None => {
            let manager = Arc::new(create_temp_directory_manager().await?);
            match GLOBAL_MANAGER.set(manager.clone()) {
                Ok(()) => Ok(manager),
                Err(_) => {
                    // Another thread set it first, get the existing one
                    Ok(GLOBAL_MANAGER.get().unwrap().clone())
                }
            }
        }
    }
}

/// Initialize the global manager with custom configuration
pub async fn initialize_global_temp_directory_manager(
    config: TempDirectoryManagerConfig,
) -> Result<Arc<TempDirectoryManager>, TempDirError> {
    let manager = Arc::new(TempDirectoryManager::with_full_config(config).await?);
    match GLOBAL_MANAGER.set(manager.clone()) {
        Ok(()) => Ok(manager),
        Err(_) => {
            // Global manager was already initialized
            Err(TempDirError::ConfigurationError {
                message: "Global temp directory manager is already initialized".to_string(),
            })
        }
    }
}

/// Shutdown the global manager if it exists
pub async fn shutdown_global_temp_directory_manager() -> Result<(), TempDirError> {
    if let Some(manager) = GLOBAL_MANAGER.get() {
        manager.shutdown().await?;
    }
    Ok(())
}

// ================================================================================================
// Convenience Macros (Optional)
// ================================================================================================

/// Convenience macro for allocating temporary directories with automatic cleanup
#[macro_export]
macro_rules! with_temp_directories {
    ($count:expr, $test_id:expr, $body:block) => {{
        let guard = $crate::resource_management::temp_dir_manager::allocate_temp_directories_with_guard(
            $count, $test_id
        ).await?;

        let directories = guard.directories().to_vec();
        let result = async move {
            $body
        }.await;

        // Guard automatically cleans up on drop
        result
    }};
}

// ================================================================================================
// Testing Utilities and Helpers
// ================================================================================================

#[cfg(test)]
pub mod test_helpers {
    use super::*;
    use std::env;

    /// Create a test manager with temporary base directory
    pub async fn create_test_manager() -> Result<TempDirectoryManager, TempDirError> {
        let test_base = env::temp_dir().join(format!("test_temp_dir_{}", uuid::Uuid::new_v4()));

        let config = TempDirPoolConfig {
            base_path: test_base,
            max_directories: 10,
            max_directory_size_bytes: 10 * 1024 * 1024, // 10MB
            default_cleanup_policy: TempDirectoryCleanupPolicy::Immediate,
            enable_auto_cleanup: true,
            cleanup_interval_secs: 1,
        };

        TempDirectoryManager::new(config).await
    }

    /// Clean up test base directories
    pub async fn cleanup_test_base_directories() -> Result<usize, TempDirError> {
        let temp_dir = env::temp_dir();
        let mut cleaned_count = 0;

        if let Ok(entries) = fs::read_dir(&temp_dir) {
            for entry in entries {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    if let Some(name) = path.file_name() {
                        if name.to_string_lossy().starts_with("test_temp_dir_") {
                            if let Err(e) = fs::remove_dir_all(&path) {
                                warn!(path = %path.display(), error = %e, "Failed to clean test directory");
                            } else {
                                cleaned_count += 1;
                            }
                        }
                    }
                }
            }
        }

        Ok(cleaned_count)
    }
}

// ================================================================================================
// Module Tests
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use test_helpers::*;

    #[tokio::test]
    async fn test_backward_compatibility() {
        // Test that old interfaces still work
        let config = TempDirPoolConfig::default();
        let manager = create_legacy_temp_directory_manager(config).await;
        assert!(manager.is_ok());

        let cleanup_scheduler = create_legacy_cleanup_scheduler();
        assert!(cleanup_scheduler.get_pending_task_count().await == 0);

        let conflict_resolver = create_legacy_conflict_resolver();
        assert!(conflict_resolver.get_active_conflicts().await.is_empty());
    }

    #[tokio::test]
    async fn test_factory_functions() {
        let testing_manager = create_testing_temp_directory_manager().await;
        assert!(testing_manager.is_ok());

        let debug_manager = create_debug_temp_directory_manager().await;
        assert!(debug_manager.is_ok());
    }

    #[tokio::test]
    async fn test_global_manager() {
        let manager1 = get_global_temp_directory_manager().await;
        let manager2 = get_global_temp_directory_manager().await;

        assert!(manager1.is_ok());
        assert!(manager2.is_ok());

        // Should be the same instance
        let m1 = manager1.unwrap();
        let m2 = manager2.unwrap();
        assert!(Arc::ptr_eq(&m1, &m2));
    }

    #[tokio::test]
    async fn test_directory_allocation_and_cleanup() {
        let manager = create_test_manager().await.unwrap();

        let test_id = "integration_test";
        let directories = manager.allocate_directories(2, test_id).await.unwrap();

        assert_eq!(directories.len(), 2);

        // Verify directories exist
        for dir in &directories {
            let path = Path::new(dir);
            assert!(path.exists() && path.is_dir());
        }

        // Clean up
        let cleaned_count = manager.deallocate_directories_for_test(test_id).await.unwrap();
        assert_eq!(cleaned_count, 2);
    }

    #[tokio::test]
    async fn test_system_integration() {
        let config = TempDirectoryManagerConfig::default();
        let system = TempDirectorySystem::new(config).await.unwrap();

        let report = system.generate_system_report().await;
        assert!(!report.is_empty());
        assert!(report.contains("Temporary Directory Manager"));

        let maintenance_result = system.perform_maintenance().await.unwrap();
        // Should be able to perform maintenance without errors

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_directory_guard() {
        let guard = allocate_temp_directories_with_guard(1, "guard_test").await.unwrap();

        assert_eq!(guard.directories().len(), 1);
        assert_eq!(guard.test_id(), "guard_test");

        let dir_path = &guard.directories()[0];
        let path = Path::new(dir_path);
        assert!(path.exists());

        // Guard will automatically clean up when dropped
        drop(guard);

        // Give some time for async cleanup
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}