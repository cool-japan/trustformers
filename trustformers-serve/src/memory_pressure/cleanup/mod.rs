//! # Memory Cleanup System
//!
//! This module provides a comprehensive cleanup system for memory pressure mitigation.
//! It includes various cleanup strategies, handlers, and processing logic to free
//! memory when pressure is detected.
//!
//! ## Core Components
//!
//! - **CleanupHandler Trait**: Base interface for all cleanup implementations
//! - **CacheManager Trait**: Interface for cache-specific cleanup operations
//! - **Cleanup Strategies**: Various approaches to memory cleanup (GC, cache eviction, compaction)
//! - **Processing Logic**: Queue-based cleanup execution with prioritization
//!
//! ## Cleanup Strategies
//!
//! The system supports multiple cleanup strategies with different characteristics:
//!
//! - **Cache Eviction**: Remove cached data based on usage patterns
//! - **Garbage Collection**: Force garbage collection to reclaim unreferenced memory
//! - **Buffer Compaction**: Compact fragmented buffers to reduce memory usage
//! - **Request Rejection**: Reject new requests to prevent further memory allocation
//! - **Model Unloading**: Unload unused ML models from memory
//! - **Memory Defragmentation**: Reorganize memory to reduce fragmentation
//!
//! ## Usage Examples
//!
//! ### Custom Cleanup Handler
//!
//! ```rust
//! use trustformers_serve::memory_pressure::cleanup::{CleanupHandler, MemoryPressureLevel};
//! use anyhow::Result;
//!
//! struct CustomCleanupHandler;
//!
//! impl CleanupHandler for CustomCleanupHandler {
//!     fn cleanup(&self, pressure_level: MemoryPressureLevel) -> Result<u64> {
//!         // Implement custom cleanup logic
//!         match pressure_level {
//!             MemoryPressureLevel::High => {
//!                 // Aggressive cleanup
//!                 Ok(100 * 1024 * 1024) // 100MB freed
//!             },
//!             _ => {
//!                 // Conservative cleanup
//!                 Ok(50 * 1024 * 1024) // 50MB freed
//!             }
//!         }
//!     }
//!
//!     fn estimate_memory_freed(&self) -> u64 {
//!         75 * 1024 * 1024 // 75MB estimate
//!     }
//!
//!     fn get_priority(&self) -> u32 {
//!         100 // Medium priority
//!     }
//! }
//! ```
//!
//! ### Cache Manager Implementation
//!
//! ```rust
//! use trustformers_serve::memory_pressure::cleanup::{CacheManager, MemoryPressureLevel};
//! use anyhow::Result;
//!
//! struct ModelCacheManager {
//!     // Cache implementation
//! }
//!
//! impl CacheManager for ModelCacheManager {
//!     fn evict_cache(&self, pressure_level: MemoryPressureLevel) -> Result<u64> {
//!         // Implement cache eviction logic
//!         Ok(200 * 1024 * 1024) // 200MB freed
//!     }
//!
//!     fn get_cache_size(&self) -> u64 {
//!         1024 * 1024 * 1024 // 1GB total cache
//!     }
//!
//!     fn get_evictable_size(&self) -> u64 {
//!         512 * 1024 * 1024 // 512MB evictable
//!     }
//! }
//! ```

use super::config::*;
use anyhow::Result;

// Re-export submodules
pub mod cache;
pub mod handlers;
pub mod strategies;

// Re-export key types from submodules
pub use cache::*;
pub use handlers::*;
pub use strategies::*;

// =============================================================================
// Core Cleanup Traits
// =============================================================================

/// Trait for memory cleanup handlers
///
/// This trait defines the interface that all cleanup implementations must follow.
/// Cleanup handlers are responsible for freeing memory when pressure is detected,
/// with different strategies optimized for different types of memory usage.
///
/// ## Design Principles
///
/// - **Pressure-Aware**: Cleanup intensity varies based on memory pressure level
/// - **Predictable**: Handlers should provide reliable estimates of memory freed
/// - **Prioritized**: Different handlers have different execution priorities
/// - **Thread-Safe**: All handlers must be safe for concurrent execution
///
/// ## Implementation Guidelines
///
/// - Cleanup operations should be non-blocking where possible
/// - Memory freed estimates should be conservative but realistic
/// - Higher priority values are executed first (0-255 range recommended)
/// - Error handling should be robust to prevent cleanup chain failures
pub trait CleanupHandler: Send + Sync + std::fmt::Debug {
    /// Execute cleanup operation
    ///
    /// This method performs the actual cleanup work, freeing memory based on
    /// the current pressure level. The amount of cleanup should scale with
    /// the pressure level, with more aggressive cleanup for higher pressure.
    ///
    /// # Arguments
    ///
    /// * `pressure_level` - Current memory pressure level
    ///
    /// # Returns
    ///
    /// Returns the amount of memory freed in bytes, or an error if cleanup failed.
    ///
    /// # Implementation Notes
    ///
    /// - Should be idempotent (safe to call multiple times)
    /// - Should complete quickly to avoid blocking other cleanup operations
    /// - Should handle pressure levels gracefully (more aggressive at higher levels)
    fn cleanup(&self, pressure_level: MemoryPressureLevel) -> Result<u64>;

    /// Estimate memory that can be freed
    ///
    /// This method provides an estimate of how much memory this handler can
    /// free without actually performing the cleanup. This is used for cleanup
    /// planning and strategy selection.
    ///
    /// # Returns
    ///
    /// Estimated bytes that can be freed by this handler.
    ///
    /// # Implementation Notes
    ///
    /// - Should be fast (no expensive calculations)
    /// - Should be conservative (actual cleanup may free less)
    /// - Should be consistent (repeated calls should return similar values)
    fn estimate_memory_freed(&self) -> u64;

    /// Get cleanup priority
    ///
    /// Returns the priority level for this cleanup handler. Higher priority
    /// handlers are executed first during cleanup operations.
    ///
    /// # Returns
    ///
    /// Priority value (0-255, where 255 is highest priority)
    ///
    /// # Priority Guidelines
    ///
    /// - 200-255: Critical handlers (emergency cleanup only)
    /// - 150-199: High priority (buffer compaction, critical cache eviction)
    /// - 100-149: Medium priority (garbage collection, general cache eviction)
    /// - 50-99: Low priority (background cleanup, optimization)
    /// - 0-49: Very low priority (non-essential cleanup)
    fn get_priority(&self) -> u32;

    /// Check if handler can be executed at current pressure level
    ///
    /// Some cleanup handlers may only be appropriate at certain pressure levels.
    /// This method allows handlers to indicate when they should be skipped.
    ///
    /// # Arguments
    ///
    /// * `pressure_level` - Current memory pressure level
    ///
    /// # Returns
    ///
    /// True if the handler should be executed, false to skip.
    fn should_execute(&self, pressure_level: MemoryPressureLevel) -> bool {
        // Default implementation: execute for any pressure level above normal
        pressure_level > MemoryPressureLevel::Normal
    }

    /// Get handler name for logging and debugging
    ///
    /// Returns a human-readable name for this cleanup handler.
    fn name(&self) -> &'static str {
        "UnknownCleanupHandler"
    }
}

/// Trait for cache management operations
///
/// This trait provides a specialized interface for cache-related cleanup
/// operations. It extends beyond basic cleanup to provide cache-specific
/// functionality like selective eviction and cache size management.
///
/// ## Cache Management Strategies
///
/// - **LRU Eviction**: Remove least recently used items first
/// - **Size-Based Eviction**: Remove largest items to maximize memory freed
/// - **Priority-Based Eviction**: Remove low-priority items first
/// - **Age-Based Eviction**: Remove oldest items first
/// - **Frequency-Based Eviction**: Remove least frequently used items
pub trait CacheManager: Send + Sync + std::fmt::Debug {
    /// Evict cache entries based on pressure level
    ///
    /// This method removes cache entries to free memory, with eviction
    /// aggressiveness based on the current memory pressure level.
    ///
    /// # Arguments
    ///
    /// * `pressure_level` - Current memory pressure level
    ///
    /// # Returns
    ///
    /// Amount of memory freed by cache eviction in bytes.
    fn evict_cache(&self, pressure_level: MemoryPressureLevel) -> Result<u64>;

    /// Get total cache size in bytes
    ///
    /// Returns the current total size of all cached data.
    fn get_cache_size(&self) -> u64;

    /// Get evictable cache size in bytes
    ///
    /// Returns the size of cache data that can be safely evicted
    /// without affecting critical operations.
    fn get_evictable_size(&self) -> u64;

    /// Get cache hit rate
    ///
    /// Returns the cache hit rate as a value between 0.0 and 1.0.
    /// This can be used to inform eviction strategies.
    fn get_hit_rate(&self) -> f32 {
        0.0 // Default implementation
    }

    /// Get number of cache entries
    ///
    /// Returns the total number of items currently in the cache.
    fn get_entry_count(&self) -> usize {
        0 // Default implementation
    }

    /// Evict specific percentage of cache
    ///
    /// Evicts approximately the specified percentage of cache data.
    ///
    /// # Arguments
    ///
    /// * `percentage` - Percentage to evict (0.0-1.0)
    ///
    /// # Returns
    ///
    /// Amount of memory freed in bytes.
    fn evict_percentage(&self, percentage: f32) -> Result<u64> {
        let target_bytes = (self.get_evictable_size() as f32 * percentage.clamp(0.0, 1.0)) as u64;
        self.evict_bytes(target_bytes)
    }

    /// Evict specific amount of cache data
    ///
    /// Evicts approximately the specified number of bytes from cache.
    ///
    /// # Arguments
    ///
    /// * `target_bytes` - Target amount to evict in bytes
    ///
    /// # Returns
    ///
    /// Actual amount of memory freed in bytes.
    fn evict_bytes(&self, _target_bytes: u64) -> Result<u64> {
        // Default implementation: evict based on low pressure
        self.evict_cache(MemoryPressureLevel::Low)
    }
}

// =============================================================================
// Cleanup Action Management
// =============================================================================

/// Builder for creating cleanup actions
///
/// Provides a fluent interface for creating cleanup actions with proper
/// configuration and validation.
#[derive(Debug, Clone)]
pub struct CleanupActionBuilder {
    strategy: Option<CleanupStrategy>,
    priority: Option<u32>,
    estimated_memory_freed: Option<u64>,
    gpu_device_id: Option<u32>,
}

impl CleanupActionBuilder {
    /// Create a new cleanup action builder
    pub fn new() -> Self {
        Self {
            strategy: None,
            priority: None,
            estimated_memory_freed: None,
            gpu_device_id: None,
        }
    }

    /// Set the cleanup strategy
    pub fn strategy(mut self, strategy: CleanupStrategy) -> Self {
        self.strategy = Some(strategy);
        self
    }

    /// Set the action priority
    pub fn priority(mut self, priority: u32) -> Self {
        self.priority = Some(priority);
        self
    }

    /// Set estimated memory to be freed
    pub fn estimated_memory_freed(mut self, bytes: u64) -> Self {
        self.estimated_memory_freed = Some(bytes);
        self
    }

    /// Set target GPU device (for GPU cleanup strategies)
    pub fn gpu_device_id(mut self, device_id: u32) -> Self {
        self.gpu_device_id = Some(device_id);
        self
    }

    /// Build the cleanup action
    pub fn build(self) -> Result<CleanupAction> {
        let strategy =
            self.strategy.ok_or_else(|| anyhow::anyhow!("Cleanup strategy is required"))?;

        Ok(CleanupAction {
            strategy,
            priority: self.priority.unwrap_or(100),
            estimated_memory_freed: self.estimated_memory_freed.unwrap_or(0),
            gpu_device_id: self.gpu_device_id,
            queued_at: chrono::Utc::now(),
        })
    }
}

impl Default for CleanupActionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Cleanup Execution Context
// =============================================================================

/// Context information for cleanup execution
///
/// Provides contextual information to cleanup handlers about the current
/// system state and cleanup requirements.
#[derive(Debug, Clone)]
pub struct CleanupContext {
    /// Current memory pressure level
    pub pressure_level: MemoryPressureLevel,

    /// Current memory utilization (0.0-1.0)
    pub utilization: f32,

    /// Available memory in bytes
    pub available_memory: u64,

    /// Target memory to free in bytes
    pub target_memory_freed: u64,

    /// Whether this is an emergency cleanup
    pub is_emergency: bool,

    /// Maximum time allowed for cleanup (in milliseconds)
    pub timeout_ms: Option<u64>,
}

impl CleanupContext {
    /// Create a new cleanup context
    pub fn new(
        pressure_level: MemoryPressureLevel,
        utilization: f32,
        available_memory: u64,
    ) -> Self {
        Self {
            pressure_level,
            utilization,
            available_memory,
            target_memory_freed: 0,
            is_emergency: pressure_level >= MemoryPressureLevel::Critical,
            timeout_ms: None,
        }
    }

    /// Set target memory to free
    pub fn with_target_memory_freed(mut self, target_bytes: u64) -> Self {
        self.target_memory_freed = target_bytes;
        self
    }

    /// Set emergency flag
    pub fn with_emergency(mut self, is_emergency: bool) -> Self {
        self.is_emergency = is_emergency;
        self
    }

    /// Set cleanup timeout
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }

    /// Check if cleanup should be aggressive
    pub fn should_be_aggressive(&self) -> bool {
        self.is_emergency || self.pressure_level >= MemoryPressureLevel::High
    }

    /// Get cleanup urgency score (0.0-1.0)
    pub fn get_urgency_score(&self) -> f32 {
        match self.pressure_level {
            MemoryPressureLevel::Normal => 0.0,
            MemoryPressureLevel::Low => 0.2,
            MemoryPressureLevel::Medium => 0.4,
            MemoryPressureLevel::High => 0.7,
            MemoryPressureLevel::Critical => 0.9,
            MemoryPressureLevel::Emergency => 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct MockCleanupHandler {
        name: &'static str,
        priority: u32,
        memory_freed: u64,
    }

    impl CleanupHandler for MockCleanupHandler {
        fn cleanup(&self, _pressure_level: MemoryPressureLevel) -> Result<u64> {
            Ok(self.memory_freed)
        }

        fn estimate_memory_freed(&self) -> u64 {
            self.memory_freed
        }

        fn get_priority(&self) -> u32 {
            self.priority
        }

        fn name(&self) -> &'static str {
            self.name
        }
    }

    #[test]
    fn test_cleanup_action_builder() {
        let action = CleanupActionBuilder::new()
            .strategy(CleanupStrategy::CacheEviction)
            .priority(150)
            .estimated_memory_freed(1024 * 1024)
            .build()
            .unwrap();

        assert_eq!(action.strategy, CleanupStrategy::CacheEviction);
        assert_eq!(action.priority, 150);
        assert_eq!(action.estimated_memory_freed, 1024 * 1024);
    }

    #[test]
    fn test_cleanup_context() {
        let context = CleanupContext::new(MemoryPressureLevel::High, 0.85, 1024 * 1024 * 1024)
            .with_target_memory_freed(100 * 1024 * 1024)
            .with_emergency(true);

        assert_eq!(context.pressure_level, MemoryPressureLevel::High);
        assert_eq!(context.utilization, 0.85);
        assert!(context.is_emergency);
        assert!(context.should_be_aggressive());
        assert_eq!(context.get_urgency_score(), 0.7);
    }

    #[test]
    fn test_mock_cleanup_handler() {
        let handler = MockCleanupHandler {
            name: "TestHandler",
            priority: 100,
            memory_freed: 50 * 1024 * 1024,
        };

        assert_eq!(handler.name(), "TestHandler");
        assert_eq!(handler.get_priority(), 100);
        assert_eq!(handler.estimate_memory_freed(), 50 * 1024 * 1024);
        assert!(handler.should_execute(MemoryPressureLevel::Medium));
        assert!(!handler.should_execute(MemoryPressureLevel::Normal));

        let result = handler.cleanup(MemoryPressureLevel::High).unwrap();
        assert_eq!(result, 50 * 1024 * 1024);
    }

    #[test]
    fn test_urgency_scores() {
        let contexts = vec![
            (MemoryPressureLevel::Normal, 0.0),
            (MemoryPressureLevel::Low, 0.2),
            (MemoryPressureLevel::Medium, 0.4),
            (MemoryPressureLevel::High, 0.7),
            (MemoryPressureLevel::Critical, 0.9),
            (MemoryPressureLevel::Emergency, 1.0),
        ];

        for (level, expected_score) in contexts {
            let context = CleanupContext::new(level, 0.5, 1024 * 1024 * 1024);
            assert_eq!(context.get_urgency_score(), expected_score);
        }
    }
}
