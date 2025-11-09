//! # Memory Pressure Configuration and Types
//!
//! This module contains all configuration structures, enums, and data types
//! for the memory pressure handling system. It provides comprehensive configuration
//! options for system and GPU memory monitoring, adaptive thresholds, cleanup
//! strategies, and predictive memory management.
//!
//! ## Core Components
//!
//! - **MemoryPressureConfig**: Main configuration structure
//! - **MemoryPressureThresholds**: Adaptive threshold configuration
//! - **Data Types**: Memory statistics, GPU stats, usage patterns
//! - **Enums**: Pressure levels, cleanup strategies, device strategies
//! - **Events**: Memory pressure event types for monitoring
//!
//! ## Usage Examples
//!
//! ### Basic Configuration
//!
//! ```rust
//! use trustformers_serve::memory_pressure::config::MemoryPressureConfig;
//!
//! let config = MemoryPressureConfig::default();
//! assert!(config.enabled);
//! assert_eq!(config.monitoring_interval_seconds, 1);
//! ```
//!
//! ### Custom Thresholds
//!
//! ```rust
//! use trustformers_serve::memory_pressure::config::{
//!     MemoryPressureConfig, MemoryPressureThresholds
//! };
//!
//! let mut config = MemoryPressureConfig::default();
//! config.pressure_thresholds = MemoryPressureThresholds {
//!     low: 0.5,
//!     medium: 0.7,
//!     high: 0.8,
//!     critical: 0.9,
//!     ..MemoryPressureThresholds::default()
//! };
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    time::Duration,
};

// =============================================================================
// Core Configuration Structures
// =============================================================================

/// Memory pressure configuration
///
/// Comprehensive configuration for memory pressure monitoring and handling,
/// including system memory, GPU memory, cleanup strategies, and predictive
/// memory management features.
///
/// ## Key Features
///
/// - **Adaptive Thresholds**: Dynamic adjustment based on system behavior
/// - **Multi-GPU Support**: Individual device monitoring and cleanup
/// - **Predictive Management**: ML-inspired memory allocation prediction
/// - **Comprehensive Cleanup**: Multiple cleanup strategies with priority
/// - **Emergency Handling**: Critical memory situation management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPressureConfig {
    /// Enable memory pressure monitoring
    pub enabled: bool,

    /// Memory pressure thresholds (0.0-1.0)
    pub pressure_thresholds: MemoryPressureThresholds,

    /// Monitoring interval in seconds
    pub monitoring_interval_seconds: u64,

    /// Memory cleanup strategies
    pub cleanup_strategies: Vec<CleanupStrategy>,

    /// Emergency memory threshold (0.0-1.0)
    pub emergency_threshold: f32,

    /// Memory buffer size in MB
    pub memory_buffer_mb: usize,

    /// Enable aggressive cleanup
    pub enable_aggressive_cleanup: bool,

    /// GC trigger threshold
    pub gc_trigger_threshold: f32,

    /// Cache eviction threshold
    pub cache_eviction_threshold: f32,

    /// Request rejection threshold
    pub request_rejection_threshold: f32,

    /// Enable memory compaction
    pub enable_memory_compaction: bool,

    /// Swap usage threshold
    pub swap_usage_threshold: f32,

    /// Enable GPU memory pressure monitoring
    pub enable_gpu_monitoring: bool,

    /// GPU memory pressure thresholds
    pub gpu_pressure_thresholds: MemoryPressureThresholds,

    /// GPU cleanup strategies
    pub gpu_cleanup_strategies: Vec<GpuCleanupStrategy>,

    /// GPU memory buffer size in MB
    pub gpu_memory_buffer_mb: usize,

    /// GPU fragmentation threshold
    pub gpu_fragmentation_threshold: f32,

    /// GPU device selection strategy
    pub gpu_device_strategy: GpuDeviceStrategy,
}

impl Default for MemoryPressureConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            pressure_thresholds: MemoryPressureThresholds::default(),
            monitoring_interval_seconds: 1,
            cleanup_strategies: vec![
                CleanupStrategy::CacheEviction,
                CleanupStrategy::GarbageCollection,
                CleanupStrategy::BufferCompaction,
                CleanupStrategy::RequestRejection,
            ],
            emergency_threshold: 0.95,
            memory_buffer_mb: 256,
            enable_aggressive_cleanup: true,
            gc_trigger_threshold: 0.8,
            cache_eviction_threshold: 0.75,
            request_rejection_threshold: 0.9,
            enable_memory_compaction: true,
            swap_usage_threshold: 0.5,
            enable_gpu_monitoring: true,
            gpu_pressure_thresholds: MemoryPressureThresholds {
                low: 0.7, // GPU memory pressure starts higher
                medium: 0.8,
                high: 0.9,
                critical: 0.95,
                ..MemoryPressureThresholds::default()
            },
            gpu_cleanup_strategies: vec![
                GpuCleanupStrategy::GpuCacheEviction,
                GpuCleanupStrategy::GpuBufferCompaction,
                GpuCleanupStrategy::GpuModelUnloading,
                GpuCleanupStrategy::GpuVramCompaction,
            ],
            gpu_memory_buffer_mb: 512, // Larger buffer for GPU
            gpu_fragmentation_threshold: 0.8,
            gpu_device_strategy: GpuDeviceStrategy::All,
        }
    }
}

/// Memory pressure thresholds with adaptive adjustment
///
/// Configurable thresholds for different memory pressure levels with
/// adaptive adjustment capabilities that learn from system behavior
/// and automatically optimize threshold values over time.
///
/// ## Adaptive Features
///
/// - **Learning Rate**: Controls how quickly thresholds adapt
/// - **Base Values**: Fallback values for threshold adjustment
/// - **Adjustment Factor**: Scaling factor for threshold modifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPressureThresholds {
    /// Low pressure threshold (0.0-1.0)
    pub low: f32,

    /// Medium pressure threshold (0.0-1.0)
    pub medium: f32,

    /// High pressure threshold (0.0-1.0)
    pub high: f32,

    /// Critical pressure threshold (0.0-1.0)
    pub critical: f32,

    /// Enable adaptive threshold adjustment
    pub adaptive: bool,

    /// Base thresholds for adaptive adjustment
    pub base_low: f32,
    pub base_medium: f32,
    pub base_high: f32,
    pub base_critical: f32,

    /// Adaptive adjustment factor (0.1-2.0)
    pub adjustment_factor: f32,

    /// Learning rate for adaptive adjustment
    pub learning_rate: f32,
}

impl Default for MemoryPressureThresholds {
    fn default() -> Self {
        Self {
            low: 0.6,
            medium: 0.75,
            high: 0.85,
            critical: 0.95,
            adaptive: true,
            base_low: 0.6,
            base_medium: 0.75,
            base_high: 0.85,
            base_critical: 0.95,
            adjustment_factor: 1.0,
            learning_rate: 0.01,
        }
    }
}

// =============================================================================
// Enumerations and Type Definitions
// =============================================================================

/// Memory pressure level
///
/// Hierarchical pressure levels representing the severity of memory
/// constraints. Each level triggers different cleanup strategies
/// and system responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MemoryPressureLevel {
    /// No memory pressure
    Normal,

    /// Low memory pressure
    Low,

    /// Medium memory pressure
    Medium,

    /// High memory pressure
    High,

    /// Critical memory pressure
    Critical,

    /// Emergency memory pressure
    Emergency,
}

/// Cleanup strategies
///
/// Various strategies for reducing memory usage when pressure is detected.
/// Each strategy has different performance characteristics and effectiveness
/// depending on the system state and workload type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum CleanupStrategy {
    /// Cache eviction
    CacheEviction,

    /// Garbage collection
    GarbageCollection,

    /// Buffer compaction
    BufferCompaction,

    /// Request rejection
    RequestRejection,

    /// Model unloading
    ModelUnloading,

    /// Connection throttling
    ConnectionThrottling,

    /// Batch size reduction
    BatchSizeReduction,

    /// Memory defragmentation
    MemoryDefragmentation,
}

/// GPU-specific cleanup strategies
///
/// Specialized cleanup strategies for GPU memory management,
/// taking into account GPU-specific characteristics like VRAM
/// fragmentation, context switching costs, and stream management.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum GpuCleanupStrategy {
    /// GPU cache eviction
    GpuCacheEviction,

    /// GPU buffer compaction
    GpuBufferCompaction,

    /// GPU model unloading
    GpuModelUnloading,

    /// GPU memory defragmentation
    GpuMemoryDefragmentation,

    /// GPU context switching
    GpuContextSwitching,

    /// GPU stream cleanup
    GpuStreamCleanup,

    /// GPU texture memory cleanup
    GpuTextureCleanup,

    /// GPU VRAM compaction
    GpuVramCompaction,

    /// GPU memory pool reset
    GpuMemoryPoolReset,

    /// GPU batch size reduction
    GpuBatchSizeReduction,
}

/// GPU device selection strategy
///
/// Different strategies for selecting which GPU devices to monitor
/// and manage for memory pressure. Supports single device, multi-device,
/// and intelligent selection based on memory or utilization characteristics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GpuDeviceStrategy {
    /// Monitor all available GPU devices
    All,

    /// Monitor primary GPU only
    Primary,

    /// Monitor specific GPU devices by index
    Specific(Vec<u32>),

    /// Monitor GPUs with most memory
    HighestMemory,

    /// Monitor GPUs with highest utilization
    HighestUtilization,

    /// Load-balanced monitoring across devices
    LoadBalanced,
}

/// Pattern recognition type
///
/// Types of temporal patterns that can be detected in memory usage
/// for predictive memory management and optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    Hourly,
    Daily,
    Weekly,
    Custom(String),
}

// =============================================================================
// Data Structures and Statistics
// =============================================================================

/// GPU memory statistics for a single device
///
/// Comprehensive statistics for GPU memory usage, performance metrics,
/// and pressure indicators. Includes both memory utilization and
/// performance characteristics like compute utilization and temperature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMemoryStats {
    /// GPU device index
    pub device_id: u32,

    /// GPU device name
    pub device_name: String,

    /// Total GPU memory in bytes
    pub total_memory: u64,

    /// Available GPU memory in bytes
    pub available_memory: u64,

    /// Used GPU memory in bytes
    pub used_memory: u64,

    /// GPU memory utilization (0.0-1.0)
    pub utilization: f32,

    /// GPU compute utilization (0.0-1.0)
    pub compute_utilization: f32,

    /// GPU memory bandwidth utilization (0.0-1.0)
    pub bandwidth_utilization: f32,

    /// GPU temperature in Celsius
    pub temperature: f32,

    /// GPU power consumption in watts
    pub power_consumption: f32,

    /// GPU memory fragmentation level (0.0-1.0)
    pub fragmentation_level: f32,

    /// GPU memory pressure level
    pub pressure_level: MemoryPressureLevel,

    /// Number of active contexts
    pub active_contexts: u32,

    /// Number of active streams
    pub active_streams: u32,

    /// Allocated memory by type
    pub allocated_by_type: HashMap<String, u64>,

    /// Memory pressure events count
    pub pressure_events: u64,

    /// Last cleanup timestamp
    pub last_cleanup: Option<DateTime<Utc>>,
}

/// Memory statistics
///
/// Comprehensive system memory statistics including system memory,
/// process-specific memory, GPU memory across all devices,
/// and swap usage information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total system memory in bytes
    pub total_memory: u64,

    /// Available memory in bytes
    pub available_memory: u64,

    /// Used memory in bytes
    pub used_memory: u64,

    /// Memory utilization (0.0-1.0)
    pub utilization: f32,

    /// Process memory usage in bytes
    pub process_memory: u64,

    /// Heap memory usage in bytes
    pub heap_memory: u64,

    /// Stack memory usage in bytes
    pub stack_memory: u64,

    /// Total GPU memory usage across all devices in bytes
    pub gpu_memory: u64,

    /// Detailed GPU memory statistics per device
    pub gpu_stats: HashMap<u32, GpuMemoryStats>,

    /// Swap usage in bytes
    pub swap_usage: u64,

    /// Memory pressure level
    pub pressure_level: MemoryPressureLevel,

    /// Last updated timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Memory usage pattern for ML-based prediction
///
/// Historical memory usage data and pattern analysis for predictive
/// memory management. Includes trend analysis, confidence metrics,
/// and seasonal pattern detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsagePattern {
    /// Time window in seconds
    pub window_seconds: u64,

    /// Memory usage samples (timestamp, utilization)
    pub samples: VecDeque<(DateTime<Utc>, f32)>,

    /// Trend direction (-1: decreasing, 0: stable, 1: increasing)
    pub trend: f32,

    /// Prediction confidence (0.0-1.0)
    pub confidence: f32,

    /// Predicted utilization in next window
    pub predicted_utilization: f32,

    /// Seasonal pattern detected
    pub seasonal_pattern: Option<SeasonalPattern>,
}

/// Seasonal memory usage pattern
///
/// Detected seasonal patterns in memory usage for more accurate
/// prediction and proactive memory management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalPattern {
    /// Pattern type (hourly, daily, weekly)
    pub pattern_type: PatternType,

    /// Pattern cycle duration in seconds
    pub cycle_duration: u64,

    /// Peak usage times (offset from cycle start in seconds)
    pub peak_times: Vec<u64>,

    /// Low usage times
    pub low_times: Vec<u64>,

    /// Pattern strength (0.0-1.0)
    pub strength: f32,
}

/// Memory allocation predictor using ML-inspired techniques
///
/// Predictive memory management using historical data analysis,
/// regression techniques, and adaptive threshold adjustment
/// for proactive memory pressure prevention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPredictor {
    /// Historical usage patterns
    pub patterns: HashMap<String, MemoryUsagePattern>,

    /// Linear regression weights for simple prediction
    pub regression_weights: Vec<f32>,

    /// Moving averages for trend analysis
    pub short_term_average: f32,
    pub medium_term_average: f32,
    pub long_term_average: f32,

    /// Prediction accuracy tracking
    pub prediction_accuracy: f32,

    /// Last prediction error
    pub last_error: f32,

    /// Adaptive threshold multipliers
    pub threshold_multipliers: HashMap<MemoryPressureLevel, f32>,

    /// Current predicted memory utilization
    pub predicted_utilization: f32,
}

impl Default for MemoryPredictor {
    fn default() -> Self {
        let mut threshold_multipliers = HashMap::new();
        threshold_multipliers.insert(MemoryPressureLevel::Low, 1.0);
        threshold_multipliers.insert(MemoryPressureLevel::Medium, 1.0);
        threshold_multipliers.insert(MemoryPressureLevel::High, 1.0);
        threshold_multipliers.insert(MemoryPressureLevel::Critical, 1.0);

        Self {
            patterns: HashMap::new(),
            regression_weights: vec![0.1, 0.2, 0.3, 0.4], // Simple 4-point regression
            short_term_average: 0.0,
            medium_term_average: 0.0,
            long_term_average: 0.0,
            prediction_accuracy: 1.0,
            last_error: 0.0,
            threshold_multipliers,
            predicted_utilization: 0.0,
        }
    }
}

/// Smart memory preallocation strategy
///
/// Intelligent memory preallocation based on usage patterns
/// to reduce allocation overhead and improve performance
/// during high memory pressure situations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreallocationStrategy {
    /// Enable preallocation
    pub enabled: bool,

    /// Preallocation buffer size in MB
    pub buffer_size_mb: usize,

    /// Preallocation trigger threshold (0.0-1.0)
    pub trigger_threshold: f32,

    /// Maximum preallocation size in MB
    pub max_preallocation_mb: usize,

    /// Preallocation pools by size
    pub pools: HashMap<usize, usize>, // size -> count

    /// Pool usage statistics
    pub pool_stats: HashMap<usize, PoolStats>,
}

/// Memory pool statistics
///
/// Performance metrics for memory pool efficiency and utilization
/// to optimize pool sizing and allocation strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolStats {
    /// Total allocations from pool
    pub total_allocations: u64,

    /// Current pool utilization
    pub utilization: f32,

    /// Hit rate (successful allocations from pool)
    pub hit_rate: f32,

    /// Pool efficiency score
    pub efficiency: f32,
}

impl Default for PreallocationStrategy {
    fn default() -> Self {
        let mut pools = HashMap::new();
        pools.insert(1024, 100); // 1KB pool with 100 slots
        pools.insert(4096, 50); // 4KB pool with 50 slots
        pools.insert(16384, 25); // 16KB pool with 25 slots
        pools.insert(65536, 10); // 64KB pool with 10 slots

        Self {
            enabled: true,
            buffer_size_mb: 64,
            trigger_threshold: 0.7,
            max_preallocation_mb: 256,
            pools,
            pool_stats: HashMap::new(),
        }
    }
}

// =============================================================================
// Event Types
// =============================================================================

/// Memory pressure event
///
/// Events generated by the memory pressure system for monitoring,
/// logging, and integration with external systems. Includes both
/// system memory and GPU memory events.
#[derive(Debug, Clone, Serialize)]
pub enum MemoryPressureEvent {
    /// Memory pressure level changed
    PressureLevelChanged {
        old_level: MemoryPressureLevel,
        new_level: MemoryPressureLevel,
        utilization: f32,
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    /// GPU memory pressure level changed
    GpuPressureLevelChanged {
        device_id: u32,
        pressure_level: MemoryPressureLevel,
        utilization: f32,
        available_memory: u64,
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    /// Cleanup strategy triggered
    CleanupTriggered {
        strategy: CleanupStrategy,
        memory_freed: u64,
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    /// GPU cleanup strategy triggered
    GpuCleanupTriggered {
        device_id: u32,
        strategy: GpuCleanupStrategy,
        memory_freed: u64,
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    /// Memory allocation failed
    AllocationFailed {
        requested_size: u64,
        available_memory: u64,
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    /// Emergency memory cleanup
    EmergencyCleanup {
        memory_freed: u64,
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    /// Memory compaction completed
    CompactionCompleted {
        memory_compacted: u64,
        duration: Duration,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
}

// =============================================================================
// Helper Types for Handler Implementation
// =============================================================================

/// Cleanup action for the cleanup queue
///
/// Represents a cleanup action to be executed by the memory pressure handler.
/// Includes priority, strategy type, and estimated memory impact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupAction {
    /// Cleanup strategy to execute
    pub strategy: CleanupStrategy,

    /// Action priority (higher values executed first)
    pub priority: u32,

    /// Estimated memory to be freed in bytes
    pub estimated_memory_freed: u64,

    /// Target GPU device (if applicable)
    pub gpu_device_id: Option<u32>,

    /// Timestamp when action was queued
    pub queued_at: chrono::DateTime<chrono::Utc>,
}

/// Memory allocation information
///
/// Information about memory allocations for tracking and optimization.
/// Used by the memory pressure handler to understand allocation patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationInfo {
    /// Allocation size in bytes
    pub size: u64,

    /// Allocation type (e.g., "model", "cache", "buffer")
    pub allocation_type: String,

    /// Allocation timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Allocation lifetime hint in seconds
    pub lifetime_hint: Option<u64>,

    /// Allocation priority (0-100, higher is more important)
    pub priority: u32,

    /// GPU device ID (if GPU allocation)
    pub gpu_device_id: Option<u32>,
}

/// Memory pressure snapshot
///
/// Point-in-time snapshot of memory pressure state for analysis
/// and trend detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PressureSnapshot {
    /// Snapshot timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Memory utilization at snapshot time
    pub utilization: f32,

    /// Pressure level at snapshot time
    pub pressure_level: MemoryPressureLevel,

    /// Available memory in bytes
    pub available_memory: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MemoryPressureConfig::default();
        assert!(config.enabled);
        assert_eq!(config.monitoring_interval_seconds, 1);
        assert!(!config.cleanup_strategies.is_empty());
        assert!(config.enable_gpu_monitoring);
    }

    #[test]
    fn test_pressure_level_ordering() {
        assert!(MemoryPressureLevel::Normal < MemoryPressureLevel::Low);
        assert!(MemoryPressureLevel::Low < MemoryPressureLevel::Medium);
        assert!(MemoryPressureLevel::Medium < MemoryPressureLevel::High);
        assert!(MemoryPressureLevel::High < MemoryPressureLevel::Critical);
        assert!(MemoryPressureLevel::Critical < MemoryPressureLevel::Emergency);
    }

    #[test]
    fn test_threshold_defaults() {
        let thresholds = MemoryPressureThresholds::default();
        assert!(thresholds.adaptive);
        assert!(thresholds.low < thresholds.medium);
        assert!(thresholds.medium < thresholds.high);
        assert!(thresholds.high < thresholds.critical);
    }

    #[test]
    fn test_predictor_initialization() {
        let predictor = MemoryPredictor::default();
        assert_eq!(predictor.regression_weights.len(), 4);
        assert_eq!(predictor.prediction_accuracy, 1.0);
        assert!(predictor.threshold_multipliers.len() > 0);
    }

    #[test]
    fn test_preallocation_strategy_pools() {
        let strategy = PreallocationStrategy::default();
        assert!(strategy.enabled);
        assert!(!strategy.pools.is_empty());
        assert!(strategy.pools.contains_key(&1024));
        assert!(strategy.pools.contains_key(&4096));
    }
}
