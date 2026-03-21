//! Model cache type definitions

use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Model cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCacheConfig {
    /// Maximum number of models to cache
    pub max_models: usize,
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
    /// Model eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Enable model preloading
    pub enable_preloading: bool,
    /// Model warm-up timeout in seconds
    pub warmup_timeout_sec: u64,
    /// Enable model versioning
    pub enable_versioning: bool,
    /// Automatic cleanup interval in seconds
    pub cleanup_interval_sec: u64,
    /// Enable model health checks
    pub enable_health_checks: bool,
    /// Health check interval in seconds
    pub health_check_interval_sec: u64,
}

impl Default for ModelCacheConfig {
    fn default() -> Self {
        Self {
            max_models: 10,
            max_memory_mb: 8192, // 8GB
            eviction_policy: EvictionPolicy::LRU,
            enable_preloading: true,
            warmup_timeout_sec: 30,
            enable_versioning: true,
            cleanup_interval_sec: 300, // 5 minutes
            enable_health_checks: true,
            health_check_interval_sec: 60, // 1 minute
        }
    }
}

/// Model eviction policies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First In, First Out
    FIFO,
    /// Time-based expiration
    TTL,
    /// Memory pressure-based
    MemoryPressure,
}

/// Model cache entry
#[derive(Debug, Clone)]
pub struct ModelCacheEntry {
    pub model_id: String,
    pub model_handle: usize,
    pub model_path: String,
    pub model_config: String, // JSON configuration
    pub version: String,
    pub size_mb: usize,
    pub load_time: Instant,
    pub load_duration_ms: f64, // Duration it took to load the model in milliseconds
    pub last_accessed: Instant,
    pub access_count: u64,
    pub warmup_completed: bool,
    pub health_status: ModelHealthStatus,
    pub metadata: ModelMetadata,
}

/// Model health status
#[derive(Debug, Clone, PartialEq)]
pub enum ModelHealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub architecture: String,
    pub framework: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub parameters_count: u64,
    pub quantized: bool,
    pub precision: String,
    pub supported_backends: Vec<String>,
}

/// Model cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCacheStats {
    pub total_models: usize,
    pub loaded_models: usize,
    pub memory_usage_mb: usize,
    pub memory_limit_mb: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_hit_ratio: f64,
    pub evictions: u64,
    pub load_errors: u64,
    pub health_check_failures: u64,
    pub average_load_time_ms: f64,
    pub total_access_count: u64,
    pub uptime_sec: u64,
}

/// Model preloading request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPreloadRequest {
    pub model_id: String,
    pub model_path: String,
    pub config: String, // JSON configuration
    pub version: String,
    pub priority: LoadPriority,
}

/// Model loading priority
#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}
