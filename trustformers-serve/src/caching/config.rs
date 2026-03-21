//! Cache Configuration
//!
//! Configuration structures for all cache types and policies.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Main cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Result cache configuration
    pub result_cache: TierConfig,

    /// Embedding cache configuration
    pub embedding_cache: TierConfig,

    /// KV cache configuration
    pub kv_cache: KVCacheConfig,

    /// Distributed cache configuration
    pub distributed: DistributedConfig,

    /// Cache warming configuration
    pub warming: WarmingConfig,

    /// Enable distributed caching
    pub enable_distributed: bool,

    /// Enable cache warming
    pub enable_warming: bool,

    /// Global cache mode
    pub cache_mode: CacheMode,

    /// Consistency level for distributed caching
    pub consistency_level: ConsistencyLevel,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            result_cache: TierConfig {
                max_size_bytes: 1024 * 1024 * 1024, // 1GB
                max_entries: 100_000,
                default_ttl: Duration::from_secs(3600), // 1 hour
                eviction_policy: EvictionPolicy::LRU,
                compression_enabled: true,
                tier_name: "result_cache".to_string(),
            },
            embedding_cache: TierConfig {
                max_size_bytes: 512 * 1024 * 1024, // 512MB
                max_entries: 50_000,
                default_ttl: Duration::from_secs(7200), // 2 hours
                eviction_policy: EvictionPolicy::LFU,
                compression_enabled: false,
                tier_name: "embedding_cache".to_string(),
            },
            kv_cache: KVCacheConfig {
                max_size_bytes: 2 * 1024 * 1024 * 1024, // 2GB
                max_sequences: 1000,
                max_layers: 80,
                max_sequence_length: 8192,
                sharing_enabled: true,
                compression_enabled: false,
                eviction_policy: EvictionPolicy::LRU,
            },
            distributed: DistributedConfig::default(),
            warming: WarmingConfig::default(),
            enable_distributed: false,
            enable_warming: true,
            cache_mode: CacheMode::Performance,
            consistency_level: ConsistencyLevel::Eventual,
        }
    }
}

/// Configuration for individual cache tiers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierConfig {
    /// Maximum cache size in bytes
    pub max_size_bytes: usize,

    /// Maximum number of entries
    pub max_entries: usize,

    /// Default time-to-live for entries
    pub default_ttl: Duration,

    /// Eviction policy
    pub eviction_policy: EvictionPolicy,

    /// Enable compression for stored data
    pub compression_enabled: bool,

    /// Tier name for metrics
    pub tier_name: String,
}

/// KV cache specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVCacheConfig {
    /// Maximum cache size in bytes
    pub max_size_bytes: usize,

    /// Maximum number of sequences to cache
    pub max_sequences: usize,

    /// Maximum number of layers to cache
    pub max_layers: usize,

    /// Maximum sequence length to cache
    pub max_sequence_length: usize,

    /// Enable sharing of KV cache between sequences
    pub sharing_enabled: bool,

    /// Enable compression for KV cache
    pub compression_enabled: bool,

    /// Eviction policy for KV cache
    pub eviction_policy: EvictionPolicy,
}

/// Distributed cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Cache node addresses
    pub nodes: Vec<String>,

    /// Replication factor
    pub replication_factor: usize,

    /// Consistency level
    pub consistency_level: ConsistencyLevel,

    /// Connection timeout
    pub connection_timeout: Duration,

    /// Request timeout
    pub request_timeout: Duration,

    /// Retry policy
    pub retry_attempts: usize,

    /// Enable automatic failover
    pub enable_failover: bool,

    /// Health check interval
    pub health_check_interval: Duration,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            nodes: vec!["localhost:6379".to_string()],
            replication_factor: 1,
            consistency_level: ConsistencyLevel::Eventual,
            connection_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(2),
            retry_attempts: 3,
            enable_failover: true,
            health_check_interval: Duration::from_secs(30),
        }
    }
}

/// Cache warming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmingConfig {
    /// Enable cache warming
    pub enabled: bool,

    /// Warming strategies to use
    pub strategies: Vec<WarmingStrategy>,

    /// Warming schedule
    pub schedule: WarmingSchedule,

    /// Maximum warming concurrency
    pub max_concurrent_requests: usize,

    /// Warming request timeout
    pub request_timeout: Duration,

    /// Popular queries file path
    pub popular_queries_file: Option<String>,

    /// Minimum hit rate threshold for warming
    pub min_hit_rate_threshold: f32,
}

impl Default for WarmingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strategies: vec![
                WarmingStrategy::PopularQueries,
                WarmingStrategy::RecentQueries,
            ],
            schedule: WarmingSchedule::Interval(Duration::from_secs(3600)),
            max_concurrent_requests: 10,
            request_timeout: Duration::from_secs(30),
            popular_queries_file: None,
            min_hit_rate_threshold: 0.7,
        }
    }
}

/// Cache eviction policies
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,

    /// Least Frequently Used
    LFU,

    /// Time To Live based
    TTL,

    /// Priority based (custom scoring)
    Priority,

    /// Random eviction
    Random,

    /// First In First Out
    FIFO,
}

/// Cache modes for different optimization targets
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum CacheMode {
    /// Optimize for maximum performance
    Performance,

    /// Optimize for memory efficiency
    Memory,

    /// Balance between performance and memory
    Balanced,

    /// Optimize for cache hit rate
    HitRate,

    /// Custom mode with specific parameters
    Custom,
}

/// Consistency levels for distributed caching
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ConsistencyLevel {
    /// Strong consistency - all nodes must agree
    Strong,

    /// Eventual consistency - updates propagate eventually
    Eventual,

    /// Session consistency - consistent within a session
    Session,

    /// Weak consistency - no guarantees
    Weak,
}

/// Cache warming strategies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WarmingStrategy {
    /// Warm cache with popular queries
    PopularQueries,

    /// Warm cache with recent queries
    RecentQueries,

    /// Warm cache with predicted queries
    PredictiveQueries,

    /// Warm cache with user-provided queries
    CustomQueries(Vec<String>),

    /// Warm cache based on access patterns
    AccessPatterns,
}

/// Cache warming schedule
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WarmingSchedule {
    /// Warm at fixed intervals
    Interval(Duration),

    /// Warm at specific times of day
    TimeOfDay(Vec<String>), // HH:MM format

    /// Warm on startup only
    Startup,

    /// Manual warming only
    Manual,

    /// Adaptive warming based on hit rates
    Adaptive {
        min_hit_rate: f32,
        check_interval: Duration,
    },
}

/// Cache tier levels for hierarchical caching
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum CacheTier {
    /// L1 cache (fastest, smallest)
    L1,

    /// L2 cache (medium speed/size)
    L2,

    /// L3 cache (slower, largest)
    L3,

    /// Distributed cache
    Distributed,
}

/// Cache operation types for metrics
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum CacheOperation {
    Get,
    Put,
    Delete,
    Clear,
    Evict,
    Warm,
    Invalidate,
}
