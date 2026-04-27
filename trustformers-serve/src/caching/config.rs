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

#[cfg(test)]
mod tests {
    use super::*;

    // --- CacheConfig tests ---

    #[test]
    fn test_cache_config_default_result_cache_size() {
        let cfg = CacheConfig::default();
        assert_eq!(cfg.result_cache.max_size_bytes, 1024 * 1024 * 1024);
    }

    #[test]
    fn test_cache_config_default_embedding_cache_entries() {
        let cfg = CacheConfig::default();
        assert_eq!(cfg.embedding_cache.max_entries, 50_000);
    }

    #[test]
    fn test_cache_config_default_kv_cache_layers() {
        let cfg = CacheConfig::default();
        assert_eq!(cfg.kv_cache.max_layers, 80);
    }

    #[test]
    fn test_cache_config_default_distributed_disabled() {
        let cfg = CacheConfig::default();
        assert!(!cfg.enable_distributed);
    }

    #[test]
    fn test_cache_config_default_warming_enabled() {
        let cfg = CacheConfig::default();
        assert!(cfg.enable_warming);
    }

    #[test]
    fn test_cache_config_default_mode_is_performance() {
        let cfg = CacheConfig::default();
        assert_eq!(cfg.cache_mode, CacheMode::Performance);
    }

    #[test]
    fn test_cache_config_default_consistency_eventual() {
        let cfg = CacheConfig::default();
        assert_eq!(cfg.consistency_level, ConsistencyLevel::Eventual);
    }

    // --- KVCacheConfig tests ---

    #[test]
    fn test_kv_cache_config_sharing_enabled() {
        let cfg = CacheConfig::default();
        assert!(cfg.kv_cache.sharing_enabled);
    }

    #[test]
    fn test_kv_cache_config_compression_disabled_by_default() {
        let cfg = CacheConfig::default();
        assert!(!cfg.kv_cache.compression_enabled);
    }

    #[test]
    fn test_kv_cache_config_eviction_lru() {
        let cfg = CacheConfig::default();
        assert_eq!(cfg.kv_cache.eviction_policy, EvictionPolicy::LRU);
    }

    // --- DistributedConfig tests ---

    #[test]
    fn test_distributed_config_default_nodes() {
        let cfg = DistributedConfig::default();
        assert_eq!(cfg.nodes.len(), 1);
        assert_eq!(cfg.nodes[0], "localhost:6379");
    }

    #[test]
    fn test_distributed_config_default_replication_factor() {
        let cfg = DistributedConfig::default();
        assert_eq!(cfg.replication_factor, 1);
    }

    #[test]
    fn test_distributed_config_failover_enabled_by_default() {
        let cfg = DistributedConfig::default();
        assert!(cfg.enable_failover);
    }

    #[test]
    fn test_distributed_config_retry_attempts() {
        let cfg = DistributedConfig::default();
        assert_eq!(cfg.retry_attempts, 3);
    }

    // --- WarmingConfig tests ---

    #[test]
    fn test_warming_config_default_enabled() {
        let cfg = WarmingConfig::default();
        assert!(cfg.enabled);
    }

    #[test]
    fn test_warming_config_default_has_two_strategies() {
        let cfg = WarmingConfig::default();
        assert_eq!(cfg.strategies.len(), 2);
    }

    #[test]
    fn test_warming_config_default_strategies_include_popular() {
        let cfg = WarmingConfig::default();
        assert!(cfg.strategies.contains(&WarmingStrategy::PopularQueries));
    }

    #[test]
    fn test_warming_config_default_strategies_include_recent() {
        let cfg = WarmingConfig::default();
        assert!(cfg.strategies.contains(&WarmingStrategy::RecentQueries));
    }

    #[test]
    fn test_warming_config_default_max_concurrent() {
        let cfg = WarmingConfig::default();
        assert_eq!(cfg.max_concurrent_requests, 10);
    }

    #[test]
    fn test_warming_config_hit_rate_threshold() {
        let cfg = WarmingConfig::default();
        assert!((cfg.min_hit_rate_threshold - 0.7).abs() < 1e-6);
    }

    // --- EvictionPolicy tests ---

    #[test]
    fn test_eviction_policy_equality_lru() {
        let p1 = EvictionPolicy::LRU;
        let p2 = EvictionPolicy::LRU;
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_eviction_policy_inequality() {
        assert_ne!(EvictionPolicy::LRU, EvictionPolicy::LFU);
        assert_ne!(EvictionPolicy::TTL, EvictionPolicy::FIFO);
        assert_ne!(EvictionPolicy::Priority, EvictionPolicy::Random);
    }

    // --- CacheMode tests ---

    #[test]
    fn test_cache_mode_equality() {
        assert_eq!(CacheMode::Performance, CacheMode::Performance);
        assert_ne!(CacheMode::Memory, CacheMode::Balanced);
    }

    // --- ConsistencyLevel tests ---

    #[test]
    fn test_consistency_level_equality() {
        assert_eq!(ConsistencyLevel::Strong, ConsistencyLevel::Strong);
        assert_ne!(ConsistencyLevel::Eventual, ConsistencyLevel::Weak);
    }

    // --- WarmingSchedule tests ---

    #[test]
    fn test_warming_schedule_startup_equality() {
        assert_eq!(WarmingSchedule::Startup, WarmingSchedule::Startup);
        assert_ne!(WarmingSchedule::Manual, WarmingSchedule::Startup);
    }

    // --- PowerScalingFactors (no direct dep but TierConfig clone) ---

    #[test]
    fn test_tier_config_clone() {
        let tier = TierConfig {
            max_size_bytes: 100,
            max_entries: 10,
            default_ttl: Duration::from_secs(60),
            eviction_policy: EvictionPolicy::LRU,
            compression_enabled: false,
            tier_name: "test".to_string(),
        };
        let cloned = tier.clone();
        assert_eq!(cloned.max_size_bytes, 100);
        assert_eq!(cloned.tier_name, "test");
    }
}
