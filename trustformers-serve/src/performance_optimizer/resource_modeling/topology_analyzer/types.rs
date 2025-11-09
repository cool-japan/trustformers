//! Core types for topology analysis

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Validation levels for topology analysis
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValidationLevel {
    /// Basic validation with minimal overhead
    Basic,

    /// Comprehensive validation with detailed checks
    Comprehensive,

    /// Enterprise-grade validation with exhaustive analysis
    Enterprise,
}

/// Precision levels for analysis
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PrecisionLevel {
    /// Fast analysis with reduced precision
    Fast,

    /// Balanced analysis with good precision/performance ratio
    Balanced,

    /// High precision analysis with detailed measurements
    HighPrecision,

    /// Maximum precision with exhaustive measurements
    Maximum,
}

/// Cache types supported in analysis
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CacheType {
    /// Instruction cache
    Instruction,

    /// Data cache
    Data,

    /// Unified cache (both instruction and data)
    Unified,

    /// Translation Lookaside Buffer
    TLB,

    /// Trace cache
    Trace,
}

/// Cache replacement policies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CacheReplacementPolicy {
    /// Least Recently Used
    LRU,

    /// First In, First Out
    FIFO,

    /// Least Frequently Used
    LFU,

    /// Random replacement
    Random,

    /// Pseudo-LRU
    PseudoLRU,

    /// Adaptive replacement cache
    ARC,
}

/// Cache write policies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CacheWritePolicy {
    /// Write-through policy
    WriteThrough,

    /// Write-back policy
    WriteBack,

    /// Write-around policy
    WriteAround,
}

/// Cache coherency protocol types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CoherencyProtocolType {
    /// Modified, Shared, Invalid
    MSI,

    /// Modified, Exclusive, Shared, Invalid
    MESI,

    /// Modified, Owned, Exclusive, Shared, Invalid
    MOESI,

    /// Forward, Owned, Exclusive, Shared, Invalid
    MOESIF,

    /// Directory-based protocol
    Directory,
}

/// Topology analysis cache for performance optimization
#[derive(Debug, Default)]
pub struct TopologyAnalysisCache {
    /// Cached topology analysis results
    pub cached_results: HashMap<String, Vec<u8>>,

    /// Cache hit/miss statistics
    pub cache_stats: CacheStatistics,

    /// Cache configuration
    pub cache_config: CacheConfig,
}

/// Cache statistics
#[derive(Debug, Default)]
pub struct CacheStatistics {
    /// Number of cache hits
    pub hits: u64,

    /// Number of cache misses
    pub misses: u64,

    /// Total cache accesses
    pub total_accesses: u64,

    /// Average access time
    pub avg_access_time: Duration,
}

/// Cache configuration
#[derive(Debug)]
pub struct CacheConfig {
    /// Maximum cache size in bytes
    pub max_size: usize,

    /// Cache entry TTL
    pub ttl: Duration,

    /// Enable cache compression
    pub enable_compression: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_size: 128 * 1024 * 1024,    // 128MB default
            ttl: Duration::from_secs(3600), // 1 hour default
            enable_compression: true,
        }
    }
}

// =============================================================================
// Additional Types for resource_modeling.rs Imports
// =============================================================================

/// Cache bandwidth characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheBandwidthCharacteristics {
    /// Read bandwidth in bytes per second
    pub read_bandwidth: u64,

    /// Write bandwidth in bytes per second
    pub write_bandwidth: u64,

    /// Bidirectional bandwidth in bytes per second
    pub bidirectional_bandwidth: u64,

    /// Latency in nanoseconds
    pub latency_ns: u64,
}

/// Cache hierarchy node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheHierarchyNode {
    /// Cache level (L1, L2, L3, etc.)
    pub level: u8,

    /// Cache type
    pub cache_type: CacheType,

    /// Cache size in bytes
    pub size: usize,

    /// Line size in bytes
    pub line_size: usize,

    /// Associativity
    pub associativity: usize,

    /// Sharing level (cores, sockets, etc.)
    pub sharing_level: String,

    /// Children in hierarchy
    pub children: Vec<String>,
}

/// Cache latency characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheLatencyCharacteristics {
    /// Hit latency in nanoseconds
    pub hit_latency_ns: u64,

    /// Miss latency in nanoseconds
    pub miss_latency_ns: u64,

    /// Average access latency
    pub avg_access_latency_ns: f64,

    /// Load latency
    pub load_latency_ns: u64,

    /// Store latency
    pub store_latency_ns: u64,
}

/// Cache sharing pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheSharingPattern {
    /// Shared by cores
    pub shared_cores: Vec<usize>,

    /// Sharing granularity
    pub granularity: String,

    /// Exclusive mode enabled
    pub exclusive: bool,

    /// Inclusive mode enabled
    pub inclusive: bool,
}

/// Cache topology mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheTopologyMapping {
    /// Mapping from cache level to nodes
    pub level_to_nodes: HashMap<u8, Vec<String>>,

    /// Mapping from core to caches
    pub core_to_caches: HashMap<usize, Vec<String>>,

    /// Total cache size
    pub total_cache_size: usize,

    /// Number of cache levels
    pub num_levels: u8,
}

/// NUMA domain metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaDomainMetrics {
    /// Domain ID
    pub domain_id: usize,

    /// Total memory in bytes
    pub total_memory: u64,

    /// Available memory in bytes
    pub available_memory: u64,

    /// CPU cores in this domain
    pub cpu_cores: Vec<usize>,

    /// Local access latency in nanoseconds
    pub local_latency_ns: u64,

    /// Remote access latency in nanoseconds
    pub remote_latency_ns: u64,

    /// Bandwidth to other domains
    pub inter_domain_bandwidth: HashMap<usize, u64>,
}
