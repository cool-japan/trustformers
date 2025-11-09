/// Weight Loading Configuration
///
/// This module contains all configuration structs and enums for weight loading functionality.
use std::path::PathBuf;
use std::time::Duration;

/// Supported weight file formats
#[derive(Debug, Clone, PartialEq)]
pub enum WeightFormat {
    HuggingFaceBin, // PyTorch .bin files
    SafeTensors,    // SafeTensors format
    ONNX,           // ONNX format
    TensorFlow,     // TensorFlow SavedModel
    GGUF,           // GGUF quantized format
    Custom(String), // Custom format
}

/// Weight loading configuration
#[derive(Debug, Clone)]
pub struct WeightLoadingConfig {
    pub format: Option<WeightFormat>,
    pub lazy_loading: bool,
    pub memory_mapped: bool,
    pub streaming: bool,
    pub device: String,
    pub dtype: WeightDataType,
    pub quantization: Option<QuantizationConfig>,
    pub cache_dir: Option<PathBuf>,
    pub verify_checksums: bool,
    pub distributed: Option<DistributedConfig>,
}

/// Distributed weight loading configuration
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// List of worker nodes for distributed loading
    pub nodes: Vec<NodeConfig>,
    /// Load balancing strategy
    pub load_balancer: LoadBalancingStrategy,
    /// Fault tolerance settings
    pub fault_tolerance: FaultToleranceConfig,
    /// Network settings
    pub network: NetworkConfig,
    /// Caching strategy across nodes
    pub distributed_cache: DistributedCacheConfig,
    /// Enable compression for network transfer
    pub compression: bool,
}

/// Configuration for individual nodes
#[derive(Debug, Clone)]
pub struct NodeConfig {
    pub id: String,
    pub address: String,
    pub port: u16,
    pub weight_capacity: u64, // Maximum weights this node can hold (bytes)
    pub bandwidth: f64,       // Network bandwidth to this node (MB/s)
    pub priority: u8,         // Higher priority nodes are preferred (0-255)
    pub storage_paths: Vec<PathBuf>, // Paths where weights are stored on this node
}

/// Load balancing strategies for distributed weight loading
#[derive(Debug, Clone, PartialEq)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    WeightedRoundRobin,
    ConsistentHashing,
    LocalityAware,
    PerformanceBased,
    Adaptive,
}

/// Fault tolerance configuration
#[derive(Debug, Clone)]
pub struct FaultToleranceConfig {
    pub max_retries: u32,
    pub retry_delay: Duration,
    pub timeout: Duration,
    pub enable_failover: bool,
    pub health_check_interval: Duration,
    pub backup_nodes: Vec<String>, // Node IDs to use as backups
}

/// Network configuration for distributed loading
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    pub max_concurrent_connections: usize,
    pub connection_timeout: Duration,
    pub read_timeout: Duration,
    pub chunk_size: usize,
    pub enable_keepalive: bool,
    pub compression_threshold: usize, // Compress transfers larger than this
}

/// Distributed cache configuration
#[derive(Debug, Clone)]
pub struct DistributedCacheConfig {
    pub cache_strategy: CacheStrategy,
    pub replication_factor: u8, // How many nodes should cache each tensor
    pub eviction_policy: CacheEvictionPolicy,
    pub consistency_level: ConsistencyLevel,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CacheStrategy {
    None,
    ReadThrough,
    WriteThrough,
    WriteBack,
    ReadAround,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CacheEvictionPolicy {
    LRU,
    LFU,
    FIFO,
    Random,
    TTL,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConsistencyLevel {
    Strong,
    Eventual,
    Weak,
}

#[derive(Debug, Clone)]
pub enum WeightDataType {
    Float32,
    Float16,
    BFloat16,
    Int8,
    Int4,
}

#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    pub bits: u8,
    pub group_size: Option<usize>,
    pub symmetric: bool,
}

impl Default for WeightLoadingConfig {
    fn default() -> Self {
        Self {
            format: None,
            lazy_loading: false,
            memory_mapped: false,
            streaming: false,
            device: "cpu".to_string(),
            dtype: WeightDataType::Float32,
            quantization: None,
            cache_dir: None,
            verify_checksums: true,
            distributed: None,
        }
    }
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_delay: Duration::from_millis(1000),
            timeout: Duration::from_secs(30),
            enable_failover: true,
            health_check_interval: Duration::from_secs(60),
            backup_nodes: Vec::new(),
        }
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            max_concurrent_connections: 10,
            connection_timeout: Duration::from_secs(30),
            read_timeout: Duration::from_secs(60),
            chunk_size: 8192,
            enable_keepalive: true,
            compression_threshold: 1024 * 1024, // 1MB
        }
    }
}

impl Default for DistributedCacheConfig {
    fn default() -> Self {
        Self {
            cache_strategy: CacheStrategy::ReadThrough,
            replication_factor: 2,
            eviction_policy: CacheEvictionPolicy::LRU,
            consistency_level: ConsistencyLevel::Eventual,
        }
    }
}
