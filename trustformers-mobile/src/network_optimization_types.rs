//! Configuration and data types for [`crate::network_optimization`].
//!
//! Split out of `network_optimization.rs` (which had grown to 2013 lines,
//! the only file in the workspace over the <2000-line policy) purely to
//! keep the manager/impl file under that limit -- no behavior change.
//! Every type here is re-exported with `pub use types::*;` at the top of
//! `network_optimization.rs`, so every existing
//! `network_optimization::TypeName` path (including the crate-root
//! re-export list in `lib.rs`) resolves identically to before the split.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Network optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkOptimizationConfig {
    /// Enable resumable downloads
    pub enable_resumable_downloads: bool,
    /// Enable bandwidth-aware downloading
    pub enable_bandwidth_awareness: bool,
    /// Enable P2P model sharing
    pub enable_p2p_sharing: bool,
    /// Enable edge server integration
    pub enable_edge_servers: bool,
    /// Offline-first configuration
    pub offline_first: OfflineFirstConfig,
    /// Download optimization settings
    pub download_optimization: DownloadOptimizationConfig,
    /// P2P sharing configuration
    pub p2p_config: P2PConfig,
    /// Edge server configuration
    pub edge_config: EdgeServerConfig,
    /// Network quality monitoring
    pub quality_monitoring: NetworkQualityConfig,
}

/// Offline-first design configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfflineFirstConfig {
    /// Enable offline mode
    pub enable_offline_mode: bool,
    /// Offline cache size in MB
    pub offline_cache_size_mb: usize,
    /// Offline fallback models
    pub fallback_models: Vec<String>,
    /// Sync strategy when coming online
    pub sync_strategy: OfflineSyncStrategy,
    /// Data retention policy for offline mode
    pub offline_retention: OfflineRetentionPolicy,
}

/// Offline synchronization strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OfflineSyncStrategy {
    /// Sync immediately when online
    Immediate,
    /// Sync during optimal conditions
    Opportunistic,
    /// Sync on user demand
    Manual,
    /// Sync in background
    Background,
    /// Adaptive based on connection
    Adaptive,
}

/// Offline data retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfflineRetentionPolicy {
    /// Retain models for days
    pub model_retention_days: usize,
    /// Retain inference cache for hours
    pub cache_retention_hours: usize,
    /// Auto-cleanup when storage low
    pub auto_cleanup_on_low_storage: bool,
    /// Minimum storage to maintain (MB)
    pub min_storage_threshold_mb: usize,
}

/// Download optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadOptimizationConfig {
    /// Chunk size for downloads (KB)
    pub chunk_size_kb: usize,
    /// Maximum concurrent downloads
    pub max_concurrent_downloads: usize,
    /// Download timeout in seconds
    pub download_timeout_seconds: f64,
    /// Retry configuration
    pub retry_config: DownloadRetryConfig,
    /// Compression settings
    pub compression: DownloadCompressionConfig,
    /// Bandwidth adaptation
    pub bandwidth_adaptation: BandwidthAdaptationConfig,
}

/// Download retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadRetryConfig {
    /// Maximum retry attempts
    pub max_retries: usize,
    /// Initial retry delay in milliseconds
    pub initial_delay_ms: f64,
    /// Maximum retry delay in milliseconds
    pub max_delay_ms: f64,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    /// Jitter factor (0.0-1.0)
    pub jitter_factor: f64,
}

/// Download compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadCompressionConfig {
    /// Enable download compression
    pub enable_compression: bool,
    /// Preferred compression algorithms (in order)
    pub preferred_algorithms: Vec<CompressionAlgorithm>,
    /// Minimum file size for compression (bytes)
    pub min_size_for_compression: usize,
    /// Enable on-the-fly decompression
    pub enable_streaming_decompression: bool,
}

/// Compression algorithms for downloads
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// GZIP compression
    Gzip,
    /// Brotli compression
    Brotli,
    /// LZ4 compression
    LZ4,
    /// ZSTD compression
    Zstd,
    /// No compression
    None,
}

/// Bandwidth adaptation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthAdaptationConfig {
    /// Enable automatic bandwidth detection
    pub enable_auto_detection: bool,
    /// Bandwidth monitoring interval (seconds)
    pub monitoring_interval_seconds: f64,
    /// Adaptation thresholds
    pub adaptation_thresholds: BandwidthThresholds,
    /// Quality adaptation settings
    pub quality_adaptation: QualityAdaptationConfig,
}

/// Bandwidth threshold configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthThresholds {
    /// Low bandwidth threshold (Kbps)
    pub low_bandwidth_kbps: f64,
    /// Medium bandwidth threshold (Kbps)
    pub medium_bandwidth_kbps: f64,
    /// High bandwidth threshold (Kbps)
    pub high_bandwidth_kbps: f64,
    /// Ultra-high bandwidth threshold (Kbps)
    pub ultra_high_bandwidth_kbps: f64,
}

/// Quality adaptation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAdaptationConfig {
    /// Enable dynamic quality adjustment
    pub enable_dynamic_quality: bool,
    /// Quality levels for different bandwidths
    pub quality_levels: HashMap<BandwidthTier, QualityLevel>,
    /// Adaptation strategy
    pub adaptation_strategy: QualityAdaptationStrategy,
}

/// Bandwidth tiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BandwidthTier {
    /// Very low bandwidth
    VeryLow,
    /// Low bandwidth
    Low,
    /// Medium bandwidth
    Medium,
    /// High bandwidth
    High,
    /// Ultra-high bandwidth
    UltraHigh,
}

/// Quality levels for adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityLevel {
    /// Model quantization level
    pub quantization_level: u8,
    /// Model compression ratio
    pub compression_ratio: f64,
    /// Maximum model size (MB)
    pub max_model_size_mb: usize,
    /// Enable model pruning
    pub enable_pruning: bool,
}

/// Quality adaptation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityAdaptationStrategy {
    /// Conservative adaptation
    Conservative,
    /// Aggressive adaptation
    Aggressive,
    /// Balanced adaptation
    Balanced,
    /// User-controlled adaptation
    Manual,
}

/// P2P sharing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct P2PConfig {
    /// Enable P2P discovery
    pub enable_discovery: bool,
    /// P2P protocol to use
    pub protocol: P2PProtocol,
    /// Maximum peers to connect to
    pub max_peers: usize,
    /// Security settings
    pub security: P2PSecurityConfig,
    /// Sharing policy
    pub sharing_policy: P2PSharingPolicy,
    /// Resource limits
    pub resource_limits: P2PResourceLimits,
}

/// P2P protocols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum P2PProtocol {
    /// BitTorrent-like protocol
    BitTorrent,
    /// Gossip protocol
    Gossip,
    /// DHT-based protocol
    DHT,
    /// Hybrid protocol
    Hybrid,
}

/// P2P security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct P2PSecurityConfig {
    /// Enable encryption
    pub enable_encryption: bool,
    /// Enable peer authentication
    pub enable_peer_authentication: bool,
    /// Trusted peer whitelist
    pub trusted_peers: Vec<String>,
    /// Enable content verification
    pub enable_content_verification: bool,
    /// Security level
    pub security_level: P2PSecurityLevel,
}

/// P2P security levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum P2PSecurityLevel {
    /// No security
    None,
    /// Basic security
    Basic,
    /// Standard security
    Standard,
    /// High security
    High,
    /// Maximum security
    Maximum,
}

/// P2P sharing policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct P2PSharingPolicy {
    /// Models allowed to share
    pub shareable_models: Vec<String>,
    /// Maximum upload bandwidth (Kbps)
    pub max_upload_bandwidth_kbps: f64,
    /// Sharing time restrictions
    pub time_restrictions: P2PTimeRestrictions,
    /// Battery-aware sharing
    pub battery_aware_sharing: bool,
    /// Network-aware sharing
    pub network_aware_sharing: bool,
}

/// P2P time restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct P2PTimeRestrictions {
    /// Enable time-based restrictions
    pub enable_restrictions: bool,
    /// Allowed hours (0-23)
    pub allowed_hours: Vec<usize>,
    /// Allowed days of week (0-6, Sunday=0)
    pub allowed_days: Vec<usize>,
    /// Timezone for restrictions
    pub timezone: String,
}

/// P2P resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct P2PResourceLimits {
    /// Maximum CPU usage for P2P (%)
    pub max_cpu_usage_percent: f64,
    /// Maximum memory usage for P2P (MB)
    pub max_memory_usage_mb: usize,
    /// Maximum storage for P2P cache (MB)
    pub max_storage_mb: usize,
    /// Maximum connections
    pub max_connections: usize,
}

/// Edge server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeServerConfig {
    /// Enable edge server discovery
    pub enable_discovery: bool,
    /// Edge server endpoints
    pub server_endpoints: Vec<EdgeServerEndpoint>,
    /// Load balancing strategy
    pub load_balancing: EdgeLoadBalancingStrategy,
    /// Failover configuration
    pub failover: EdgeFailoverConfig,
    /// Caching configuration
    pub caching: EdgeCachingConfig,
}

/// Edge server endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeServerEndpoint {
    /// Server URL
    pub url: String,
    /// Server priority (1-10)
    pub priority: u8,
    /// Geographic region
    pub region: String,
    /// Supported capabilities
    pub capabilities: Vec<String>,
    /// Health check endpoint
    pub health_check_url: Option<String>,
}

/// Edge load balancing strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeLoadBalancingStrategy {
    /// Round robin
    RoundRobin,
    /// Lowest latency
    LowestLatency,
    /// Geographically closest
    Geographic,
    /// Least loaded
    LeastLoaded,
    /// Random selection
    Random,
    /// Weighted round robin
    WeightedRoundRobin,
}

/// Edge failover configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeFailoverConfig {
    /// Enable automatic failover
    pub enable_auto_failover: bool,
    /// Health check interval (seconds)
    pub health_check_interval_seconds: f64,
    /// Failure threshold count
    pub failure_threshold: usize,
    /// Recovery check interval (seconds)
    pub recovery_check_interval_seconds: f64,
    /// Failover timeout (seconds)
    pub failover_timeout_seconds: f64,
}

/// Edge caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeCachingConfig {
    /// Enable edge caching
    pub enable_caching: bool,
    /// Cache TTL in hours
    pub cache_ttl_hours: f64,
    /// Maximum cache size (MB)
    pub max_cache_size_mb: usize,
    /// Cache eviction strategy
    pub eviction_strategy: CacheEvictionStrategy,
}

/// Cache eviction strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheEvictionStrategy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First In, First Out
    FIFO,
    /// Time-based expiration
    TTL,
    /// Size-based eviction
    SizeBased,
}

/// Network quality monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkQualityConfig {
    /// Enable continuous monitoring
    pub enable_continuous_monitoring: bool,
    /// Monitoring interval (seconds)
    pub monitoring_interval_seconds: f64,
    /// Quality metrics to track
    pub tracked_metrics: Vec<NetworkMetric>,
    /// Quality thresholds
    pub quality_thresholds: NetworkQualityThresholds,
    /// Adaptive behavior settings
    pub adaptive_behavior: AdaptiveBehaviorConfig,
}

/// Network metrics to monitor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkMetric {
    /// Bandwidth (download)
    BandwidthDown,
    /// Bandwidth (upload)
    BandwidthUp,
    /// Latency/ping
    Latency,
    /// Packet loss
    PacketLoss,
    /// Jitter
    Jitter,
    /// Connection stability
    Stability,
    /// Signal strength
    SignalStrength,
}

/// Network quality thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkQualityThresholds {
    /// Excellent quality thresholds
    pub excellent: QualityThresholds,
    /// Good quality thresholds
    pub good: QualityThresholds,
    /// Fair quality thresholds
    pub fair: QualityThresholds,
    /// Poor quality thresholds
    pub poor: QualityThresholds,
}

/// Quality threshold values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Minimum bandwidth (Kbps)
    pub min_bandwidth_kbps: f64,
    /// Maximum latency (ms)
    pub max_latency_ms: f64,
    /// Maximum packet loss (%)
    pub max_packet_loss_percent: f64,
    /// Maximum jitter (ms)
    pub max_jitter_ms: f64,
}

/// Adaptive behavior configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveBehaviorConfig {
    /// Enable adaptive downloads
    pub enable_adaptive_downloads: bool,
    /// Enable adaptive model selection
    pub enable_adaptive_model_selection: bool,
    /// Enable adaptive caching
    pub enable_adaptive_caching: bool,
    /// Adaptation responsiveness (0.0-1.0)
    pub adaptation_responsiveness: f64,
    /// Stability window (seconds)
    pub stability_window_seconds: f64,
}

/// Download request for resumable downloads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResumableDownloadRequest {
    /// Unique download ID
    pub download_id: String,
    /// Source URL
    pub url: String,
    /// Destination path
    pub destination_path: String,
    /// Expected file size (bytes)
    pub expected_size: Option<usize>,
    /// Checksum for verification
    pub checksum: Option<String>,
    /// Download priority
    pub priority: DownloadPriority,
    /// Constraints
    pub constraints: DownloadConstraints,
}

/// Download priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DownloadPriority {
    /// Low priority
    Low = 1,
    /// Normal priority
    Normal = 2,
    /// High priority
    High = 3,
    /// Critical priority
    Critical = 4,
}

/// Download constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadConstraints {
    /// Only download on WiFi
    pub wifi_only: bool,
    /// Only download when charging
    pub charging_only: bool,
    /// Maximum bandwidth usage (Kbps)
    pub max_bandwidth_kbps: Option<f64>,
    /// Allowed time windows
    pub time_windows: Vec<TimeWindow>,
}

/// Time window for downloads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindow {
    /// Start hour (0-23)
    pub start_hour: usize,
    /// End hour (0-23)
    pub end_hour: usize,
    /// Days of week (0-6, Sunday=0)
    pub days_of_week: Vec<usize>,
}

/// Download progress information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    /// Download ID
    pub download_id: String,
    /// Bytes downloaded
    pub bytes_downloaded: usize,
    /// Total bytes
    pub total_bytes: usize,
    /// Download speed (Kbps)
    pub speed_kbps: f64,
    /// Estimated time remaining (seconds)
    pub eta_seconds: f64,
    /// Current status
    pub status: DownloadStatus,
    /// Error information (if any)
    pub error: Option<String>,
}

/// Download status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DownloadStatus {
    /// Download pending
    Pending,
    /// Download in progress
    InProgress,
    /// Download paused
    Paused,
    /// Download completed
    Completed,
    /// Download failed
    Failed,
    /// Download cancelled
    Cancelled,
}
