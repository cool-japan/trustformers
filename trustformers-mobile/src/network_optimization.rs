//! Network Optimization Features for TrustformeRS Mobile
//!
//! This module provides comprehensive network optimization capabilities including
//! resumable downloads, bandwidth-aware transfers, P2P model sharing, edge server
//! integration, and offline-first design patterns for mobile ML deployments.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use trustformers_core::error::{CoreError, Result};
use trustformers_core::TrustformersError;

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

/// Network Optimization Manager
pub struct NetworkOptimizationManager {
    config: NetworkOptimizationConfig,
    download_manager: Arc<Mutex<DownloadManager>>,
    p2p_manager: Arc<Mutex<P2PManager>>,
    edge_manager: Arc<Mutex<EdgeManager>>,
    quality_monitor: Arc<Mutex<NetworkQualityMonitor>>,
    offline_manager: Arc<Mutex<OfflineManager>>,
}

/// Download manager for resumable downloads
#[derive(Debug)]
struct DownloadManager {
    active_downloads: HashMap<String, ActiveDownload>,
    download_queue: std::collections::VecDeque<ResumableDownloadRequest>,
    download_history: HashMap<String, DownloadProgress>,
    bandwidth_monitor: BandwidthMonitor,
}

/// Active download tracking
#[derive(Debug, Clone)]
struct ActiveDownload {
    request: ResumableDownloadRequest,
    progress: DownloadProgress,
    start_time: std::time::Instant,
    last_checkpoint: usize,
    resume_data: Option<Vec<u8>>,
}

/// Bandwidth monitoring
#[derive(Debug, Clone)]
struct BandwidthMonitor {
    current_bandwidth_kbps: f64,
    average_bandwidth_kbps: f64,
    bandwidth_history: Vec<BandwidthSample>,
    last_measurement: std::time::Instant,
}

/// Bandwidth measurement sample
#[derive(Debug, Clone)]
struct BandwidthSample {
    timestamp: std::time::Instant,
    bandwidth_kbps: f64,
    connection_type: String,
}

/// P2P manager
#[derive(Debug)]
struct P2PManager {
    peer_connections: HashMap<String, PeerConnection>,
    shared_models: HashMap<String, SharedModel>,
    discovery_service: P2PDiscoveryService,
    security_manager: P2PSecurityManager,
}

/// Peer connection information
#[derive(Debug, Clone)]
struct PeerConnection {
    peer_id: String,
    address: String,
    connection_quality: f64,
    last_seen: std::time::Instant,
    shared_models: Vec<String>,
    trust_score: f64,
}

/// Shared model information
#[derive(Debug, Clone)]
struct SharedModel {
    model_id: String,
    model_hash: String,
    size_bytes: usize,
    availability_score: f64,
    peer_sources: Vec<String>,
}

/// P2P discovery service
#[derive(Debug)]
struct P2PDiscoveryService {
    discovered_peers: HashMap<String, PeerInfo>,
    discovery_protocol: P2PProtocol,
    last_discovery: std::time::Instant,
}

/// Peer information from discovery
#[derive(Debug, Clone)]
struct PeerInfo {
    peer_id: String,
    address: String,
    capabilities: Vec<String>,
    discovery_time: std::time::Instant,
}

/// P2P security manager
#[derive(Debug)]
struct P2PSecurityManager {
    trusted_peers: HashMap<String, TrustedPeer>,
    security_level: P2PSecurityLevel,
    encryption_keys: HashMap<String, Vec<u8>>,
}

/// Trusted peer information
#[derive(Debug, Clone)]
struct TrustedPeer {
    peer_id: String,
    public_key: Vec<u8>,
    trust_level: f64,
    last_verified: std::time::Instant,
}

/// Edge server manager
#[derive(Debug)]
struct EdgeManager {
    available_servers: HashMap<String, EdgeServerInfo>,
    current_server: Option<String>,
    load_balancer: EdgeLoadBalancer,
    health_monitor: EdgeHealthMonitor,
}

/// Edge server information
#[derive(Debug, Clone)]
struct EdgeServerInfo {
    endpoint: EdgeServerEndpoint,
    health_status: EdgeServerHealth,
    performance_metrics: EdgePerformanceMetrics,
    last_health_check: std::time::Instant,
}

/// Edge server health status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EdgeServerHealth {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Edge server performance metrics
#[derive(Debug, Clone)]
struct EdgePerformanceMetrics {
    average_latency_ms: f64,
    success_rate: f64,
    throughput_kbps: f64,
    load_percentage: f64,
}

/// Edge load balancer
#[derive(Debug)]
struct EdgeLoadBalancer {
    strategy: EdgeLoadBalancingStrategy,
    server_weights: HashMap<String, f64>,
    round_robin_index: usize,
}

/// Edge health monitor
#[derive(Debug)]
struct EdgeHealthMonitor {
    health_checks: HashMap<String, Vec<HealthCheckResult>>,
    monitoring_interval: std::time::Duration,
    last_check: std::time::Instant,
}

/// Health check result
#[derive(Debug, Clone)]
struct HealthCheckResult {
    timestamp: std::time::Instant,
    success: bool,
    latency_ms: f64,
    error_message: Option<String>,
}

/// Network quality monitor
#[derive(Debug)]
struct NetworkQualityMonitor {
    current_quality: NetworkQuality,
    quality_history: Vec<NetworkQualityMeasurement>,
    active_measurements: HashMap<NetworkMetric, f64>,
    last_measurement: std::time::Instant,
}

/// Network quality assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
enum NetworkQuality {
    Excellent,
    Good,
    Fair,
    Poor,
    Offline,
}

/// Network quality measurement
#[derive(Debug, Clone)]
struct NetworkQualityMeasurement {
    timestamp: std::time::Instant,
    quality: NetworkQuality,
    metrics: HashMap<NetworkMetric, f64>,
    connection_type: String,
}

/// Offline manager
#[derive(Debug)]
struct OfflineManager {
    offline_cache: HashMap<String, OfflineCacheEntry>,
    sync_queue: Vec<OfflineSyncItem>,
    fallback_models: HashMap<String, FallbackModelInfo>,
    last_online: Option<std::time::Instant>,
}

/// Offline cache entry
#[derive(Debug, Clone)]
struct OfflineCacheEntry {
    key: String,
    data: Vec<u8>,
    timestamp: std::time::Instant,
    expiry: Option<std::time::Instant>,
    size_bytes: usize,
}

/// Offline sync item
#[derive(Debug, Clone)]
struct OfflineSyncItem {
    item_id: String,
    sync_type: OfflineSyncType,
    priority: u8,
    created_at: std::time::Instant,
    retry_count: usize,
}

/// Offline sync types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OfflineSyncType {
    ModelUpdate,
    CacheSync,
    MetricsUpload,
    ConfigSync,
}

/// Fallback model information
#[derive(Debug, Clone)]
struct FallbackModelInfo {
    model_id: String,
    model_path: String,
    capabilities: Vec<String>,
    last_updated: std::time::Instant,
}

impl NetworkOptimizationManager {
    /// Create new network optimization manager
    pub fn new(config: NetworkOptimizationConfig) -> Result<Self> {
        config.validate()?;

        let download_manager = Arc::new(Mutex::new(DownloadManager::new(
            &config.download_optimization,
        )));
        let p2p_manager = Arc::new(Mutex::new(P2PManager::new(&config.p2p_config)));
        let edge_manager = Arc::new(Mutex::new(EdgeManager::new(&config.edge_config)));
        let quality_monitor = Arc::new(Mutex::new(NetworkQualityMonitor::new(
            &config.quality_monitoring,
        )));
        let offline_manager = Arc::new(Mutex::new(OfflineManager::new(&config.offline_first)));

        Ok(Self {
            config,
            download_manager,
            p2p_manager,
            edge_manager,
            quality_monitor,
            offline_manager,
        })
    }

    /// Start resumable download
    pub async fn start_resumable_download(
        &self,
        request: ResumableDownloadRequest,
    ) -> Result<String> {
        tracing::info!("Starting resumable download: {}", request.download_id);

        // Check if download is already in progress
        {
            let manager = self.download_manager.lock().expect("Operation failed");
            if manager.active_downloads.contains_key(&request.download_id) {
                return Err(TrustformersError::runtime_error(
                    "Download already in progress".into(),
                )
                .into());
            }
        }

        // Check constraints
        if !self.check_download_constraints(&request.constraints).await? {
            return Err(
                TrustformersError::runtime_error("Download constraints not met".into()).into(),
            );
        }

        // Add to download queue
        {
            let mut manager = self.download_manager.lock().expect("Operation failed");
            manager.enqueue_download(request.clone());
        }

        // Start download processing
        self.process_download_queue().await?;

        Ok(request.download_id)
    }

    /// Get download progress
    pub fn get_download_progress(&self, download_id: &str) -> Result<Option<DownloadProgress>> {
        let manager = self.download_manager.lock().expect("Operation failed");
        Ok(manager.get_download_progress(download_id))
    }

    /// Pause download
    pub async fn pause_download(&self, download_id: &str) -> Result<bool> {
        let mut manager = self.download_manager.lock().expect("Operation failed");
        manager.pause_download(download_id)
    }

    /// Resume download
    pub async fn resume_download(&self, download_id: &str) -> Result<bool> {
        let mut manager = self.download_manager.lock().expect("Operation failed");
        manager.resume_download(download_id)
    }

    /// Cancel download
    pub async fn cancel_download(&self, download_id: &str) -> Result<bool> {
        let mut manager = self.download_manager.lock().expect("Operation failed");
        manager.cancel_download(download_id)
    }

    /// Enable P2P model sharing
    pub async fn enable_p2p_sharing(&self, model_id: &str) -> Result<()> {
        if !self.config.enable_p2p_sharing {
            return Err(TrustformersError::config_error(
                "P2P sharing not enabled",
                "enable_p2p_sharing",
            )
            .into());
        }

        let mut p2p_manager = self.p2p_manager.lock().expect("Operation failed");
        p2p_manager.add_shared_model(model_id)?;

        tracing::info!("Enabled P2P sharing for model: {}", model_id);
        Ok(())
    }

    /// Discover P2P peers
    pub async fn discover_p2p_peers(&self) -> Result<Vec<String>> {
        if !self.config.enable_p2p_sharing {
            return Ok(Vec::new());
        }

        let mut p2p_manager = self.p2p_manager.lock().expect("Operation failed");
        let peers = p2p_manager.discover_peers().await?;

        Ok(peers)
    }

    /// Get optimal edge server
    pub async fn get_optimal_edge_server(&self) -> Result<Option<String>> {
        if !self.config.enable_edge_servers {
            return Ok(None);
        }

        let mut edge_manager = self.edge_manager.lock().expect("Operation failed");
        let server = edge_manager.select_optimal_server().await?;

        Ok(server)
    }

    /// Check network quality
    pub async fn check_network_quality(&self) -> Result<String> {
        let mut monitor = self.quality_monitor.lock().expect("Operation failed");
        let quality = monitor.measure_quality().await?;

        let quality_json = serde_json::json!({
            "quality": quality.quality,
            "metrics": quality.metrics,
            "connection_type": quality.connection_type,
            "timestamp": quality.timestamp.elapsed().as_secs()
        });

        Ok(quality_json.to_string())
    }

    /// Enter offline mode
    pub async fn enter_offline_mode(&self) -> Result<()> {
        if !self.config.offline_first.enable_offline_mode {
            return Err(TrustformersError::config_error(
                "Offline mode not enabled",
                "enter_offline_mode",
            )
            .into());
        }

        let mut offline_manager = self.offline_manager.lock().expect("Operation failed");
        offline_manager.enter_offline_mode().await?;

        tracing::info!("Entered offline mode");
        Ok(())
    }

    /// Exit offline mode and sync
    pub async fn exit_offline_mode(&self) -> Result<()> {
        let mut offline_manager = self.offline_manager.lock().expect("Operation failed");
        offline_manager.exit_offline_mode().await?;

        // Start synchronization
        self.sync_offline_data().await?;

        tracing::info!("Exited offline mode and started sync");
        Ok(())
    }

    /// Sync offline data
    pub async fn sync_offline_data(&self) -> Result<()> {
        let strategy = self.config.offline_first.sync_strategy;

        match strategy {
            OfflineSyncStrategy::Immediate => self.sync_immediate().await,
            OfflineSyncStrategy::Opportunistic => self.sync_opportunistic().await,
            OfflineSyncStrategy::Background => self.sync_background().await,
            OfflineSyncStrategy::Adaptive => self.sync_adaptive().await,
            OfflineSyncStrategy::Manual => Ok(()), // Manual sync requires explicit trigger
        }
    }

    /// Get network optimization statistics
    pub fn get_optimization_statistics(&self) -> Result<String> {
        let download_stats = {
            let manager = self.download_manager.lock().expect("Operation failed");
            manager.get_statistics()
        };

        let p2p_stats = {
            let manager = self.p2p_manager.lock().expect("Operation failed");
            manager.get_statistics()
        };

        let edge_stats = {
            let manager = self.edge_manager.lock().expect("Operation failed");
            manager.get_statistics()
        };

        let quality_stats = {
            let monitor = self.quality_monitor.lock().expect("Operation failed");
            monitor.get_statistics()
        };

        let stats_json = serde_json::json!({
            "download_manager": download_stats,
            "p2p_manager": p2p_stats,
            "edge_manager": edge_stats,
            "quality_monitor": quality_stats
        });

        Ok(stats_json.to_string())
    }

    // Private helper methods

    async fn check_download_constraints(&self, constraints: &DownloadConstraints) -> Result<bool> {
        // Check WiFi constraint
        if constraints.wifi_only && !self.is_wifi_connected() {
            return Ok(false);
        }

        // Check charging constraint
        if constraints.charging_only && !self.is_device_charging() {
            return Ok(false);
        }

        // Check bandwidth constraint
        if let Some(max_bandwidth) = constraints.max_bandwidth_kbps {
            let current_bandwidth = self.get_current_bandwidth().await;
            if current_bandwidth > max_bandwidth {
                return Ok(false);
            }
        }

        // Check time windows
        if !constraints.time_windows.is_empty() {
            let current_time = self.get_current_time_info();
            let in_allowed_window = constraints
                .time_windows
                .iter()
                .any(|window| self.is_time_in_window(&current_time, window));
            if !in_allowed_window {
                return Ok(false);
            }
        }

        Ok(true)
    }

    async fn process_download_queue(&self) -> Result<()> {
        let mut manager = self.download_manager.lock().expect("Operation failed");
        manager.process_queue().await
    }

    async fn sync_immediate(&self) -> Result<()> {
        // Implement immediate sync
        Ok(())
    }

    async fn sync_opportunistic(&self) -> Result<()> {
        // Implement opportunistic sync
        Ok(())
    }

    async fn sync_background(&self) -> Result<()> {
        // Implement background sync
        Ok(())
    }

    async fn sync_adaptive(&self) -> Result<()> {
        // Implement adaptive sync based on network conditions
        let quality = {
            let monitor = self.quality_monitor.lock().expect("Operation failed");
            monitor.current_quality
        };

        match quality {
            NetworkQuality::Excellent | NetworkQuality::Good => self.sync_immediate().await,
            NetworkQuality::Fair => self.sync_opportunistic().await,
            NetworkQuality::Poor => self.sync_background().await,
            NetworkQuality::Offline => Ok(()),
        }
    }

    fn is_wifi_connected(&self) -> bool {
        // Platform-specific WiFi detection
        true // Placeholder
    }

    fn is_device_charging(&self) -> bool {
        // Platform-specific charging detection
        false // Placeholder
    }

    async fn get_current_bandwidth(&self) -> f64 {
        let manager = self.download_manager.lock().expect("Operation failed");
        manager.bandwidth_monitor.current_bandwidth_kbps
    }

    fn get_current_time_info(&self) -> CurrentTimeInfo {
        CurrentTimeInfo {
            hour: 12,       // Placeholder
            day_of_week: 1, // Placeholder
        }
    }

    fn is_time_in_window(&self, time: &CurrentTimeInfo, window: &TimeWindow) -> bool {
        // Check if current time is within allowed window
        let hour_in_range = if window.start_hour <= window.end_hour {
            time.hour >= window.start_hour && time.hour <= window.end_hour
        } else {
            // Wrap around midnight
            time.hour >= window.start_hour || time.hour <= window.end_hour
        };

        let day_allowed =
            window.days_of_week.is_empty() || window.days_of_week.contains(&time.day_of_week);

        hour_in_range && day_allowed
    }
}

/// Current time information
struct CurrentTimeInfo {
    hour: usize,
    day_of_week: usize,
}

// Implementation details for helper structs

impl DownloadManager {
    fn new(config: &DownloadOptimizationConfig) -> Self {
        Self {
            active_downloads: HashMap::new(),
            download_queue: std::collections::VecDeque::new(),
            download_history: HashMap::new(),
            bandwidth_monitor: BandwidthMonitor::new(),
        }
    }

    fn enqueue_download(&mut self, request: ResumableDownloadRequest) {
        self.download_queue.push_back(request);
    }

    fn get_download_progress(&self, download_id: &str) -> Option<DownloadProgress> {
        self.active_downloads
            .get(download_id)
            .map(|download| download.progress.clone())
            .or_else(|| self.download_history.get(download_id).cloned())
    }

    fn pause_download(&mut self, download_id: &str) -> Result<bool> {
        if let Some(download) = self.active_downloads.get_mut(download_id) {
            download.progress.status = DownloadStatus::Paused;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn resume_download(&mut self, download_id: &str) -> Result<bool> {
        if let Some(download) = self.active_downloads.get_mut(download_id) {
            if download.progress.status == DownloadStatus::Paused {
                download.progress.status = DownloadStatus::InProgress;
                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            Ok(false)
        }
    }

    fn cancel_download(&mut self, download_id: &str) -> Result<bool> {
        if let Some(mut download) = self.active_downloads.remove(download_id) {
            download.progress.status = DownloadStatus::Cancelled;
            self.download_history.insert(download_id.to_string(), download.progress);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    async fn process_queue(&mut self) -> Result<()> {
        // Process download queue
        while let Some(request) = self.download_queue.pop_front() {
            if self.active_downloads.len() < 3 {
                // Max concurrent downloads
                self.start_download(request).await?;
            } else {
                // Put back in queue
                self.download_queue.push_front(request);
                break;
            }
        }
        Ok(())
    }

    async fn start_download(&mut self, request: ResumableDownloadRequest) -> Result<()> {
        let progress = DownloadProgress {
            download_id: request.download_id.clone(),
            bytes_downloaded: 0,
            total_bytes: request.expected_size.unwrap_or(0),
            speed_kbps: 0.0,
            eta_seconds: 0.0,
            status: DownloadStatus::InProgress,
            error: None,
        };

        let active_download = ActiveDownload {
            request: request.clone(),
            progress,
            start_time: std::time::Instant::now(),
            last_checkpoint: 0,
            resume_data: None,
        };

        self.active_downloads.insert(request.download_id.clone(), active_download);
        Ok(())
    }

    fn get_statistics(&self) -> serde_json::Value {
        serde_json::json!({
            "active_downloads": self.active_downloads.len(),
            "queued_downloads": self.download_queue.len(),
            "completed_downloads": self.download_history.len(),
            "current_bandwidth_kbps": self.bandwidth_monitor.current_bandwidth_kbps
        })
    }
}

impl BandwidthMonitor {
    fn new() -> Self {
        Self {
            current_bandwidth_kbps: 0.0,
            average_bandwidth_kbps: 0.0,
            bandwidth_history: Vec::new(),
            last_measurement: std::time::Instant::now(),
        }
    }
}

impl P2PManager {
    fn new(config: &P2PConfig) -> Self {
        Self {
            peer_connections: HashMap::new(),
            shared_models: HashMap::new(),
            discovery_service: P2PDiscoveryService::new(config.protocol),
            security_manager: P2PSecurityManager::new(&config.security),
        }
    }

    fn add_shared_model(&mut self, model_id: &str) -> Result<()> {
        // Add model to shared models
        let shared_model = SharedModel {
            model_id: model_id.to_string(),
            model_hash: "placeholder_hash".to_string(),
            size_bytes: 1024 * 1024, // Placeholder
            availability_score: 1.0,
            peer_sources: Vec::new(),
        };

        self.shared_models.insert(model_id.to_string(), shared_model);
        Ok(())
    }

    async fn discover_peers(&mut self) -> Result<Vec<String>> {
        self.discovery_service.discover_peers().await
    }

    fn get_statistics(&self) -> serde_json::Value {
        serde_json::json!({
            "connected_peers": self.peer_connections.len(),
            "shared_models": self.shared_models.len(),
            "discovery_protocol": self.discovery_service.discovery_protocol
        })
    }
}

impl P2PDiscoveryService {
    fn new(protocol: P2PProtocol) -> Self {
        Self {
            discovered_peers: HashMap::new(),
            discovery_protocol: protocol,
            last_discovery: std::time::Instant::now(),
        }
    }

    async fn discover_peers(&mut self) -> Result<Vec<String>> {
        // Implement peer discovery
        Ok(Vec::new())
    }
}

impl P2PSecurityManager {
    fn new(config: &P2PSecurityConfig) -> Self {
        Self {
            trusted_peers: HashMap::new(),
            security_level: config.security_level,
            encryption_keys: HashMap::new(),
        }
    }
}

impl EdgeManager {
    fn new(config: &EdgeServerConfig) -> Self {
        Self {
            available_servers: HashMap::new(),
            current_server: None,
            load_balancer: EdgeLoadBalancer::new(config.load_balancing),
            health_monitor: EdgeHealthMonitor::new(),
        }
    }

    async fn select_optimal_server(&mut self) -> Result<Option<String>> {
        self.load_balancer.select_server(&self.available_servers)
    }

    fn get_statistics(&self) -> serde_json::Value {
        serde_json::json!({
            "available_servers": self.available_servers.len(),
            "current_server": self.current_server,
            "load_balancing_strategy": self.load_balancer.strategy
        })
    }
}

impl EdgeLoadBalancer {
    fn new(strategy: EdgeLoadBalancingStrategy) -> Self {
        Self {
            strategy,
            server_weights: HashMap::new(),
            round_robin_index: 0,
        }
    }

    fn select_server(
        &mut self,
        servers: &HashMap<String, EdgeServerInfo>,
    ) -> Result<Option<String>> {
        if servers.is_empty() {
            return Ok(None);
        }

        match self.strategy {
            EdgeLoadBalancingStrategy::RoundRobin => {
                let server_ids: Vec<_> = servers.keys().collect();
                if server_ids.is_empty() {
                    return Ok(None);
                }
                let selected = server_ids[self.round_robin_index % server_ids.len()];
                self.round_robin_index += 1;
                Ok(Some(selected.clone()))
            },
            EdgeLoadBalancingStrategy::LowestLatency => {
                // Select server with lowest latency
                let best_server = servers
                    .iter()
                    .min_by(|(_, a), (_, b)| {
                        a.performance_metrics
                            .average_latency_ms
                            .partial_cmp(&b.performance_metrics.average_latency_ms)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(id, _)| id.clone());
                Ok(best_server)
            },
            _ => {
                // Implement other strategies
                Ok(servers.keys().next().cloned())
            },
        }
    }
}

impl EdgeHealthMonitor {
    fn new() -> Self {
        Self {
            health_checks: HashMap::new(),
            monitoring_interval: std::time::Duration::from_secs(30),
            last_check: std::time::Instant::now(),
        }
    }
}

impl NetworkQualityMonitor {
    fn new(config: &NetworkQualityConfig) -> Self {
        Self {
            current_quality: NetworkQuality::Good,
            quality_history: Vec::new(),
            active_measurements: HashMap::new(),
            last_measurement: std::time::Instant::now(),
        }
    }

    async fn measure_quality(&mut self) -> Result<NetworkQualityMeasurement> {
        // Implement network quality measurement
        let measurement = NetworkQualityMeasurement {
            timestamp: std::time::Instant::now(),
            quality: NetworkQuality::Good,
            metrics: HashMap::new(),
            connection_type: "WiFi".to_string(),
        };

        self.quality_history.push(measurement.clone());
        self.current_quality = measurement.quality;

        Ok(measurement)
    }

    fn get_statistics(&self) -> serde_json::Value {
        serde_json::json!({
            "current_quality": self.current_quality,
            "measurement_count": self.quality_history.len(),
            "last_measurement_elapsed_ms": self.last_measurement.elapsed().as_millis() as u64
        })
    }
}

impl OfflineManager {
    fn new(config: &OfflineFirstConfig) -> Self {
        Self {
            offline_cache: HashMap::new(),
            sync_queue: Vec::new(),
            fallback_models: HashMap::new(),
            last_online: Some(std::time::Instant::now()),
        }
    }

    async fn enter_offline_mode(&mut self) -> Result<()> {
        self.last_online = Some(std::time::Instant::now());
        // Prepare offline cache and fallback models
        Ok(())
    }

    async fn exit_offline_mode(&mut self) -> Result<()> {
        // Prepare for online sync
        Ok(())
    }
}

impl Default for NetworkOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_resumable_downloads: true,
            enable_bandwidth_awareness: true,
            enable_p2p_sharing: false, // Disabled by default for security
            enable_edge_servers: true,
            offline_first: OfflineFirstConfig {
                enable_offline_mode: true,
                offline_cache_size_mb: 500,
                fallback_models: vec!["lightweight_model".to_string()],
                sync_strategy: OfflineSyncStrategy::Adaptive,
                offline_retention: OfflineRetentionPolicy {
                    model_retention_days: 7,
                    cache_retention_hours: 24,
                    auto_cleanup_on_low_storage: true,
                    min_storage_threshold_mb: 100,
                },
            },
            download_optimization: DownloadOptimizationConfig {
                chunk_size_kb: 1024,
                max_concurrent_downloads: 3,
                download_timeout_seconds: 300.0,
                retry_config: DownloadRetryConfig {
                    max_retries: 3,
                    initial_delay_ms: 1000.0,
                    max_delay_ms: 30000.0,
                    backoff_multiplier: 2.0,
                    jitter_factor: 0.1,
                },
                compression: DownloadCompressionConfig {
                    enable_compression: true,
                    preferred_algorithms: vec![
                        CompressionAlgorithm::Brotli,
                        CompressionAlgorithm::Gzip,
                        CompressionAlgorithm::LZ4,
                    ],
                    min_size_for_compression: 1024,
                    enable_streaming_decompression: true,
                },
                bandwidth_adaptation: BandwidthAdaptationConfig {
                    enable_auto_detection: true,
                    monitoring_interval_seconds: 10.0,
                    adaptation_thresholds: BandwidthThresholds {
                        low_bandwidth_kbps: 100.0,
                        medium_bandwidth_kbps: 1000.0,
                        high_bandwidth_kbps: 10000.0,
                        ultra_high_bandwidth_kbps: 100000.0,
                    },
                    quality_adaptation: QualityAdaptationConfig {
                        enable_dynamic_quality: true,
                        quality_levels: HashMap::new(),
                        adaptation_strategy: QualityAdaptationStrategy::Balanced,
                    },
                },
            },
            p2p_config: P2PConfig {
                enable_discovery: false,
                protocol: P2PProtocol::Hybrid,
                max_peers: 10,
                security: P2PSecurityConfig {
                    enable_encryption: true,
                    enable_peer_authentication: true,
                    trusted_peers: Vec::new(),
                    enable_content_verification: true,
                    security_level: P2PSecurityLevel::Standard,
                },
                sharing_policy: P2PSharingPolicy {
                    shareable_models: Vec::new(),
                    max_upload_bandwidth_kbps: 1000.0,
                    time_restrictions: P2PTimeRestrictions {
                        enable_restrictions: false,
                        allowed_hours: (0..24).collect(),
                        allowed_days: (0..7).collect(),
                        timezone: "UTC".to_string(),
                    },
                    battery_aware_sharing: true,
                    network_aware_sharing: true,
                },
                resource_limits: P2PResourceLimits {
                    max_cpu_usage_percent: 20.0,
                    max_memory_usage_mb: 100,
                    max_storage_mb: 500,
                    max_connections: 10,
                },
            },
            edge_config: EdgeServerConfig {
                enable_discovery: true,
                server_endpoints: Vec::new(),
                load_balancing: EdgeLoadBalancingStrategy::LowestLatency,
                failover: EdgeFailoverConfig {
                    enable_auto_failover: true,
                    health_check_interval_seconds: 30.0,
                    failure_threshold: 3,
                    recovery_check_interval_seconds: 60.0,
                    failover_timeout_seconds: 10.0,
                },
                caching: EdgeCachingConfig {
                    enable_caching: true,
                    cache_ttl_hours: 24.0,
                    max_cache_size_mb: 1000,
                    eviction_strategy: CacheEvictionStrategy::LRU,
                },
            },
            quality_monitoring: NetworkQualityConfig {
                enable_continuous_monitoring: true,
                monitoring_interval_seconds: 30.0,
                tracked_metrics: vec![
                    NetworkMetric::BandwidthDown,
                    NetworkMetric::BandwidthUp,
                    NetworkMetric::Latency,
                    NetworkMetric::PacketLoss,
                ],
                quality_thresholds: NetworkQualityThresholds {
                    excellent: QualityThresholds {
                        min_bandwidth_kbps: 10000.0,
                        max_latency_ms: 50.0,
                        max_packet_loss_percent: 0.1,
                        max_jitter_ms: 10.0,
                    },
                    good: QualityThresholds {
                        min_bandwidth_kbps: 1000.0,
                        max_latency_ms: 100.0,
                        max_packet_loss_percent: 1.0,
                        max_jitter_ms: 25.0,
                    },
                    fair: QualityThresholds {
                        min_bandwidth_kbps: 100.0,
                        max_latency_ms: 300.0,
                        max_packet_loss_percent: 5.0,
                        max_jitter_ms: 50.0,
                    },
                    poor: QualityThresholds {
                        min_bandwidth_kbps: 10.0,
                        max_latency_ms: 1000.0,
                        max_packet_loss_percent: 10.0,
                        max_jitter_ms: 100.0,
                    },
                },
                adaptive_behavior: AdaptiveBehaviorConfig {
                    enable_adaptive_downloads: true,
                    enable_adaptive_model_selection: true,
                    enable_adaptive_caching: true,
                    adaptation_responsiveness: 0.5,
                    stability_window_seconds: 60.0,
                },
            },
        }
    }
}

impl NetworkOptimizationConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.download_optimization.max_concurrent_downloads == 0 {
            return Err(TrustformersError::config_error(
                "Max concurrent downloads must be > 0",
                "validate",
            )
            .into());
        }

        if self.download_optimization.max_concurrent_downloads > 10 {
            return Err(TrustformersError::config_error(
                "Too many concurrent downloads",
                "validate",
            )
            .into());
        }

        if self.offline_first.offline_cache_size_mb < 50 {
            return Err(TrustformersError::config_error(
                "Offline cache size too small",
                "validate",
            )
            .into());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_optimization_config_default() {
        let config = NetworkOptimizationConfig::default();
        assert!(config.enable_resumable_downloads);
        assert!(config.enable_bandwidth_awareness);
        assert!(!config.enable_p2p_sharing); // Should be disabled by default
        assert!(config.enable_edge_servers);
    }

    #[test]
    fn test_network_optimization_config_validation() {
        let mut config = NetworkOptimizationConfig::default();
        assert!(config.validate().is_ok());

        config.download_optimization.max_concurrent_downloads = 0;
        assert!(config.validate().is_err());

        config.download_optimization.max_concurrent_downloads = 15;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_download_priority_ordering() {
        assert!(DownloadPriority::Critical > DownloadPriority::High);
        assert!(DownloadPriority::High > DownloadPriority::Normal);
        assert!(DownloadPriority::Normal > DownloadPriority::Low);
    }

    #[tokio::test]
    async fn test_network_optimization_manager_creation() {
        let config = NetworkOptimizationConfig::default();
        let result = NetworkOptimizationManager::new(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_bandwidth_thresholds() {
        let thresholds = BandwidthThresholds {
            low_bandwidth_kbps: 100.0,
            medium_bandwidth_kbps: 1000.0,
            high_bandwidth_kbps: 10000.0,
            ultra_high_bandwidth_kbps: 100000.0,
        };

        assert!(thresholds.ultra_high_bandwidth_kbps > thresholds.high_bandwidth_kbps);
        assert!(thresholds.high_bandwidth_kbps > thresholds.medium_bandwidth_kbps);
        assert!(thresholds.medium_bandwidth_kbps > thresholds.low_bandwidth_kbps);
    }
}
