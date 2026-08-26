//! Network Optimization Features for TrustformeRS Mobile
//!
//! This module provides comprehensive network optimization capabilities including
//! resumable downloads, bandwidth-aware transfers, P2P model sharing, edge server
//! integration, and offline-first design patterns for mobile ML deployments.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use trustformers_core::error::{CoreError, Result};
use trustformers_core::TrustformersError;

#[path = "network_optimization_types.rs"]
mod types;
pub use types::*;

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
            let manager = self.download_manager.lock().unwrap_or_else(|p| p.into_inner());
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
            let mut manager = self.download_manager.lock().unwrap_or_else(|p| p.into_inner());
            manager.enqueue_download(request.clone());
        }

        // Start download processing
        self.process_download_queue().await?;

        Ok(request.download_id)
    }

    /// Get download progress
    pub fn get_download_progress(&self, download_id: &str) -> Result<Option<DownloadProgress>> {
        let manager = self.download_manager.lock().unwrap_or_else(|p| p.into_inner());
        Ok(manager.get_download_progress(download_id))
    }

    /// Pause download
    pub async fn pause_download(&self, download_id: &str) -> Result<bool> {
        let mut manager = self.download_manager.lock().unwrap_or_else(|p| p.into_inner());
        manager.pause_download(download_id)
    }

    /// Resume download
    pub async fn resume_download(&self, download_id: &str) -> Result<bool> {
        let mut manager = self.download_manager.lock().unwrap_or_else(|p| p.into_inner());
        manager.resume_download(download_id)
    }

    /// Cancel download
    pub async fn cancel_download(&self, download_id: &str) -> Result<bool> {
        let mut manager = self.download_manager.lock().unwrap_or_else(|p| p.into_inner());
        manager.cancel_download(download_id)
    }

    /// Enable P2P model sharing for the model file at `model_path`.
    ///
    /// Takes a path (not bytes): this manager has no `ModelManager` of its
    /// own to resolve `model_id` to a location or to hold pre-loaded bytes,
    /// and a shared model can be hundreds of megabytes, so the caller passes
    /// where it already lives and `add_shared_model` streams it.
    pub async fn enable_p2p_sharing(&self, model_id: &str, model_path: &Path) -> Result<()> {
        if !self.config.enable_p2p_sharing {
            return Err(TrustformersError::config_error(
                "P2P sharing not enabled",
                "enable_p2p_sharing",
            )
            .into());
        }

        let mut p2p_manager = self.p2p_manager.lock().unwrap_or_else(|p| p.into_inner());
        p2p_manager.add_shared_model(model_id, model_path)?;

        tracing::info!("Enabled P2P sharing for model: {}", model_id);
        Ok(())
    }

    /// Discover P2P peers
    pub async fn discover_p2p_peers(&self) -> Result<Vec<String>> {
        if !self.config.enable_p2p_sharing {
            return Ok(Vec::new());
        }

        let mut p2p_manager = self.p2p_manager.lock().unwrap_or_else(|p| p.into_inner());
        let peers = p2p_manager.discover_peers().await?;

        Ok(peers)
    }

    /// Get optimal edge server
    pub async fn get_optimal_edge_server(&self) -> Result<Option<String>> {
        if !self.config.enable_edge_servers {
            return Ok(None);
        }

        let mut edge_manager = self.edge_manager.lock().unwrap_or_else(|p| p.into_inner());
        let server = edge_manager.select_optimal_server().await?;

        Ok(server)
    }

    /// Check network quality
    pub async fn check_network_quality(&self) -> Result<String> {
        let mut monitor = self.quality_monitor.lock().unwrap_or_else(|p| p.into_inner());
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

        let mut offline_manager = self.offline_manager.lock().unwrap_or_else(|p| p.into_inner());
        offline_manager.enter_offline_mode().await?;

        tracing::info!("Entered offline mode");
        Ok(())
    }

    /// Exit offline mode and sync
    pub async fn exit_offline_mode(&self) -> Result<()> {
        let mut offline_manager = self.offline_manager.lock().unwrap_or_else(|p| p.into_inner());
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
            let manager = self.download_manager.lock().unwrap_or_else(|p| p.into_inner());
            manager.get_statistics()
        };

        let p2p_stats = {
            let manager = self.p2p_manager.lock().unwrap_or_else(|p| p.into_inner());
            manager.get_statistics()
        };

        let edge_stats = {
            let manager = self.edge_manager.lock().unwrap_or_else(|p| p.into_inner());
            manager.get_statistics()
        };

        let quality_stats = {
            let monitor = self.quality_monitor.lock().unwrap_or_else(|p| p.into_inner());
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
        // Check WiFi constraint. `is_wifi_connected` reports `None` when
        // connectivity cannot be verified on this platform; a `wifi_only`
        // download must fail closed in that case (refuse, with a
        // structured error naming exactly what could not be checked)
        // rather than silently proceed on a connection that might be
        // metered cellular -- the previous `true // Placeholder` let every
        // `wifi_only` download through unconditionally.
        if constraints.wifi_only {
            match self.is_wifi_connected() {
                Some(true) => {},
                Some(false) => return Ok(false),
                None => {
                    return Err(TrustformersError::runtime_error(
                        "wifi_only download requested but WiFi connectivity cannot be verified \
                         on this platform; refusing rather than risk downloading over a \
                         metered connection"
                            .into(),
                    )
                    .into());
                },
            }
        }

        // Check charging constraint. Same fail-closed shape: a
        // `charging_only` download that can never be evaluated must say so
        // explicitly rather than collapse into the same `Ok(false)` an
        // ordinary "not currently charging" evaluation returns -- the
        // previous `false // Placeholder` made every `charging_only`
        // download return that same silent `Ok(false)` forever, with no
        // way for a caller to tell "not right now" apart from "this can
        // never be checked on this build".
        if constraints.charging_only {
            match self.is_device_charging() {
                Some(true) => {},
                Some(false) => return Ok(false),
                None => {
                    return Err(TrustformersError::runtime_error(
                        "charging_only download requested but charging state cannot be \
                         verified on this platform; refusing rather than risk draining the \
                         battery"
                            .into(),
                    )
                    .into());
                },
            }
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
        let mut manager = self.download_manager.lock().unwrap_or_else(|p| p.into_inner());
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
            let monitor = self.quality_monitor.lock().unwrap_or_else(|p| p.into_inner());
            monitor.current_quality
        };

        match quality {
            NetworkQuality::Excellent | NetworkQuality::Good => self.sync_immediate().await,
            NetworkQuality::Fair => self.sync_opportunistic().await,
            NetworkQuality::Poor => self.sync_background().await,
            NetworkQuality::Offline => Ok(()),
        }
    }

    /// `None` when WiFi connectivity cannot be verified on this platform
    /// (always, today -- no platform-specific WiFi detector is wired up
    /// yet). Previously `true` unconditionally, which made every
    /// `wifi_only` constraint check pass regardless of the real
    /// connection. Reporting `None` rather than guessing lets
    /// `check_download_constraints` fail closed instead.
    fn is_wifi_connected(&self) -> Option<bool> {
        None
    }

    /// `None` when charging state cannot be verified on this platform
    /// (always, today). Previously `false` unconditionally, which
    /// collapsed "verified not charging" and "cannot tell" into the same
    /// silent `Ok(false)` from `check_download_constraints`. Reporting
    /// `None` lets that caller distinguish the two and fail closed with a
    /// structured error for the latter.
    fn is_device_charging(&self) -> Option<bool> {
        None
    }

    async fn get_current_bandwidth(&self) -> f64 {
        let manager = self.download_manager.lock().unwrap_or_else(|p| p.into_inner());
        manager.bandwidth_monitor.current_bandwidth_kbps
    }

    /// Real local wall-clock hour/weekday via `chrono` (already a workspace
    /// dependency, the same crate `trustformers-serve`'s cache warmer uses
    /// for the identical "which hour is it right now" question). Previously
    /// a hardcoded `hour: 12, day_of_week: 1` regardless of the actual
    /// time -- on the live `check_download_constraints` path
    /// (`is_time_in_window`, below), that made a `time_windows`-restricted
    /// download (e.g. "only overnight, 2am-6am") either always allowed or
    /// always refused depending on whether noon-Monday happened to fall
    /// inside the configured window, never actually gated by the real
    /// clock. `day_of_week` uses the `0..=6, Sunday=0` convention this
    /// struct's own doc comment (and `TimeWindow::days_of_week`) already
    /// declares, matching `chrono::Weekday::num_days_from_sunday()`.
    fn get_current_time_info(&self) -> CurrentTimeInfo {
        use chrono::{Datelike, Timelike};

        let now = chrono::Local::now();
        CurrentTimeInfo {
            hour: now.hour() as usize,
            day_of_week: now.weekday().num_days_from_sunday() as usize,
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

    /// Real, content-derived hash and size for the model file at
    /// `model_path`, streamed through `sha2::Sha256` in fixed-size chunks
    /// (never loaded fully into memory) and hex-encoded -- same crates and
    /// idiom as `nnapi_converter.rs::compute_model_hash`. Errors (via
    /// `io::Error`'s `TrustformersError` conversion) instead of recording a
    /// fabricated hash/size when `model_path` cannot be opened or read.
    fn add_shared_model(&mut self, model_id: &str, model_path: &Path) -> Result<()> {
        use sha2::{Digest, Sha256};
        use std::io::Read;

        let mut file = std::fs::File::open(model_path).map_err(TrustformersError::from)?;
        let mut hasher = Sha256::new();
        let mut size_bytes: u64 = 0;
        let mut chunk = [0u8; 65536];
        loop {
            let read = file.read(&mut chunk).map_err(TrustformersError::from)?;
            if read == 0 {
                break;
            }
            hasher.update(&chunk[..read]);
            size_bytes += read as u64;
        }
        let model_hash = hex::encode(hasher.finalize());

        // Add model to shared models
        let shared_model = SharedModel {
            model_id: model_id.to_string(),
            model_hash,
            // Real files fit within `usize::MAX` on every target platform.
            size_bytes: size_bytes as usize,
            // Locally available the instant it is shared (this point is only
            // reached after reading `model_path` in full above) -- a real
            // fact about local possession, not a network replication score;
            // no peer has received it yet, hence `peer_sources` below.
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
#[path = "network_optimization_tests.rs"]
mod tests;
