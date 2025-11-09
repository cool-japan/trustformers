use crate::error::{Result, TrustformersError};
use reqwest;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime};
use tokio::fs as async_fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::time::{interval, sleep};
use tokio_stream::StreamExt;
use trustformers_core::errors::TrustformersError as CoreTrustformersError;

/// Local hub mirror for caching and serving TrustformeRS models locally
/// Provides offline access, bandwidth optimization, and fast local model loading

/// Mirror configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirrorConfig {
    pub storage_path: PathBuf,
    pub remote_hub_url: String,
    pub sync_interval: Duration,
    pub max_storage_size_gb: f64,
    pub compression_enabled: bool,
    pub auto_cleanup: bool,
    pub cleanup_threshold: f64, // When to trigger cleanup (0.0-1.0 of max size)
    pub parallel_downloads: usize,
    pub retry_attempts: u32,
    pub retry_delay: Duration,
    pub bandwidth_limit_mbps: Option<f64>,
    pub priority_models: Vec<String>, // Models to always keep cached
}

impl Default for MirrorConfig {
    fn default() -> Self {
        Self {
            storage_path: PathBuf::from("./hub_mirror"),
            remote_hub_url: "https://hub.trustformers.ai".to_string(),
            sync_interval: Duration::from_secs(3600), // 1 hour
            max_storage_size_gb: 100.0,
            compression_enabled: true,
            auto_cleanup: true,
            cleanup_threshold: 0.8,
            parallel_downloads: 4,
            retry_attempts: 3,
            retry_delay: Duration::from_secs(5),
            bandwidth_limit_mbps: None,
            priority_models: Vec::new(),
        }
    }
}

/// Cached model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedModel {
    pub model_id: String,
    pub version: String,
    pub local_path: PathBuf,
    pub remote_url: String,
    pub cached_at: SystemTime,
    pub last_accessed: SystemTime,
    pub access_count: u64,
    pub file_size: u64,
    pub checksum: String,
    pub metadata: ModelMetadata,
    pub is_priority: bool,
    pub download_complete: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub description: Option<String>,
    pub architecture: String,
    pub task: String,
    pub language: Option<String>,
    pub license: Option<String>,
    pub tags: Vec<String>,
    pub performance_metrics: HashMap<String, f64>,
    pub size_mb: f64,
    pub dependencies: Vec<String>,
}

/// Mirror statistics and metrics
#[derive(Debug, Clone, Default)]
pub struct MirrorStats {
    pub total_models: usize,
    pub total_size_gb: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub downloads_completed: u64,
    pub downloads_failed: u64,
    pub last_sync: Option<SystemTime>,
    pub sync_errors: u64,
    pub bandwidth_saved_gb: f64,
    pub average_download_speed_mbps: f64,
}

/// Download progress tracking
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub model_id: String,
    pub version: String,
    pub bytes_downloaded: u64,
    pub total_bytes: u64,
    pub progress_percent: f64,
    pub download_speed_mbps: f64,
    pub eta_seconds: u64,
    pub status: DownloadStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DownloadStatus {
    Queued,
    Downloading,
    Completed,
    Failed(String),
    Cancelled,
}

/// Hub mirror implementation
pub struct HubMirror {
    config: MirrorConfig,
    cache: Arc<RwLock<HashMap<String, CachedModel>>>,
    stats: Arc<Mutex<MirrorStats>>,
    download_queue: Arc<RwLock<HashMap<String, DownloadProgress>>>,
    http_client: reqwest::Client,
    sync_handle: Option<tokio::task::JoinHandle<()>>,
}

impl HubMirror {
    pub fn new(config: MirrorConfig) -> Result<Self> {
        // Create storage directory
        fs::create_dir_all(&config.storage_path).map_err(|e| {
            TrustformersError::Core(CoreTrustformersError::other(format!(
                "Failed to create storage directory: {}",
                e
            )))
        })?;

        // Create HTTP client with optional bandwidth limiting
        let client_builder = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .user_agent("TrustformeRS-Mirror/1.0");

        if let Some(bandwidth_limit) = config.bandwidth_limit_mbps {
            // In a real implementation, we'd configure bandwidth limiting here
            tracing::info!("Bandwidth limit set to {} Mbps", bandwidth_limit);
        }

        let http_client = client_builder.build().map_err(|e| {
            TrustformersError::Core(CoreTrustformersError::other(format!(
                "Failed to create HTTP client: {}",
                e
            )))
        })?;

        let mirror = Self {
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(Mutex::new(MirrorStats::default())),
            download_queue: Arc::new(RwLock::new(HashMap::new())),
            http_client,
            sync_handle: None,
        };

        Ok(mirror)
    }

    /// Initialize the mirror by loading existing cache and starting background tasks
    pub async fn initialize(&mut self) -> Result<()> {
        self.load_cache().await?;
        self.start_background_sync().await?;
        self.cleanup_if_needed().await?;

        tracing::info!(
            "Hub mirror initialized with {} cached models",
            self.cache.read().unwrap().len()
        );
        Ok(())
    }

    /// Get a model from cache or download it
    pub async fn get_model(&self, model_id: &str, version: Option<&str>) -> Result<PathBuf> {
        let version = version.unwrap_or("latest");
        let cache_key = format!("{}:{}", model_id, version);

        // Check cache first
        {
            let mut cache = self.cache.write().unwrap();
            if let Some(cached_model) = cache.get_mut(&cache_key) {
                if cached_model.download_complete && cached_model.local_path.exists() {
                    // Update access statistics
                    cached_model.last_accessed = SystemTime::now();
                    cached_model.access_count += 1;

                    let mut stats = self.stats.lock().unwrap();
                    stats.cache_hits += 1;

                    tracing::debug!("Cache hit for {}:{}", model_id, version);
                    return Ok(cached_model.local_path.clone());
                }
            }
        }

        // Cache miss - download the model
        {
            let mut stats = self.stats.lock().unwrap();
            stats.cache_misses += 1;
        }

        tracing::info!("Cache miss for {}:{}, downloading...", model_id, version);
        self.download_model(model_id, version).await
    }

    /// Download a model from the remote hub
    async fn download_model(&self, model_id: &str, version: &str) -> Result<PathBuf> {
        let cache_key = format!("{}:{}", model_id, version);

        // Check if already downloading
        {
            let queue = self.download_queue.read().unwrap();
            if let Some(progress) = queue.get(&cache_key) {
                if progress.status == DownloadStatus::Downloading {
                    // Wait for existing download to complete
                    return self.wait_for_download(&cache_key).await;
                }
            }
        }

        // Start new download
        let download_url = format!(
            "{}/models/{}/versions/{}/download",
            self.config.remote_hub_url, model_id, version
        );

        // Get model metadata first
        let metadata = self.fetch_model_metadata(model_id, version).await?;

        // Initialize download progress
        {
            let mut queue = self.download_queue.write().unwrap();
            queue.insert(
                cache_key.clone(),
                DownloadProgress {
                    model_id: model_id.to_string(),
                    version: version.to_string(),
                    bytes_downloaded: 0,
                    total_bytes: (metadata.size_mb * 1024.0 * 1024.0) as u64,
                    progress_percent: 0.0,
                    download_speed_mbps: 0.0,
                    eta_seconds: 0,
                    status: DownloadStatus::Queued,
                },
            );
        }

        // Perform download
        let local_path = self
            .config
            .storage_path
            .join("models")
            .join(model_id)
            .join(version)
            .join("model.safetensors");

        fs::create_dir_all(local_path.parent().unwrap()).map_err(|e| {
            TrustformersError::Core(CoreTrustformersError::other(format!(
                "Failed to create model directory: {}",
                e
            )))
        })?;

        let download_result =
            self.download_with_progress(&download_url, &local_path, &cache_key).await;

        match download_result {
            Ok(()) => {
                // Verify download
                let file_size = local_path
                    .metadata()
                    .map_err(|e| {
                        TrustformersError::Core(CoreTrustformersError::other(format!(
                            "Failed to get file metadata: {}",
                            e
                        )))
                    })?
                    .len();

                let checksum = self.calculate_file_hash(&local_path).await?;

                // Add to cache
                let cached_model = CachedModel {
                    model_id: model_id.to_string(),
                    version: version.to_string(),
                    local_path: local_path.clone(),
                    remote_url: download_url,
                    cached_at: SystemTime::now(),
                    last_accessed: SystemTime::now(),
                    access_count: 1,
                    file_size,
                    checksum,
                    metadata,
                    is_priority: self.config.priority_models.contains(&model_id.to_string()),
                    download_complete: true,
                };

                {
                    let mut cache = self.cache.write().unwrap();
                    cache.insert(cache_key.clone(), cached_model);
                }

                // Update statistics
                {
                    let mut stats = self.stats.lock().unwrap();
                    stats.downloads_completed += 1;
                    stats.total_models = self.cache.read().unwrap().len();
                    stats.total_size_gb += file_size as f64 / (1024.0 * 1024.0 * 1024.0);
                }

                // Remove from download queue
                {
                    let mut queue = self.download_queue.write().unwrap();
                    queue.remove(&cache_key);
                }

                // Save cache to disk
                self.save_cache().await?;

                tracing::info!(
                    "Successfully downloaded {}:{} ({} MB)",
                    model_id,
                    version,
                    file_size / 1024 / 1024
                );

                Ok(local_path)
            },
            Err(e) => {
                // Update download queue with error
                {
                    let mut queue = self.download_queue.write().unwrap();
                    if let Some(progress) = queue.get_mut(&cache_key) {
                        progress.status = DownloadStatus::Failed(e.to_string());
                    }
                }

                // Update statistics
                {
                    let mut stats = self.stats.lock().unwrap();
                    stats.downloads_failed += 1;
                }

                Err(e)
            },
        }
    }

    /// Download file with progress tracking
    async fn download_with_progress(
        &self,
        url: &str,
        local_path: &Path,
        cache_key: &str,
    ) -> Result<()> {
        let mut attempt = 0;

        while attempt < self.config.retry_attempts {
            attempt += 1;

            match self.attempt_download(url, local_path, cache_key).await {
                Ok(()) => return Ok(()),
                Err(e) => {
                    if attempt < self.config.retry_attempts {
                        tracing::warn!(
                            "Download attempt {} failed for {}: {}. Retrying in {:?}...",
                            attempt,
                            cache_key,
                            e,
                            self.config.retry_delay
                        );
                        sleep(self.config.retry_delay).await;
                    } else {
                        return Err(e);
                    }
                },
            }
        }

        Err(TrustformersError::Core(CoreTrustformersError::other(
            format!(
                "Download failed after {} attempts",
                self.config.retry_attempts
            )
            .to_string(),
        )))
    }

    async fn attempt_download(&self, url: &str, local_path: &Path, cache_key: &str) -> Result<()> {
        // Update status to downloading
        {
            let mut queue = self.download_queue.write().unwrap();
            if let Some(progress) = queue.get_mut(cache_key) {
                progress.status = DownloadStatus::Downloading;
            }
        }

        let response = self.http_client.get(url).send().await.map_err(|e| {
            TrustformersError::Core(CoreTrustformersError::other(format!(
                "Failed to start download: {}",
                e
            )))
        })?;

        if !response.status().is_success() {
            return Err(TrustformersError::Core(CoreTrustformersError::other(
                format!("Download failed with status: {}", response.status()),
            )));
        }

        let total_size = response.content_length().unwrap_or(0);
        let mut file = async_fs::File::create(local_path).await.map_err(|e| {
            TrustformersError::Core(CoreTrustformersError::other(format!(
                "Failed to create file: {}",
                e
            )))
        })?;

        let mut stream = response.bytes_stream();
        let mut downloaded = 0u64;
        let start_time = std::time::Instant::now();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| {
                TrustformersError::Core(CoreTrustformersError::other(format!(
                    "Failed to read chunk: {}",
                    e
                )))
            })?;

            file.write_all(&chunk).await.map_err(|e| {
                TrustformersError::Core(CoreTrustformersError::other(format!(
                    "Failed to write chunk: {}",
                    e
                )))
            })?;

            downloaded += chunk.len() as u64;

            // Update progress
            let elapsed = start_time.elapsed().as_secs_f64();
            let speed_mbps = if elapsed > 0.0 {
                (downloaded as f64 / elapsed) / (1024.0 * 1024.0)
            } else {
                0.0
            };

            let progress_percent = if total_size > 0 {
                (downloaded as f64 / total_size as f64) * 100.0
            } else {
                0.0
            };

            let eta_seconds = if speed_mbps > 0.0 && total_size > 0 {
                ((total_size - downloaded) as f64 / (speed_mbps * 1024.0 * 1024.0)) as u64
            } else {
                0
            };

            {
                let mut queue = self.download_queue.write().unwrap();
                if let Some(progress) = queue.get_mut(cache_key) {
                    progress.bytes_downloaded = downloaded;
                    progress.progress_percent = progress_percent;
                    progress.download_speed_mbps = speed_mbps;
                    progress.eta_seconds = eta_seconds;
                }
            }

            // Apply bandwidth limiting if configured
            if let Some(limit_mbps) = self.config.bandwidth_limit_mbps {
                let target_delay = (chunk.len() as f64) / (limit_mbps * 1024.0 * 1024.0);
                if elapsed < target_delay {
                    sleep(Duration::from_secs_f64(target_delay - elapsed)).await;
                }
            }
        }

        file.flush().await.map_err(|e| {
            TrustformersError::Core(CoreTrustformersError::other(format!(
                "Failed to flush file: {}",
                e
            )))
        })?;

        Ok(())
    }

    /// Wait for an ongoing download to complete
    async fn wait_for_download(&self, cache_key: &str) -> Result<PathBuf> {
        let mut interval = interval(Duration::from_millis(100));

        loop {
            interval.tick().await;

            let queue = self.download_queue.read().unwrap();
            if let Some(progress) = queue.get(cache_key) {
                match &progress.status {
                    DownloadStatus::Completed => {
                        drop(queue);
                        let cache = self.cache.read().unwrap();
                        if let Some(cached_model) = cache.get(cache_key) {
                            return Ok(cached_model.local_path.clone());
                        } else {
                            return Err(TrustformersError::Core(CoreTrustformersError::other(
                                "Download completed but model not in cache".to_string(),
                            )));
                        }
                    },
                    DownloadStatus::Failed(error) => {
                        return Err(TrustformersError::Core(CoreTrustformersError::other(
                            format!("Download failed: {}", error),
                        )));
                    },
                    DownloadStatus::Cancelled => {
                        return Err(TrustformersError::Core(CoreTrustformersError::other(
                            "Download was cancelled".to_string(),
                        )));
                    },
                    _ => {
                        // Still downloading, continue waiting
                        continue;
                    },
                }
            } else {
                // Download no longer in queue, check cache
                let cache = self.cache.read().unwrap();
                if let Some(cached_model) = cache.get(cache_key) {
                    return Ok(cached_model.local_path.clone());
                } else {
                    return Err(TrustformersError::Core(CoreTrustformersError::other(
                        "Download not found in queue or cache".to_string(),
                    )));
                }
            }
        }
    }

    /// Fetch model metadata from remote hub
    async fn fetch_model_metadata(&self, model_id: &str, version: &str) -> Result<ModelMetadata> {
        let metadata_url = format!(
            "{}/models/{}/versions/{}/metadata",
            self.config.remote_hub_url, model_id, version
        );

        let response = self.http_client.get(&metadata_url).send().await.map_err(|e| {
            TrustformersError::Core(CoreTrustformersError::other(format!(
                "Failed to fetch metadata: {}",
                e
            )))
        })?;

        if !response.status().is_success() {
            return Err(TrustformersError::Core(CoreTrustformersError::other(
                format!("Failed to fetch metadata: {}", response.status()),
            )));
        }

        let metadata: ModelMetadata = response.json().await.map_err(|e| {
            TrustformersError::Core(CoreTrustformersError::other(format!(
                "Failed to parse metadata: {}",
                e
            )))
        })?;

        Ok(metadata)
    }

    /// Start background synchronization task
    async fn start_background_sync(&mut self) -> Result<()> {
        let cache = self.cache.clone();
        let stats = self.stats.clone();
        let config = self.config.clone();
        let http_client = self.http_client.clone();

        let handle = tokio::spawn(async move {
            let mut sync_interval = interval(config.sync_interval);

            loop {
                sync_interval.tick().await;

                if let Err(e) = Self::sync_with_remote(&cache, &stats, &config, &http_client).await
                {
                    tracing::error!("Background sync failed: {}", e);
                    let mut stats_lock = stats.lock().unwrap();
                    stats_lock.sync_errors += 1;
                }
            }
        });

        self.sync_handle = Some(handle);
        Ok(())
    }

    /// Synchronize with remote hub
    async fn sync_with_remote(
        cache: &Arc<RwLock<HashMap<String, CachedModel>>>,
        stats: &Arc<Mutex<MirrorStats>>,
        config: &MirrorConfig,
        http_client: &reqwest::Client,
    ) -> Result<()> {
        tracing::info!("Starting background sync with remote hub");

        // Fetch list of available models from remote
        let models_url = format!("{}/models", config.remote_hub_url);
        let response = http_client.get(&models_url).send().await.map_err(|e| {
            TrustformersError::Core(CoreTrustformersError::other(format!(
                "Failed to fetch models list: {}",
                e
            )))
        })?;

        if !response.status().is_success() {
            return Err(TrustformersError::Core(CoreTrustformersError::other(
                format!("Failed to fetch models list: {}", response.status()),
            )));
        }

        let remote_models: Vec<RemoteModelInfo> = response.json().await.map_err(|e| {
            TrustformersError::Core(CoreTrustformersError::other(format!(
                "Failed to parse models list: {}",
                e
            )))
        })?;

        // Check for updates to cached models
        let mut updates_found = 0;
        {
            let cache_read = cache.read().unwrap();
            for cached_model in cache_read.values() {
                if let Some(remote_model) =
                    remote_models.iter().find(|m| m.model_id == cached_model.model_id)
                {
                    if remote_model.latest_version != cached_model.version {
                        tracing::info!(
                            "Update available for {}: {} -> {}",
                            cached_model.model_id,
                            cached_model.version,
                            remote_model.latest_version
                        );
                        updates_found += 1;
                    }
                }
            }
        }

        // Update statistics
        {
            let mut stats_lock = stats.lock().unwrap();
            stats_lock.last_sync = Some(SystemTime::now());
        }

        tracing::info!("Sync completed. Found {} updates available", updates_found);
        Ok(())
    }

    /// Clean up cache if storage is getting full
    async fn cleanup_if_needed(&self) -> Result<()> {
        let current_size = self.calculate_total_size().await?;
        let max_size = self.config.max_storage_size_gb * 1024.0 * 1024.0 * 1024.0;

        if current_size > max_size * self.config.cleanup_threshold {
            tracing::info!(
                "Cache cleanup triggered. Current size: {:.2} GB, Max: {:.2} GB",
                current_size / (1024.0 * 1024.0 * 1024.0),
                self.config.max_storage_size_gb
            );

            self.cleanup_cache().await?;
        }

        Ok(())
    }

    /// Perform cache cleanup using LRU strategy
    async fn cleanup_cache(&self) -> Result<()> {
        let mut models_to_remove = Vec::new();
        let max_size = self.config.max_storage_size_gb * 1024.0 * 1024.0 * 1024.0;
        let target_size = max_size * 0.7; // Clean up to 70% of max size

        {
            let cache = self.cache.read().unwrap();
            let mut cache_items: Vec<_> = cache.values().collect();

            // Sort by priority (keep priority models) and last access time
            cache_items.sort_by(|a, b| {
                match (a.is_priority, b.is_priority) {
                    (true, false) => std::cmp::Ordering::Greater,
                    (false, true) => std::cmp::Ordering::Less,
                    _ => a.last_accessed.cmp(&b.last_accessed), // LRU for same priority
                }
            });

            let mut current_size = self.calculate_total_size().await?;

            for model in cache_items {
                if current_size <= target_size || model.is_priority {
                    break;
                }

                models_to_remove.push((model.model_id.clone(), model.version.clone()));
                current_size -= model.file_size as f64;
            }
        }

        // Remove selected models
        for (model_id, version) in models_to_remove {
            self.remove_model(&model_id, &version).await?;
            tracing::info!("Removed {}:{} during cleanup", model_id, version);
        }

        Ok(())
    }

    /// Remove a model from cache
    pub async fn remove_model(&self, model_id: &str, version: &str) -> Result<()> {
        let cache_key = format!("{}:{}", model_id, version);

        let local_path = {
            let cache = self.cache.read().unwrap();
            cache.get(&cache_key).map(|m| m.local_path.clone())
        };

        if let Some(path) = local_path {
            // Remove file
            if path.exists() {
                async_fs::remove_file(&path).await.map_err(|e| {
                    TrustformersError::Core(CoreTrustformersError::other(format!(
                        "Failed to remove file: {}",
                        e
                    )))
                })?;
            }

            // Remove from cache
            {
                let mut cache = self.cache.write().unwrap();
                cache.remove(&cache_key);
            }

            // Update statistics
            {
                let mut stats = self.stats.lock().unwrap();
                stats.total_models = self.cache.read().unwrap().len();
            }

            self.save_cache().await?;
        }

        Ok(())
    }

    /// Get mirror statistics
    pub fn get_stats(&self) -> MirrorStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get download progress for all active downloads
    pub fn get_download_progress(&self) -> Vec<DownloadProgress> {
        self.download_queue.read().unwrap().values().cloned().collect()
    }

    /// Calculate total cache size
    async fn calculate_total_size(&self) -> Result<f64> {
        let cache = self.cache.read().unwrap();
        Ok(cache.values().map(|m| m.file_size as f64).sum())
    }

    /// Calculate file hash
    async fn calculate_file_hash(&self, path: &Path) -> Result<String> {
        let mut file = async_fs::File::open(path).await.map_err(|e| {
            TrustformersError::Core(CoreTrustformersError::other(format!(
                "Failed to open file for hashing: {}",
                e
            )))
        })?;

        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 8192];

        loop {
            let n = file.read(&mut buffer).await.map_err(|e| {
                TrustformersError::Core(CoreTrustformersError::other(format!(
                    "Failed to read file for hashing: {}",
                    e
                )))
            })?;

            if n == 0 {
                break;
            }

            hasher.update(&buffer[..n]);
        }

        Ok(format!("{:x}", hasher.finalize()))
    }

    /// Load cache from disk
    async fn load_cache(&self) -> Result<()> {
        let cache_file = self.config.storage_path.join("cache.json");

        if cache_file.exists() {
            let content = async_fs::read_to_string(&cache_file).await.map_err(|e| {
                TrustformersError::Core(CoreTrustformersError::other(format!(
                    "Failed to read cache file: {}",
                    e
                )))
            })?;

            let cached_models: HashMap<String, CachedModel> = serde_json::from_str(&content)
                .map_err(|e| {
                    TrustformersError::Core(CoreTrustformersError::other(format!(
                        "Failed to parse cache file: {}",
                        e
                    )))
                })?;

            // Verify cached files still exist
            let mut valid_models = HashMap::new();
            for (key, model) in cached_models {
                if model.local_path.exists() {
                    valid_models.insert(key, model);
                }
            }

            {
                let mut cache = self.cache.write().unwrap();
                *cache = valid_models;
            }

            tracing::info!(
                "Loaded {} models from cache",
                self.cache.read().unwrap().len()
            );
        }

        Ok(())
    }

    /// Save cache to disk
    async fn save_cache(&self) -> Result<()> {
        let cache_file = self.config.storage_path.join("cache.json");
        let cache = self.cache.read().unwrap();

        let content = serde_json::to_string_pretty(&*cache).map_err(|e| {
            TrustformersError::Core(CoreTrustformersError::other(format!(
                "Failed to serialize cache: {}",
                e
            )))
        })?;

        async_fs::write(&cache_file, content).await.map_err(|e| {
            TrustformersError::Core(CoreTrustformersError::other(format!(
                "Failed to write cache file: {}",
                e
            )))
        })?;

        Ok(())
    }

    /// Shutdown the mirror gracefully
    pub async fn shutdown(&mut self) -> Result<()> {
        if let Some(handle) = self.sync_handle.take() {
            handle.abort();
        }

        self.save_cache().await?;
        tracing::info!("Hub mirror shut down gracefully");
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct RemoteModelInfo {
    model_id: String,
    latest_version: String,
    size_mb: f64,
    updated_at: String,
}

/// Global mirror instance
static HUB_MIRROR: std::sync::OnceLock<Arc<tokio::sync::Mutex<HubMirror>>> =
    std::sync::OnceLock::new();

/// Initialize global hub mirror
pub async fn init_hub_mirror(config: MirrorConfig) -> Result<()> {
    let mut mirror = HubMirror::new(config)?;
    mirror.initialize().await?;

    HUB_MIRROR.set(Arc::new(tokio::sync::Mutex::new(mirror))).map_err(|_| {
        TrustformersError::Core(CoreTrustformersError::other(
            "Hub mirror already initialized".to_string(),
        ))
    })?;

    Ok(())
}

/// Get global hub mirror
pub fn get_hub_mirror() -> Result<Arc<tokio::sync::Mutex<HubMirror>>> {
    HUB_MIRROR.get().cloned().ok_or_else(|| {
        TrustformersError::Core(CoreTrustformersError::other(
            "Hub mirror not initialized".to_string(),
        ))
    })
}

/// Convenience function to get a model from the mirror
pub async fn get_model_from_mirror(model_id: &str, version: Option<&str>) -> Result<PathBuf> {
    let mirror = get_hub_mirror()?;
    let mirror_lock = mirror.lock().await;
    mirror_lock.get_model(model_id, version).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_mirror_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = MirrorConfig {
            storage_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let mirror = HubMirror::new(config).unwrap();
        assert_eq!(mirror.cache.read().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_cache_operations() {
        let temp_dir = TempDir::new().unwrap();
        let config = MirrorConfig {
            storage_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let mirror = HubMirror::new(config).unwrap();

        // Test cache save/load
        mirror.save_cache().await.unwrap();
        mirror.load_cache().await.unwrap();
    }

    #[test]
    fn test_download_progress() {
        let progress = DownloadProgress {
            model_id: "test-model".to_string(),
            version: "1.0".to_string(),
            bytes_downloaded: 50,
            total_bytes: 100,
            progress_percent: 50.0,
            download_speed_mbps: 10.0,
            eta_seconds: 5,
            status: DownloadStatus::Downloading,
        };

        assert_eq!(progress.progress_percent, 50.0);
        assert_eq!(progress.status, DownloadStatus::Downloading);
    }

    #[test]
    fn test_mirror_config() {
        let config = MirrorConfig::default();
        assert_eq!(config.max_storage_size_gb, 100.0);
        assert_eq!(config.parallel_downloads, 4);
        assert!(config.auto_cleanup);
    }
}
