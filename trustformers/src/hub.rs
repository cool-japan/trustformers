use crate::error::{Result, TrustformersError};
#[cfg(feature = "hub")]
use futures::stream::{self, StreamExt};
#[cfg(feature = "hub")]
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
#[cfg(feature = "hub")]
use reqwest::{blocking::Client, Client as AsyncClient};
use serde::{Deserialize, Serialize};
#[cfg(feature = "hub")]
use sha2::{Digest, Sha256};
use std::fs;
#[cfg(feature = "hub")]
use std::fs::{File, OpenOptions};
#[cfg(feature = "hub")]
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
#[cfg(feature = "hub")]
use std::sync::Arc;
use std::time::{Duration, Instant};
#[cfg(feature = "hub")]
use tokio::sync::Semaphore;
use trustformers_core::errors::TrustformersError as CoreTrustformersError;

const HF_HUB_URL: &str = "https://huggingface.co";

/// Options for downloading models from the Hugging Face Hub
#[derive(Clone, Debug)]
pub struct HubOptions {
    pub revision: Option<String>,
    pub cache_dir: Option<PathBuf>,
    pub force_download: bool,
    pub token: Option<String>,
    pub parallel_downloads: bool,
    pub max_concurrent_downloads: usize,
    pub enable_resumable_downloads: bool,
    pub enable_delta_compression: bool,
    pub chunk_size: usize,
    pub timeout_seconds: u64,
    pub retry_attempts: usize,
    pub use_cdn: bool,
    pub cdn_urls: Vec<String>,
    pub smart_caching: bool,
}

impl Default for HubOptions {
    fn default() -> Self {
        Self {
            revision: Some("main".to_string()),
            cache_dir: None,
            force_download: false,
            token: None,
            parallel_downloads: true,
            max_concurrent_downloads: 4,
            enable_resumable_downloads: true,
            enable_delta_compression: true,
            chunk_size: 8 * 1024 * 1024, // 8MB chunks
            timeout_seconds: 300,
            retry_attempts: 3,
            use_cdn: true,
            cdn_urls: vec![
                "https://cdn-lfs.huggingface.co".to_string(),
                "https://cdn.huggingface.co".to_string(),
            ],
            smart_caching: true,
        }
    }
}

/// Advanced download configuration
#[derive(Clone, Debug)]
pub struct DownloadConfig {
    pub parallel_downloads: bool,
    pub max_concurrent: usize,
    pub enable_resumable: bool,
    pub enable_compression: bool,
    pub chunk_size: usize,
    pub timeout: Duration,
    pub retry_attempts: usize,
    pub verify_checksums: bool,
    pub progress_reporting: bool,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            parallel_downloads: true,
            max_concurrent: 4,
            enable_resumable: true,
            enable_compression: true,
            chunk_size: 8 * 1024 * 1024,
            timeout: Duration::from_secs(300),
            retry_attempts: 3,
            verify_checksums: true,
            progress_reporting: true,
        }
    }
}

/// Download statistics and metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DownloadStats {
    pub total_files: usize,
    pub downloaded_files: usize,
    pub failed_files: usize,
    pub total_bytes: u64,
    pub downloaded_bytes: u64,
    #[serde(skip)]
    pub start_time: Option<Instant>,
    #[serde(skip)]
    pub end_time: Option<Instant>,
    pub average_speed_mbps: f64,
    pub parallel_efficiency: f64,
    pub cache_hit_rate: f64,
    pub compression_ratio: f64,
    pub resume_count: usize,
}

impl DownloadStats {
    pub fn duration(&self) -> Option<Duration> {
        if let (Some(start), Some(end)) = (self.start_time, self.end_time) {
            Some(end.duration_since(start))
        } else {
            None
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_files > 0 {
            self.downloaded_files as f64 / self.total_files as f64
        } else {
            0.0
        }
    }
}

/// Resume information for downloads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResumeInfo {
    pub url: String,
    pub local_path: PathBuf,
    pub expected_size: u64,
    pub downloaded_size: u64,
    pub checksum: Option<String>,
    pub last_modified: Option<String>,
    #[serde(skip, default = "Instant::now")]
    pub created_at: Instant,
}

impl ResumeInfo {
    pub fn can_resume(&self, max_age: Duration) -> bool {
        self.created_at.elapsed() < max_age && self.downloaded_size > 0
    }
}

/// Delta compression info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaInfo {
    pub base_version: String,
    pub target_version: String,
    pub delta_url: String,
    pub compression_ratio: f64,
    pub delta_size: u64,
    pub full_size: u64,
    /// SHA-256 (hex) of the delta file's raw bytes, if the server provided
    /// one. When present, the delta download itself is checksum-verified in
    /// addition to the TFDELTA1 format's own embedded base/target hashes.
    #[serde(default)]
    pub delta_checksum: Option<String>,
}

/// CDN configuration and routing
#[derive(Debug, Clone)]
pub struct CdnConfig {
    pub primary_urls: Vec<String>,
    pub fallback_urls: Vec<String>,
    pub health_check_interval: Duration,
    pub latency_threshold: Duration,
    pub enable_geographic_routing: bool,
    pub region_preferences: Vec<String>,
}

impl Default for CdnConfig {
    fn default() -> Self {
        Self {
            primary_urls: vec![
                "https://cdn-lfs.huggingface.co".to_string(),
                "https://cdn.huggingface.co".to_string(),
            ],
            fallback_urls: vec!["https://huggingface.co".to_string()],
            health_check_interval: Duration::from_secs(300),
            latency_threshold: Duration::from_millis(1000),
            enable_geographic_routing: true,
            region_preferences: vec!["us".to_string(), "eu".to_string()],
        }
    }
}

/// Smart cache management
#[derive(Debug, Clone)]
pub struct SmartCacheConfig {
    pub max_cache_size_gb: f64,
    pub cleanup_threshold: f64,
    pub access_weight: f64,
    pub frequency_weight: f64,
    pub recency_weight: f64,
    pub size_penalty: f64,
    pub enable_predictive_caching: bool,
    pub enable_compression: bool,
}

impl Default for SmartCacheConfig {
    fn default() -> Self {
        Self {
            max_cache_size_gb: 50.0,
            cleanup_threshold: 0.9,
            access_weight: 0.4,
            frequency_weight: 0.3,
            recency_weight: 0.2,
            size_penalty: 0.1,
            enable_predictive_caching: true,
            enable_compression: true,
        }
    }
}

/// File information from the Hub API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoFile {
    pub path: String,
    pub size: u64,
    #[serde(rename = "lfs")]
    pub lfs: Option<LfsInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LfsInfo {
    pub sha256: String,
    pub size: u64,
    #[serde(rename = "pointerSize")]
    pub pointer_size: u64,
}

/// Model information from the Hub
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model_id: String,
    pub sha: String,
    pub pipeline_tag: Option<String>,
    pub library_name: Option<String>,
    pub downloads: u64,
    pub likes: u64,
}

/// Model card information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    pub license: Option<String>,
    pub language: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
    pub datasets: Option<Vec<String>>,
    pub metrics: Option<Vec<String>>,
    pub widget: Option<Vec<serde_json::Value>>,
    pub model_index: Option<Vec<serde_json::Value>>,
    pub thumbnail: Option<String>,
    pub pipeline_tag: Option<String>,
    pub inference: Option<bool>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Get the cache directory for models
pub fn get_cache_dir() -> Result<PathBuf> {
    if let Ok(cache_dir) = std::env::var("TRUSTFORMERS_CACHE") {
        Ok(PathBuf::from(cache_dir))
    } else if let Some(cache_dir) = dirs::cache_dir() {
        Ok(cache_dir.join("trustformers"))
    } else if let Ok(home) = std::env::var("HOME") {
        Ok(PathBuf::from(home).join(".cache").join("trustformers"))
    } else {
        Err(TrustformersError::Core(CoreTrustformersError::other(
            "Could not determine cache directory".to_string(),
        )))
    }
}

/// Check if a model exists in the cache
pub fn is_cached(model_id: &str, revision: Option<&str>) -> Result<bool> {
    let cache_dir = get_cache_dir()?;
    let model_dir = cache_dir
        .join("models")
        .join(model_id.replace('/', "--"))
        .join(revision.unwrap_or("main"));

    Ok(model_dir.exists())
}

/// Enhanced download manager with parallel and resumable downloads
///
/// Note: this does not yet route through [`CdnConfig`] or persist
/// [`ResumeInfo`] — real resumable downloads work off the on-disk file's own
/// `fs::metadata().len()` in `download_single_file_async`, and `RepoFile`
/// URLs are used as-is rather than being routed through a CDN's
/// primary/fallback host list. `HubOptions::use_cdn` therefore does not yet
/// change request routing; both types remain available (`CdnConfig`,
/// `ResumeInfo`) for a future CDN-routing/resume-persistence implementation.
#[cfg(feature = "hub")]
pub struct DownloadManager {
    config: DownloadConfig,
    cache_config: SmartCacheConfig,
    client: AsyncClient,
    stats: DownloadStats,
}

#[cfg(feature = "hub")]
impl DownloadManager {
    pub fn new(config: DownloadConfig) -> Self {
        let client = AsyncClient::builder()
            .timeout(config.timeout)
            .build()
            .unwrap_or_else(|_| AsyncClient::new());

        Self {
            config,
            cache_config: SmartCacheConfig::default(),
            client,
            stats: DownloadStats::default(),
        }
    }

    /// Download multiple files in parallel
    pub async fn download_files_parallel(
        &mut self,
        downloads: Vec<DownloadTask>,
        token: Option<&str>,
    ) -> Result<DownloadStats> {
        self.stats.start_time = Some(Instant::now());
        self.stats.total_files = downloads.len();
        self.stats.total_bytes = downloads.iter().map(|d| d.expected_size).sum();

        let multi_progress = MultiProgress::new();
        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrent));

        // Create progress bars for each download
        let progress_bars: Vec<_> = downloads
            .iter()
            .map(|task| {
                let pb = multi_progress.add(ProgressBar::new(task.expected_size));
                pb.set_style(
                    ProgressStyle::default_bar()
                        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} {msg}")
                        .unwrap_or_else(|_| ProgressStyle::default_bar())
                        .progress_chars("#>-"),
                );
                pb.set_message(task.filename.clone());
                pb
            })
            .collect();

        // Execute downloads concurrently
        let results = stream::iter(downloads.into_iter().enumerate())
            .map(|(index, task)| {
                let semaphore = semaphore.clone();
                let client = self.client.clone();
                let config = self.config.clone();
                let pb = progress_bars[index].clone();
                let token = token.map(|s| s.to_string());

                async move {
                    let _permit = match semaphore.acquire().await {
                        Ok(permit) => permit,
                        Err(_) => {
                            return Err(TrustformersError::resource(
                                "Download semaphore closed unexpectedly",
                                "semaphore",
                            ));
                        },
                    };
                    Self::download_single_file_async(client, task, token.as_deref(), config, pb)
                        .await
                }
            })
            .buffer_unordered(self.config.max_concurrent)
            .collect::<Vec<_>>()
            .await;

        // Process results
        for result in results {
            match result {
                Ok(_) => self.stats.downloaded_files += 1,
                Err(_) => self.stats.failed_files += 1,
            }
        }

        self.stats.end_time = Some(Instant::now());
        self.calculate_final_stats();

        Ok(self.stats.clone())
    }

    /// Download a single file with resumable support
    async fn download_single_file_async(
        client: AsyncClient,
        task: DownloadTask,
        token: Option<&str>,
        config: DownloadConfig,
        progress_bar: ProgressBar,
    ) -> Result<()> {
        let mut resume_offset = 0u64;
        let mut file = Self::prepare_file_for_download(&task.local_path, config.enable_resumable)?;

        // Check for resumable download
        if config.enable_resumable {
            if let Ok(metadata) = std::fs::metadata(&task.local_path) {
                resume_offset = metadata.len();
                progress_bar.set_position(resume_offset);
            }
        }

        let mut attempt = 0;
        while attempt < config.retry_attempts {
            match Self::attempt_download(
                &client,
                &task,
                token,
                resume_offset,
                &mut file,
                &progress_bar,
                &config,
            )
            .await
            {
                Ok(_) => return Ok(()),
                Err(e) => {
                    attempt += 1;
                    if attempt >= config.retry_attempts {
                        progress_bar.finish_with_message("Failed");
                        return Err(e);
                    }
                    // Exponential backoff
                    tokio::time::sleep(Duration::from_secs(2u64.pow(attempt as u32))).await;
                },
            }
        }

        Err(TrustformersError::Core(CoreTrustformersError::other(
            format!("Download failed after {} attempts", config.retry_attempts),
        )))
    }

    async fn attempt_download(
        client: &AsyncClient,
        task: &DownloadTask,
        token: Option<&str>,
        resume_offset: u64,
        file: &mut File,
        progress_bar: &ProgressBar,
        config: &DownloadConfig,
    ) -> Result<()> {
        let mut request = client.get(&task.url);

        if let Some(token) = token {
            request = request.bearer_auth(token);
        }

        // Add range header for resumable downloads
        if resume_offset > 0 {
            request = request.header("Range", format!("bytes={}-", resume_offset));
        }

        let response = request.send().await.map_err(|e| {
            TrustformersError::Core(CoreTrustformersError::other(format!(
                "Failed to send request: {}",
                e
            )))
        })?;

        if !response.status().is_success() && response.status().as_u16() != 206 {
            return Err(TrustformersError::Core(CoreTrustformersError::other(
                format!("Download failed with status: {}", response.status()),
            )));
        }

        // Seek to resume position if necessary
        if resume_offset > 0 {
            file.seek(SeekFrom::Start(resume_offset)).map_err(|e| TrustformersError::Io {
                message: format!("Failed to seek file: {}", e),
                path: Some(task.local_path.to_string_lossy().to_string()),
                suggestion: Some("Check file permissions and disk space".to_string()),
            })?;
        }

        let mut hasher = Sha256::new();
        let mut bytes_stream = response.bytes_stream();

        while let Some(chunk) = bytes_stream.next().await {
            let chunk = chunk.map_err(|e| TrustformersError::Network {
                message: format!("Failed to read chunk: {}", e),
                url: Some(task.url.clone()),
                status_code: None,
                suggestion: Some("Check network connection and retry".to_string()),
                retry_recommended: true,
            })?;

            hasher.update(&chunk);
            file.write_all(&chunk).map_err(|e| TrustformersError::Io {
                message: format!("Failed to write chunk: {}", e),
                path: Some(task.local_path.to_string_lossy().to_string()),
                suggestion: Some("Check disk space and file permissions".to_string()),
            })?;

            progress_bar.inc(chunk.len() as u64);
        }

        file.flush().map_err(|e| TrustformersError::Io {
            message: format!("Failed to flush file: {}", e),
            path: Some(task.local_path.to_string_lossy().to_string()),
            suggestion: Some("Check disk space and file permissions".to_string()),
        })?;

        // Verify checksum if provided
        if let Some(expected_sha) = &task.expected_checksum {
            if config.verify_checksums {
                let calculated_sha = hex::encode(hasher.finalize());
                if &calculated_sha != expected_sha {
                    fs::remove_file(&task.local_path).ok();
                    return Err(TrustformersError::Core(CoreTrustformersError::other(
                        format!(
                            "Checksum mismatch: expected {}, got {}",
                            expected_sha, calculated_sha
                        ),
                    )));
                }
            }
        }

        progress_bar.finish_with_message("Completed");
        Ok(())
    }

    fn prepare_file_for_download(path: &Path, enable_resumable: bool) -> Result<File> {
        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| TrustformersError::Io {
                message: format!("Failed to create directory: {}", e),
                path: Some(parent.to_string_lossy().to_string()),
                suggestion: Some("Check permissions and available disk space".to_string()),
            })?;
        }

        let file = if enable_resumable && path.exists() {
            OpenOptions::new().append(true).open(path).map_err(|e| TrustformersError::Io {
                message: format!("Failed to open file for resume: {}", e),
                path: Some(path.to_string_lossy().to_string()),
                suggestion: Some("Check file permissions".to_string()),
            })?
        } else {
            File::create(path).map_err(|e| TrustformersError::Io {
                message: format!("Failed to create file: {}", e),
                path: Some(path.to_string_lossy().to_string()),
                suggestion: Some("Check directory permissions and disk space".to_string()),
            })?
        };

        Ok(file)
    }

    fn calculate_final_stats(&mut self) {
        if let Some(duration) = self.stats.duration() {
            let duration_secs = duration.as_secs_f64();
            if duration_secs > 0.0 {
                let bytes_per_sec = self.stats.downloaded_bytes as f64 / duration_secs;
                self.stats.average_speed_mbps = bytes_per_sec / (1024.0 * 1024.0);
            }
        }

        // Calculate parallel efficiency (simplified)
        if self.stats.total_files > 1 {
            self.stats.parallel_efficiency =
                self.config.max_concurrent as f64 / self.stats.total_files as f64;
            if self.stats.parallel_efficiency > 1.0 {
                self.stats.parallel_efficiency = 1.0;
            }
        }
    }

    /// Apply delta compression if available.
    ///
    /// The downloaded delta is a self-verifying TFDELTA1 file (see
    /// [`crate::hub_delta_codec`]): [`apply_binary_delta`](Self::apply_binary_delta)
    /// refuses to reconstruct anything unless both the base and the
    /// reconstructed target pass their embedded SHA-256 checks, so a
    /// mismatched or corrupted delta can never silently produce a bad model
    /// file. The downloaded delta file is always cleaned up, whether or not
    /// applying it succeeded.
    pub async fn apply_delta_compression(
        &self,
        delta_info: &DeltaInfo,
        base_path: &Path,
        target_path: &Path,
    ) -> Result<()> {
        // Download delta file
        let delta_path = target_path.with_extension("delta");
        let task = DownloadTask {
            url: delta_info.delta_url.clone(),
            local_path: delta_path.clone(),
            filename: "delta".to_string(),
            expected_size: delta_info.delta_size,
            expected_checksum: delta_info.delta_checksum.clone(),
        };

        Self::download_single_file_async(
            self.client.clone(),
            task,
            None,
            self.config.clone(),
            ProgressBar::hidden(),
        )
        .await?;

        let result = self.apply_binary_delta(&delta_path, base_path, target_path).await;

        // Always clean up the downloaded delta file, on both success and
        // failure — never leave it behind because reconstruction errored out.
        fs::remove_file(&delta_path).ok();

        result
    }

    async fn apply_binary_delta(
        &self,
        delta_path: &Path,
        base_path: &Path,
        target_path: &Path,
    ) -> Result<()> {
        let delta_data = fs::read(delta_path).map_err(|e| TrustformersError::Io {
            message: format!("Failed to read delta file: {}", e),
            path: Some(delta_path.to_string_lossy().to_string()),
            suggestion: Some("Check file existence and permissions".to_string()),
        })?;

        let base_data = fs::read(base_path).map_err(|e| TrustformersError::Io {
            message: format!("Failed to read base file: {}", e),
            path: Some(base_path.to_string_lossy().to_string()),
            suggestion: Some("Check file existence and permissions".to_string()),
        })?;

        // Reconstructs the target from a real block-copy/insert delta,
        // verifying both the base and the reconstructed target against the
        // checksums embedded in the delta itself. Never returns bytes that
        // haven't passed both checks.
        let target_data = reconstruct_from_delta(&base_data, &delta_data)?;

        // Write to a temp file and rename into place, so a crash or a later
        // error never leaves a partially-written or corrupted file sitting at
        // `target_path` — by the time we reach this line, `target_data` has
        // already been checksum-verified against the delta's embedded
        // `target_sha256`.
        let mut tmp_name = target_path.file_name().map(|n| n.to_os_string()).unwrap_or_default();
        tmp_name.push(".tmp-delta");
        let tmp_path = target_path.with_file_name(tmp_name);

        fs::write(&tmp_path, &target_data).map_err(|e| TrustformersError::Io {
            message: format!("Failed to write reconstructed target file: {}", e),
            path: Some(tmp_path.to_string_lossy().to_string()),
            suggestion: Some("Check permissions and disk space".to_string()),
        })?;
        fs::rename(&tmp_path, target_path).map_err(|e| {
            fs::remove_file(&tmp_path).ok();
            TrustformersError::Io {
                message: format!("Failed to move reconstructed target file into place: {}", e),
                path: Some(target_path.to_string_lossy().to_string()),
                suggestion: Some("Check permissions and disk space".to_string()),
            }
        })?;

        Ok(())
    }

    /// Smart cache management
    pub fn manage_smart_cache(&mut self, cache_dir: &Path) -> Result<()> {
        let cache_usage = self.calculate_cache_usage(cache_dir)?;
        let max_size_bytes =
            (self.cache_config.max_cache_size_gb * 1024.0 * 1024.0 * 1024.0) as u64;

        if cache_usage.total_size
            > (max_size_bytes as f64 * self.cache_config.cleanup_threshold) as u64
        {
            self.cleanup_cache(cache_dir, &cache_usage, max_size_bytes)?;
        }

        Ok(())
    }

    fn calculate_cache_usage(&self, cache_dir: &Path) -> Result<CacheUsage> {
        let mut usage = CacheUsage::default();
        self.scan_cache_directory(cache_dir, &mut usage)?;
        Ok(usage)
    }

    fn scan_cache_directory(&self, dir: &Path, usage: &mut CacheUsage) -> Result<()> {
        for entry in fs::read_dir(dir).map_err(|e| TrustformersError::Io {
            message: format!("Failed to read cache directory: {}", e),
            path: Some(dir.to_string_lossy().to_string()),
            suggestion: Some("Check directory existence and permissions".to_string()),
        })? {
            let entry = entry.map_err(|e| TrustformersError::Io {
                message: format!("Failed to read directory entry: {}", e),
                path: Some(dir.to_string_lossy().to_string()),
                suggestion: Some("Check directory permissions".to_string()),
            })?;
            let path = entry.path();

            if path.is_file() {
                if let Ok(metadata) = fs::metadata(&path) {
                    usage.total_size += metadata.len();
                    usage.file_count += 1;

                    let access_time =
                        metadata.accessed().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                    let file_info = CacheFileInfo {
                        path: path.clone(),
                        size: metadata.len(),
                        access_time,
                        score: self.calculate_cache_score(&metadata),
                    };
                    usage.files.push(file_info);
                }
            } else if path.is_dir() {
                self.scan_cache_directory(&path, usage)?;
            }
        }
        Ok(())
    }

    fn calculate_cache_score(&self, metadata: &fs::Metadata) -> f64 {
        let now = std::time::SystemTime::now();
        let access_time = metadata.accessed().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        let recency = now.duration_since(access_time).unwrap_or(Duration::ZERO).as_secs() as f64;

        // Simple scoring based on recency and size
        let recency_score = 1.0 / (1.0 + recency / 86400.0); // Decay over days
        let size_penalty = (metadata.len() as f64).log10() * self.cache_config.size_penalty;

        (recency_score * self.cache_config.recency_weight) - size_penalty
    }

    fn cleanup_cache(&mut self, cache_dir: &Path, usage: &CacheUsage, max_size: u64) -> Result<()> {
        let mut files = usage.files.clone();
        files.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));

        let target_size = (max_size as f64 * 0.8) as u64; // Clean to 80% of max
        let mut current_size = usage.total_size;
        tracing::info!(
            "{}",
            format_cache_cleanup_start_message(cache_dir, current_size, target_size)
        );

        for file_info in files {
            if current_size <= target_size {
                break;
            }

            if fs::remove_file(&file_info.path).is_ok() {
                current_size -= file_info.size;
                tracing::info!("Removed cached file: {:?}", file_info.path);
            }
        }

        Ok(())
    }
}

/// Format the diagnostic logged when smart-cache cleanup starts.
///
/// Pulled out of `DownloadManager::cleanup_cache` so the message content
/// (which directory is being cleaned, and to what target) is unit-testable
/// without capturing `tracing` output.
///
/// `cfg`-gated on `hub` because its only caller, `DownloadManager::cleanup_cache`,
/// lives on the `hub`-gated `DownloadManager` impl block: without networking
/// there is no smart cache to clean.
#[cfg(feature = "hub")]
fn format_cache_cleanup_start_message(
    cache_dir: &Path,
    current_size: u64,
    target_size: u64,
) -> String {
    format!(
        "Cleaning cache directory {}: {} bytes -> target {} bytes",
        cache_dir.display(),
        current_size,
        target_size
    )
}

/// Download task definition
#[derive(Debug, Clone)]
pub struct DownloadTask {
    pub url: String,
    pub local_path: PathBuf,
    pub filename: String,
    pub expected_size: u64,
    pub expected_checksum: Option<String>,
}

/// Cache usage information
#[derive(Debug, Clone, Default)]
pub struct CacheUsage {
    pub total_size: u64,
    pub file_count: usize,
    pub files: Vec<CacheFileInfo>,
}

/// Cache file information
#[derive(Debug, Clone)]
pub struct CacheFileInfo {
    pub path: PathBuf,
    pub size: u64,
    pub access_time: std::time::SystemTime,
    pub score: f64,
}

// ─── Binary delta codec (TFDELTA1) ─────────────────────────────────────────
//
// Real implementation lives in `crate::hub_delta_codec` so this file stays
// under the workspace's 2000-line-per-file limit; these are the public,
// bytes-level entry points. Neither needs the `hub` feature — both are pure
// local computation — so they work even in builds with networking disabled.

/// Reconstruct a target file's bytes from a base file's bytes and a TFDELTA1
/// binary delta (as produced by [`create_binary_delta`]).
///
/// Both the base and the reconstructed target are checksum-verified against
/// the SHA-256 hashes embedded in the delta itself. A delta that doesn't
/// match the supplied base, that is truncated or malformed, or that doesn't
/// replay to the bytes it claims to, is a [`TrustformersError`] — never a
/// silently corrupted result. This is what makes
/// [`DownloadManager::apply_binary_delta`] safe: it is the only way this
/// module ever produces reconstructed bytes.
pub fn reconstruct_from_delta(base_data: &[u8], delta_data: &[u8]) -> Result<Vec<u8>> {
    crate::hub_delta_codec::apply_delta(base_data, delta_data).map_err(|message| {
        TrustformersError::Core(CoreTrustformersError::other(format!(
            "failed to apply binary delta: {message}"
        )))
    })
}

/// Encode a TFDELTA1 binary delta that reconstructs `target_path` from
/// `base_path`.
///
/// Hosting the result at a [`DeltaInfo::delta_url`] lets a future download of
/// `target_path`, starting from a peer that already has `base_path`, transfer
/// only the delta; [`reconstruct_from_delta`] is the inverse operation.
pub fn create_binary_delta(base_path: &Path, target_path: &Path) -> Result<Vec<u8>> {
    let base_data = fs::read(base_path).map_err(|e| TrustformersError::Io {
        message: format!("Failed to read base file: {}", e),
        path: Some(base_path.to_string_lossy().to_string()),
        suggestion: Some("Check file existence and permissions".to_string()),
    })?;
    let target_data = fs::read(target_path).map_err(|e| TrustformersError::Io {
        message: format!("Failed to read target file: {}", e),
        path: Some(target_path.to_string_lossy().to_string()),
        suggestion: Some("Check file existence and permissions".to_string()),
    })?;
    Ok(crate::hub_delta_codec::encode_delta(
        &base_data,
        &target_data,
    ))
}

/// Local-only fallback when the `hub` feature (networking) is disabled.
///
/// Callers such as [`download_file_from_hub`] check the on-disk cache first and
/// only reach this when an actual network download would be required, so local
/// and pre-cached models keep working; only remote fetches are refused.
#[cfg(not(feature = "hub"))]
fn download_file(
    url: &str,
    _path: &Path,
    _token: Option<&str>,
    _expected_sha: Option<&str>,
) -> Result<()> {
    Err(TrustformersError::Hub {
        message: "Remote model download is disabled: the `hub` feature is not enabled".to_string(),
        model_id: String::new(),
        endpoint: Some(url.to_string()),
        suggestion: Some(
            "Rebuild with the `hub` feature (e.g. `--features hub`) to download models, \
             or provide a local path / pre-populate the model cache"
                .to_string(),
        ),
        recovery_actions: vec![],
    })
}

/// Legacy synchronous download function (maintained for compatibility)
#[cfg(feature = "hub")]
fn download_file(
    url: &str,
    path: &Path,
    token: Option<&str>,
    expected_sha: Option<&str>,
) -> Result<()> {
    // Create a basic download task and use the async implementation
    let rt = tokio::runtime::Runtime::new().map_err(|e| {
        TrustformersError::runtime_error(format!("Failed to create tokio runtime: {}", e))
    })?;
    rt.block_on(async {
        let client = AsyncClient::new();
        let config = DownloadConfig::default();
        let pb = ProgressBar::new(0);

        let task = DownloadTask {
            url: url.to_string(),
            local_path: path.to_path_buf(),
            filename: path.file_name().unwrap_or_default().to_string_lossy().to_string(),
            expected_size: 0,
            expected_checksum: expected_sha.map(|s| s.to_string()),
        };

        DownloadManager::download_single_file_async(client, task, token, config, pb).await
    })
}

/// List files in a repository
#[cfg(feature = "hub")]
fn list_repo_files(model_id: &str, revision: &str, token: Option<&str>) -> Result<Vec<RepoFile>> {
    let client = Client::new();
    let url = format!("{}/api/models/{}/tree/{}", HF_HUB_URL, model_id, revision);

    let mut request = client.get(&url);
    if let Some(token) = token {
        request = request.bearer_auth(token);
    }

    let response = request.send().map_err(|e| TrustformersError::Hub {
        message: format!("Failed to list repo files: {}", e),
        model_id: model_id.to_string(),
        endpoint: Some(url.clone()),
        suggestion: Some("Check network connection and model ID".to_string()),
        recovery_actions: vec![],
    })?;

    if !response.status().is_success() {
        return Err(TrustformersError::Core(CoreTrustformersError::other(
            format!("Failed to list repo files: HTTP {}", response.status()),
        )));
    }

    let files: Vec<RepoFile> = response.json().map_err(|e| {
        TrustformersError::invalid_input(
            format!("Failed to parse repo files response: {}", e),
            Some("api_response"),
            Some("valid JSON array of RepoFile objects"),
            Some("invalid JSON format"),
        )
    })?;

    Ok(files)
}

/// Download a model from the Hugging Face Hub (legacy implementation)
#[cfg(feature = "hub")]
pub fn download_model(model_id: &str, options: Option<HubOptions>) -> Result<PathBuf> {
    let opts = options.unwrap_or_default();
    let revision = opts.revision.as_deref().unwrap_or("main");

    // Get cache directory
    let cache_dir = match opts.cache_dir {
        Some(dir) => dir,
        None => get_cache_dir()?,
    };
    let model_dir = cache_dir.join("models").join(model_id.replace('/', "--")).join(revision);

    // Check if already cached and not forcing download
    if !opts.force_download && model_dir.exists() {
        tracing::info!("Model {} already cached at {:?}", model_id, model_dir);
        return Ok(model_dir);
    }

    // Create model directory
    fs::create_dir_all(&model_dir).map_err(|e| TrustformersError::Io {
        message: format!("Failed to create model directory: {}", e),
        path: Some(model_dir.to_string_lossy().to_string()),
        suggestion: Some("Check cache directory permissions and disk space".to_string()),
    })?;

    // List files in the repository
    let files = list_repo_files(model_id, revision, opts.token.as_deref())?;

    // Download essential files
    let essential_files = [
        "config.json",
        "pytorch_model.bin",
        "model.safetensors",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
    ];

    for file in files.iter() {
        if essential_files.contains(&file.path.as_str()) || file.path.ends_with(".safetensors") {
            let file_path = model_dir.join(&file.path);

            // Skip if file already exists and not forcing download
            if !opts.force_download && file_path.exists() {
                tracing::info!("File {} already exists, skipping", file.path);
                continue;
            }

            let download_url = if file.lfs.is_some() {
                format!(
                    "{}/{}/resolve/{}/{}",
                    HF_HUB_URL, model_id, revision, file.path
                )
            } else {
                format!("{}/{}/raw/{}/{}", HF_HUB_URL, model_id, revision, file.path)
            };

            tracing::info!("Downloading {} from {}", file.path, download_url);

            let expected_sha = file.lfs.as_ref().map(|lfs| lfs.sha256.as_str());
            download_file(
                &download_url,
                &file_path,
                opts.token.as_deref(),
                expected_sha,
            )?;
        }
    }

    Ok(model_dir)
}

/// Enhanced model download with parallel downloads and advanced features
#[cfg(feature = "hub")]
pub async fn download_model_enhanced(
    model_id: &str,
    options: Option<HubOptions>,
) -> Result<(PathBuf, DownloadStats)> {
    let opts = options.unwrap_or_default();
    let revision = opts.revision.as_deref().unwrap_or("main");

    // Get cache directory
    let cache_dir = match opts.cache_dir {
        Some(dir) => dir,
        None => get_cache_dir()?,
    };
    let model_dir = cache_dir.join("models").join(model_id.replace('/', "--")).join(revision);

    // Check if already cached and not forcing download
    if !opts.force_download && model_dir.exists() {
        tracing::info!("Model {} already cached at {:?}", model_id, model_dir);
        return Ok((model_dir, DownloadStats::default()));
    }

    // Create model directory
    fs::create_dir_all(&model_dir).map_err(|e| TrustformersError::Io {
        message: format!("Failed to create model directory: {}", e),
        path: Some(model_dir.to_string_lossy().to_string()),
        suggestion: Some("Check cache directory permissions and disk space".to_string()),
    })?;

    // Create download configuration
    let download_config = DownloadConfig {
        parallel_downloads: opts.parallel_downloads,
        max_concurrent: opts.max_concurrent_downloads,
        enable_resumable: opts.enable_resumable_downloads,
        enable_compression: opts.enable_delta_compression,
        chunk_size: opts.chunk_size,
        timeout: Duration::from_secs(opts.timeout_seconds),
        retry_attempts: opts.retry_attempts,
        verify_checksums: true,
        progress_reporting: true,
    };

    let mut download_manager = DownloadManager::new(download_config);

    // Enable smart caching if requested
    if opts.smart_caching {
        download_manager.manage_smart_cache(&cache_dir)?;
    }

    // List files in the repository
    let files = list_repo_files(model_id, revision, opts.token.as_deref())?;

    // Filter and prepare download tasks
    let essential_files = [
        "config.json",
        "pytorch_model.bin",
        "model.safetensors",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
    ];

    let mut download_tasks = Vec::new();

    for file in files.iter() {
        if essential_files.contains(&file.path.as_str()) || file.path.ends_with(".safetensors") {
            let file_path = model_dir.join(&file.path);

            // Skip if file already exists and not forcing download
            if !opts.force_download && file_path.exists() {
                tracing::info!("File {} already exists, skipping", file.path);
                continue;
            }

            // Choose optimal download URL
            let download_url = if opts.use_cdn && !opts.cdn_urls.is_empty() {
                // Try CDN first
                if file.lfs.is_some() {
                    format!(
                        "{}/{}/resolve/{}/{}",
                        opts.cdn_urls[0], model_id, revision, file.path
                    )
                } else {
                    format!(
                        "{}/{}/raw/{}/{}",
                        opts.cdn_urls[0], model_id, revision, file.path
                    )
                }
            } else {
                // Use main hub URL
                if file.lfs.is_some() {
                    format!(
                        "{}/{}/resolve/{}/{}",
                        HF_HUB_URL, model_id, revision, file.path
                    )
                } else {
                    format!("{}/{}/raw/{}/{}", HF_HUB_URL, model_id, revision, file.path)
                }
            };

            let expected_checksum = file.lfs.as_ref().map(|lfs| lfs.sha256.clone());

            download_tasks.push(DownloadTask {
                url: download_url,
                local_path: file_path,
                filename: file.path.clone(),
                expected_size: file.size,
                expected_checksum,
            });
        }
    }

    // Execute downloads
    let stats = if opts.parallel_downloads && download_tasks.len() > 1 {
        tracing::info!(
            "Starting parallel download of {} files",
            download_tasks.len()
        );
        download_manager
            .download_files_parallel(download_tasks, opts.token.as_deref())
            .await?
    } else {
        tracing::info!(
            "Starting sequential download of {} files",
            download_tasks.len()
        );
        let mut sequential_stats = DownloadStats::default();
        sequential_stats.start_time = Some(Instant::now());
        sequential_stats.total_files = download_tasks.len();

        for task in download_tasks {
            let pb = ProgressBar::new(task.expected_size);
            match DownloadManager::download_single_file_async(
                download_manager.client.clone(),
                task,
                opts.token.as_deref(),
                download_manager.config.clone(),
                pb,
            )
            .await
            {
                Ok(_) => sequential_stats.downloaded_files += 1,
                Err(_) => sequential_stats.failed_files += 1,
            }
        }

        sequential_stats.end_time = Some(Instant::now());
        sequential_stats
    };

    tracing::info!("Download completed. Stats: {:#?}", stats);

    Ok((model_dir, stats))
}

/// Create optimized download configuration for different scenarios
pub fn create_download_config_for_scenario(scenario: DownloadScenario) -> DownloadConfig {
    match scenario {
        DownloadScenario::FastDevelopment => DownloadConfig {
            parallel_downloads: true,
            max_concurrent: 8,
            enable_resumable: true,
            enable_compression: false,    // Skip compression for speed
            chunk_size: 16 * 1024 * 1024, // 16MB chunks
            timeout: Duration::from_secs(120),
            retry_attempts: 2,
            verify_checksums: false, // Skip verification for speed
            progress_reporting: true,
        },
        DownloadScenario::Production => DownloadConfig {
            parallel_downloads: true,
            max_concurrent: 4,
            enable_resumable: true,
            enable_compression: true,
            chunk_size: 8 * 1024 * 1024,
            timeout: Duration::from_secs(600),
            retry_attempts: 5,
            verify_checksums: true,
            progress_reporting: false, // Reduce overhead in production
        },
        DownloadScenario::BandwidthLimited => DownloadConfig {
            parallel_downloads: false, // Sequential to reduce bandwidth usage
            max_concurrent: 1,
            enable_resumable: true,
            enable_compression: true,
            chunk_size: 1024 * 1024, // 1MB chunks
            timeout: Duration::from_secs(1200),
            retry_attempts: 10,
            verify_checksums: true,
            progress_reporting: true,
        },
        DownloadScenario::Reliable => DownloadConfig {
            parallel_downloads: true,
            max_concurrent: 2,
            enable_resumable: true,
            enable_compression: true,
            chunk_size: 4 * 1024 * 1024,
            timeout: Duration::from_secs(900),
            retry_attempts: 8,
            verify_checksums: true,
            progress_reporting: true,
        },
    }
}

/// Download scenarios for optimized configurations
#[derive(Debug, Clone, Copy)]
pub enum DownloadScenario {
    FastDevelopment,
    Production,
    BandwidthLimited,
    Reliable,
}

/// Get download statistics for a model
#[cfg(feature = "hub")]
pub async fn get_download_stats(
    model_id: &str,
    revision: Option<&str>,
) -> Result<ModelDownloadInfo> {
    let client = AsyncClient::new();
    let revision = revision.unwrap_or("main");
    let url = format!("{}/api/models/{}", HF_HUB_URL, model_id);

    let response = client.get(&url).send().await.map_err(|e| TrustformersError::Hub {
        message: format!("Failed to get model info: {}", e),
        model_id: model_id.to_string(),
        endpoint: Some(url.clone()),
        suggestion: Some("Check network connection and model ID".to_string()),
        recovery_actions: vec![],
    })?;

    if !response.status().is_success() {
        return Err(TrustformersError::Core(CoreTrustformersError::other(
            format!("Failed to get model info: HTTP {}", response.status()),
        )));
    }

    let model_info: serde_json::Value = response.json().await.map_err(|e| {
        TrustformersError::invalid_input(
            format!("Failed to parse model info response: {}", e),
            Some("api_response"),
            Some("valid JSON model info object"),
            Some("invalid JSON format"),
        )
    })?;

    // Extract relevant information
    let downloads = model_info.get("downloads").and_then(|d| d.as_u64()).unwrap_or(0);
    let likes = model_info.get("likes").and_then(|l| l.as_u64()).unwrap_or(0);
    let pipeline_tag =
        model_info.get("pipeline_tag").and_then(|p| p.as_str()).map(|s| s.to_string());

    // Get file information for size calculation
    let files = list_repo_files(model_id, revision, None)?;
    let total_size: u64 = files.iter().map(|f| f.size).sum();
    let file_count = files.len();

    Ok(ModelDownloadInfo {
        model_id: model_id.to_string(),
        revision: revision.to_string(),
        total_size,
        file_count,
        downloads,
        likes,
        pipeline_tag,
        essential_files: files.iter().filter(|f| is_essential_file(&f.path)).cloned().collect(),
        estimated_download_time: estimate_download_time(total_size),
    })
}

/// Check if a file is essential for model operation
#[cfg(feature = "hub")]
fn is_essential_file(filename: &str) -> bool {
    let essential_files = [
        "config.json",
        "pytorch_model.bin",
        "model.safetensors",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
    ];

    essential_files.contains(&filename) || filename.ends_with(".safetensors")
}

/// Estimate download time based on file size
#[cfg(feature = "hub")]
fn estimate_download_time(total_size: u64) -> Duration {
    // Assume average download speed of 10 MB/s
    let average_speed_mbps = 10.0 * 1024.0 * 1024.0;
    let estimated_seconds = total_size as f64 / average_speed_mbps;
    Duration::from_secs(estimated_seconds as u64)
}

/// Model download information
#[derive(Debug, Clone)]
pub struct ModelDownloadInfo {
    pub model_id: String,
    pub revision: String,
    pub total_size: u64,
    pub file_count: usize,
    pub downloads: u64,
    pub likes: u64,
    pub pipeline_tag: Option<String>,
    pub essential_files: Vec<RepoFile>,
    pub estimated_download_time: Duration,
}

impl ModelDownloadInfo {
    pub fn size_mb(&self) -> f64 {
        self.total_size as f64 / (1024.0 * 1024.0)
    }

    pub fn size_gb(&self) -> f64 {
        self.total_size as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Check if delta compression is available for a model update
#[cfg(feature = "hub")]
pub async fn check_delta_availability(
    model_id: &str,
    from_revision: &str,
    to_revision: &str,
) -> Result<Option<DeltaInfo>> {
    // A real GET against a hypothetical delta-serving endpoint. The public
    // Hugging Face Hub does not currently expose `/api/models/*/deltas/*`, so
    // in practice this will honestly resolve to `Ok(None)` (a 404) rather
    // than fabricating delta availability; a Hub-compatible server that does
    // implement the endpoint would be answered for real, with no code change
    // needed here.
    let delta_url = format!(
        "{}/api/models/{}/deltas/{}/{}",
        HF_HUB_URL, model_id, from_revision, to_revision
    );

    let client = AsyncClient::new();
    let response = client.get(&delta_url).send().await;

    match response {
        Ok(resp) if resp.status().is_success() => {
            let delta_info: DeltaInfo = resp.json().await.map_err(|e| {
                TrustformersError::invalid_input(
                    format!("Failed to parse delta info: {}", e),
                    Some("delta_response"),
                    Some("valid DeltaInfo JSON object"),
                    Some("invalid JSON format"),
                )
            })?;
            Ok(Some(delta_info))
        },
        _ => Ok(None), // Delta not available
    }
}

/// Download a specific file from the Hub
pub fn download_file_from_hub(
    model_id: &str,
    filename: &str,
    options: Option<HubOptions>,
) -> Result<PathBuf> {
    let opts = options.unwrap_or_default();
    let revision = opts.revision.as_deref().unwrap_or("main");

    // Get cache directory
    let cache_dir = match opts.cache_dir {
        Some(dir) => dir,
        None => get_cache_dir()?,
    };
    let model_dir = cache_dir.join("models").join(model_id.replace('/', "--")).join(revision);

    let file_path = model_dir.join(filename);

    // Check if already cached and not forcing download
    if !opts.force_download && file_path.exists() {
        return Ok(file_path);
    }

    // Create model directory
    fs::create_dir_all(&model_dir).map_err(|e| TrustformersError::Io {
        message: format!("Failed to create model directory: {}", e),
        path: Some(model_dir.to_string_lossy().to_string()),
        suggestion: Some("Check cache directory permissions and disk space".to_string()),
    })?;

    // Download the file
    let download_url = format!(
        "{}/{}/resolve/{}/{}",
        HF_HUB_URL, model_id, revision, filename
    );

    download_file(&download_url, &file_path, opts.token.as_deref(), None)?;

    Ok(file_path)
}

/// Load a model configuration from the Hub
pub fn load_config_from_hub(
    model_id: &str,
    options: Option<HubOptions>,
) -> Result<serde_json::Value> {
    let config_path = download_file_from_hub(model_id, "config.json", options)?;
    let config_str = std::fs::read_to_string(&config_path).map_err(|e| TrustformersError::Io {
        message: format!("Failed to read config file: {}", e),
        path: Some(config_path.to_string_lossy().to_string()),
        suggestion: Some("Check file existence and permissions".to_string()),
    })?;
    serde_json::from_str(&config_str).map_err(|e| {
        TrustformersError::invalid_input(
            format!("Failed to parse config: {}", e),
            Some("config_json"),
            Some("valid JSON format"),
            Some("invalid JSON"),
        )
    })
}

/// Load model weights from the Hub (supports SafeTensors format)
pub fn load_weights_from_hub(
    model_id: &str,
    options: Option<HubOptions>,
) -> Result<Box<dyn crate::core::traits::WeightReader>> {
    // Try SafeTensors first
    let safetensors_path = download_file_from_hub(model_id, "model.safetensors", options.clone());

    if let Ok(path) = safetensors_path {
        let reader = crate::core::utils::weight_loading::SafeTensorsReader::from_file(&path)?;
        return Ok(Box::new(reader));
    }

    // Fall back to PyTorch format if SafeTensors not available
    let pytorch_formats = ["pytorch_model.bin", "model.pt", "pytorch_model.pt"];

    for pytorch_file in &pytorch_formats {
        if let Ok(path) = download_file_from_hub(model_id, pytorch_file, options.clone()) {
            let reader = crate::core::utils::weight_loading::PyTorchReader::from_file(&path)?;
            return Ok(Box::new(reader));
        }
    }

    // If neither SafeTensors nor PyTorch formats are found
    Err(TrustformersError::Core(CoreTrustformersError::other(
        format!("No supported weight format found for model {}: Tried SafeTensors (.safetensors), PyTorch (.bin, .pt)", model_id)
    )))
}

/// Parse model card from README.md
fn parse_model_card_from_readme(content: &str) -> Result<ModelCard> {
    // Look for YAML frontmatter in the README
    if let Some(yaml_start) = content.find("---\n") {
        if let Some(yaml_end) = content[yaml_start + 4..].find("\n---") {
            let yaml_content = &content[yaml_start + 4..yaml_start + 4 + yaml_end];

            // Parse YAML frontmatter
            let yaml_value: serde_yaml_ng::Value =
                serde_yaml_ng::from_str(yaml_content).map_err(|e| {
                    TrustformersError::invalid_input(
                        format!("Failed to parse YAML frontmatter: {}", e),
                        Some("yaml_frontmatter"),
                        Some("valid YAML format"),
                        Some("invalid YAML"),
                    )
                })?;

            // Convert YAML to JSON for easier handling with serde_json
            let json_value = serde_json::to_value(yaml_value).map_err(|e| {
                TrustformersError::invalid_input(
                    format!("Failed to convert YAML to JSON: {}", e),
                    Some("yaml_content"),
                    Some("YAML convertible to JSON"),
                    Some("incompatible YAML structure"),
                )
            })?;

            // Parse as ModelCard
            let model_card: ModelCard = serde_json::from_value(json_value).map_err(|e| {
                TrustformersError::invalid_input(
                    format!("Failed to parse model card: {}", e),
                    Some("model_card_json"),
                    Some("valid ModelCard structure"),
                    Some("invalid model card format"),
                )
            })?;

            return Ok(model_card);
        }
    }

    // If no YAML frontmatter found, return empty model card
    Ok(ModelCard {
        license: None,
        language: None,
        tags: None,
        datasets: None,
        metrics: None,
        widget: None,
        model_index: None,
        thumbnail: None,
        pipeline_tag: None,
        inference: None,
        extra: serde_json::Map::new(),
    })
}

/// Load a model card from the Hub
pub fn load_model_card_from_hub(model_id: &str, options: Option<HubOptions>) -> Result<ModelCard> {
    // Try to download README.md
    let readme_path = download_file_from_hub(model_id, "README.md", options);

    if let Ok(path) = readme_path {
        let readme_content = std::fs::read_to_string(&path).map_err(|e| TrustformersError::Io {
            message: format!("Failed to read README.md: {}", e),
            path: Some(path.to_string_lossy().to_string()),
            suggestion: Some("Check file existence and permissions".to_string()),
        })?;

        parse_model_card_from_readme(&readme_content)
    } else {
        // If README.md not found, return empty model card
        Ok(ModelCard {
            license: None,
            language: None,
            tags: None,
            datasets: None,
            metrics: None,
            widget: None,
            model_index: None,
            thumbnail: None,
            pipeline_tag: None,
            inference: None,
            extra: serde_json::Map::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_dir() {
        let cache_dir = get_cache_dir();
        assert!(cache_dir.is_ok());
    }

    #[test]
    fn test_is_cached() {
        let result = is_cached("bert-base-uncased", None);
        assert!(result.is_ok());
    }

    #[test]
    #[cfg(feature = "hub")]
    fn test_format_cache_cleanup_start_message_reports_directory_and_sizes() {
        // Regression test: `cleanup_cache`'s `cache_dir` parameter used to be
        // computed and passed in but never read. Guard against that regressing.
        let message =
            format_cache_cleanup_start_message(Path::new("/tmp/trustformers-cache"), 1_000, 800);
        assert!(message.contains("/tmp/trustformers-cache"));
        assert!(message.contains("1000 bytes"));
        assert!(message.contains("target 800 bytes"));
    }

    #[test]
    fn test_hub_options_default() {
        let opts = HubOptions::default();
        assert_eq!(opts.revision, Some("main".to_string()));
        assert!(opts.cache_dir.is_none());
        assert!(!opts.force_download);
        assert!(opts.token.is_none());
        assert!(opts.parallel_downloads);
        assert_eq!(opts.max_concurrent_downloads, 4);
        assert!(opts.enable_resumable_downloads);
        assert!(opts.enable_delta_compression);
        assert_eq!(opts.chunk_size, 8 * 1024 * 1024);
        assert_eq!(opts.timeout_seconds, 300);
        assert_eq!(opts.retry_attempts, 3);
        assert!(opts.use_cdn);
        assert!(opts.smart_caching);
    }

    #[test]
    fn test_download_config_default() {
        let config = DownloadConfig::default();
        assert!(config.parallel_downloads);
        assert_eq!(config.max_concurrent, 4);
        assert!(config.enable_resumable);
        assert!(config.enable_compression);
        assert_eq!(config.chunk_size, 8 * 1024 * 1024);
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert_eq!(config.retry_attempts, 3);
        assert!(config.verify_checksums);
        assert!(config.progress_reporting);
    }

    #[test]
    fn test_download_stats_default() {
        let stats = DownloadStats::default();
        assert_eq!(stats.total_files, 0);
        assert_eq!(stats.downloaded_files, 0);
        assert_eq!(stats.failed_files, 0);
        assert_eq!(stats.total_bytes, 0);
        assert_eq!(stats.downloaded_bytes, 0);
    }

    #[test]
    fn test_download_stats_success_rate_empty() {
        let stats = DownloadStats::default();
        assert!((stats.success_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_download_stats_success_rate_partial() {
        let stats = DownloadStats {
            total_files: 10,
            downloaded_files: 7,
            failed_files: 3,
            ..DownloadStats::default()
        };
        assert!((stats.success_rate() - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_download_stats_success_rate_all() {
        let stats = DownloadStats {
            total_files: 5,
            downloaded_files: 5,
            failed_files: 0,
            ..DownloadStats::default()
        };
        assert!((stats.success_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_download_stats_duration_none() {
        let stats = DownloadStats::default();
        assert!(stats.duration().is_none());
    }

    #[test]
    fn test_download_stats_duration_with_times() {
        let start = Instant::now();
        let stats = DownloadStats {
            start_time: Some(start),
            end_time: Some(start),
            ..DownloadStats::default()
        };
        let dur = stats.duration();
        assert!(dur.is_some());
    }

    #[test]
    fn test_cdn_config_default() {
        let config = CdnConfig::default();
        assert!(!config.primary_urls.is_empty());
        assert!(!config.fallback_urls.is_empty());
        assert_eq!(config.health_check_interval, Duration::from_secs(300));
        assert_eq!(config.latency_threshold, Duration::from_millis(1000));
        assert!(config.enable_geographic_routing);
        assert!(!config.region_preferences.is_empty());
    }

    #[test]
    fn test_smart_cache_config_default() {
        let config = SmartCacheConfig::default();
        assert!((config.max_cache_size_gb - 50.0).abs() < f64::EPSILON);
        assert!((config.cleanup_threshold - 0.9).abs() < f64::EPSILON);
        // Weights should sum to approximately 1.0
        let total = config.access_weight
            + config.frequency_weight
            + config.recency_weight
            + config.size_penalty;
        assert!((total - 1.0).abs() < f64::EPSILON);
        assert!(config.enable_predictive_caching);
        assert!(config.enable_compression);
    }

    #[test]
    fn test_resume_info_can_resume_recent() {
        let info = ResumeInfo {
            url: "https://example.com/file".to_string(),
            local_path: std::env::temp_dir().join("file"),
            expected_size: 1000,
            downloaded_size: 500,
            checksum: None,
            last_modified: None,
            created_at: Instant::now(),
        };
        assert!(info.can_resume(Duration::from_secs(3600)));
    }

    #[test]
    fn test_resume_info_cannot_resume_zero_downloaded() {
        let info = ResumeInfo {
            url: "https://example.com/file".to_string(),
            local_path: std::env::temp_dir().join("file"),
            expected_size: 1000,
            downloaded_size: 0,
            checksum: None,
            last_modified: None,
            created_at: Instant::now(),
        };
        assert!(!info.can_resume(Duration::from_secs(3600)));
    }

    #[test]
    fn test_delta_info_creation() {
        let delta = DeltaInfo {
            base_version: "v1".to_string(),
            target_version: "v2".to_string(),
            delta_url: "https://example.com/delta".to_string(),
            compression_ratio: 0.3,
            delta_size: 30_000_000,
            delta_checksum: None,
            full_size: 100_000_000,
        };
        assert!(delta.delta_size < delta.full_size);
        assert!((delta.compression_ratio - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_model_info_creation() {
        let info = ModelInfo {
            model_id: "bert-base".to_string(),
            sha: "abc123".to_string(),
            pipeline_tag: Some("text-classification".to_string()),
            library_name: Some("trustformers".to_string()),
            downloads: 10000,
            likes: 500,
        };
        assert_eq!(info.model_id, "bert-base");
        assert!(info.pipeline_tag.is_some());
        assert!(info.downloads > 0);
    }

    #[test]
    fn test_hub_options_custom() {
        let opts = HubOptions {
            revision: Some("develop".to_string()),
            cache_dir: Some(PathBuf::from("/custom/cache")),
            force_download: true,
            token: Some("hf_token".to_string()),
            parallel_downloads: false,
            max_concurrent_downloads: 1,
            enable_resumable_downloads: false,
            enable_delta_compression: false,
            chunk_size: 1024 * 1024,
            timeout_seconds: 60,
            retry_attempts: 1,
            use_cdn: false,
            cdn_urls: vec![],
            smart_caching: false,
        };
        assert!(opts.force_download);
        assert!(!opts.parallel_downloads);
        assert_eq!(opts.max_concurrent_downloads, 1);
    }

    #[test]
    fn test_download_config_custom() {
        let config = DownloadConfig {
            parallel_downloads: false,
            max_concurrent: 1,
            enable_resumable: false,
            enable_compression: false,
            chunk_size: 1024,
            timeout: Duration::from_secs(10),
            retry_attempts: 0,
            verify_checksums: false,
            progress_reporting: false,
        };
        assert!(!config.parallel_downloads);
        assert_eq!(config.retry_attempts, 0);
    }

    #[test]
    fn test_is_cached_nonexistent_model() {
        let result = is_cached("nonexistent-model-xyz-123", None);
        assert!(result.is_ok());
        assert!(!result.expect("Operation failed"));
    }

    #[test]
    fn test_is_cached_with_revision() {
        let result = is_cached("bert-base", Some("v1.0"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_cdn_config_primary_url_count() {
        let config = CdnConfig::default();
        assert_eq!(config.primary_urls.len(), 2);
    }

    #[test]
    fn test_hub_options_cdn_urls_default() {
        let opts = HubOptions::default();
        assert_eq!(opts.cdn_urls.len(), 2);
    }

    #[cfg(feature = "hub")]
    #[test]
    fn test_download_manager_creation() {
        let config = DownloadConfig::default();
        let manager = DownloadManager::new(config);
        assert_eq!(manager.stats.total_files, 0);
    }

    // ── Binary delta codec (create_binary_delta / reconstruct_from_delta) ──
    //
    // Regression tests for the historical bug: `reconstruct_from_delta` used
    // to XOR the delta bytes over the base file and return whatever came out,
    // with no checksum check, so *any* base/delta pair "succeeded" — even a
    // delta that had nothing to do with the base. These tests would all have
    // failed against that code.

    fn delta_test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("trustformers_hub_delta_{name}"));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn test_create_and_reconstruct_from_delta_round_trips() {
        let dir = delta_test_dir("round_trip");
        let base_path = dir.join("base.bin");
        let target_path = dir.join("target.bin");
        fs::write(
            &base_path,
            b"the quick brown fox jumps over the lazy dog".repeat(8),
        )
        .expect("write base");
        let mut target_content = b"the quick brown fox jumps over the lazy dog".repeat(8);
        target_content.extend_from_slice(b" ...with a tail appended for the new version");
        fs::write(&target_path, &target_content).expect("write target");

        let delta = create_binary_delta(&base_path, &target_path).expect("create_binary_delta");
        // A real delta between two very similar files should be much smaller
        // than the target itself, proving actual matching happened.
        assert!(delta.len() < target_content.len());

        let base_data = fs::read(&base_path).expect("read base");
        let reconstructed = reconstruct_from_delta(&base_data, &delta).expect("reconstruct");
        assert_eq!(reconstructed, target_content);

        fs::remove_dir_all(&dir).ok();
    }

    /// The old implementation would XOR *any* delta over *any* base and
    /// return `Ok(_)`. A delta generated for one base must now be refused
    /// when applied against an unrelated base, instead of silently producing
    /// a corrupted result.
    #[test]
    fn test_reconstruct_from_delta_rejects_mismatched_base() {
        let dir = delta_test_dir("mismatched_base");
        let base_path = dir.join("base.bin");
        let target_path = dir.join("target.bin");
        fs::write(&base_path, b"original base content for this model version").expect("write base");
        fs::write(
            &target_path,
            b"updated target content for the next model version",
        )
        .expect("write target");

        let delta = create_binary_delta(&base_path, &target_path).expect("create_binary_delta");

        let unrelated_base = b"a totally unrelated file that is not the real base".to_vec();
        let result = reconstruct_from_delta(&unrelated_base, &delta);
        assert!(
            result.is_err(),
            "applying a delta to the wrong base must error, not silently corrupt"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_reconstruct_from_delta_rejects_non_delta_bytes() {
        let base = b"any base file content".to_vec();
        let not_a_delta = b"this is just some file that happens to exist on disk".to_vec();
        let result = reconstruct_from_delta(&base, &not_a_delta);
        assert!(
            result.is_err(),
            "arbitrary bytes must never be accepted as a delta"
        );
    }

    #[test]
    fn test_create_binary_delta_missing_base_file_errors() {
        let dir = delta_test_dir("missing_base");
        let target_path = dir.join("target.bin");
        fs::write(&target_path, b"target content").expect("write target");

        let result = create_binary_delta(&dir.join("does-not-exist.bin"), &target_path);
        assert!(result.is_err());

        fs::remove_dir_all(&dir).ok();
    }

    /// End-to-end at the `DownloadManager` level: given a base file and a
    /// (locally-produced) delta file on disk, `apply_binary_delta` writes a
    /// target file whose *actual bytes on disk* match the original target —
    /// exercising the temp-file-then-rename path, not just the in-memory
    /// codec functions above.
    #[cfg(feature = "hub")]
    #[tokio::test]
    async fn test_download_manager_apply_binary_delta_writes_verified_target() {
        let dir = delta_test_dir("apply_binary_delta");
        let base_path = dir.join("base.safetensors");
        let target_path = dir.join("target.safetensors");
        let delta_path = dir.join("update.delta");

        let base_content: Vec<u8> = (0..4096u32).map(|i| (i % 256) as u8).collect();
        let mut target_content = base_content.clone();
        target_content.truncate(2048);
        target_content.extend_from_slice(b"freshly appended tensor bytes for the new revision");
        fs::write(&base_path, &base_content).expect("write base");
        fs::write(&target_path, &target_content).expect("write target");

        let delta = create_binary_delta(&base_path, &target_path).expect("create_binary_delta");
        fs::write(&delta_path, &delta).expect("write delta");
        // Overwrite the "real" target so we can prove apply_binary_delta
        // reconstructs it fresh rather than the file already being correct.
        fs::write(&target_path, b"stale content that must be replaced").expect("clobber target");

        let manager = DownloadManager::new(DownloadConfig::default());
        manager
            .apply_binary_delta(&delta_path, &base_path, &target_path)
            .await
            .expect("apply_binary_delta");

        let final_bytes = fs::read(&target_path).expect("read final target");
        assert_eq!(final_bytes, target_content);

        fs::remove_dir_all(&dir).ok();
    }

    /// Regression test: applying a delta against the wrong base must leave
    /// whatever was already at `target_path` untouched rather than
    /// overwriting it with corrupted bytes — verifying the temp-file-then-
    /// rename ordering actually protects the destination on failure.
    #[cfg(feature = "hub")]
    #[tokio::test]
    async fn test_download_manager_apply_binary_delta_never_corrupts_target_on_mismatch() {
        let dir = delta_test_dir("apply_binary_delta_failure");
        let base_path = dir.join("base.safetensors");
        let target_path = dir.join("target.safetensors");
        let delta_path = dir.join("update.delta");
        let wrong_base_path = dir.join("wrong_base.safetensors");

        fs::write(&base_path, b"the real base file contents").expect("write base");
        fs::write(&wrong_base_path, b"a completely different base file").expect("write wrong base");
        let target_content = b"the real, correct target file contents".to_vec();
        fs::write(&target_path, &target_content).expect("write target");

        let delta = create_binary_delta(&base_path, &target_path).expect("create_binary_delta");
        fs::write(&delta_path, &delta).expect("write delta");

        let sentinel = b"pre-existing target bytes that must survive a failed apply".to_vec();
        fs::write(&target_path, &sentinel).expect("seed target with sentinel content");

        let manager = DownloadManager::new(DownloadConfig::default());
        let result = manager.apply_binary_delta(&delta_path, &wrong_base_path, &target_path).await;
        assert!(result.is_err(), "applying against the wrong base must fail");

        // target_path must be untouched: still the sentinel, not corrupted.
        let bytes_after = fs::read(&target_path).expect("read target after failed apply");
        assert_eq!(bytes_after, sentinel);

        fs::remove_dir_all(&dir).ok();
    }
}
