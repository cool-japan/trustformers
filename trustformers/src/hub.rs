use crate::error::{Result, TrustformersError};
use futures::stream::{self, StreamExt};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use reqwest::{blocking::Client, Client as AsyncClient};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
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
pub struct DownloadManager {
    config: DownloadConfig,
    cdn_config: CdnConfig,
    cache_config: SmartCacheConfig,
    client: AsyncClient,
    resume_db: HashMap<String, ResumeInfo>,
    stats: DownloadStats,
}

impl DownloadManager {
    pub fn new(config: DownloadConfig) -> Self {
        let client = AsyncClient::builder().timeout(config.timeout).build().unwrap();

        Self {
            config,
            cdn_config: CdnConfig::default(),
            cache_config: SmartCacheConfig::default(),
            client,
            resume_db: HashMap::new(),
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
                        .unwrap()
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
                    let _permit = semaphore.acquire().await.unwrap();
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

    /// Apply delta compression if available
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
            expected_checksum: None,
        };

        Self::download_single_file_async(
            self.client.clone(),
            task,
            None,
            self.config.clone(),
            ProgressBar::hidden(),
        )
        .await?;

        // Apply delta (simplified implementation)
        // In a real implementation, this would use a proper binary diff algorithm
        self.apply_binary_delta(&delta_path, base_path, target_path).await?;

        // Clean up delta file
        fs::remove_file(delta_path).ok();

        Ok(())
    }

    async fn apply_binary_delta(
        &self,
        delta_path: &Path,
        base_path: &Path,
        target_path: &Path,
    ) -> Result<()> {
        // Simplified delta application - in reality you'd use bsdiff/xdelta3 or similar
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

        // This is a placeholder - real delta application would reconstruct the target file
        let target_data = Self::reconstruct_from_delta(&base_data, &delta_data)?;

        fs::write(target_path, target_data).map_err(|e| TrustformersError::Io {
            message: format!("Failed to write target file: {}", e),
            path: Some(target_path.to_string_lossy().to_string()),
            suggestion: Some("Check permissions and disk space".to_string()),
        })?;

        Ok(())
    }

    fn reconstruct_from_delta(base_data: &[u8], delta_data: &[u8]) -> Result<Vec<u8>> {
        // Placeholder implementation - real implementation would use proper binary diff
        // For now, just return the base data modified by delta
        let mut result = base_data.to_vec();

        // Simple example: XOR the delta with the base
        for (i, &delta_byte) in delta_data.iter().enumerate() {
            if i < result.len() {
                result[i] ^= delta_byte;
            }
        }

        Ok(result)
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

/// Legacy synchronous download function (maintained for compatibility)
fn download_file(
    url: &str,
    path: &Path,
    token: Option<&str>,
    expected_sha: Option<&str>,
) -> Result<()> {
    // Create a basic download task and use the async implementation
    let rt = tokio::runtime::Runtime::new().unwrap();
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
pub fn download_model(model_id: &str, options: Option<HubOptions>) -> Result<PathBuf> {
    let opts = options.unwrap_or_default();
    let revision = opts.revision.as_deref().unwrap_or("main");

    // Get cache directory
    let cache_dir = opts.cache_dir.unwrap_or_else(|| get_cache_dir().unwrap());
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
pub async fn download_model_enhanced(
    model_id: &str,
    options: Option<HubOptions>,
) -> Result<(PathBuf, DownloadStats)> {
    let opts = options.unwrap_or_default();
    let revision = opts.revision.as_deref().unwrap_or("main");

    // Get cache directory
    let cache_dir = opts.cache_dir.unwrap_or_else(|| get_cache_dir().unwrap());
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
pub async fn check_delta_availability(
    model_id: &str,
    from_revision: &str,
    to_revision: &str,
) -> Result<Option<DeltaInfo>> {
    // This is a placeholder implementation
    // In reality, you would check the hub API for delta information
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
    let cache_dir = opts.cache_dir.unwrap_or_else(|| get_cache_dir().unwrap());
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
            let yaml_value: serde_yaml::Value =
                serde_yaml::from_str(yaml_content).map_err(|e| {
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
}
