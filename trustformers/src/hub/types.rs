//! Plain configuration and data-transfer types for Hub downloads: options, stats, repo/model/LFS metadata.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Cache file information
#[derive(Debug, Clone)]
pub struct CacheFileInfo {
    pub path: PathBuf,
    pub size: u64,
    pub access_time: std::time::SystemTime,
    pub score: f64,
}
/// Cache usage information
#[derive(Debug, Clone, Default)]
pub struct CacheUsage {
    pub total_size: u64,
    pub file_count: usize,
    pub files: Vec<CacheFileInfo>,
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
/// Download task definition
#[derive(Debug, Clone)]
pub struct DownloadTask {
    pub url: String,
    pub local_path: PathBuf,
    pub filename: String,
    pub expected_size: u64,
    pub expected_checksum: Option<String>,
}
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LfsInfo {
    pub sha256: String,
    pub size: u64,
    #[serde(rename = "pointerSize")]
    pub pointer_size: u64,
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
/// File information from the Hub API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoFile {
    pub path: String,
    pub size: u64,
    #[serde(rename = "lfs")]
    pub lfs: Option<LfsInfo>,
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
pub(super) const HF_HUB_URL: &str = "https://huggingface.co";
