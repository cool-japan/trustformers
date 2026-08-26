//! The `DownloadManager` engine: concurrent, resumable, cache-aware Hub downloads (requires the `hub` feature).
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

// This whole module (the `DownloadManager` engine and its cache-cleanup
// helper) is `#[cfg(feature = "hub")]`-gated, so every import below is too —
// unlike in the original single-file `hub.rs`, none of these are shared with
// non-hub-gated code living elsewhere in this file any more.
#[cfg(feature = "hub")]
use crate::error::{Result, TrustformersError};
#[cfg(feature = "hub")]
use futures::stream::{self, StreamExt};
#[cfg(feature = "hub")]
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
#[cfg(feature = "hub")]
use reqwest::Client as AsyncClient;
#[cfg(feature = "hub")]
use sha2::{Digest, Sha256};
#[cfg(feature = "hub")]
use std::fs;
#[cfg(feature = "hub")]
use std::fs::{File, OpenOptions};
#[cfg(feature = "hub")]
use std::io::{Seek, SeekFrom, Write};
#[cfg(feature = "hub")]
use std::path::Path;
#[cfg(feature = "hub")]
use std::sync::Arc;
#[cfg(feature = "hub")]
use std::time::{Duration, Instant};
#[cfg(feature = "hub")]
use tokio::sync::Semaphore;
#[cfg(feature = "hub")]
use trustformers_core::errors::TrustformersError as CoreTrustformersError;

#[cfg(feature = "hub")]
use super::delta::{reconstruct_from_delta, DeltaInfo};
#[cfg(feature = "hub")]
use super::types::{
    CacheFileInfo, CacheUsage, DownloadConfig, DownloadStats, DownloadTask, SmartCacheConfig,
};

/// Enhanced download manager with parallel and resumable downloads
///
/// Note: this does not yet route through [`CdnConfig`](super::types::CdnConfig) or
/// persist [`ResumeInfo`](super::types::ResumeInfo) — real resumable downloads
/// work off the on-disk file's own `fs::metadata().len()` in
/// `download_single_file_async`, and `RepoFile`
/// URLs are used as-is rather than being routed through a CDN's
/// primary/fallback host list. `HubOptions::use_cdn` therefore does not yet
/// change request routing; both types remain available (`CdnConfig`,
/// `ResumeInfo`) for a future CDN-routing/resume-persistence implementation.
#[cfg(feature = "hub")]
pub struct DownloadManager {
    pub(super) config: DownloadConfig,
    pub(super) cache_config: SmartCacheConfig,
    pub(super) client: AsyncClient,
    pub(super) stats: DownloadStats,
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
    pub(super) async fn download_single_file_async(
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

    pub(super) async fn attempt_download(
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

    pub(super) fn prepare_file_for_download(path: &Path, enable_resumable: bool) -> Result<File> {
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

    pub(super) fn calculate_final_stats(&mut self) {
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
    /// `crate::hub_delta_codec`): `apply_binary_delta`
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

    pub(super) async fn apply_binary_delta(
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

    pub(super) fn calculate_cache_usage(&self, cache_dir: &Path) -> Result<CacheUsage> {
        let mut usage = CacheUsage::default();
        self.scan_cache_directory(cache_dir, &mut usage)?;
        Ok(usage)
    }

    pub(super) fn scan_cache_directory(&self, dir: &Path, usage: &mut CacheUsage) -> Result<()> {
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

    pub(super) fn calculate_cache_score(&self, metadata: &fs::Metadata) -> f64 {
        let now = std::time::SystemTime::now();
        let access_time = metadata.accessed().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        let recency = now.duration_since(access_time).unwrap_or(Duration::ZERO).as_secs() as f64;

        // Simple scoring based on recency and size
        let recency_score = 1.0 / (1.0 + recency / 86400.0); // Decay over days
        let size_penalty = (metadata.len() as f64).log10() * self.cache_config.size_penalty;

        (recency_score * self.cache_config.recency_weight) - size_penalty
    }

    pub(super) fn cleanup_cache(
        &mut self,
        cache_dir: &Path,
        usage: &CacheUsage,
        max_size: u64,
    ) -> Result<()> {
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
pub(super) fn format_cache_cleanup_start_message(
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
