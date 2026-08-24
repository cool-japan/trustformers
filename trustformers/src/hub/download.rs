//! Free functions for locating, downloading, and reporting on Hub model files.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, TrustformersError};
#[cfg(feature = "hub")]
use indicatif::ProgressBar;
#[cfg(feature = "hub")]
use reqwest::{blocking::Client, Client as AsyncClient};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
#[cfg(feature = "hub")]
use std::time::Instant;
use trustformers_core::errors::TrustformersError as CoreTrustformersError;

#[cfg(feature = "hub")]
use super::manager::DownloadManager;
use super::types::{DownloadConfig, HubOptions, HF_HUB_URL};
#[cfg(feature = "hub")]
use super::types::{DownloadStats, DownloadTask, ModelDownloadInfo, RepoFile};

/// Download scenarios for optimized configurations
#[derive(Debug, Clone, Copy)]
pub enum DownloadScenario {
    FastDevelopment,
    Production,
    BandwidthLimited,
    Reliable,
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
