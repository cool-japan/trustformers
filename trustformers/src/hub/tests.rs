//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use std::fs;
#[cfg(feature = "hub")]
use std::path::Path;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use super::*;

#[cfg(test)]
mod tests_2 {
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
