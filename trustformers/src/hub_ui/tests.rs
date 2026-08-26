//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;

    #[test]
    fn test_model_repository_creation() {
        let repo = ModelRepository::new("test/model".to_string(), "test_user".to_string());

        assert_eq!(repo.model_id, "test/model");
        assert_eq!(repo.metadata.owner, "test_user");
        assert_eq!(repo.versions.len(), 0);
        assert_eq!(repo.version_history.len(), 0);
    }

    #[test]
    fn test_version_creation() {
        let version = ModelVersion {
            version: "v1.0.0".to_string(),
            name: Some("Initial release".to_string()),
            description: Some("First stable version".to_string()),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs(),
            modified_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs(),
            author: Some("test_user".to_string()),
            tags: vec!["stable".to_string()],
            metrics: None,
            changes: Vec::new(),
            parent_version: None,
            download_stats: None,
            size_bytes: 1024 * 1024 * 100, // 100MB
            checksum: Some("abc123".to_string()),
            status: VersionStatus::Stable,
            compatibility: CompatibilityInfo {
                framework_version: Some("trustformers>=0.1.0".to_string()),
                python_version: Some(">=3.8".to_string()),
                cuda_version: None,
                hardware_requirements: Vec::new(),
                breaking_changes: Vec::new(),
                migration_notes: None,
            },
        };

        assert_eq!(version.version, "v1.0.0");
        assert_eq!(version.status, VersionStatus::Stable);
        assert!(version.name.is_some());
    }

    fn test_version(version: &str) -> ModelVersion {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs();
        ModelVersion {
            version: version.to_string(),
            name: None,
            description: None,
            created_at: now,
            modified_at: now,
            author: None,
            tags: Vec::new(),
            metrics: None,
            changes: Vec::new(),
            parent_version: None,
            download_stats: None,
            size_bytes: 0,
            checksum: None,
            status: VersionStatus::Stable,
            compatibility: CompatibilityInfo {
                framework_version: None,
                python_version: None,
                cuda_version: None,
                hardware_requirements: Vec::new(),
                breaking_changes: Vec::new(),
                migration_notes: None,
            },
        }
    }

    /// Regression test: `create_version` used to ignore the URL's
    /// `:version` path segment entirely, so POSTing to `.../versions/v1`
    /// with a body claiming to be `v2` silently created `v2` instead of
    /// rejecting the mismatch.
    #[tokio::test]
    async fn test_create_version_rejects_url_body_version_mismatch() {
        let state = HubUiState::new(HubUiConfig::default(), std::env::temp_dir());
        state
            .add_repository(ModelRepository::new(
                "test/model".to_string(),
                "test_user".to_string(),
            ))
            .expect("add_repository should succeed");

        let result = create_version(
            State(state.clone()),
            Path(("test/model".to_string(), "v1".to_string())),
            Json(test_version("v2")),
        )
        .await;

        assert_eq!(result.err(), Some(StatusCode::BAD_REQUEST));
        // Confirm nothing was actually created under either name.
        let repo = state.get_repository("test/model").expect("repository should still exist");
        assert!(repo.get_version("v1").is_none());
        assert!(repo.get_version("v2").is_none());
    }

    /// Regression test: with a matching URL/body version, `create_version`
    /// must still succeed (the mismatch check above must not be overly
    /// strict).
    #[tokio::test]
    async fn test_create_version_accepts_matching_url_and_body_version() {
        let state = HubUiState::new(HubUiConfig::default(), std::env::temp_dir());
        state
            .add_repository(ModelRepository::new(
                "test/model".to_string(),
                "test_user".to_string(),
            ))
            .expect("add_repository should succeed");

        let result = create_version(
            State(state.clone()),
            Path(("test/model".to_string(), "v1".to_string())),
            Json(test_version("v1")),
        )
        .await;

        assert!(result.is_ok());
        let repo = state.get_repository("test/model").expect("repository should still exist");
        assert!(repo.get_version("v1").is_some());
    }

    #[test]
    fn test_hub_ui_state() {
        let config = HubUiConfig::default();
        let cache_dir = std::env::temp_dir();
        let state = HubUiState::new(config, cache_dir);

        let repo = ModelRepository::new("test/model".to_string(), "test_user".to_string());
        assert!(state.add_repository(repo).is_ok());

        let retrieved = state.get_repository("test/model");
        assert!(retrieved.is_some());

        let repos = state.list_repositories();
        assert_eq!(repos.len(), 1);
    }

    #[test]
    fn test_version_comparison() {
        let mut repo = ModelRepository::new("test/model".to_string(), "test_user".to_string());

        let v1 = ModelVersion {
            version: "v1.0.0".to_string(),
            name: Some("V1".to_string()),
            description: None,
            created_at: 1000,
            modified_at: 1000,
            author: None,
            tags: Vec::new(),
            metrics: Some(ModelMetrics {
                accuracy: Some(0.9),
                loss: Some(0.1),
                inference_speed: Some(100.0),
                memory_usage: Some(1000.0),
                parameter_count: None,
                custom_metrics: HashMap::new(),
                benchmarks: Vec::new(),
            }),
            changes: Vec::new(),
            parent_version: None,
            download_stats: None,
            size_bytes: 1000000,
            checksum: Some("abc".to_string()),
            status: VersionStatus::Stable,
            compatibility: CompatibilityInfo {
                framework_version: None,
                python_version: None,
                cuda_version: None,
                hardware_requirements: Vec::new(),
                breaking_changes: Vec::new(),
                migration_notes: None,
            },
        };

        let v2 = ModelVersion {
            version: "v2.0.0".to_string(),
            name: Some("V2".to_string()),
            description: None,
            created_at: 2000,
            modified_at: 2000,
            author: None,
            tags: Vec::new(),
            metrics: Some(ModelMetrics {
                accuracy: Some(0.95),
                loss: Some(0.05),
                inference_speed: Some(120.0),
                memory_usage: Some(1200.0),
                parameter_count: None,
                custom_metrics: HashMap::new(),
                benchmarks: Vec::new(),
            }),
            changes: Vec::new(),
            parent_version: Some("v1.0.0".to_string()),
            download_stats: None,
            size_bytes: 1200000,
            checksum: Some("def".to_string()),
            status: VersionStatus::Stable,
            compatibility: CompatibilityInfo {
                framework_version: None,
                python_version: None,
                cuda_version: None,
                hardware_requirements: Vec::new(),
                breaking_changes: Vec::new(),
                migration_notes: None,
            },
        };

        repo.versions.insert("v1.0.0".to_string(), v1);
        repo.versions.insert("v2.0.0".to_string(), v2);
        repo.version_history = vec!["v1.0.0".to_string(), "v2.0.0".to_string()];

        let comparison =
            repo.compare_versions("v1.0.0", "v2.0.0").expect("operation failed in test");

        assert_eq!(comparison.from_version, "v1.0.0");
        assert_eq!(comparison.to_version, "v2.0.0");
        assert_eq!(comparison.size_diff, 200000);
        assert!(
            (comparison.performance_diff.accuracy_diff.expect("operation failed in test") - 0.05)
                .abs()
                < 1e-10
        );
        assert!(
            (comparison.performance_diff.loss_diff.expect("operation failed in test") + 0.05).abs()
                < 1e-10
        );
    }

    fn make_version(version: &str, checksum: Option<&str>, size_bytes: u64) -> ModelVersion {
        ModelVersion {
            version: version.to_string(),
            name: None,
            description: None,
            created_at: 0,
            modified_at: 0,
            author: None,
            tags: Vec::new(),
            metrics: None,
            changes: Vec::new(),
            parent_version: None,
            download_stats: None,
            size_bytes,
            checksum: checksum.map(str::to_string),
            status: VersionStatus::Stable,
            compatibility: CompatibilityInfo {
                framework_version: None,
                python_version: None,
                cuda_version: None,
                hardware_requirements: Vec::new(),
                breaking_changes: Vec::new(),
                migration_notes: None,
            },
        }
    }

    fn repo_for_changes_tests() -> ModelRepository {
        ModelRepository::new("test/model".to_string(), "test_user".to_string())
    }

    #[test]
    fn test_compute_changes_no_change_when_checksum_and_size_identical() {
        let repo = repo_for_changes_tests();
        let from = make_version("v1", Some("abc"), 1000);
        let to = make_version("v2", Some("abc"), 1000);
        assert!(repo.compute_changes(&from, &to).is_empty());
    }

    /// Regression test for the P2 bug: the old implementation always
    /// reported a hardcoded `"model.safetensors"` path — a fabricated,
    /// plausible-looking filename with no basis in what actually changed.
    /// It must never appear now; the reported path must clearly signal
    /// "whole model", not a specific (guessed) file.
    #[test]
    fn test_compute_changes_never_fabricates_a_specific_filename() {
        let repo = repo_for_changes_tests();
        let from = make_version("v1", Some("abc"), 1000);
        let to = make_version("v2", Some("def"), 1200);

        let changes = repo.compute_changes(&from, &to);
        assert_eq!(changes.len(), 1);
        assert_ne!(
            changes[0].path, "model.safetensors",
            "must not fabricate a specific filename that was never actually inspected"
        );
        assert_eq!(changes[0].old_size, Some(1000));
        assert_eq!(changes[0].new_size, 1200);
        assert!(changes[0].description.as_deref().unwrap_or_default().contains("checksum"));
    }

    #[test]
    fn test_compute_changes_detects_size_only_change() {
        let repo = repo_for_changes_tests();
        let from = make_version("v1", Some("abc"), 1000);
        let to = make_version("v2", Some("abc"), 2000);

        let changes = repo.compute_changes(&from, &to);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].new_size, 2000);
    }

    #[test]
    fn test_compute_changes_detects_checksum_added() {
        let repo = repo_for_changes_tests();
        let from = make_version("v1", None, 1000);
        let to = make_version("v2", Some("abc"), 1000);

        let changes = repo.compute_changes(&from, &to);
        assert_eq!(changes.len(), 1);
        assert_eq!(
            changes[0].description.as_deref(),
            Some("Integrity checksum added")
        );
    }

    // --- format_timestamp ---

    #[test]
    fn test_format_timestamp_just_now() {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).expect("after epoch").as_secs();
        assert_eq!(format_timestamp(now), "just now");
    }

    #[test]
    fn test_format_timestamp_hours_ago() {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).expect("after epoch").as_secs();
        let three_hours_ago = now - 3 * 3600;
        let result = format_timestamp(three_hours_ago);
        assert!(result.contains("hour"), "{result}");
        assert!(result.ends_with("ago"), "{result}");
    }

    /// Regression test for the underflow bug: the old implementation
    /// computed `now - timestamp` as plain `u64` subtraction, which panics
    /// for any `timestamp` in the future. This must return a normal string
    /// instead.
    #[test]
    fn test_format_timestamp_future_does_not_panic() {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).expect("after epoch").as_secs();
        // Comfortably past the minute/hour boundary so the few milliseconds
        // between computing `now` here and inside `format_timestamp` can
        // never flip which unit gets chosen.
        let two_hours_from_now = now + 2 * 3600 + 120;
        let result = format_timestamp(two_hours_from_now);
        assert!(result.contains("hour"), "{result}");
        assert!(result.ends_with("from now"), "{result}");
    }
}
