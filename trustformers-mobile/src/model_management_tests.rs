#[cfg(test)]
mod tests {
    use crate::model_management::*;
    use std::path::PathBuf;

    fn make_test_config() -> ModelManagerConfig {
        ModelManagerConfig {
            update_server_url: "https://example.com/models".to_string(),
            api_key: None,
            storage_directory: std::env::temp_dir().join("trustformers_model_mgmt_test"),
            max_storage_size_mb: 512,
            enable_auto_updates: false,
            update_check_interval_seconds: 3600,
            enable_differential_updates: true,
            require_signature_verification: false,
            download_timeout_seconds: 60,
            max_concurrent_downloads: 2,
            enable_compression: true,
            download_retry_attempts: 3,
        }
    }

    fn make_compatibility() -> ModelCompatibility {
        ModelCompatibility {
            min_android_api: Some(28),
            min_ios_version: Some("15.0".to_string()),
            required_features: vec!["neon".to_string()],
            min_memory_mb: 256,
            supported_architectures: vec!["arm64".to_string(), "x86_64".to_string()],
        }
    }

    // --- ModelManagerConfig Tests ---

    #[test]
    fn test_model_manager_config_creation() {
        let config = make_test_config();
        assert_eq!(config.max_storage_size_mb, 512);
        assert!(!config.enable_auto_updates);
        assert!(config.enable_differential_updates);
    }

    #[test]
    fn test_model_manager_config_serialization() {
        let config = make_test_config();
        let json = serde_json::to_string(&config).expect("Failed to serialize");
        let deserialized: ModelManagerConfig =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized.max_storage_size_mb, 512);
        assert_eq!(deserialized.download_retry_attempts, 3);
    }

    #[test]
    fn test_model_manager_config_with_api_key() {
        let config = ModelManagerConfig {
            api_key: Some("test-key-123".to_string()),
            ..make_test_config()
        };
        assert!(config.api_key.is_some());
        assert_eq!(config.api_key.as_ref().expect("expected key"), "test-key-123");
    }

    #[test]
    fn test_model_manager_config_clone() {
        let config = make_test_config();
        let cloned = config.clone();
        assert_eq!(cloned.max_storage_size_mb, config.max_storage_size_mb);
        assert_eq!(cloned.update_server_url, config.update_server_url);
    }

    // --- ModelCompatibility Tests ---

    #[test]
    fn test_model_compatibility_creation() {
        let compat = make_compatibility();
        assert_eq!(compat.min_android_api, Some(28));
        assert_eq!(compat.min_memory_mb, 256);
        assert_eq!(compat.supported_architectures.len(), 2);
    }

    #[test]
    fn test_model_compatibility_no_ios() {
        let compat = ModelCompatibility {
            min_android_api: Some(30),
            min_ios_version: None,
            required_features: vec![],
            min_memory_mb: 128,
            supported_architectures: vec!["arm64".to_string()],
        };
        assert!(compat.min_ios_version.is_none());
    }

    #[test]
    fn test_model_compatibility_serialization() {
        let compat = make_compatibility();
        let json = serde_json::to_string(&compat).expect("Failed to serialize");
        let deserialized: ModelCompatibility =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized.min_memory_mb, 256);
    }

    // --- DownloadStatus Tests ---

    #[test]
    fn test_download_status_equality() {
        assert_eq!(DownloadStatus::Pending, DownloadStatus::Pending);
        assert_ne!(DownloadStatus::Pending, DownloadStatus::Downloading);
        assert_eq!(DownloadStatus::Completed, DownloadStatus::Completed);
    }

    #[test]
    fn test_download_status_failed() {
        let status = DownloadStatus::Failed("Network error".to_string());
        if let DownloadStatus::Failed(msg) = &status {
            assert_eq!(msg, "Network error");
        } else {
            panic!("Expected Failed variant");
        }
    }

    #[test]
    fn test_download_status_variants() {
        let statuses = vec![
            DownloadStatus::Pending,
            DownloadStatus::Downloading,
            DownloadStatus::Verifying,
            DownloadStatus::Installing,
            DownloadStatus::Completed,
            DownloadStatus::Failed("err".to_string()),
            DownloadStatus::Cancelled,
        ];
        assert_eq!(statuses.len(), 7);
    }

    // --- UpdateType Tests ---

    #[test]
    fn test_update_type_equality() {
        assert_eq!(UpdateType::Full, UpdateType::Full);
        assert_ne!(UpdateType::Full, UpdateType::Differential);
        assert_ne!(UpdateType::Differential, UpdateType::ConfigOnly);
    }

    #[test]
    fn test_update_type_serialization() {
        let update_type = UpdateType::Differential;
        let json = serde_json::to_string(&update_type).expect("Failed to serialize");
        let deserialized: UpdateType =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized, UpdateType::Differential);
    }

    // --- UpdatePriority Tests ---

    #[test]
    fn test_update_priority_equality() {
        assert_eq!(UpdatePriority::Critical, UpdatePriority::Critical);
        assert_ne!(UpdatePriority::Low, UpdatePriority::High);
    }

    #[test]
    fn test_update_priority_serialization() {
        let priority = UpdatePriority::Critical;
        let json = serde_json::to_string(&priority).expect("Failed to serialize");
        let deserialized: UpdatePriority =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized, UpdatePriority::Critical);
    }

    // --- DownloadProgress Tests ---

    #[test]
    fn test_download_progress_creation() {
        let progress = DownloadProgress {
            model_id: "model-001".to_string(),
            total_bytes: 1024 * 1024,
            downloaded_bytes: 512 * 1024,
            download_speed_bps: 1024.0 * 100.0,
            eta_seconds: 5.0,
            status: DownloadStatus::Downloading,
        };
        assert_eq!(progress.model_id, "model-001");
        assert_eq!(progress.total_bytes, 1024 * 1024);
        assert_eq!(progress.downloaded_bytes, 512 * 1024);
    }

    #[test]
    fn test_download_progress_complete() {
        let progress = DownloadProgress {
            model_id: "model-002".to_string(),
            total_bytes: 1000,
            downloaded_bytes: 1000,
            download_speed_bps: 0.0,
            eta_seconds: 0.0,
            status: DownloadStatus::Completed,
        };
        assert_eq!(progress.total_bytes, progress.downloaded_bytes);
        assert_eq!(progress.status, DownloadStatus::Completed);
    }

    #[test]
    fn test_download_progress_clone() {
        let progress = DownloadProgress {
            model_id: "model-003".to_string(),
            total_bytes: 2000,
            downloaded_bytes: 500,
            download_speed_bps: 100.0,
            eta_seconds: 15.0,
            status: DownloadStatus::Downloading,
        };
        let cloned = progress.clone();
        assert_eq!(cloned.model_id, "model-003");
        assert_eq!(cloned.downloaded_bytes, 500);
    }

    // --- ModelManager Tests ---

    #[test]
    fn test_model_manager_creation() {
        let config = make_test_config();
        let manager = ModelManager::new(config);
        assert!(manager.is_ok());
        // Clean up
        let _ = std::fs::remove_dir_all(
            std::env::temp_dir().join("trustformers_model_mgmt_test"),
        );
    }

    #[test]
    fn test_model_manager_with_existing_dir() {
        let dir = std::env::temp_dir().join("trustformers_model_mgmt_test2");
        let _ = std::fs::create_dir_all(&dir);
        let config = ModelManagerConfig {
            storage_directory: dir.clone(),
            ..make_test_config()
        };
        let manager = ModelManager::new(config);
        assert!(manager.is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
