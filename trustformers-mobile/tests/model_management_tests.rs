//! Model management tests for trustformers-mobile
//!
//! Tests model metadata, versioning, update priority, cache config,
//! and storage statistics without actual network downloads.

use std::env::temp_dir;
use trustformers_mobile::model_management::{
    DownloadStatus, ModelCompatibility, ModelManagerConfig, ModelMetadata, ModelUpdate,
    UpdatePriority, UpdateType,
};
use trustformers_mobile::MobileConfig;

fn make_model_manager_config() -> ModelManagerConfig {
    let storage_dir = temp_dir().join("trustformers_mobile_model_management_test");
    ModelManagerConfig {
        update_server_url: "https://models.example.com".to_string(),
        api_key: Some("test-key-abc123".to_string()),
        storage_directory: storage_dir,
        max_storage_size_mb: 512,
        enable_auto_updates: true,
        update_check_interval_seconds: 3600,
        enable_differential_updates: true,
        require_signature_verification: true,
        download_timeout_seconds: 30,
        max_concurrent_downloads: 2,
        enable_compression: true,
        download_retry_attempts: 3,
    }
}

fn make_model_metadata(id: &str, version: &str) -> ModelMetadata {
    ModelMetadata {
        model_id: id.to_string(),
        version: version.to_string(),
        model_type: "bert-base".to_string(),
        size_bytes: 1024 * 1024 * 256, // 256 MB
        checksum: "abc123def456".to_string(),
        signature: None,
        download_url: format!("https://models.example.com/{id}/{version}/model.bin"),
        differential_url: None,
        description: "Test model".to_string(),
        required_config: MobileConfig::default(),
        compatibility: ModelCompatibility {
            min_android_api: Some(26),
            min_ios_version: Some("14.0".to_string()),
            required_features: vec!["fp16".to_string()],
            min_memory_mb: 512,
            supported_architectures: vec!["arm64".to_string(), "x86_64".to_string()],
        },
        release_timestamp: 1700000000,
        deprecation_timestamp: None,
        tags: vec!["nlp".to_string(), "bert".to_string()],
    }
}

#[test]
fn test_model_manager_config_creation() {
    let config = make_model_manager_config();
    assert!(!config.update_server_url.is_empty());
    assert!(config.max_storage_size_mb > 0);
    assert!(config.download_retry_attempts > 0);
}

#[test]
fn test_model_manager_config_temp_dir_is_valid() {
    let config = make_model_manager_config();
    // The parent (temp_dir) should exist
    let parent = config.storage_directory.parent().expect("should have parent");
    assert!(parent.exists(), "temp_dir parent should exist");
}

#[test]
fn test_model_metadata_creation() {
    let meta = make_model_metadata("bert-base", "1.0.0");
    assert_eq!(meta.model_id, "bert-base");
    assert_eq!(meta.version, "1.0.0");
    assert!(!meta.download_url.is_empty());
}

#[test]
fn test_model_metadata_size_positive() {
    let meta = make_model_metadata("bert-base", "1.0.0");
    assert!(meta.size_bytes > 0, "model size must be positive");
}

#[test]
fn test_model_compatibility_min_memory_positive() {
    let meta = make_model_metadata("bert-base", "1.0.0");
    assert!(
        meta.compatibility.min_memory_mb > 0,
        "min memory requirement must be positive"
    );
}

#[test]
fn test_model_update_full_vs_differential() {
    let current = make_model_metadata("bert-base", "1.0.0");
    let update = make_model_metadata("bert-base", "1.1.0");
    let model_update = ModelUpdate {
        current: current.clone(),
        update: update.clone(),
        update_type: UpdateType::Full,
        priority: UpdatePriority::Normal,
        update_size_bytes: update.size_bytes,
    };
    assert_eq!(model_update.update_type, UpdateType::Full);
    assert_ne!(model_update.current.version, model_update.update.version);
}

#[test]
fn test_update_priority_ordering() {
    // Critical > High > Normal > Low
    // They are PartialEq enum variants (not PartialOrd), so just test inequality
    assert_ne!(UpdatePriority::Critical, UpdatePriority::Low);
    assert_ne!(UpdatePriority::High, UpdatePriority::Normal);
}

#[test]
fn test_update_priority_variants_exist() {
    let _critical = UpdatePriority::Critical;
    let _high = UpdatePriority::High;
    let _normal = UpdatePriority::Normal;
    let _low = UpdatePriority::Low;
}

#[test]
fn test_update_type_variants_exist() {
    let _full = UpdateType::Full;
    let _differential = UpdateType::Differential;
    let _config_only = UpdateType::ConfigOnly;
}

#[test]
fn test_download_status_variants() {
    let _pending = DownloadStatus::Pending;
    let _downloading = DownloadStatus::Downloading;
    let _verifying = DownloadStatus::Verifying;
    let _installing = DownloadStatus::Installing;
    let _completed = DownloadStatus::Completed;
    let _failed = DownloadStatus::Failed("network error".to_string());
    let _cancelled = DownloadStatus::Cancelled;
}

#[test]
fn test_model_metadata_serialization_roundtrip() {
    let meta = make_model_metadata("bert-base", "2.0.0");
    let json = serde_json::to_string(&meta).expect("serialization should succeed");
    let restored: ModelMetadata =
        serde_json::from_str(&json).expect("deserialization should succeed");
    assert_eq!(restored.model_id, meta.model_id);
    assert_eq!(restored.version, meta.version);
    assert_eq!(restored.size_bytes, meta.size_bytes);
    assert_eq!(restored.checksum, meta.checksum);
}

#[test]
fn test_model_tags_can_contain_multiple_values() {
    let meta = make_model_metadata("bert-base", "1.0.0");
    assert!(!meta.tags.is_empty(), "model should have at least one tag");
}

#[test]
fn test_supported_architectures_not_empty() {
    let meta = make_model_metadata("bert-base", "1.0.0");
    assert!(
        !meta.compatibility.supported_architectures.is_empty(),
        "model should support at least one architecture"
    );
}

#[test]
fn test_model_manager_config_timeout_positive() {
    let config = make_model_manager_config();
    assert!(
        config.download_timeout_seconds > 0,
        "download timeout must be positive"
    );
}
