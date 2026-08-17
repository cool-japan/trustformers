//! Non-network unit tests for `hub_upload`.
//!
//! Anything that needs to assert the *shape* of a real HTTP request against
//! the Hub API lives in `trustformers/tests/hub_upload_mock.rs` instead,
//! against a local mock server — this module only covers validation,
//! config/result shapes, dry-run behavior, and pure helpers, none of which
//! should ever need the network.

use super::*;
use std::fs;

fn temp_dir() -> PathBuf {
    std::env::temp_dir()
}

fn make_test_file(dir: &Path, name: &str, content: &[u8]) -> PathBuf {
    fs::create_dir_all(dir).ok();
    let path = dir.join(name);
    fs::write(&path, content).expect("Failed to write test file");
    path
}

fn valid_config() -> UploadConfig {
    UploadConfig {
        token: "hf_test_token".to_string(),
        repo_id: "testuser/test-model".to_string(),
        repo_type: RepoType::Model,
        revision: "main".to_string(),
        commit_message: "Test upload".to_string(),
        create_if_missing: true,
        private: false,
        base_url: "https://huggingface.co".to_string(),
        dry_run: false,
    }
}

fn dry_run_config() -> UploadConfig {
    UploadConfig {
        dry_run: true,
        ..valid_config()
    }
}

#[test]
fn test_repo_type_as_str() {
    assert_eq!(RepoType::Model.as_str(), "model");
    assert_eq!(RepoType::Dataset.as_str(), "dataset");
    assert_eq!(RepoType::Space.as_str(), "space");
}

#[test]
fn test_upload_config_default() {
    let config = UploadConfig::default();
    assert_eq!(config.revision, "main");
    assert!(!config.private);
    assert!(config.create_if_missing);
    assert_eq!(config.repo_type, RepoType::Model);
    assert_eq!(config.base_url, "https://huggingface.co");
    assert!(!config.dry_run, "dry_run must default to false");
}

#[test]
fn test_validate_empty_token() {
    let mut config = valid_config();
    config.token = String::new();
    let uploader = HubUploader::new(config);
    assert!(uploader.validate().is_err());
}

/// Regression test: an empty token must be reported as a *missing
/// credentials* condition — distinguishable from a generic validation
/// failure like an empty repo_id — matching the mission's explicit
/// `MissingCredentials` requirement.
#[test]
fn test_validate_empty_token_is_missing_credentials_not_generic_invalid_input() {
    let mut config = valid_config();
    config.token = String::new();
    let uploader = HubUploader::new(config);
    let err = uploader.validate().expect_err("empty token must fail validation");
    match err {
        TrustformersError::Hub { message, .. } => {
            assert!(message.to_lowercase().contains("missing"), "{message}");
            assert!(message.to_lowercase().contains("credential"), "{message}");
        },
        other => panic!("expected TrustformersError::Hub (missing credentials), got {other:?}"),
    }

    // Contrast: an empty repo_id is a different, generic validation error.
    let mut config2 = valid_config();
    config2.repo_id = String::new();
    let uploader2 = HubUploader::new(config2);
    let err2 = uploader2.validate().expect_err("empty repo_id must fail validation");
    assert!(matches!(err2, TrustformersError::InvalidInput { .. }));
}

#[test]
fn test_validate_empty_repo_id() {
    let mut config = valid_config();
    config.repo_id = String::new();
    let uploader = HubUploader::new(config);
    assert!(uploader.validate().is_err());
}

#[test]
fn test_validate_repo_id_missing_slash() {
    let mut config = valid_config();
    config.repo_id = "no-slash-repo".to_string();
    let uploader = HubUploader::new(config);
    assert!(uploader.validate().is_err());
}

#[test]
fn test_validate_empty_revision() {
    let mut config = valid_config();
    config.revision = String::new();
    let uploader = HubUploader::new(config);
    assert!(uploader.validate().is_err());
}

#[test]
fn test_upload_file_not_found() {
    let uploader = HubUploader::new(valid_config());
    let file = UploadFile::new("/nonexistent/path/model.bin", "model.bin");
    assert!(uploader.upload_file(&file).is_err());
}

/// Regression test for the historical bug: `upload_file` used to validate the
/// path, log `"(simulated)"`, and return `Ok(UploadResult::simulated(...))`
/// with a commit URL built from the all-zeros SHA — indistinguishable from a
/// real success. Without `dry_run` and without the `hub` feature compiled
/// in, it must now fail instead of fabricating success.
#[cfg(not(feature = "hub"))]
#[test]
fn test_upload_file_without_hub_feature_fails_instead_of_fabricating_success() {
    let dir = temp_dir().join("trustformers_upload_test_no_hub_feature");
    let path = make_test_file(&dir, "config.json", br#"{"model": "test"}"#);

    let uploader = HubUploader::new(valid_config()); // dry_run: false
    let file = UploadFile::new(path, "config.json");
    let err = uploader.upload_file(&file).expect_err(
        "without the `hub` feature and without dry_run, upload_file must fail, not fabricate a \
         result",
    );
    // Must not be confusable with a real success: assert it really is an error
    // carrying an explanation, not just any Err.
    assert!(!err.to_string().is_empty());

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_upload_file_dry_run_reports_success_without_network() {
    let dir = temp_dir().join("trustformers_upload_test_dry_run_file");
    let path = make_test_file(&dir, "config.json", br#"{"model": "test"}"#);

    let uploader = HubUploader::new(dry_run_config());
    let file = UploadFile::new(path, "config.json");
    let result = uploader.upload_file(&file).expect("dry run must succeed without any network");

    assert!(result.dry_run);
    assert_eq!(result.repo_id, "testuser/test-model");
    assert_eq!(result.revision, "main");
    assert_eq!(result.files_uploaded, vec!["config.json"]);
    // A dry run must not claim a real commit happened.
    assert_eq!(result.commit_url, None);
    assert_eq!(result.commit_oid, None);

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_upload_files_empty_list() {
    let uploader = HubUploader::new(valid_config());
    assert!(uploader.upload_files(&[]).is_err());
}

#[test]
fn test_upload_files_multiple_dry_run() {
    let dir = temp_dir().join("trustformers_upload_test_multi");
    let path1 = make_test_file(&dir, "config.json", b"{}");
    let path2 = make_test_file(&dir, "model.safetensors", b"weights");

    let uploader = HubUploader::new(dry_run_config());
    let files = vec![
        UploadFile::new(path1, "config.json"),
        UploadFile::new(path2, "model.safetensors"),
    ];
    let result = uploader.upload_files(&files).expect("dry run upload_files");

    assert!(result.dry_run);
    assert_eq!(result.files_uploaded.len(), 2);
    assert!(result.files_uploaded.contains(&"config.json".to_string()));
    assert!(result.files_uploaded.contains(&"model.safetensors".to_string()));

    fs::remove_dir_all(&dir).ok();
}

/// Regression test: a file at or above the inline-upload threshold must be
/// refused with a clear error (before any network attempt — this applies
/// even in a dry run, since it's a real limitation of *this* upload path,
/// not something a dry run should hide).
#[test]
fn test_upload_files_rejects_file_at_or_above_lfs_threshold() {
    let dir = temp_dir().join("trustformers_upload_test_large_file");
    let big_content = vec![0u8; api::LFS_INLINE_THRESHOLD_BYTES as usize];
    let path = make_test_file(&dir, "huge.safetensors", &big_content);

    let uploader = HubUploader::new(dry_run_config());
    let file = UploadFile::new(path, "huge.safetensors");
    let err = uploader
        .upload_file(&file)
        .expect_err("a file at the LFS threshold must be refused, not silently accepted");
    assert!(err.to_string().to_lowercase().contains("lfs"), "{err}");

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_upload_directory_dry_run() {
    let base = temp_dir().join("trustformers_upload_test_dir");
    make_test_file(&base, "config.json", b"{}");
    make_test_file(&base, "tokenizer.json", b"{}");

    let uploader = HubUploader::new(dry_run_config());
    let result = uploader.upload_directory(&base, "").expect("dry run upload_directory");

    assert!(result.dry_run);
    assert_eq!(result.files_uploaded.len(), 2);

    fs::remove_dir_all(&base).ok();
}

#[test]
fn test_upload_directory_with_prefix_dry_run() {
    let base = temp_dir().join("trustformers_upload_test_prefix");
    make_test_file(&base, "weights.bin", b"binary");

    let uploader = HubUploader::new(dry_run_config());
    let result = uploader.upload_directory(&base, "models/v1").expect("dry run");

    assert!(result.files_uploaded[0].starts_with("models/v1/"));

    fs::remove_dir_all(&base).ok();
}

#[test]
fn test_upload_directory_not_a_dir() {
    let dir = temp_dir().join("trustformers_upload_test_notdir");
    let file = make_test_file(&dir, "file.txt", b"content");

    let uploader = HubUploader::new(valid_config());
    assert!(uploader.upload_directory(&file, "").is_err());

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_upload_directory_empty_dry_run_reports_no_files() {
    let base = temp_dir().join("trustformers_upload_test_empty_dir_dry_run");
    fs::create_dir_all(&base).expect("create empty dir");

    let uploader = HubUploader::new(dry_run_config());
    let result = uploader.upload_directory(&base, "").expect("empty directory is not an error");
    assert!(result.files_uploaded.is_empty());
    assert!(result.dry_run);

    fs::remove_dir_all(&base).ok();
}

#[test]
fn test_delete_file_empty_path() {
    let uploader = HubUploader::new(valid_config());
    assert!(uploader.delete_file("").is_err());
}

#[test]
fn test_delete_file_dry_run() {
    let uploader = HubUploader::new(dry_run_config());
    assert!(uploader.delete_file("model.bin").is_ok());
}

#[test]
fn test_repo_exists_dry_run_is_false() {
    let uploader = HubUploader::new(dry_run_config());
    assert!(!uploader.repo_exists().expect("dry run repo_exists"));
}

#[cfg(not(feature = "hub"))]
#[test]
fn test_repo_exists_without_hub_feature_errors() {
    let uploader = HubUploader::new(valid_config());
    assert!(uploader.repo_exists().is_err());
}

#[test]
fn test_create_repo_dry_run_returns_url() {
    let uploader = HubUploader::new(dry_run_config());
    let url = uploader.create_repo().expect("dry run create_repo");
    assert!(url.contains("testuser/test-model"));
}

#[cfg(not(feature = "hub"))]
#[test]
fn test_create_repo_without_hub_feature_errors() {
    let uploader = HubUploader::new(valid_config());
    assert!(uploader.create_repo().is_err());
}

#[test]
fn test_builder_missing_slash_in_repo_id() {
    let result = HubUploaderBuilder::new("token", "noslash").build();
    assert!(result.is_err());
}

#[test]
fn test_builder_success() {
    let uploader = HubUploaderBuilder::new("hf_token", "user/repo")
        .repo_type(RepoType::Dataset)
        .commit_message("Initial upload")
        .private(true)
        .create_if_missing(false)
        .revision("v1")
        .base_url("http://127.0.0.1:9")
        .dry_run(true)
        .build()
        .unwrap();

    assert_eq!(uploader.config.repo_type, RepoType::Dataset);
    assert!(uploader.config.private);
    assert!(!uploader.config.create_if_missing);
    assert_eq!(uploader.config.revision, "v1");
    assert_eq!(uploader.config.base_url, "http://127.0.0.1:9");
    assert!(uploader.config.dry_run);
}

#[test]
fn test_upload_file_empty_repo_path() {
    let dir = temp_dir().join("trustformers_upload_empty_rpath");
    let path = make_test_file(&dir, "x.bin", b"data");

    let uploader = HubUploader::new(valid_config());
    let file = UploadFile::new(path, "");
    assert!(uploader.upload_file(&file).is_err());

    fs::remove_dir_all(&dir).ok();
}

// ── HubError, HubUploadConfig, HubUploadProgress, sha256 ──

#[test]
fn test_hub_error_display_unauthorized() {
    let e = HubError::Unauthorized {
        message: "bad token".to_string(),
    };
    assert!(e.to_string().contains("Unauthorized"));
    assert!(e.to_string().contains("bad token"));
}

#[test]
fn test_hub_error_display_missing_credentials() {
    let e = HubError::MissingCredentials {
        message: "no token given".to_string(),
    };
    assert!(e.to_string().to_lowercase().contains("missing credentials"));
}

#[test]
fn test_hub_error_display_feature_unavailable() {
    let e = HubError::FeatureUnavailable {
        message: "hub feature disabled".to_string(),
    };
    assert!(e.to_string().contains("Feature unavailable"));
}

#[test]
fn test_hub_error_display_lfs_required() {
    let e = HubError::LfsRequired {
        path: "big.bin".to_string(),
        size: 20_000_000,
    };
    let s = e.to_string();
    assert!(s.contains("big.bin"));
    assert!(s.contains("20000000") || s.contains("20,000,000"));
}

#[test]
fn test_hub_error_display_not_found_with_path() {
    let e = HubError::NotFound {
        repo_id: "user/repo".to_string(),
        path: Some("model.bin".to_string()),
    };
    assert!(e.to_string().contains("user/repo"));
    assert!(e.to_string().contains("model.bin"));
}

#[test]
fn test_hub_error_display_not_found_without_path() {
    let e = HubError::NotFound {
        repo_id: "user/repo".to_string(),
        path: None,
    };
    assert!(e.to_string().contains("user/repo"));
}

#[test]
fn test_hub_error_display_request_failed() {
    let e = HubError::RequestFailed {
        status_code: 403,
        message: "forbidden".to_string(),
    };
    assert!(e.to_string().contains("403"));
    assert!(e.to_string().contains("forbidden"));
}

#[test]
fn test_hub_upload_config_new() {
    let cfg = HubUploadConfig::new("user/model", "tok", "init commit");
    assert_eq!(cfg.repo_id, "user/model");
    assert_eq!(cfg.token, "tok");
    assert_eq!(cfg.commit_message, "init commit");
    assert!(!cfg.private);
    assert!(cfg.revision.is_none());
    assert_eq!(cfg.effective_revision(), "main");
    assert!(!cfg.dry_run);
}

#[test]
fn test_hub_upload_config_with_revision() {
    let cfg = HubUploadConfig::new("user/model", "tok", "msg").with_revision("v2");
    assert_eq!(cfg.effective_revision(), "v2");
}

#[test]
fn test_hub_upload_config_with_base_url_and_dry_run() {
    let cfg = HubUploadConfig::new("user/model", "tok", "msg")
        .with_base_url("http://127.0.0.1:1234")
        .with_dry_run(true);
    assert_eq!(cfg.base_url.as_deref(), Some("http://127.0.0.1:1234"));
    assert!(cfg.dry_run);
}

#[test]
fn test_hub_upload_config_validate_ok() {
    let cfg = HubUploadConfig::new("user/model", "hf_token", "msg");
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_hub_upload_config_validate_empty_token_is_missing_credentials() {
    let cfg = HubUploadConfig::new("user/model", "", "msg");
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, HubError::MissingCredentials { .. }));
}

#[test]
fn test_hub_upload_config_validate_missing_slash() {
    let cfg = HubUploadConfig::new("noslash", "tok", "msg");
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, HubError::InvalidInput { .. }));
}

#[test]
fn test_hub_upload_progress_tracking() {
    let mut progress = HubUploadProgress::new(3, 300);
    assert_eq!(progress.fraction(), 0.0);
    assert!(!progress.is_complete());

    progress.record_file(100);
    progress.record_file(100);
    progress.record_file(100);

    assert!((progress.fraction() - 1.0).abs() < 1e-9);
    assert!(progress.is_complete());
}

#[test]
fn test_hub_upload_progress_zero_bytes() {
    let progress = HubUploadProgress::new(0, 0);
    assert_eq!(progress.fraction(), 1.0);
}

#[test]
fn test_hub_upload_progress_files_only() {
    let mut progress = HubUploadProgress::new(2, 0);
    progress.record_file(0);
    assert!((progress.fraction() - 0.5).abs() < 1e-9);
    assert!(!progress.is_complete());
    progress.record_file(0);
    assert!(progress.is_complete());
}

#[test]
fn test_sha256_deterministic() {
    let data = b"hello, trustformers!";
    let h1 = sha256(data);
    let h2 = sha256(data);
    assert_eq!(h1, h2);
    assert_eq!(h1.len(), 64);
    assert!(h1.chars().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn test_sha256_different_inputs() {
    let h1 = sha256(b"foo");
    let h2 = sha256(b"bar");
    assert_ne!(h1, h2);
}

#[test]
fn test_sha256_known_vectors() {
    assert_eq!(
        sha256(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(sha256(b"hello").len(), 64);
}

#[test]
fn test_sha256_file() {
    let dir = temp_dir().join("trustformers_sha256_test");
    let path = make_test_file(&dir, "data.bin", b"test file content for hashing");

    let hash = sha256_file(&path).unwrap();
    assert_eq!(hash.len(), 64);
    let hash2 = sha256_file(&path).unwrap();
    assert_eq!(hash, hash2);

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_upload_file_path_dry_run() {
    let dir = temp_dir().join("trustformers_upload_path_test");
    let path = dir.join("weights.bin");
    fs::create_dir_all(&dir).ok();
    fs::write(&path, b"fake weights data").unwrap();

    let uploader = HubUploader::new(dry_run_config());
    let result = uploader.upload_file_path(path.to_str().unwrap(), "weights.bin").unwrap();

    assert_eq!(result.file_size, 17);
    assert!(result.remote_url.contains("testuser/test-model"));
    assert_eq!(result.commit_url, None);
    assert_eq!(result.sha256.len(), 64);

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_upload_file_path_not_found() {
    let uploader = HubUploader::new(valid_config());
    let err = uploader.upload_file_path("/nonexistent/path.bin", "path.bin").unwrap_err();
    assert!(matches!(err, HubError::Io { .. }));
}

#[test]
fn test_upload_file_path_empty_remote() {
    let dir = temp_dir().join("trustformers_upload_empty_remote");
    let path = dir.join("x.bin");
    fs::create_dir_all(&dir).ok();
    fs::write(&path, b"x").unwrap();

    let uploader = HubUploader::new(valid_config());
    let err = uploader.upload_file_path(path.to_str().unwrap(), "").unwrap_err();
    assert!(matches!(err, HubError::InvalidInput { .. }));

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_upload_model_directory_dry_run() {
    let dir = temp_dir().join("trustformers_upload_model_dir");
    fs::create_dir_all(&dir).ok();
    fs::write(dir.join("config.json"), "{}").unwrap();
    fs::write(dir.join("model.safetensors"), "weights").unwrap();
    fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    fs::write(dir.join("notes.txt"), "ignore me").unwrap();

    let uploader = HubUploader::new(dry_run_config());
    let results = uploader.upload_model(dir.to_str().unwrap()).unwrap();

    assert!(results.iter().any(|r| r.remote_url.contains("config.json")));
    assert!(results.iter().any(|r| r.remote_url.contains("model.safetensors")));
    let names: Vec<_> = results.iter().map(|r| r.remote_url.clone()).collect();
    assert!(!names.iter().any(|u| u.contains("notes.txt")));

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_upload_tokenizer_directory_dry_run() {
    let dir = temp_dir().join("trustformers_upload_tok_dir");
    fs::create_dir_all(&dir).ok();
    fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
    fs::write(dir.join("vocab.txt"), "hello\nworld").unwrap();
    fs::write(dir.join("model.safetensors"), "weights").unwrap();

    let uploader = HubUploader::new(dry_run_config());
    let results = uploader.upload_tokenizer(dir.to_str().unwrap()).unwrap();

    assert!(results.iter().any(|r| r.remote_url.contains("tokenizer.json")));
    assert!(results.iter().any(|r| r.remote_url.contains("vocab.txt")));
    assert!(!results.iter().any(|r| r.remote_url.contains("model.safetensors")));

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_create_repo_typed_dry_run() {
    let uploader = HubUploader::new(dry_run_config());
    let url = uploader.create_repo_typed(RepoType::Dataset).unwrap();
    assert!(url.contains("testuser/test-model"));
}

#[test]
fn test_delete_remote_file_dry_run() {
    let uploader = HubUploader::new(dry_run_config());
    assert!(uploader.delete_remote_file("model.bin").is_ok());
}

#[test]
fn test_from_hub_config() {
    let cfg = HubUploadConfig::new("user/my-model", "hf_tok", "first upload")
        .with_private(true)
        .with_revision("dev")
        .with_dry_run(true);
    let uploader = HubUploader::from_hub_config(cfg).unwrap();
    assert_eq!(uploader.config.repo_id, "user/my-model");
    assert!(uploader.config.private);
    assert_eq!(uploader.config.revision, "dev");
    assert!(uploader.config.dry_run);
}

#[test]
fn test_from_hub_config_validation_fail() {
    let cfg = HubUploadConfig::new("no-slash", "tok", "msg");
    assert!(HubUploader::from_hub_config(cfg).is_err());
}
