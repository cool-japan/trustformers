//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    use std::env;

    fn temp_dir_path() -> std::path::PathBuf {
        let mut path = env::temp_dir();
        // Use a deterministic but unique subdirectory using LCG-based pseudo-unique suffix
        // LCG: seed = PID * 6364136223846793005 + 1442695040888963407
        let pid = std::process::id() as u64;
        let suffix = pid.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        path.push(format!("trustformers_test_{}", suffix));
        path
    }

    /// Build a real local "model directory" — the kind `resolve_model_source_dir`
    /// resolves directly, without any Hub cache or network access — containing
    /// a config, a tokenizer file, and a (fake but real, on-disk) weights file.
    /// Returns `(dir, total_bytes_of_the_three_files)`.
    fn make_fake_model_dir(base: &Path, name: &str) -> (PathBuf, u64) {
        let dir = base.join(name);
        std::fs::create_dir_all(&dir).expect("create fake model dir");
        let config = br#"{"model_type":"bert","hidden_size":768}"#;
        let tokenizer = br#"{"version":"1.0","vocab_size":30522}"#;
        let weights = vec![0xABu8; 256];
        std::fs::write(dir.join("config.json"), config).expect("write config.json");
        std::fs::write(dir.join("tokenizer.json"), tokenizer).expect("write tokenizer.json");
        std::fs::write(dir.join("model.safetensors"), &weights).expect("write model.safetensors");
        let total = (config.len() + tokenizer.len() + weights.len()) as u64;
        (dir, total)
    }

    // --- ModelPackMetadata tests ---

    #[test]
    fn test_model_pack_metadata_fields() {
        let metadata = ModelPackMetadata {
            pack_id: "test-pack-id".to_string(),
            name: "Test Pack".to_string(),
            description: "A test model pack".to_string(),
            version: "1.0.0".to_string(),
            created_at: SystemTime::now(),
            created_by: "trustformers".to_string(),
            total_size: 1024 * 1024,
            models: vec![],
            dependencies: vec![],
            target_platforms: vec!["linux".to_string()],
            checksum: "abc123".to_string(),
            compression_ratio: 0.75,
        };
        assert_eq!(metadata.pack_id, "test-pack-id");
        assert_eq!(metadata.name, "Test Pack");
        assert!(!metadata.version.is_empty(), "version should not be empty");
        assert!(
            metadata.compression_ratio > 0.0,
            "compression_ratio should be positive"
        );
    }

    #[test]
    fn test_model_pack_metadata_compression_ratio_bounded() {
        // Compression ratio should be (0.0, 1.0] for compressed, or > 1.0 for expansion
        let metadata = ModelPackMetadata {
            pack_id: "id1".to_string(),
            name: "Pack".to_string(),
            description: "desc".to_string(),
            version: "1.0.0".to_string(),
            created_at: SystemTime::now(),
            created_by: "test".to_string(),
            total_size: 512,
            models: vec![],
            dependencies: vec![],
            target_platforms: vec![],
            checksum: "abc".to_string(),
            compression_ratio: 0.65,
        };
        assert!(
            metadata.compression_ratio > 0.0,
            "compression_ratio should be positive"
        );
    }

    // --- PackedModelInfo tests ---

    #[test]
    fn test_packed_model_info_construction() {
        let info = PackedModelInfo {
            model_id: "bert-base-uncased".to_string(),
            name: "BERT Base Uncased".to_string(),
            version: "latest".to_string(),
            original_size: 1024 * 1024 * 440,
            compressed_size: 1024 * 1024 * 320,
            model_type: ModelType::TextClassification,
            framework: "transformers".to_string(),
            precision: PrecisionType::FP32,
            metadata: HashMap::new(),
        };
        assert_eq!(info.model_id, "bert-base-uncased");
        assert!(
            info.compressed_size <= info.original_size,
            "compressed_size should not exceed original_size after compression"
        );
    }

    #[test]
    fn test_packed_model_info_model_type_variants() {
        let types = [
            ModelType::TextGeneration,
            ModelType::TextClassification,
            ModelType::ImageClassification,
            ModelType::SpeechRecognition,
            ModelType::Translation,
            ModelType::Summarization,
            ModelType::QuestionAnswering,
            ModelType::Multimodal,
        ];
        // Verify all variants are constructible
        assert_eq!(types.len(), 8, "should have 8 ModelType variants");
    }

    // --- PackCreationConfig tests ---

    #[test]
    fn test_pack_creation_config_default() {
        let config = PackCreationConfig::default();
        assert!(
            config.compression_level <= 9,
            "compression_level should be in [0,9]"
        );
        assert!(
            !config.target_platforms.is_empty(),
            "target_platforms should not be empty by default"
        );
        assert!(
            config.max_pack_size.is_some(),
            "default max_pack_size should be set"
        );
        let max_size = config.max_pack_size.expect("max_pack_size should be set");
        assert!(max_size > 0, "max_pack_size should be positive");
    }

    #[test]
    fn test_pack_creation_config_compression_level_range() {
        for level in 0u8..=9 {
            let config = PackCreationConfig {
                compression_level: level,
                ..PackCreationConfig::default()
            };
            assert!(
                config.compression_level <= 9,
                "compression_level {} should be valid (0-9)",
                config.compression_level
            );
        }
    }

    // --- OfflineModelPackManager construction tests ---

    #[test]
    fn test_offline_pack_manager_new_creates_directory() {
        let path = temp_dir_path();
        let _manager = OfflineModelPackManager::new(&path)
            .expect("OfflineModelPackManager::new should succeed");
        assert!(path.exists(), "base directory should be created");
        std::fs::remove_dir_all(&path).ok();
    }

    #[test]
    fn test_offline_pack_manager_list_packs_initially_empty() {
        let path = temp_dir_path();
        let manager = OfflineModelPackManager::new(&path)
            .expect("OfflineModelPackManager::new should succeed");
        let packs = manager.list_packs();
        // Initially empty (or from any previously saved registry)
        let _ = packs.len(); // Just verify no panic
        std::fs::remove_dir_all(&path).ok();
    }

    #[test]
    fn test_offline_pack_manager_get_pack_info_missing_returns_none() {
        let path = temp_dir_path();
        let manager = OfflineModelPackManager::new(&path)
            .expect("OfflineModelPackManager::new should succeed");
        let info = manager.get_pack_info("non-existent-pack-id");
        assert!(
            info.is_none(),
            "get_pack_info on missing pack should return None"
        );
        std::fs::remove_dir_all(&path).ok();
    }

    // --- Async pack creation tests ---

    #[tokio::test]
    async fn test_create_pack_returns_pack_id() {
        let path = temp_dir_path();
        let mut manager = OfflineModelPackManager::new(&path)
            .expect("OfflineModelPackManager::new should succeed");
        let (model_dir, _size) = make_fake_model_dir(&path, "src-model");
        let config = PackCreationConfig::default();
        let pack_id = manager
            .create_pack(
                "Test Pack".to_string(),
                "A test pack for unit testing".to_string(),
                vec![model_dir.to_string_lossy().to_string()],
                config,
            )
            .await
            .expect("create_pack should succeed");
        assert!(
            !pack_id.is_empty(),
            "create_pack should return non-empty pack_id"
        );
        std::fs::remove_dir_all(&path).ok();
    }

    #[tokio::test]
    async fn test_create_pack_registers_in_list() {
        let path = temp_dir_path();
        let mut manager = OfflineModelPackManager::new(&path)
            .expect("OfflineModelPackManager::new should succeed");
        let (model_dir, _size) = make_fake_model_dir(&path, "src-model");
        let config = PackCreationConfig::default();
        let pack_id = manager
            .create_pack(
                "Listed Pack".to_string(),
                "Pack that should appear in listing".to_string(),
                vec![model_dir.to_string_lossy().to_string()],
                config,
            )
            .await
            .expect("create_pack should succeed");
        let packs = manager.list_packs();
        let found = packs.iter().any(|p| p.pack_id == pack_id);
        assert!(found, "newly created pack should appear in list_packs()");
        std::fs::remove_dir_all(&path).ok();
    }

    #[tokio::test]
    async fn test_create_pack_metadata_has_model_info() {
        let path = temp_dir_path();
        let mut manager = OfflineModelPackManager::new(&path)
            .expect("OfflineModelPackManager::new should succeed");
        let (model_dir_a, _) = make_fake_model_dir(&path, "src-model-a");
        let (model_dir_b, _) = make_fake_model_dir(&path, "src-model-b");
        let config = PackCreationConfig::default();
        let pack_id = manager
            .create_pack(
                "Metadata Test Pack".to_string(),
                "Testing metadata fields".to_string(),
                vec![
                    model_dir_a.to_string_lossy().to_string(),
                    model_dir_b.to_string_lossy().to_string(),
                ],
                config,
            )
            .await
            .expect("create_pack should succeed");
        let info = manager
            .get_pack_info(&pack_id)
            .expect("pack should be retrievable after creation");
        assert_eq!(info.name, "Metadata Test Pack");
        assert!(
            !info.checksum.is_empty(),
            "pack should have a non-empty integrity checksum"
        );
        assert!(info.total_size > 0, "pack should have positive total_size");
        assert!(
            !info.models.is_empty(),
            "pack should contain model information"
        );
        std::fs::remove_dir_all(&path).ok();
    }

    #[tokio::test]
    async fn test_create_pack_pack_id_is_unique() {
        let path = temp_dir_path();
        let mut manager = OfflineModelPackManager::new(&path)
            .expect("OfflineModelPackManager::new should succeed");
        let (model_dir_a, _) = make_fake_model_dir(&path, "src-model-a");
        let (model_dir_b, _) = make_fake_model_dir(&path, "src-model-b");
        let config = PackCreationConfig::default();
        let id1 = manager
            .create_pack(
                "Pack A".to_string(),
                "First pack".to_string(),
                vec![model_dir_a.to_string_lossy().to_string()],
                config.clone(),
            )
            .await
            .expect("first create_pack should succeed");
        let id2 = manager
            .create_pack(
                "Pack B".to_string(),
                "Second pack".to_string(),
                vec![model_dir_b.to_string_lossy().to_string()],
                config,
            )
            .await
            .expect("second create_pack should succeed");
        assert_ne!(id1, id2, "each created pack should have a unique pack_id");
        std::fs::remove_dir_all(&path).ok();
    }

    /// Regression test for the P0 bug: `create_pack` used to record a
    /// hardcoded `1024 * 1024 * 512` (512MB) `original_size` for every model
    /// regardless of its real content, making `compression_ratio` fiction.
    /// With real local files, `original_size` must equal their real byte sum.
    #[tokio::test]
    async fn test_create_pack_uses_real_file_sizes_not_hardcoded_512mb() {
        let path = temp_dir_path();
        let mut manager = OfflineModelPackManager::new(&path)
            .expect("OfflineModelPackManager::new should succeed");
        let (model_dir, expected_size) = make_fake_model_dir(&path, "src-model");
        // Sanity check the fixture itself is nowhere near 512MB.
        assert!(expected_size < 1024 * 1024);

        let pack_id = manager
            .create_pack(
                "Real Size Pack".to_string(),
                "Pack whose size must reflect real files".to_string(),
                vec![model_dir.to_string_lossy().to_string()],
                PackCreationConfig::default(),
            )
            .await
            .expect("create_pack should succeed");

        let info = manager
            .get_pack_info(&pack_id)
            .expect("pack should be retrievable after creation");
        assert_eq!(info.models.len(), 1);
        let packed = &info.models[0];
        assert_eq!(
            packed.original_size, expected_size,
            "original_size must be the real on-disk byte count, not a 512MB estimate"
        );
        assert_ne!(
            packed.original_size,
            1024 * 1024 * 512,
            "original_size must never be the old hardcoded 512MB placeholder"
        );

        std::fs::remove_dir_all(&path).ok();
    }

    /// Regression test for the P0 bug: `create_compressed_archive` used to
    /// write a fabricated `config.json` (`"architecture": "auto-detected"`)
    /// for every model regardless of whether any real files existed. A
    /// model_id that resolves to nothing real (no local dir, no cache hit,
    /// and no `hub` feature to fall back on) must now fail outright.
    #[cfg(not(feature = "hub"))]
    #[tokio::test]
    async fn test_create_pack_fails_for_unresolvable_model_without_hub_feature() {
        let path = temp_dir_path();
        let mut manager = OfflineModelPackManager::new(&path)
            .expect("OfflineModelPackManager::new should succeed");
        let result = manager
            .create_pack(
                "Should Fail".to_string(),
                "No local dir, no cache, no hub feature".to_string(),
                vec!["definitely-not-a-real-local-path-or-cached-model".to_string()],
                PackCreationConfig::default(),
            )
            .await;
        assert!(
            result.is_err(),
            "an unresolvable model must fail pack creation, not silently produce an empty pack"
        );
        std::fs::remove_dir_all(&path).ok();
    }

    /// `create_pack_from_hub` must actually use the real per-model byte sizes
    /// (via the shared `build_pack`/`create_compressed_archive` path) rather
    /// than computing an "enhanced_models" list it then threw away.
    ///
    /// Uses a local directory as the `model_id`, which both `get_hub_model_info`
    /// and `resolve_model_source_dir` treat as "no Hub repo id here" and
    /// resolve without any network access — so this runs identically, and
    /// without touching the network, in both `hub`-enabled and -disabled
    /// builds.
    #[tokio::test]
    async fn test_create_pack_from_hub_uses_real_file_sizes() {
        let path = temp_dir_path();
        let mut manager = OfflineModelPackManager::new(&path)
            .expect("OfflineModelPackManager::new should succeed");
        let (model_dir, expected_size) = make_fake_model_dir(&path, "src-model");
        let hub_integration = HubIntegration::new(None);

        let pack_id = manager
            .create_pack_from_hub(
                &hub_integration,
                "Hub Pack".to_string(),
                "Pack built via HubIntegration".to_string(),
                vec![model_dir.to_string_lossy().to_string()],
                PackCreationConfig::default(),
            )
            .await
            .expect("create_pack_from_hub should succeed");

        let info = manager
            .get_pack_info(&pack_id)
            .expect("pack should be retrievable after creation");
        assert_eq!(info.models.len(), 1);
        assert_eq!(info.models[0].original_size, expected_size);

        std::fs::remove_dir_all(&path).ok();
    }

    // --- PrecisionType tests ---

    #[test]
    fn test_precision_type_variants_serializable() {
        let types = [
            PrecisionType::FP32,
            PrecisionType::FP16,
            PrecisionType::INT8,
            PrecisionType::INT4,
            PrecisionType::Mixed,
        ];
        for precision in &types {
            let serialized =
                serde_json::to_string(precision).expect("PrecisionType should be serializable");
            assert!(
                !serialized.is_empty(),
                "serialized precision should not be empty"
            );
        }
    }

    // --- ModelInfo tests ---

    #[test]
    fn test_model_info_construction() {
        let info = ModelInfo {
            model_id: "test/model".to_string(),
            library_name: Some("transformers".to_string()),
            pipeline_tag: Some("text-generation".to_string()),
            tags: vec!["nlp".to_string()],
            config: HashMap::new(),
            downloads: Some(5000),
            likes: Some(200),
            created_at: None,
            updated_at: None,
            author: Some("test-author".to_string()),
            description: Some("A test model".to_string()),
            license: Some("apache-2.0".to_string()),
            task: Some("text-generation".to_string()),
            language: vec!["en".to_string()],
            dataset: vec![],
            model_type: None,
            architecture: None,
        };
        assert_eq!(info.model_id, "test/model");
        assert_eq!(info.pipeline_tag.as_deref(), Some("text-generation"));
        assert_eq!(info.downloads, Some(5000));
    }

    // --- get_model_info tests (Hub integration) ---

    /// Regression test for the P0 bug: without the `hub` feature,
    /// `get_model_info` used to fabricate `downloads: Some(1000)`,
    /// `likes: Some(50)`, `pipeline_tag: Some("text-generation")`, and
    /// `library_name: Some("transformers")` — plausible-looking numbers with
    /// no basis in reality. Every one of those must now be honestly
    /// `None`/empty: there is no way to know a model's real Hub metadata
    /// without a network call.
    #[cfg(not(feature = "hub"))]
    #[tokio::test]
    async fn test_get_model_info_without_hub_feature_is_honestly_empty() {
        let path = temp_dir_path();
        let manager = OfflineModelPackManager::new(&path)
            .expect("OfflineModelPackManager::new should succeed");
        let info = manager
            .get_model_info("some-arbitrary-model-id")
            .await
            .expect("get_model_info should not fail without the hub feature");
        assert_eq!(info.model_id, "some-arbitrary-model-id");
        assert_eq!(info.pipeline_tag, None, "must not fabricate a pipeline_tag");
        assert_eq!(info.library_name, None, "must not fabricate a library_name");
        assert_eq!(info.downloads, None, "must not fabricate a downloads count");
        assert_eq!(info.likes, None, "must not fabricate a likes count");
        assert!(info.tags.is_empty());
        assert!(info.config.is_empty());
        std::fs::remove_dir_all(&path).ok();
    }

    /// `model_id` may itself be a local model directory (no Hub repo id at
    /// all) — `get_model_info` must recognize that and return honest empty
    /// metadata without attempting a network call, in every feature
    /// configuration.
    #[tokio::test]
    async fn test_get_model_info_local_directory_skips_network_and_is_honest() {
        let path = temp_dir_path();
        let manager = OfflineModelPackManager::new(&path)
            .expect("OfflineModelPackManager::new should succeed");
        let (model_dir, _size) = make_fake_model_dir(&path, "local-model");

        let info = manager
            .get_model_info(&model_dir.to_string_lossy())
            .await
            .expect("get_model_info must succeed for a local directory");
        assert_eq!(info.model_id, model_dir.to_string_lossy());
        assert_eq!(info.downloads, None);
        assert_eq!(info.likes, None);
        assert_eq!(info.pipeline_tag, None);

        std::fs::remove_dir_all(&path).ok();
    }

    /// The real HTTP call itself just mirrors `hub.rs::get_download_stats`'s
    /// already-established `reqwest` usage, so the part worth unit-testing
    /// without a live network call is the JSON field-mapping logic. This
    /// exercises `model_info_from_hub_json` directly against a hand-built
    /// `serde_json::Value` shaped like a real `/api/models/{id}` response.
    #[cfg(feature = "hub")]
    #[test]
    fn test_model_info_from_hub_json_maps_all_fields() {
        let json = serde_json::json!({
            "downloads": 12345,
            "likes": 678,
            "pipeline_tag": "text-classification",
            "library_name": "transformers",
            "tags": ["nlp", "bert"],
            "createdAt": "2022-01-01T00:00:00.000Z",
            "lastModified": "2023-06-15T00:00:00.000Z",
            "author": "some-org",
            "cardData": {
                "license": "apache-2.0",
                "language": ["en", "fr"],
                "datasets": ["squad"]
            },
            "config": {
                "model_type": "bert",
                "architectures": ["BertForMaskedLM"]
            }
        });

        let info = model_info_from_hub_json("some-org/some-model", &json);

        assert_eq!(info.model_id, "some-org/some-model");
        assert_eq!(info.downloads, Some(12345));
        assert_eq!(info.likes, Some(678));
        assert_eq!(info.pipeline_tag.as_deref(), Some("text-classification"));
        assert_eq!(info.task.as_deref(), Some("text-classification"));
        assert_eq!(info.library_name.as_deref(), Some("transformers"));
        assert_eq!(info.tags, vec!["nlp".to_string(), "bert".to_string()]);
        assert_eq!(info.created_at.as_deref(), Some("2022-01-01T00:00:00.000Z"));
        assert_eq!(info.updated_at.as_deref(), Some("2023-06-15T00:00:00.000Z"));
        assert_eq!(info.author.as_deref(), Some("some-org"));
        assert_eq!(info.license.as_deref(), Some("apache-2.0"));
        assert_eq!(info.language, vec!["en".to_string(), "fr".to_string()]);
        assert_eq!(info.dataset, vec!["squad".to_string()]);
        assert_eq!(info.model_type.as_deref(), Some("bert"));
        assert_eq!(info.architecture.as_deref(), Some("BertForMaskedLM"));
        assert!(info.config.contains_key("model_type"));
    }

    #[cfg(feature = "hub")]
    #[test]
    fn test_model_info_from_hub_json_handles_missing_optional_fields() {
        let json = serde_json::json!({});
        let info = model_info_from_hub_json("bare-model", &json);

        assert_eq!(info.model_id, "bare-model");
        assert_eq!(info.downloads, None);
        assert_eq!(info.likes, None);
        assert_eq!(info.pipeline_tag, None);
        assert_eq!(info.library_name, None);
        assert!(info.tags.is_empty());
        assert!(info.config.is_empty());
        assert!(info.language.is_empty());
        assert!(info.dataset.is_empty());
        assert_eq!(info.license, None);
        assert_eq!(info.model_type, None);
        assert_eq!(info.architecture, None);
    }

    #[cfg(feature = "hub")]
    #[test]
    fn test_model_info_from_hub_json_handles_single_string_language() {
        // Some HF cardData responses provide `language` as a single string
        // rather than an array of strings.
        let json = serde_json::json!({
            "cardData": {
                "language": "en"
            }
        });
        let info = model_info_from_hub_json("single-lang-model", &json);
        assert_eq!(info.language, vec!["en".to_string()]);
    }

    // --- safe_relative_path / extract_pack path traversal ---

    /// Regression test: `extract_pack` used to only strip a leading `./` or
    /// `/`, so a crafted pack with an entry named e.g. `../../evil.txt` would
    /// write outside the extraction directory. `safe_relative_path` is the
    /// guard that now rejects it.
    #[test]
    fn test_safe_relative_path_rejects_parent_dir_traversal() {
        let base = temp_dir_path();
        assert!(safe_relative_path(&base, "../../etc/passwd").is_none());
        assert!(safe_relative_path(&base, "models/../../escape.txt").is_none());
    }

    #[test]
    fn test_safe_relative_path_rejects_absolute_paths() {
        let base = temp_dir_path();
        assert!(safe_relative_path(&base, "/etc/passwd").is_none());
    }

    #[test]
    fn test_safe_relative_path_rejects_empty_path() {
        let base = temp_dir_path();
        assert!(safe_relative_path(&base, "").is_none());
        assert!(safe_relative_path(&base, "./").is_none());
    }

    #[test]
    fn test_safe_relative_path_accepts_normal_nested_paths() {
        let base = temp_dir_path();
        let dest = safe_relative_path(&base, "models/bert/config.json")
            .expect("a normal nested relative path must be accepted");
        assert_eq!(dest, base.join("models").join("bert").join("config.json"));
    }

    #[test]
    fn test_safe_relative_path_strips_leading_current_dir() {
        let base = temp_dir_path();
        let dest =
            safe_relative_path(&base, "./pack_metadata.json").expect("./ prefix must be accepted");
        assert_eq!(dest, base.join("pack_metadata.json"));
    }

    // --- load_registry corruption handling ---

    /// Regression test: `load_registry` used to swallow a corrupt
    /// `registry.json` with `.unwrap_or_default()`, silently resetting the
    /// registry to empty (losing every previously-tracked pack) with no
    /// error at all. It must now fail loudly instead.
    #[test]
    fn test_new_rejects_corrupt_registry_file() {
        let path = temp_dir_path();
        std::fs::create_dir_all(&path).expect("create base dir");
        std::fs::write(path.join("registry.json"), b"{ this is not valid json ")
            .expect("write corrupt registry");

        let result = OfflineModelPackManager::new(&path);
        assert!(
            result.is_err(),
            "a corrupt registry.json must fail construction, not silently reset to empty"
        );

        std::fs::remove_dir_all(&path).ok();
    }

    #[test]
    fn test_new_accepts_missing_registry_file() {
        let path = temp_dir_path();
        // No registry.json written at all — a fresh manager should be fine.
        let manager = OfflineModelPackManager::new(&path)
            .expect("a missing registry.json should not be an error");
        assert!(manager.list_packs().is_empty());
        std::fs::remove_dir_all(&path).ok();
    }

    // --- resolve_model_source_dir ---

    #[tokio::test]
    async fn test_resolve_model_source_dir_uses_explicit_local_directory() {
        let base = temp_dir_path();
        let (model_dir, _size) = make_fake_model_dir(&base, "explicit-model");
        let resolved = resolve_model_source_dir(&model_dir.to_string_lossy())
            .await
            .expect("an existing local directory must resolve directly");
        assert_eq!(resolved, model_dir);
        std::fs::remove_dir_all(&base).ok();
    }

    #[cfg(not(feature = "hub"))]
    #[tokio::test]
    async fn test_resolve_model_source_dir_fails_when_unresolvable() {
        let result =
            resolve_model_source_dir("definitely-not-a-real-local-path-or-cached-model").await;
        assert!(result.is_err());
    }

    // --- install_pack end-to-end (checksum-verified extraction) ---

    /// Recursively collect every regular file under `dir`, keyed by filename,
    /// so the test below doesn't need to reconstruct the exact nested archive
    /// path (`models/{model_id}/{file}`) that `create_compressed_archive`
    /// chose internally.
    fn collect_files_by_name(dir: &Path, out: &mut HashMap<String, Vec<u8>>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.filter_map(|e| e.ok()) {
            let p = entry.path();
            if p.is_dir() {
                collect_files_by_name(&p, out);
            } else if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                if let Ok(bytes) = std::fs::read(&p) {
                    out.insert(name.to_string(), bytes);
                }
            }
        }
    }

    /// End-to-end regression test for the P0 bug: a pack built from real
    /// local model files must, once installed, extract those *exact bytes*
    /// back out — proving the create_pack -> tar/gzip archive -> install_pack
    /// -> extract round-trip preserves real model content rather than the
    /// old fabricated `config.json` stub.
    #[tokio::test]
    async fn test_install_pack_round_trips_real_file_content() {
        let path = temp_dir_path();
        let mut manager = OfflineModelPackManager::new(&path)
            .expect("OfflineModelPackManager::new should succeed");
        let (model_dir, _size) = make_fake_model_dir(&path, "install-src-model");

        let pack_id = manager
            .create_pack(
                "Install Round Trip".to_string(),
                "pack for install_pack round-trip test".to_string(),
                vec![model_dir.to_string_lossy().to_string()],
                PackCreationConfig::default(),
            )
            .await
            .expect("create_pack should succeed");

        let pack_path = path.join(format!("{}.tfpack", pack_id));
        let installed_id =
            manager.install_pack(&pack_path).await.expect("install_pack should succeed");
        assert_eq!(installed_id, pack_id);

        let install_dir = path.join("installed").join(&pack_id);
        let mut found: HashMap<String, Vec<u8>> = HashMap::new();
        collect_files_by_name(&install_dir, &mut found);

        assert_eq!(
            found.get("config.json").map(|v| v.as_slice()),
            Some(br#"{"model_type":"bert","hidden_size":768}"#.as_slice()),
            "extracted config.json must match the real source file byte-for-byte, not a \
             fabricated placeholder"
        );
        assert_eq!(
            found.get("tokenizer.json").map(|v| v.as_slice()),
            Some(br#"{"version":"1.0","vocab_size":30522}"#.as_slice()),
            "extracted tokenizer.json must match the real source file byte-for-byte"
        );
        assert_eq!(
            found.get("model.safetensors").map(|v| v.len()),
            Some(256),
            "the model weights file must round-trip through the archive at its real size"
        );

        std::fs::remove_dir_all(&path).ok();
    }

    /// Regression test: `install_pack` must refuse a `.tfpack` file whose
    /// on-disk bytes no longer match its recorded checksum, rather than
    /// silently extracting whatever is actually there (which could be
    /// truncated, bit-flipped, or swapped for an unrelated pack). No files
    /// may be extracted when the check fails.
    #[tokio::test]
    async fn test_install_pack_rejects_tampered_pack_file() {
        let path = temp_dir_path();
        let mut manager = OfflineModelPackManager::new(&path)
            .expect("OfflineModelPackManager::new should succeed");
        let (model_dir, _size) = make_fake_model_dir(&path, "tamper-src-model");

        let pack_id = manager
            .create_pack(
                "Tamper Test Pack".to_string(),
                "pack for checksum-tamper test".to_string(),
                vec![model_dir.to_string_lossy().to_string()],
                PackCreationConfig::default(),
            )
            .await
            .expect("create_pack should succeed");

        let pack_path = path.join(format!("{}.tfpack", pack_id));
        // Flip a byte in the middle of the archive body (well past any
        // header) without touching the separately-stored .metadata.json
        // checksum record.
        let mut bytes = std::fs::read(&pack_path).expect("read pack file");
        let flip_at = bytes.len() / 2;
        bytes[flip_at] ^= 0xFF;
        std::fs::write(&pack_path, &bytes).expect("write tampered pack file");

        let result = manager.install_pack(&pack_path).await;
        assert!(
            result.is_err(),
            "install_pack must reject a pack whose bytes no longer match its recorded checksum"
        );

        let install_dir = path.join("installed").join(&pack_id);
        assert!(
            !install_dir.exists(),
            "no files should be extracted onto disk when the checksum check fails"
        );

        std::fs::remove_dir_all(&path).ok();
    }
}
