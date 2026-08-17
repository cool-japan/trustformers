use crate::error::{Result, TrustformersError};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use uuid::Uuid;

/// Model information structure for Hub integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub model_id: String,
    pub library_name: Option<String>,
    pub pipeline_tag: Option<String>,
    pub tags: Vec<String>,
    pub config: HashMap<String, serde_json::Value>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
    pub author: Option<String>,
    pub description: Option<String>,
    pub license: Option<String>,
    pub task: Option<String>,
    pub language: Vec<String>,
    pub dataset: Vec<String>,
    pub model_type: Option<String>,
    pub architecture: Option<String>,
}

/// Parse a Hugging Face Hub `/api/models/{id}` JSON response into a [`ModelInfo`].
///
/// This is a pure, network-free mapping function so the field-extraction logic
/// (including the nested `cardData`/`config` lookups) can be unit-tested
/// without making any HTTP calls. Only the `hub`-feature body of
/// `OfflineModelPackManager::get_model_info` performs the actual request;
/// this function just maps the resulting JSON.
///
/// Field mapping mirrors `hub.rs::get_download_stats`'s manual-field-pull
/// pattern for this exact same endpoint:
/// - `downloads`, `likes`, `pipeline_tag`, `tags`, `library_name` map directly
/// - `createdAt` -> `created_at`, `lastModified` -> `updated_at`
/// - `cardData.license` / `cardData.language` / `cardData.datasets` -> `license` / `language` / `dataset`
/// - `config.model_type` -> `model_type`, `config.architectures[0]` -> `architecture`
#[cfg(feature = "hub")]
fn model_info_from_hub_json(model_id: &str, json: &serde_json::Value) -> ModelInfo {
    // `cardData.language`/`cardData.datasets` may be a single string or an
    // array of strings depending on the model's metadata; normalize both.
    fn as_string_vec(value: Option<&serde_json::Value>) -> Vec<String> {
        match value {
            Some(serde_json::Value::String(s)) => vec![s.clone()],
            Some(serde_json::Value::Array(items)) => {
                items.iter().filter_map(|item| item.as_str().map(str::to_string)).collect()
            },
            _ => Vec::new(),
        }
    }

    let pipeline_tag = json.get("pipeline_tag").and_then(|v| v.as_str()).map(str::to_string);
    let card_data = json.get("cardData");
    let config_obj = json.get("config");

    ModelInfo {
        model_id: model_id.to_string(),
        library_name: json.get("library_name").and_then(|v| v.as_str()).map(str::to_string),
        pipeline_tag: pipeline_tag.clone(),
        tags: json
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|items| items.iter().filter_map(|t| t.as_str().map(str::to_string)).collect())
            .unwrap_or_default(),
        config: config_obj
            .and_then(|c| c.as_object())
            .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default(),
        downloads: json.get("downloads").and_then(|v| v.as_u64()),
        likes: json.get("likes").and_then(|v| v.as_u64()),
        created_at: json.get("createdAt").and_then(|v| v.as_str()).map(str::to_string),
        updated_at: json.get("lastModified").and_then(|v| v.as_str()).map(str::to_string),
        author: json.get("author").and_then(|v| v.as_str()).map(str::to_string),
        description: None,
        license: card_data
            .and_then(|c| c.get("license"))
            .and_then(|v| v.as_str())
            .map(str::to_string),
        task: pipeline_tag,
        language: as_string_vec(card_data.and_then(|c| c.get("language"))),
        dataset: as_string_vec(card_data.and_then(|c| c.get("datasets"))),
        model_type: config_obj
            .and_then(|c| c.get("model_type"))
            .and_then(|v| v.as_str())
            .map(str::to_string),
        architecture: config_obj
            .and_then(|c| c.get("architectures"))
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_str())
            .map(str::to_string),
    }
}

/// Build a "we don't know anything about this model" [`ModelInfo`]: every
/// Hub-side field is `None`/empty rather than a guessed placeholder. Shared
/// by every path that genuinely has no Hub metadata to report — `model_id`
/// being a local directory (no Hub repo id to query at all), the `hub`
/// feature being disabled, or a failed Hub lookup.
fn empty_model_info(model_id: &str) -> ModelInfo {
    ModelInfo {
        model_id: model_id.to_string(),
        library_name: None,
        pipeline_tag: None,
        tags: vec![],
        config: HashMap::new(),
        downloads: None,
        likes: None,
        created_at: None,
        updated_at: None,
        author: None,
        description: None,
        license: None,
        task: None,
        language: vec![],
        dataset: vec![],
        model_type: None,
        architecture: None,
    }
}

/// Resolve `model_id` to a real, on-disk directory containing that model's
/// files, without ever fabricating one:
///
/// 1. If `model_id` is itself an existing local directory, use it directly —
///    this is how a caller points at a model that was never downloaded from
///    the Hub at all.
/// 2. Otherwise, check the Hub's on-disk cache (the same layout
///    [`crate::hub::download_model_enhanced`] writes to).
/// 3. With the `hub` feature enabled and no local cache hit, download the
///    model's essential files into that cache, then use it.
///
/// Returns an error — never an empty or synthetic directory — if none of the
/// above produces a real directory containing at least one file.
async fn resolve_model_source_dir(model_id: &str) -> Result<PathBuf> {
    let explicit = Path::new(model_id);
    if explicit.is_dir() {
        return Ok(explicit.to_path_buf());
    }

    let cache_dir = crate::hub::get_cache_dir()?;
    let cached_model_dir = cache_dir.join("models").join(model_id.replace('/', "--")).join("main");
    if cached_model_dir.is_dir() && has_any_files(&cached_model_dir) {
        return Ok(cached_model_dir);
    }

    #[cfg(feature = "hub")]
    {
        let (downloaded_dir, _stats) =
            crate::hub::download_model_enhanced(model_id, None).await.map_err(|e| {
                TrustformersError::io_error(format!(
                    "Failed to download model '{model_id}' to build an offline pack: {e}"
                ))
            })?;
        if has_any_files(&downloaded_dir) {
            return Ok(downloaded_dir);
        }
        Err(TrustformersError::file_not_found(format!(
            "Downloaded model directory for '{model_id}' contains no files"
        )))
    }

    #[cfg(not(feature = "hub"))]
    {
        Err(TrustformersError::invalid_input_simple(format!(
            "Cannot build an offline pack for '{model_id}': it is not a local directory, it is \
             not present in the local cache ({}), and the `hub` feature is disabled so it cannot \
             be downloaded. Provide a local model directory, pre-populate the cache, or rebuild \
             with `--features hub`.",
            cached_model_dir.display()
        )))
    }
}

/// Whether `dir` contains at least one regular file directly inside it.
fn has_any_files(dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    entries.filter_map(|e| e.ok()).any(|e| e.path().is_file())
}

/// Lexically normalize a tar entry's path and confirm it cannot escape
/// `base`, without touching the filesystem (the destination doesn't exist yet
/// during extraction, so canonicalization isn't an option). Any `..`
/// component or absolute-path component is rejected outright rather than
/// "resolved" — the simplest policy that is unambiguously safe against a
/// crafted archive with entries like `../../etc/passwd`.
fn safe_relative_path(base: &Path, entry_name: &str) -> Option<PathBuf> {
    let mut normalized = PathBuf::new();
    for component in Path::new(entry_name).components() {
        match component {
            std::path::Component::Normal(part) => normalized.push(part),
            std::path::Component::CurDir => {},
            std::path::Component::ParentDir
            | std::path::Component::RootDir
            | std::path::Component::Prefix(_) => return None,
        }
    }
    if normalized.as_os_str().is_empty() {
        return None;
    }
    Some(base.join(normalized))
}

/// Offline Model Pack System for TrustformeRS
/// Enables packaging and distribution of model collections for offline deployment

/// Metadata for an offline model pack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPackMetadata {
    pub pack_id: String,
    pub name: String,
    pub description: String,
    pub version: String,
    pub created_at: SystemTime,
    pub created_by: String,
    pub total_size: u64,
    pub models: Vec<PackedModelInfo>,
    pub dependencies: Vec<String>,
    pub target_platforms: Vec<String>,
    pub checksum: String,
    pub compression_ratio: f64,
}

/// Information about a model within a pack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackedModelInfo {
    pub model_id: String,
    pub name: String,
    pub version: String,
    pub original_size: u64,
    pub compressed_size: u64,
    pub model_type: ModelType,
    pub framework: String,
    pub precision: PrecisionType,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    TextGeneration,
    TextClassification,
    ImageClassification,
    SpeechRecognition,
    Translation,
    Summarization,
    QuestionAnswering,
    Multimodal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrecisionType {
    FP32,
    FP16,
    INT8,
    INT4,
    Mixed,
}

/// Configuration for creating model packs
#[derive(Debug, Clone)]
pub struct PackCreationConfig {
    pub compression_level: u8, // 0-9, 9 being highest compression
    pub include_cache: bool,
    pub include_examples: bool,
    pub include_documentation: bool,
    pub target_platforms: Vec<String>,
    pub max_pack_size: Option<u64>, // Maximum pack size in bytes
    pub split_large_packs: bool,
}

impl Default for PackCreationConfig {
    fn default() -> Self {
        Self {
            compression_level: 6,
            include_cache: false,
            include_examples: true,
            include_documentation: true,
            target_platforms: vec![
                "linux".to_string(),
                "windows".to_string(),
                "macos".to_string(),
            ],
            max_pack_size: Some(2 * 1024 * 1024 * 1024), // 2GB default
            split_large_packs: true,
        }
    }
}

/// Offline model pack manager
pub struct OfflineModelPackManager {
    base_path: PathBuf,
    registry: HashMap<String, ModelPackMetadata>,
}

impl OfflineModelPackManager {
    /// Create a new offline model pack manager
    pub fn new(base_path: impl AsRef<Path>) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_path)?;

        let mut manager = Self {
            base_path,
            registry: HashMap::new(),
        };

        manager.load_registry()?;
        Ok(manager)
    }

    /// Create a new model pack from a list of models.
    ///
    /// Model metadata comes from the real Hub API (`get_model_info`, behind
    /// the `hub` feature) or, without it, an honestly-empty `ModelInfo` — see
    /// [`build_pack`](Self::build_pack) for how that combines with each
    /// model's real on-disk files.
    pub async fn create_pack(
        &mut self,
        name: String,
        description: String,
        model_ids: Vec<String>,
        config: PackCreationConfig,
    ) -> Result<String> {
        let mut model_infos = Vec::with_capacity(model_ids.len());
        for model_id in &model_ids {
            let info = self.get_model_info(model_id).await?;
            model_infos.push((model_id.clone(), info));
        }
        self.build_pack(name, description, model_ids, config, model_infos).await
    }

    /// Shared pack-building core used by both [`create_pack`](Self::create_pack)
    /// and [`create_pack_from_hub`](Self::create_pack_from_hub): given each
    /// model's already-resolved [`ModelInfo`], resolve its *real* on-disk
    /// files via [`resolve_model_source_dir`], archive them, and compute
    /// every size (`original_size`, `compressed_size`, `compression_ratio`)
    /// from those real bytes — never from a hardcoded estimate.
    async fn build_pack(
        &mut self,
        name: String,
        description: String,
        model_ids: Vec<String>,
        config: PackCreationConfig,
        model_infos: Vec<(String, ModelInfo)>,
    ) -> Result<String> {
        let pack_id = Uuid::new_v4().to_string();
        let pack_path = self.base_path.join(format!("{}.tfpack", pack_id));

        // Create compressed archive from each model's real files, getting
        // back the real (uncompressed) byte count actually packed per model.
        let (compressed_size, per_model_original_size) =
            self.create_compressed_archive(&model_ids, &pack_path, &config).await?;
        let total_original_size: u64 = per_model_original_size.values().sum();

        let compression_ratio = if total_original_size > 0 {
            compressed_size as f64 / total_original_size as f64
        } else {
            1.0
        };

        let mut models = Vec::with_capacity(model_infos.len());
        for (model_id, model_info) in &model_infos {
            let original_size = per_model_original_size.get(model_id).copied().unwrap_or(0);
            models.push(PackedModelInfo {
                model_id: model_id.clone(),
                name: model_info.model_id.clone(),
                version: "latest".to_string(), // Could be made configurable
                original_size,
                compressed_size: (original_size as f64 * compression_ratio) as u64,
                model_type: self.infer_model_type(model_info),
                framework: model_info
                    .library_name
                    .clone()
                    .unwrap_or_else(|| "transformers".to_string()),
                precision: PrecisionType::FP32, // Default, could be detected
                metadata: self.extract_metadata_from_model_info(model_info),
            });
        }

        // Generate checksum
        let checksum = self.calculate_file_checksum(&pack_path)?;

        // Create metadata
        let metadata = ModelPackMetadata {
            pack_id: pack_id.clone(),
            name: name.clone(),
            description,
            version: "1.0.0".to_string(),
            created_at: SystemTime::now(),
            created_by: "trustformers".to_string(),
            total_size: compressed_size,
            models,
            dependencies: Vec::new(), // Could be enhanced to detect dependencies
            target_platforms: config.target_platforms.clone(),
            checksum,
            compression_ratio,
        };

        // Save metadata
        self.save_pack_metadata(&metadata)?;
        self.registry.insert(pack_id.clone(), metadata);

        Ok(pack_id)
    }

    /// Install a model pack
    pub async fn install_pack(&mut self, pack_path: impl AsRef<Path>) -> Result<String> {
        let pack_path = pack_path.as_ref();

        // Verify pack integrity
        let metadata = self.load_pack_metadata(pack_path)?;
        self.verify_pack_integrity(pack_path, &metadata)?;

        // Extract pack to installation directory
        let install_path = self.base_path.join("installed").join(&metadata.pack_id);
        std::fs::create_dir_all(&install_path)?;

        self.extract_pack(pack_path, &install_path).await?;

        // Register pack
        self.registry.insert(metadata.pack_id.clone(), metadata.clone());
        self.save_registry()?;

        Ok(metadata.pack_id)
    }

    /// List available packs
    pub fn list_packs(&self) -> Vec<&ModelPackMetadata> {
        self.registry.values().collect()
    }

    /// Get pack information
    pub fn get_pack_info(&self, pack_id: &str) -> Option<&ModelPackMetadata> {
        self.registry.get(pack_id)
    }

    /// Remove a pack
    pub async fn remove_pack(&mut self, pack_id: &str) -> Result<()> {
        if let Some(metadata) = self.registry.remove(pack_id) {
            // Remove installed files
            let install_path = self.base_path.join("installed").join(&metadata.pack_id);
            if install_path.exists() {
                tokio::fs::remove_dir_all(&install_path).await?;
            }

            // Remove pack file
            let pack_path = self.base_path.join(format!("{}.tfpack", pack_id));
            if pack_path.exists() {
                tokio::fs::remove_file(&pack_path).await?;
            }

            self.save_registry()?;
        }

        Ok(())
    }

    /// Create a curated pack for specific use cases
    pub async fn create_curated_pack(
        &mut self,
        pack_type: CuratedPackType,
        config: PackCreationConfig,
    ) -> Result<String> {
        let (name, description, model_ids) = match pack_type {
            CuratedPackType::NLP => (
                "NLP Essentials".to_string(),
                "Essential models for natural language processing tasks".to_string(),
                vec![
                    "bert-base-uncased".to_string(),
                    "gpt2".to_string(),
                    "distilbert-base-uncased".to_string(),
                    "roberta-base".to_string(),
                ],
            ),
            CuratedPackType::Vision => (
                "Computer Vision Pack".to_string(),
                "Essential models for computer vision tasks".to_string(),
                vec![
                    "vit-base-patch16-224".to_string(),
                    "resnet-50".to_string(),
                    "clip-vit-base-patch32".to_string(),
                ],
            ),
            CuratedPackType::Multimodal => (
                "Multimodal AI Pack".to_string(),
                "Models for cross-modal understanding and generation".to_string(),
                vec![
                    "clip-vit-base-patch32".to_string(),
                    "blip-image-captioning-base".to_string(),
                    "layoutlm-base-uncased".to_string(),
                ],
            ),
            CuratedPackType::EdgeOptimized => (
                "Edge Deployment Pack".to_string(),
                "Optimized models for edge and mobile deployment".to_string(),
                vec![
                    "distilbert-base-uncased".to_string(),
                    "mobilenet-v2".to_string(),
                    "efficientnet-b0".to_string(),
                ],
            ),
        };

        self.create_pack(name, description, model_ids, config).await
    }

    /// Update a pack with new models or versions
    pub async fn update_pack(
        &mut self,
        pack_id: &str,
        additional_models: Vec<String>,
    ) -> Result<String> {
        let existing_metadata = self
            .registry
            .get(pack_id)
            .ok_or_else(|| {
                TrustformersError::file_not_found(format!("Pack {} not found", pack_id))
            })?
            .clone();

        // Combine existing and new models
        let mut all_models: Vec<String> =
            existing_metadata.models.iter().map(|m| m.model_id.clone()).collect();
        all_models.extend(additional_models);

        // Create new pack with updated content
        let new_pack_id = self
            .create_pack(
                format!("{} (Updated)", existing_metadata.name),
                existing_metadata.description,
                all_models,
                PackCreationConfig::default(),
            )
            .await?;

        // Remove old pack
        self.remove_pack(pack_id).await?;

        Ok(new_pack_id)
    }

    // Private helper methods

    /// Look up a model's Hub metadata, or — when `model_id` is itself a local
    /// directory (the same convention [`resolve_model_source_dir`] uses) —
    /// skip the network entirely: there is no Hub repo id to query, and
    /// sending a local filesystem path to the Hub API as a "model id" would
    /// be both pointless and, in tests, an unwanted network call.
    async fn get_model_info(&self, model_id: &str) -> Result<ModelInfo> {
        if Path::new(model_id).is_dir() {
            return Ok(empty_model_info(model_id));
        }
        self.get_model_info_remote(model_id).await
    }

    /// Query the real Hugging Face Hub API for model metadata.
    ///
    /// Mirrors `hub.rs::get_download_stats`'s existing pattern for this exact
    /// endpoint (`GET /api/models/{id}`): fetch, parse as a generic
    /// `serde_json::Value`, then hand off to [`model_info_from_hub_json`] for
    /// the actual field mapping.
    #[cfg(feature = "hub")]
    async fn get_model_info_remote(&self, model_id: &str) -> Result<ModelInfo> {
        let url = format!("https://huggingface.co/api/models/{model_id}");
        let client = reqwest::Client::new();

        let response = client.get(&url).send().await.map_err(|e| TrustformersError::Hub {
            message: format!("Failed to fetch model info for '{}': {}", model_id, e),
            model_id: model_id.to_string(),
            endpoint: Some(url.clone()),
            suggestion: Some(
                "Check network connectivity and that the model ID is correct".to_string(),
            ),
            recovery_actions: vec![],
        })?;

        if !response.status().is_success() {
            return Err(TrustformersError::Hub {
                message: format!(
                    "Failed to fetch model info for '{}': HTTP {}",
                    model_id,
                    response.status()
                ),
                model_id: model_id.to_string(),
                endpoint: Some(url.clone()),
                suggestion: Some(
                    "Check that the model ID exists on the Hugging Face Hub".to_string(),
                ),
                recovery_actions: vec![],
            });
        }

        let json: serde_json::Value = response.json().await.map_err(|e| {
            TrustformersError::invalid_input(
                format!(
                    "Failed to parse model info response for '{}': {}",
                    model_id, e
                ),
                Some("api_response"),
                Some("valid JSON model info object"),
                Some("invalid JSON format"),
            )
        })?;

        Ok(model_info_from_hub_json(model_id, &json))
    }

    /// Honest empty model info used when the `hub` feature (networking) is
    /// disabled: there is no way to know a model's real Hub metadata
    /// (downloads, likes, pipeline tag, ...) without a network call, so every
    /// such field is `None`/empty rather than a guessed placeholder. Pack
    /// creation itself still works without the `hub` feature — real files
    /// are resolved via `resolve_model_source_dir`, which also checks the
    /// local cache and an explicit local directory — only this Hub-side
    /// metadata is genuinely unavailable.
    #[cfg(not(feature = "hub"))]
    async fn get_model_info_remote(&self, model_id: &str) -> Result<ModelInfo> {
        Ok(empty_model_info(model_id))
    }

    /// Build the compressed pack archive from each model's *real* on-disk
    /// files (resolved via [`resolve_model_source_dir`] — an explicit local
    /// directory, the Hub cache, or, with the `hub` feature, a fresh
    /// download). Fabricating a placeholder `config.json` is not an option:
    /// a model whose files can't be resolved fails the whole pack rather
    /// than silently producing an empty entry.
    ///
    /// Returns `(compressed_archive_size, per_model_original_size)` — the
    /// latter is the real sum of bytes packed for each model, used by
    /// [`build_pack`](Self::build_pack) instead of a hardcoded estimate.
    async fn create_compressed_archive(
        &self,
        model_ids: &[String],
        output_path: &Path,
        config: &PackCreationConfig,
    ) -> Result<(u64, HashMap<String, u64>)> {
        use oxiarc_archive::tar::TarWriter;
        use oxiarc_deflate::streaming::GzipStreamEncoder;

        let file = File::create(output_path)?;
        let encoder = GzipStreamEncoder::new(file, 6);
        let mut tar_writer = TarWriter::new(encoder);

        // Create pack metadata
        let metadata = serde_json::json!({
            "version": "1.0",
            "compression": format!("{:?}", config.compression_level),
            "models": model_ids.len(),
            "created": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            "split_large_packs": config.split_large_packs,
            "model_ids": model_ids
        });

        // Add metadata file to archive
        let metadata_content = serde_json::to_string_pretty(&metadata)?;
        tar_writer
            .add_file_with_mode("pack_metadata.json", metadata_content.as_bytes(), 0o644)
            .map_err(|e| TrustformersError::invalid_input_simple(e.to_string()))?;

        // Add each model's real files to the archive.
        let mut per_model_original_size = HashMap::with_capacity(model_ids.len());
        for model_id in model_ids {
            let source_dir = resolve_model_source_dir(model_id).await?;

            let entries = std::fs::read_dir(&source_dir).map_err(|e| TrustformersError::Io {
                message: format!(
                    "Failed to read model directory '{}': {e}",
                    source_dir.display()
                ),
                path: Some(source_dir.to_string_lossy().to_string()),
                suggestion: None,
            })?;

            let mut model_bytes: u64 = 0;
            let mut file_count = 0usize;
            for entry in entries {
                let entry = entry.map_err(|e| TrustformersError::io_error(e.to_string()))?;
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                let Some(file_name) = path.file_name().and_then(|n| n.to_str()) else {
                    continue;
                };
                if file_name.starts_with('.') {
                    continue; // skip hidden/lock files
                }

                let content = std::fs::read(&path).map_err(|e| TrustformersError::Io {
                    message: format!("Failed to read model file '{}': {e}", path.display()),
                    path: Some(path.to_string_lossy().to_string()),
                    suggestion: None,
                })?;
                let archive_path = format!("models/{model_id}/{file_name}");
                tar_writer
                    .add_file_with_mode(&archive_path, &content, 0o644)
                    .map_err(|e| TrustformersError::invalid_input_simple(e.to_string()))?;

                model_bytes += content.len() as u64;
                file_count += 1;
            }

            if file_count == 0 {
                return Err(TrustformersError::invalid_input_simple(format!(
                    "Model directory '{}' for '{model_id}' contains no files to pack",
                    source_dir.display()
                )));
            }
            per_model_original_size.insert(model_id.clone(), model_bytes);
        }

        // Consume tar_writer, writing the trailing zero blocks and returning the encoder
        let encoder = tar_writer
            .into_inner()
            .map_err(|e| TrustformersError::invalid_input_simple(e.to_string()))?;

        // Flush and finalise the gzip stream
        encoder
            .finish()
            .map_err(|e| TrustformersError::invalid_input_simple(e.to_string()))?;

        // Calculate final archive size from the real bytes written.
        let final_size = output_path.metadata()?.len();

        Ok((final_size, per_model_original_size))
    }

    fn calculate_file_checksum(&self, file_path: &Path) -> Result<String> {
        let mut file = File::open(file_path)?;
        let mut hasher = Sha256::new();
        let mut buffer = [0; 8192];

        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }

        Ok(hex::encode(hasher.finalize()))
    }

    fn save_pack_metadata(&self, metadata: &ModelPackMetadata) -> Result<()> {
        let metadata_path = self.base_path.join(format!("{}.metadata.json", metadata.pack_id));
        let file = File::create(metadata_path)?;
        serde_json::to_writer_pretty(file, metadata)?;
        Ok(())
    }

    /// Infer model type from model information
    fn infer_model_type(&self, model_info: &ModelInfo) -> ModelType {
        match model_info.pipeline_tag.as_deref() {
            Some("text-generation") => ModelType::TextGeneration,
            Some("text-classification") => ModelType::TextClassification,
            Some("image-classification") => ModelType::ImageClassification,
            Some("automatic-speech-recognition") => ModelType::SpeechRecognition,
            Some("translation") => ModelType::Translation,
            Some("summarization") => ModelType::Summarization,
            Some("question-answering") => ModelType::QuestionAnswering,
            _ => ModelType::TextGeneration, // Default fallback
        }
    }

    fn load_pack_metadata(&self, pack_path: &Path) -> Result<ModelPackMetadata> {
        // Extract metadata from pack or look for accompanying .metadata.json file
        let pack_stem = pack_path.file_stem().ok_or_else(|| {
            TrustformersError::invalid_input_simple("Invalid pack file name".to_string())
        })?;
        let metadata_path =
            pack_path.with_file_name(format!("{}.metadata.json", pack_stem.to_string_lossy()));

        if metadata_path.exists() {
            let file = File::open(metadata_path)?;
            let metadata: ModelPackMetadata = serde_json::from_reader(file)?;
            Ok(metadata)
        } else {
            Err(TrustformersError::invalid_input_simple(
                "Pack metadata not found".to_string(),
            ))
        }
    }

    fn verify_pack_integrity(&self, pack_path: &Path, metadata: &ModelPackMetadata) -> Result<()> {
        let calculated_checksum = self.calculate_file_checksum(pack_path)?;
        if calculated_checksum != metadata.checksum {
            return Err(TrustformersError::invalid_input_simple(
                "Pack checksum mismatch".to_string(),
            ));
        }
        Ok(())
    }

    async fn extract_pack(&self, pack_path: &Path, extract_path: &Path) -> Result<()> {
        use oxiarc_archive::tar::TarStreamReader;
        use oxiarc_deflate::streaming::GzipStreamDecoder;
        use std::io::Read as _;

        // TAR typeflag constants
        const TAR_REGULAR_FILE: u8 = b'0';
        const TAR_REGULAR_FILE_ALT: u8 = 0;
        const TAR_DIRECTORY: u8 = b'5';

        std::fs::create_dir_all(extract_path)?;

        let file = File::open(pack_path)?;
        let decoder = GzipStreamDecoder::new(file);
        let mut stream = TarStreamReader::new(decoder);

        // Extract all entries from the archive manually
        while let Some(mut entry) = stream
            .next_entry()
            .map_err(|e| TrustformersError::invalid_input_simple(e.to_string()))?
        {
            let entry_name = entry.header.name.clone();
            let typeflag = entry.header.typeflag;

            // Reject any entry whose path cannot be safely joined onto
            // `extract_path` (`..` components, absolute paths, ...) — a tar
            // entry named e.g. `../../etc/passwd` must never be allowed to
            // write outside the extraction directory.
            let Some(dest) = safe_relative_path(extract_path, &entry_name) else {
                return Err(TrustformersError::invalid_input_simple(format!(
                    "Refusing to extract pack entry with an unsafe path: '{entry_name}'"
                )));
            };

            match typeflag {
                TAR_DIRECTORY => {
                    std::fs::create_dir_all(&dest)?;
                },
                TAR_REGULAR_FILE | TAR_REGULAR_FILE_ALT => {
                    // Ensure parent directories exist
                    if let Some(parent) = dest.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    let mut out_file = File::create(&dest)?;
                    let mut buf = Vec::new();
                    entry
                        .read_to_end(&mut buf)
                        .map_err(|e| TrustformersError::invalid_input_simple(e.to_string()))?;
                    std::io::Write::write_all(&mut out_file, &buf)?;
                },
                // Skip symlinks (b'2'), hardlinks (b'1'), and unknown types for security
                _ => {},
            }
        }

        // Read the pack metadata that was extracted
        let metadata_path = extract_path.join("pack_metadata.json");
        let manifest = if metadata_path.exists() {
            // Use the extracted metadata as manifest
            let metadata_content = std::fs::read_to_string(&metadata_path)?;
            let mut metadata: serde_json::Value = serde_json::from_str(&metadata_content)?;

            // Add extraction time
            metadata["extraction_time"] = serde_json::json!(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs());

            metadata
        } else {
            // Fallback manifest if no metadata found
            serde_json::json!({
                "extraction_time": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                "pack_source": pack_path.display().to_string()
            })
        };

        // Write extraction manifest
        let manifest_path = extract_path.join("manifest.json");
        std::fs::write(manifest_path, serde_json::to_string_pretty(&manifest)?)?;

        Ok(())
    }

    /// Load the pack registry from disk.
    ///
    /// A missing registry file is fine (a fresh, empty registry). A *present
    /// but corrupt* one is not silently discarded — that would quietly drop
    /// every previously-tracked pack with no indication anything was wrong —
    /// so it is a hard error instead.
    fn load_registry(&mut self) -> Result<()> {
        let registry_path = self.base_path.join("registry.json");
        if registry_path.exists() {
            let file = File::open(&registry_path)?;
            self.registry = serde_json::from_reader(file).map_err(|e| {
                TrustformersError::invalid_input_simple(format!(
                    "Pack registry at '{}' is corrupt and could not be parsed: {e}. Remove or \
                     repair the file to continue.",
                    registry_path.display()
                ))
            })?;
        }
        Ok(())
    }

    fn save_registry(&self) -> Result<()> {
        let registry_path = self.base_path.join("registry.json");
        let file = File::create(registry_path)?;
        serde_json::to_writer_pretty(file, &self.registry)?;
        Ok(())
    }

    /// Extract metadata from model information
    fn extract_metadata_from_model_info(&self, model_info: &ModelInfo) -> HashMap<String, String> {
        let mut metadata = HashMap::new();

        // Basic information
        if let Some(author) = &model_info.author {
            metadata.insert("author".to_string(), author.clone());
        }

        if let Some(description) = &model_info.description {
            metadata.insert("description".to_string(), description.clone());
        }

        if let Some(license) = &model_info.license {
            metadata.insert("license".to_string(), license.clone());
        }

        if let Some(created_at) = &model_info.created_at {
            metadata.insert("created_at".to_string(), created_at.clone());
        }

        if let Some(updated_at) = &model_info.updated_at {
            metadata.insert("updated_at".to_string(), updated_at.clone());
        }

        // Statistics
        if let Some(downloads) = model_info.downloads {
            metadata.insert("downloads".to_string(), downloads.to_string());
        }

        if let Some(likes) = model_info.likes {
            metadata.insert("likes".to_string(), likes.to_string());
        }

        // Task and architecture information
        if let Some(task) = &model_info.task {
            metadata.insert("task".to_string(), task.clone());
        }

        if let Some(architecture) = &model_info.architecture {
            metadata.insert("architecture".to_string(), architecture.clone());
        }

        if let Some(model_type) = &model_info.model_type {
            metadata.insert("model_type".to_string(), model_type.clone());
        }

        if let Some(pipeline_tag) = &model_info.pipeline_tag {
            metadata.insert("pipeline_tag".to_string(), pipeline_tag.clone());
        }

        // Language and datasets
        if !model_info.language.is_empty() {
            metadata.insert("language".to_string(), model_info.language.join(", "));
        }

        if !model_info.dataset.is_empty() {
            metadata.insert("datasets".to_string(), model_info.dataset.join(", "));
        }

        // Tags
        if !model_info.tags.is_empty() {
            metadata.insert("tags".to_string(), model_info.tags.join(", "));
        }

        // Configuration details (convert JSON values to strings)
        for (key, value) in &model_info.config {
            match value {
                serde_json::Value::String(s) => {
                    metadata.insert(format!("config_{}", key), s.clone());
                },
                serde_json::Value::Number(n) => {
                    metadata.insert(format!("config_{}", key), n.to_string());
                },
                serde_json::Value::Bool(b) => {
                    metadata.insert(format!("config_{}", key), b.to_string());
                },
                _ => {
                    metadata.insert(format!("config_{}", key), value.to_string());
                },
            }
        }

        metadata
    }
}

/// Curated pack types for common use cases
#[derive(Debug, Clone)]
pub enum CuratedPackType {
    NLP,
    Vision,
    Multimodal,
    EdgeOptimized,
}

/// Factory functions for creating specialized packs
impl OfflineModelPackManager {
    /// Create a development pack with essential models for prototyping
    pub async fn create_development_pack(&mut self) -> Result<String> {
        self.create_curated_pack(
            CuratedPackType::NLP,
            PackCreationConfig {
                compression_level: 9,
                include_examples: true,
                include_documentation: true,
                ..Default::default()
            },
        )
        .await
    }

    /// Create a production pack optimized for deployment
    pub async fn create_production_pack(&mut self, target_platform: String) -> Result<String> {
        self.create_curated_pack(
            CuratedPackType::EdgeOptimized,
            PackCreationConfig {
                compression_level: 9,
                include_cache: false,
                include_examples: false,
                include_documentation: false,
                target_platforms: vec![target_platform],
                max_pack_size: Some(1024 * 1024 * 1024), // 1GB for production
                ..Default::default()
            },
        )
        .await
    }
}

/// Hub integration for offline packs
/// Provides bridge between online Hub functionality and offline model packs
pub struct HubIntegration {
    pub hub_options: crate::hub::HubOptions,
}

impl HubIntegration {
    /// Create a new Hub integration instance
    pub fn new(options: Option<crate::hub::HubOptions>) -> Self {
        Self {
            hub_options: options.unwrap_or_default(),
        }
    }

    /// Download model from Hub and add it to an offline pack
    pub async fn download_model_to_pack(
        &self,
        pack_manager: &mut OfflineModelPackManager,
        model_id: &str,
        pack_id: &str,
    ) -> Result<()> {
        // Download model from Hub using existing hub functionality
        let _model_path = crate::hub::download_file_from_hub(
            model_id,
            "config.json",
            Some(self.hub_options.clone()),
        )
        .map_err(|e| TrustformersError::io_error(format!("Hub download failed: {}", e)))?;

        // Get model info from Hub
        let model_info = self.get_hub_model_info(model_id).await?;

        // Update existing pack with new model
        let additional_models = vec![model_id.to_string()];
        pack_manager.update_pack(pack_id, additional_models).await?;

        Ok(())
    }

    /// Create a pack from Hub model collection
    pub async fn create_pack_from_hub_collection(
        &self,
        pack_manager: &mut OfflineModelPackManager,
        collection_name: &str,
        model_ids: Vec<String>,
        config: PackCreationConfig,
    ) -> Result<String> {
        // Verify all models exist on Hub before creating pack
        for model_id in &model_ids {
            let _ = self.get_hub_model_info(model_id).await?;
        }

        // Create pack using verified models
        pack_manager
            .create_pack(
                format!("Hub Collection: {}", collection_name),
                format!(
                    "Model pack created from Hub collection: {}",
                    collection_name
                ),
                model_ids,
                config,
            )
            .await
    }

    /// Get model information from Hub.
    ///
    /// When `model_id` is itself a local directory (the same convention
    /// [`resolve_model_source_dir`] uses), there is no Hub repo id to look up
    /// a model card for, so the network call is skipped entirely rather than
    /// sending a filesystem path to `crate::hub::load_model_card_from_hub`.
    async fn get_hub_model_info(&self, model_id: &str) -> Result<ModelInfo> {
        if Path::new(model_id).is_dir() {
            return Ok(empty_model_info(model_id));
        }

        // Try to load model card from Hub
        match crate::hub::load_model_card_from_hub(model_id, Some(self.hub_options.clone())) {
            Ok(model_card) => {
                // Convert model card to ModelInfo
                Ok(ModelInfo {
                    model_id: model_id.to_string(),
                    library_name: Some("transformers".to_string()),
                    pipeline_tag: model_card.pipeline_tag.clone(),
                    tags: model_card.tags.unwrap_or_default(),
                    config: model_card.extra.into_iter().collect(),
                    downloads: None, // Not available in model card
                    likes: None,     // Not available in model card
                    created_at: None,
                    updated_at: None,
                    author: None,
                    description: None,
                    license: model_card.license,
                    task: model_card.pipeline_tag,
                    language: model_card.language.unwrap_or_default(),
                    dataset: model_card.datasets.unwrap_or_default(),
                    model_type: None,
                    architecture: None,
                })
            },
            Err(_) => {
                // No network access (or the model card genuinely doesn't
                // exist): we don't know anything about this model beyond its
                // id, so every Hub-side field is honestly `None`/empty
                // rather than a guessed placeholder.
                Ok(empty_model_info(model_id))
            },
        }
    }
}

/// Enhanced OfflineModelPackManager with Hub integration
impl OfflineModelPackManager {
    /// Create a new pack manager with Hub integration
    pub fn with_hub_integration(
        base_path: impl AsRef<Path>,
        hub_options: Option<crate::hub::HubOptions>,
    ) -> Result<(Self, HubIntegration)> {
        let manager = Self::new(base_path)?;
        let hub_integration = HubIntegration::new(hub_options);
        Ok((manager, hub_integration))
    }

    /// Create pack from Hub models using integration.
    ///
    /// Unlike [`create_pack`](Self::create_pack) (which uses the plain
    /// `/api/models/{id}` lookup), each model's [`ModelInfo`] here comes from
    /// `hub_integration.get_hub_model_info` (the model card). Both funnel
    /// into the same [`build_pack`](Self::build_pack), so the real file
    /// resolution, archiving, and size accounting are identical — only the
    /// metadata source differs.
    pub async fn create_pack_from_hub(
        &mut self,
        hub_integration: &HubIntegration,
        name: String,
        description: String,
        model_ids: Vec<String>,
        config: PackCreationConfig,
    ) -> Result<String> {
        let mut model_infos = Vec::with_capacity(model_ids.len());
        for model_id in &model_ids {
            let info = hub_integration.get_hub_model_info(model_id).await?;
            model_infos.push((model_id.clone(), info));
        }
        self.build_pack(name, description, model_ids, config, model_infos).await
    }
}

// ================================================================================================
// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
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
