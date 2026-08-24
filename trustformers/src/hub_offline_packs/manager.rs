//! The `OfflineModelPackManager` engine: creating, curating, listing, verifying and extracting offline model packs.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, TrustformersError};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use uuid::Uuid;

use super::hub_integration::HubIntegration;
use super::pack_types::{
    ModelPackMetadata, ModelType, PackCreationConfig, PackedModelInfo, PrecisionType,
};
#[cfg(feature = "hub")]
use super::resolution::model_info_from_hub_json;
use super::resolution::{
    empty_model_info, resolve_model_source_dir, safe_relative_path, ModelInfo,
};

/// Curated pack types for common use cases
#[derive(Debug, Clone)]
pub enum CuratedPackType {
    NLP,
    Vision,
    Multimodal,
    EdgeOptimized,
}
/// Offline model pack manager
pub struct OfflineModelPackManager {
    pub(super) base_path: PathBuf,
    pub(super) registry: HashMap<String, ModelPackMetadata>,
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
    pub(super) async fn build_pack(
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
    pub(super) async fn get_model_info(&self, model_id: &str) -> Result<ModelInfo> {
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
    pub(super) async fn get_model_info_remote(&self, model_id: &str) -> Result<ModelInfo> {
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
    pub(super) async fn get_model_info_remote(&self, model_id: &str) -> Result<ModelInfo> {
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
    pub(super) async fn create_compressed_archive(
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

    pub(super) fn calculate_file_checksum(&self, file_path: &Path) -> Result<String> {
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

    pub(super) fn save_pack_metadata(&self, metadata: &ModelPackMetadata) -> Result<()> {
        let metadata_path = self.base_path.join(format!("{}.metadata.json", metadata.pack_id));
        let file = File::create(metadata_path)?;
        serde_json::to_writer_pretty(file, metadata)?;
        Ok(())
    }

    /// Infer model type from model information
    pub(super) fn infer_model_type(&self, model_info: &ModelInfo) -> ModelType {
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

    pub(super) fn load_pack_metadata(&self, pack_path: &Path) -> Result<ModelPackMetadata> {
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

    pub(super) fn verify_pack_integrity(
        &self,
        pack_path: &Path,
        metadata: &ModelPackMetadata,
    ) -> Result<()> {
        let calculated_checksum = self.calculate_file_checksum(pack_path)?;
        if calculated_checksum != metadata.checksum {
            return Err(TrustformersError::invalid_input_simple(
                "Pack checksum mismatch".to_string(),
            ));
        }
        Ok(())
    }

    pub(super) async fn extract_pack(&self, pack_path: &Path, extract_path: &Path) -> Result<()> {
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
    pub(super) fn load_registry(&mut self) -> Result<()> {
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

    pub(super) fn save_registry(&self) -> Result<()> {
        let registry_path = self.base_path.join("registry.json");
        let file = File::create(registry_path)?;
        serde_json::to_writer_pretty(file, &self.registry)?;
        Ok(())
    }

    /// Extract metadata from model information
    pub(super) fn extract_metadata_from_model_info(
        &self,
        model_info: &ModelInfo,
    ) -> HashMap<String, String> {
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
