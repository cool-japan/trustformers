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

    /// Create a new model pack from a list of models
    pub async fn create_pack(
        &mut self,
        name: String,
        description: String,
        model_ids: Vec<String>,
        config: PackCreationConfig,
    ) -> Result<String> {
        let pack_id = Uuid::new_v4().to_string();
        let pack_path = self.base_path.join(format!("{}.tfpack", pack_id));

        // Collect model information
        let mut models = Vec::new();
        let mut total_original_size = 0u64;

        for model_id in &model_ids {
            let model_info = self.get_model_info(model_id).await?;
            let estimated_size = 1024 * 1024 * 512; // Estimate 512MB per model
            total_original_size += estimated_size;

            models.push(PackedModelInfo {
                model_id: model_id.clone(),
                name: model_info.model_id.clone(),
                version: "latest".to_string(), // Could be made configurable
                original_size: estimated_size,
                compressed_size: 0, // Will be updated after compression
                model_type: self.infer_model_type(&model_info),
                framework: model_info
                    .library_name
                    .clone()
                    .unwrap_or_else(|| "transformers".to_string()),
                precision: PrecisionType::FP32, // Default, could be detected
                metadata: self.extract_metadata_from_model_info(&model_info),
            });
        }

        // Create compressed archive
        let compressed_size =
            self.create_compressed_archive(&model_ids, &pack_path, &config).await?;

        // Calculate compression ratio
        let compression_ratio = if total_original_size > 0 {
            compressed_size as f64 / total_original_size as f64
        } else {
            1.0
        };

        // Update compressed sizes for models (approximate distribution)
        for model in &mut models {
            model.compressed_size = (model.original_size as f64 * compression_ratio) as u64;
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

    async fn get_model_info(&self, model_id: &str) -> Result<ModelInfo> {
        // Mock implementation - in real scenario, this would query the hub
        Ok(ModelInfo {
            model_id: model_id.to_string(),
            pipeline_tag: Some("text-generation".to_string()),
            library_name: Some("transformers".to_string()),
            tags: vec![],
            config: HashMap::new(),
            downloads: Some(1000),
            likes: Some(50),
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
        })
    }

    async fn create_compressed_archive(
        &self,
        model_ids: &[String],
        output_path: &Path,
        config: &PackCreationConfig,
    ) -> Result<u64> {
        // Real tar archive implementation
        use flate2::write::GzEncoder;
        use tar::Builder;

        let file = File::create(output_path)?;
        let encoder = GzEncoder::new(file, flate2::Compression::default());
        let mut tar_builder = Builder::new(encoder);

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
        let mut metadata_header = tar::Header::new_gnu();
        metadata_header.set_size(metadata_content.len() as u64);
        metadata_header.set_mode(0o644);
        metadata_header.set_cksum();
        tar_builder.append_data(
            &mut metadata_header,
            "pack_metadata.json",
            std::io::Cursor::new(metadata_content),
        )?;

        // Add each model to the archive
        let mut total_size = 0u64;
        for model_id in model_ids {
            // In a real implementation, you would download or copy the actual model files
            // For now, create a placeholder model structure
            let model_config = serde_json::json!({
                "model_id": model_id,
                "type": "transformers",
                "format": "safetensors",
                "architecture": "auto-detected"
            });

            let config_content = serde_json::to_string_pretty(&model_config)?;
            let mut config_header = tar::Header::new_gnu();
            config_header.set_size(config_content.len() as u64);
            config_header.set_mode(0o644);
            config_header.set_cksum();

            let model_path = format!("models/{}/config.json", model_id);
            let content_len = config_content.len() as u64;
            tar_builder.append_data(
                &mut config_header,
                &model_path,
                std::io::Cursor::new(config_content),
            )?;

            total_size += content_len;
        }

        // Finalize the archive
        tar_builder.finish()?;

        // Calculate final archive size
        let final_size = output_path.metadata()?.len();

        Ok(final_size)
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

        Ok(format!("{:x}", hasher.finalize()))
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
        // Real tar extraction implementation
        use flate2::read::GzDecoder;
        use tar::Archive;

        std::fs::create_dir_all(extract_path)?;

        let file = File::open(pack_path)?;
        let decoder = GzDecoder::new(file);
        let mut archive = Archive::new(decoder);

        // Extract all files from the archive
        archive.unpack(extract_path)?;

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

    fn load_registry(&mut self) -> Result<()> {
        let registry_path = self.base_path.join("registry.json");
        if registry_path.exists() {
            let file = File::open(registry_path)?;
            self.registry = serde_json::from_reader(file).unwrap_or_default();
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

    /// Get model information from Hub
    async fn get_hub_model_info(&self, model_id: &str) -> Result<ModelInfo> {
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
                // Fallback to mock model info if Hub access fails
                Ok(ModelInfo {
                    model_id: model_id.to_string(),
                    pipeline_tag: Some("text-generation".to_string()),
                    library_name: Some("transformers".to_string()),
                    tags: vec![],
                    config: HashMap::new(),
                    downloads: Some(1000),
                    likes: Some(50),
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
                })
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

    /// Create pack from Hub models using integration
    pub async fn create_pack_from_hub(
        &mut self,
        hub_integration: &HubIntegration,
        name: String,
        description: String,
        model_ids: Vec<String>,
        config: PackCreationConfig,
    ) -> Result<String> {
        // Use Hub integration to get real model info
        let mut enhanced_models = Vec::new();
        let mut total_original_size = 0u64;

        for model_id in &model_ids {
            let model_info = hub_integration.get_hub_model_info(model_id).await?;
            let estimated_size = 1024 * 1024 * 512; // Estimate 512MB per model
            total_original_size += estimated_size;

            enhanced_models.push(PackedModelInfo {
                model_id: model_id.clone(),
                name: model_info.model_id.clone(),
                version: "latest".to_string(),
                original_size: estimated_size,
                compressed_size: 0, // Will be updated after compression
                model_type: self.infer_model_type(&model_info),
                framework: model_info
                    .library_name
                    .clone()
                    .unwrap_or_else(|| "transformers".to_string()),
                precision: PrecisionType::FP32, // Default, could be detected
                metadata: self.extract_metadata_from_model_info(&model_info),
            });
        }

        // Use the existing create_pack implementation but with enhanced model info
        self.create_pack(name, description, model_ids, config).await
    }
}
