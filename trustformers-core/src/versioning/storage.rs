//! Model storage backend for artifacts and metadata

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use uuid::Uuid;

/// Storage backend trait for model artifacts
#[async_trait]
pub trait ModelStorage: Send + Sync {
    /// Store artifacts and return their IDs
    async fn store_artifacts(&self, artifacts: &[Artifact]) -> Result<Vec<Uuid>>;

    /// Retrieve an artifact by ID
    async fn get_artifact(&self, artifact_id: Uuid) -> Result<Option<Artifact>>;

    /// Delete artifacts by IDs
    async fn delete_artifacts(&self, artifact_ids: &[Uuid]) -> Result<()>;

    /// Archive artifacts for a version
    async fn archive_version(&self, version_id: Uuid) -> Result<()>;

    /// Delete all artifacts for a version
    async fn delete_version(&self, version_id: Uuid) -> Result<()>;

    /// List all artifacts for a version
    async fn list_artifacts(&self, version_id: Uuid) -> Result<Vec<Artifact>>;
}

/// Model artifact types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ArtifactType {
    /// Model weights/parameters
    Model,
    /// Model configuration
    Config,
    /// Tokenizer files
    Tokenizer,
    /// Vocabulary files
    Vocabulary,
    /// Training checkpoints
    Checkpoint,
    /// Optimization state
    OptimizerState,
    /// Model architecture definition
    Architecture,
    /// Preprocessing pipeline
    Preprocessing,
    /// Evaluation metrics
    Metrics,
    /// Documentation
    Documentation,
    /// Custom artifact type
    Custom(String),
}

impl ArtifactType {
    /// Get file extension for artifact type
    pub fn default_extension(&self) -> &'static str {
        match self {
            ArtifactType::Model => "bin",
            ArtifactType::Config => "json",
            ArtifactType::Tokenizer => "json",
            ArtifactType::Vocabulary => "txt",
            ArtifactType::Checkpoint => "ckpt",
            ArtifactType::OptimizerState => "bin",
            ArtifactType::Architecture => "json",
            ArtifactType::Preprocessing => "json",
            ArtifactType::Metrics => "json",
            ArtifactType::Documentation => "md",
            ArtifactType::Custom(_) => "bin",
        }
    }

    /// Check if artifact type is required for deployment
    pub fn is_required_for_deployment(&self) -> bool {
        matches!(self, ArtifactType::Model | ArtifactType::Config)
    }
}

/// Model artifact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// Unique identifier
    pub id: Uuid,
    /// Artifact type
    pub artifact_type: ArtifactType,
    /// Original file path
    pub file_path: PathBuf,
    /// File size in bytes
    pub size_bytes: u64,
    /// Content hash (SHA256)
    pub content_hash: String,
    /// MIME type
    pub mime_type: String,
    /// Binary content
    pub content: Vec<u8>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Optional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Artifact {
    /// Create a new artifact
    pub fn new(artifact_type: ArtifactType, file_path: PathBuf, content: Vec<u8>) -> Self {
        let content_hash = Self::compute_hash(&content);
        let mime_type = Self::detect_mime_type(&file_path, &artifact_type);

        Self {
            id: Uuid::new_v4(),
            artifact_type,
            size_bytes: content.len() as u64,
            content_hash,
            mime_type,
            content,
            file_path,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Create artifact from file
    pub async fn from_file(artifact_type: ArtifactType, file_path: PathBuf) -> Result<Self> {
        let content = fs::read(&file_path).await?;
        Ok(Self::new(artifact_type, file_path, content))
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Compute SHA256 hash of content
    fn compute_hash(content: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(content);
        format!("{:x}", hasher.finalize())
    }

    /// Detect MIME type
    fn detect_mime_type(file_path: &Path, artifact_type: &ArtifactType) -> String {
        // Simple MIME type detection based on extension and artifact type
        if let Some(extension) = file_path.extension().and_then(|s| s.to_str()) {
            match extension.to_lowercase().as_str() {
                "json" => "application/json".to_string(),
                "bin" | "pt" | "pth" => "application/octet-stream".to_string(),
                "txt" => "text/plain".to_string(),
                "md" => "text/markdown".to_string(),
                "yaml" | "yml" => "application/x-yaml".to_string(),
                _ => "application/octet-stream".to_string(),
            }
        } else {
            match artifact_type {
                ArtifactType::Config
                | ArtifactType::Tokenizer
                | ArtifactType::Architecture
                | ArtifactType::Preprocessing
                | ArtifactType::Metrics => "application/json".to_string(),
                ArtifactType::Documentation => "text/markdown".to_string(),
                ArtifactType::Vocabulary => "text/plain".to_string(),
                _ => "application/octet-stream".to_string(),
            }
        }
    }

    /// Verify content integrity
    pub fn verify_integrity(&self) -> bool {
        Self::compute_hash(&self.content) == self.content_hash
    }

    /// Get file extension
    pub fn file_extension(&self) -> Option<&str> {
        self.file_path.extension()?.to_str()
    }
}

/// File system storage backend
pub struct FileSystemStorage {
    base_path: PathBuf,
    archive_path: PathBuf,
    metadata_cache: tokio::sync::RwLock<HashMap<Uuid, Artifact>>,
}

impl FileSystemStorage {
    /// Create a new filesystem storage backend
    pub fn new(base_path: PathBuf) -> Self {
        let archive_path = base_path.join("archive");
        Self {
            base_path,
            archive_path,
            metadata_cache: tokio::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Initialize storage directories
    pub async fn initialize(&self) -> Result<()> {
        fs::create_dir_all(&self.base_path).await?;
        fs::create_dir_all(&self.archive_path).await?;
        Ok(())
    }

    /// Get storage path for an artifact
    fn get_artifact_path(&self, artifact_id: Uuid) -> PathBuf {
        let id_str = artifact_id.to_string();
        let prefix = &id_str[0..2];
        self.base_path.join("artifacts").join(prefix).join(&id_str)
    }

    /// Get archive path for an artifact
    fn get_archive_path(&self, artifact_id: Uuid) -> PathBuf {
        let id_str = artifact_id.to_string();
        let prefix = &id_str[0..2];
        self.archive_path.join("artifacts").join(prefix).join(&id_str)
    }

    /// Store artifact metadata
    async fn store_metadata(&self, artifact: &Artifact) -> Result<()> {
        let metadata_path = self.get_artifact_path(artifact.id).with_extension("meta");
        if let Some(parent) = metadata_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        let metadata_json = serde_json::to_string_pretty(artifact)?;
        fs::write(metadata_path, metadata_json).await?;

        // Cache metadata
        self.metadata_cache.write().await.insert(artifact.id, artifact.clone());
        Ok(())
    }

    /// Load artifact metadata
    async fn load_metadata(&self, artifact_id: Uuid) -> Result<Option<Artifact>> {
        // Check cache first
        if let Some(artifact) = self.metadata_cache.read().await.get(&artifact_id) {
            return Ok(Some(artifact.clone()));
        }

        // Load from disk
        let metadata_path = self.get_artifact_path(artifact_id).with_extension("meta");
        if !metadata_path.exists() {
            return Ok(None);
        }

        let metadata_json = fs::read_to_string(metadata_path).await?;
        let mut artifact: Artifact = serde_json::from_str(&metadata_json)?;

        // Load content if needed
        let content_path = self.get_artifact_path(artifact_id).with_extension("bin");
        if content_path.exists() {
            artifact.content = fs::read(content_path).await?;
        }

        // Cache metadata
        self.metadata_cache.write().await.insert(artifact_id, artifact.clone());
        Ok(Some(artifact))
    }
}

#[async_trait]
impl ModelStorage for FileSystemStorage {
    async fn store_artifacts(&self, artifacts: &[Artifact]) -> Result<Vec<Uuid>> {
        let mut artifact_ids = Vec::new();

        for artifact in artifacts {
            // Store content
            let content_path = self.get_artifact_path(artifact.id).with_extension("bin");
            if let Some(parent) = content_path.parent() {
                fs::create_dir_all(parent).await?;
            }
            fs::write(&content_path, &artifact.content).await?;

            // Store metadata
            self.store_metadata(artifact).await?;

            artifact_ids.push(artifact.id);
            tracing::debug!("Stored artifact {} at {:?}", artifact.id, content_path);
        }

        Ok(artifact_ids)
    }

    async fn get_artifact(&self, artifact_id: Uuid) -> Result<Option<Artifact>> {
        self.load_metadata(artifact_id).await
    }

    async fn delete_artifacts(&self, artifact_ids: &[Uuid]) -> Result<()> {
        for &artifact_id in artifact_ids {
            let content_path = self.get_artifact_path(artifact_id).with_extension("bin");
            let metadata_path = self.get_artifact_path(artifact_id).with_extension("meta");

            if content_path.exists() {
                fs::remove_file(content_path).await?;
            }
            if metadata_path.exists() {
                fs::remove_file(metadata_path).await?;
            }

            // Remove from cache
            self.metadata_cache.write().await.remove(&artifact_id);
            tracing::debug!("Deleted artifact {}", artifact_id);
        }
        Ok(())
    }

    async fn archive_version(&self, version_id: Uuid) -> Result<()> {
        // Move artifacts to archive directory
        let artifacts = self.list_artifacts(version_id).await?;

        for artifact in artifacts {
            let src_content = self.get_artifact_path(artifact.id).with_extension("bin");
            let src_metadata = self.get_artifact_path(artifact.id).with_extension("meta");

            let dst_content = self.get_archive_path(artifact.id).with_extension("bin");
            let dst_metadata = self.get_archive_path(artifact.id).with_extension("meta");

            if let Some(parent) = dst_content.parent() {
                fs::create_dir_all(parent).await?;
            }

            if src_content.exists() {
                fs::rename(src_content, dst_content).await?;
            }
            if src_metadata.exists() {
                fs::rename(src_metadata, dst_metadata).await?;
            }

            // Remove from cache
            self.metadata_cache.write().await.remove(&artifact.id);
        }

        tracing::info!("Archived version {}", version_id);
        Ok(())
    }

    async fn delete_version(&self, version_id: Uuid) -> Result<()> {
        let artifacts = self.list_artifacts(version_id).await?;
        let artifact_ids: Vec<Uuid> = artifacts.iter().map(|a| a.id).collect();
        self.delete_artifacts(&artifact_ids).await?;

        tracing::info!("Deleted version {}", version_id);
        Ok(())
    }

    async fn list_artifacts(&self, _version_id: Uuid) -> Result<Vec<Artifact>> {
        // This would normally query a database or index
        // For now, return artifacts from cache
        let cache = self.metadata_cache.read().await;
        Ok(cache.values().cloned().collect())
    }
}

/// In-memory storage backend for testing
pub struct InMemoryStorage {
    artifacts: tokio::sync::RwLock<HashMap<Uuid, Artifact>>,
    archived: tokio::sync::RwLock<HashMap<Uuid, Artifact>>,
}

impl InMemoryStorage {
    pub fn new() -> Self {
        Self {
            artifacts: tokio::sync::RwLock::new(HashMap::new()),
            archived: tokio::sync::RwLock::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl ModelStorage for InMemoryStorage {
    async fn store_artifacts(&self, artifacts: &[Artifact]) -> Result<Vec<Uuid>> {
        let mut artifact_ids = Vec::new();
        let mut storage = self.artifacts.write().await;

        for artifact in artifacts {
            storage.insert(artifact.id, artifact.clone());
            artifact_ids.push(artifact.id);
        }

        Ok(artifact_ids)
    }

    async fn get_artifact(&self, artifact_id: Uuid) -> Result<Option<Artifact>> {
        let storage = self.artifacts.read().await;
        Ok(storage.get(&artifact_id).cloned())
    }

    async fn delete_artifacts(&self, artifact_ids: &[Uuid]) -> Result<()> {
        let mut storage = self.artifacts.write().await;
        for &artifact_id in artifact_ids {
            storage.remove(&artifact_id);
        }
        Ok(())
    }

    async fn archive_version(&self, version_id: Uuid) -> Result<()> {
        let artifacts = self.list_artifacts(version_id).await?;

        let mut storage = self.artifacts.write().await;
        let mut archived = self.archived.write().await;

        for artifact in artifacts {
            if let Some(artifact) = storage.remove(&artifact.id) {
                archived.insert(artifact.id, artifact);
            }
        }

        Ok(())
    }

    async fn delete_version(&self, version_id: Uuid) -> Result<()> {
        let artifacts = self.list_artifacts(version_id).await?;
        let artifact_ids: Vec<Uuid> = artifacts.iter().map(|a| a.id).collect();
        self.delete_artifacts(&artifact_ids).await
    }

    async fn list_artifacts(&self, _version_id: Uuid) -> Result<Vec<Artifact>> {
        let storage = self.artifacts.read().await;
        Ok(storage.values().cloned().collect())
    }
}

impl Default for InMemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_artifact_creation() {
        let content = b"test model data".to_vec();
        let artifact = Artifact::new(
            ArtifactType::Model,
            PathBuf::from("model.bin"),
            content.clone(),
        );

        assert_eq!(artifact.artifact_type, ArtifactType::Model);
        assert_eq!(artifact.content, content);
        assert_eq!(artifact.size_bytes, content.len() as u64);
        assert!(!artifact.content_hash.is_empty());
        assert!(artifact.verify_integrity());
    }

    #[tokio::test]
    async fn test_filesystem_storage() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileSystemStorage::new(temp_dir.path().to_path_buf());
        storage.initialize().await.unwrap();

        let artifact = Artifact::new(
            ArtifactType::Model,
            PathBuf::from("test_model.bin"),
            b"test content".to_vec(),
        );

        // Store artifact
        let ids = storage.store_artifacts(&[artifact.clone()]).await.unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], artifact.id);

        // Retrieve artifact
        let retrieved = storage.get_artifact(artifact.id).await.unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.content, artifact.content);
        assert_eq!(retrieved.content_hash, artifact.content_hash);

        // Delete artifact
        storage.delete_artifacts(&[artifact.id]).await.unwrap();
        let deleted = storage.get_artifact(artifact.id).await.unwrap();
        assert!(deleted.is_none());
    }

    #[tokio::test]
    async fn test_inmemory_storage() {
        let storage = InMemoryStorage::new();

        let artifact = Artifact::new(
            ArtifactType::Config,
            PathBuf::from("config.json"),
            b"{}".to_vec(),
        );

        // Store and retrieve
        let ids = storage.store_artifacts(&[artifact.clone()]).await.unwrap();
        assert_eq!(ids[0], artifact.id);

        let retrieved = storage.get_artifact(artifact.id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, artifact.content);
    }

    #[test]
    fn test_artifact_types() {
        assert_eq!(ArtifactType::Model.default_extension(), "bin");
        assert_eq!(ArtifactType::Config.default_extension(), "json");
        assert!(ArtifactType::Model.is_required_for_deployment());
        assert!(ArtifactType::Config.is_required_for_deployment());
        assert!(!ArtifactType::Documentation.is_required_for_deployment());
    }

    #[test]
    fn test_mime_type_detection() {
        let json_artifact = Artifact::new(
            ArtifactType::Config,
            PathBuf::from("config.json"),
            b"{}".to_vec(),
        );
        assert_eq!(json_artifact.mime_type, "application/json");

        let bin_artifact = Artifact::new(
            ArtifactType::Model,
            PathBuf::from("model.bin"),
            b"binary data".to_vec(),
        );
        assert_eq!(bin_artifact.mime_type, "application/octet-stream");
    }
}
