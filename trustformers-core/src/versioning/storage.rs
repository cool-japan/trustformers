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
    /// Optional metadata.
    ///
    /// The key `version_id` associates the artifact with a model version and is
    /// what [`ModelStorage::list_artifacts`] filters on. Set it with
    /// [`Artifact::assign_version`]; an artifact without it belongs to no
    /// version and is never returned by a version query (and therefore never
    /// deleted by [`ModelStorage::delete_version`]).
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Metadata key associating an artifact with a model version.
pub const ARTIFACT_VERSION_KEY: &str = "version_id";

impl Artifact {
    /// Associate this artifact with a model version.
    pub fn assign_version(&mut self, version_id: Uuid) {
        self.metadata.insert(
            ARTIFACT_VERSION_KEY.to_string(),
            serde_json::Value::String(version_id.to_string()),
        );
    }

    /// The model version this artifact belongs to, if any.
    pub fn version_id(&self) -> Option<Uuid> {
        self.metadata
            .get(ARTIFACT_VERSION_KEY)
            .and_then(|value| value.as_str())
            .and_then(|text| Uuid::parse_str(text).ok())
    }

    /// Whether this artifact belongs to `version_id`.
    pub fn belongs_to(&self, version_id: Uuid) -> bool {
        self.version_id() == Some(version_id)
    }
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
        hex::encode(hasher.finalize())
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

    /// Read every `.meta` file under the artifact directory.
    ///
    /// Content bytes are deliberately not loaded: callers of `list_artifacts`
    /// want the index, not every blob.
    async fn scan_stored_metadata(&self) -> Result<Vec<Artifact>> {
        let root = self.base_path.join("artifacts");
        let mut artifacts = Vec::new();

        let mut prefixes = match fs::read_dir(&root).await {
            Ok(entries) => entries,
            // No artifact directory yet: nothing is stored.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(artifacts),
            Err(error) => return Err(error.into()),
        };

        while let Some(prefix_entry) = prefixes.next_entry().await? {
            if !prefix_entry.file_type().await?.is_dir() {
                continue;
            }
            let mut files = fs::read_dir(prefix_entry.path()).await?;
            while let Some(file_entry) = files.next_entry().await? {
                let path = file_entry.path();
                if path.extension().and_then(|extension| extension.to_str()) != Some("meta") {
                    continue;
                }
                let contents = fs::read_to_string(&path).await?;
                match serde_json::from_str::<Artifact>(&contents) {
                    Ok(artifact) => artifacts.push(artifact),
                    Err(error) => {
                        tracing::warn!(path = %path.display(), "skipping unreadable artifact metadata: {error}");
                    },
                }
            }
        }

        Ok(artifacts)
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

    /// List the artifacts belonging to `version_id`.
    ///
    /// The on-disk metadata directory is scanned, not just the warm cache, so a
    /// cold process sees the same set a warm one does. Artifacts that carry no
    /// `version_id` metadata belong to no version and are never returned — this
    /// is what stops [`Self::delete_version`] from deleting unrelated
    /// artifacts, which the previous cache dump did.
    async fn list_artifacts(&self, version_id: Uuid) -> Result<Vec<Artifact>> {
        let mut found: HashMap<Uuid, Artifact> = HashMap::new();

        for artifact in self.metadata_cache.read().await.values() {
            if artifact.belongs_to(version_id) {
                found.insert(artifact.id, artifact.clone());
            }
        }

        for artifact in self.scan_stored_metadata().await? {
            if artifact.belongs_to(version_id) {
                found.entry(artifact.id).or_insert(artifact);
            }
        }

        let mut artifacts: Vec<Artifact> = found.into_values().collect();
        artifacts.sort_by_key(|artifact| artifact.created_at);
        Ok(artifacts)
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

    /// List the artifacts belonging to `version_id`.
    ///
    /// Filters on the artifact's `version_id` metadata, exactly like the
    /// filesystem backend; returning every stored artifact regardless of
    /// version would make `delete_version` delete everything.
    async fn list_artifacts(&self, version_id: Uuid) -> Result<Vec<Artifact>> {
        let storage = self.artifacts.read().await;
        let mut artifacts: Vec<Artifact> = storage
            .values()
            .filter(|artifact| artifact.belongs_to(version_id))
            .cloned()
            .collect();
        artifacts.sort_by_key(|artifact| artifact.created_at);
        Ok(artifacts)
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

    /// Regression test: `list_artifacts` ignored `version_id` and returned the
    /// whole cache, so `delete_version` deleted every artifact it could see.
    #[tokio::test]
    async fn test_list_artifacts_filters_by_version() {
        let storage = InMemoryStorage::new();
        let version_a = Uuid::new_v4();
        let version_b = Uuid::new_v4();

        let mut artifact_a = sample_artifact("a");
        artifact_a.assign_version(version_a);
        let mut artifact_b = sample_artifact("b");
        artifact_b.assign_version(version_b);
        // An artifact that belongs to no version at all.
        let orphan = sample_artifact("orphan");

        storage
            .store_artifacts(&[artifact_a.clone(), artifact_b.clone(), orphan.clone()])
            .await
            .expect("store failed");

        let listed = storage.list_artifacts(version_a).await.expect("list failed");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, artifact_a.id);

        // Deleting version A must leave B and the orphan alone.
        storage.delete_version(version_a).await.expect("delete failed");
        assert!(storage.get_artifact(artifact_a.id).await.expect("get").is_none());
        assert!(storage.get_artifact(artifact_b.id).await.expect("get").is_some());
        assert!(storage.get_artifact(orphan.id).await.expect("get").is_some());
    }

    /// The filesystem backend must see artifacts written by another process,
    /// not only whatever happens to be in its own cache.
    #[tokio::test]
    async fn test_filesystem_list_artifacts_reads_from_disk() {
        let root = std::env::temp_dir().join(format!(
            "trustformers_storage_{}_{}",
            std::process::id(),
            Uuid::new_v4()
        ));
        let version = Uuid::new_v4();

        {
            let storage = FileSystemStorage::new(root.clone());
            storage.initialize().await.expect("initialize failed");
            let mut artifact = sample_artifact("on-disk");
            artifact.assign_version(version);
            storage.store_artifacts(&[artifact]).await.expect("store failed");
        }

        // A brand new backend has a cold cache.
        let cold = FileSystemStorage::new(root.clone());
        let listed = cold.list_artifacts(version).await.expect("list failed");
        assert_eq!(
            listed.len(),
            1,
            "a cold cache must still find stored artifacts"
        );
        assert!(cold.list_artifacts(Uuid::new_v4()).await.expect("list failed").is_empty());

        tokio::fs::remove_dir_all(&root).await.ok();
    }

    fn sample_artifact(name: &str) -> Artifact {
        Artifact {
            id: Uuid::new_v4(),
            artifact_type: ArtifactType::Config,
            file_path: PathBuf::from(name),
            size_bytes: 3,
            content_hash: "deadbeef".to_string(),
            mime_type: "application/octet-stream".to_string(),
            content: vec![1, 2, 3],
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }
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
        let temp_dir = TempDir::new().expect("temp file creation failed");
        let storage = FileSystemStorage::new(temp_dir.path().to_path_buf());
        storage.initialize().await.expect("async operation failed");

        let artifact = Artifact::new(
            ArtifactType::Model,
            PathBuf::from("test_model.bin"),
            b"test content".to_vec(),
        );

        // Store artifact
        let ids = storage
            .store_artifacts(std::slice::from_ref(&artifact))
            .await
            .expect("async operation failed");
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], artifact.id);

        // Retrieve artifact
        let retrieved = storage.get_artifact(artifact.id).await.expect("async operation failed");
        assert!(retrieved.is_some());
        let retrieved = retrieved.expect("operation failed in test");
        assert_eq!(retrieved.content, artifact.content);
        assert_eq!(retrieved.content_hash, artifact.content_hash);

        // Delete artifact
        storage.delete_artifacts(&[artifact.id]).await.expect("async operation failed");
        let deleted = storage.get_artifact(artifact.id).await.expect("async operation failed");
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
        let ids = storage
            .store_artifacts(std::slice::from_ref(&artifact))
            .await
            .expect("async operation failed");
        assert_eq!(ids[0], artifact.id);

        let retrieved = storage.get_artifact(artifact.id).await.expect("async operation failed");
        assert!(retrieved.is_some());
        assert_eq!(
            retrieved.expect("operation failed in test").content,
            artifact.content
        );
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
