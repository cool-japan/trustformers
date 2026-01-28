//! Model Registry
//!
//! Manages model metadata, versioning, and lifecycle tracking.

use crate::model_management::{
    DeploymentStrategy, ModelError, ModelMetadata, ModelMetrics, ModelResult, ModelStatus,
};
use chrono::Utc;
use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
    sync::{Arc, RwLock},
};
use tokio::fs;
use uuid::Uuid;

/// Thread-safe model registry
pub struct ModelRegistry {
    /// Models indexed by ID
    models: Arc<RwLock<HashMap<String, ModelMetadata>>>,

    /// Models indexed by name and version
    by_name_version: Arc<RwLock<HashMap<String, BTreeMap<String, String>>>>,

    /// Active model mappings (name -> model_id)
    active_models: Arc<RwLock<HashMap<String, String>>>,

    /// Metadata storage directory
    metadata_dir: String,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new(metadata_dir: String) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            by_name_version: Arc::new(RwLock::new(HashMap::new())),
            active_models: Arc::new(RwLock::new(HashMap::new())),
            metadata_dir,
        }
    }

    /// Initialize registry by loading existing metadata
    pub async fn initialize(&self) -> ModelResult<()> {
        // Create metadata directory if it doesn't exist
        fs::create_dir_all(&self.metadata_dir).await?;

        // Load existing metadata files
        let mut dir = fs::read_dir(&self.metadata_dir).await?;
        while let Some(entry) = dir.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Ok(content) = fs::read_to_string(&path).await {
                    if let Ok(metadata) = serde_json::from_str::<ModelMetadata>(&content) {
                        self.add_model_to_indexes(metadata)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Register a new model
    pub async fn register_model(
        &self,
        name: String,
        version: String,
        path: String,
        config: String,
        deployment_strategy: DeploymentStrategy,
        tags: HashMap<String, String>,
    ) -> ModelResult<String> {
        let model_id = Uuid::new_v4().to_string();
        let now = Utc::now();

        // Check if model with same name and version exists
        if self.get_model_by_name_version(&name, &version).is_some() {
            return Err(ModelError::ModelAlreadyExists {
                id: format!("{}:{}", name, version),
            });
        }

        let metadata = ModelMetadata {
            id: model_id.clone(),
            name: name.clone(),
            version: version.clone(),
            path,
            config,
            created_at: now,
            updated_at: now,
            status: ModelStatus::Loading,
            deployment_strategy,
            tags,
            metrics: ModelMetrics::default(),
        };

        // Add to indexes
        self.add_model_to_indexes(metadata.clone())?;

        // Persist metadata
        self.persist_metadata(&metadata).await?;

        Ok(model_id)
    }

    /// Update model metadata
    pub async fn update_model(&self, model_id: &str, metadata: ModelMetadata) -> ModelResult<()> {
        {
            let mut models = self.models.write().map_err(|e| ModelError::InvalidConfig {
                error: format!("Lock poisoned: {}", e),
            })?;
            if !models.contains_key(model_id) {
                return Err(ModelError::ModelNotFound {
                    id: model_id.to_string(),
                });
            }
            models.insert(model_id.to_string(), metadata.clone());
        }

        // Update version index
        {
            let mut by_name_version =
                self.by_name_version.write().map_err(|e| ModelError::InvalidConfig {
                    error: format!("Lock poisoned: {}", e),
                })?;
            by_name_version
                .entry(metadata.name.clone())
                .or_default()
                .insert(metadata.version.clone(), model_id.to_string());
        }

        // Persist updated metadata
        self.persist_metadata(&metadata).await?;

        Ok(())
    }

    /// Get model metadata by ID
    pub fn get_model(&self, model_id: &str) -> Option<ModelMetadata> {
        self.models.read().ok()?.get(model_id).cloned()
    }

    /// Get model by name and version
    pub fn get_model_by_name_version(&self, name: &str, version: &str) -> Option<ModelMetadata> {
        let by_name_version = self.by_name_version.read().ok()?;
        if let Some(versions) = by_name_version.get(name) {
            if let Some(model_id) = versions.get(version) {
                return self.get_model(model_id);
            }
        }
        None
    }

    /// Get latest version of a model by name
    pub fn get_latest_model(&self, name: &str) -> Option<ModelMetadata> {
        let by_name_version = self.by_name_version.read().ok()?;
        if let Some(versions) = by_name_version.get(name) {
            if let Some((_, model_id)) = versions.iter().last() {
                return self.get_model(model_id);
            }
        }
        None
    }

    /// List all models
    pub fn list_models(&self) -> Vec<ModelMetadata> {
        self.models
            .read()
            .ok()
            .map(|m| m.values().cloned().collect())
            .unwrap_or_default()
    }

    /// List models by name
    pub fn list_models_by_name(&self, name: &str) -> Vec<ModelMetadata> {
        let by_name_version = match self.by_name_version.read() {
            Ok(guard) => guard,
            Err(_) => return Vec::new(),
        };
        if let Some(versions) = by_name_version.get(name) {
            return versions.values().filter_map(|model_id| self.get_model(model_id)).collect();
        }
        Vec::new()
    }

    /// Update model status
    pub async fn update_status(&self, model_id: &str, status: ModelStatus) -> ModelResult<()> {
        if let Some(mut metadata) = self.get_model(model_id) {
            metadata.status = status;
            metadata.updated_at = Utc::now();
            self.update_model(model_id, metadata).await?;
        } else {
            return Err(ModelError::ModelNotFound {
                id: model_id.to_string(),
            });
        }
        Ok(())
    }

    /// Update model metrics
    pub async fn update_metrics(&self, model_id: &str, metrics: ModelMetrics) -> ModelResult<()> {
        if let Some(mut metadata) = self.get_model(model_id) {
            metadata.metrics = metrics;
            metadata.updated_at = Utc::now();
            self.update_model(model_id, metadata).await?;
        } else {
            return Err(ModelError::ModelNotFound {
                id: model_id.to_string(),
            });
        }
        Ok(())
    }

    /// Set active model for a name
    pub fn set_active_model(&self, name: &str, model_id: &str) -> ModelResult<()> {
        // Verify model exists
        if self.get_model(model_id).is_none() {
            return Err(ModelError::ModelNotFound {
                id: model_id.to_string(),
            });
        }

        let mut active_models =
            self.active_models.write().map_err(|e| ModelError::InvalidConfig {
                error: format!("Lock poisoned: {}", e),
            })?;
        active_models.insert(name.to_string(), model_id.to_string());

        Ok(())
    }

    /// Get active model for a name
    pub fn get_active_model(&self, name: &str) -> Option<ModelMetadata> {
        let active_models = self.active_models.read().ok()?;
        if let Some(model_id) = active_models.get(name) {
            return self.get_model(model_id);
        }
        None
    }

    /// Remove model from registry
    pub async fn remove_model(&self, model_id: &str) -> ModelResult<()> {
        let metadata = {
            let mut models = self.models.write().map_err(|e| ModelError::InvalidConfig {
                error: format!("Lock poisoned: {}", e),
            })?;
            models.remove(model_id)
        };

        if let Some(metadata) = metadata {
            // Remove from version index
            {
                let mut by_name_version =
                    self.by_name_version.write().map_err(|e| ModelError::InvalidConfig {
                        error: format!("Lock poisoned: {}", e),
                    })?;
                if let Some(versions) = by_name_version.get_mut(&metadata.name) {
                    versions.remove(&metadata.version);
                    if versions.is_empty() {
                        by_name_version.remove(&metadata.name);
                    }
                }
            }

            // Remove from active models if it was active
            {
                let mut active_models =
                    self.active_models.write().map_err(|e| ModelError::InvalidConfig {
                        error: format!("Lock poisoned: {}", e),
                    })?;
                active_models.retain(|_, id| id != model_id);
            }

            // Remove metadata file
            let metadata_path = Path::new(&self.metadata_dir).join(format!("{}.json", model_id));
            if metadata_path.exists() {
                fs::remove_file(metadata_path).await?;
            }

            Ok(())
        } else {
            Err(ModelError::ModelNotFound {
                id: model_id.to_string(),
            })
        }
    }

    /// Cleanup old versions
    pub async fn cleanup_old_versions(
        &self,
        max_versions_per_model: usize,
    ) -> ModelResult<Vec<String>> {
        let mut removed_models = Vec::new();
        let models_by_name = {
            let by_name_version =
                self.by_name_version.read().map_err(|e| ModelError::InvalidConfig {
                    error: format!("Lock poisoned: {}", e),
                })?;
            by_name_version.clone()
        };

        for (_name, versions) in models_by_name {
            if versions.len() > max_versions_per_model {
                // Keep only the latest versions
                let mut version_list: Vec<_> = versions.into_iter().collect();
                version_list.sort_by(|a, b| a.0.cmp(&b.0)); // Sort by version string

                // Calculate how many to remove
                let remove_count = version_list.len() - max_versions_per_model;

                // Remove oldest versions
                for (_, model_id) in version_list.into_iter().take(remove_count) {
                    if let Ok(()) = self.remove_model(&model_id).await {
                        removed_models.push(model_id);
                    }
                }
            }
        }

        Ok(removed_models)
    }

    /// Get models by status
    pub fn get_models_by_status(&self, status: ModelStatus) -> Vec<ModelMetadata> {
        self.models
            .read()
            .ok()
            .map(|m| m.values().filter(|v| v.status == status).cloned().collect())
            .unwrap_or_default()
    }

    /// Get models by tags
    pub fn get_models_by_tags(&self, tags: &HashMap<String, String>) -> Vec<ModelMetadata> {
        self.models
            .read()
            .ok()
            .map(|m| {
                m.values()
                    .filter(|v| tags.iter().all(|(key, value)| v.tags.get(key) == Some(value)))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Add model to internal indexes
    fn add_model_to_indexes(&self, metadata: ModelMetadata) -> ModelResult<()> {
        {
            let mut models = self.models.write().map_err(|e| ModelError::InvalidConfig {
                error: format!("Lock poisoned: {}", e),
            })?;
            models.insert(metadata.id.clone(), metadata.clone());
        }

        {
            let mut by_name_version =
                self.by_name_version.write().map_err(|e| ModelError::InvalidConfig {
                    error: format!("Lock poisoned: {}", e),
                })?;
            by_name_version
                .entry(metadata.name.clone())
                .or_default()
                .insert(metadata.version.clone(), metadata.id.clone());
        }

        Ok(())
    }

    /// Persist metadata to disk
    async fn persist_metadata(&self, metadata: &ModelMetadata) -> ModelResult<()> {
        let metadata_path = Path::new(&self.metadata_dir).join(format!("{}.json", metadata.id));
        let content = serde_json::to_string_pretty(metadata)?;
        fs::write(metadata_path, content).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_model_registry() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let registry = ModelRegistry::new(temp_dir.path().to_string_lossy().to_string());

        registry.initialize().await.expect("Failed to initialize");

        // Test model registration
        let model_id = registry
            .register_model(
                "test-model".to_string(),
                "1.0.0".to_string(),
                "/path/to/model".to_string(),
                "{}".to_string(),
                DeploymentStrategy::Replace,
                HashMap::new(),
            )
            .await
            .expect("Failed to register model");

        // Test model retrieval
        let metadata = registry.get_model(&model_id).expect("Failed to get model");
        assert_eq!(metadata.name, "test-model");
        assert_eq!(metadata.version, "1.0.0");

        // Test version lookup
        let metadata2 = registry
            .get_model_by_name_version("test-model", "1.0.0")
            .expect("Failed to get model by name/version");
        assert_eq!(metadata.id, metadata2.id);

        // Test active model setting
        registry
            .set_active_model("test-model", &model_id)
            .expect("Failed to set active model");
        let active = registry.get_active_model("test-model").expect("Failed to get active model");
        assert_eq!(active.id, model_id);
    }
}
