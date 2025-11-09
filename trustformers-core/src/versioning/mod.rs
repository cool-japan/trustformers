//! Model Versioning System for TrustformeRS
//!
//! This module provides comprehensive model versioning capabilities including:
//! - Model version management with metadata
//! - Artifact storage and retrieval
//! - Version lifecycle management
//! - Rollback and promotion strategies
//! - Integration with A/B testing

mod deployment;
mod integration;
mod lifecycle;
mod metadata;
mod registry;
mod storage;

pub use deployment::{
    ActiveDeployment, DeploymentConfig, DeploymentEvent, DeploymentEventType, DeploymentManager,
    DeploymentStatistics, DeploymentStatus, DeploymentStrategy, Environment, HealthStatus,
};
pub use integration::{
    ModelRoutingResult, PromotionResult, VersionExperimentConfig, VersionExperimentResult,
    VersionMetricType, VersionedABTestManager, VersionedExperiment, VersionedExperimentStatus,
};
pub use lifecycle::{
    LifecycleEvent, LifecyclePolicies, LifecycleStatistics, VersionLifecycle, VersionStatus,
    VersionTransition,
};
pub use metadata::{ModelMetadata, ModelSource, ModelTag, VersionedModel};
pub use registry::{
    DateRange, ModelRegistry, RegistryStatistics, SortBy, SortOrder, TagMatchMode, VersionFilter,
    VersionQuery,
};
pub use storage::{Artifact, ArtifactType, FileSystemStorage, InMemoryStorage, ModelStorage};

use anyhow::Result;
use std::sync::Arc;
use uuid::Uuid;

/// Main model versioning manager
pub struct ModelVersionManager {
    registry: Arc<ModelRegistry>,
    storage: Arc<dyn ModelStorage>,
    deployment_manager: Arc<DeploymentManager>,
    lifecycle: Arc<VersionLifecycle>,
}

impl ModelVersionManager {
    /// Create a new model version manager
    pub fn new(storage: Arc<dyn ModelStorage>) -> Self {
        Self {
            registry: Arc::new(ModelRegistry::new()),
            storage: storage.clone(),
            deployment_manager: Arc::new(DeploymentManager::new(storage)),
            lifecycle: Arc::new(VersionLifecycle::new()),
        }
    }

    /// Register a new model version
    pub async fn register_version(
        &self,
        model_name: &str,
        version: &str,
        metadata: ModelMetadata,
        artifacts: Vec<Artifact>,
    ) -> Result<Uuid> {
        // Store artifacts
        let artifact_ids = self.storage.store_artifacts(&artifacts).await?;

        // Create versioned model
        let versioned_model = VersionedModel::new(
            model_name.to_string(),
            version.to_string(),
            metadata,
            artifact_ids,
        );

        // Register in registry
        let version_id = self.registry.register(versioned_model).await?;

        // Initialize lifecycle
        self.lifecycle.initialize_version(version_id).await?;

        tracing::info!(
            "Registered model version: {}:{} ({})",
            model_name,
            version,
            version_id
        );
        Ok(version_id)
    }

    /// Get a specific model version
    pub async fn get_version(&self, version_id: Uuid) -> Result<Option<VersionedModel>> {
        self.registry.get_version(version_id).await
    }

    /// Get model version by name and version
    pub async fn get_version_by_name(
        &self,
        model_name: &str,
        version: &str,
    ) -> Result<Option<VersionedModel>> {
        self.registry.get_version_by_name(model_name, version).await
    }

    /// List all versions for a model
    pub async fn list_versions(&self, model_name: &str) -> Result<Vec<VersionedModel>> {
        self.registry.list_versions(model_name).await
    }

    /// Query versions with filters
    pub async fn query_versions(&self, query: VersionQuery) -> Result<Vec<VersionedModel>> {
        self.registry.query_versions(query).await
    }

    /// Promote a version to production
    pub async fn promote_to_production(&self, version_id: Uuid) -> Result<()> {
        // Check current status
        let current_status = self.lifecycle.get_status(version_id).await?;
        if current_status != VersionStatus::Staging {
            anyhow::bail!("Can only promote versions from staging to production");
        }

        // Transition to production
        self.lifecycle.transition(version_id, VersionTransition::Promote).await?;

        // Deploy to production
        let version = self
            .registry
            .get_version(version_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Version not found"))?;

        self.deployment_manager.deploy_to_production(version_id, &version).await?;

        tracing::info!("Promoted version {} to production", version_id);
        Ok(())
    }

    /// Rollback to a previous version
    pub async fn rollback_to_version(&self, model_name: &str, target_version: &str) -> Result<()> {
        let version =
            self.get_version_by_name(model_name, target_version).await?.ok_or_else(|| {
                anyhow::anyhow!("Version not found: {}:{}", model_name, target_version)
            })?;

        // Check if target version is deployable
        let status = self.lifecycle.get_status(version.id()).await?;
        if status != VersionStatus::Production && status != VersionStatus::Staging {
            anyhow::bail!("Can only rollback to production or staging versions");
        }

        // Perform rollback
        self.deployment_manager.rollback(model_name, version.id()).await?;

        tracing::info!("Rolled back {} to version {}", model_name, target_version);
        Ok(())
    }

    /// Archive old versions
    pub async fn archive_version(&self, version_id: Uuid) -> Result<()> {
        // Check if version can be archived
        let status = self.lifecycle.get_status(version_id).await?;
        if status == VersionStatus::Production {
            anyhow::bail!("Cannot archive production version");
        }

        // Transition to archived
        self.lifecycle.transition(version_id, VersionTransition::Archive).await?;

        // Move artifacts to archive storage
        self.storage.archive_version(version_id).await?;

        tracing::info!("Archived version {}", version_id);
        Ok(())
    }

    /// Delete a version (permanent)
    pub async fn delete_version(&self, version_id: Uuid) -> Result<()> {
        // Check if version can be deleted
        let status = self.lifecycle.get_status(version_id).await?;
        if status == VersionStatus::Production {
            anyhow::bail!("Cannot delete production version");
        }

        // Remove from storage
        self.storage.delete_version(version_id).await?;

        // Remove from registry
        self.registry.remove_version(version_id).await?;

        // Clean up lifecycle
        self.lifecycle.cleanup_version(version_id).await?;

        tracing::info!("Deleted version {}", version_id);
        Ok(())
    }

    /// Get version statistics
    pub async fn get_version_stats(&self, model_name: &str) -> Result<VersionStats> {
        let versions = self.list_versions(model_name).await?;

        let mut stats = VersionStats {
            model_name: model_name.to_string(),
            total_versions: versions.len(),
            production_versions: 0,
            staging_versions: 0,
            development_versions: 0,
            archived_versions: 0,
            latest_version: None,
            oldest_version: None,
        };

        if !versions.is_empty() {
            // Find latest and oldest
            stats.latest_version = versions
                .iter()
                .max_by_key(|v| v.metadata().created_at)
                .map(|v| v.version().to_string());

            stats.oldest_version = versions
                .iter()
                .min_by_key(|v| v.metadata().created_at)
                .map(|v| v.version().to_string());

            // Count by status
            for version in &versions {
                let status = self.lifecycle.get_status(version.id()).await?;
                match status {
                    VersionStatus::Production => stats.production_versions += 1,
                    VersionStatus::Staging => stats.staging_versions += 1,
                    VersionStatus::Development => stats.development_versions += 1,
                    VersionStatus::Archived => stats.archived_versions += 1,
                    _ => {},
                }
            }
        }

        Ok(stats)
    }

    /// Get registry reference
    pub fn registry(&self) -> Arc<ModelRegistry> {
        self.registry.clone()
    }

    /// Get storage reference
    pub fn storage(&self) -> Arc<dyn ModelStorage> {
        self.storage.clone()
    }

    /// Get deployment manager reference
    pub fn deployment_manager(&self) -> Arc<DeploymentManager> {
        self.deployment_manager.clone()
    }

    /// Get lifecycle manager reference
    pub fn lifecycle(&self) -> Arc<VersionLifecycle> {
        self.lifecycle.clone()
    }
}

/// Version statistics for a model
#[derive(Debug, Clone)]
pub struct VersionStats {
    pub model_name: String,
    pub total_versions: usize,
    pub production_versions: usize,
    pub staging_versions: usize,
    pub development_versions: usize,
    pub archived_versions: usize,
    pub latest_version: Option<String>,
    pub oldest_version: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // Mock storage for testing
    struct MockStorage;

    #[async_trait::async_trait]
    impl ModelStorage for MockStorage {
        async fn store_artifacts(&self, _artifacts: &[Artifact]) -> Result<Vec<Uuid>> {
            Ok(vec![Uuid::new_v4()])
        }

        async fn get_artifact(&self, _artifact_id: Uuid) -> Result<Option<Artifact>> {
            Ok(None)
        }

        async fn delete_artifacts(&self, _artifact_ids: &[Uuid]) -> Result<()> {
            Ok(())
        }

        async fn archive_version(&self, _version_id: Uuid) -> Result<()> {
            Ok(())
        }

        async fn delete_version(&self, _version_id: Uuid) -> Result<()> {
            Ok(())
        }

        async fn list_artifacts(&self, _version_id: Uuid) -> Result<Vec<Artifact>> {
            Ok(vec![])
        }
    }

    #[tokio::test]
    async fn test_version_registration() {
        let storage = Arc::new(MockStorage);
        let manager = ModelVersionManager::new(storage);

        let metadata = ModelMetadata::builder()
            .description("Test model".to_string())
            .created_by("test_user".to_string())
            .model_type("transformer".to_string())
            .build();

        let artifacts = vec![Artifact::new(
            ArtifactType::Model,
            PathBuf::from("model.bin"),
            vec![1, 2, 3],
        )];

        let version_id = manager
            .register_version("test_model", "1.0.0", metadata, artifacts)
            .await
            .unwrap();
        assert!(!version_id.is_nil());

        let retrieved = manager.get_version(version_id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().version(), "1.0.0");
    }

    #[tokio::test]
    async fn test_version_lifecycle() {
        let storage = Arc::new(MockStorage);
        let manager = ModelVersionManager::new(storage);

        let metadata = ModelMetadata::builder()
            .description("Test model".to_string())
            .created_by("test_user".to_string())
            .model_type("transformer".to_string())
            .build();

        let version_id =
            manager.register_version("test_model", "1.0.0", metadata, vec![]).await.unwrap();

        // Should start in development
        let status = manager.lifecycle.get_status(version_id).await.unwrap();
        assert_eq!(status, VersionStatus::Development);

        // Move to staging
        manager
            .lifecycle
            .transition(version_id, VersionTransition::ToStaging)
            .await
            .unwrap();
        let status = manager.lifecycle.get_status(version_id).await.unwrap();
        assert_eq!(status, VersionStatus::Staging);
    }
}
