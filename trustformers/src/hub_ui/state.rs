//! Server-wide UI state and configuration: `HubUiState`, `HubUiConfig`, `ThemeConfig`, `FeatureFlags`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::TrustformersError;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use super::repository::ModelRepository;
use super::types::{ModelVersion, RepositoryMetadata};

/// Feature flags
#[derive(Debug, Clone)]
pub struct FeatureFlags {
    /// Enable model comparison
    pub enable_comparison: bool,
    /// Enable version branching
    pub enable_branching: bool,
    /// Enable performance tracking
    pub enable_performance_tracking: bool,
    /// Enable collaborative features
    pub enable_collaboration: bool,
    /// Enable CI/CD integration
    pub enable_cicd: bool,
}
impl Default for FeatureFlags {
    fn default() -> Self {
        Self {
            enable_comparison: true,
            enable_branching: true,
            enable_performance_tracking: true,
            enable_collaboration: true,
            enable_cicd: false,
        }
    }
}
/// Hub UI configuration
#[derive(Debug, Clone)]
pub struct HubUiConfig {
    /// Server bind address
    pub bind_address: String,
    /// Server port
    pub port: u16,
    /// Enable authentication
    pub enable_auth: bool,
    /// Static files directory
    pub static_dir: Option<PathBuf>,
    /// Theme configuration
    pub theme: ThemeConfig,
    /// Feature flags
    pub features: FeatureFlags,
}
impl Default for HubUiConfig {
    fn default() -> Self {
        Self {
            bind_address: "127.0.0.1".to_string(),
            port: 8080,
            enable_auth: false,
            static_dir: None,
            theme: ThemeConfig::default(),
            features: FeatureFlags::default(),
        }
    }
}
/// Hub UI server state
#[derive(Debug, Clone)]
pub struct HubUiState {
    /// Model repositories
    pub(super) repositories: Arc<Mutex<HashMap<String, ModelRepository>>>,
    /// UI configuration
    pub(super) config: HubUiConfig,
    /// Cache directory
    pub(super) cache_dir: PathBuf,
}
impl HubUiState {
    /// Create new Hub UI state
    pub fn new(config: HubUiConfig, cache_dir: PathBuf) -> Self {
        Self {
            repositories: Arc::new(Mutex::new(HashMap::new())),
            config,
            cache_dir,
        }
    }
    /// Add a model repository
    pub fn add_repository(&self, repository: ModelRepository) -> Result<(), TrustformersError> {
        let mut repos = self.repositories.lock().unwrap_or_else(|p| p.into_inner());
        repos.insert(repository.model_id.clone(), repository);
        Ok(())
    }
    /// Get a model repository
    pub fn get_repository(&self, model_id: &str) -> Option<ModelRepository> {
        let repos = self.repositories.lock().unwrap_or_else(|p| p.into_inner());
        repos.get(model_id).cloned()
    }
    /// List all repositories
    pub fn list_repositories(&self) -> Vec<ModelRepository> {
        let repos = self.repositories.lock().unwrap_or_else(|p| p.into_inner());
        repos.values().cloned().collect()
    }
    /// Local directory where downloaded model files are cached.
    pub fn cache_dir(&self) -> &std::path::Path {
        &self.cache_dir
    }
    /// Add a version to a repository
    pub fn add_version(
        &self,
        model_id: &str,
        version: ModelVersion,
    ) -> Result<(), TrustformersError> {
        let mut repos = self.repositories.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(repo) = repos.get_mut(model_id) {
            repo.versions.insert(version.version.clone(), version.clone());
            repo.version_history.push(version.version);
            repo.version_history.sort_by(|a, b| {
                let a_created = repo.versions.get(a).map(|v| v.created_at);
                let b_created = repo.versions.get(b).map(|v| v.created_at);
                a_created.cmp(&b_created)
            });
            repo.metadata.updated_at =
                SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
            repo.metadata.stats.version_count = repo.versions.len();
            Ok(())
        } else {
            Err(TrustformersError::hub(
                format!("Model not found: {}", model_id),
                model_id.to_string(),
            ))
        }
    }
    /// Update repository metadata
    pub fn update_repository(
        &self,
        model_id: &str,
        metadata: RepositoryMetadata,
    ) -> Result<ModelRepository, TrustformersError> {
        let mut repos = self.repositories.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(repo) = repos.get_mut(model_id) {
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
            repo.metadata = metadata;
            repo.metadata.updated_at = now;
            repo.metadata.stats.last_activity = now;
            Ok(repo.clone())
        } else {
            Err(TrustformersError::hub(
                format!("Model not found: {}", model_id),
                model_id.to_string(),
            ))
        }
    }
    /// Delete a repository
    pub fn delete_repository(&self, model_id: &str) -> Result<(), TrustformersError> {
        let mut repos = self.repositories.lock().unwrap_or_else(|p| p.into_inner());
        if repos.remove(model_id).is_some() {
            Ok(())
        } else {
            Err(TrustformersError::hub(
                format!("Model not found: {}", model_id),
                model_id.to_string(),
            ))
        }
    }
    /// Update a specific version in a repository
    pub fn update_version(
        &self,
        model_id: &str,
        version_id: &str,
        updated_version: ModelVersion,
    ) -> Result<ModelVersion, TrustformersError> {
        let mut repos = self.repositories.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(repo) = repos.get_mut(model_id) {
            if repo.versions.contains_key(version_id) {
                repo.versions.insert(version_id.to_string(), updated_version.clone());
                repo.metadata.updated_at =
                    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
                Ok(updated_version)
            } else {
                Err(TrustformersError::hub(
                    format!("Version {} not found in {}", version_id, model_id),
                    model_id.to_string(),
                ))
            }
        } else {
            Err(TrustformersError::hub(
                format!("Model not found: {}", model_id),
                model_id.to_string(),
            ))
        }
    }
    /// Delete a specific version from a repository
    pub fn delete_version(
        &self,
        model_id: &str,
        version_id: &str,
    ) -> Result<(), TrustformersError> {
        let mut repos = self.repositories.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(repo) = repos.get_mut(model_id) {
            if repo.versions.remove(version_id).is_some() {
                repo.version_history.retain(|v| v != version_id);
                repo.metadata.updated_at =
                    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
                repo.metadata.stats.version_count = repo.versions.len();
                Ok(())
            } else {
                Err(TrustformersError::hub(
                    format!("Version {} not found in {}", version_id, model_id),
                    model_id.to_string(),
                ))
            }
        } else {
            Err(TrustformersError::hub(
                format!("Model not found: {}", model_id),
                model_id.to_string(),
            ))
        }
    }
}
/// Theme configuration
#[derive(Debug, Clone)]
pub struct ThemeConfig {
    /// Primary color
    pub primary_color: String,
    /// Secondary color
    pub secondary_color: String,
    /// Dark mode support
    pub dark_mode: bool,
    /// Custom CSS
    pub custom_css: Option<String>,
}
impl Default for ThemeConfig {
    fn default() -> Self {
        Self {
            primary_color: "#3b82f6".to_string(),
            secondary_color: "#64748b".to_string(),
            dark_mode: true,
            custom_css: None,
        }
    }
}
