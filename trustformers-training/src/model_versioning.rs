use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    pub version_id: String,
    pub model_name: String,
    pub version_number: u32,
    pub created_at: u64,
    pub created_by: String,
    pub description: String,
    pub tags: Vec<String>,
    pub metadata: HashMap<String, String>,
    pub model_hash: String,
    pub file_path: PathBuf,
    pub parent_version: Option<String>,
    pub training_config: TrainingConfig,
    pub performance_metrics: PerformanceMetrics,
    pub model_size: u64,
    pub status: ModelStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub epochs: u32,
    pub optimizer: String,
    pub loss_function: String,
    pub regularization: HashMap<String, f32>,
    pub hyperparameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub accuracy: f32,
    pub loss: f32,
    pub validation_accuracy: f32,
    pub validation_loss: f32,
    pub f1_score: Option<f32>,
    pub precision: Option<f32>,
    pub recall: Option<f32>,
    pub custom_metrics: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelStatus {
    Training,
    Trained,
    Validated,
    Deployed,
    Archived,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistry {
    pub models: HashMap<String, Vec<ModelVersion>>,
    pub latest_versions: HashMap<String, String>,
    pub registry_path: PathBuf,
}

pub struct ModelVersioningManager {
    registry: ModelRegistry,
    storage_root: PathBuf,
}

impl ModelVersioningManager {
    pub fn new(storage_root: PathBuf) -> Result<Self> {
        let registry_path = storage_root.join("model_registry.json");

        let registry = if registry_path.exists() {
            let registry_data =
                std::fs::read_to_string(&registry_path).context("Failed to read model registry")?;
            serde_json::from_str(&registry_data).context("Failed to parse model registry")?
        } else {
            ModelRegistry {
                models: HashMap::new(),
                latest_versions: HashMap::new(),
                registry_path: registry_path.clone(),
            }
        };

        // Create storage root if it doesn't exist
        std::fs::create_dir_all(&storage_root)
            .context("Failed to create storage root directory")?;

        Ok(Self {
            registry,
            storage_root,
        })
    }

    pub fn create_version(
        &mut self,
        model_name: String,
        model_data: &[u8],
        description: String,
        created_by: String,
        tags: Vec<String>,
        training_config: TrainingConfig,
        performance_metrics: PerformanceMetrics,
        metadata: HashMap<String, String>,
    ) -> Result<ModelVersion> {
        // Generate version number
        let version_number = self.get_next_version_number(&model_name);

        // Generate version ID
        let version_id = format!("{}_{:04}", model_name, version_number);

        // Calculate model hash
        let mut hasher = Sha256::new();
        hasher.update(model_data);
        let model_hash = format!("{:x}", hasher.finalize());

        // Create file path
        let file_name = format!("{}.model", version_id);
        let file_path = self.storage_root.join(&model_name).join(&file_name);

        // Create model directory if it doesn't exist
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent).context("Failed to create model directory")?;
        }

        // Save model data
        std::fs::write(&file_path, model_data).context("Failed to save model data")?;

        // Get current timestamp
        let created_at = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

        // Get parent version (latest version of the same model)
        let parent_version = self.registry.latest_versions.get(&model_name).cloned();

        // Create version record
        let version = ModelVersion {
            version_id: version_id.clone(),
            model_name: model_name.clone(),
            version_number,
            created_at,
            created_by,
            description,
            tags,
            metadata,
            model_hash,
            file_path,
            parent_version,
            training_config,
            performance_metrics,
            model_size: model_data.len() as u64,
            status: ModelStatus::Trained,
        };

        // Add to registry
        self.registry
            .models
            .entry(model_name.clone())
            .or_default()
            .push(version.clone());

        // Update latest version
        self.registry.latest_versions.insert(model_name, version_id);

        // Save registry
        self.save_registry()?;

        Ok(version)
    }

    pub fn get_version(&self, model_name: &str, version_id: &str) -> Option<&ModelVersion> {
        self.registry
            .models
            .get(model_name)?
            .iter()
            .find(|v| v.version_id == version_id)
    }

    pub fn get_latest_version(&self, model_name: &str) -> Option<&ModelVersion> {
        let latest_version_id = self.registry.latest_versions.get(model_name)?;
        self.get_version(model_name, latest_version_id)
    }

    pub fn list_versions(&self, model_name: &str) -> Vec<&ModelVersion> {
        self.registry
            .models
            .get(model_name)
            .map(|versions| {
                let mut sorted_versions: Vec<_> = versions.iter().collect();
                sorted_versions.sort_by(|a, b| b.created_at.cmp(&a.created_at));
                sorted_versions
            })
            .unwrap_or_default()
    }

    pub fn list_models(&self) -> Vec<String> {
        self.registry.models.keys().cloned().collect()
    }

    pub fn update_status(
        &mut self,
        model_name: &str,
        version_id: &str,
        status: ModelStatus,
    ) -> Result<()> {
        let versions = self.registry.models.get_mut(model_name).context("Model not found")?;

        let version = versions
            .iter_mut()
            .find(|v| v.version_id == version_id)
            .context("Version not found")?;

        version.status = status;
        self.save_registry()?;

        Ok(())
    }

    pub fn add_tag(&mut self, model_name: &str, version_id: &str, tag: String) -> Result<()> {
        let versions = self.registry.models.get_mut(model_name).context("Model not found")?;

        let version = versions
            .iter_mut()
            .find(|v| v.version_id == version_id)
            .context("Version not found")?;

        if !version.tags.contains(&tag) {
            version.tags.push(tag);
            self.save_registry()?;
        }

        Ok(())
    }

    pub fn remove_tag(&mut self, model_name: &str, version_id: &str, tag: &str) -> Result<()> {
        let versions = self.registry.models.get_mut(model_name).context("Model not found")?;

        let version = versions
            .iter_mut()
            .find(|v| v.version_id == version_id)
            .context("Version not found")?;

        version.tags.retain(|t| t != tag);
        self.save_registry()?;

        Ok(())
    }

    pub fn find_versions_by_tag(&self, tag: &str) -> Vec<&ModelVersion> {
        self.registry
            .models
            .values()
            .flatten()
            .filter(|version| version.tags.contains(&tag.to_string()))
            .collect()
    }

    pub fn find_versions_by_performance(
        &self,
        metric_name: &str,
        min_value: f32,
        max_value: Option<f32>,
    ) -> Vec<&ModelVersion> {
        self.registry
            .models
            .values()
            .flatten()
            .filter(|version| {
                if let Some(value) = version.performance_metrics.custom_metrics.get(metric_name) {
                    *value >= min_value && max_value.map_or(true, |max| *value <= max)
                } else {
                    // Check standard metrics
                    match metric_name {
                        "accuracy" => {
                            let value = version.performance_metrics.accuracy;
                            value >= min_value && max_value.map_or(true, |max| value <= max)
                        },
                        "loss" => {
                            let value = version.performance_metrics.loss;
                            value >= min_value && max_value.map_or(true, |max| value <= max)
                        },
                        "validation_accuracy" => {
                            let value = version.performance_metrics.validation_accuracy;
                            value >= min_value && max_value.map_or(true, |max| value <= max)
                        },
                        "validation_loss" => {
                            let value = version.performance_metrics.validation_loss;
                            value >= min_value && max_value.map_or(true, |max| value <= max)
                        },
                        _ => false,
                    }
                }
            })
            .collect()
    }

    pub fn delete_version(&mut self, model_name: &str, version_id: &str) -> Result<()> {
        // Get the version to delete
        let version =
            self.get_version(model_name, version_id).context("Version not found")?.clone();

        // Remove from registry
        if let Some(versions) = self.registry.models.get_mut(model_name) {
            versions.retain(|v| v.version_id != version_id);

            // If this was the latest version, update the latest version
            if self.registry.latest_versions.get(model_name) == Some(&version_id.to_string()) {
                if let Some(latest) = versions.iter().max_by_key(|v| v.created_at) {
                    self.registry
                        .latest_versions
                        .insert(model_name.to_string(), latest.version_id.clone());
                } else {
                    self.registry.latest_versions.remove(model_name);
                }
            }
        }

        // Delete the model file
        if version.file_path.exists() {
            std::fs::remove_file(&version.file_path).context("Failed to delete model file")?;
        }

        self.save_registry()?;

        Ok(())
    }

    pub fn load_model_data(&self, model_name: &str, version_id: &str) -> Result<Vec<u8>> {
        let version = self.get_version(model_name, version_id).context("Version not found")?;

        std::fs::read(&version.file_path).context("Failed to read model data")
    }

    pub fn get_version_lineage(&self, model_name: &str, version_id: &str) -> Vec<&ModelVersion> {
        let mut lineage = Vec::new();
        let mut current_version_id = Some(version_id.to_string());

        while let Some(vid) = current_version_id {
            if let Some(version) = self.get_version(model_name, &vid) {
                lineage.push(version);
                current_version_id = version.parent_version.clone();
            } else {
                break;
            }
        }

        lineage
    }

    pub fn compare_versions(
        &self,
        model_name: &str,
        version_id1: &str,
        version_id2: &str,
    ) -> Result<VersionComparison> {
        let version1 =
            self.get_version(model_name, version_id1).context("First version not found")?;
        let version2 =
            self.get_version(model_name, version_id2).context("Second version not found")?;

        Ok(VersionComparison {
            version1: version1.clone(),
            version2: version2.clone(),
            accuracy_diff: version2.performance_metrics.accuracy
                - version1.performance_metrics.accuracy,
            loss_diff: version2.performance_metrics.loss - version1.performance_metrics.loss,
            size_diff: version2.model_size as i64 - version1.model_size as i64,
            config_changes: self
                .compare_training_configs(&version1.training_config, &version2.training_config),
        })
    }

    fn compare_training_configs(
        &self,
        config1: &TrainingConfig,
        config2: &TrainingConfig,
    ) -> Vec<String> {
        let mut changes = Vec::new();

        if config1.learning_rate != config2.learning_rate {
            changes.push(format!(
                "Learning rate: {} -> {}",
                config1.learning_rate, config2.learning_rate
            ));
        }

        if config1.batch_size != config2.batch_size {
            changes.push(format!(
                "Batch size: {} -> {}",
                config1.batch_size, config2.batch_size
            ));
        }

        if config1.epochs != config2.epochs {
            changes.push(format!("Epochs: {} -> {}", config1.epochs, config2.epochs));
        }

        if config1.optimizer != config2.optimizer {
            changes.push(format!(
                "Optimizer: {} -> {}",
                config1.optimizer, config2.optimizer
            ));
        }

        if config1.loss_function != config2.loss_function {
            changes.push(format!(
                "Loss function: {} -> {}",
                config1.loss_function, config2.loss_function
            ));
        }

        changes
    }

    fn get_next_version_number(&self, model_name: &str) -> u32 {
        self.registry
            .models
            .get(model_name)
            .map(|versions| versions.iter().map(|v| v.version_number).max().unwrap_or(0) + 1)
            .unwrap_or(1)
    }

    fn save_registry(&self) -> Result<()> {
        let registry_data =
            serde_json::to_string_pretty(&self.registry).context("Failed to serialize registry")?;

        std::fs::write(&self.registry.registry_path, registry_data)
            .context("Failed to save registry")?;

        Ok(())
    }

    pub fn get_statistics(&self) -> ModelRegistryStatistics {
        let total_models = self.registry.models.len();
        let total_versions = self.registry.models.values().map(|v| v.len()).sum();
        let total_size: u64 = self.registry.models.values().flatten().map(|v| v.model_size).sum();

        let status_counts =
            self.registry
                .models
                .values()
                .flatten()
                .fold(HashMap::new(), |mut acc, version| {
                    *acc.entry(format!("{:?}", version.status)).or_insert(0) += 1;
                    acc
                });

        ModelRegistryStatistics {
            total_models,
            total_versions,
            total_size,
            status_counts,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VersionComparison {
    pub version1: ModelVersion,
    pub version2: ModelVersion,
    pub accuracy_diff: f32,
    pub loss_diff: f32,
    pub size_diff: i64,
    pub config_changes: Vec<String>,
}

#[derive(Debug)]
pub struct ModelRegistryStatistics {
    pub total_models: usize,
    pub total_versions: usize,
    pub total_size: u64,
    pub status_counts: HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_model_versioning_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelVersioningManager::new(temp_dir.path().to_path_buf()).unwrap();
        assert_eq!(manager.list_models().len(), 0);
    }

    #[test]
    fn test_create_version() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = ModelVersioningManager::new(temp_dir.path().to_path_buf()).unwrap();

        let training_config = TrainingConfig {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 10,
            optimizer: "Adam".to_string(),
            loss_function: "CrossEntropy".to_string(),
            regularization: HashMap::new(),
            hyperparameters: HashMap::new(),
        };

        let performance_metrics = PerformanceMetrics {
            accuracy: 0.95,
            loss: 0.05,
            validation_accuracy: 0.93,
            validation_loss: 0.07,
            f1_score: Some(0.94),
            precision: Some(0.96),
            recall: Some(0.92),
            custom_metrics: HashMap::new(),
        };

        let model_data = b"fake model data";
        let version = manager
            .create_version(
                "test_model".to_string(),
                model_data,
                "Test model version".to_string(),
                "test_user".to_string(),
                vec!["test".to_string()],
                training_config,
                performance_metrics,
                HashMap::new(),
            )
            .unwrap();

        assert_eq!(version.model_name, "test_model");
        assert_eq!(version.version_number, 1);
        assert_eq!(version.description, "Test model version");
        assert_eq!(version.tags, vec!["test"]);
    }

    #[test]
    fn test_get_latest_version() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = ModelVersioningManager::new(temp_dir.path().to_path_buf()).unwrap();

        let training_config = TrainingConfig {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 10,
            optimizer: "Adam".to_string(),
            loss_function: "CrossEntropy".to_string(),
            regularization: HashMap::new(),
            hyperparameters: HashMap::new(),
        };

        let performance_metrics = PerformanceMetrics {
            accuracy: 0.95,
            loss: 0.05,
            validation_accuracy: 0.93,
            validation_loss: 0.07,
            f1_score: None,
            precision: None,
            recall: None,
            custom_metrics: HashMap::new(),
        };

        // Create first version
        let model_data1 = b"fake model data v1";
        manager
            .create_version(
                "test_model".to_string(),
                model_data1,
                "Version 1".to_string(),
                "test_user".to_string(),
                vec![],
                training_config.clone(),
                performance_metrics.clone(),
                HashMap::new(),
            )
            .unwrap();

        // Create second version
        let model_data2 = b"fake model data v2";
        let version2 = manager
            .create_version(
                "test_model".to_string(),
                model_data2,
                "Version 2".to_string(),
                "test_user".to_string(),
                vec![],
                training_config,
                performance_metrics,
                HashMap::new(),
            )
            .unwrap();

        let latest = manager.get_latest_version("test_model").unwrap();
        assert_eq!(latest.version_id, version2.version_id);
        assert_eq!(latest.version_number, 2);
    }

    #[test]
    fn test_version_lineage() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = ModelVersioningManager::new(temp_dir.path().to_path_buf()).unwrap();

        let training_config = TrainingConfig {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 10,
            optimizer: "Adam".to_string(),
            loss_function: "CrossEntropy".to_string(),
            regularization: HashMap::new(),
            hyperparameters: HashMap::new(),
        };

        let performance_metrics = PerformanceMetrics {
            accuracy: 0.95,
            loss: 0.05,
            validation_accuracy: 0.93,
            validation_loss: 0.07,
            f1_score: None,
            precision: None,
            recall: None,
            custom_metrics: HashMap::new(),
        };

        // Create multiple versions
        manager
            .create_version(
                "test_model".to_string(),
                b"v1",
                "Version 1".to_string(),
                "user".to_string(),
                vec![],
                training_config.clone(),
                performance_metrics.clone(),
                HashMap::new(),
            )
            .unwrap();

        manager
            .create_version(
                "test_model".to_string(),
                b"v2",
                "Version 2".to_string(),
                "user".to_string(),
                vec![],
                training_config.clone(),
                performance_metrics.clone(),
                HashMap::new(),
            )
            .unwrap();

        let version3 = manager
            .create_version(
                "test_model".to_string(),
                b"v3",
                "Version 3".to_string(),
                "user".to_string(),
                vec![],
                training_config,
                performance_metrics,
                HashMap::new(),
            )
            .unwrap();

        let lineage = manager.get_version_lineage("test_model", &version3.version_id);
        assert_eq!(lineage.len(), 3);
        assert_eq!(lineage[0].version_number, 3);
        assert_eq!(lineage[1].version_number, 2);
        assert_eq!(lineage[2].version_number, 1);
    }

    #[test]
    fn test_find_versions_by_tag() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = ModelVersioningManager::new(temp_dir.path().to_path_buf()).unwrap();

        let training_config = TrainingConfig {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 10,
            optimizer: "Adam".to_string(),
            loss_function: "CrossEntropy".to_string(),
            regularization: HashMap::new(),
            hyperparameters: HashMap::new(),
        };

        let performance_metrics = PerformanceMetrics {
            accuracy: 0.95,
            loss: 0.05,
            validation_accuracy: 0.93,
            validation_loss: 0.07,
            f1_score: None,
            precision: None,
            recall: None,
            custom_metrics: HashMap::new(),
        };

        // Create version with production tag
        manager
            .create_version(
                "model1".to_string(),
                b"data",
                "Production model".to_string(),
                "user".to_string(),
                vec!["production".to_string()],
                training_config.clone(),
                performance_metrics.clone(),
                HashMap::new(),
            )
            .unwrap();

        // Create version with development tag
        manager
            .create_version(
                "model2".to_string(),
                b"data",
                "Dev model".to_string(),
                "user".to_string(),
                vec!["development".to_string()],
                training_config,
                performance_metrics,
                HashMap::new(),
            )
            .unwrap();

        let production_versions = manager.find_versions_by_tag("production");
        assert_eq!(production_versions.len(), 1);
        assert_eq!(production_versions[0].model_name, "model1");

        let dev_versions = manager.find_versions_by_tag("development");
        assert_eq!(dev_versions.len(), 1);
        assert_eq!(dev_versions[0].model_name, "model2");
    }
}
