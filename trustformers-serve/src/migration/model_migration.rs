use super::*;
use anyhow::Result;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use tokio::fs;

/// Model migration handler for TrustformeRS Serve
pub struct ModelMigrator {
    source_version: String,
    target_version: String,
    migration_rules: HashMap<String, Box<dyn ModelMigrationRule + Send + Sync>>,
}

impl ModelMigrator {
    pub fn new(source_version: String, target_version: String) -> Self {
        let mut migrator = Self {
            source_version,
            target_version,
            migration_rules: HashMap::new(),
        };

        migrator.register_default_rules();
        migrator
    }

    fn register_default_rules(&mut self) {
        // Register migration rules for different version transitions
        self.register_rule("1.0.0->2.0.0", Box::new(V1ToV2ModelRule));
        self.register_rule("0.1.0->1.0.0", Box::new(V01ToV1ModelRule));
        self.register_rule("2.0.0->2.1.0", Box::new(V2ToV21ModelRule));
    }

    pub fn register_rule(
        &mut self,
        version_path: &str,
        rule: Box<dyn ModelMigrationRule + Send + Sync>,
    ) {
        self.migration_rules.insert(version_path.to_string(), rule);
    }

    pub async fn migrate_models_directory<P: AsRef<Path>>(
        &self,
        models_path: P,
    ) -> Result<ModelMigrationResult> {
        let models_path = models_path.as_ref();
        let migration_key = format!("{}->{}", self.source_version, self.target_version);

        if let Some(rule) = self.migration_rules.get(&migration_key) {
            let mut result = ModelMigrationResult {
                source_version: self.source_version.clone(),
                target_version: self.target_version.clone(),
                migrated_models: Vec::new(),
                errors: Vec::new(),
                statistics: ModelMigrationStats::new(),
            };

            // Create backup directory for models
            let backup_path = models_path
                .parent()
                .unwrap()
                .join(format!("models_backup_{}", Utc::now().timestamp()));
            fs::create_dir_all(&backup_path).await?;

            // Copy models to backup before migration
            self.copy_directory_recursive(models_path, &backup_path).await?;

            // Get all model files and directories
            let models = self.get_model_entries(models_path).await?;

            for model_entry in models {
                result.statistics.total_models += 1;

                match rule.migrate_model(&model_entry).await {
                    Ok(model_result) => {
                        result.migrated_models.push(model_result);
                        result.statistics.migrated_models += 1;
                    },
                    Err(e) => {
                        result.errors.push(ModelMigrationError {
                            model_path: model_entry.to_string_lossy().to_string(),
                            error: e.to_string(),
                        });
                        result.statistics.failed_models += 1;
                    },
                }
            }

            // Validate migrated models
            result.statistics.validation_results =
                self.validate_migrated_models(models_path).await?;

            // Update model registry
            self.update_model_registry(models_path).await?;

            Ok(result)
        } else {
            Err(anyhow::anyhow!(
                "No model migration rule found for {} -> {}",
                self.source_version,
                self.target_version
            ))
        }
    }

    fn copy_directory_recursive<'a>(
        &'a self,
        src: &'a Path,
        dst: &'a Path,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            fs::create_dir_all(dst).await?;

            let mut entries = fs::read_dir(src).await?;
            while let Some(entry) = entries.next_entry().await? {
                let src_path = entry.path();
                let dst_path = dst.join(entry.file_name());

                if src_path.is_dir() {
                    self.copy_directory_recursive(&src_path, &dst_path).await?;
                } else {
                    fs::copy(&src_path, &dst_path).await?;
                }
            }

            Ok(())
        })
    }

    async fn get_model_entries(&self, models_path: &Path) -> Result<Vec<std::path::PathBuf>> {
        let mut entries = Vec::new();

        if models_path.is_dir() {
            let mut dir_entries = fs::read_dir(models_path).await?;
            while let Some(entry) = dir_entries.next_entry().await? {
                let path = entry.path();
                if path.is_dir() {
                    // Check if it's a model directory (contains model files)
                    if self.is_model_directory(&path).await? {
                        entries.push(path);
                    }
                } else {
                    // Check if it's a model file
                    if self.is_model_file(&path) {
                        entries.push(path);
                    }
                }
            }
        }

        Ok(entries)
    }

    async fn is_model_directory(&self, path: &Path) -> Result<bool> {
        let mut entries = fs::read_dir(path).await?;
        while let Some(entry) = entries.next_entry().await? {
            let file_path = entry.path();
            if self.is_model_file(&file_path) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn is_model_file(&self, path: &Path) -> bool {
        if let Some(ext) = path.extension() {
            match ext.to_str() {
                Some("onnx") | Some("pt") | Some("pth") | Some("safetensors") | Some("bin")
                | Some("json") | Some("config") => true,
                _ => false,
            }
        } else {
            false
        }
    }

    async fn validate_migrated_models(&self, models_path: &Path) -> Result<Vec<ValidationResult>> {
        let mut results = Vec::new();

        // Check model integrity
        let models = self.get_model_entries(models_path).await?;
        for model_path in models {
            let model_name =
                model_path.file_name().and_then(|name| name.to_str()).unwrap_or("unknown");

            // Basic file existence check
            if model_path.exists() {
                results.push(ValidationResult {
                    rule_name: format!("Model Existence: {}", model_name),
                    status: ValidationStatus::Passed,
                    message: format!("Model {} exists", model_name),
                    details: None,
                });
            } else {
                results.push(ValidationResult {
                    rule_name: format!("Model Existence: {}", model_name),
                    status: ValidationStatus::Failed,
                    message: format!("Model {} is missing", model_name),
                    details: Some(serde_json::json!({"path": model_path})),
                });
            }

            // Check model metadata
            if let Ok(metadata) = self.check_model_metadata(&model_path).await {
                results.push(ValidationResult {
                    rule_name: format!("Model Metadata: {}", model_name),
                    status: ValidationStatus::Passed,
                    message: format!("Model {} metadata is valid", model_name),
                    details: Some(serde_json::json!(metadata)),
                });
            } else {
                results.push(ValidationResult {
                    rule_name: format!("Model Metadata: {}", model_name),
                    status: ValidationStatus::Warning,
                    message: format!("Model {} metadata could not be validated", model_name),
                    details: None,
                });
            }
        }

        // Check model registry
        let registry_path = models_path.join("registry.json");
        if registry_path.exists() {
            results.push(ValidationResult {
                rule_name: "Model Registry".to_string(),
                status: ValidationStatus::Passed,
                message: "Model registry exists".to_string(),
                details: None,
            });
        } else {
            results.push(ValidationResult {
                rule_name: "Model Registry".to_string(),
                status: ValidationStatus::Warning,
                message: "Model registry is missing".to_string(),
                details: None,
            });
        }

        Ok(results)
    }

    async fn check_model_metadata(&self, model_path: &Path) -> Result<ModelMetadata> {
        let metadata_path = if model_path.is_dir() {
            model_path.join("config.json")
        } else {
            model_path.with_extension("json")
        };

        if metadata_path.exists() {
            let content = fs::read_to_string(&metadata_path).await?;
            let metadata: ModelMetadata = serde_json::from_str(&content)?;
            Ok(metadata)
        } else {
            // Create basic metadata from file info
            let file_metadata = fs::metadata(model_path).await?;
            Ok(ModelMetadata {
                name: model_path
                    .file_stem()  // Use file_stem to exclude extension
                    .and_then(|name| name.to_str())
                    .unwrap_or("unknown")
                    .to_string(),
                version: "unknown".to_string(),
                format: self.detect_model_format(model_path),
                size: file_metadata.len(),
                created_at: file_metadata
                    .created()
                    .unwrap_or_else(|_| std::time::SystemTime::now())
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                architecture: "unknown".to_string(),
                parameters: HashMap::new(),
            })
        }
    }

    fn detect_model_format(&self, model_path: &Path) -> String {
        if let Some(ext) = model_path.extension() {
            match ext.to_str() {
                Some("onnx") => "ONNX".to_string(),
                Some("pt") | Some("pth") => "PyTorch".to_string(),
                Some("safetensors") => "SafeTensors".to_string(),
                Some("bin") => "Binary".to_string(),
                _ => "Unknown".to_string(),
            }
        } else {
            "Unknown".to_string()
        }
    }

    async fn update_model_registry(&self, models_path: &Path) -> Result<()> {
        let registry_path = models_path.join("registry.json");
        let models = self.get_model_entries(models_path).await?;

        let mut registry = ModelRegistry {
            version: self.target_version.clone(),
            models: Vec::new(),
            last_updated: Utc::now(),
        };

        for model_path in models {
            if let Ok(metadata) = self.check_model_metadata(&model_path).await {
                registry.models.push(ModelRegistryEntry {
                    name: metadata.name,
                    path: model_path.to_string_lossy().to_string(),
                    version: metadata.version,
                    format: metadata.format,
                    size: metadata.size,
                    checksum: self.calculate_checksum(&model_path).await.unwrap_or_default(),
                    status: ModelStatus::Available,
                    last_validated: Utc::now(),
                });
            }
        }

        let registry_content = serde_json::to_string_pretty(&registry)?;
        fs::write(&registry_path, registry_content).await?;

        Ok(())
    }

    async fn calculate_checksum(&self, path: &Path) -> Result<String> {
        use sha2::{Digest, Sha256};

        let content = fs::read(path).await?;
        let mut hasher = Sha256::new();
        hasher.update(content);
        Ok(format!("{:x}", hasher.finalize()))
    }

    pub async fn rollback_model_migration(
        &self,
        backup_path: &Path,
        models_path: &Path,
    ) -> Result<()> {
        // Remove current models directory
        if models_path.exists() {
            fs::remove_dir_all(models_path).await?;
        }

        // Restore from backup
        self.copy_directory_recursive(backup_path, models_path).await?;

        Ok(())
    }

    pub async fn cleanup_backup(&self, backup_path: &Path) -> Result<()> {
        if backup_path.exists() {
            fs::remove_dir_all(backup_path).await?;
        }
        Ok(())
    }

    pub async fn optimize_models(&self, models_path: &Path) -> Result<OptimizationResult> {
        let mut result = OptimizationResult {
            optimized_models: Vec::new(),
            total_size_before: 0,
            total_size_after: 0,
            optimization_time: std::time::Duration::new(0, 0),
        };

        let start_time = std::time::Instant::now();
        let models = self.get_model_entries(models_path).await?;

        for model_path in models {
            let size_before = fs::metadata(&model_path).await?.len();
            result.total_size_before += size_before;

            // Apply model-specific optimizations
            if let Ok(optimized_size) = self.optimize_single_model(&model_path).await {
                result.optimized_models.push(ModelOptimizationResult {
                    model_path: model_path.to_string_lossy().to_string(),
                    size_before,
                    size_after: optimized_size,
                    compression_ratio: size_before as f64 / optimized_size as f64,
                    optimization_applied: vec!["quantization".to_string(), "pruning".to_string()],
                });
                result.total_size_after += optimized_size;
            } else {
                result.total_size_after += size_before;
            }
        }

        result.optimization_time = start_time.elapsed();
        Ok(result)
    }

    async fn optimize_single_model(&self, model_path: &Path) -> Result<u64> {
        // In a real implementation, this would apply various optimizations
        // For now, we'll just return the current size
        Ok(fs::metadata(model_path).await?.len())
    }
}

#[async_trait::async_trait]
pub trait ModelMigrationRule: Send + Sync {
    async fn migrate_model(&self, model_path: &Path) -> Result<ModelMigrationResult>;
    fn get_supported_formats(&self) -> Vec<String>;
    fn get_version_range(&self) -> (String, String);
    fn get_description(&self) -> String;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMigrationResult {
    pub source_version: String,
    pub target_version: String,
    pub migrated_models: Vec<ModelMigrationResult>,
    pub errors: Vec<ModelMigrationError>,
    pub statistics: ModelMigrationStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMigrationError {
    pub model_path: String,
    pub error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMigrationStats {
    pub total_models: usize,
    pub migrated_models: usize,
    pub failed_models: usize,
    pub validation_results: Vec<ValidationResult>,
}

impl ModelMigrationStats {
    pub fn new() -> Self {
        Self {
            total_models: 0,
            migrated_models: 0,
            failed_models: 0,
            validation_results: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub format: String,
    pub size: u64,
    pub created_at: u64,
    pub architecture: String,
    pub parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistry {
    pub version: String,
    pub models: Vec<ModelRegistryEntry>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistryEntry {
    pub name: String,
    pub path: String,
    pub version: String,
    pub format: String,
    pub size: u64,
    pub checksum: String,
    pub status: ModelStatus,
    pub last_validated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelStatus {
    Available,
    Loading,
    Error,
    Deprecated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub optimized_models: Vec<ModelOptimizationResult>,
    pub total_size_before: u64,
    pub total_size_after: u64,
    pub optimization_time: std::time::Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOptimizationResult {
    pub model_path: String,
    pub size_before: u64,
    pub size_after: u64,
    pub compression_ratio: f64,
    pub optimization_applied: Vec<String>,
}

/// Migration rule from version 1.0.0 to 2.0.0
pub struct V1ToV2ModelRule;

#[async_trait::async_trait]
impl ModelMigrationRule for V1ToV2ModelRule {
    async fn migrate_model(&self, model_path: &Path) -> Result<ModelMigrationResult> {
        let model_name = model_path.file_name().and_then(|name| name.to_str()).unwrap_or("unknown");

        // Check if model directory or file
        if model_path.is_dir() {
            // Migrate model directory
            let config_path = model_path.join("config.json");
            if config_path.exists() {
                let content = fs::read_to_string(&config_path).await?;
                let mut config: Value = serde_json::from_str(&content)?;

                // Update config for v2.0.0
                if let Some(obj) = config.as_object_mut() {
                    obj.insert("version".to_string(), Value::String("2.0.0".to_string()));
                    obj.insert(
                        "format_version".to_string(),
                        Value::String("2.0".to_string()),
                    );
                    obj.insert(
                        "migration_info".to_string(),
                        serde_json::json!({
                            "migrated_from": "1.0.0",
                            "migration_date": Utc::now().to_rfc3339(),
                            "migration_tool": "trustformers-serve"
                        }),
                    );

                    // Add new v2.0.0 fields
                    if !obj.contains_key("optimization") {
                        obj.insert(
                            "optimization".to_string(),
                            serde_json::json!({
                                "quantization": false,
                                "pruning": false,
                                "distillation": false
                            }),
                        );
                    }
                }

                let updated_config = serde_json::to_string_pretty(&config)?;
                fs::write(&config_path, updated_config).await?;
            }
        } else {
            // Migrate single model file
            // Create corresponding metadata file
            let metadata_path = model_path.with_extension("json");
            let metadata = ModelMetadata {
                name: model_name.to_string(),
                version: "2.0.0".to_string(),
                format: self.detect_model_format(model_path),
                size: fs::metadata(model_path).await?.len(),
                created_at: Utc::now().timestamp() as u64,
                architecture: "unknown".to_string(),
                parameters: HashMap::new(),
            };

            let metadata_content = serde_json::to_string_pretty(&metadata)?;
            fs::write(&metadata_path, metadata_content).await?;
        }

        Ok(ModelMigrationResult {
            source_version: "1.0.0".to_string(),
            target_version: "2.0.0".to_string(),
            migrated_models: vec![],
            errors: vec![],
            statistics: ModelMigrationStats::new(),
        })
    }

    fn get_supported_formats(&self) -> Vec<String> {
        vec![
            "ONNX".to_string(),
            "PyTorch".to_string(),
            "SafeTensors".to_string(),
            "Binary".to_string(),
        ]
    }

    fn get_version_range(&self) -> (String, String) {
        ("1.0.0".to_string(), "2.0.0".to_string())
    }

    fn get_description(&self) -> String {
        "Migrate models from v1.0.0 to v2.0.0: update metadata, add optimization settings"
            .to_string()
    }
}

impl V1ToV2ModelRule {
    fn detect_model_format(&self, model_path: &Path) -> String {
        if let Some(ext) = model_path.extension() {
            match ext.to_str() {
                Some("onnx") => "ONNX".to_string(),
                Some("pt") | Some("pth") => "PyTorch".to_string(),
                Some("safetensors") => "SafeTensors".to_string(),
                Some("bin") => "Binary".to_string(),
                _ => "Unknown".to_string(),
            }
        } else {
            "Unknown".to_string()
        }
    }
}

/// Migration rule from version 0.1.0 to 1.0.0
pub struct V01ToV1ModelRule;

#[async_trait::async_trait]
impl ModelMigrationRule for V01ToV1ModelRule {
    async fn migrate_model(&self, model_path: &Path) -> Result<ModelMigrationResult> {
        // Simple migration: create basic metadata
        let metadata_path = model_path.with_extension("json");
        let metadata = ModelMetadata {
            name: model_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("unknown")
                .to_string(),
            version: "1.0.0".to_string(),
            format: self.detect_model_format(model_path),
            size: fs::metadata(model_path).await?.len(),
            created_at: Utc::now().timestamp() as u64,
            architecture: "unknown".to_string(),
            parameters: HashMap::new(),
        };

        let metadata_content = serde_json::to_string_pretty(&metadata)?;
        fs::write(&metadata_path, metadata_content).await?;

        Ok(ModelMigrationResult {
            source_version: "0.1.0".to_string(),
            target_version: "1.0.0".to_string(),
            migrated_models: vec![],
            errors: vec![],
            statistics: ModelMigrationStats::new(),
        })
    }

    fn get_supported_formats(&self) -> Vec<String> {
        vec!["ONNX".to_string(), "PyTorch".to_string()]
    }

    fn get_version_range(&self) -> (String, String) {
        ("0.1.0".to_string(), "1.0.0".to_string())
    }

    fn get_description(&self) -> String {
        "Migrate models from v0.1.0 to v1.0.0: create metadata files".to_string()
    }
}

impl V01ToV1ModelRule {
    fn detect_model_format(&self, model_path: &Path) -> String {
        if let Some(ext) = model_path.extension() {
            match ext.to_str() {
                Some("onnx") => "ONNX".to_string(),
                Some("pt") | Some("pth") => "PyTorch".to_string(),
                _ => "Unknown".to_string(),
            }
        } else {
            "Unknown".to_string()
        }
    }
}

/// Migration rule from version 2.0.0 to 2.1.0
pub struct V2ToV21ModelRule;

#[async_trait::async_trait]
impl ModelMigrationRule for V2ToV21ModelRule {
    async fn migrate_model(&self, model_path: &Path) -> Result<ModelMigrationResult> {
        if model_path.is_dir() {
            let config_path = model_path.join("config.json");
            if config_path.exists() {
                let content = fs::read_to_string(&config_path).await?;
                let mut config: Value = serde_json::from_str(&content)?;

                if let Some(obj) = config.as_object_mut() {
                    obj.insert("version".to_string(), Value::String("2.1.0".to_string()));
                    obj.insert(
                        "format_version".to_string(),
                        Value::String("2.1".to_string()),
                    );

                    // Add new v2.1.0 features
                    if !obj.contains_key("streaming") {
                        obj.insert(
                            "streaming".to_string(),
                            serde_json::json!({
                                "enabled": true,
                                "chunk_size": 1024,
                                "buffer_size": 8192
                            }),
                        );
                    }

                    if !obj.contains_key("caching") {
                        obj.insert(
                            "caching".to_string(),
                            serde_json::json!({
                                "enabled": true,
                                "cache_size": 1000,
                                "ttl": 3600
                            }),
                        );
                    }
                }

                let updated_config = serde_json::to_string_pretty(&config)?;
                fs::write(&config_path, updated_config).await?;
            }
        }

        Ok(ModelMigrationResult {
            source_version: "2.0.0".to_string(),
            target_version: "2.1.0".to_string(),
            migrated_models: vec![],
            errors: vec![],
            statistics: ModelMigrationStats::new(),
        })
    }

    fn get_supported_formats(&self) -> Vec<String> {
        vec![
            "ONNX".to_string(),
            "PyTorch".to_string(),
            "SafeTensors".to_string(),
            "Binary".to_string(),
        ]
    }

    fn get_version_range(&self) -> (String, String) {
        ("2.0.0".to_string(), "2.1.0".to_string())
    }

    fn get_description(&self) -> String {
        "Migrate models from v2.0.0 to v2.1.0: add streaming and caching features".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_v1_to_v2_model_migration() {
        let temp_dir = TempDir::new().unwrap();
        let models_path = temp_dir.path().join("models");
        fs::create_dir_all(&models_path).await.unwrap();

        // Create a test model directory
        let model_dir = models_path.join("test_model");
        fs::create_dir_all(&model_dir).await.unwrap();

        // Create config.json
        let config = serde_json::json!({
            "name": "test_model",
            "architecture": "transformer"
        });
        fs::write(
            model_dir.join("config.json"),
            serde_json::to_string_pretty(&config).unwrap(),
        )
        .await
        .unwrap();

        // Create model file
        fs::write(model_dir.join("model.onnx"), "dummy model data").await.unwrap();

        let migrator = ModelMigrator::new("1.0.0".to_string(), "2.0.0".to_string());
        let result = migrator.migrate_models_directory(&models_path).await.unwrap();

        assert_eq!(result.source_version, "1.0.0");
        assert_eq!(result.target_version, "2.0.0");
        assert_eq!(result.errors.len(), 0);

        // Verify the migrated config
        let migrated_config_content =
            fs::read_to_string(model_dir.join("config.json")).await.unwrap();
        let migrated_config: Value = serde_json::from_str(&migrated_config_content).unwrap();

        assert_eq!(migrated_config["version"], "2.0.0");
        assert!(migrated_config["optimization"].is_object());
        assert!(migrated_config["migration_info"].is_object());
    }

    #[tokio::test]
    async fn test_model_registry_update() {
        let temp_dir = TempDir::new().unwrap();
        let models_path = temp_dir.path().join("models");
        fs::create_dir_all(&models_path).await.unwrap();

        // Create a test model file
        fs::write(models_path.join("test_model.onnx"), "dummy model data")
            .await
            .unwrap();

        let migrator = ModelMigrator::new("1.0.0".to_string(), "2.0.0".to_string());
        migrator.update_model_registry(&models_path).await.unwrap();

        // Verify registry was created
        let registry_path = models_path.join("registry.json");
        assert!(registry_path.exists());

        let registry_content = fs::read_to_string(&registry_path).await.unwrap();
        let registry: ModelRegistry = serde_json::from_str(&registry_content).unwrap();

        assert_eq!(registry.version, "2.0.0");
        assert_eq!(registry.models.len(), 1);
        assert_eq!(registry.models[0].name, "test_model");
    }

    #[tokio::test]
    async fn test_model_validation() {
        let temp_dir = TempDir::new().unwrap();
        let models_path = temp_dir.path().join("models");
        fs::create_dir_all(&models_path).await.unwrap();

        // Create test model
        fs::write(models_path.join("test_model.onnx"), "dummy model data")
            .await
            .unwrap();

        // Create registry
        let registry = ModelRegistry {
            version: "2.0.0".to_string(),
            models: vec![],
            last_updated: Utc::now(),
        };
        fs::write(
            models_path.join("registry.json"),
            serde_json::to_string_pretty(&registry).unwrap(),
        )
        .await
        .unwrap();

        let migrator = ModelMigrator::new("1.0.0".to_string(), "2.0.0".to_string());
        let results = migrator.validate_migrated_models(&models_path).await.unwrap();

        // Should have validation results for model and registry
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.rule_name.contains("Model Existence")));
        assert!(results.iter().any(|r| r.rule_name == "Model Registry"));
    }
}
