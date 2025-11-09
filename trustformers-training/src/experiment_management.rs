//! Experiment Management Framework
//!
//! This module provides comprehensive experiment tracking and management including:
//! - Experiment versioning and lineage tracking
//! - Hyperparameter versioning and comparison
//! - Model lineage and provenance tracking
//! - Reproducibility guarantees
//! - A/B testing framework
//! - Experiment metadata and artifact management

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Experiment metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentMetadata {
    /// Unique experiment identifier
    pub id: String,
    /// Experiment name
    pub name: String,
    /// Experiment description
    pub description: String,
    /// Experiment tags
    pub tags: Vec<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,
    /// Experiment creator
    pub creator: String,
    /// Experiment status
    pub status: ExperimentStatus,
    /// Parent experiment ID (for versioning)
    pub parent_id: Option<String>,
    /// Version number
    pub version: u32,
    /// Git commit hash (for reproducibility)
    pub git_commit: Option<String>,
    /// Environment information
    pub environment: EnvironmentInfo,
    /// Custom metadata
    pub custom_metadata: HashMap<String, String>,
}

/// Experiment status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExperimentStatus {
    /// Experiment is being planned
    Planning,
    /// Experiment is currently running
    Running,
    /// Experiment completed successfully
    Completed,
    /// Experiment failed
    Failed,
    /// Experiment was cancelled
    Cancelled,
    /// Experiment is paused
    Paused,
}

/// Environment information for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    /// Python version
    pub python_version: String,
    /// CUDA version
    pub cuda_version: Option<String>,
    /// PyTorch version
    pub pytorch_version: String,
    /// Hardware information
    pub hardware: HardwareInfo,
    /// Installed packages
    pub packages: HashMap<String, String>,
    /// System information
    pub system: SystemInfo,
}

/// Hardware information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    /// CPU model
    pub cpu_model: String,
    /// Number of CPU cores
    pub cpu_cores: u32,
    /// Total RAM in GB
    pub total_ram_gb: f32,
    /// GPU information
    pub gpus: Vec<GPUInfo>,
}

/// GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUInfo {
    /// GPU model
    pub model: String,
    /// GPU memory in GB
    pub memory_gb: f32,
    /// CUDA compute capability
    pub compute_capability: Option<String>,
}

/// System information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// Operating system
    pub os: String,
    /// OS version
    pub os_version: String,
    /// Architecture
    pub architecture: String,
    /// Hostname
    pub hostname: String,
}

/// Hyperparameter configuration with versioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterConfig {
    /// Configuration ID
    pub id: String,
    /// Configuration name
    pub name: String,
    /// Hyperparameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Configuration version
    pub version: u32,
    /// Parent configuration ID
    pub parent_id: Option<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Configuration hash (for change detection)
    pub config_hash: String,
    /// Comments or notes
    pub notes: Option<String>,
}

/// Model lineage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLineage {
    /// Model ID
    pub model_id: String,
    /// Model name
    pub model_name: String,
    /// Model version
    pub version: u32,
    /// Parent model ID
    pub parent_id: Option<String>,
    /// Training experiment ID
    pub training_experiment_id: String,
    /// Model architecture
    pub architecture: String,
    /// Training dataset ID
    pub training_dataset_id: String,
    /// Model artifacts
    pub artifacts: Vec<ModelArtifact>,
    /// Model metrics
    pub metrics: HashMap<String, f64>,
    /// Model size information
    pub size_info: ModelSizeInfo,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Model provenance
    pub provenance: ModelProvenance,
}

/// Model artifact information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArtifact {
    /// Artifact type
    pub artifact_type: ArtifactType,
    /// File path
    pub file_path: String,
    /// File size in bytes
    pub file_size: u64,
    /// File hash
    pub file_hash: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

/// Types of model artifacts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactType {
    /// Model weights
    Weights,
    /// Model configuration
    Config,
    /// Tokenizer
    Tokenizer,
    /// Training logs
    TrainingLogs,
    /// Evaluation results
    EvaluationResults,
    /// Checkpoints
    Checkpoint,
    /// Metrics
    Metrics,
    /// Other custom artifacts
    Custom(String),
}

/// Model size information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSizeInfo {
    /// Total number of parameters
    pub total_parameters: u64,
    /// Number of trainable parameters
    pub trainable_parameters: u64,
    /// Model size in MB
    pub model_size_mb: f32,
    /// Memory footprint in MB
    pub memory_footprint_mb: f32,
}

/// Model provenance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProvenance {
    /// Data lineage
    pub data_lineage: Vec<DataLineage>,
    /// Training pipeline
    pub training_pipeline: TrainingPipeline,
    /// Validation methodology
    pub validation_methodology: String,
    /// Quality assurance steps
    pub quality_assurance: Vec<QualityAssuranceStep>,
}

/// Data lineage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLineage {
    /// Dataset ID
    pub dataset_id: String,
    /// Dataset name
    pub dataset_name: String,
    /// Dataset version
    pub version: String,
    /// Data source
    pub source: String,
    /// Data preprocessing steps
    pub preprocessing_steps: Vec<String>,
    /// Data splits
    pub splits: HashMap<String, DataSplit>,
}

/// Data split information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSplit {
    /// Number of samples
    pub num_samples: u64,
    /// Split percentage
    pub percentage: f32,
    /// Split hash
    pub split_hash: String,
}

/// Training pipeline information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingPipeline {
    /// Pipeline steps
    pub steps: Vec<PipelineStep>,
    /// Optimizer configuration
    pub optimizer_config: serde_json::Value,
    /// Loss function
    pub loss_function: String,
    /// Training duration
    pub training_duration: chrono::Duration,
    /// Resource usage
    pub resource_usage: ResourceUsage,
}

/// Pipeline step information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStep {
    /// Step name
    pub name: String,
    /// Step type
    pub step_type: String,
    /// Step parameters
    pub parameters: serde_json::Value,
    /// Step duration
    pub duration: chrono::Duration,
    /// Step status
    pub status: String,
}

/// Resource usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Peak memory usage in MB
    pub peak_memory_mb: f32,
    /// Average CPU usage percentage
    pub avg_cpu_usage: f32,
    /// Average GPU usage percentage
    pub avg_gpu_usage: f32,
    /// Total compute time in seconds
    pub total_compute_time: f32,
    /// Energy consumption in kWh
    pub energy_consumption_kwh: f32,
}

/// Quality assurance step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssuranceStep {
    /// Step name
    pub name: String,
    /// Step description
    pub description: String,
    /// Step result
    pub result: String,
    /// Step timestamp
    pub timestamp: DateTime<Utc>,
    /// Step passed
    pub passed: bool,
}

/// A/B testing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestConfig {
    /// Test ID
    pub test_id: String,
    /// Test name
    pub test_name: String,
    /// Test description
    pub description: String,
    /// Control experiment ID
    pub control_experiment_id: String,
    /// Treatment experiment IDs
    pub treatment_experiment_ids: Vec<String>,
    /// Traffic allocation
    pub traffic_allocation: HashMap<String, f32>,
    /// Success metrics
    pub success_metrics: Vec<String>,
    /// Statistical significance threshold
    pub significance_threshold: f32,
    /// Minimum sample size
    pub minimum_sample_size: u64,
    /// Test duration
    pub test_duration: chrono::Duration,
    /// Test status
    pub status: ABTestStatus,
    /// Test results
    pub results: Option<ABTestResults>,
}

/// A/B test status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ABTestStatus {
    /// Test is being planned
    Planning,
    /// Test is currently running
    Running,
    /// Test completed successfully
    Completed,
    /// Test was stopped early
    Stopped,
    /// Test failed
    Failed,
}

/// A/B test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestResults {
    /// Test duration
    pub test_duration: chrono::Duration,
    /// Total samples
    pub total_samples: u64,
    /// Metric results by experiment
    pub metric_results: HashMap<String, ExperimentResults>,
    /// Statistical significance
    pub statistical_significance: HashMap<String, f32>,
    /// Confidence intervals
    pub confidence_intervals: HashMap<String, (f32, f32)>,
    /// Winner experiment ID
    pub winner_experiment_id: Option<String>,
    /// Test conclusion
    pub conclusion: String,
}

/// Experiment results for A/B testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResults {
    /// Number of samples
    pub samples: u64,
    /// Metric values
    pub metrics: HashMap<String, f64>,
    /// Conversion rates
    pub conversion_rates: HashMap<String, f32>,
    /// Statistical significance
    pub statistical_significance: f32,
}

/// Experiment management system
pub struct ExperimentManager {
    /// Experiments storage
    experiments: HashMap<String, ExperimentMetadata>,
    /// Hyperparameter configurations
    hyperparameter_configs: HashMap<String, HyperparameterConfig>,
    /// Model lineage
    model_lineage: HashMap<String, ModelLineage>,
    /// A/B tests
    ab_tests: HashMap<String, ABTestConfig>,
    /// Current experiment ID
    current_experiment_id: Option<String>,
}

impl ExperimentManager {
    /// Create a new experiment manager
    pub fn new() -> Self {
        Self {
            experiments: HashMap::new(),
            hyperparameter_configs: HashMap::new(),
            model_lineage: HashMap::new(),
            ab_tests: HashMap::new(),
            current_experiment_id: None,
        }
    }

    /// Create a new experiment
    pub fn create_experiment(
        &mut self,
        name: String,
        description: String,
        tags: Vec<String>,
        creator: String,
        parent_id: Option<String>,
        environment: EnvironmentInfo,
    ) -> String {
        let experiment_id = Uuid::new_v4().to_string();
        let now = Utc::now();

        let version = if let Some(parent_id) = &parent_id {
            if let Some(parent) = self.experiments.get(parent_id) {
                parent.version + 1
            } else {
                1
            }
        } else {
            1
        };

        let experiment = ExperimentMetadata {
            id: experiment_id.clone(),
            name,
            description,
            tags,
            created_at: now,
            updated_at: now,
            creator,
            status: ExperimentStatus::Planning,
            parent_id,
            version,
            git_commit: None,
            environment,
            custom_metadata: HashMap::new(),
        };

        self.experiments.insert(experiment_id.clone(), experiment);
        experiment_id
    }

    /// Update experiment status
    pub fn update_experiment_status(
        &mut self,
        experiment_id: &str,
        status: ExperimentStatus,
    ) -> Result<(), String> {
        if let Some(experiment) = self.experiments.get_mut(experiment_id) {
            experiment.status = status;
            experiment.updated_at = Utc::now();
            Ok(())
        } else {
            Err(format!("Experiment {} not found", experiment_id))
        }
    }

    /// Add custom metadata to experiment
    pub fn add_experiment_metadata(
        &mut self,
        experiment_id: &str,
        key: String,
        value: String,
    ) -> Result<(), String> {
        if let Some(experiment) = self.experiments.get_mut(experiment_id) {
            experiment.custom_metadata.insert(key, value);
            experiment.updated_at = Utc::now();
            Ok(())
        } else {
            Err(format!("Experiment {} not found", experiment_id))
        }
    }

    /// Set current experiment
    pub fn set_current_experiment(&mut self, experiment_id: String) -> Result<(), String> {
        if self.experiments.contains_key(&experiment_id) {
            self.current_experiment_id = Some(experiment_id);
            Ok(())
        } else {
            Err(format!("Experiment {} not found", experiment_id))
        }
    }

    /// Get current experiment
    pub fn get_current_experiment(&self) -> Option<&ExperimentMetadata> {
        self.current_experiment_id.as_ref().and_then(|id| self.experiments.get(id))
    }

    /// Create hyperparameter configuration
    pub fn create_hyperparameter_config(
        &mut self,
        name: String,
        parameters: HashMap<String, serde_json::Value>,
        parent_id: Option<String>,
        notes: Option<String>,
    ) -> String {
        let config_id = Uuid::new_v4().to_string();
        let config_hash = Self::compute_config_hash(&parameters);

        let version = if let Some(parent_id) = &parent_id {
            if let Some(parent) = self.hyperparameter_configs.get(parent_id) {
                parent.version + 1
            } else {
                1
            }
        } else {
            1
        };

        let config = HyperparameterConfig {
            id: config_id.clone(),
            name,
            parameters,
            version,
            parent_id,
            created_at: Utc::now(),
            config_hash,
            notes,
        };

        self.hyperparameter_configs.insert(config_id.clone(), config);
        config_id
    }

    /// Compare hyperparameter configurations
    pub fn compare_hyperparameter_configs(
        &self,
        config_id1: &str,
        config_id2: &str,
    ) -> Result<HyperparameterComparison, String> {
        let config1 = self
            .hyperparameter_configs
            .get(config_id1)
            .ok_or_else(|| format!("Configuration {} not found", config_id1))?;
        let config2 = self
            .hyperparameter_configs
            .get(config_id2)
            .ok_or_else(|| format!("Configuration {} not found", config_id2))?;

        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut modified = Vec::new();
        let mut unchanged = Vec::new();

        // Find added parameters
        for key in config2.parameters.keys() {
            if !config1.parameters.contains_key(key) {
                added.push(key.clone());
            }
        }

        // Find removed and modified parameters
        for (key, value1) in &config1.parameters {
            if let Some(value2) = config2.parameters.get(key) {
                if value1 != value2 {
                    modified.push(ParameterChange {
                        key: key.clone(),
                        old_value: value1.clone(),
                        new_value: value2.clone(),
                    });
                } else {
                    unchanged.push(key.clone());
                }
            } else {
                removed.push(key.clone());
            }
        }

        Ok(HyperparameterComparison {
            config1_id: config_id1.to_string(),
            config2_id: config_id2.to_string(),
            added,
            removed,
            modified,
            unchanged,
        })
    }

    /// Create model lineage
    pub fn create_model_lineage(
        &mut self,
        model_name: String,
        architecture: String,
        training_experiment_id: String,
        training_dataset_id: String,
        parent_id: Option<String>,
        artifacts: Vec<ModelArtifact>,
        metrics: HashMap<String, f64>,
        size_info: ModelSizeInfo,
        provenance: ModelProvenance,
    ) -> String {
        let model_id = Uuid::new_v4().to_string();

        let version = if let Some(parent_id) = &parent_id {
            if let Some(parent) = self.model_lineage.get(parent_id) {
                parent.version + 1
            } else {
                1
            }
        } else {
            1
        };

        let lineage = ModelLineage {
            model_id: model_id.clone(),
            model_name,
            version,
            parent_id,
            training_experiment_id,
            architecture,
            training_dataset_id,
            artifacts,
            metrics,
            size_info,
            created_at: Utc::now(),
            provenance,
        };

        self.model_lineage.insert(model_id.clone(), lineage);
        model_id
    }

    /// Create A/B test
    pub fn create_ab_test(
        &mut self,
        test_name: String,
        description: String,
        control_experiment_id: String,
        treatment_experiment_ids: Vec<String>,
        traffic_allocation: HashMap<String, f32>,
        success_metrics: Vec<String>,
        significance_threshold: f32,
        minimum_sample_size: u64,
        test_duration: chrono::Duration,
    ) -> String {
        let test_id = Uuid::new_v4().to_string();

        let test_config = ABTestConfig {
            test_id: test_id.clone(),
            test_name,
            description,
            control_experiment_id,
            treatment_experiment_ids,
            traffic_allocation,
            success_metrics,
            significance_threshold,
            minimum_sample_size,
            test_duration,
            status: ABTestStatus::Planning,
            results: None,
        };

        self.ab_tests.insert(test_id.clone(), test_config);
        test_id
    }

    /// Update A/B test results
    pub fn update_ab_test_results(
        &mut self,
        test_id: &str,
        results: ABTestResults,
    ) -> Result<(), String> {
        if let Some(test) = self.ab_tests.get_mut(test_id) {
            test.results = Some(results);
            test.status = ABTestStatus::Completed;
            Ok(())
        } else {
            Err(format!("A/B test {} not found", test_id))
        }
    }

    /// Get experiment by ID
    pub fn get_experiment(&self, experiment_id: &str) -> Option<&ExperimentMetadata> {
        self.experiments.get(experiment_id)
    }

    /// Get hyperparameter configuration by ID
    pub fn get_hyperparameter_config(&self, config_id: &str) -> Option<&HyperparameterConfig> {
        self.hyperparameter_configs.get(config_id)
    }

    /// Get model lineage by ID
    pub fn get_model_lineage(&self, model_id: &str) -> Option<&ModelLineage> {
        self.model_lineage.get(model_id)
    }

    /// Get A/B test by ID
    pub fn get_ab_test(&self, test_id: &str) -> Option<&ABTestConfig> {
        self.ab_tests.get(test_id)
    }

    /// List experiments with filters
    pub fn list_experiments(&self, filters: Option<ExperimentFilters>) -> Vec<&ExperimentMetadata> {
        let mut experiments: Vec<&ExperimentMetadata> = self.experiments.values().collect();

        if let Some(filters) = filters {
            experiments.retain(|exp| {
                let mut include = true;

                if let Some(status) = &filters.status {
                    include = include && exp.status == *status;
                }

                if let Some(creator) = &filters.creator {
                    include = include && exp.creator == *creator;
                }

                if let Some(tags) = &filters.tags {
                    include = include && tags.iter().all(|tag| exp.tags.contains(tag));
                }

                if let Some(date_range) = &filters.date_range {
                    include = include
                        && exp.created_at >= date_range.start
                        && exp.created_at <= date_range.end;
                }

                include
            });
        }

        experiments.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        experiments
    }

    /// Generate experiment report
    pub fn generate_experiment_report(
        &self,
        experiment_id: &str,
    ) -> Result<ExperimentReport, String> {
        let experiment = self
            .get_experiment(experiment_id)
            .ok_or_else(|| format!("Experiment {} not found", experiment_id))?;

        // Find related hyperparameter configs
        let related_configs: Vec<&HyperparameterConfig> = self
            .hyperparameter_configs
            .values()
            .filter(|config| {
                // This is a simplified check - in reality, you'd have better linkage
                config.name.contains(&experiment.name)
                    || experiment.custom_metadata.contains_key("config_id")
                        && experiment.custom_metadata.get("config_id") == Some(&config.id)
            })
            .collect();

        // Find related models
        let related_models: Vec<&ModelLineage> = self
            .model_lineage
            .values()
            .filter(|model| model.training_experiment_id == experiment_id)
            .collect();

        // Find related A/B tests
        let related_ab_tests: Vec<&ABTestConfig> = self
            .ab_tests
            .values()
            .filter(|test| {
                test.control_experiment_id == experiment_id
                    || test.treatment_experiment_ids.contains(&experiment_id.to_string())
            })
            .collect();

        Ok(ExperimentReport {
            experiment: experiment.clone(),
            related_configs: related_configs.into_iter().cloned().collect(),
            related_models: related_models.into_iter().cloned().collect(),
            related_ab_tests: related_ab_tests.into_iter().cloned().collect(),
        })
    }

    /// Compute configuration hash
    fn compute_config_hash(parameters: &HashMap<String, serde_json::Value>) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        let serialized = serde_json::to_string(parameters).unwrap_or_default();
        serialized.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

/// Experiment filters
#[derive(Debug, Clone)]
pub struct ExperimentFilters {
    pub status: Option<ExperimentStatus>,
    pub creator: Option<String>,
    pub tags: Option<Vec<String>>,
    pub date_range: Option<DateRange>,
}

/// Date range filter
#[derive(Debug, Clone)]
pub struct DateRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

/// Hyperparameter comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterComparison {
    pub config1_id: String,
    pub config2_id: String,
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub modified: Vec<ParameterChange>,
    pub unchanged: Vec<String>,
}

/// Parameter change information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterChange {
    pub key: String,
    pub old_value: serde_json::Value,
    pub new_value: serde_json::Value,
}

/// Experiment report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentReport {
    pub experiment: ExperimentMetadata,
    pub related_configs: Vec<HyperparameterConfig>,
    pub related_models: Vec<ModelLineage>,
    pub related_ab_tests: Vec<ABTestConfig>,
}

impl Default for ExperimentManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experiment_creation() {
        let mut manager = ExperimentManager::new();

        let environment = EnvironmentInfo {
            python_version: "3.9.0".to_string(),
            cuda_version: Some("11.4".to_string()),
            pytorch_version: "1.12.0".to_string(),
            hardware: HardwareInfo {
                cpu_model: "Intel Core i7".to_string(),
                cpu_cores: 8,
                total_ram_gb: 32.0,
                gpus: vec![GPUInfo {
                    model: "NVIDIA RTX 3080".to_string(),
                    memory_gb: 10.0,
                    compute_capability: Some("8.6".to_string()),
                }],
            },
            packages: HashMap::new(),
            system: SystemInfo {
                os: "Linux".to_string(),
                os_version: "Ubuntu 20.04".to_string(),
                architecture: "x86_64".to_string(),
                hostname: "test-machine".to_string(),
            },
        };

        let experiment_id = manager.create_experiment(
            "Test Experiment".to_string(),
            "A test experiment".to_string(),
            vec!["test".to_string()],
            "test_user".to_string(),
            None,
            environment,
        );

        assert!(manager.get_experiment(&experiment_id).is_some());
        let experiment = manager.get_experiment(&experiment_id).unwrap();
        assert_eq!(experiment.name, "Test Experiment");
        assert_eq!(experiment.version, 1);
        assert_eq!(experiment.status, ExperimentStatus::Planning);
    }

    #[test]
    fn test_hyperparameter_config_creation() {
        let mut manager = ExperimentManager::new();

        let mut parameters = HashMap::new();
        parameters.insert("learning_rate".to_string(), serde_json::json!(0.001));
        parameters.insert("batch_size".to_string(), serde_json::json!(32));

        let config_id = manager.create_hyperparameter_config(
            "Test Config".to_string(),
            parameters,
            None,
            Some("Test configuration".to_string()),
        );

        assert!(manager.get_hyperparameter_config(&config_id).is_some());
        let config = manager.get_hyperparameter_config(&config_id).unwrap();
        assert_eq!(config.name, "Test Config");
        assert_eq!(config.version, 1);
    }

    #[test]
    fn test_hyperparameter_comparison() {
        let mut manager = ExperimentManager::new();

        let mut parameters1 = HashMap::new();
        parameters1.insert("learning_rate".to_string(), serde_json::json!(0.001));
        parameters1.insert("batch_size".to_string(), serde_json::json!(32));

        let mut parameters2 = HashMap::new();
        parameters2.insert("learning_rate".to_string(), serde_json::json!(0.002));
        parameters2.insert("batch_size".to_string(), serde_json::json!(32));
        parameters2.insert("weight_decay".to_string(), serde_json::json!(0.01));

        let config_id1 =
            manager.create_hyperparameter_config("Config 1".to_string(), parameters1, None, None);

        let config_id2 =
            manager.create_hyperparameter_config("Config 2".to_string(), parameters2, None, None);

        let comparison = manager.compare_hyperparameter_configs(&config_id1, &config_id2).unwrap();

        assert_eq!(comparison.added.len(), 1);
        assert!(comparison.added.contains(&"weight_decay".to_string()));
        assert_eq!(comparison.modified.len(), 1);
        assert_eq!(comparison.modified[0].key, "learning_rate");
        assert_eq!(comparison.unchanged.len(), 1);
        assert!(comparison.unchanged.contains(&"batch_size".to_string()));
    }

    #[test]
    fn test_ab_test_creation() {
        let mut manager = ExperimentManager::new();

        let mut traffic_allocation = HashMap::new();
        traffic_allocation.insert("control".to_string(), 0.5);
        traffic_allocation.insert("treatment".to_string(), 0.5);

        let test_id = manager.create_ab_test(
            "Test A/B Test".to_string(),
            "Testing model performance".to_string(),
            "control_exp_id".to_string(),
            vec!["treatment_exp_id".to_string()],
            traffic_allocation,
            vec!["accuracy".to_string()],
            0.05,
            1000,
            chrono::Duration::days(7),
        );

        assert!(manager.get_ab_test(&test_id).is_some());
        let test = manager.get_ab_test(&test_id).unwrap();
        assert_eq!(test.test_name, "Test A/B Test");
        assert_eq!(test.status, ABTestStatus::Planning);
    }

    #[test]
    fn test_experiment_filters() {
        let mut manager = ExperimentManager::new();

        let environment = EnvironmentInfo {
            python_version: "3.9.0".to_string(),
            cuda_version: Some("11.4".to_string()),
            pytorch_version: "1.12.0".to_string(),
            hardware: HardwareInfo {
                cpu_model: "Intel Core i7".to_string(),
                cpu_cores: 8,
                total_ram_gb: 32.0,
                gpus: vec![],
            },
            packages: HashMap::new(),
            system: SystemInfo {
                os: "Linux".to_string(),
                os_version: "Ubuntu 20.04".to_string(),
                architecture: "x86_64".to_string(),
                hostname: "test-machine".to_string(),
            },
        };

        let exp_id1 = manager.create_experiment(
            "Experiment 1".to_string(),
            "Test 1".to_string(),
            vec!["test".to_string()],
            "user1".to_string(),
            None,
            environment.clone(),
        );

        let _exp_id2 = manager.create_experiment(
            "Experiment 2".to_string(),
            "Test 2".to_string(),
            vec!["prod".to_string()],
            "user2".to_string(),
            None,
            environment,
        );

        manager.update_experiment_status(&exp_id1, ExperimentStatus::Completed).unwrap();

        let filters = ExperimentFilters {
            status: Some(ExperimentStatus::Completed),
            creator: None,
            tags: None,
            date_range: None,
        };

        let filtered_experiments = manager.list_experiments(Some(filters));
        assert_eq!(filtered_experiments.len(), 1);
        assert_eq!(filtered_experiments[0].id, exp_id1);
    }
}
