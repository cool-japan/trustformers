//! Integration between versioning and A/B testing systems

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use super::{ModelVersionManager, VersionedModel};
use crate::ab_testing::{ABTestManager, ExperimentConfig, MetricType, MetricValue, Variant};

/// Enhanced A/B testing manager with versioning integration
pub struct VersionedABTestManager {
    version_manager: Arc<ModelVersionManager>,
    ab_test_manager: Arc<ABTestManager>,
    active_experiments: tokio::sync::RwLock<HashMap<String, VersionedExperiment>>,
}

impl VersionedABTestManager {
    /// Create a new versioned A/B test manager
    pub fn new(version_manager: Arc<ModelVersionManager>) -> Self {
        Self {
            version_manager,
            ab_test_manager: Arc::new(ABTestManager::new()),
            active_experiments: tokio::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Create an A/B test between model versions
    pub async fn create_version_experiment(
        &self,
        config: VersionExperimentConfig,
    ) -> Result<String> {
        // Validate that all versions exist
        let control_version = self
            .version_manager
            .get_version(config.control_version_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Control version not found"))?;

        let mut treatment_versions = Vec::new();
        for &version_id in &config.treatment_version_ids {
            let version = self
                .version_manager
                .get_version(version_id)
                .await?
                .ok_or_else(|| anyhow::anyhow!("Treatment version {} not found", version_id))?;
            treatment_versions.push(version);
        }

        // Create A/B test variants
        let control_variant = Variant::new("control", &control_version.qualified_name());

        let treatment_variants: Vec<Variant> = treatment_versions
            .iter()
            .enumerate()
            .map(|(i, v)| Variant::new(&format!("treatment_{}", i), &v.qualified_name()))
            .collect();

        // Create experiment config
        let experiment_config = ExperimentConfig {
            name: config.name.clone(),
            description: config.description.clone(),
            control_variant,
            treatment_variants,
            traffic_percentage: config.traffic_percentage,
            min_sample_size: config.min_sample_size,
            max_duration_hours: config.max_duration_hours,
        };

        // Create the experiment
        let experiment_id = self.ab_test_manager.create_experiment(experiment_config)?;

        // Track versioned experiment
        let versioned_experiment = VersionedExperiment {
            experiment_id: experiment_id.clone(),
            model_name: control_version.model_name().to_string(),
            control_version_id: config.control_version_id,
            treatment_version_ids: config.treatment_version_ids.clone(),
            config: config.clone(),
            started_at: Utc::now(),
            status: VersionedExperimentStatus::Running,
            metrics_collected: HashMap::new(),
        };

        {
            let mut experiments = self.active_experiments.write().await;
            experiments.insert(experiment_id.clone(), versioned_experiment);
        }

        tracing::info!(
            "Created versioned A/B test: {} ({})",
            config.name,
            experiment_id
        );
        Ok(experiment_id)
    }

    /// Route a request to the appropriate model version
    pub async fn route_request(
        &self,
        experiment_id: &str,
        user_id: &str,
    ) -> Result<ModelRoutingResult> {
        // Get the variant from A/B test manager
        let variant = self.ab_test_manager.route_request(experiment_id, user_id)?;

        // Get the versioned experiment
        let experiments = self.active_experiments.read().await;
        let versioned_experiment = experiments
            .get(experiment_id)
            .ok_or_else(|| anyhow::anyhow!("Versioned experiment not found"))?;

        // Map variant to version ID
        let version_id = if variant.name() == "control" {
            versioned_experiment.control_version_id
        } else {
            // Parse treatment index from variant name
            let treatment_index = variant
                .name()
                .strip_prefix("treatment_")
                .and_then(|s| s.parse::<usize>().ok())
                .ok_or_else(|| anyhow::anyhow!("Invalid treatment variant name"))?;

            *versioned_experiment
                .treatment_version_ids
                .get(treatment_index)
                .ok_or_else(|| anyhow::anyhow!("Treatment index out of bounds"))?
        };

        // Get the actual model version
        let model_version = self
            .version_manager
            .get_version(version_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Model version not found"))?;

        Ok(ModelRoutingResult {
            experiment_id: experiment_id.to_string(),
            variant: variant.clone(),
            version_id,
            model_version,
            user_id: user_id.to_string(),
        })
    }

    /// Record metrics for a versioned experiment
    pub async fn record_version_metric(
        &self,
        experiment_id: &str,
        user_id: &str,
        metric_type: VersionMetricType,
        value: f64,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<()> {
        // Get routing information to determine variant
        let routing_result = self.route_request(experiment_id, user_id).await?;

        // Convert to A/B test metric
        let ab_metric_type = match &metric_type {
            VersionMetricType::Latency => MetricType::Latency,
            VersionMetricType::Accuracy => MetricType::Accuracy,
            VersionMetricType::Throughput => MetricType::Throughput,
            VersionMetricType::ErrorRate => MetricType::ErrorRate,
            VersionMetricType::MemoryUsage => MetricType::MemoryUsage,
            VersionMetricType::CustomMetric(name) => MetricType::Custom(name.clone()),
        };

        let ab_metric_value = match metric_type {
            VersionMetricType::Latency => MetricValue::Duration(value as u64),
            _ => MetricValue::Numeric(value),
        };

        // Record in A/B test manager
        self.ab_test_manager.record_metric(
            experiment_id,
            &routing_result.variant,
            ab_metric_type,
            ab_metric_value,
        )?;

        // Update versioned experiment metrics
        {
            let mut experiments = self.active_experiments.write().await;
            if let Some(experiment) = experiments.get_mut(experiment_id) {
                let metric_key = format!("{}:{}", routing_result.variant.name(), metric_type);
                experiment.metrics_collected.entry(metric_key).or_default().push(
                    VersionMetricRecord {
                        value,
                        timestamp: Utc::now(),
                        user_id: user_id.to_string(),
                        metadata: metadata.unwrap_or_default(),
                    },
                );
            }
        }

        Ok(())
    }

    /// Analyze experiment results with version context
    pub async fn analyze_version_experiment(
        &self,
        experiment_id: &str,
    ) -> Result<VersionExperimentResult> {
        // Get A/B test results
        let ab_result = self.ab_test_manager.analyze_experiment(experiment_id)?;

        // Get versioned experiment
        let experiments = self.active_experiments.read().await;
        let versioned_experiment = experiments
            .get(experiment_id)
            .ok_or_else(|| anyhow::anyhow!("Versioned experiment not found"))?;

        // Get version details
        let control_version = self
            .version_manager
            .get_version(versioned_experiment.control_version_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Control version not found"))?;

        let mut treatment_versions = Vec::new();
        for &version_id in &versioned_experiment.treatment_version_ids {
            let version = self
                .version_manager
                .get_version(version_id)
                .await?
                .ok_or_else(|| anyhow::anyhow!("Treatment version not found"))?;
            treatment_versions.push(version);
        }

        // Create version-enhanced result
        Ok(VersionExperimentResult {
            experiment_id: experiment_id.to_string(),
            model_name: versioned_experiment.model_name.clone(),
            control_version,
            treatment_versions,
            ab_test_result: ab_result,
            experiment_duration: Utc::now() - versioned_experiment.started_at,
            total_requests: versioned_experiment
                .metrics_collected
                .values()
                .map(|records| records.len())
                .sum(),
            version_performance_comparison: self
                .compare_version_performance(versioned_experiment)
                .await?,
        })
    }

    /// Promote winning version based on experiment results
    pub async fn promote_winning_version(&self, experiment_id: &str) -> Result<PromotionResult> {
        let result = self.analyze_version_experiment(experiment_id).await?;

        // Determine winner based on A/B test recommendation
        let winning_version_id = match &result.ab_test_result.recommendation {
            crate::ab_testing::TestRecommendation::AdoptTreatment { variant, .. } => {
                // Find the treatment version ID
                let treatment_index = variant
                    .strip_prefix("treatment_")
                    .and_then(|s| s.parse::<usize>().ok())
                    .ok_or_else(|| anyhow::anyhow!("Invalid treatment variant name"))?;

                result
                    .treatment_versions
                    .get(treatment_index)
                    .map(|v| v.id())
                    .ok_or_else(|| anyhow::anyhow!("Treatment index out of bounds"))?
            },
            crate::ab_testing::TestRecommendation::KeepControl { .. } => {
                result.control_version.id()
            },
            _ => {
                return Ok(PromotionResult {
                    promoted: false,
                    version_id: None,
                    reason: "No clear winner determined".to_string(),
                });
            },
        };

        // Promote the winning version to production
        self.version_manager.promote_to_production(winning_version_id).await?;

        // Mark experiment as concluded
        {
            let mut experiments = self.active_experiments.write().await;
            if let Some(experiment) = experiments.get_mut(experiment_id) {
                experiment.status = VersionedExperimentStatus::Concluded;
            }
        }

        Ok(PromotionResult {
            promoted: true,
            version_id: Some(winning_version_id),
            reason: "Version promoted based on A/B test results".to_string(),
        })
    }

    /// List active versioned experiments
    pub async fn list_experiments(&self) -> Result<Vec<VersionedExperiment>> {
        let experiments = self.active_experiments.read().await;
        Ok(experiments.values().cloned().collect())
    }

    /// Stop a versioned experiment
    pub async fn stop_experiment(&self, experiment_id: &str) -> Result<()> {
        let mut experiments = self.active_experiments.write().await;
        if let Some(experiment) = experiments.get_mut(experiment_id) {
            experiment.status = VersionedExperimentStatus::Stopped;
        }
        Ok(())
    }

    // Helper methods

    async fn compare_version_performance(
        &self,
        experiment: &VersionedExperiment,
    ) -> Result<HashMap<String, VersionPerformanceMetrics>> {
        let mut comparison = HashMap::new();

        for (metric_key, records) in &experiment.metrics_collected {
            let parts: Vec<&str> = metric_key.split(':').collect();
            if parts.len() == 2 {
                let variant_name = parts[0];
                let metric_type = parts[1];

                let values: Vec<f64> = records.iter().map(|r| r.value).collect();

                if !values.is_empty() {
                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                        / values.len() as f64;
                    let std_dev = variance.sqrt();
                    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                    let metrics = VersionPerformanceMetrics {
                        metric_type: metric_type.to_string(),
                        sample_count: values.len(),
                        mean,
                        std_dev,
                        min,
                        max,
                        p95: calculate_percentile(&values, 0.95),
                        p99: calculate_percentile(&values, 0.99),
                    };

                    comparison.insert(format!("{}:{}", variant_name, metric_type), metrics);
                }
            }
        }

        Ok(comparison)
    }
}

/// Calculate percentile from a list of values
fn calculate_percentile(values: &[f64], percentile: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let index = (percentile * (sorted.len() - 1) as f64) as usize;
    sorted[index]
}

/// Configuration for version-based A/B test experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionExperimentConfig {
    pub name: String,
    pub description: String,
    pub control_version_id: Uuid,
    pub treatment_version_ids: Vec<Uuid>,
    pub traffic_percentage: f64,
    pub min_sample_size: usize,
    pub max_duration_hours: u64,
}

/// Status of a versioned experiment
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VersionedExperimentStatus {
    Running,
    Stopped,
    Concluded,
    Failed,
}

/// A versioned A/B test experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedExperiment {
    pub experiment_id: String,
    pub model_name: String,
    pub control_version_id: Uuid,
    pub treatment_version_ids: Vec<Uuid>,
    pub config: VersionExperimentConfig,
    pub started_at: DateTime<Utc>,
    pub status: VersionedExperimentStatus,
    pub metrics_collected: HashMap<String, Vec<VersionMetricRecord>>,
}

/// Metric types for version experiments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VersionMetricType {
    Latency,
    Accuracy,
    Throughput,
    ErrorRate,
    MemoryUsage,
    CustomMetric(String),
}

impl std::fmt::Display for VersionMetricType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VersionMetricType::Latency => write!(f, "latency"),
            VersionMetricType::Accuracy => write!(f, "accuracy"),
            VersionMetricType::Throughput => write!(f, "throughput"),
            VersionMetricType::ErrorRate => write!(f, "error_rate"),
            VersionMetricType::MemoryUsage => write!(f, "memory_usage"),
            VersionMetricType::CustomMetric(name) => write!(f, "{}", name),
        }
    }
}

/// Metric record for version experiments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionMetricRecord {
    pub value: f64,
    pub timestamp: DateTime<Utc>,
    pub user_id: String,
    pub metadata: HashMap<String, String>,
}

/// Result of routing a request to a model version
#[derive(Debug, Clone)]
pub struct ModelRoutingResult {
    pub experiment_id: String,
    pub variant: Variant,
    pub version_id: Uuid,
    pub model_version: VersionedModel,
    pub user_id: String,
}

/// Result of analyzing a version experiment
#[derive(Debug, Clone)]
pub struct VersionExperimentResult {
    pub experiment_id: String,
    pub model_name: String,
    pub control_version: VersionedModel,
    pub treatment_versions: Vec<VersionedModel>,
    pub ab_test_result: crate::ab_testing::TestResult,
    pub experiment_duration: chrono::Duration,
    pub total_requests: usize,
    pub version_performance_comparison: HashMap<String, VersionPerformanceMetrics>,
}

/// Performance metrics for a version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionPerformanceMetrics {
    pub metric_type: String,
    pub sample_count: usize,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub p95: f64,
    pub p99: f64,
}

/// Result of promoting a version
#[derive(Debug, Clone)]
pub struct PromotionResult {
    pub promoted: bool,
    pub version_id: Option<Uuid>,
    pub reason: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::versioning::{
        metadata::{ModelMetadata, ModelTag},
        storage::InMemoryStorage,
    };

    async fn create_test_version_manager() -> Arc<ModelVersionManager> {
        let storage = Arc::new(InMemoryStorage::new());
        Arc::new(ModelVersionManager::new(storage))
    }

    async fn create_test_version(manager: &ModelVersionManager, name: &str, version: &str) -> Uuid {
        let metadata = ModelMetadata::builder()
            .description(format!("Test model {}", name))
            .created_by("test_user".to_string())
            .model_type("transformer".to_string())
            .tag(ModelTag::new("test"))
            .build();

        manager.register_version(name, version, metadata, vec![]).await.unwrap()
    }

    #[tokio::test]
    async fn test_version_experiment_creation() {
        let version_manager = create_test_version_manager().await;
        let ab_manager = VersionedABTestManager::new(version_manager.clone());

        // Create test versions
        let control_id = create_test_version(&version_manager, "test_model", "1.0.0").await;
        let treatment_id = create_test_version(&version_manager, "test_model", "1.1.0").await;

        // Create experiment
        let config = VersionExperimentConfig {
            name: "Model v1.1 Test".to_string(),
            description: "Testing improved model".to_string(),
            control_version_id: control_id,
            treatment_version_ids: vec![treatment_id],
            traffic_percentage: 50.0,
            min_sample_size: 100,
            max_duration_hours: 24,
        };

        let experiment_id = ab_manager.create_version_experiment(config).await.unwrap();
        assert!(!experiment_id.is_empty());

        // Check that experiment is tracked
        let experiments = ab_manager.list_experiments().await.unwrap();
        assert_eq!(experiments.len(), 1);
        assert_eq!(experiments[0].experiment_id, experiment_id);
    }

    #[tokio::test]
    async fn test_request_routing() {
        let version_manager = create_test_version_manager().await;
        let ab_manager = VersionedABTestManager::new(version_manager.clone());

        let control_id = create_test_version(&version_manager, "test_model", "1.0.0").await;
        let treatment_id = create_test_version(&version_manager, "test_model", "1.1.0").await;

        let config = VersionExperimentConfig {
            name: "Routing Test".to_string(),
            description: "Test routing".to_string(),
            control_version_id: control_id,
            treatment_version_ids: vec![treatment_id],
            traffic_percentage: 100.0,
            min_sample_size: 10,
            max_duration_hours: 1,
        };

        let experiment_id = ab_manager.create_version_experiment(config).await.unwrap();

        // Route a request
        let routing_result = ab_manager.route_request(&experiment_id, "test_user").await.unwrap();
        assert_eq!(routing_result.experiment_id, experiment_id);
        assert_eq!(routing_result.user_id, "test_user");
        assert!(
            routing_result.version_id == control_id || routing_result.version_id == treatment_id
        );
    }

    #[tokio::test]
    async fn test_metric_recording() {
        let version_manager = create_test_version_manager().await;
        let ab_manager = VersionedABTestManager::new(version_manager.clone());

        let control_id = create_test_version(&version_manager, "test_model", "1.0.0").await;
        let treatment_id = create_test_version(&version_manager, "test_model", "1.1.0").await;

        let config = VersionExperimentConfig {
            name: "Metrics Test".to_string(),
            description: "Test metrics".to_string(),
            control_version_id: control_id,
            treatment_version_ids: vec![treatment_id],
            traffic_percentage: 100.0,
            min_sample_size: 10,
            max_duration_hours: 1,
        };

        let experiment_id = ab_manager.create_version_experiment(config).await.unwrap();

        // Record a metric
        ab_manager
            .record_version_metric(
                &experiment_id,
                "test_user",
                VersionMetricType::Latency,
                120.0,
                None,
            )
            .await
            .unwrap();

        // Check that metric was recorded
        let experiments = ab_manager.list_experiments().await.unwrap();
        let experiment = &experiments[0];
        assert!(!experiment.metrics_collected.is_empty());
    }
}
