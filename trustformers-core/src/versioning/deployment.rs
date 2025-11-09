//! Model deployment management for production environments

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use super::metadata::VersionedModel;
use super::storage::ModelStorage;
use crate::ab_testing::{ABTestManager, ExperimentConfig, Variant};

/// Deployment manager for model versions
pub struct DeploymentManager {
    #[allow(dead_code)] // Reserved for future persistent storage integration
    storage: Arc<dyn ModelStorage>,
    deployments: RwLock<HashMap<String, ActiveDeployment>>,
    deployment_history: RwLock<HashMap<String, Vec<DeploymentEvent>>>,
    ab_test_manager: Arc<ABTestManager>,
}

impl DeploymentManager {
    /// Create a new deployment manager
    pub fn new(storage: Arc<dyn ModelStorage>) -> Self {
        Self {
            storage,
            deployments: RwLock::new(HashMap::new()),
            deployment_history: RwLock::new(HashMap::new()),
            ab_test_manager: Arc::new(ABTestManager::new()),
        }
    }

    /// Deploy a model version to production
    pub async fn deploy_to_production(
        &self,
        version_id: Uuid,
        model: &VersionedModel,
    ) -> Result<String> {
        let deployment_id = format!("{}:production", model.model_name());

        let deployment = ActiveDeployment {
            deployment_id: deployment_id.clone(),
            model_name: model.model_name().to_string(),
            version_id,
            environment: Environment::Production,
            strategy: DeploymentStrategy::BlueGreen,
            status: DeploymentStatus::Deploying,
            traffic_percentage: 100.0,
            deployment_time: Utc::now(),
            health_check_url: None,
            rollback_version: None,
            config_overrides: HashMap::new(),
        };

        // Record deployment event
        self.record_deployment_event(
            &deployment_id,
            DeploymentEvent {
                event_type: DeploymentEventType::Deploy,
                version_id,
                timestamp: Utc::now(),
                message: format!(
                    "Deploying {}:{} to production",
                    model.model_name(),
                    model.version()
                ),
                triggered_by: "system".to_string(),
                metadata: HashMap::new(),
            },
        )
        .await;

        // Update active deployments
        {
            let mut deployments = self.deployments.write().await;
            deployments.insert(deployment_id.clone(), deployment);
        }

        // Mark as active after deployment completes
        tokio::time::sleep(std::time::Duration::from_secs(1)).await; // Simulate deployment time
        self.mark_deployment_active(&deployment_id).await?;

        tracing::info!(
            "Deployed {}:{} to production",
            model.model_name(),
            model.version()
        );
        Ok(deployment_id)
    }

    /// Deploy using specific strategy
    pub async fn deploy_with_strategy(
        &self,
        version_id: Uuid,
        model: &VersionedModel,
        config: DeploymentConfig,
    ) -> Result<String> {
        let _deployment_id = format!("{}:{}", model.model_name(), config.environment);

        match config.strategy {
            DeploymentStrategy::Canary => self.deploy_canary(version_id, model, config).await,
            DeploymentStrategy::BlueGreen => {
                self.deploy_blue_green(version_id, model, config).await
            },
            DeploymentStrategy::RollingUpdate => {
                self.deploy_rolling_update(version_id, model, config).await
            },
            DeploymentStrategy::ABTest => self.deploy_ab_test(version_id, model, config).await,
        }
    }

    /// Rollback to a previous version
    pub async fn rollback(&self, model_name: &str, target_version_id: Uuid) -> Result<()> {
        let deployment_id = format!("{}:production", model_name);

        // Get current deployment
        let current_deployment = {
            let deployments = self.deployments.read().await;
            deployments.get(&deployment_id).cloned()
        };

        if let Some(mut deployment) = current_deployment {
            let previous_version = deployment.version_id;
            deployment.rollback_version = Some(previous_version);
            deployment.version_id = target_version_id;
            deployment.status = DeploymentStatus::RollingBack;

            // Update deployment
            {
                let mut deployments = self.deployments.write().await;
                deployments.insert(deployment_id.clone(), deployment);
            }

            // Record rollback event
            self.record_deployment_event(
                &deployment_id,
                DeploymentEvent {
                    event_type: DeploymentEventType::Rollback,
                    version_id: target_version_id,
                    timestamp: Utc::now(),
                    message: format!(
                        "Rolling back {} from {} to {}",
                        model_name, previous_version, target_version_id
                    ),
                    triggered_by: "user".to_string(),
                    metadata: HashMap::new(),
                },
            )
            .await;

            // Simulate rollback process
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            self.mark_deployment_active(&deployment_id).await?;

            tracing::info!(
                "Rolled back {} to version {}",
                model_name,
                target_version_id
            );
            Ok(())
        } else {
            anyhow::bail!("No active deployment found for model {}", model_name);
        }
    }

    /// Get active deployment for a model
    pub async fn get_active_deployment(
        &self,
        model_name: &str,
    ) -> Result<Option<ActiveDeployment>> {
        let deployment_id = format!("{}:production", model_name);
        let deployments = self.deployments.read().await;
        Ok(deployments.get(&deployment_id).cloned())
    }

    /// List all active deployments
    pub async fn list_deployments(&self) -> Result<Vec<ActiveDeployment>> {
        let deployments = self.deployments.read().await;
        Ok(deployments.values().cloned().collect())
    }

    /// Get deployment history
    pub async fn get_deployment_history(&self, model_name: &str) -> Result<Vec<DeploymentEvent>> {
        let deployment_id = format!("{}:production", model_name);
        let history = self.deployment_history.read().await;
        Ok(history.get(&deployment_id).cloned().unwrap_or_default())
    }

    /// Update traffic percentage for gradual rollouts
    pub async fn update_traffic_percentage(
        &self,
        deployment_id: &str,
        percentage: f64,
    ) -> Result<()> {
        if !(0.0..=100.0).contains(&percentage) {
            anyhow::bail!("Traffic percentage must be between 0 and 100");
        }

        let mut deployments = self.deployments.write().await;
        if let Some(deployment) = deployments.get_mut(deployment_id) {
            deployment.traffic_percentage = percentage;

            self.record_deployment_event(
                deployment_id,
                DeploymentEvent {
                    event_type: DeploymentEventType::TrafficUpdate,
                    version_id: deployment.version_id,
                    timestamp: Utc::now(),
                    message: format!("Updated traffic to {:.1}%", percentage),
                    triggered_by: "user".to_string(),
                    metadata: [("traffic_percentage".to_string(), percentage.into())].into(),
                },
            )
            .await;

            tracing::info!(
                "Updated traffic for {} to {:.1}%",
                deployment_id,
                percentage
            );
            Ok(())
        } else {
            anyhow::bail!("Deployment {} not found", deployment_id);
        }
    }

    /// Health check for a deployment
    pub async fn health_check(&self, deployment_id: &str) -> Result<HealthStatus> {
        let deployments = self.deployments.read().await;
        if let Some(deployment) = deployments.get(deployment_id) {
            // Simulate health check
            let is_healthy = deployment.status == DeploymentStatus::Active;

            Ok(HealthStatus {
                deployment_id: deployment_id.to_string(),
                is_healthy,
                last_check: Utc::now(),
                response_time_ms: 120,
                error_rate_percent: if is_healthy { 0.1 } else { 5.0 },
                metrics: HashMap::new(),
            })
        } else {
            anyhow::bail!("Deployment {} not found", deployment_id);
        }
    }

    /// Get deployment statistics
    pub async fn get_deployment_stats(&self) -> Result<DeploymentStatistics> {
        let deployments = self.deployments.read().await;

        let mut stats = DeploymentStatistics {
            total_deployments: deployments.len(),
            active_deployments: 0,
            failed_deployments: 0,
            deploying_count: 0,
            rolling_back_count: 0,
            environments: HashMap::new(),
        };

        for deployment in deployments.values() {
            match deployment.status {
                DeploymentStatus::Active => stats.active_deployments += 1,
                DeploymentStatus::Failed => stats.failed_deployments += 1,
                DeploymentStatus::Deploying => stats.deploying_count += 1,
                DeploymentStatus::RollingBack => stats.rolling_back_count += 1,
                _ => {},
            }

            let env_name = deployment.environment.to_string();
            *stats.environments.entry(env_name).or_insert(0) += 1;
        }

        Ok(stats)
    }

    // Private helper methods

    async fn deploy_canary(
        &self,
        version_id: Uuid,
        model: &VersionedModel,
        config: DeploymentConfig,
    ) -> Result<String> {
        let deployment_id = format!("{}:canary", model.model_name());

        let deployment = ActiveDeployment {
            deployment_id: deployment_id.clone(),
            model_name: model.model_name().to_string(),
            version_id,
            environment: config.environment,
            strategy: DeploymentStrategy::Canary,
            status: DeploymentStatus::Deploying,
            traffic_percentage: config.initial_traffic_percentage.unwrap_or(5.0),
            deployment_time: Utc::now(),
            health_check_url: config.health_check_url,
            rollback_version: None,
            config_overrides: config.config_overrides,
        };

        {
            let mut deployments = self.deployments.write().await;
            deployments.insert(deployment_id.clone(), deployment);
        }

        // Start with low traffic
        self.mark_deployment_active(&deployment_id).await?;

        tracing::info!(
            "Started canary deployment for {}:{}",
            model.model_name(),
            model.version()
        );
        Ok(deployment_id)
    }

    async fn deploy_blue_green(
        &self,
        version_id: Uuid,
        model: &VersionedModel,
        config: DeploymentConfig,
    ) -> Result<String> {
        let deployment_id = format!("{}:blue-green", model.model_name());

        // Deploy to "blue" environment first
        let deployment = ActiveDeployment {
            deployment_id: deployment_id.clone(),
            model_name: model.model_name().to_string(),
            version_id,
            environment: config.environment,
            strategy: DeploymentStrategy::BlueGreen,
            status: DeploymentStatus::Deploying,
            traffic_percentage: 0.0, // No traffic initially
            deployment_time: Utc::now(),
            health_check_url: config.health_check_url,
            rollback_version: None,
            config_overrides: config.config_overrides,
        };

        {
            let mut deployments = self.deployments.write().await;
            deployments.insert(deployment_id.clone(), deployment);
        }

        // Health check before switching traffic
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        let health = self.health_check(&deployment_id).await?;

        if health.is_healthy {
            // Switch traffic to blue environment
            self.update_traffic_percentage(&deployment_id, 100.0).await?;
            self.mark_deployment_active(&deployment_id).await?;
        } else {
            self.mark_deployment_failed(&deployment_id, "Health check failed").await?;
            anyhow::bail!("Blue-green deployment failed health check");
        }

        tracing::info!(
            "Completed blue-green deployment for {}:{}",
            model.model_name(),
            model.version()
        );
        Ok(deployment_id)
    }

    async fn deploy_rolling_update(
        &self,
        version_id: Uuid,
        model: &VersionedModel,
        config: DeploymentConfig,
    ) -> Result<String> {
        let deployment_id = format!("{}:rolling", model.model_name());

        let deployment = ActiveDeployment {
            deployment_id: deployment_id.clone(),
            model_name: model.model_name().to_string(),
            version_id,
            environment: config.environment,
            strategy: DeploymentStrategy::RollingUpdate,
            status: DeploymentStatus::Deploying,
            traffic_percentage: 0.0,
            deployment_time: Utc::now(),
            health_check_url: config.health_check_url,
            rollback_version: None,
            config_overrides: config.config_overrides,
        };

        {
            let mut deployments = self.deployments.write().await;
            deployments.insert(deployment_id.clone(), deployment);
        }

        // Gradual traffic increase
        let steps = vec![25.0, 50.0, 75.0, 100.0];
        for &percentage in &steps {
            self.update_traffic_percentage(&deployment_id, percentage).await?;
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;

            let health = self.health_check(&deployment_id).await?;
            if !health.is_healthy {
                self.mark_deployment_failed(
                    &deployment_id,
                    "Health check failed during rolling update",
                )
                .await?;
                anyhow::bail!("Rolling update failed at {}% traffic", percentage);
            }
        }

        self.mark_deployment_active(&deployment_id).await?;
        tracing::info!(
            "Completed rolling update for {}:{}",
            model.model_name(),
            model.version()
        );
        Ok(deployment_id)
    }

    async fn deploy_ab_test(
        &self,
        version_id: Uuid,
        model: &VersionedModel,
        config: DeploymentConfig,
    ) -> Result<String> {
        let deployment_id = format!("{}:ab-test", model.model_name());

        // Create A/B test experiment
        let control_variant = Variant::new("control", "current-production");
        let treatment_variant = Variant::new("treatment", &format!("version-{}", version_id));

        let experiment_config = ExperimentConfig {
            name: format!("A/B Test: {}", model.qualified_name()),
            description: format!(
                "Testing new version {} against current production",
                model.version()
            ),
            control_variant,
            treatment_variants: vec![treatment_variant],
            traffic_percentage: config.initial_traffic_percentage.unwrap_or(50.0),
            min_sample_size: config.min_sample_size.unwrap_or(1000),
            max_duration_hours: config.max_duration_hours.unwrap_or(24),
        };

        let experiment_id = self.ab_test_manager.create_experiment(experiment_config)?;

        let deployment = ActiveDeployment {
            deployment_id: deployment_id.clone(),
            model_name: model.model_name().to_string(),
            version_id,
            environment: config.environment,
            strategy: DeploymentStrategy::ABTest,
            status: DeploymentStatus::Active,
            traffic_percentage: config.initial_traffic_percentage.unwrap_or(50.0),
            deployment_time: Utc::now(),
            health_check_url: config.health_check_url,
            rollback_version: None,
            config_overrides: config.config_overrides,
        };

        {
            let mut deployments = self.deployments.write().await;
            deployments.insert(deployment_id.clone(), deployment);
        }

        tracing::info!(
            "Started A/B test deployment for {}:{} (experiment: {})",
            model.model_name(),
            model.version(),
            experiment_id
        );
        Ok(deployment_id)
    }

    async fn mark_deployment_active(&self, deployment_id: &str) -> Result<()> {
        let mut deployments = self.deployments.write().await;
        if let Some(deployment) = deployments.get_mut(deployment_id) {
            deployment.status = DeploymentStatus::Active;

            self.record_deployment_event(
                deployment_id,
                DeploymentEvent {
                    event_type: DeploymentEventType::Activate,
                    version_id: deployment.version_id,
                    timestamp: Utc::now(),
                    message: "Deployment activated".to_string(),
                    triggered_by: "system".to_string(),
                    metadata: HashMap::new(),
                },
            )
            .await;
        }
        Ok(())
    }

    async fn mark_deployment_failed(&self, deployment_id: &str, reason: &str) -> Result<()> {
        let mut deployments = self.deployments.write().await;
        if let Some(deployment) = deployments.get_mut(deployment_id) {
            deployment.status = DeploymentStatus::Failed;

            self.record_deployment_event(
                deployment_id,
                DeploymentEvent {
                    event_type: DeploymentEventType::Fail,
                    version_id: deployment.version_id,
                    timestamp: Utc::now(),
                    message: format!("Deployment failed: {}", reason),
                    triggered_by: "system".to_string(),
                    metadata: [("failure_reason".to_string(), reason.into())].into(),
                },
            )
            .await;
        }
        Ok(())
    }

    async fn record_deployment_event(&self, deployment_id: &str, event: DeploymentEvent) {
        let mut history = self.deployment_history.write().await;
        history.entry(deployment_id.to_string()).or_default().push(event);
    }
}

/// Active deployment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveDeployment {
    pub deployment_id: String,
    pub model_name: String,
    pub version_id: Uuid,
    pub environment: Environment,
    pub strategy: DeploymentStrategy,
    pub status: DeploymentStatus,
    pub traffic_percentage: f64,
    pub deployment_time: DateTime<Utc>,
    pub health_check_url: Option<String>,
    pub rollback_version: Option<Uuid>,
    pub config_overrides: HashMap<String, serde_json::Value>,
}

/// Deployment configuration
#[derive(Debug, Clone)]
pub struct DeploymentConfig {
    pub environment: Environment,
    pub strategy: DeploymentStrategy,
    pub initial_traffic_percentage: Option<f64>,
    pub health_check_url: Option<String>,
    pub config_overrides: HashMap<String, serde_json::Value>,
    pub min_sample_size: Option<usize>,
    pub max_duration_hours: Option<u64>,
}

/// Deployment environments
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Environment {
    Development,
    Testing,
    Staging,
    Production,
    Canary,
}

impl std::fmt::Display for Environment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Environment::Development => write!(f, "development"),
            Environment::Testing => write!(f, "testing"),
            Environment::Staging => write!(f, "staging"),
            Environment::Production => write!(f, "production"),
            Environment::Canary => write!(f, "canary"),
        }
    }
}

/// Deployment strategies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DeploymentStrategy {
    /// Replace all instances at once
    BlueGreen,
    /// Gradual rollout with increasing traffic
    Canary,
    /// Progressive replacement of instances
    RollingUpdate,
    /// A/B testing between versions
    ABTest,
}

/// Deployment status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DeploymentStatus {
    Deploying,
    Active,
    Failed,
    RollingBack,
    Archived,
}

/// Deployment event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentEventType {
    Deploy,
    Activate,
    Fail,
    Rollback,
    TrafficUpdate,
    HealthCheck,
}

/// Deployment event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentEvent {
    pub event_type: DeploymentEventType,
    pub version_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub message: String,
    pub triggered_by: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Health check status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub deployment_id: String,
    pub is_healthy: bool,
    pub last_check: DateTime<Utc>,
    pub response_time_ms: u64,
    pub error_rate_percent: f64,
    pub metrics: HashMap<String, f64>,
}

/// Deployment statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentStatistics {
    pub total_deployments: usize,
    pub active_deployments: usize,
    pub failed_deployments: usize,
    pub deploying_count: usize,
    pub rolling_back_count: usize,
    pub environments: HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::versioning::metadata::ModelMetadata;
    use crate::versioning::storage::InMemoryStorage;

    #[tokio::test]
    async fn test_basic_deployment() {
        let storage = Arc::new(InMemoryStorage::new());
        let manager = DeploymentManager::new(storage);

        let metadata = ModelMetadata::builder()
            .description("Test model".to_string())
            .created_by("test".to_string())
            .model_type("transformer".to_string())
            .build();

        let model = VersionedModel::new(
            "test_model".to_string(),
            "1.0.0".to_string(),
            metadata,
            vec![],
        );

        let deployment_id = manager.deploy_to_production(model.id(), &model).await.unwrap();
        assert!(!deployment_id.is_empty());

        let deployment = manager.get_active_deployment("test_model").await.unwrap();
        assert!(deployment.is_some());
        assert_eq!(deployment.unwrap().status, DeploymentStatus::Active);
    }

    #[tokio::test]
    async fn test_canary_deployment() {
        let storage = Arc::new(InMemoryStorage::new());
        let manager = DeploymentManager::new(storage);

        let metadata = ModelMetadata::builder()
            .description("Test model".to_string())
            .created_by("test".to_string())
            .model_type("transformer".to_string())
            .build();

        let model = VersionedModel::new(
            "test_model".to_string(),
            "1.0.0".to_string(),
            metadata,
            vec![],
        );

        let config = DeploymentConfig {
            environment: Environment::Production,
            strategy: DeploymentStrategy::Canary,
            initial_traffic_percentage: Some(10.0),
            health_check_url: None,
            config_overrides: HashMap::new(),
            min_sample_size: None,
            max_duration_hours: None,
        };

        let deployment_id = manager.deploy_with_strategy(model.id(), &model, config).await.unwrap();
        assert!(!deployment_id.is_empty());

        // Check initial traffic percentage
        let deployments = manager.list_deployments().await.unwrap();
        let canary_deployment = deployments.iter().find(|d| d.deployment_id == deployment_id);
        assert!(canary_deployment.is_some());
        assert_eq!(canary_deployment.unwrap().traffic_percentage, 10.0);
    }

    #[tokio::test]
    async fn test_rollback() {
        let storage = Arc::new(InMemoryStorage::new());
        let manager = DeploymentManager::new(storage);

        let metadata = ModelMetadata::builder()
            .description("Test model".to_string())
            .created_by("test".to_string())
            .model_type("transformer".to_string())
            .build();

        let model = VersionedModel::new(
            "test_model".to_string(),
            "1.0.0".to_string(),
            metadata,
            vec![],
        );

        // Deploy initial version
        manager.deploy_to_production(model.id(), &model).await.unwrap();

        // Create new version
        let new_metadata = ModelMetadata::builder()
            .description("Updated test model".to_string())
            .created_by("test".to_string())
            .model_type("transformer".to_string())
            .build();

        let new_model = VersionedModel::new(
            "test_model".to_string(),
            "1.1.0".to_string(),
            new_metadata,
            vec![],
        );

        // Deploy new version
        manager.deploy_to_production(new_model.id(), &new_model).await.unwrap();

        // Rollback to original version
        manager.rollback("test_model", model.id()).await.unwrap();

        let deployment = manager.get_active_deployment("test_model").await.unwrap();
        assert!(deployment.is_some());
        assert_eq!(deployment.unwrap().version_id, model.id());
    }

    #[tokio::test]
    async fn test_traffic_update() {
        let storage = Arc::new(InMemoryStorage::new());
        let manager = DeploymentManager::new(storage);

        let metadata = ModelMetadata::builder()
            .description("Test model".to_string())
            .created_by("test".to_string())
            .model_type("transformer".to_string())
            .build();

        let model = VersionedModel::new(
            "test_model".to_string(),
            "1.0.0".to_string(),
            metadata,
            vec![],
        );

        let deployment_id = manager.deploy_to_production(model.id(), &model).await.unwrap();

        // Update traffic percentage
        manager.update_traffic_percentage(&deployment_id, 75.0).await.unwrap();

        let deployment = manager.get_active_deployment("test_model").await.unwrap();
        assert!(deployment.is_some());
        assert_eq!(deployment.unwrap().traffic_percentage, 75.0);
    }

    #[tokio::test]
    async fn test_health_check() {
        let storage = Arc::new(InMemoryStorage::new());
        let manager = DeploymentManager::new(storage);

        let metadata = ModelMetadata::builder()
            .description("Test model".to_string())
            .created_by("test".to_string())
            .model_type("transformer".to_string())
            .build();

        let model = VersionedModel::new(
            "test_model".to_string(),
            "1.0.0".to_string(),
            metadata,
            vec![],
        );

        let deployment_id = manager.deploy_to_production(model.id(), &model).await.unwrap();

        let health = manager.health_check(&deployment_id).await.unwrap();
        assert!(health.is_healthy);
        assert_eq!(health.deployment_id, deployment_id);
    }
}
