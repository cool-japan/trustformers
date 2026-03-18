//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use super::functions::ServerlessProviderTrait;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfrastructureHealth {
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub network_latency_ms: f64,
    pub error_rate: f64,
    pub availability: f64,
    pub scaling_events: u32,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Architecture {
    X86_64,
    ARM64,
    Auto,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventSourceMapping {
    pub batch_size: Option<u32>,
    pub maximum_batching_window_in_seconds: Option<u32>,
    pub starting_position: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAlert {
    pub threshold_usd: f64,
    pub period_hours: u32,
    pub notification_endpoint: String,
}
pub struct ServerlessOrchestrator {
    deployments: Arc<RwLock<HashMap<Uuid, ServerlessDeployment>>>,
    providers:
        Arc<RwLock<HashMap<ServerlessProvider, Box<dyn ServerlessProviderTrait + Send + Sync>>>>,
    metrics: Arc<RwLock<HashMap<Uuid, ServerlessMetrics>>>,
    analytics: Arc<RwLock<HashMap<Uuid, DeploymentAnalytics>>>,
    cost_tracker: Arc<RwLock<HashMap<Uuid, f64>>>,
}
impl ServerlessOrchestrator {
    pub fn new() -> Self {
        Self {
            deployments: Arc::new(RwLock::new(HashMap::new())),
            providers: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(HashMap::new())),
            analytics: Arc::new(RwLock::new(HashMap::new())),
            cost_tracker: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    pub async fn register_provider(
        &self,
        provider_type: ServerlessProvider,
        provider: Box<dyn ServerlessProviderTrait + Send + Sync>,
    ) -> Result<()> {
        let mut providers = self.providers.write().await;
        providers.insert(provider_type, provider);
        Ok(())
    }
    pub async fn deploy_function(&self, config: ServerlessConfig) -> Result<Uuid> {
        let deployment_id = Uuid::new_v4();
        let deployment = ServerlessDeployment {
            id: deployment_id,
            config: config.clone(),
            status: DeploymentStatus::Pending,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            function_arn: None,
            function_url: None,
            version: "1.0.0".to_string(),
            last_deployment: None,
            deployment_logs: vec![],
        };
        {
            let mut deployments = self.deployments.write().await;
            deployments.insert(deployment_id, deployment);
        }
        self.update_deployment_status(deployment_id, DeploymentStatus::Deploying)
            .await?;
        let providers = self.providers.read().await;
        if let Some(provider) = providers.get(&config.provider) {
            match provider.deploy(&config).await {
                Ok(deployment_result) => {
                    self.update_deployment_with_result(deployment_id, deployment_result).await?;
                    self.update_deployment_status(deployment_id, DeploymentStatus::Active).await?;
                },
                Err(e) => {
                    self.add_deployment_log(
                        deployment_id,
                        "ERROR".to_string(),
                        format!("Deployment failed: {}", e),
                        Some(serde_json::json!({ "error" : e.to_string() })),
                    )
                    .await?;
                    self.update_deployment_status(deployment_id, DeploymentStatus::Failed).await?;
                    return Err(e);
                },
            }
        } else {
            return Err(anyhow!("Provider not registered: {:?}", config.provider));
        }
        Ok(deployment_id)
    }
    pub async fn update_function(
        &self,
        deployment_id: Uuid,
        config: ServerlessConfig,
    ) -> Result<()> {
        self.update_deployment_status(deployment_id, DeploymentStatus::Updating).await?;
        let providers = self.providers.read().await;
        if let Some(provider) = providers.get(&config.provider) {
            match provider.update(&config).await {
                Ok(deployment_result) => {
                    self.update_deployment_with_result(deployment_id, deployment_result).await?;
                    self.update_deployment_status(deployment_id, DeploymentStatus::Active).await?;
                },
                Err(e) => {
                    self.add_deployment_log(
                        deployment_id,
                        "ERROR".to_string(),
                        format!("Update failed: {}", e),
                        Some(serde_json::json!({ "error" : e.to_string() })),
                    )
                    .await?;
                    self.update_deployment_status(deployment_id, DeploymentStatus::Failed).await?;
                    return Err(e);
                },
            }
        } else {
            return Err(anyhow!("Provider not registered"));
        }
        Ok(())
    }
    pub async fn delete_function(&self, deployment_id: Uuid) -> Result<()> {
        self.update_deployment_status(deployment_id, DeploymentStatus::Deleting).await?;
        let deployment = {
            let deployments = self.deployments.read().await;
            deployments.get(&deployment_id).cloned()
        };
        if let Some(deployment) = deployment {
            let providers = self.providers.read().await;
            if let Some(provider) = providers.get(&deployment.config.provider) {
                match provider.delete(&deployment.config).await {
                    Ok(_) => {
                        self.update_deployment_status(deployment_id, DeploymentStatus::Deleted)
                            .await?;
                    },
                    Err(e) => {
                        self.add_deployment_log(
                            deployment_id,
                            "ERROR".to_string(),
                            format!("Deletion failed: {}", e),
                            Some(serde_json::json!({ "error" : e.to_string() })),
                        )
                        .await?;
                        return Err(e);
                    },
                }
            }
        }
        Ok(())
    }
    pub async fn get_deployment(&self, deployment_id: Uuid) -> Option<ServerlessDeployment> {
        let deployments = self.deployments.read().await;
        deployments.get(&deployment_id).cloned()
    }
    pub async fn list_deployments(&self) -> Vec<ServerlessDeployment> {
        let deployments = self.deployments.read().await;
        deployments.values().cloned().collect()
    }
    pub async fn get_metrics(&self, deployment_id: Uuid) -> Option<ServerlessMetrics> {
        let metrics = self.metrics.read().await;
        metrics.get(&deployment_id).cloned()
    }
    pub async fn invoke_function(
        &self,
        deployment_id: Uuid,
        payload: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let deployment = {
            let deployments = self.deployments.read().await;
            deployments.get(&deployment_id).cloned()
        };
        if let Some(deployment) = deployment {
            let providers = self.providers.read().await;
            if let Some(provider) = providers.get(&deployment.config.provider) {
                return provider.invoke(&deployment.config, payload).await;
            }
        }
        Err(anyhow!("Deployment not found or provider not available"))
    }
    async fn update_deployment_status(
        &self,
        deployment_id: Uuid,
        status: DeploymentStatus,
    ) -> Result<()> {
        let mut deployments = self.deployments.write().await;
        if let Some(deployment) = deployments.get_mut(&deployment_id) {
            deployment.status = status;
            deployment.updated_at = Utc::now();
        }
        Ok(())
    }
    async fn update_deployment_with_result(
        &self,
        deployment_id: Uuid,
        result: DeploymentResult,
    ) -> Result<()> {
        let mut deployments = self.deployments.write().await;
        if let Some(deployment) = deployments.get_mut(&deployment_id) {
            deployment.function_arn = result.function_arn;
            deployment.function_url = result.function_url;
            deployment.last_deployment = Some(Utc::now());
            deployment.updated_at = Utc::now();
        }
        Ok(())
    }
    async fn add_deployment_log(
        &self,
        deployment_id: Uuid,
        level: String,
        message: String,
        details: Option<serde_json::Value>,
    ) -> Result<()> {
        let mut deployments = self.deployments.write().await;
        if let Some(deployment) = deployments.get_mut(&deployment_id) {
            deployment.deployment_logs.push(DeploymentLog {
                timestamp: Utc::now(),
                level,
                message,
                details,
            });
        }
        Ok(())
    }
    pub async fn collect_metrics(&self) -> Result<()> {
        let deployments = self.deployments.read().await;
        let providers = self.providers.read().await;
        for (deployment_id, deployment) in deployments.iter() {
            if let Some(provider) = providers.get(&deployment.config.provider) {
                if let Ok(metrics) = provider.get_metrics(&deployment.config).await {
                    let mut metrics_map = self.metrics.write().await;
                    metrics_map.insert(*deployment_id, metrics);
                }
            }
        }
        Ok(())
    }
    pub async fn optimize_cold_starts(&self, deployment_id: Uuid) -> Result<()> {
        let deployment = {
            let deployments = self.deployments.read().await;
            deployments.get(&deployment_id).cloned()
        };
        if let Some(deployment) = deployment {
            if let Some(cold_start_config) = &deployment.config.cold_start {
                let providers = self.providers.read().await;
                if let Some(provider) = providers.get(&deployment.config.provider) {
                    if cold_start_config.enable_provisioned_concurrency {
                        provider
                            .configure_provisioned_concurrency(
                                &deployment.config,
                                cold_start_config.provisioned_concurrency_count.unwrap_or(1),
                            )
                            .await?;
                    }
                    if let Some(schedule) = &cold_start_config.warmup_schedule {
                        provider.configure_warmup_schedule(&deployment.config, schedule).await?;
                    }
                    if let Some(requests_per_minute) =
                        cold_start_config.keep_warm_requests_per_minute
                    {
                        provider
                            .configure_keep_warm(&deployment.config, requests_per_minute)
                            .await?;
                    }
                }
            }
        }
        Ok(())
    }
    pub async fn track_costs(&self, deployment_id: Uuid) -> Result<f64> {
        let metrics = self.get_metrics(deployment_id).await;
        if let Some(metrics) = metrics {
            let total_cost = metrics.cost_usd;
            {
                let mut cost_tracker = self.cost_tracker.write().await;
                cost_tracker.insert(deployment_id, total_cost);
            }
            let deployment = self.get_deployment(deployment_id).await;
            if let Some(deployment) = deployment {
                if let Some(cost_config) = &deployment.config.cost_optimization {
                    for alert in &cost_config.cost_alerts {
                        if total_cost >= alert.threshold_usd {
                            self.send_cost_alert(deployment_id, total_cost, alert).await?;
                        }
                    }
                }
            }
            Ok(total_cost)
        } else {
            Ok(0.0)
        }
    }
    pub async fn optimize_costs(&self, deployment_id: Uuid) -> Result<CostOptimizationResult> {
        let deployment = {
            let deployments = self.deployments.read().await;
            deployments.get(&deployment_id).cloned()
        };
        if let Some(deployment) = deployment {
            let current_metrics = self.get_metrics(deployment_id).await;
            let mut recommendations = Vec::new();
            let mut potential_savings = 0.0;
            let current_cost = current_metrics.as_ref().map(|m| m.cost_usd).unwrap_or(0.0);
            if let Some(metrics) = current_metrics {
                if metrics.memory_utilization < 0.6 {
                    let new_memory = (deployment.config.memory_mb as f64 * 0.8) as u32;
                    let savings = metrics.cost_usd * 0.2;
                    recommendations.push(format!(
                        "Reduce memory from {}MB to {}MB (potential savings: ${:.2})",
                        deployment.config.memory_mb, new_memory, savings
                    ));
                    potential_savings += savings;
                }
                if let Some(cost_config) = &deployment.config.cost_optimization {
                    if cost_config.architecture == Architecture::X86_64
                        && cost_config.enable_arm_graviton
                    {
                        let savings = metrics.cost_usd * 0.15;
                        recommendations.push(format!(
                            "Switch to ARM64 (Graviton) architecture (potential savings: ${:.2})",
                            savings
                        ));
                        potential_savings += savings;
                    }
                }
                if metrics.cold_starts > metrics.invocations / 10 {
                    recommendations
                        .push("Enable provisioned concurrency to reduce cold starts".to_string());
                }
            }
            let optimization_confidence = if recommendations.len() > 2 { 0.8 } else { 0.6 };
            Ok(CostOptimizationResult {
                deployment_id,
                current_cost,
                potential_savings,
                recommendations,
                optimization_confidence,
            })
        } else {
            Err(anyhow!("Deployment not found"))
        }
    }
    pub async fn get_deployment_analytics(
        &self,
        deployment_id: Uuid,
    ) -> Option<DeploymentAnalytics> {
        let analytics = self.analytics.read().await;
        analytics.get(&deployment_id).cloned()
    }
    pub async fn update_deployment_analytics(&self, deployment_id: Uuid) -> Result<()> {
        let deployment = self.get_deployment(deployment_id).await;
        let metrics = self.get_metrics(deployment_id).await;
        if let (Some(_deployment), Some(metrics)) = (deployment, metrics) {
            let analytics = DeploymentAnalytics {
                deployment_id,
                deployment_duration_seconds: 30,
                deployment_success_rate: if metrics.errors == 0 {
                    1.0
                } else {
                    (metrics.invocations - metrics.errors) as f64 / metrics.invocations as f64
                },
                rollback_count: 0,
                performance_improvement: 0.0,
                cost_impact_usd: metrics.cost_usd,
                user_satisfaction_score: None,
                infrastructure_health: InfrastructureHealth {
                    cpu_utilization: 0.0,
                    memory_utilization: metrics.memory_utilization,
                    network_latency_ms: metrics.duration_ms * 0.1,
                    error_rate: metrics.errors as f64 / metrics.invocations as f64,
                    availability: if metrics.errors == 0 { 99.9 } else { 99.0 },
                    scaling_events: 0,
                },
            };
            let mut analytics_map = self.analytics.write().await;
            analytics_map.insert(deployment_id, analytics);
        }
        Ok(())
    }
    pub async fn generate_optimization_recommendations(
        &self,
        deployment_id: Uuid,
    ) -> Result<Vec<OptimizationRecommendation>> {
        let metrics = self.get_metrics(deployment_id).await;
        let deployment = self.get_deployment(deployment_id).await;
        let mut recommendations = Vec::new();
        if let (Some(metrics), Some(_deployment)) = (metrics, deployment) {
            if metrics.duration_ms > 10000.0 {
                recommendations
                    .push(OptimizationRecommendation {
                        category: "Performance".to_string(),
                        priority: RecommendationPriority::High,
                        description: "Function execution time is high. Consider optimizing code or increasing memory allocation."
                            .to_string(),
                        impact: "Reduce latency by up to 50%".to_string(),
                        effort: RecommendationEffort::Medium,
                    });
            }
            if metrics.cold_starts > metrics.invocations / 5 {
                recommendations
                    .push(OptimizationRecommendation {
                        category: "Cold Start".to_string(),
                        priority: RecommendationPriority::High,
                        description: "High cold start rate detected. Enable provisioned concurrency or optimize initialization."
                            .to_string(),
                        impact: "Reduce cold starts by 80%".to_string(),
                        effort: RecommendationEffort::Low,
                    });
            }
            if metrics.memory_utilization < 0.5 {
                recommendations.push(OptimizationRecommendation {
                    category: "Cost".to_string(),
                    priority: RecommendationPriority::Medium,
                    description: format!(
                        "Memory utilization is low ({}%). Consider reducing memory allocation.",
                        (metrics.memory_utilization * 100.0) as u32
                    ),
                    impact: "Reduce costs by 20-30%".to_string(),
                    effort: RecommendationEffort::Low,
                });
            }
            if metrics.errors > metrics.invocations / 20 {
                recommendations
                    .push(OptimizationRecommendation {
                        category: "Reliability".to_string(),
                        priority: RecommendationPriority::Critical,
                        description: "High error rate detected. Review function logs and implement error handling."
                            .to_string(),
                        impact: "Improve reliability and user experience".to_string(),
                        effort: RecommendationEffort::High,
                    });
            }
        }
        Ok(recommendations)
    }
    async fn send_cost_alert(
        &self,
        deployment_id: Uuid,
        cost: f64,
        alert: &CostAlert,
    ) -> Result<()> {
        tracing::warn!(
            "Cost alert triggered for deployment {}: ${} >= ${} threshold",
            deployment_id,
            cost,
            alert.threshold_usd
        );
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerType {
    Http,
    ApiGateway,
    S3,
    DynamoDB,
    Kinesis,
    SQS,
    EventBridge,
    CloudWatch,
    PubSub,
    ServiceBus,
    Custom(String),
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PackageType {
    Zip,
    Image,
    Source,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdStartConfig {
    pub enable_provisioned_concurrency: bool,
    pub provisioned_concurrency_count: Option<u32>,
    pub warmup_schedule: Option<String>,
    pub warmup_endpoint: Option<String>,
    pub keep_warm_requests_per_minute: Option<u32>,
    pub pre_initialization_handler: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub category: String,
    pub priority: RecommendationPriority,
    pub description: String,
    pub impact: String,
    pub effort: RecommendationEffort,
}
pub struct AwsLambdaProvider {
    pub(super) lambda_client: Option<aws_sdk_lambda::Client>,
    _cloudwatch_client: Option<aws_sdk_cloudwatch::Client>,
    _region: String,
}
impl AwsLambdaProvider {
    pub fn new(region: String) -> Self {
        Self {
            lambda_client: None,
            _cloudwatch_client: None,
            _region: region,
        }
    }
    pub async fn with_aws_config(mut self) -> Result<Self> {
        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        self.lambda_client = Some(aws_sdk_lambda::Client::new(&config));
        Ok(self)
    }
    pub fn with_clients(
        lambda_client: aws_sdk_lambda::Client,
        cloudwatch_client: Option<aws_sdk_cloudwatch::Client>,
        region: String,
    ) -> Self {
        Self {
            lambda_client: Some(lambda_client),
            _cloudwatch_client: cloudwatch_client,
            _region: region,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ServerlessProvider {
    AwsLambda,
    GoogleCloudFunctions,
    AzureFunctions,
    Custom(String),
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VpcConfig {
    pub subnet_ids: Vec<String>,
    pub security_group_ids: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UsagePlan {
    OnDemand,
    Reserved,
    Spot,
    Mixed,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentLog {
    pub timestamp: DateTime<Utc>,
    pub level: String,
    pub message: String,
    pub details: Option<serde_json::Value>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerlessMetrics {
    pub invocations: u64,
    pub duration_ms: f64,
    pub errors: u64,
    pub throttles: u64,
    pub concurrent_executions: u64,
    pub memory_utilization: f64,
    pub cold_starts: u64,
    pub billed_duration_ms: f64,
    pub cost_usd: f64,
    pub init_duration_ms: f64,
    pub max_memory_used_mb: u64,
    pub p99_duration_ms: f64,
    pub p95_duration_ms: f64,
    pub p50_duration_ms: f64,
    pub success_rate: f64,
    pub provisioned_concurrency_invocations: u64,
    pub provisioned_concurrency_spillover: u64,
    pub dead_letter_errors: u64,
    pub iterator_age_ms: Option<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedMetrics {
    pub basic_metrics: ServerlessMetrics,
    pub performance_breakdown: PerformanceBreakdown,
    pub cost_breakdown: CostBreakdown,
    pub resource_utilization: ResourceUtilization,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enable_logging: bool,
    pub log_level: String,
    pub enable_tracing: bool,
    pub enable_metrics: bool,
    pub custom_metrics: Vec<String>,
    pub enable_xray: bool,
    pub enable_insights: bool,
    pub log_retention_days: Option<u32>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostOptimizationConfig {
    pub architecture: Architecture,
    pub enable_arm_graviton: bool,
    pub optimize_for_cost: bool,
    pub cost_budget_usd: Option<f64>,
    pub cost_alerts: Vec<CostAlert>,
    pub usage_plan: UsagePlan,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentAnalytics {
    pub deployment_id: Uuid,
    pub deployment_duration_seconds: u64,
    pub deployment_success_rate: f64,
    pub rollback_count: u32,
    pub performance_improvement: f64,
    pub cost_impact_usd: f64,
    pub user_satisfaction_score: Option<f64>,
    pub infrastructure_health: InfrastructureHealth,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerlessConfig {
    pub provider: ServerlessProvider,
    pub function_name: String,
    pub runtime: String,
    pub memory_mb: u32,
    pub timeout_seconds: u32,
    pub environment_variables: HashMap<String, String>,
    pub vpc_config: Option<VpcConfig>,
    pub deployment_package: DeploymentPackage,
    pub triggers: Vec<Trigger>,
    pub scaling: ScalingConfig,
    pub monitoring: MonitoringConfig,
    pub cold_start: Option<ColdStartConfig>,
    pub cost_optimization: Option<CostOptimizationConfig>,
    pub region: Option<String>,
    pub tags: HashMap<String, String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentPackage {
    pub package_type: PackageType,
    pub source_location: String,
    pub handler: String,
    pub layers: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: u64,
    pub network_in_mb: f64,
    pub network_out_mb: f64,
    pub disk_usage_mb: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingConfig {
    pub min_instances: u32,
    pub max_instances: u32,
    pub target_utilization: f64,
    pub scale_down_delay_seconds: u32,
    pub scale_up_delay_seconds: u32,
    pub concurrency_limit: Option<u32>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trigger {
    pub trigger_type: TriggerType,
    pub source_arn: Option<String>,
    pub event_source_mapping: Option<EventSourceMapping>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerlessDeployment {
    pub id: Uuid,
    pub config: ServerlessConfig,
    pub status: DeploymentStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub function_arn: Option<String>,
    pub function_url: Option<String>,
    pub version: String,
    pub last_deployment: Option<DateTime<Utc>>,
    pub deployment_logs: Vec<DeploymentLog>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DeploymentStatus {
    Pending,
    Deploying,
    Active,
    Failed,
    Updating,
    Deleting,
    Deleted,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostOptimizationResult {
    pub deployment_id: Uuid,
    pub current_cost: f64,
    pub potential_savings: f64,
    pub recommendations: Vec<String>,
    pub optimization_confidence: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBreakdown {
    pub initialization_ms: f64,
    pub execution_ms: f64,
    pub overhead_ms: f64,
    pub network_latency_ms: f64,
    pub queue_time_ms: f64,
}
pub struct GoogleCloudFunctionsProvider {
    pub(crate) project_id: String,
}
impl GoogleCloudFunctionsProvider {
    pub fn new(project_id: String) -> Self {
        Self { project_id }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
}
pub struct AzureFunctionsProvider {
    pub(crate) subscription_id: String,
    pub(crate) resource_group: String,
}
impl AzureFunctionsProvider {
    pub fn new(subscription_id: String, resource_group: String) -> Self {
        Self {
            subscription_id,
            resource_group,
        }
    }
}
#[derive(Debug, Clone)]
pub struct DeploymentResult {
    pub function_arn: Option<String>,
    pub function_url: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationEffort {
    Low,
    Medium,
    High,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBreakdown {
    pub compute_cost: f64,
    pub request_cost: f64,
    pub data_transfer_cost: f64,
    pub storage_cost: f64,
    pub additional_services_cost: f64,
}
