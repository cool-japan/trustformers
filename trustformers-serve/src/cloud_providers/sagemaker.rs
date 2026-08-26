//! AWS SageMaker provider.
//!
//! Inference calls `InvokeEndpoint` on the SageMaker **runtime** API; deployment
//! calls `CreateModel` + `CreateEndpointConfig` + `CreateEndpoint` on the
//! SageMaker control-plane API; status calls `DescribeEndpoint`; teardown calls
//! `DeleteEndpoint`. Without SDK clients every operation returns
//! [`CloudProviderError::MissingCredentials`] — no synthesized ARN, no
//! `https://example.com/endpoint`, no invented latency.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Instant;

use anyhow::Result;
use async_trait::async_trait;
use aws_sdk_sagemakerruntime::primitives::Blob;
use chrono::Utc;
use tokio::sync::RwLock;

use super::errors::CloudProviderError;
use super::support::{metadata, performance, require_text, unreported_cost, ObservedStats};
use super::{
    CloudInferenceRequest, CloudInferenceResponse, CloudProvider, CloudProviderType,
    DeploymentStatus, HealthStatus, ModelDeploymentRequest, ModelDeploymentResponse, ModelInfo,
    OutputData, ProviderConfig, ProviderMetrics, ResourceUtilization,
};

const PROVIDER: &str = "AwsSagemakerProvider";

/// AWS SageMaker inference and deployment provider.
#[derive(Debug)]
pub struct AwsSagemakerProvider {
    config: RwLock<Option<ProviderConfig>>,
    runtime_client: Option<aws_sdk_sagemakerruntime::Client>,
    control_client: Option<aws_sdk_sagemaker::Client>,
    stats: ObservedStats,
}

impl AwsSagemakerProvider {
    /// Create an unconfigured provider with no SDK clients.
    ///
    /// # Errors
    ///
    /// Never fails; the signature matches the other providers.
    pub async fn new() -> Result<Self> {
        Ok(Self {
            config: RwLock::new(None),
            runtime_client: None,
            control_client: None,
            stats: ObservedStats::default(),
        })
    }

    /// Build a provider whose clients come from the ambient AWS configuration.
    ///
    /// # Errors
    ///
    /// Never fails today; returns `Result` so credential resolution can surface
    /// errors in future SDK versions without a breaking change.
    pub async fn with_aws_config() -> Result<Self> {
        let aws = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        Ok(Self {
            config: RwLock::new(None),
            runtime_client: Some(aws_sdk_sagemakerruntime::Client::new(&aws)),
            control_client: Some(aws_sdk_sagemaker::Client::new(&aws)),
            stats: ObservedStats::default(),
        })
    }

    /// Build a provider from pre-constructed SDK clients.
    ///
    /// Tests use this with clients whose `endpoint_url` points at a local mock.
    pub fn with_clients(
        runtime_client: aws_sdk_sagemakerruntime::Client,
        control_client: Option<aws_sdk_sagemaker::Client>,
    ) -> Self {
        Self {
            config: RwLock::new(None),
            runtime_client: Some(runtime_client),
            control_client,
            stats: ObservedStats::default(),
        }
    }

    fn runtime(
        &self,
        operation: &'static str,
    ) -> Result<&aws_sdk_sagemakerruntime::Client, CloudProviderError> {
        self.runtime_client.as_ref().ok_or_else(|| {
            CloudProviderError::missing_credentials(
                PROVIDER,
                operation,
                "build the provider with `with_aws_config()` or `with_clients(..)`",
            )
        })
    }

    fn control(
        &self,
        operation: &'static str,
    ) -> Result<&aws_sdk_sagemaker::Client, CloudProviderError> {
        self.control_client.as_ref().ok_or_else(|| {
            CloudProviderError::missing_credentials(
                PROVIDER,
                operation,
                "build the provider with `with_aws_config()` or supply a SageMaker \
                 control-plane client to `with_clients(..)`",
            )
        })
    }

    /// The endpoint name to invoke: `endpoints.inference_endpoint` when set,
    /// otherwise the requested model name.
    async fn endpoint_name(&self, model_name: &str) -> String {
        match self.config.read().await.as_ref() {
            Some(config) if !config.endpoints.inference_endpoint.trim().is_empty() => {
                config.endpoints.inference_endpoint.clone()
            },
            _ => model_name.to_string(),
        }
    }
}

#[async_trait]
impl CloudProvider for AwsSagemakerProvider {
    async fn initialize(&self, config: &ProviderConfig) -> Result<()> {
        *self.config.write().await = Some(config.clone());
        Ok(())
    }

    async fn inference(&self, request: CloudInferenceRequest) -> Result<CloudInferenceResponse> {
        const OP: &str = "inference";
        let client = self.runtime(OP)?;
        let prompt = require_text(PROVIDER, OP, &request.input_data)?;
        let endpoint = self.endpoint_name(&request.model_name).await;

        let payload = serde_json::json!({
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": request.output_config.max_tokens.unwrap_or(1024),
                "temperature": request.output_config.temperature,
                "top_p": request.output_config.top_p,
            }
        });

        let started = Instant::now();
        let output = client
            .invoke_endpoint()
            .endpoint_name(&endpoint)
            .content_type("application/json")
            .accept("application/json")
            .body(Blob::new(serde_json::to_vec(&payload)?))
            .send()
            .await;

        let output = match output {
            Ok(output) => output,
            Err(e) => {
                self.stats.record(false, started.elapsed());
                return Err(CloudProviderError::ApiError {
                    provider: PROVIDER,
                    operation: OP,
                    status: 0,
                    body: format!("{e:?}"),
                }
                .into());
            },
        };
        let latency = started.elapsed();

        let Some(body) = output.body() else {
            self.stats.record(false, latency);
            return Err(CloudProviderError::InvalidResponse {
                provider: PROVIDER,
                operation: OP,
                detail: "InvokeEndpoint returned no body".to_string(),
            }
            .into());
        };

        let parsed: serde_json::Value = serde_json::from_slice(body.as_ref()).map_err(|e| {
            CloudProviderError::InvalidResponse {
                provider: PROVIDER,
                operation: OP,
                detail: format!("endpoint response is not JSON: {e}"),
            }
        })?;

        // SageMaker text-generation containers answer either
        // `[{"generated_text": ...}]` or `{"generated_text": ...}`.
        let text = parsed
            .as_array()
            .and_then(|items| items.first())
            .and_then(|item| item["generated_text"].as_str())
            .or_else(|| parsed["generated_text"].as_str());
        let Some(text) = text else {
            self.stats.record(false, latency);
            return Err(CloudProviderError::InvalidResponse {
                provider: PROVIDER,
                operation: OP,
                detail: "endpoint response contained no generated_text".to_string(),
            }
            .into());
        };

        let mut provider_metadata = HashMap::new();
        if let Some(invoked_version) = output.invoked_production_variant() {
            provider_metadata.insert(
                "invoked_production_variant".to_string(),
                serde_json::json!(invoked_version),
            );
        }
        self.stats.record(true, latency);

        Ok(CloudInferenceResponse {
            request_id: request.request_id,
            provider: PROVIDER.to_string(),
            model_name: request.model_name,
            model_version: request.model_version,
            output_data: OutputData::Text(text.to_string()),
            // SageMaker does not return token accounting.
            metadata: metadata(latency, None, None, None, provider_metadata),
            performance: performance(latency, None),
            cost: unreported_cost(),
            timestamp: Utc::now(),
        })
    }

    async fn batch_inference(
        &self,
        requests: Vec<CloudInferenceRequest>,
    ) -> Result<Vec<CloudInferenceResponse>> {
        let mut responses = Vec::with_capacity(requests.len());
        for request in requests {
            responses.push(self.inference(request).await?);
        }
        Ok(responses)
    }

    async fn deploy_model(
        &self,
        request: ModelDeploymentRequest,
    ) -> Result<ModelDeploymentResponse> {
        const OP: &str = "deploy_model";
        let client = self.control(OP)?;
        let config = self.config.read().await.clone();
        let role_arn = config
            .as_ref()
            .and_then(|config| config.credentials.session_token.clone())
            .or_else(|| request.deployment_config.security_config.iam_role_arn.clone())
            .ok_or_else(|| CloudProviderError::MissingConfiguration {
                provider: PROVIDER,
                operation: OP,
                detail: "CreateModel requires an execution role: set \
                         deployment_config.security_config.iam_role_arn"
                    .to_string(),
            })?;

        let model_name = format!("{}-{}", request.model_name, request.model_version);
        let endpoint_config_name = format!("{}-config", request.deployment_id);

        let mut container = aws_sdk_sagemaker::types::ContainerDefinition::builder()
            .model_data_url(&request.model_artifact_uri);
        for (key, value) in &request.environment_variables {
            container = container.environment(key, value);
        }

        client
            .create_model()
            .model_name(&model_name)
            .execution_role_arn(&role_arn)
            .primary_container(container.build())
            .send()
            .await
            .map_err(|e| CloudProviderError::ApiError {
                provider: PROVIDER,
                operation: "CreateModel",
                status: 0,
                body: format!("{e:?}"),
            })?;

        let variant = aws_sdk_sagemaker::types::ProductionVariant::builder()
            .variant_name("AllTraffic")
            .model_name(&model_name)
            .initial_instance_count(request.instance_count as i32)
            .instance_type(
                aws_sdk_sagemaker::types::ProductionVariantInstanceType::from(
                    request.instance_type.as_str(),
                ),
            )
            .initial_variant_weight(1.0)
            .build();

        client
            .create_endpoint_config()
            .endpoint_config_name(&endpoint_config_name)
            .production_variants(variant)
            .send()
            .await
            .map_err(|e| CloudProviderError::ApiError {
                provider: PROVIDER,
                operation: "CreateEndpointConfig",
                status: 0,
                body: format!("{e:?}"),
            })?;

        let endpoint = client
            .create_endpoint()
            .endpoint_name(&request.deployment_id)
            .endpoint_config_name(&endpoint_config_name)
            .send()
            .await
            .map_err(|e| CloudProviderError::ApiError {
                provider: PROVIDER,
                operation: "CreateEndpoint",
                status: 0,
                body: format!("{e:?}"),
            })?;

        self.stats.active_deployments.fetch_add(1, Ordering::Relaxed);

        Ok(ModelDeploymentResponse {
            deployment_id: request.deployment_id,
            // CreateEndpoint returns immediately; the endpoint is still creating.
            deployment_status: DeploymentStatus::InProgress,
            // The ARN is the only URL SageMaker gives back; invocations go
            // through the SDK, so there is no public HTTPS endpoint to report.
            endpoint_arn: endpoint.endpoint_arn().map(str::to_string),
            endpoint_url: None,
            deployment_time: Utc::now(),
            // AWS does not return a price with CreateEndpoint.
            estimated_cost_per_hour: None,
            performance_estimate: None,
            monitoring_dashboard_url: None,
        })
    }

    async fn update_deployment(
        &self,
        deployment_id: &str,
        config: ModelDeploymentRequest,
    ) -> Result<ModelDeploymentResponse> {
        const OP: &str = "update_deployment";
        let client = self.control(OP)?;
        let endpoint_config_name = format!("{}-config", config.deployment_id);
        let endpoint = client
            .update_endpoint()
            .endpoint_name(deployment_id)
            .endpoint_config_name(&endpoint_config_name)
            .send()
            .await
            .map_err(|e| CloudProviderError::ApiError {
                provider: PROVIDER,
                operation: "UpdateEndpoint",
                status: 0,
                body: format!("{e:?}"),
            })?;

        Ok(ModelDeploymentResponse {
            deployment_id: deployment_id.to_string(),
            deployment_status: DeploymentStatus::Updating,
            endpoint_arn: endpoint.endpoint_arn().map(str::to_string),
            endpoint_url: None,
            deployment_time: Utc::now(),
            estimated_cost_per_hour: None,
            performance_estimate: None,
            monitoring_dashboard_url: None,
        })
    }

    async fn delete_deployment(&self, deployment_id: &str) -> Result<()> {
        const OP: &str = "delete_deployment";
        let client = self.control(OP)?;
        client
            .delete_endpoint()
            .endpoint_name(deployment_id)
            .send()
            .await
            .map_err(|e| CloudProviderError::ApiError {
                provider: PROVIDER,
                operation: "DeleteEndpoint",
                status: 0,
                body: format!("{e:?}"),
            })?;
        let previous = self.stats.active_deployments.load(Ordering::Relaxed);
        self.stats
            .active_deployments
            .store(previous.saturating_sub(1), Ordering::Relaxed);
        Ok(())
    }

    async fn get_deployment_status(&self, deployment_id: &str) -> Result<DeploymentStatus> {
        const OP: &str = "get_deployment_status";
        let client = self.control(OP)?;
        let described = client
            .describe_endpoint()
            .endpoint_name(deployment_id)
            .send()
            .await
            .map_err(|e| CloudProviderError::ApiError {
                provider: PROVIDER,
                operation: "DescribeEndpoint",
                status: 0,
                body: format!("{e:?}"),
            })?;

        use aws_sdk_sagemaker::types::EndpointStatus;
        Ok(match described.endpoint_status() {
            Some(EndpointStatus::Creating) | Some(EndpointStatus::SystemUpdating) => {
                DeploymentStatus::InProgress
            },
            Some(EndpointStatus::Updating) | Some(EndpointStatus::RollingBack) => {
                DeploymentStatus::Updating
            },
            Some(EndpointStatus::InService) => DeploymentStatus::Completed,
            Some(EndpointStatus::Deleting) => DeploymentStatus::Deleting,
            Some(EndpointStatus::Failed) => DeploymentStatus::Failed,
            Some(EndpointStatus::OutOfService) => DeploymentStatus::Failed,
            _ => DeploymentStatus::Pending,
        })
    }

    async fn list_deployments(&self) -> Result<Vec<ModelDeploymentResponse>> {
        const OP: &str = "list_deployments";
        let client = self.control(OP)?;
        let listed =
            client.list_endpoints().send().await.map_err(|e| CloudProviderError::ApiError {
                provider: PROVIDER,
                operation: "ListEndpoints",
                status: 0,
                body: format!("{e:?}"),
            })?;

        use aws_sdk_sagemaker::types::EndpointStatus;
        Ok(listed
            .endpoints()
            .iter()
            .map(|endpoint| ModelDeploymentResponse {
                deployment_id: endpoint.endpoint_name().unwrap_or_default().to_string(),
                deployment_status: match endpoint.endpoint_status() {
                    Some(EndpointStatus::InService) => DeploymentStatus::Completed,
                    Some(EndpointStatus::Creating) => DeploymentStatus::InProgress,
                    Some(EndpointStatus::Updating) => DeploymentStatus::Updating,
                    Some(EndpointStatus::Deleting) => DeploymentStatus::Deleting,
                    Some(EndpointStatus::Failed) => DeploymentStatus::Failed,
                    _ => DeploymentStatus::Pending,
                },
                endpoint_arn: endpoint.endpoint_arn().map(str::to_string),
                endpoint_url: None,
                deployment_time: endpoint
                    .creation_time()
                    .and_then(|time| {
                        chrono::DateTime::from_timestamp(time.secs(), time.subsec_nanos())
                    })
                    .unwrap_or_else(Utc::now),
                estimated_cost_per_hour: None,
                performance_estimate: None,
                monitoring_dashboard_url: None,
            })
            .collect())
    }

    async fn get_model_info(&self, model_name: &str) -> Result<ModelInfo> {
        const OP: &str = "get_model_info";
        let client = self.control(OP)?;
        let described =
            client.describe_model().model_name(model_name).send().await.map_err(|e| {
                CloudProviderError::ApiError {
                    provider: PROVIDER,
                    operation: "DescribeModel",
                    status: 0,
                    body: format!("{e:?}"),
                }
            })?;

        Ok(ModelInfo {
            name: described.model_name().unwrap_or(model_name).to_string(),
            version: described.creation_time().map(|t| t.secs().to_string()).unwrap_or_default(),
            description: described.model_arn().map(str::to_string),
            model_type: "sagemaker-model".to_string(),
            input_schema: serde_json::json!({"type": "string"}),
            output_schema: serde_json::json!({"type": "string"}),
            supported_formats: vec!["text".to_string()],
            max_input_size: None,
            max_output_size: None,
            // SageMaker does not publish pricing through DescribeModel.
            pricing: None,
            performance_characteristics: None,
        })
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        let requests = self.stats.requests.load(Ordering::Relaxed);
        let status = if requests == 0 {
            "unknown"
        } else if self.stats.error_ratio() == 0.0 {
            "healthy"
        } else if self.stats.success_ratio() > 0.0 {
            "degraded"
        } else {
            "unhealthy"
        };
        Ok(HealthStatus {
            provider: PROVIDER.to_string(),
            status: status.to_string(),
            availability: self.stats.success_ratio(),
            last_check: Utc::now(),
            response_time_ms: self.stats.average_latency_ms(),
            error_rate: self.stats.error_ratio(),
            active_deployments: self.stats.active_deployments.load(Ordering::Relaxed) as u32,
            region_status: HashMap::new(),
        })
    }

    async fn get_metrics(&self) -> Result<ProviderMetrics> {
        Ok(ProviderMetrics {
            provider: PROVIDER.to_string(),
            requests_per_second: self.stats.requests_per_second(),
            average_latency_ms: self.stats.average_latency_ms(),
            error_rate: self.stats.error_ratio(),
            cost_per_hour: None,
            active_connections: 0,
            queue_depth: 0,
            throughput_tokens_per_second: 0.0,
            resource_utilization: None::<ResourceUtilization>,
        })
    }

    async fn get_cost_estimate(&self, _request: &CloudInferenceRequest) -> Result<f64> {
        Err(CloudProviderError::unsupported(PROVIDER, "get_cost_estimate").into())
    }

    fn get_provider_type(&self) -> CloudProviderType {
        CloudProviderType::AwsSagemaker
    }

    fn supports_feature(&self, feature: &str) -> bool {
        matches!(feature, "batch" | "deployment")
    }
}
