//! REST-backed cloud inference providers.
//!
//! Each provider here speaks its vendor's real HTTP API:
//!
//! | Provider | Endpoint | Auth |
//! |---|---|---|
//! | [`OpenAiProvider`] | `POST {base}/v1/chat/completions` | `Authorization: Bearer` |
//! | [`AnthropicProvider`] | `POST {base}/v1/messages` | `x-api-key` + `anthropic-version` |
//! | [`AzureMachineLearningProvider`] | `POST {base}/chat/completions` | `api-key` |
//! | [`HuggingFaceProvider`] | `POST {base}/models/{model}` | `Authorization: Bearer` |
//! | [`GoogleVertexAiProvider`] | `POST {base}/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent` | `Authorization: Bearer` |
//!
//! Point `ProviderConfig::endpoints::inference_endpoint` at a different host to
//! target a compatible gateway — that is also how the tests aim a provider at a
//! local mock server. Without credentials every provider returns
//! [`CloudProviderError::MissingCredentials`]; none of them fabricates a
//! completion, a cost figure or an endpoint URL.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Instant;

use anyhow::Result;
use async_trait::async_trait;
use chrono::Utc;
use serde_json::json;

use super::errors::CloudProviderError;
use super::support::{
    api_error, metadata, performance, require_text, unreported_cost, ProviderState,
};
use super::{
    CloudInferenceRequest, CloudInferenceResponse, CloudProvider, CloudProviderType,
    DeploymentStatus, HealthStatus, ModelDeploymentRequest, ModelDeploymentResponse, ModelInfo,
    OutputData, ProviderConfig, ProviderMetrics,
};

/// Default OpenAI API host.
pub const OPENAI_DEFAULT_BASE: &str = "https://api.openai.com";
/// Default Anthropic API host.
pub const ANTHROPIC_DEFAULT_BASE: &str = "https://api.anthropic.com";
/// Default HuggingFace Inference API host.
pub const HUGGINGFACE_DEFAULT_BASE: &str = "https://api-inference.huggingface.co";
/// Default Vertex AI host.
pub const VERTEX_DEFAULT_BASE: &str = "https://us-central1-aiplatform.googleapis.com";
/// Anthropic API version header value.
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Extract `max_tokens`/`temperature`/`top_p` from the request's output config.
fn sampling(request: &CloudInferenceRequest) -> (u32, Option<f32>, Option<f32>) {
    (
        request.output_config.max_tokens.unwrap_or(1024),
        request.output_config.temperature,
        request.output_config.top_p,
    )
}

fn provider_metadata(
    value: &serde_json::Value,
    keys: &[&str],
) -> HashMap<String, serde_json::Value> {
    let mut map = HashMap::new();
    for key in keys {
        if let Some(found) = value.get(*key) {
            map.insert((*key).to_string(), found.clone());
        }
    }
    map
}

macro_rules! rest_provider_boilerplate {
    ($provider:ident, $name:literal, $type:expr) => {
        impl $provider {
            /// Create an uninitialized provider.
            ///
            /// # Errors
            ///
            /// Never fails; the signature matches the other providers.
            pub async fn new() -> Result<Self> {
                Ok(Self {
                    state: ProviderState::new($name),
                })
            }

            /// Access the shared provider state (used by tests).
            pub fn state(&self) -> &ProviderState {
                &self.state
            }
        }

        impl $provider {
            async fn store_config(&self, config: &ProviderConfig) -> Result<()> {
                *self.state.config.write().await = Some(config.clone());
                Ok(())
            }
        }
    };
}

// ── OpenAI ───────────────────────────────────────────────────────────────────

/// OpenAI Chat Completions provider.
#[derive(Debug)]
pub struct OpenAiProvider {
    state: ProviderState,
}

rest_provider_boilerplate!(
    OpenAiProvider,
    "OpenAiProvider",
    CloudProviderType::OpenAiApi
);

#[async_trait]
impl CloudProvider for OpenAiProvider {
    async fn initialize(&self, config: &ProviderConfig) -> Result<()> {
        self.store_config(config).await
    }

    async fn inference(&self, request: CloudInferenceRequest) -> Result<CloudInferenceResponse> {
        const OP: &str = "inference";
        let api_key = self.state.api_key(OP).await?;
        let base = self.state.base_url(OP, OPENAI_DEFAULT_BASE).await?;
        let prompt = require_text("OpenAiProvider", OP, &request.input_data)?;
        let (max_tokens, temperature, top_p) = sampling(&request);

        let mut body = json!({
            "model": request.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        });
        if let Some(temperature) = temperature {
            body["temperature"] = json!(temperature);
        }
        if let Some(top_p) = top_p {
            body["top_p"] = json!(top_p);
        }

        let started = Instant::now();
        let mut http = self
            .state
            .http
            .post(format!("{base}/v1/chat/completions"))
            .bearer_auth(&api_key)
            .json(&body);
        for (name, value) in self.state.custom_headers(OP).await? {
            http = http.header(name, value);
        }
        let response = match http.send().await {
            Ok(response) => response,
            Err(e) => {
                self.state.stats.record(false, started.elapsed());
                return Err(CloudProviderError::Transport {
                    provider: "OpenAiProvider",
                    operation: OP,
                    detail: e.to_string(),
                }
                .into());
            },
        };

        if !response.status().is_success() {
            self.state.stats.record(false, started.elapsed());
            return Err(api_error("OpenAiProvider", OP, response).await.into());
        }

        let payload: serde_json::Value = match response.json().await {
            Ok(payload) => payload,
            Err(e) => {
                self.state.stats.record(false, started.elapsed());
                return Err(CloudProviderError::InvalidResponse {
                    provider: "OpenAiProvider",
                    operation: OP,
                    detail: e.to_string(),
                }
                .into());
            },
        };
        let latency = started.elapsed();

        let Some(text) = payload["choices"][0]["message"]["content"].as_str() else {
            self.state.stats.record(false, latency);
            return Err(CloudProviderError::InvalidResponse {
                provider: "OpenAiProvider",
                operation: OP,
                detail: "response contained no choices[0].message.content".to_string(),
            }
            .into());
        };

        let input_tokens = payload["usage"]["prompt_tokens"].as_u64().map(|v| v as u32);
        let output_tokens = payload["usage"]["completion_tokens"].as_u64().map(|v| v as u32);
        let finish_reason = payload["choices"][0]["finish_reason"].as_str().map(str::to_string);

        self.state.stats.record(true, latency);

        Ok(CloudInferenceResponse {
            request_id: request.request_id,
            provider: "OpenAiProvider".to_string(),
            model_name: payload["model"].as_str().map(str::to_string).unwrap_or(request.model_name),
            model_version: request.model_version,
            output_data: OutputData::Text(text.to_string()),
            metadata: metadata(
                latency,
                input_tokens,
                output_tokens,
                finish_reason,
                provider_metadata(&payload, &["id", "system_fingerprint", "usage"]),
            ),
            performance: performance(latency, output_tokens),
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
        _request: ModelDeploymentRequest,
    ) -> Result<ModelDeploymentResponse> {
        Err(CloudProviderError::unsupported("OpenAiProvider", "deploy_model").into())
    }

    async fn update_deployment(
        &self,
        _deployment_id: &str,
        _config: ModelDeploymentRequest,
    ) -> Result<ModelDeploymentResponse> {
        Err(CloudProviderError::unsupported("OpenAiProvider", "update_deployment").into())
    }

    async fn delete_deployment(&self, _deployment_id: &str) -> Result<()> {
        Err(CloudProviderError::unsupported("OpenAiProvider", "delete_deployment").into())
    }

    async fn get_deployment_status(&self, _deployment_id: &str) -> Result<DeploymentStatus> {
        Err(CloudProviderError::unsupported("OpenAiProvider", "get_deployment_status").into())
    }

    async fn list_deployments(&self) -> Result<Vec<ModelDeploymentResponse>> {
        Err(CloudProviderError::unsupported("OpenAiProvider", "list_deployments").into())
    }

    async fn get_model_info(&self, model_name: &str) -> Result<ModelInfo> {
        const OP: &str = "get_model_info";
        let api_key = self.state.api_key(OP).await?;
        let base = self.state.base_url(OP, OPENAI_DEFAULT_BASE).await?;
        let started = Instant::now();
        let response = self
            .state
            .http
            .get(format!("{base}/v1/models/{model_name}"))
            .bearer_auth(&api_key)
            .send()
            .await
            .map_err(|e| CloudProviderError::Transport {
                provider: "OpenAiProvider",
                operation: OP,
                detail: e.to_string(),
            })?;
        if !response.status().is_success() {
            self.state.stats.record(false, started.elapsed());
            return Err(api_error("OpenAiProvider", OP, response).await.into());
        }
        let payload: serde_json::Value =
            response.json().await.map_err(|e| CloudProviderError::InvalidResponse {
                provider: "OpenAiProvider",
                operation: OP,
                detail: e.to_string(),
            })?;
        self.state.stats.record(true, started.elapsed());

        Ok(ModelInfo {
            name: payload["id"].as_str().unwrap_or(model_name).to_string(),
            version: payload["created"].as_u64().map(|v| v.to_string()).unwrap_or_default(),
            description: payload["owned_by"].as_str().map(|owner| format!("owned by {owner}")),
            model_type: payload["object"].as_str().unwrap_or("model").to_string(),
            input_schema: json!({"type": "string"}),
            output_schema: json!({"type": "string"}),
            supported_formats: vec!["text".to_string()],
            // The models endpoint publishes no size limits.
            max_input_size: None,
            max_output_size: None,
            // OpenAI does not publish pricing through the API.
            pricing: None,
            performance_characteristics: None,
        })
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        const OP: &str = "health_check";
        let api_key = self.state.api_key(OP).await?;
        let base = self.state.base_url(OP, OPENAI_DEFAULT_BASE).await?;
        let started = Instant::now();
        let outcome = self
            .state
            .http
            .get(format!("{base}/v1/models"))
            .bearer_auth(&api_key)
            .send()
            .await;
        let success = matches!(&outcome, Ok(response) if response.status().is_success());
        self.state.stats.record(success, started.elapsed());
        Ok(self.state.health())
    }

    async fn get_metrics(&self) -> Result<ProviderMetrics> {
        Ok(self.state.metrics())
    }

    async fn get_cost_estimate(&self, _request: &CloudInferenceRequest) -> Result<f64> {
        Err(CloudProviderError::unsupported("OpenAiProvider", "get_cost_estimate").into())
    }

    fn get_provider_type(&self) -> CloudProviderType {
        CloudProviderType::OpenAiApi
    }

    fn supports_feature(&self, feature: &str) -> bool {
        matches!(feature, "streaming" | "batch")
    }
}

// ── Anthropic ────────────────────────────────────────────────────────────────

/// Anthropic Messages API provider.
#[derive(Debug)]
pub struct AnthropicProvider {
    state: ProviderState,
}

rest_provider_boilerplate!(
    AnthropicProvider,
    "AnthropicProvider",
    CloudProviderType::AnthropicClaude
);

#[async_trait]
impl CloudProvider for AnthropicProvider {
    async fn initialize(&self, config: &ProviderConfig) -> Result<()> {
        self.store_config(config).await
    }

    async fn inference(&self, request: CloudInferenceRequest) -> Result<CloudInferenceResponse> {
        const OP: &str = "inference";
        let api_key = self.state.api_key(OP).await?;
        let base = self.state.base_url(OP, ANTHROPIC_DEFAULT_BASE).await?;
        let prompt = require_text("AnthropicProvider", OP, &request.input_data)?;
        let (max_tokens, temperature, top_p) = sampling(&request);

        let mut body = json!({
            "model": request.model_name,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        });
        if let Some(temperature) = temperature {
            body["temperature"] = json!(temperature);
        }
        if let Some(top_p) = top_p {
            body["top_p"] = json!(top_p);
        }

        let started = Instant::now();
        let mut http = self
            .state
            .http
            .post(format!("{base}/v1/messages"))
            .header("x-api-key", &api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&body);
        for (name, value) in self.state.custom_headers(OP).await? {
            http = http.header(name, value);
        }
        let response = match http.send().await {
            Ok(response) => response,
            Err(e) => {
                self.state.stats.record(false, started.elapsed());
                return Err(CloudProviderError::Transport {
                    provider: "AnthropicProvider",
                    operation: OP,
                    detail: e.to_string(),
                }
                .into());
            },
        };
        if !response.status().is_success() {
            self.state.stats.record(false, started.elapsed());
            return Err(api_error("AnthropicProvider", OP, response).await.into());
        }
        let payload: serde_json::Value = match response.json().await {
            Ok(payload) => payload,
            Err(e) => {
                self.state.stats.record(false, started.elapsed());
                return Err(CloudProviderError::InvalidResponse {
                    provider: "AnthropicProvider",
                    operation: OP,
                    detail: e.to_string(),
                }
                .into());
            },
        };
        let latency = started.elapsed();

        // The Messages API returns a list of content blocks; concatenate the
        // text blocks in order.
        let text: String = payload["content"]
            .as_array()
            .map(|blocks| {
                blocks
                    .iter()
                    .filter(|block| block["type"].as_str() == Some("text"))
                    .filter_map(|block| block["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("")
            })
            .unwrap_or_default();
        if text.is_empty() && payload["content"].as_array().map(Vec::len).unwrap_or(0) == 0 {
            self.state.stats.record(false, latency);
            return Err(CloudProviderError::InvalidResponse {
                provider: "AnthropicProvider",
                operation: OP,
                detail: "response contained no content blocks".to_string(),
            }
            .into());
        }

        let input_tokens = payload["usage"]["input_tokens"].as_u64().map(|v| v as u32);
        let output_tokens = payload["usage"]["output_tokens"].as_u64().map(|v| v as u32);
        let finish_reason = payload["stop_reason"].as_str().map(str::to_string);

        self.state.stats.record(true, latency);

        Ok(CloudInferenceResponse {
            request_id: request.request_id,
            provider: "AnthropicProvider".to_string(),
            model_name: payload["model"].as_str().map(str::to_string).unwrap_or(request.model_name),
            model_version: request.model_version,
            output_data: OutputData::Text(text),
            metadata: metadata(
                latency,
                input_tokens,
                output_tokens,
                finish_reason,
                provider_metadata(&payload, &["id", "stop_sequence", "usage"]),
            ),
            performance: performance(latency, output_tokens),
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
        _request: ModelDeploymentRequest,
    ) -> Result<ModelDeploymentResponse> {
        Err(CloudProviderError::unsupported("AnthropicProvider", "deploy_model").into())
    }

    async fn update_deployment(
        &self,
        _deployment_id: &str,
        _config: ModelDeploymentRequest,
    ) -> Result<ModelDeploymentResponse> {
        Err(CloudProviderError::unsupported("AnthropicProvider", "update_deployment").into())
    }

    async fn delete_deployment(&self, _deployment_id: &str) -> Result<()> {
        Err(CloudProviderError::unsupported("AnthropicProvider", "delete_deployment").into())
    }

    async fn get_deployment_status(&self, _deployment_id: &str) -> Result<DeploymentStatus> {
        Err(CloudProviderError::unsupported("AnthropicProvider", "get_deployment_status").into())
    }

    async fn list_deployments(&self) -> Result<Vec<ModelDeploymentResponse>> {
        Err(CloudProviderError::unsupported("AnthropicProvider", "list_deployments").into())
    }

    async fn get_model_info(&self, model_name: &str) -> Result<ModelInfo> {
        const OP: &str = "get_model_info";
        let api_key = self.state.api_key(OP).await?;
        let base = self.state.base_url(OP, ANTHROPIC_DEFAULT_BASE).await?;
        let started = Instant::now();
        let response = self
            .state
            .http
            .get(format!("{base}/v1/models/{model_name}"))
            .header("x-api-key", &api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .send()
            .await
            .map_err(|e| CloudProviderError::Transport {
                provider: "AnthropicProvider",
                operation: OP,
                detail: e.to_string(),
            })?;
        if !response.status().is_success() {
            self.state.stats.record(false, started.elapsed());
            return Err(api_error("AnthropicProvider", OP, response).await.into());
        }
        let payload: serde_json::Value =
            response.json().await.map_err(|e| CloudProviderError::InvalidResponse {
                provider: "AnthropicProvider",
                operation: OP,
                detail: e.to_string(),
            })?;
        self.state.stats.record(true, started.elapsed());

        Ok(ModelInfo {
            name: payload["id"].as_str().unwrap_or(model_name).to_string(),
            version: payload["created_at"].as_str().unwrap_or_default().to_string(),
            description: payload["display_name"].as_str().map(str::to_string),
            model_type: payload["type"].as_str().unwrap_or("model").to_string(),
            input_schema: json!({"type": "string"}),
            output_schema: json!({"type": "string"}),
            supported_formats: vec!["text".to_string()],
            max_input_size: None,
            max_output_size: None,
            pricing: None,
            performance_characteristics: None,
        })
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        const OP: &str = "health_check";
        let api_key = self.state.api_key(OP).await?;
        let base = self.state.base_url(OP, ANTHROPIC_DEFAULT_BASE).await?;
        let started = Instant::now();
        let outcome = self
            .state
            .http
            .get(format!("{base}/v1/models"))
            .header("x-api-key", &api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .send()
            .await;
        let success = matches!(&outcome, Ok(response) if response.status().is_success());
        self.state.stats.record(success, started.elapsed());
        Ok(self.state.health())
    }

    async fn get_metrics(&self) -> Result<ProviderMetrics> {
        Ok(self.state.metrics())
    }

    async fn get_cost_estimate(&self, _request: &CloudInferenceRequest) -> Result<f64> {
        Err(CloudProviderError::unsupported("AnthropicProvider", "get_cost_estimate").into())
    }

    fn get_provider_type(&self) -> CloudProviderType {
        CloudProviderType::AnthropicClaude
    }

    fn supports_feature(&self, feature: &str) -> bool {
        matches!(feature, "streaming" | "batch")
    }
}

// ── Azure Machine Learning / Azure AI model inference ────────────────────────

/// Azure AI model-inference provider (chat-completions shaped).
#[derive(Debug)]
pub struct AzureMachineLearningProvider {
    state: ProviderState,
}

rest_provider_boilerplate!(
    AzureMachineLearningProvider,
    "AzureMachineLearningProvider",
    CloudProviderType::AzureMachineLearning
);

#[async_trait]
impl CloudProvider for AzureMachineLearningProvider {
    async fn initialize(&self, config: &ProviderConfig) -> Result<()> {
        self.store_config(config).await
    }

    async fn inference(&self, request: CloudInferenceRequest) -> Result<CloudInferenceResponse> {
        const OP: &str = "inference";
        let config = self.state.config(OP).await?;
        let api_key = self.state.api_key(OP).await?;
        if config.endpoints.inference_endpoint.trim().is_empty() {
            return Err(CloudProviderError::MissingConfiguration {
                provider: "AzureMachineLearningProvider",
                operation: OP,
                detail: "Azure endpoints are per-deployment: set \
                         endpoints.inference_endpoint to the scoring URI"
                    .to_string(),
            }
            .into());
        }
        let base = self.state.base_url(OP, "").await?;
        let prompt = require_text("AzureMachineLearningProvider", OP, &request.input_data)?;
        let (max_tokens, temperature, top_p) = sampling(&request);

        let mut body = json!({
            "model": request.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        });
        if let Some(temperature) = temperature {
            body["temperature"] = json!(temperature);
        }
        if let Some(top_p) = top_p {
            body["top_p"] = json!(top_p);
        }

        let started = Instant::now();
        let mut http = self
            .state
            .http
            .post(format!("{base}/chat/completions"))
            .header("api-key", &api_key)
            .header("content-type", "application/json")
            .json(&body);
        for (name, value) in self.state.custom_headers(OP).await? {
            http = http.header(name, value);
        }
        let response = match http.send().await {
            Ok(response) => response,
            Err(e) => {
                self.state.stats.record(false, started.elapsed());
                return Err(CloudProviderError::Transport {
                    provider: "AzureMachineLearningProvider",
                    operation: OP,
                    detail: e.to_string(),
                }
                .into());
            },
        };
        if !response.status().is_success() {
            self.state.stats.record(false, started.elapsed());
            return Err(api_error("AzureMachineLearningProvider", OP, response).await.into());
        }
        let payload: serde_json::Value = match response.json().await {
            Ok(payload) => payload,
            Err(e) => {
                self.state.stats.record(false, started.elapsed());
                return Err(CloudProviderError::InvalidResponse {
                    provider: "AzureMachineLearningProvider",
                    operation: OP,
                    detail: e.to_string(),
                }
                .into());
            },
        };
        let latency = started.elapsed();

        let Some(text) = payload["choices"][0]["message"]["content"].as_str() else {
            self.state.stats.record(false, latency);
            return Err(CloudProviderError::InvalidResponse {
                provider: "AzureMachineLearningProvider",
                operation: OP,
                detail: "response contained no choices[0].message.content".to_string(),
            }
            .into());
        };

        let input_tokens = payload["usage"]["prompt_tokens"].as_u64().map(|v| v as u32);
        let output_tokens = payload["usage"]["completion_tokens"].as_u64().map(|v| v as u32);
        let finish_reason = payload["choices"][0]["finish_reason"].as_str().map(str::to_string);
        self.state.stats.record(true, latency);

        Ok(CloudInferenceResponse {
            request_id: request.request_id,
            provider: "AzureMachineLearningProvider".to_string(),
            model_name: payload["model"].as_str().map(str::to_string).unwrap_or(request.model_name),
            model_version: request.model_version,
            output_data: OutputData::Text(text.to_string()),
            metadata: metadata(
                latency,
                input_tokens,
                output_tokens,
                finish_reason,
                provider_metadata(&payload, &["id", "usage"]),
            ),
            performance: performance(latency, output_tokens),
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
        _request: ModelDeploymentRequest,
    ) -> Result<ModelDeploymentResponse> {
        Err(CloudProviderError::unsupported("AzureMachineLearningProvider", "deploy_model").into())
    }

    async fn update_deployment(
        &self,
        _deployment_id: &str,
        _config: ModelDeploymentRequest,
    ) -> Result<ModelDeploymentResponse> {
        Err(
            CloudProviderError::unsupported("AzureMachineLearningProvider", "update_deployment")
                .into(),
        )
    }

    async fn delete_deployment(&self, _deployment_id: &str) -> Result<()> {
        Err(
            CloudProviderError::unsupported("AzureMachineLearningProvider", "delete_deployment")
                .into(),
        )
    }

    async fn get_deployment_status(&self, _deployment_id: &str) -> Result<DeploymentStatus> {
        Err(CloudProviderError::unsupported(
            "AzureMachineLearningProvider",
            "get_deployment_status",
        )
        .into())
    }

    async fn list_deployments(&self) -> Result<Vec<ModelDeploymentResponse>> {
        Err(
            CloudProviderError::unsupported("AzureMachineLearningProvider", "list_deployments")
                .into(),
        )
    }

    async fn get_model_info(&self, _model_name: &str) -> Result<ModelInfo> {
        Err(
            CloudProviderError::unsupported("AzureMachineLearningProvider", "get_model_info")
                .into(),
        )
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        Ok(self.state.health())
    }

    async fn get_metrics(&self) -> Result<ProviderMetrics> {
        Ok(self.state.metrics())
    }

    async fn get_cost_estimate(&self, _request: &CloudInferenceRequest) -> Result<f64> {
        Err(
            CloudProviderError::unsupported("AzureMachineLearningProvider", "get_cost_estimate")
                .into(),
        )
    }

    fn get_provider_type(&self) -> CloudProviderType {
        CloudProviderType::AzureMachineLearning
    }

    fn supports_feature(&self, feature: &str) -> bool {
        matches!(feature, "batch")
    }
}

// ── HuggingFace Inference API ────────────────────────────────────────────────

/// HuggingFace Inference API provider.
#[derive(Debug)]
pub struct HuggingFaceProvider {
    state: ProviderState,
}

rest_provider_boilerplate!(
    HuggingFaceProvider,
    "HuggingFaceProvider",
    CloudProviderType::HuggingFaceInference
);

#[async_trait]
impl CloudProvider for HuggingFaceProvider {
    async fn initialize(&self, config: &ProviderConfig) -> Result<()> {
        self.store_config(config).await
    }

    async fn inference(&self, request: CloudInferenceRequest) -> Result<CloudInferenceResponse> {
        const OP: &str = "inference";
        let api_key = self.state.api_key(OP).await?;
        let base = self.state.base_url(OP, HUGGINGFACE_DEFAULT_BASE).await?;
        let prompt = require_text("HuggingFaceProvider", OP, &request.input_data)?;
        let (max_tokens, temperature, top_p) = sampling(&request);

        let mut parameters = json!({ "max_new_tokens": max_tokens, "return_full_text": false });
        if let Some(temperature) = temperature {
            parameters["temperature"] = json!(temperature);
        }
        if let Some(top_p) = top_p {
            parameters["top_p"] = json!(top_p);
        }
        let body = json!({ "inputs": prompt, "parameters": parameters });

        let started = Instant::now();
        let mut http = self
            .state
            .http
            .post(format!("{base}/models/{}", request.model_name))
            .bearer_auth(&api_key)
            .json(&body);
        for (name, value) in self.state.custom_headers(OP).await? {
            http = http.header(name, value);
        }
        let response = match http.send().await {
            Ok(response) => response,
            Err(e) => {
                self.state.stats.record(false, started.elapsed());
                return Err(CloudProviderError::Transport {
                    provider: "HuggingFaceProvider",
                    operation: OP,
                    detail: e.to_string(),
                }
                .into());
            },
        };
        if !response.status().is_success() {
            self.state.stats.record(false, started.elapsed());
            return Err(api_error("HuggingFaceProvider", OP, response).await.into());
        }
        let payload: serde_json::Value = match response.json().await {
            Ok(payload) => payload,
            Err(e) => {
                self.state.stats.record(false, started.elapsed());
                return Err(CloudProviderError::InvalidResponse {
                    provider: "HuggingFaceProvider",
                    operation: OP,
                    detail: e.to_string(),
                }
                .into());
            },
        };
        let latency = started.elapsed();

        // The text-generation task answers with `[{"generated_text": "..."}]`.
        let text = payload
            .as_array()
            .and_then(|items| items.first())
            .and_then(|item| item["generated_text"].as_str())
            .or_else(|| payload["generated_text"].as_str());
        let Some(text) = text else {
            self.state.stats.record(false, latency);
            return Err(CloudProviderError::InvalidResponse {
                provider: "HuggingFaceProvider",
                operation: OP,
                detail: "response contained no generated_text field".to_string(),
            }
            .into());
        };
        self.state.stats.record(true, latency);

        Ok(CloudInferenceResponse {
            request_id: request.request_id,
            provider: "HuggingFaceProvider".to_string(),
            model_name: request.model_name,
            model_version: request.model_version,
            output_data: OutputData::Text(text.to_string()),
            // The Inference API returns no token accounting.
            metadata: metadata(latency, None, None, None, HashMap::new()),
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
        _request: ModelDeploymentRequest,
    ) -> Result<ModelDeploymentResponse> {
        Err(CloudProviderError::unsupported("HuggingFaceProvider", "deploy_model").into())
    }

    async fn update_deployment(
        &self,
        _deployment_id: &str,
        _config: ModelDeploymentRequest,
    ) -> Result<ModelDeploymentResponse> {
        Err(CloudProviderError::unsupported("HuggingFaceProvider", "update_deployment").into())
    }

    async fn delete_deployment(&self, _deployment_id: &str) -> Result<()> {
        Err(CloudProviderError::unsupported("HuggingFaceProvider", "delete_deployment").into())
    }

    async fn get_deployment_status(&self, _deployment_id: &str) -> Result<DeploymentStatus> {
        Err(CloudProviderError::unsupported("HuggingFaceProvider", "get_deployment_status").into())
    }

    async fn list_deployments(&self) -> Result<Vec<ModelDeploymentResponse>> {
        Err(CloudProviderError::unsupported("HuggingFaceProvider", "list_deployments").into())
    }

    async fn get_model_info(&self, model_name: &str) -> Result<ModelInfo> {
        const OP: &str = "get_model_info";
        let config = self.state.config(OP).await?;
        // Model metadata lives on the Hub host, not the inference host.
        let base = config
            .endpoints
            .model_management_endpoint
            .clone()
            .unwrap_or_else(|| "https://huggingface.co".to_string());
        let started = Instant::now();
        let mut http = self.state.http.get(format!(
            "{}/api/models/{model_name}",
            base.trim_end_matches('/')
        ));
        if let Some(api_key) = &config.credentials.api_key {
            http = http.bearer_auth(api_key);
        }
        let response = http.send().await.map_err(|e| CloudProviderError::Transport {
            provider: "HuggingFaceProvider",
            operation: OP,
            detail: e.to_string(),
        })?;
        if !response.status().is_success() {
            self.state.stats.record(false, started.elapsed());
            return Err(api_error("HuggingFaceProvider", OP, response).await.into());
        }
        let payload: serde_json::Value =
            response.json().await.map_err(|e| CloudProviderError::InvalidResponse {
                provider: "HuggingFaceProvider",
                operation: OP,
                detail: e.to_string(),
            })?;
        self.state.stats.record(true, started.elapsed());

        Ok(ModelInfo {
            name: payload["id"].as_str().unwrap_or(model_name).to_string(),
            version: payload["sha"].as_str().unwrap_or_default().to_string(),
            description: payload["pipeline_tag"].as_str().map(str::to_string),
            model_type: payload["pipeline_tag"].as_str().unwrap_or("model").to_string(),
            input_schema: json!({"type": "string"}),
            output_schema: json!({"type": "string"}),
            supported_formats: vec!["text".to_string()],
            max_input_size: None,
            max_output_size: None,
            pricing: None,
            performance_characteristics: None,
        })
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        Ok(self.state.health())
    }

    async fn get_metrics(&self) -> Result<ProviderMetrics> {
        Ok(self.state.metrics())
    }

    async fn get_cost_estimate(&self, _request: &CloudInferenceRequest) -> Result<f64> {
        Err(CloudProviderError::unsupported("HuggingFaceProvider", "get_cost_estimate").into())
    }

    fn get_provider_type(&self) -> CloudProviderType {
        CloudProviderType::HuggingFaceInference
    }

    fn supports_feature(&self, feature: &str) -> bool {
        matches!(feature, "batch")
    }
}

// ── Google Vertex AI ─────────────────────────────────────────────────────────

/// Google Vertex AI `generateContent` provider.
#[derive(Debug)]
pub struct GoogleVertexAiProvider {
    state: ProviderState,
}

rest_provider_boilerplate!(
    GoogleVertexAiProvider,
    "GoogleVertexAiProvider",
    CloudProviderType::GoogleVertexAi
);

impl GoogleVertexAiProvider {
    /// The GCP project id, read from `credentials.client_id`.
    async fn project(&self, operation: &'static str) -> Result<String, CloudProviderError> {
        let config = self.state.config(operation).await?;
        config.credentials.client_id.clone().ok_or_else(|| {
            CloudProviderError::MissingConfiguration {
                provider: "GoogleVertexAiProvider",
                operation,
                detail: "set credentials.client_id to the GCP project id".to_string(),
            }
        })
    }

    /// OAuth bearer token, read from `credentials.oauth_token`.
    async fn token(&self, operation: &'static str) -> Result<String, CloudProviderError> {
        let config = self.state.config(operation).await?;
        config
            .credentials
            .oauth_token
            .clone()
            .or_else(|| config.credentials.api_key.clone())
            .ok_or_else(|| {
                CloudProviderError::missing_credentials(
                    "GoogleVertexAiProvider",
                    operation,
                    "set credentials.oauth_token to a Google OAuth2 access token",
                )
            })
    }
}

#[async_trait]
impl CloudProvider for GoogleVertexAiProvider {
    async fn initialize(&self, config: &ProviderConfig) -> Result<()> {
        self.store_config(config).await
    }

    async fn inference(&self, request: CloudInferenceRequest) -> Result<CloudInferenceResponse> {
        const OP: &str = "inference";
        let config = self.state.config(OP).await?;
        let token = self.token(OP).await?;
        let project = self.project(OP).await?;
        let location = if config.region.trim().is_empty() {
            "us-central1".to_string()
        } else {
            config.region.clone()
        };
        let base = self.state.base_url(OP, VERTEX_DEFAULT_BASE).await?;
        let prompt = require_text("GoogleVertexAiProvider", OP, &request.input_data)?;
        let (max_tokens, temperature, top_p) = sampling(&request);

        let mut generation_config = json!({ "maxOutputTokens": max_tokens });
        if let Some(temperature) = temperature {
            generation_config["temperature"] = json!(temperature);
        }
        if let Some(top_p) = top_p {
            generation_config["topP"] = json!(top_p);
        }
        let body = json!({
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        });

        let url = format!(
            "{base}/v1/projects/{project}/locations/{location}/publishers/google/models/{}:generateContent",
            request.model_name
        );

        let started = Instant::now();
        let mut http = self.state.http.post(url).bearer_auth(&token).json(&body);
        for (name, value) in self.state.custom_headers(OP).await? {
            http = http.header(name, value);
        }
        let response = match http.send().await {
            Ok(response) => response,
            Err(e) => {
                self.state.stats.record(false, started.elapsed());
                return Err(CloudProviderError::Transport {
                    provider: "GoogleVertexAiProvider",
                    operation: OP,
                    detail: e.to_string(),
                }
                .into());
            },
        };
        if !response.status().is_success() {
            self.state.stats.record(false, started.elapsed());
            return Err(api_error("GoogleVertexAiProvider", OP, response).await.into());
        }
        let payload: serde_json::Value = match response.json().await {
            Ok(payload) => payload,
            Err(e) => {
                self.state.stats.record(false, started.elapsed());
                return Err(CloudProviderError::InvalidResponse {
                    provider: "GoogleVertexAiProvider",
                    operation: OP,
                    detail: e.to_string(),
                }
                .into());
            },
        };
        let latency = started.elapsed();

        let text: String = payload["candidates"][0]["content"]["parts"]
            .as_array()
            .map(|parts| {
                parts
                    .iter()
                    .filter_map(|part| part["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("")
            })
            .unwrap_or_default();
        if text.is_empty() {
            self.state.stats.record(false, latency);
            return Err(CloudProviderError::InvalidResponse {
                provider: "GoogleVertexAiProvider",
                operation: OP,
                detail: "response contained no candidates[0].content.parts[].text".to_string(),
            }
            .into());
        }

        let input_tokens = payload["usageMetadata"]["promptTokenCount"].as_u64().map(|v| v as u32);
        let output_tokens =
            payload["usageMetadata"]["candidatesTokenCount"].as_u64().map(|v| v as u32);
        let finish_reason = payload["candidates"][0]["finishReason"].as_str().map(str::to_string);
        self.state.stats.record(true, latency);

        Ok(CloudInferenceResponse {
            request_id: request.request_id,
            provider: "GoogleVertexAiProvider".to_string(),
            model_name: request.model_name,
            model_version: request.model_version,
            output_data: OutputData::Text(text),
            metadata: metadata(
                latency,
                input_tokens,
                output_tokens,
                finish_reason,
                provider_metadata(&payload, &["usageMetadata", "modelVersion"]),
            ),
            performance: performance(latency, output_tokens),
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
        _request: ModelDeploymentRequest,
    ) -> Result<ModelDeploymentResponse> {
        Err(CloudProviderError::unsupported("GoogleVertexAiProvider", "deploy_model").into())
    }

    async fn update_deployment(
        &self,
        _deployment_id: &str,
        _config: ModelDeploymentRequest,
    ) -> Result<ModelDeploymentResponse> {
        Err(CloudProviderError::unsupported("GoogleVertexAiProvider", "update_deployment").into())
    }

    async fn delete_deployment(&self, _deployment_id: &str) -> Result<()> {
        Err(CloudProviderError::unsupported("GoogleVertexAiProvider", "delete_deployment").into())
    }

    async fn get_deployment_status(&self, _deployment_id: &str) -> Result<DeploymentStatus> {
        Err(
            CloudProviderError::unsupported("GoogleVertexAiProvider", "get_deployment_status")
                .into(),
        )
    }

    async fn list_deployments(&self) -> Result<Vec<ModelDeploymentResponse>> {
        Err(CloudProviderError::unsupported("GoogleVertexAiProvider", "list_deployments").into())
    }

    async fn get_model_info(&self, _model_name: &str) -> Result<ModelInfo> {
        Err(CloudProviderError::unsupported("GoogleVertexAiProvider", "get_model_info").into())
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        Ok(self.state.health())
    }

    async fn get_metrics(&self) -> Result<ProviderMetrics> {
        Ok(self.state.metrics())
    }

    async fn get_cost_estimate(&self, _request: &CloudInferenceRequest) -> Result<f64> {
        Err(CloudProviderError::unsupported("GoogleVertexAiProvider", "get_cost_estimate").into())
    }

    fn get_provider_type(&self) -> CloudProviderType {
        CloudProviderType::GoogleVertexAi
    }

    fn supports_feature(&self, feature: &str) -> bool {
        matches!(feature, "streaming" | "batch")
    }
}

// ── Custom (OpenAI-compatible gateway) ───────────────────────────────────────

/// A user-named provider speaking the OpenAI chat-completions protocol.
#[derive(Debug)]
pub struct CustomProvider {
    name: String,
    state: ProviderState,
}

impl CustomProvider {
    /// Create an uninitialized custom provider named `name`.
    ///
    /// # Errors
    ///
    /// Never fails; the signature matches the other providers.
    pub async fn new(name: &str) -> Result<Self> {
        Ok(Self {
            name: name.to_string(),
            state: ProviderState::new("CustomProvider"),
        })
    }

    /// Access the shared provider state (used by tests).
    pub fn state(&self) -> &ProviderState {
        &self.state
    }
}

#[async_trait]
impl CloudProvider for CustomProvider {
    async fn initialize(&self, config: &ProviderConfig) -> Result<()> {
        *self.state.config.write().await = Some(config.clone());
        Ok(())
    }

    async fn inference(&self, request: CloudInferenceRequest) -> Result<CloudInferenceResponse> {
        const OP: &str = "inference";
        let config = self.state.config(OP).await?;
        if config.endpoints.inference_endpoint.trim().is_empty() {
            return Err(CloudProviderError::MissingConfiguration {
                provider: "CustomProvider",
                operation: OP,
                detail: "set endpoints.inference_endpoint to the gateway base URL".to_string(),
            }
            .into());
        }
        let base = self.state.base_url(OP, "").await?;
        let prompt = require_text("CustomProvider", OP, &request.input_data)?;
        let (max_tokens, temperature, top_p) = sampling(&request);

        let mut body = json!({
            "model": request.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        });
        if let Some(temperature) = temperature {
            body["temperature"] = json!(temperature);
        }
        if let Some(top_p) = top_p {
            body["top_p"] = json!(top_p);
        }

        let started = Instant::now();
        let mut http = self.state.http.post(format!("{base}/v1/chat/completions")).json(&body);
        if let Some(api_key) = &config.credentials.api_key {
            http = http.bearer_auth(api_key);
        }
        for (name, value) in config.credentials.custom_headers.iter() {
            http = http.header(name, value);
        }
        let response = match http.send().await {
            Ok(response) => response,
            Err(e) => {
                self.state.stats.record(false, started.elapsed());
                return Err(CloudProviderError::Transport {
                    provider: "CustomProvider",
                    operation: OP,
                    detail: e.to_string(),
                }
                .into());
            },
        };
        if !response.status().is_success() {
            self.state.stats.record(false, started.elapsed());
            return Err(api_error("CustomProvider", OP, response).await.into());
        }
        let payload: serde_json::Value = match response.json().await {
            Ok(payload) => payload,
            Err(e) => {
                self.state.stats.record(false, started.elapsed());
                return Err(CloudProviderError::InvalidResponse {
                    provider: "CustomProvider",
                    operation: OP,
                    detail: e.to_string(),
                }
                .into());
            },
        };
        let latency = started.elapsed();

        let Some(text) = payload["choices"][0]["message"]["content"].as_str() else {
            self.state.stats.record(false, latency);
            return Err(CloudProviderError::InvalidResponse {
                provider: "CustomProvider",
                operation: OP,
                detail: "response contained no choices[0].message.content".to_string(),
            }
            .into());
        };
        let input_tokens = payload["usage"]["prompt_tokens"].as_u64().map(|v| v as u32);
        let output_tokens = payload["usage"]["completion_tokens"].as_u64().map(|v| v as u32);
        self.state.stats.record(true, latency);

        Ok(CloudInferenceResponse {
            request_id: request.request_id,
            provider: self.name.clone(),
            model_name: request.model_name,
            model_version: request.model_version,
            output_data: OutputData::Text(text.to_string()),
            metadata: metadata(
                latency,
                input_tokens,
                output_tokens,
                payload["choices"][0]["finish_reason"].as_str().map(str::to_string),
                HashMap::new(),
            ),
            performance: performance(latency, output_tokens),
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
        _request: ModelDeploymentRequest,
    ) -> Result<ModelDeploymentResponse> {
        Err(CloudProviderError::unsupported("CustomProvider", "deploy_model").into())
    }

    async fn update_deployment(
        &self,
        _deployment_id: &str,
        _config: ModelDeploymentRequest,
    ) -> Result<ModelDeploymentResponse> {
        Err(CloudProviderError::unsupported("CustomProvider", "update_deployment").into())
    }

    async fn delete_deployment(&self, _deployment_id: &str) -> Result<()> {
        Err(CloudProviderError::unsupported("CustomProvider", "delete_deployment").into())
    }

    async fn get_deployment_status(&self, _deployment_id: &str) -> Result<DeploymentStatus> {
        Err(CloudProviderError::unsupported("CustomProvider", "get_deployment_status").into())
    }

    async fn list_deployments(&self) -> Result<Vec<ModelDeploymentResponse>> {
        Err(CloudProviderError::unsupported("CustomProvider", "list_deployments").into())
    }

    async fn get_model_info(&self, _model_name: &str) -> Result<ModelInfo> {
        Err(CloudProviderError::unsupported("CustomProvider", "get_model_info").into())
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        let mut health = self.state.health();
        health.provider = self.name.clone();
        Ok(health)
    }

    async fn get_metrics(&self) -> Result<ProviderMetrics> {
        let mut metrics = self.state.metrics();
        metrics.provider = self.name.clone();
        Ok(metrics)
    }

    async fn get_cost_estimate(&self, _request: &CloudInferenceRequest) -> Result<f64> {
        Err(CloudProviderError::unsupported("CustomProvider", "get_cost_estimate").into())
    }

    fn get_provider_type(&self) -> CloudProviderType {
        CloudProviderType::Custom(self.name.clone())
    }

    fn supports_feature(&self, feature: &str) -> bool {
        matches!(feature, "batch")
    }
}

/// Number of deployments this process created through `provider`.
pub fn active_deployments(state: &ProviderState) -> u64 {
    state.stats.active_deployments.load(Ordering::Relaxed)
}
