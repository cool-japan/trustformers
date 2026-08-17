//! Cloud provider tests against local mock servers.
//!
//! Each test spins an in-process axum server, points the provider's
//! `endpoints.inference_endpoint` at it, and asserts both the request the
//! provider really sent and the response it really parsed. No network access
//! and no credentials are involved.

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::{get, post};
use axum::Router;
use tokio::sync::Mutex;

use super::errors::CloudProviderError;
use super::*;

/// One request captured by a mock endpoint.
#[derive(Debug, Clone)]
struct Captured {
    path: String,
    headers: HashMap<String, String>,
    body: serde_json::Value,
}

type Recorder = Arc<Mutex<Vec<Captured>>>;

async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind loopback");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    format!("http://{addr}")
}

fn header_map(headers: &HeaderMap) -> HashMap<String, String> {
    headers
        .iter()
        .map(|(name, value)| {
            (
                name.as_str().to_ascii_lowercase(),
                value.to_str().unwrap_or_default().to_string(),
            )
        })
        .collect()
}

fn capture(path: &str, headers: &HeaderMap, body: &Bytes) -> Captured {
    Captured {
        path: path.to_string(),
        headers: header_map(headers),
        body: serde_json::from_slice(body).unwrap_or(serde_json::Value::Null),
    }
}

fn json_response(value: serde_json::Value) -> (StatusCode, HeaderMap, String) {
    let mut headers = HeaderMap::new();
    headers.insert("content-type", "application/json".parse().expect("header"));
    (StatusCode::OK, headers, value.to_string())
}

fn provider_config(name: &str, provider_type: CloudProviderType, base: &str) -> ProviderConfig {
    ProviderConfig {
        name: name.to_string(),
        provider_type,
        enabled: true,
        priority: 1,
        region: "us-central1".to_string(),
        credentials: CredentialsConfig {
            access_key_id: None,
            secret_access_key: None,
            session_token: None,
            service_account_key: None,
            client_id: Some("test-project".to_string()),
            client_secret: None,
            tenant_id: None,
            api_key: Some("test-api-key".to_string()),
            oauth_token: Some("test-oauth-token".to_string()),
            custom_headers: HashMap::new(),
        },
        endpoints: EndpointConfig {
            inference_endpoint: base.to_string(),
            model_management_endpoint: Some(base.to_string()),
            training_endpoint: None,
            monitoring_endpoint: None,
            custom_endpoints: HashMap::new(),
        },
        limits: LimitsConfig {
            max_requests_per_second: 10,
            max_concurrent_requests: 4,
            max_payload_size_bytes: 1 << 20,
            max_response_size_bytes: 1 << 20,
            request_timeout_seconds: 30,
            batch_size_limit: 8,
        },
        features: FeatureConfig {
            supports_streaming: false,
            supports_batch_inference: true,
            supports_model_deployment: false,
            supports_auto_scaling: false,
            supports_monitoring: false,
            supports_a_b_testing: false,
            supports_custom_models: false,
            supported_model_formats: vec!["text".to_string()],
            supported_data_types: vec!["text".to_string()],
        },
    }
}

fn text_request(model: &str, text: &str) -> CloudInferenceRequest {
    CloudInferenceRequest {
        request_id: "req-1".to_string(),
        model_name: model.to_string(),
        model_version: None,
        input_data: InputData::Text(text.to_string()),
        parameters: HashMap::new(),
        output_config: OutputConfig {
            format: OutputFormat::Text,
            max_tokens: Some(64),
            temperature: Some(0.3),
            top_p: Some(0.9),
            top_k: None,
            stream: false,
            include_probabilities: false,
            return_metadata: true,
        },
        priority: RequestPriority::Normal,
        timeout_seconds: Some(30),
        callback_url: None,
        metadata: HashMap::new(),
    }
}

fn text_of(response: &CloudInferenceResponse) -> String {
    match &response.output_data {
        OutputData::Text(text) => text.clone(),
        other => panic!("expected text output, got {other:?}"),
    }
}

// ── OpenAI ───────────────────────────────────────────────────────────────────

async fn openai_chat(
    State(recorder): State<Recorder>,
    headers: HeaderMap,
    body: Bytes,
) -> (StatusCode, HeaderMap, String) {
    recorder.lock().await.push(capture("/v1/chat/completions", &headers, &body));
    json_response(serde_json::json!({
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "the real completion"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}
    }))
}

#[tokio::test]
async fn test_openai_provider_calls_chat_completions() {
    let recorder: Recorder = Arc::new(Mutex::new(Vec::new()));
    let base = spawn(
        Router::new()
            .route("/v1/chat/completions", post(openai_chat))
            .with_state(Arc::clone(&recorder)),
    )
    .await;

    let provider = OpenAiProvider::new().await.expect("provider");
    provider
        .initialize(&provider_config(
            "openai",
            CloudProviderType::OpenAiApi,
            &base,
        ))
        .await
        .expect("initialize");

    let response = provider
        .inference(text_request("gpt-4o-mini", "hello there"))
        .await
        .expect("inference");

    assert_eq!(text_of(&response), "the real completion");
    assert_ne!(text_of(&response), "Mock response");
    assert_eq!(response.metadata.input_tokens, Some(11));
    assert_eq!(response.metadata.output_tokens, Some(4));
    assert_eq!(response.metadata.finish_reason.as_deref(), Some("stop"));
    assert_eq!(
        response.metadata.confidence_score, None,
        "a text API returns no confidence score"
    );
    assert!(
        !response.cost.reported_by_provider,
        "OpenAI returns no billing data with a completion"
    );
    assert_eq!(response.cost.cost_usd, 0.0);

    let captured = recorder.lock().await.clone();
    let request = captured.first().expect("a request must have been sent");
    assert_eq!(request.path, "/v1/chat/completions");
    assert_eq!(
        request.headers.get("authorization").map(String::as_str),
        Some("Bearer test-api-key")
    );
    assert_eq!(request.body["model"], serde_json::json!("gpt-4o-mini"));
    assert_eq!(
        request.body["messages"][0]["content"],
        serde_json::json!("hello there")
    );
    assert_eq!(request.body["max_tokens"], serde_json::json!(64));
}

#[tokio::test]
async fn test_openai_provider_without_credentials_is_rejected() {
    let provider = OpenAiProvider::new().await.expect("provider");
    let mut config = provider_config("openai", CloudProviderType::OpenAiApi, "http://127.0.0.1:1");
    config.credentials.api_key = None;
    provider.initialize(&config).await.expect("initialize");

    let err = provider
        .inference(text_request("gpt-4o-mini", "hi"))
        .await
        .expect_err("no API key means no completion");
    assert!(matches!(
        err.downcast_ref::<CloudProviderError>(),
        Some(CloudProviderError::MissingCredentials { .. })
    ));
}

#[tokio::test]
async fn test_uninitialized_provider_is_rejected() {
    let provider = OpenAiProvider::new().await.expect("provider");
    let err = provider
        .inference(text_request("gpt-4o-mini", "hi"))
        .await
        .expect_err("an uninitialized provider cannot serve");
    assert!(matches!(
        err.downcast_ref::<CloudProviderError>(),
        Some(CloudProviderError::NotInitialized { .. })
    ));
}

#[tokio::test]
async fn test_openai_provider_surfaces_api_errors() {
    let base = spawn(Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            (
                StatusCode::TOO_MANY_REQUESTS,
                "{\"error\":{\"message\":\"rate limited\"}}",
            )
        }),
    ))
    .await;

    let provider = OpenAiProvider::new().await.expect("provider");
    provider
        .initialize(&provider_config(
            "openai",
            CloudProviderType::OpenAiApi,
            &base,
        ))
        .await
        .expect("initialize");

    let err = provider
        .inference(text_request("gpt-4o-mini", "hi"))
        .await
        .expect_err("HTTP 429 must not be reported as a completion");
    match err.downcast_ref::<CloudProviderError>() {
        Some(CloudProviderError::ApiError { status, .. }) => assert_eq!(*status, 429),
        other => panic!("expected ApiError, got {other:?}"),
    }
}

#[tokio::test]
async fn test_metrics_and_health_reflect_observed_requests() {
    let recorder: Recorder = Arc::new(Mutex::new(Vec::new()));
    let base = spawn(
        Router::new()
            .route("/v1/chat/completions", post(openai_chat))
            .route(
                "/v1/models",
                get(|| async { "{\"object\":\"list\",\"data\":[]}" }),
            )
            .with_state(Arc::clone(&recorder)),
    )
    .await;

    let provider = OpenAiProvider::new().await.expect("provider");
    provider
        .initialize(&provider_config(
            "openai",
            CloudProviderType::OpenAiApi,
            &base,
        ))
        .await
        .expect("initialize");

    // Before any request, nothing may be claimed. `get_metrics` is a pure
    // read of the observed counters and issues no probe of its own.
    let metrics = provider.get_metrics().await.expect("metrics");
    assert_eq!(metrics.average_latency_ms, 0);
    assert_eq!(metrics.requests_per_second, 0.0);
    assert_eq!(
        metrics.cost_per_hour, None,
        "spend is a billing figure the client cannot observe"
    );
    assert!(
        metrics.resource_utilization.is_none(),
        "a cloud API exposes no host utilisation to its clients"
    );

    // `health_check` really probes the API, so it turns "unknown" into a
    // measured verdict.
    let health = provider.health_check().await.expect("health");
    assert_eq!(health.status, "healthy");
    assert!(health.response_time_ms > 0 || health.availability == 1.0);
    assert_eq!(health.error_rate, 0.0);

    provider.inference(text_request("gpt-4o-mini", "hi")).await.expect("inference");
    let metrics = provider.get_metrics().await.expect("metrics");
    assert_eq!(metrics.error_rate, 0.0);
    assert_ne!(
        metrics.provider, "",
        "metrics must name the provider that produced them"
    );

    // A failing probe must move the verdict, not be swallowed.
    let broken = OpenAiProvider::new().await.expect("provider");
    broken
        .initialize(&provider_config(
            "openai",
            CloudProviderType::OpenAiApi,
            "http://127.0.0.1:1",
        ))
        .await
        .expect("initialize");
    let health = broken.health_check().await.expect("health");
    assert_eq!(health.status, "unhealthy");
    assert_eq!(health.availability, 0.0);
}

#[tokio::test]
async fn test_health_is_unknown_before_anything_is_observed() {
    let provider = OpenAiProvider::new().await.expect("provider");
    provider
        .initialize(&provider_config(
            "openai",
            CloudProviderType::OpenAiApi,
            "http://127.0.0.1:1",
        ))
        .await
        .expect("initialize");
    // Regression test: the old provider always reported availability 0.99 and
    // five active deployments, whatever had happened.
    let health = provider.state().health();
    assert_eq!(health.status, "unknown");
    assert_eq!(health.active_deployments, 0);
    assert_eq!(health.response_time_ms, 0);
}

// ── Anthropic ────────────────────────────────────────────────────────────────

async fn anthropic_messages(
    State(recorder): State<Recorder>,
    headers: HeaderMap,
    body: Bytes,
) -> (StatusCode, HeaderMap, String) {
    recorder.lock().await.push(capture("/v1/messages", &headers, &body));
    json_response(serde_json::json!({
        "id": "msg_mock",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4",
        "content": [
            {"type": "text", "text": "claude says "},
            {"type": "text", "text": "hello"}
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 9, "output_tokens": 3}
    }))
}

#[tokio::test]
async fn test_anthropic_provider_calls_v1_messages_with_the_right_headers() {
    let recorder: Recorder = Arc::new(Mutex::new(Vec::new()));
    let base = spawn(
        Router::new()
            .route("/v1/messages", post(anthropic_messages))
            .with_state(Arc::clone(&recorder)),
    )
    .await;

    let provider = AnthropicProvider::new().await.expect("provider");
    provider
        .initialize(&provider_config(
            "anthropic",
            CloudProviderType::AnthropicClaude,
            &base,
        ))
        .await
        .expect("initialize");

    let response = provider
        .inference(text_request("claude-sonnet-4", "hello"))
        .await
        .expect("inference");

    assert_eq!(text_of(&response), "claude says hello");
    assert_eq!(response.metadata.input_tokens, Some(9));
    assert_eq!(response.metadata.output_tokens, Some(3));
    assert_eq!(response.metadata.finish_reason.as_deref(), Some("end_turn"));

    let captured = recorder.lock().await.clone();
    let request = captured.first().expect("a request must have been sent");
    assert_eq!(request.path, "/v1/messages");
    assert_eq!(
        request.headers.get("x-api-key").map(String::as_str),
        Some("test-api-key"),
        "Anthropic authenticates with x-api-key, not a bearer token"
    );
    assert_eq!(
        request.headers.get("anthropic-version").map(String::as_str),
        Some("2023-06-01")
    );
    assert_eq!(request.body["max_tokens"], serde_json::json!(64));
    assert_eq!(
        request.body["messages"][0]["content"],
        serde_json::json!("hello")
    );
}

// ── HuggingFace ──────────────────────────────────────────────────────────────

async fn hf_generate(
    State(recorder): State<Recorder>,
    headers: HeaderMap,
    body: Bytes,
) -> (StatusCode, HeaderMap, String) {
    recorder.lock().await.push(capture("/models", &headers, &body));
    json_response(serde_json::json!([{"generated_text": "hf output"}]))
}

#[tokio::test]
async fn test_huggingface_provider_posts_to_the_model_route() {
    let recorder: Recorder = Arc::new(Mutex::new(Vec::new()));
    let base = spawn(
        Router::new()
            .route("/models/{*model}", post(hf_generate))
            .with_state(Arc::clone(&recorder)),
    )
    .await;

    let provider = HuggingFaceProvider::new().await.expect("provider");
    provider
        .initialize(&provider_config(
            "hf",
            CloudProviderType::HuggingFaceInference,
            &base,
        ))
        .await
        .expect("initialize");

    let response = provider
        .inference(text_request("meta-llama/Llama-3-8B", "hello"))
        .await
        .expect("inference");
    assert_eq!(text_of(&response), "hf output");
    assert_eq!(
        response.metadata.input_tokens, None,
        "the Inference API returns no token accounting, so none may be claimed"
    );

    let captured = recorder.lock().await.clone();
    let request = captured.first().expect("a request must have been sent");
    assert_eq!(
        request.headers.get("authorization").map(String::as_str),
        Some("Bearer test-api-key")
    );
    assert_eq!(request.body["inputs"], serde_json::json!("hello"));
    assert_eq!(
        request.body["parameters"]["max_new_tokens"],
        serde_json::json!(64)
    );
}

// ── Vertex AI ────────────────────────────────────────────────────────────────

async fn vertex_generate(
    State(recorder): State<Recorder>,
    headers: HeaderMap,
    body: Bytes,
) -> (StatusCode, HeaderMap, String) {
    recorder.lock().await.push(capture("/vertex", &headers, &body));
    json_response(serde_json::json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "vertex output"}]},
            "finishReason": "STOP"
        }],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2}
    }))
}

#[tokio::test]
async fn test_vertex_provider_uses_generate_content() {
    let recorder: Recorder = Arc::new(Mutex::new(Vec::new()));
    let base = spawn(
        Router::new()
            .route("/v1/projects/{*rest}", post(vertex_generate))
            .with_state(Arc::clone(&recorder)),
    )
    .await;

    let provider = GoogleVertexAiProvider::new().await.expect("provider");
    provider
        .initialize(&provider_config(
            "vertex",
            CloudProviderType::GoogleVertexAi,
            &base,
        ))
        .await
        .expect("initialize");

    let response = provider
        .inference(text_request("gemini-2.0-flash", "hello"))
        .await
        .expect("inference");
    assert_eq!(text_of(&response), "vertex output");
    assert_eq!(response.metadata.input_tokens, Some(5));
    assert_eq!(response.metadata.output_tokens, Some(2));

    let captured = recorder.lock().await.clone();
    let request = captured.first().expect("a request must have been sent");
    assert_eq!(
        request.headers.get("authorization").map(String::as_str),
        Some("Bearer test-oauth-token")
    );
    assert_eq!(
        request.body["contents"][0]["parts"][0]["text"],
        serde_json::json!("hello")
    );
    assert_eq!(
        request.body["generationConfig"]["maxOutputTokens"],
        serde_json::json!(64)
    );
}

#[tokio::test]
async fn test_vertex_provider_requires_a_project_id() {
    let provider = GoogleVertexAiProvider::new().await.expect("provider");
    let mut config = provider_config(
        "vertex",
        CloudProviderType::GoogleVertexAi,
        "http://127.0.0.1:1",
    );
    config.credentials.client_id = None;
    provider.initialize(&config).await.expect("initialize");

    let err = provider
        .inference(text_request("gemini", "hi"))
        .await
        .expect_err("no project id means no call");
    assert!(matches!(
        err.downcast_ref::<CloudProviderError>(),
        Some(CloudProviderError::MissingConfiguration { .. })
    ));
}

// ── Azure ────────────────────────────────────────────────────────────────────

async fn azure_chat(
    State(recorder): State<Recorder>,
    headers: HeaderMap,
    body: Bytes,
) -> (StatusCode, HeaderMap, String) {
    recorder.lock().await.push(capture("/chat/completions", &headers, &body));
    json_response(serde_json::json!({
        "id": "azure-mock",
        "model": "phi-4",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "azure output"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2}
    }))
}

#[tokio::test]
async fn test_azure_provider_uses_the_api_key_header() {
    let recorder: Recorder = Arc::new(Mutex::new(Vec::new()));
    let base = spawn(
        Router::new()
            .route("/chat/completions", post(azure_chat))
            .with_state(Arc::clone(&recorder)),
    )
    .await;

    let provider = AzureMachineLearningProvider::new().await.expect("provider");
    provider
        .initialize(&provider_config(
            "azure",
            CloudProviderType::AzureMachineLearning,
            &base,
        ))
        .await
        .expect("initialize");

    let response = provider.inference(text_request("phi-4", "hello")).await.expect("inference");
    assert_eq!(text_of(&response), "azure output");

    let captured = recorder.lock().await.clone();
    let request = captured.first().expect("a request must have been sent");
    assert_eq!(
        request.headers.get("api-key").map(String::as_str),
        Some("test-api-key")
    );
}

#[tokio::test]
async fn test_azure_provider_requires_an_endpoint() {
    let provider = AzureMachineLearningProvider::new().await.expect("provider");
    let mut config = provider_config("azure", CloudProviderType::AzureMachineLearning, "");
    config.endpoints.inference_endpoint = String::new();
    provider.initialize(&config).await.expect("initialize");
    let err = provider
        .inference(text_request("phi-4", "hi"))
        .await
        .expect_err("Azure endpoints are per-deployment");
    assert!(matches!(
        err.downcast_ref::<CloudProviderError>(),
        Some(CloudProviderError::MissingConfiguration { .. })
    ));
}

// ── Custom gateway ───────────────────────────────────────────────────────────

#[tokio::test]
async fn test_custom_provider_talks_to_an_openai_compatible_gateway() {
    let recorder: Recorder = Arc::new(Mutex::new(Vec::new()));
    let base = spawn(
        Router::new()
            .route("/v1/chat/completions", post(openai_chat))
            .with_state(Arc::clone(&recorder)),
    )
    .await;

    let provider = CustomProvider::new("my-gateway").await.expect("provider");
    provider
        .initialize(&provider_config(
            "my-gateway",
            CloudProviderType::Custom("my-gateway".to_string()),
            &base,
        ))
        .await
        .expect("initialize");

    let response = provider.inference(text_request("local", "hello")).await.expect("inference");
    assert_eq!(response.provider, "my-gateway");
    assert_eq!(text_of(&response), "the real completion");
    assert_ne!(text_of(&response), "Custom provider response");
}

// ── Cross-provider honesty checks ────────────────────────────────────────────

#[tokio::test]
async fn test_deployment_is_unsupported_on_api_only_providers() {
    let request = ModelDeploymentRequest {
        deployment_id: "d1".to_string(),
        model_name: "m".to_string(),
        model_version: "1".to_string(),
        model_artifact_uri: "s3://bucket/model.tar.gz".to_string(),
        instance_type: "ml.g5.xlarge".to_string(),
        instance_count: 1,
        auto_scaling_config: None,
        environment_variables: HashMap::new(),
        resource_requirements: ResourceRequirements {
            cpu_cores: 4.0,
            memory_gb: 16.0,
            gpu_count: 1,
            gpu_type: None,
            storage_gb: 100.0,
            network_bandwidth_mbps: None,
        },
        deployment_config: DeploymentConfig {
            enable_logging: true,
            enable_monitoring: true,
            enable_data_capture: false,
            health_check_grace_period_seconds: 60,
            rolling_update_strategy: RollingUpdateStrategy {
                max_unavailable_percent: 0,
                max_surge_percent: 100,
                update_interval_seconds: 30,
            },
            security_config: SecurityConfig {
                enable_vpc: false,
                vpc_id: None,
                subnet_ids: vec![],
                security_group_ids: vec![],
                enable_encryption: true,
                kms_key_id: None,
                iam_role_arn: None,
            },
        },
    };

    let provider = OpenAiProvider::new().await.expect("provider");
    let err = provider
        .deploy_model(request)
        .await
        .expect_err("the OpenAI API has no model-deployment endpoint");
    match err.downcast_ref::<CloudProviderError>() {
        Some(CloudProviderError::UnsupportedOperation { .. }) => {},
        other => panic!("expected UnsupportedOperation, got {other:?}"),
    }
    assert!(
        !err.to_string().contains("example.com"),
        "no fabricated endpoint URL may appear"
    );
}

#[tokio::test]
async fn test_cost_estimate_is_not_invented() {
    let provider = AnthropicProvider::new().await.expect("provider");
    let err = provider
        .get_cost_estimate(&text_request("claude", "hi"))
        .await
        .expect_err("no price list is configured, so no estimate may be produced");
    assert!(matches!(
        err.downcast_ref::<CloudProviderError>(),
        Some(CloudProviderError::UnsupportedOperation { .. })
    ));
}

#[tokio::test]
async fn test_non_text_input_is_rejected_by_text_providers() {
    let base = spawn(Router::new().route("/v1/chat/completions", post(|| async { "{}" }))).await;
    let provider = OpenAiProvider::new().await.expect("provider");
    provider
        .initialize(&provider_config(
            "openai",
            CloudProviderType::OpenAiApi,
            &base,
        ))
        .await
        .expect("initialize");

    let mut request = text_request("gpt-4o-mini", "hi");
    request.input_data = InputData::Audio(vec![0, 1, 2]);
    let err = provider
        .inference(request)
        .await
        .expect_err("a chat-completions endpoint cannot take raw audio");
    match err.downcast_ref::<CloudProviderError>() {
        Some(CloudProviderError::UnsupportedInput { input_kind, .. }) => {
            assert_eq!(*input_kind, "audio")
        },
        other => panic!("expected UnsupportedInput, got {other:?}"),
    }
}

#[tokio::test]
async fn test_sagemaker_without_clients_reports_missing_credentials() {
    let provider = AwsSagemakerProvider::new().await.expect("provider");
    let err = provider
        .inference(text_request("my-endpoint", "hi"))
        .await
        .expect_err("no SDK client means no invocation");
    assert!(matches!(
        err.downcast_ref::<CloudProviderError>(),
        Some(CloudProviderError::MissingCredentials { .. })
    ));
    assert!(
        !err.to_string().contains("Mock response"),
        "no fabricated completion may be produced"
    );
}

#[tokio::test]
async fn test_manager_routes_to_a_real_provider() {
    let recorder: Recorder = Arc::new(Mutex::new(Vec::new()));
    let base = spawn(
        Router::new()
            .route("/v1/chat/completions", post(openai_chat))
            .route("/v1/models", get(|| async { "{\"data\":[]}" }))
            .with_state(Arc::clone(&recorder)),
    )
    .await;

    let mut config = CloudProviderConfig::default();
    config.providers = vec![provider_config(
        "openai",
        CloudProviderType::OpenAiApi,
        &base,
    )];
    config.default_provider = Some("openai".to_string());

    let manager = CloudProviderManager::new(config).await.expect("manager");
    let response = manager
        .inference(text_request("gpt-4o-mini", "hello"))
        .await
        .expect("manager inference");
    assert_eq!(text_of(&response), "the real completion");

    let stats = manager.get_stats().await;
    assert_eq!(stats.total_requests, 1);
    assert_eq!(stats.successful_requests, 1);
    assert_eq!(
        stats.total_cost_usd, 0.0,
        "an unreported cost must not accumulate into a spend figure"
    );
}
