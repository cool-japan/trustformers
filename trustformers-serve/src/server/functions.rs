//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{
    batching::aggregator::{ProcessingOutput, RequestInput},
    health::HealthStatus,
    openapi::ErrorResponse,
    polling::{LongPollRequest, LongPollResponse, LongPollingStats},
    shadow::{ShadowComparison, ShadowStats},
    ServerConfig, ServerError,
};
use anyhow::Result;
use axum::{
    extract::{Extension, Path, Query, WebSocketUpgrade},
    http::{header, StatusCode},
    response::{IntoResponse, Json, Response},
};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};
use sysinfo::System;

use super::jobs::{run_job, JobState};
use super::streams::run_stream;
use super::system_stats::{disk_usage_percentage, measure_host_async};
use super::types::{
    AsyncInferenceRequest, AsyncInferenceResponse, BatchInferenceRequest, BatchInferenceResponse,
    DetailedHealthResponse, FailoverRequest, HealthResponse, InferenceRequest, InferenceResponse,
    JobStatusResponse, ModelLoadRequest, ModelLoadResponse, ServiceHealthInfo, StatsResponse,
    TokenRequest, TrustformerServer,
};

static REQUEST_CACHE: LazyLock<Mutex<HashMap<String, String>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Grace period a streaming producer waits for the client to attach to the SSE
/// connection before it starts emitting chunks.
const STREAM_SUBSCRIBER_GRACE: std::time::Duration = std::time::Duration::from_millis(50);

/// Map an inference error onto the status code that describes it.
///
/// A missing model is a service-configuration problem (`503`), a prompt that
/// does not fit the model is a client problem (`400`), and anything else is an
/// internal fault (`500`).
pub(super) fn processing_error_to_server_error(error: &str) -> ServerError {
    if error.contains("no model configured") {
        ServerError::Overloaded
    } else if error.contains(crate::batching::processor::CONTEXT_WINDOW_EXCEEDED)
        || error.contains("requires a tokenizer")
        || error.contains("tokenized to zero tokens")
        || error.contains("empty token id sequence")
        || error.contains("are not supported by the text batch executor")
    {
        ServerError::InvalidRequest(error.to_string())
    } else {
        ServerError::Internal(anyhow::anyhow!("Processing error: {}", error))
    }
}

/// Count every HTTP request the server serves.
///
/// Registered unconditionally, unlike the auth middleware, so `/admin/stats`
/// reports a real request total whether or not authentication is configured.
pub(super) async fn request_counting_middleware(
    Extension(server): Extension<Arc<TrustformerServer>>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    server.record_request();
    next.run(request).await
}
/// Basic health check endpoint
#[utoipa::path(
    get,
    path = "/health",
    tag = "health",
    responses(
        (status = 200, description = "Service is healthy", body = HealthResponse),
        (status = 503, description = "Service is unhealthy", body = ErrorResponse)
    )
)]
pub(super) async fn health_check(
    Extension(state): Extension<Arc<TrustformerServer>>,
) -> Result<Json<HealthResponse>, ServerError> {
    let system_health = state.ha_service.get_system_health().await;
    let status = match system_health.status {
        HealthStatus::Healthy => "healthy",
        HealthStatus::Degraded => "degraded",
        HealthStatus::Unhealthy => "unhealthy",
    };
    Ok(Json(HealthResponse {
        status: status.to_string(),
        timestamp: chrono::Utc::now(),
        version: crate::VERSION.to_string(),
        uptime_seconds: state.uptime_seconds(),
    }))
}
/// Detailed health check endpoint
#[utoipa::path(
    get,
    path = "/health/detailed",
    tag = "health",
    responses(
        (
            status = 200,
            description = "Detailed health information",
            body = DetailedHealthResponse
        ),
        (status = 503, description = "Service is unhealthy", body = ErrorResponse)
    )
)]
pub(super) async fn detailed_health_check(
    Extension(state): Extension<Arc<TrustformerServer>>,
) -> Result<Json<DetailedHealthResponse>, ServerError> {
    let system_health = state.ha_service.get_system_health().await;
    let status = match system_health.status {
        HealthStatus::Healthy => "healthy",
        HealthStatus::Degraded => "degraded",
        HealthStatus::Unhealthy => "unhealthy",
    };
    // Every subsystem string below is derived from that subsystem's own state.
    let batching_stats = state.batching_service.get_stats().await;
    let batching = if !state.has_model() {
        "unavailable: no model configured".to_string()
    } else if batching_stats.processor_stats.total_batches > 0
        && batching_stats.processor_stats.success_rate < 0.5
    {
        format!(
            "degraded: batch success rate {:.0}%",
            batching_stats.processor_stats.success_rate * 100.0
        )
    } else {
        "healthy".to_string()
    };

    let caching = match state.caching_service.get_stats().await {
        Ok(_) => "healthy".to_string(),
        Err(e) => format!("degraded: {}", e),
    };

    let streaming_stats = state.streaming_service.get_stats().await;
    let streaming = format!("healthy: {} active streams", streaming_stats.active_streams);

    let failover = match system_health.status {
        HealthStatus::Healthy => "healthy".to_string(),
        HealthStatus::Degraded => "degraded".to_string(),
        HealthStatus::Unhealthy => "unhealthy".to_string(),
    };

    Ok(Json(DetailedHealthResponse {
        status: status.to_string(),
        timestamp: chrono::Utc::now(),
        version: crate::VERSION.to_string(),
        uptime_seconds: state.uptime_seconds(),
        system_health: state.system_health_info().await,
        services: ServiceHealthInfo {
            batching,
            caching,
            streaming,
            failover,
        },
        // Real circuit-breaker state, straight from the HA service's registry.
        circuit_breakers: serde_json::to_value(
            &state.ha_service.get_stats().await.circuit_breakers,
        )
        .unwrap_or(serde_json::Value::Null),
    }))
}
/// Readiness probe endpoint
#[utoipa::path(
    get,
    path = "/health/readiness",
    tag = "health",
    responses(
        (status = 200, description = "Service is ready"),
        (status = 503, description = "Service is not ready")
    )
)]
pub(super) async fn readiness_check(
    Extension(state): Extension<Arc<TrustformerServer>>,
) -> StatusCode {
    let system_health = state.ha_service.get_system_health().await;
    match system_health.status {
        HealthStatus::Healthy | HealthStatus::Degraded => StatusCode::OK,
        HealthStatus::Unhealthy => StatusCode::SERVICE_UNAVAILABLE,
    }
}
/// Liveness probe endpoint
#[utoipa::path(
    get,
    path = "/health/liveness",
    tag = "health",
    responses((status = 200, description = "Service is alive"))
)]
pub(super) async fn liveness_check() -> StatusCode {
    StatusCode::OK
}
/// Inference endpoint
#[utoipa::path(
    post,
    path = "/v1/inference",
    tag = "inference",
    request_body = InferenceRequest,
    responses(
        (
            status = 200,
            description = "Inference completed successfully",
            body = InferenceResponse
        ),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 429, description = "Rate limit exceeded", body = ErrorResponse),
        (status = 503, description = "Service overloaded", body = ErrorResponse)
    ),
    security(("bearer_auth" = []), ("api_key" = []))
)]
#[axum::debug_handler]
pub(super) async fn inference_endpoint(
    Extension(state): Extension<Arc<TrustformerServer>>,
    Json(request): Json<InferenceRequest>,
) -> Result<Json<InferenceResponse>, ServerError> {
    let start_time = std::time::Instant::now();
    let request_id = uuid::Uuid::new_v4().to_string();
    let (text, cache_hit, processing_time_ms) = if request.enable_cache.unwrap_or(false) {
        // The key must cover every input that changes the generated text;
        // keying on the prompt alone served a cached completion produced under
        // different generation parameters.
        let cache_key = format!(
            "{:?}|max_length={:?}|temperature={:?}|top_p={:?}|model={:?}",
            request.text, request.max_length, request.temperature, request.top_p, request.model
        );
        let cached_result = {
            let cache = REQUEST_CACHE
                .lock()
                .map_err(|e| ServerError::Internal(anyhow::anyhow!("Cache lock error: {}", e)))?;
            cache.get(&cache_key).cloned()
        };
        if let Some(cached_text) = cached_result {
            let elapsed = start_time.elapsed().as_millis() as f64;
            (cached_text, Some(true), elapsed)
        } else {
            let batch_request = crate::batching::Request {
                id: crate::batching::RequestId::new(),
                input: RequestInput::Text {
                    text: request.text.clone(),
                    max_length: Some(request.max_length.unwrap_or(100)),
                },
                priority: crate::batching::config::Priority::Normal,
                submitted_at: std::time::Instant::now(),
                deadline: None,
                metadata: std::collections::HashMap::new(),
            };
            let processing_result =
                state.batching_service().submit_request(batch_request).await.map_err(|e| {
                    ServerError::Internal(anyhow::anyhow!("Batching service error: {}", e))
                })?;
            let text = match processing_result.output {
                ProcessingOutput::Text(text) => text,
                ProcessingOutput::Tokens(tokens) => format!("Tokens: {:?}", tokens),
                ProcessingOutput::Error(error) => {
                    return Err(processing_error_to_server_error(&error));
                },
                _ => "Unsupported output type".to_string(),
            };
            {
                let mut cache = REQUEST_CACHE.lock().map_err(|e| {
                    ServerError::Internal(anyhow::anyhow!("Cache lock error: {}", e))
                })?;
                cache.insert(cache_key, text.clone());
            }
            (text, Some(false), processing_result.latency_ms as f64)
        }
    } else {
        let batch_request = crate::batching::Request {
            id: crate::batching::RequestId::new(),
            input: RequestInput::Text {
                text: request.text.clone(),
                max_length: Some(request.max_length.unwrap_or(100)),
            },
            priority: crate::batching::config::Priority::Normal,
            submitted_at: std::time::Instant::now(),
            deadline: None,
            metadata: std::collections::HashMap::new(),
        };
        let processing_result =
            state.batching_service().submit_request(batch_request).await.map_err(|e| {
                ServerError::Internal(anyhow::anyhow!("Batching service error: {}", e))
            })?;
        let text = match processing_result.output {
            ProcessingOutput::Text(text) => text,
            ProcessingOutput::Tokens(tokens) => format!("Tokens: {:?}", tokens),
            ProcessingOutput::Error(error) => {
                return Err(processing_error_to_server_error(&error));
            },
            _ => "Unsupported output type".to_string(),
        };
        (text, None, processing_result.latency_ms as f64)
    };
    // Shadow mode: run the request through the real shadow service and report the
    // real comparison. Nothing is synthesized from the primary output.
    let shadow_comparison = if request.shadow_mode.unwrap_or(false) {
        let mirrored_payload = serde_json::json!({
            "text": request.text,
            "max_length": request.max_length,
        });
        let production_payload = serde_json::json!({
            "text": text.clone(),
            "tokens": text.split_whitespace().collect::<Vec<_>>(),
        });

        match state
            .shadow_service
            .process_request_blocking(
                mirrored_payload,
                production_payload,
                processing_time_ms,
                std::time::Duration::from_secs(5),
            )
            .await
        {
            Ok(Some(comparison)) => {
                Some(serde_json::to_value(&comparison).unwrap_or(serde_json::Value::Null))
            },
            // Shadow testing is disabled or this request was not sampled: say so
            // rather than inventing a comparison.
            Ok(None) => Some(serde_json::json!({
                "compared": false,
                "reason": "shadow testing is disabled or this request was not sampled",
            })),
            Err(e) => Some(serde_json::json!({
                "compared": false,
                "error": e.to_string(),
            })),
        }
    } else {
        None
    };
    let response = InferenceResponse {
        request_id,
        text: text.clone(),
        tokens: text.split_whitespace().map(String::from).collect(),
        processing_time_ms,
        cache_hit,
        shadow_comparison,
    };
    Ok(Json(response))
}
/// Batch inference endpoint
#[utoipa::path(
    post,
    path = "/v1/inference/batch",
    tag = "inference",
    request_body = BatchInferenceRequest,
    responses(
        (
            status = 200,
            description = "Batch inference completed successfully",
            body = BatchInferenceResponse
        ),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 429, description = "Rate limit exceeded", body = ErrorResponse),
        (status = 503, description = "Service overloaded", body = ErrorResponse)
    ),
    security(("bearer_auth" = []), ("api_key" = []))
)]
pub(super) async fn batch_inference_endpoint(
    Extension(state): Extension<Arc<TrustformerServer>>,
    Json(request): Json<BatchInferenceRequest>,
) -> Result<Json<BatchInferenceResponse>, ServerError> {
    let start_time = std::time::Instant::now();
    let batch_id = uuid::Uuid::new_v4().to_string();
    let mut responses = Vec::new();
    let mut futures = Vec::new();
    for (i, req) in request.requests.iter().enumerate() {
        let batch_request = crate::batching::Request {
            id: crate::batching::RequestId::new(),
            input: RequestInput::Text {
                text: req.text.clone(),
                max_length: req.max_length,
            },
            priority: crate::batching::config::Priority::Normal,
            submitted_at: std::time::Instant::now(),
            deadline: None,
            metadata: std::collections::HashMap::new(),
        };
        let batching_service = state.batching_service().clone();
        let request_id = format!("{}_{}", batch_id, i);
        let future = async move {
            let result = batching_service.submit_request(batch_request).await;
            (request_id, result)
        };
        futures.push(future);
    }
    let results = futures::future::join_all(futures).await;
    for (request_id, result) in results {
        match result {
            Ok(processing_result) => {
                let text = match processing_result.output {
                    ProcessingOutput::Text(text) => text,
                    ProcessingOutput::Tokens(tokens) => format!("Tokens: {:?}", tokens),
                    ProcessingOutput::Error(error) => format!("Error: {}", error),
                    _ => "Unsupported output type".to_string(),
                };
                responses.push(InferenceResponse {
                    request_id,
                    text: text.clone(),
                    tokens: text.split_whitespace().map(String::from).collect(),
                    processing_time_ms: processing_result.latency_ms as f64,
                    cache_hit: None,
                    shadow_comparison: None,
                });
            },
            Err(e) => {
                responses.push(InferenceResponse {
                    request_id,
                    text: format!("Error: {}", e),
                    tokens: vec!["Error".to_string()],
                    processing_time_ms: 0.0,
                    cache_hit: None,
                    shadow_comparison: None,
                });
            },
        }
    }
    let batch_size = responses.len();
    Ok(Json(BatchInferenceResponse {
        batch_id,
        results: responses,
        batch_size,
        total_processing_time_ms: start_time.elapsed().as_millis() as f64,
    }))
}
/// Server-Sent Events streaming endpoint
pub(super) async fn sse_stream_endpoint(
    Extension(state): Extension<Arc<TrustformerServer>>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Response, ServerError> {
    let request_id = params.get("request_id").cloned();
    state
        .sse_handler
        .handle_connection(request_id)
        .await
        .map_err(ServerError::Internal)
}
/// WebSocket endpoint
pub(super) async fn websocket_endpoint(
    ws: WebSocketUpgrade,
    Extension(state): Extension<Arc<TrustformerServer>>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let request_id = params.get("request_id").cloned();
    state.websocket_handler.handle_upgrade(ws, request_id).await
}
/// Statistics endpoint
#[utoipa::path(
    get,
    path = "/admin/stats",
    tag = "admin",
    responses(
        (
            status = 200,
            description = "Service statistics retrieved successfully",
            body = StatsResponse
        ),
        (status = 503, description = "Service unavailable", body = ErrorResponse)
    ),
    security(("bearer_auth" = []), ("api_key" = []))
)]
pub(super) async fn get_stats(
    Extension(state): Extension<Arc<TrustformerServer>>,
) -> Result<Json<StatsResponse>, ServerError> {
    let batching_stats = {
        let stats = state.batching_service.get_stats().await;
        serde_json::to_value(&stats).unwrap_or_else(|_| {
            serde_json::json!(
                { "active_batches" : 0, "total_processed" : 0, "status" : "running" }
            )
        })
    };
    let caching_stats = match state.caching_service.get_stats().await {
        Ok(stats) => serde_json::to_value(&stats).unwrap_or_default(),
        Err(_) => serde_json::json!({ "error" : "failed to get caching stats" }),
    };
    let streaming_stats = {
        let stats = state.streaming_service.get_stats().await;
        serde_json::to_value(&stats).unwrap_or_else(|_| {
            serde_json::json!(
                { "active_streams" : 0, "total_bytes_streamed" : 0, "status" :
                "running" }
            )
        })
    };
    let ha_stats = {
        let stats = state.ha_service.get_stats().await;
        serde_json::to_value(&stats).unwrap_or_else(|_| {
            serde_json::json!(
                { "active_instances" : 1, "failover_count" : 0, "health_status" :
                "healthy", "status" : "running" }
            )
        })
    };
    // Measured process/host usage. `disk_percent` is null when the platform
    // could not attribute the working directory to a filesystem.
    let host = measure_host_async().await;
    let resource_usage = serde_json::json!({
        "process_memory_mb": host.process_memory_bytes as f64 / 1024.0 / 1024.0,
        "system_memory_used_mb": host.used_memory_bytes as f64 / 1024.0 / 1024.0,
        "system_memory_total_mb": host.total_memory_bytes as f64 / 1024.0 / 1024.0,
        "memory_percent": host.memory_percent,
        "cpu_percent": host.cpu_percent,
        "disk_percent": host.disk_percent,
    });
    let server_stats = serde_json::json!({
        "total_requests": state.total_requests(),
        "uptime_seconds": state.uptime_seconds(),
        "version": crate::VERSION,
        "model_configured": state.has_model(),
        "async_jobs_tracked": state.job_store().len(),
        "streams_tracked": state.stream_store().len(),
    });
    Ok(Json(StatsResponse {
        batching_stats,
        caching_stats,
        streaming_stats,
        ha_stats,
        resource_usage,
        server_stats,
    }))
}
/// Configuration endpoint
#[utoipa::path(
    get,
    path = "/admin/config",
    tag = "admin",
    responses(
        (
            status = 200,
            description = "Configuration retrieved successfully",
            body = serde_json::Value
        ),
        (status = 503, description = "Service unavailable", body = ErrorResponse)
    ),
    security(("bearer_auth" = []), ("api_key" = []))
)]
pub(super) async fn get_config(
    Extension(state): Extension<Arc<TrustformerServer>>,
) -> Result<Json<ServerConfig>, ServerError> {
    Ok(Json(state.config.clone()))
}
/// Metrics endpoint (Prometheus format)
#[utoipa::path(
    get,
    path = "/metrics",
    tag = "monitoring",
    responses(
        (
            status = 200,
            description = "Metrics retrieved successfully",
            content_type = "text/plain"
        ),
        (status = 503, description = "Service unavailable", body = ErrorResponse)
    )
)]
pub(super) async fn metrics_endpoint(
    Extension(state): Extension<Arc<TrustformerServer>>,
) -> Result<Response, ServerError> {
    // The prometheus registry owned by MetricsService is the single source of
    // truth. Counters that nothing in this process increments are simply absent
    // from the exposition; they are never invented so the page looks populated.
    let mut body =
        state.metrics_service.get_metrics().await.map_err(|e| {
            ServerError::Internal(anyhow::anyhow!("metrics encoding failed: {}", e))
        })?;

    // Serving-layer gauges that are measured here rather than in the registry.
    let batching_stats = state.batching_service.get_stats().await;
    let streaming_stats = state.streaming_service.get_stats().await;
    body.push_str(&render_serving_metrics(
        state.uptime_seconds(),
        state.total_requests(),
        state.has_model(),
        batching_stats.aggregator_stats.total_batches_formed as u64,
        batching_stats.processor_stats.total_requests as u64,
        batching_stats.processor_stats.failed_batches as u64,
        streaming_stats.active_streams as u64,
        state.job_store().len() as u64,
        state.stream_store().len() as u64,
    ));

    Ok((
        StatusCode::OK,
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
        .into_response())
}

/// Render the serving-layer gauges in Prometheus text exposition format.
#[allow(clippy::too_many_arguments)]
fn render_serving_metrics(
    uptime_seconds: f64,
    total_requests: u64,
    model_configured: bool,
    batches_formed: u64,
    requests_processed: u64,
    failed_batches: u64,
    active_streams: u64,
    async_jobs: u64,
    tracked_streams: u64,
) -> String {
    let mut out = String::new();
    out.push_str("# HELP trustformers_serve_uptime_seconds Seconds since the server started.\n");
    out.push_str("# TYPE trustformers_serve_uptime_seconds gauge\n");
    out.push_str(&format!(
        "trustformers_serve_uptime_seconds {uptime_seconds}\n"
    ));

    out.push_str("# HELP trustformers_serve_http_requests_total HTTP requests served.\n");
    out.push_str("# TYPE trustformers_serve_http_requests_total counter\n");
    out.push_str(&format!(
        "trustformers_serve_http_requests_total {total_requests}\n"
    ));

    out.push_str(
        "# HELP trustformers_serve_model_configured 1 when a real model backs the batch executor.\n",
    );
    out.push_str("# TYPE trustformers_serve_model_configured gauge\n");
    out.push_str(&format!(
        "trustformers_serve_model_configured {}\n",
        u8::from(model_configured)
    ));

    out.push_str(
        "# HELP trustformers_serve_batches_formed_total Batches formed by the aggregator.\n",
    );
    out.push_str("# TYPE trustformers_serve_batches_formed_total counter\n");
    out.push_str(&format!(
        "trustformers_serve_batches_formed_total {batches_formed}\n"
    ));

    out.push_str(
        "# HELP trustformers_serve_batched_requests_total Requests executed by the processor.\n",
    );
    out.push_str("# TYPE trustformers_serve_batched_requests_total counter\n");
    out.push_str(&format!(
        "trustformers_serve_batched_requests_total {requests_processed}\n"
    ));

    out.push_str(
        "# HELP trustformers_serve_failed_batches_total Batches whose executor errored.\n",
    );
    out.push_str("# TYPE trustformers_serve_failed_batches_total counter\n");
    out.push_str(&format!(
        "trustformers_serve_failed_batches_total {failed_batches}\n"
    ));

    out.push_str("# HELP trustformers_serve_active_streams Streaming connections in progress.\n");
    out.push_str("# TYPE trustformers_serve_active_streams gauge\n");
    out.push_str(&format!(
        "trustformers_serve_active_streams {active_streams}\n"
    ));

    out.push_str("# HELP trustformers_serve_async_jobs Async inference jobs tracked in memory.\n");
    out.push_str("# TYPE trustformers_serve_async_jobs gauge\n");
    out.push_str(&format!("trustformers_serve_async_jobs {async_jobs}\n"));

    out.push_str(
        "# HELP trustformers_serve_tracked_streams Streaming inference requests tracked in memory.\n",
    );
    out.push_str("# TYPE trustformers_serve_tracked_streams gauge\n");
    out.push_str(&format!(
        "trustformers_serve_tracked_streams {tracked_streams}\n"
    ));

    out
}
/// GraphQL handler endpoint (temporarily disabled due to axum compatibility)
/// GraphQL playground endpoint (temporarily disabled due to axum compatibility)
/// Long polling endpoint
#[utoipa::path(
    get,
    path = "/v1/poll",
    tag = "polling",
    params(
        (
            "event_types" = Option<String>,
            Query,
            description = "Comma-separated list of event types to listen for"
        ),
        ("client_id" = Option<String>, Query, description = "Unique client identifier"),
        (
            "timeout_seconds" = Option<u64>,
            Query,
            description = "Polling timeout in seconds"
        ),
        (
            "last_event_id" = Option<String>,
            Query,
            description = "Last received event ID for continuation"
        )
    ),
    responses(
        (
            status = 200,
            description = "Events received or timeout reached",
            body = LongPollResponse
        ),
        (status = 400, description = "Invalid request parameters", body = ErrorResponse)
    ),
    security(("bearer_auth" = []), ("api_key" = []))
)]
pub(super) async fn long_poll_endpoint(
    Extension(state): Extension<Arc<TrustformerServer>>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<LongPollResponse>, ServerError> {
    let request = LongPollRequest {
        event_types: params
            .get("event_types")
            .map(|s| s.split(',').map(|s| s.to_string()).collect())
            .unwrap_or_else(|| vec!["*".to_string()]),
        client_id: params.get("client_id").cloned(),
        timeout_seconds: params.get("timeout_seconds").and_then(|s| s.parse().ok()),
        last_event_id: params.get("last_event_id").cloned(),
    };
    let response = state.polling_service.poll(request).await.map_err(ServerError::Internal)?;
    Ok(Json(response))
}
/// Poll statistics endpoint
#[utoipa::path(
    get,
    path = "/v1/poll/stats",
    tag = "polling",
    responses(
        (
            status = 200,
            description = "Polling statistics retrieved successfully",
            body = LongPollingStats
        ),
        (status = 503, description = "Service unavailable", body = ErrorResponse)
    ),
    security(("bearer_auth" = []), ("api_key" = []))
)]
pub(super) async fn poll_stats_endpoint(
    Extension(state): Extension<Arc<TrustformerServer>>,
) -> Result<Json<crate::polling::LongPollingStats>, ServerError> {
    let stats = state.polling_service.get_stats().await;
    Ok(Json(stats))
}
/// Shadow testing statistics endpoint
#[utoipa::path(
    get,
    path = "/v1/shadow/stats",
    tag = "shadow",
    responses(
        (
            status = 200,
            description = "Shadow testing statistics retrieved successfully",
            body = ShadowStats
        ),
        (status = 503, description = "Service unavailable", body = ErrorResponse)
    ),
    security(("bearer_auth" = []), ("api_key" = []))
)]
pub(super) async fn shadow_stats_endpoint(
    Extension(state): Extension<Arc<TrustformerServer>>,
) -> Result<Json<ShadowStats>, ServerError> {
    let stats = state.shadow_service.get_stats().await;
    Ok(Json(stats))
}
/// Shadow testing results endpoint
#[utoipa::path(
    get,
    path = "/v1/shadow/results",
    tag = "shadow",
    params(
        (
            "limit" = Option<usize>,
            Query,
            description = "Maximum number of results to return"
        )
    ),
    responses(
        (
            status = 200,
            description = "Shadow testing results retrieved successfully",
            body = Vec<ShadowComparison>
        ),
        (status = 400, description = "Invalid request parameters", body = ErrorResponse),
        (status = 503, description = "Service unavailable", body = ErrorResponse)
    ),
    security(("bearer_auth" = []), ("api_key" = []))
)]
pub(super) async fn shadow_results_endpoint(
    Extension(state): Extension<Arc<TrustformerServer>>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<Vec<ShadowComparison>>, ServerError> {
    let limit = params.get("limit").and_then(|s| s.parse().ok());
    let results = state.shadow_service.get_shadow_results(limit).await;
    Ok(Json(results))
}
/// Shadow testing comparison endpoint
#[utoipa::path(
    get,
    path = "/v1/shadow/comparison/{id}",
    tag = "shadow",
    params(("id" = String, Path, description = "Comparison ID")),
    responses(
        (
            status = 200,
            description = "Shadow comparison retrieved successfully",
            body = ShadowComparison
        ),
        (status = 404, description = "Comparison not found", body = ErrorResponse),
        (status = 503, description = "Service unavailable", body = ErrorResponse)
    ),
    security(("bearer_auth" = []), ("api_key" = []))
)]
pub(super) async fn shadow_comparison_endpoint(
    Extension(state): Extension<Arc<TrustformerServer>>,
    Path(comparison_id): Path<String>,
) -> Result<Json<ShadowComparison>, ServerError> {
    let comparison = state
        .shadow_service
        .get_comparison(&comparison_id)
        .await
        .ok_or_else(|| ServerError::NotFound("Comparison not found".to_string()))?;
    Ok(Json(comparison))
}
/// Measured disk usage percentage for the filesystem holding the current
/// working directory.
///
/// # Errors
///
/// Returns an error when no mounted filesystem contains the working directory,
/// rather than reporting a plausible figure.
pub fn get_disk_usage_percentage() -> Result<f64> {
    disk_usage_percentage()
}
/// Extension-compatible authentication middleware
pub(super) async fn auth_extension_middleware(
    Extension(server): Extension<Arc<TrustformerServer>>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Result<axum::response::Response, axum::http::StatusCode> {
    use axum::http::header;
    let auth_service = match &server.auth_service {
        Some(service) => service.clone(),
        None => return Ok(next.run(request).await),
    };
    let skip_paths = [
        "/health",
        "/health/detailed",
        "/health/readiness",
        "/health/liveness",
        "/auth/login",
        "/auth/token",
    ];
    if skip_paths.contains(&request.uri().path()) {
        return Ok(next.run(request).await);
    }
    let auth_header = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .ok_or(axum::http::StatusCode::UNAUTHORIZED)?;
    if !auth_header.starts_with("Bearer ") {
        return Err(axum::http::StatusCode::UNAUTHORIZED);
    }
    let token = &auth_header[7..];
    auth_service
        .verify_token(token)
        .map_err(|_| axum::http::StatusCode::UNAUTHORIZED)?;
    Ok(next.run(request).await)
}
/// Streaming inference endpoint.
///
/// Registers a stream, attaches a real producer, and only then returns the
/// `stream_id`. The producer runs the request through the batching service and
/// pushes each generated chunk as an SSE `token` event to any client connected
/// on `GET /stream?request_id=<stream_id>`, followed by a terminal `completion`
/// or `error` event. Progress can also be read from
/// `GET /v1/inference/stream/{id}`.
pub(super) async fn streaming_inference_endpoint(
    Extension(state): Extension<Arc<TrustformerServer>>,
    Json(request): Json<InferenceRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let stream_id = uuid::Uuid::new_v4().to_string();
    state.stream_store().register(stream_id.clone());

    let store = Arc::clone(state.stream_store());
    let batching = Arc::clone(&state.batching_service);
    let sse = Arc::clone(&state.sse_handler);
    let text = request.text.clone();
    let max_length = request.max_length;
    let producer_stream_id = stream_id.clone();
    tokio::spawn(async move {
        run_stream(
            store,
            batching,
            sse,
            producer_stream_id,
            text,
            max_length,
            STREAM_SUBSCRIBER_GRACE,
        )
        .await;
    });

    Ok(Json(serde_json::json!({
        "stream_id": stream_id,
        "status": "streaming",
        "events_url": format!("/stream?request_id={}", stream_id),
        "status_url": format!("/v1/inference/stream/{}", stream_id),
    })))
}

/// Progress and result of a streaming inference request.
pub(super) async fn stream_status_endpoint(
    Extension(state): Extension<Arc<TrustformerServer>>,
    Path(stream_id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let record = state
        .stream_store()
        .get(&stream_id)
        .ok_or_else(|| ServerError::NotFound(format!("Unknown stream '{}'", stream_id)))?;
    Ok(Json(record.to_json()))
}
/// Memory pressure endpoint
pub(super) async fn memory_pressure_endpoint(
    Extension(_state): Extension<Arc<TrustformerServer>>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let mut sys = System::new_all();
    sys.refresh_memory();
    let total_memory = sys.total_memory();
    let used_memory = sys.used_memory();
    let current_memory_mb = (used_memory as f64) / 1024.0 / 1024.0;
    let total_memory_mb = (total_memory as f64) / 1024.0 / 1024.0;
    let usage_percent = (used_memory as f64 / total_memory as f64) * 100.0;
    let pressure_level = if usage_percent < 60.0 {
        "Low"
    } else if usage_percent < 80.0 {
        "Medium"
    } else {
        "High"
    };
    Ok(Json(serde_json::json!(
        { "status" : "ok", "current_memory_mb" : current_memory_mb,
        "total_memory_mb" : total_memory_mb, "usage_percent" : usage_percent,
        "pressure_level" : pressure_level, "timestamp" : chrono::Utc::now() }
    )))
}
/// GraphQL handler
///
/// Executes the request against the real `async-graphql` schema defined in
/// [`crate::graphql`], with the live server as the resolver context.
pub(super) async fn graphql_handler(
    Extension(state): Extension<Arc<TrustformerServer>>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let request: async_graphql::Request = serde_json::from_value(body).map_err(|e| {
        ServerError::InvalidRequest(format!("malformed GraphQL request body: {}", e))
    })?;

    let schema = crate::graphql::create_schema();
    let response = schema
        .execute(request.data(crate::graphql::create_context(Arc::clone(&state))))
        .await;

    serde_json::to_value(&response)
        .map(Json)
        .map_err(|e| ServerError::Internal(anyhow::anyhow!("GraphQL encoding failed: {}", e)))
}
/// GraphiQL playground, served from the `async-graphql` bundled source.
pub(super) async fn graphql_playground_handler() -> Result<axum::response::Html<String>, ServerError>
{
    Ok(axum::response::Html(async_graphql::http::graphiql_source(
        "/graphql", None,
    )))
}
/// Async inference endpoint.
///
/// Enqueues a real job and spawns a worker that drives it through the batching
/// service; the outcome is recorded in the job store and served by
/// `GET /jobs/{id}/status`.
pub(super) async fn async_inference_endpoint(
    Extension(state): Extension<Arc<TrustformerServer>>,
    Json(request): Json<AsyncInferenceRequest>,
) -> Result<(StatusCode, Json<AsyncInferenceResponse>), ServerError> {
    let job_id = uuid::Uuid::new_v4().to_string();
    state.job_store().enqueue(
        job_id.clone(),
        request.model.clone(),
        request.callback_url.clone(),
    );

    tracing::info!(
        "Async inference job submitted: job_id={}, model={}, text_len={}, callback_url={:?}",
        job_id,
        request.model,
        request.text.len(),
        request.callback_url
    );

    let store = Arc::clone(state.job_store());
    let batching = Arc::clone(&state.batching_service);
    let worker_job_id = job_id.clone();
    let text = request.text.clone();
    tokio::spawn(async move {
        run_job(store, batching, worker_job_id, text, None).await;
    });

    Ok((
        StatusCode::ACCEPTED,
        Json(AsyncInferenceResponse {
            job_id,
            status: JobState::Pending.as_str().to_string(),
        }),
    ))
}
/// Job status endpoint.
///
/// Reads the real job registry; an id that was never submitted is a 404.
pub(super) async fn job_status_endpoint(
    Extension(state): Extension<Arc<TrustformerServer>>,
    Path(job_id): Path<String>,
) -> Result<Json<JobStatusResponse>, ServerError> {
    let record = state
        .job_store()
        .get(&job_id)
        .ok_or_else(|| ServerError::NotFound(format!("Unknown job '{}'", job_id)))?;

    Ok(Json(JobStatusResponse {
        job_id: record.job_id.clone(),
        status: record.state.as_str().to_string(),
        result: record.result_payload(),
    }))
}
/// Model load endpoint.
///
/// Delegates to the real [`crate::model_management::ModelManager`]; a failed load
/// is reported as a failure, never as success.
pub(super) async fn model_load_endpoint(
    Extension(state): Extension<Arc<TrustformerServer>>,
    Json(request): Json<ModelLoadRequest>,
) -> Result<(StatusCode, Json<ModelLoadResponse>), ServerError> {
    tracing::info!(
        "Model load requested: name={}, version={}, device={}",
        request.model_name,
        request.model_version,
        request.device
    );

    let manager = state.model_manager();
    let load_config = crate::model_management::ModelLoadConfig {
        model_path: request.model_path.clone().unwrap_or_default(),
        revision: Some(request.model_version.clone()),
        precision: request.precision.clone().unwrap_or_else(|| "fp32".to_string()),
        device: request.device.clone(),
        max_batch_size: state.config.batching_config.max_batch_size,
        max_sequence_length: state.config.model_config.max_sequence_length,
        kv_cache_size: None,
        config_overrides: std::collections::HashMap::new(),
    };

    match manager
        .load_named_model(
            &request.model_name,
            &request.model_version,
            load_config,
            crate::model_management::LoadingStrategy::Eager,
        )
        .await
    {
        Ok(report) => Ok((
            StatusCode::OK,
            Json(ModelLoadResponse {
                success: true,
                model_name: request.model_name.clone(),
                message: format!(
                    "Model {} version {} loaded on {}: {} tensors, {} bytes resident",
                    request.model_name,
                    request.model_version,
                    request.device,
                    report.tensor_count,
                    report.memory_bytes
                ),
            }),
        )),
        Err(e) => Ok((
            StatusCode::BAD_REQUEST,
            Json(ModelLoadResponse {
                success: false,
                model_name: request.model_name.clone(),
                message: format!("Model load failed: {}", e),
            }),
        )),
    }
}
/// Issue an access token for valid credentials.
///
/// When no authentication service is configured the endpoint reports
/// `503 Service Unavailable`; it never hands out a canned token.
pub(super) async fn auth_token_handler(
    Extension(server): Extension<Arc<TrustformerServer>>,
    Json(request): Json<TokenRequest>,
) -> Result<Json<crate::auth::TokenResponse>, axum::http::StatusCode> {
    if request.username.is_empty() || request.password.is_empty() {
        return Err(axum::http::StatusCode::BAD_REQUEST);
    }
    let auth_service = match &server.auth_service {
        Some(service) => service,
        None => {
            tracing::warn!(
                "token requested but no authentication service is configured on this server"
            );
            return Err(axum::http::StatusCode::SERVICE_UNAVAILABLE);
        },
    };
    let user = auth_service
        .authenticate_user(&request.username, &request.password)
        .map_err(|_| axum::http::StatusCode::UNAUTHORIZED)?;
    let claims = crate::auth::Claims::new(
        user.id.clone(),
        "trustformers-serve".to_string(),
        "trustformers-api".to_string(),
        vec!["inference".to_string()],
        3600,
    );
    let token = auth_service
        .create_token(&claims)
        .map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(crate::auth::TokenResponse {
        access_token: token,
        token_type: "Bearer".to_string(),
        expires_in: 3600,
    }))
}
/// Admin failover endpoint.
///
/// Asks the HA service to fail over to `target_node`. The response reflects what
/// the HA service actually did.
pub(super) async fn admin_failover_endpoint(
    Extension(state): Extension<Arc<TrustformerServer>>,
    Json(request): Json<FailoverRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), ServerError> {
    let target = request.target_node.trim().to_string();
    if target.is_empty() {
        return Err(ServerError::InvalidRequest(
            "target_node must not be empty".to_string(),
        ));
    }

    match state.ha_service.trigger_failover(&target).await {
        Ok(outcome) => Ok((
            StatusCode::ACCEPTED,
            Json(serde_json::json!({
                "accepted": true,
                "target_node": target,
                "previous_node": outcome.previous_node,
                "active_node": outcome.active_node,
            })),
        )),
        Err(e) => Ok((
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "accepted": false,
                "target_node": target,
                "error": e.to_string(),
            })),
        )),
    }
}
/// Admin GPU status endpoint.
///
/// Enumerates real GPUs. An empty list means none were discovered on this host,
/// which is the truthful answer on a CPU-only machine.
pub(super) async fn admin_gpu_status_endpoint(
    Extension(_state): Extension<Arc<TrustformerServer>>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let devices = crate::resource_management::gpu_manager::discover_gpu_devices().await;
    match devices {
        Ok(devices) => Ok(Json(serde_json::json!({
            "gpus": devices,
            "available": !devices.is_empty(),
            "count": devices.len(),
        }))),
        Err(e) => Ok(Json(serde_json::json!({
            "gpus": [],
            "available": false,
            "count": 0,
            "error": e.to_string(),
        }))),
    }
}
/// OpenAPI JSON endpoint.
///
/// Serves the real `utoipa` specification, so the Swagger UI at `/docs` renders
/// every annotated endpoint.
pub(super) async fn openapi_json_endpoint() -> Result<Json<serde_json::Value>, ServerError> {
    use utoipa::OpenApi;
    let mut spec = crate::openapi::ApiDoc::openapi();
    // The document must advertise the version of the crate that is serving it.
    spec.info.version = crate::VERSION.to_string();
    serde_json::to_value(&spec)
        .map(Json)
        .map_err(|e| ServerError::Internal(anyhow::anyhow!("OpenAPI encoding failed: {}", e)))
}
/// Swagger UI endpoint
pub(super) async fn swagger_ui_endpoint() -> Result<axum::response::Html<String>, ServerError> {
    let html = r#"
<!DOCTYPE html>
<html>
<head>
    <title>TrustformeRS API Documentation</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {
            const ui = SwaggerUIBundle({
                url: "/api-docs/openapi.json",
                dom_id: '#swagger-ui',
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ]
            })
        }
    </script>
</body>
</html>
    "#
    .to_string();
    Ok(axum::response::Html(html))
}
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_server_creation() {
        let config = ServerConfig::default();
        let _server = TrustformerServer::new(config);
    }

    /// Regression: `get_disk_usage_percentage` used to return the constant 80.0
    /// after a pointless full system refresh.
    #[test]
    fn disk_usage_is_not_a_constant() {
        let usage = get_disk_usage_percentage().expect("the working directory is on a filesystem");
        assert!((0.0..=100.0).contains(&usage));

        let direct = super::super::system_stats::disk_usage_percentage()
            .expect("the working directory is on a filesystem");
        assert!((usage - direct).abs() < 1.0);
    }

    /// Regression: `/metrics` must render Prometheus text exposition containing
    /// only counters this process genuinely maintains.
    #[test]
    fn serving_metrics_render_prometheus_text() {
        let body = render_serving_metrics(12.5, 7, true, 3, 9, 1, 2, 4, 5);

        for expected in [
            "# TYPE trustformers_serve_uptime_seconds gauge",
            "trustformers_serve_uptime_seconds 12.5",
            "trustformers_serve_http_requests_total 7",
            "trustformers_serve_model_configured 1",
            "trustformers_serve_batches_formed_total 3",
            "trustformers_serve_batched_requests_total 9",
            "trustformers_serve_failed_batches_total 1",
            "trustformers_serve_active_streams 2",
            "trustformers_serve_async_jobs 4",
            "trustformers_serve_tracked_streams 5",
        ] {
            assert!(body.contains(expected), "missing {expected:?} in:\n{body}");
        }

        // The fabricated JSON counters must be gone for good.
        for forbidden in ["tokens_issued", "models_loaded", "gpu_scheduler", "Mock"] {
            assert!(
                !body.contains(forbidden),
                "fabricated metric {forbidden:?} is back"
            );
        }

        // Every non-comment line must be `name value`.
        for line in body.lines().filter(|l| !l.starts_with('#') && !l.is_empty()) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            assert_eq!(parts.len(), 2, "malformed exposition line: {line:?}");
            assert!(
                parts[1].parse::<f64>().is_ok(),
                "non-numeric metric value: {line:?}"
            );
        }
    }

    /// Regression: a server with no model must report that, rather than a
    /// uniformly healthy subsystem table.
    #[test]
    fn model_configured_gauge_tracks_reality() {
        let without = render_serving_metrics(1.0, 0, false, 0, 0, 0, 0, 0, 0);
        assert!(without.contains("trustformers_serve_model_configured 0"));
    }
}
