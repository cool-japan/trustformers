//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::*;
use crate::{
    auth::AuthService,
    batching::DynamicBatchingService,
    caching::CachingService,
    health::{HAConfig, HighAvailabilityService},
    metrics::MetricsService,
    polling::LongPollingService,
    shadow::ShadowTestingService,
    streaming::{SseHandler, StreamingService, WebSocketHandler},
    ServerConfig,
};
use anyhow::Result;
use axum::{
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use sysinfo::System;
use tokio::net::TcpListener;
use tower::ServiceBuilder;
use tower_http::{cors::CorsLayer, trace::TraceLayer};

/// Batch inference request
#[derive(Debug, Deserialize, utoipa::ToSchema)]
#[schema(
    example = json!(
        {"requests":[{"text":"Hello world",
        "max_length":50,
        "temperature":0.7},
        {"text":"How are you?",
        "max_length":50,
        "temperature":0.8}]}
    )
)]
pub struct BatchInferenceRequest {
    pub(crate) requests: Vec<InferenceRequest>,
}
/// Job status response
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct JobStatusResponse {
    pub(crate) job_id: String,
    pub(crate) status: String,
    pub(crate) result: Option<serde_json::Value>,
}
/// HTTP server for TrustformeRS inference serving
#[derive(Clone)]
pub struct TrustformerServer {
    pub(crate) config: ServerConfig,
    pub(crate) batching_service: Arc<DynamicBatchingService>,
    pub(crate) caching_service: Arc<CachingService>,
    pub(crate) streaming_service: Arc<StreamingService>,
    pub(crate) sse_handler: Arc<SseHandler>,
    pub(crate) websocket_handler: Arc<WebSocketHandler>,
    pub(crate) ha_service: Arc<HighAvailabilityService>,
    pub(crate) metrics_service: Arc<MetricsService>,
    pub(crate) polling_service: Arc<LongPollingService>,
    pub(crate) shadow_service: Arc<ShadowTestingService>,
    pub(crate) auth_service: Option<Arc<AuthService>>,
    pub(crate) startup_time: Instant,
}
impl TrustformerServer {
    /// Get batching service
    pub fn batching_service(&self) -> &Arc<DynamicBatchingService> {
        &self.batching_service
    }
    /// Get caching service
    pub fn caching_service(&self) -> &Arc<CachingService> {
        &self.caching_service
    }
    /// Get streaming service
    pub fn streaming_service(&self) -> &Arc<StreamingService> {
        &self.streaming_service
    }
    /// Get HA service
    pub fn ha_service(&self) -> &Arc<HighAvailabilityService> {
        &self.ha_service
    }
    /// Get metrics service
    pub fn metrics_service(&self) -> &Arc<MetricsService> {
        &self.metrics_service
    }
    /// Create a new server instance
    pub fn new(config: ServerConfig) -> Self {
        let batching_service =
            Arc::new(DynamicBatchingService::new(config.batching_config.clone()));
        let caching_service = Arc::new(CachingService::new(config.caching_config.clone()));
        let streaming_service = Arc::new(StreamingService::new(config.streaming_config.clone()));
        let sse_handler = Arc::new(SseHandler::new(config.streaming_config.sse_config.clone()));
        let websocket_handler = Arc::new(WebSocketHandler::new(
            config.streaming_config.ws_config.clone(),
        ));
        let ha_service = Arc::new(HighAvailabilityService::new(HAConfig::default()));
        let metrics_service = Arc::new(MetricsService::default());
        let polling_service = Arc::new(LongPollingService::new(config.polling_config.clone()));
        let shadow_service = Arc::new(ShadowTestingService::new(config.shadow_config.clone()));
        Self {
            config,
            batching_service,
            caching_service,
            streaming_service,
            sse_handler,
            websocket_handler,
            ha_service,
            metrics_service,
            polling_service,
            shadow_service,
            auth_service: None,
            startup_time: Instant::now(),
        }
    }
    /// Get server uptime in seconds
    pub fn uptime_seconds(&self) -> f64 {
        self.startup_time.elapsed().as_secs_f64()
    }
    /// Get actual system metrics
    pub fn get_system_metrics(&self) -> SystemHealthInfo {
        let mut system = System::new_all();
        system.refresh_all();
        let cpu_usage = system.cpus().iter().map(|cpu| cpu.cpu_usage() as f64).sum::<f64>()
            / system.cpus().len() as f64;
        let memory_usage = if system.total_memory() > 0 {
            (system.used_memory() as f64 / system.total_memory() as f64) * 100.0
        } else {
            0.0
        };
        let disk_usage = get_disk_usage_percentage().unwrap_or(0.0);
        let active_connections = self.get_active_connection_count();
        SystemHealthInfo {
            cpu_usage,
            memory_usage,
            disk_usage,
            active_connections,
        }
    }
    /// Enable authentication
    pub fn with_auth(mut self, auth_service: AuthService) -> Self {
        self.auth_service = Some(Arc::new(auth_service));
        self
    }
    /// Start the server
    pub async fn start(self) -> Result<()> {
        self.batching_service.start().await?;
        self.ha_service.start().await?;
        self.polling_service.start().await?;
        self.shadow_service.start().await?;
        let addr = format!("{}:{}", self.config.host, self.config.port);
        let router = self.create_router().await;
        let listener = TcpListener::bind(&addr).await?;
        tracing::info!("TrustformeRS server starting on {}", addr);
        tracing::info!("Starting HTTP server on {}", listener.local_addr()?);
        match axum::serve(listener, router).await {
            Ok(_) => tracing::info!("Server shutdown gracefully"),
            Err(e) => {
                tracing::error!("Server error: {}", e);
                return Err(e.into());
            },
        }
        Ok(())
    }
    /// Create the router with all endpoints for testing
    pub async fn create_test_router(self) -> Router {
        if let Err(e) = self.batching_service.start().await {
            tracing::warn!("Failed to start batching service for tests: {}", e);
        }
        let shared_state = Arc::new(self);
        let mut router = Router::new()
            .route("/health", get(health_check))
            .route("/health/detailed", get(detailed_health_check))
            .route("/health/readiness", get(readiness_check))
            .route("/health/liveness", get(liveness_check))
            .route("/v1/inference", post(inference_endpoint))
            .route("/inference", post(inference_endpoint))
            .route("/v1/inference/batch", post(batch_inference_endpoint))
            .route("/inference/batch", post(batch_inference_endpoint))
            .route("/v1/inference/stream", post(streaming_inference_endpoint))
            .route("/inference/stream", post(streaming_inference_endpoint))
            .route("/inference/async", post(async_inference_endpoint))
            .route("/admin/stats", get(get_stats))
            .route("/admin/config", get(get_config))
            .route("/admin/memory/pressure", get(memory_pressure_endpoint))
            .route("/metrics", get(metrics_endpoint))
            .route("/stream", get(sse_stream_endpoint))
            .route("/v1/stream/sse", get(sse_stream_endpoint))
            .route("/ws", get(websocket_endpoint))
            .route("/v1/stream/ws", get(websocket_endpoint))
            .route("/poll", post(long_poll_endpoint))
            .route("/v1/poll", get(long_poll_endpoint))
            .route("/poll/stats", get(poll_stats_endpoint))
            .route("/v1/poll/stats", get(poll_stats_endpoint))
            .route("/shadow/stats", get(shadow_stats_endpoint))
            .route("/v1/shadow/stats", get(shadow_stats_endpoint))
            .route("/shadow/results", get(shadow_results_endpoint))
            .route("/v1/shadow/results", get(shadow_results_endpoint))
            .route("/shadow/compare", post(shadow_comparison_endpoint))
            .route("/graphql", post(graphql_handler))
            .route("/graphql/playground", get(graphql_playground_handler))
            .route("/jobs/{id}/status", get(job_status_endpoint))
            .route("/models/load", post(model_load_endpoint))
            .route("/admin/failover", post(admin_failover_endpoint))
            .route("/admin/gpu/status", get(admin_gpu_status_endpoint))
            .route("/api-docs/openapi.json", get(openapi_json_endpoint))
            .route("/docs", get(swagger_ui_endpoint))
            .route("/auth/token", post(mock_auth_token_handler))
            .route("/auth/login", post(mock_auth_token_handler));
        if shared_state.auth_service.is_some() {
            router = router.layer(axum::middleware::from_fn(auth_extension_middleware));
        }
        router = router.layer(axum::Extension(shared_state));
        router
    }
    /// Create the router with all endpoints
    async fn create_router(self) -> Router {
        let shared_state = Arc::new(self);
        let mut router = Router::new()
            .route("/health", get(health_check))
            .route("/health/detailed", get(detailed_health_check))
            .route("/health/readiness", get(readiness_check))
            .route("/health/liveness", get(liveness_check))
            .route("/admin/stats", get(get_stats))
            .route("/admin/config", get(get_config))
            .route("/stream", get(sse_stream_endpoint))
            .route("/ws", get(websocket_endpoint))
            .route("/poll/stats", get(poll_stats_endpoint))
            .route("/shadow/stats", get(shadow_stats_endpoint))
            .route("/shadow/results", get(shadow_results_endpoint));
        if shared_state.config.enable_metrics {
            router = router.route("/metrics", get(metrics_endpoint));
        }
        if shared_state.auth_service.is_some() {
            router = router.layer(axum::middleware::from_fn(auth_extension_middleware));
        }
        router = router.layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive()),
        );
        router
    }
}
impl TrustformerServer {
    /// Get the current number of active connections
    fn get_active_connection_count(&self) -> usize {
        let batching_queue_size = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let stats = self.batching_service.get_stats().await;
                stats.aggregator_stats.pending_requests + stats.aggregator_stats.queue_depth
            })
        });
        let mut active_connections = batching_queue_size;
        if batching_queue_size > 0 {
            active_connections += 1;
        }
        active_connections += 2;
        active_connections
    }
}
/// Async inference response
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct AsyncInferenceResponse {
    pub(crate) job_id: String,
    pub(crate) status: String,
}
/// Inference request
#[derive(Debug, Deserialize, utoipa::ToSchema)]
#[schema(
    example = json!(
        {"text":"Translate the following English text to French: Hello, how are you?",
        "max_length":100,
        "temperature":0.7,
        "top_p":0.9}
    )
)]
pub struct InferenceRequest {
    pub(crate) text: String,
    pub(crate) max_length: Option<usize>,
    #[allow(dead_code)]
    pub(crate) temperature: Option<f32>,
    #[allow(dead_code)]
    pub(crate) top_p: Option<f32>,
    #[allow(dead_code)]
    pub(crate) model: Option<String>,
    #[allow(dead_code)]
    pub(crate) enable_cache: Option<bool>,
    #[allow(dead_code)]
    pub(crate) priority: Option<u8>,
    #[allow(dead_code)]
    pub(crate) shadow_mode: Option<bool>,
    #[allow(dead_code)]
    pub(crate) parameters: Option<serde_json::Value>,
}
/// Service health information
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct ServiceHealthInfo {
    pub(crate) batching: String,
    pub(crate) caching: String,
    pub(crate) streaming: String,
    pub(crate) failover: String,
}
/// Detailed health check response
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct DetailedHealthResponse {
    pub(crate) status: String,
    pub(crate) timestamp: chrono::DateTime<chrono::Utc>,
    pub(crate) version: String,
    pub(crate) uptime_seconds: f64,
    pub(crate) system_health: SystemHealthInfo,
    pub(crate) services: ServiceHealthInfo,
    pub(crate) circuit_breakers: serde_json::Value,
}
/// Inference response
#[derive(Debug, Serialize, utoipa::ToSchema)]
#[schema(
    example = json!(
        {"request_id":"550e8400-e29b-41d4-a716-446655440000",
        "text":"Bonjour, comment allez-vous ?",
        "tokens":["Bon",
        "jour",
        ",",
        "comment",
        "allez",
        "-",
        "vous",
        "?"],
        "processing_time_ms":125.5}
    )
)]
pub struct InferenceResponse {
    pub(crate) request_id: String,
    pub(crate) text: String,
    pub(crate) tokens: Vec<String>,
    pub(crate) processing_time_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cache_hit: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) shadow_comparison: Option<serde_json::Value>,
}
/// Failover request
#[derive(Debug, Deserialize, utoipa::ToSchema)]
#[schema(example = json!({"target_node":"node-2"}))]
pub struct FailoverRequest {
    #[allow(dead_code)]
    pub(crate) target_node: String,
}
/// Model load request
#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct ModelLoadRequest {
    pub(crate) model_name: String,
    pub(crate) model_version: String,
    pub(crate) device: String,
}
/// Mock authentication token handler for testing
#[derive(Debug, serde::Deserialize)]
pub struct MockTokenRequest {
    pub username: String,
    pub password: String,
}
/// Server state for sharing between handlers
#[derive(Clone)]
#[allow(dead_code)]
struct ServerState {
    server: TrustformerServer,
}
/// Batch inference response
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct BatchInferenceResponse {
    pub(crate) batch_id: String,
    pub(crate) results: Vec<InferenceResponse>,
    pub(crate) batch_size: usize,
    pub(crate) total_processing_time_ms: f64,
}
/// Async inference request
#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct AsyncInferenceRequest {
    pub(crate) text: String,
    pub(crate) model: String,
    pub(crate) callback_url: Option<String>,
}
/// Statistics response
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct StatsResponse {
    pub(crate) batching_stats: serde_json::Value,
    pub(crate) caching_stats: serde_json::Value,
    pub(crate) streaming_stats: serde_json::Value,
    pub(crate) ha_stats: serde_json::Value,
    pub(crate) resource_usage: serde_json::Value,
    pub(crate) server_stats: serde_json::Value,
}
/// System health information
#[derive(Debug, Serialize, utoipa::ToSchema, async_graphql::SimpleObject)]
pub struct SystemHealthInfo {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_usage: f64,
    pub active_connections: usize,
}
/// Model load response
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct ModelLoadResponse {
    pub(crate) success: bool,
    pub(crate) model_name: String,
    pub(crate) message: String,
}
/// Health check response
#[derive(Debug, Serialize, utoipa::ToSchema)]
#[schema(
    example = json!(
        {"status":"healthy",
        "timestamp":"2025-07-16T10:30:00Z",
        "version":"1.0.0",
        "uptime_seconds":3600.0}
    )
)]
pub struct HealthResponse {
    pub(crate) status: String,
    pub(crate) timestamp: chrono::DateTime<chrono::Utc>,
    pub(crate) version: String,
    pub(crate) uptime_seconds: f64,
}
