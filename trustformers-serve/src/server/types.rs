//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::*;
use super::jobs::JobStore;
use super::streams::StreamStore;
use crate::{
    auth::AuthService,
    batching::{BatchExecutor, DynamicBatchingService},
    caching::CachingService,
    health::{HAConfig, HighAvailabilityService},
    metrics::MetricsService,
    model_management::{ModelManager, ModelRegistry, VersionManager},
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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
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
    /// Registry of async inference jobs; backs `/inference/async` and
    /// `/jobs/{id}/status`.
    pub(crate) job_store: Arc<JobStore>,
    /// Registry of streaming inference requests; backs `/v1/inference/stream`.
    pub(crate) stream_store: Arc<StreamStore>,
    /// Real model loader; backs `/models/load`.
    pub(crate) model_manager: Arc<ModelManager>,
    /// Requests observed by the HTTP layer, counted for `/admin/stats`.
    pub(crate) request_counter: Arc<AtomicU64>,
    /// Routing for the OpenAI-compatible surface (`/v1/chat/completions`,
    /// `/v1/completions`, `/v1/embeddings`, `/v1/models`).
    ///
    /// Built without an inference backend, so those endpoints report `503` until
    /// one is installed with [`TrustformerServer::with_openai_backend`]. They are
    /// never allowed to answer with placeholder text.
    pub(crate) openai_router: crate::openai_compat::OpenAiApiRouter,
    /// Background host-resource sampler backing `/admin/stats`' measured
    /// fields; see [`super::system_stats::HostSampler`] for why this reads
    /// a cached sample instead of measuring fresh per request.
    pub(crate) host_sampler: Arc<super::system_stats::HostSampler>,
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
    /// Create a new server instance without a model.
    ///
    /// Inference endpoints on such a server report `503 Service Unavailable`
    /// rather than returning synthesized output; use
    /// [`TrustformerServer::with_executor`] to install a real batch executor.
    pub fn new(config: ServerConfig) -> Self {
        Self::build(config, None)
    }

    /// Create a server whose batching stack is backed by `executor`.
    pub fn with_executor(config: ServerConfig, executor: Arc<dyn BatchExecutor>) -> Self {
        Self::build(config, Some(executor))
    }

    fn build(config: ServerConfig, executor: Option<Arc<dyn BatchExecutor>>) -> Self {
        let batching_service = Arc::new(match executor {
            Some(executor) => {
                DynamicBatchingService::with_executor(config.batching_config.clone(), executor)
            },
            None => DynamicBatchingService::new(config.batching_config.clone()),
        });
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
        let model_registry = Arc::new(ModelRegistry::new(
            config.model_management_config.metadata_dir.clone(),
        ));
        let model_manager = Arc::new(ModelManager::new(
            config.model_management_config.clone(),
            model_registry,
            Arc::new(VersionManager::new()),
        ));
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
            job_store: Arc::new(JobStore::new()),
            stream_store: Arc::new(StreamStore::new()),
            model_manager,
            request_counter: Arc::new(AtomicU64::new(0)),
            openai_router: crate::openai_compat::OpenAiApiRouter::new(Vec::new()),
            host_sampler: Arc::new(super::system_stats::HostSampler::new()),
        }
    }

    /// Serve the OpenAI-compatible endpoints from `backend`.
    ///
    /// Until this is called, `/v1/chat/completions`, `/v1/completions` and
    /// `/v1/embeddings` are registered but answer `503 Service Unavailable`:
    /// the deployment is incomplete, and saying so is the only honest answer.
    ///
    /// `allowed_models` restricts which model identifiers clients may name; an
    /// empty list accepts whatever the backend reports it can serve.
    pub fn with_openai_backend(
        mut self,
        backend: Arc<dyn crate::openai_compat::OpenAiInferenceBackend>,
        allowed_models: Vec<String>,
    ) -> Self {
        self.openai_router =
            crate::openai_compat::OpenAiApiRouter::new(allowed_models).with_backend(backend);
        self
    }

    /// Whether an OpenAI-compatible inference backend is installed.
    pub fn has_openai_backend(&self) -> bool {
        self.openai_router.has_backend()
    }

    /// Registry of async inference jobs.
    pub fn job_store(&self) -> &Arc<JobStore> {
        &self.job_store
    }

    /// Registry of streaming inference requests.
    pub fn stream_store(&self) -> &Arc<StreamStore> {
        &self.stream_store
    }

    /// Real model loader used by `/models/load`.
    pub fn model_manager(&self) -> &Arc<ModelManager> {
        &self.model_manager
    }

    /// Total HTTP requests observed since startup.
    pub fn total_requests(&self) -> u64 {
        self.request_counter.load(Ordering::Relaxed)
    }

    /// Record one observed HTTP request.
    pub(crate) fn record_request(&self) {
        self.request_counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Whether a real model is wired into the batching stack.
    pub fn has_model(&self) -> bool {
        self.batching_service.has_model()
    }
    /// Get server uptime in seconds
    pub fn uptime_seconds(&self) -> f64 {
        self.startup_time.elapsed().as_secs_f64()
    }
    /// Measured host metrics including live streaming connections.
    ///
    /// Reads the same background [`super::system_stats::HostSampler`] as
    /// `/admin/stats` (`server::functions::get_stats`) rather than measuring
    /// fresh: this backs both `/health/detailed` and the GraphQL
    /// `detailedHealth` query, either of which liveness/readiness probes or
    /// dashboards can hit every few seconds, so it must answer without
    /// paying `sysinfo`'s real measurement latency on the request path (see
    /// the sampler's doc comment for why that latency has no upper bound
    /// under host load).
    ///
    /// `cpu_usage`, `memory_usage`, and `disk_usage` are `None` (`null` on
    /// the wire) exactly when the corresponding reading is not available --
    /// mirroring the honest-null convention `/admin/stats` already uses for
    /// this same background sample, rather than a sentinel a client could
    /// mistake for a real measurement. That happens in two distinct cases,
    /// both real and neither bounded to a fixed sub-second window (the
    /// window before the sampler's first completed sample depends on
    /// current host load, which is precisely what makes measuring fresh on
    /// every request untenable): before the background sampler's first
    /// sample has landed, all three are `None`; once a sample exists,
    /// `disk_usage` alone can still be `None` if no mounted filesystem could
    /// be matched to the working directory. The reading returned is always
    /// the latest *completed* sample, not necessarily a current one -- see
    /// [`super::system_stats::HostSampler`] for the refresh cadence and for
    /// what happens when a background refresh itself fails (the previous
    /// reading is kept rather than being overwritten with a fabricated
    /// value).
    pub async fn system_health_info(&self) -> SystemHealthInfo {
        self.host_sampler.ensure_started(super::system_stats::DEFAULT_SAMPLE_INTERVAL);
        let snapshot = self.host_sampler.current().map(|sample| sample.snapshot);
        SystemHealthInfo {
            cpu_usage: snapshot.map(|s| s.cpu_percent),
            memory_usage: snapshot.map(|s| s.memory_percent),
            disk_usage: snapshot.and_then(|s| s.disk_percent),
            active_connections: self.active_connection_count().await,
        }
    }

    /// Requests this process is driving right now, countable synchronously.
    pub fn in_flight_work_count(&self) -> usize {
        let streaming = self.stream_store.count_in_state(super::streams::StreamState::Streaming);
        let jobs = self.job_store.count_in_state(super::jobs::JobState::Processing)
            + self.job_store.count_in_state(super::jobs::JobState::Pending);
        streaming + jobs
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
    /// The complete route table this server serves.
    ///
    /// Both the production router and the test router are built from this single
    /// table, so an endpoint can never be reachable in tests while returning 404
    /// in production.
    ///
    /// `openai` supplies the OpenAI-compatible surface. It is merged here rather
    /// than in the two callers so the two routers can never disagree about which
    /// `/v1/...` endpoints exist. With no backend attached to it, those endpoints
    /// answer `503` with an OpenAI-shaped error body — they are reachable and
    /// honest, never absent and never fabricating a completion.
    fn routes(openai: crate::openai_compat::OpenAiApiRouter) -> Router {
        Router::new()
            .merge(crate::openai_compat::openai_compat_router(openai))
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
            .route("/v1/inference/stream/{id}", get(stream_status_endpoint))
            .route("/inference/async", post(async_inference_endpoint))
            .route("/jobs/{id}/status", get(job_status_endpoint))
            .route("/admin/stats", get(get_stats))
            .route("/admin/config", get(get_config))
            .route("/admin/memory/pressure", get(memory_pressure_endpoint))
            .route("/admin/failover", post(admin_failover_endpoint))
            .route("/admin/gpu/status", get(admin_gpu_status_endpoint))
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
            .route("/models/load", post(model_load_endpoint))
            .route("/api-docs/openapi.json", get(openapi_json_endpoint))
            .route("/docs", get(swagger_ui_endpoint))
            .route("/auth/token", post(auth_token_handler))
            .route("/auth/login", post(auth_token_handler))
    }

    /// Paths served by this server, for diagnostics and for the router parity test.
    pub fn route_paths() -> &'static [&'static str] {
        &[
            "/health",
            "/health/detailed",
            "/health/readiness",
            "/health/liveness",
            "/v1/inference",
            "/inference",
            "/v1/inference/batch",
            "/inference/batch",
            "/v1/inference/stream",
            "/inference/stream",
            "/v1/inference/stream/{id}",
            "/inference/async",
            "/jobs/{id}/status",
            "/admin/stats",
            "/admin/config",
            "/admin/memory/pressure",
            "/admin/failover",
            "/admin/gpu/status",
            "/stream",
            "/v1/stream/sse",
            "/ws",
            "/v1/stream/ws",
            "/poll",
            "/v1/poll",
            "/poll/stats",
            "/v1/poll/stats",
            "/shadow/stats",
            "/v1/shadow/stats",
            "/shadow/results",
            "/v1/shadow/results",
            "/shadow/compare",
            "/graphql",
            "/graphql/playground",
            "/models/load",
            "/api-docs/openapi.json",
            "/docs",
            "/auth/token",
            "/auth/login",
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/embeddings",
            "/v1/models",
        ]
    }

    /// Create the router with all endpoints for testing
    pub async fn create_test_router(self) -> Router {
        if let Err(e) = self.batching_service.start().await {
            tracing::warn!("Failed to start batching service for tests: {}", e);
        }
        let openai = self.openai_router.clone();
        let shared_state = Arc::new(self);
        let mut router = Self::routes(openai).route("/metrics", get(metrics_endpoint));

        if shared_state.auth_service.is_some() {
            router = router.layer(axum::middleware::from_fn(auth_extension_middleware));
        }
        router = router
            .layer(axum::middleware::from_fn(request_counting_middleware))
            .layer(axum::Extension(shared_state));
        router
    }

    /// Create the production router.
    ///
    /// Serves exactly the same endpoints as [`TrustformerServer::create_test_router`];
    /// only the observability layers and the `/metrics` gate differ.
    async fn create_router(self) -> Router {
        let openai = self.openai_router.clone();
        let shared_state = Arc::new(self);
        let mut router = Self::routes(openai);

        if shared_state.config.enable_metrics {
            router = router.route("/metrics", get(metrics_endpoint));
        }
        if shared_state.auth_service.is_some() {
            router = router.layer(axum::middleware::from_fn(auth_extension_middleware));
        }
        router = router
            .layer(axum::middleware::from_fn(request_counting_middleware))
            .layer(axum::Extension(shared_state))
            .layer(
                ServiceBuilder::new()
                    .layer(TraceLayer::new_for_http())
                    .layer(CorsLayer::permissive()),
            );
        router
    }
}

impl TrustformerServer {
    /// Number of connections currently being served.
    ///
    /// Counted from the real streaming registries plus the work in flight in the
    /// batching stack. No constant is added to make the number look busier.
    pub async fn active_connection_count(&self) -> usize {
        let batching = self.batching_service.get_stats().await;
        let in_flight =
            batching.aggregator_stats.pending_requests + batching.aggregator_stats.queue_depth;
        let sse = self.sse_handler.get_stats().await.active_connections;
        let websockets = self.websocket_handler.get_stats().await.active_connections;
        in_flight + sse + websockets
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
// reason: request DTO fields are deserialized from API payloads; several are not
// read directly in the current handlers.
#[allow(dead_code)]
pub struct InferenceRequest {
    pub(crate) text: String,
    pub(crate) max_length: Option<usize>,
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) model: Option<String>,
    pub(crate) enable_cache: Option<bool>,
    pub(crate) priority: Option<u8>,
    pub(crate) shadow_mode: Option<bool>,
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
    /// Node the caller wants to fail over to. Read by `/admin/failover`.
    pub(crate) target_node: String,
}
/// Model load request
#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct ModelLoadRequest {
    pub(crate) model_name: String,
    pub(crate) model_version: String,
    #[serde(default = "default_device")]
    pub(crate) device: String,
    /// Filesystem path of the checkpoint to load. Required: the server never
    /// fabricates weights for a model it cannot find.
    #[serde(default)]
    pub(crate) model_path: Option<String>,
    /// Requested weight precision (`fp32`, `fp16`, ...).
    #[serde(default)]
    pub(crate) precision: Option<String>,
}

fn default_device() -> String {
    "cpu".to_string()
}
/// Credentials posted to the token endpoint.
///
/// Renamed from `MockTokenRequest` in 0.2.1. The name and its "for testing"
/// doc comment outlived the handler they described:
/// `auth_token_handler` has
/// authenticated against the configured
/// [`AuthService`] since the canned-token path was
/// removed, and a type called `Mock*` sitting on the crate's real
/// credential-accepting endpoint invited exactly the wrong conclusion about
/// what that endpoint does.
#[derive(Debug, serde::Deserialize)]
pub struct TokenRequest {
    /// Username to authenticate.
    pub username: String,
    /// Password to authenticate with.
    pub password: String,
}
/// Server state for sharing between handlers
// reason: retained for an in-development handler-sharing path; not yet constructed.
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
/// System health information.
///
/// `cpu_usage`, `memory_usage`, and `disk_usage` are `null` on the wire
/// whenever the corresponding reading is not available -- see
/// [`TrustformerServer::system_health_info`] for exactly which cases that
/// covers and why. `active_connections` is always populated: it is counted
/// live from this process's own request-tracking state, not `sysinfo`, so
/// it carries no "unmeasured" case.
#[derive(Debug, Serialize, utoipa::ToSchema, async_graphql::SimpleObject)]
pub struct SystemHealthInfo {
    pub cpu_usage: Option<f64>,
    pub memory_usage: Option<f64>,
    pub disk_usage: Option<f64>,
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

#[cfg(test)]
mod router_tests {
    use super::*;

    /// Regression: the production router used to register only 11 of the
    /// server's routes and no `Extension` layer, so an operator who ran
    /// `TrustformerServer::start()` got 404 (or a 500) on every endpoint the
    /// tests exercised through `create_test_router`.
    #[tokio::test]
    async fn production_router_serves_every_endpoint() {
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let mut config = ServerConfig::default();
        config.enable_metrics = true;
        let router = TrustformerServer::new(config).create_router().await;

        // Endpoints that must answer a GET without a body.
        for path in [
            "/health",
            "/health/liveness",
            "/admin/config",
            "/metrics",
            "/api-docs/openapi.json",
            "/docs",
            "/graphql/playground",
            "/admin/gpu/status",
            "/v1/models",
        ] {
            let response = router
                .clone()
                .oneshot(Request::builder().uri(path).body(Body::empty()).expect("request builds"))
                .await
                .expect("router answers");
            assert_ne!(
                response.status(),
                StatusCode::NOT_FOUND,
                "{path} is not registered on the production router"
            );
            assert_ne!(
                response.status(),
                StatusCode::INTERNAL_SERVER_ERROR,
                "{path} failed on the production router"
            );
        }

        // Endpoints that must answer a POST.
        for path in [
            "/v1/inference",
            "/inference/async",
            "/models/load",
            "/graphql",
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/embeddings",
        ] {
            let response = router
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(path)
                        .header("content-type", "application/json")
                        .body(Body::from("{}"))
                        .expect("request builds"),
                )
                .await
                .expect("router answers");
            assert_ne!(
                response.status(),
                StatusCode::NOT_FOUND,
                "{path} is not registered on the production router"
            );
            assert_ne!(
                response.status(),
                StatusCode::METHOD_NOT_ALLOWED,
                "{path} does not accept POST on the production router"
            );
        }
    }

    /// Regression: an inference request against a server with no model must be
    /// answered with 503, never with a synthesized completion.
    #[tokio::test]
    async fn inference_without_a_model_is_service_unavailable() {
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let router = TrustformerServer::new(ServerConfig::default()).create_test_router().await;
        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/inference")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"text":"hello"}"#))
                    .expect("request builds"),
            )
            .await
            .expect("router answers");

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    /// A tiny but genuine GPT-2 over a byte-level vocabulary, wired into an
    /// OpenAI-compatible backend. Real architecture, real (untrained) weights,
    /// real forward pass — no stand-in for inference anywhere.
    #[cfg(test)]
    fn tiny_openai_backend() -> Arc<dyn crate::openai_compat::OpenAiInferenceBackend> {
        use crate::batching::processor::{BatchModel, EmbeddingModel};
        use crate::batching::{ByteTokenizer, Gpt2BatchModel};
        use crate::openai_compat::BatchExecutorBackend;
        use trustformers_models::gpt2::Gpt2Config;

        let config = Gpt2Config {
            vocab_size: ByteTokenizer::VOCAB_SIZE,
            n_positions: 64,
            n_embd: 16,
            n_layer: 1,
            n_head: 2,
            n_inner: Some(32),
            resid_pdrop: 0.0,
            embd_pdrop: 0.0,
            attn_pdrop: 0.0,
            bos_token_id: ByteTokenizer::EOT_ID,
            eos_token_id: ByteTokenizer::EOT_ID,
            ..Gpt2Config::default()
        };
        let model = Arc::new(Gpt2BatchModel::untrained(config).expect("tiny GPT-2 must build"));
        Arc::new(
            BatchExecutorBackend::new(
                "tiny-gpt2",
                Arc::clone(&model) as Arc<dyn BatchModel>,
                Arc::new(ByteTokenizer),
            )
            .with_embedding_model(model as Arc<dyn EmbeddingModel>)
            .with_default_max_tokens(4),
        )
    }

    /// Regression: `openai_compat` was a fully built module registered on no
    /// router at all, so every `/v1/chat/completions` request 404'd. It must now
    /// be mounted and run a real forward pass end to end.
    #[tokio::test]
    async fn openai_chat_completions_runs_real_inference_through_the_server_router() {
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let router = TrustformerServer::new(ServerConfig::default())
            .with_openai_backend(tiny_openai_backend(), vec!["tiny-gpt2".to_string()])
            .create_test_router()
            .await;

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"tiny-gpt2","messages":[{"role":"user","content":"Hello"}],"max_tokens":4}"#,
                    ))
                    .expect("request builds"),
            )
            .await
            .expect("router answers");

        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("body readable");
        let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json body");
        assert_eq!(body["object"], serde_json::json!("chat.completion"));
        assert_eq!(body["model"], serde_json::json!("tiny-gpt2"));
        // The flattened chat prompt is "user: Hello\n" — twelve bytes, hence
        // twelve byte-level tokens. This is the real tokenizer's count; the
        // `len / 4` fallback used without a tokenizer would have said three.
        assert_eq!(body["usage"]["prompt_tokens"], serde_json::json!(12));
        assert!(body["choices"][0]["message"]["content"].is_string());
    }

    /// `/v1/completions` and `/v1/embeddings` must be reachable on the same
    /// router and produce real model output.
    #[tokio::test]
    async fn openai_completions_and_embeddings_are_mounted() {
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let router = TrustformerServer::new(ServerConfig::default())
            .with_openai_backend(tiny_openai_backend(), Vec::new())
            .create_test_router()
            .await;

        let completion = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"tiny-gpt2","prompt":"abc","max_tokens":2}"#,
                    ))
                    .expect("request builds"),
            )
            .await
            .expect("router answers");
        assert_eq!(completion.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(completion.into_body(), 1 << 20)
            .await
            .expect("body readable");
        let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json body");
        assert_eq!(body["object"], serde_json::json!("text_completion"));
        assert_eq!(body["usage"]["prompt_tokens"], serde_json::json!(3));

        let embeddings = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/embeddings")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"tiny-gpt2","input":"alpha"}"#))
                    .expect("request builds"),
            )
            .await
            .expect("router answers");
        assert_eq!(embeddings.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(embeddings.into_body(), 1 << 20)
            .await
            .expect("body readable");
        let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json body");
        let vector = body["data"][0]["embedding"].as_array().expect("embedding array");
        assert_eq!(vector.len(), 16, "the tiny model's hidden size");
        assert!(
            vector.iter().any(|v| v.as_f64().unwrap_or(0.0).abs() > f64::EPSILON),
            "a real pooled hidden state must not be an all-zero vector"
        );

        let models = router
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .expect("request builds"),
            )
            .await
            .expect("router answers");
        assert_eq!(models.status(), StatusCode::OK);
    }

    /// Regression: with no OpenAI backend installed the endpoints must still be
    /// registered and must answer an honest 503 — never 404, never placeholder
    /// text.
    #[tokio::test]
    async fn openai_endpoints_without_a_backend_are_service_unavailable() {
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let server = TrustformerServer::new(ServerConfig::default());
        assert!(!server.has_openai_backend());
        let router = server.create_test_router().await;

        for (path, payload) in [
            (
                "/v1/chat/completions",
                r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#,
            ),
            ("/v1/completions", r#"{"model":"m","prompt":"hi"}"#),
            ("/v1/embeddings", r#"{"model":"m","input":"hi"}"#),
        ] {
            let response = router
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(path)
                        .header("content-type", "application/json")
                        .body(Body::from(payload))
                        .expect("request builds"),
                )
                .await
                .expect("router answers");
            assert_eq!(
                response.status(),
                StatusCode::SERVICE_UNAVAILABLE,
                "{path} must report an incomplete deployment honestly"
            );
        }
    }

    /// Every documented path is actually registered.
    #[test]
    fn route_paths_are_declared_once() {
        let paths = TrustformerServer::route_paths();
        let mut sorted = paths.to_vec();
        sorted.sort_unstable();
        let before = sorted.len();
        sorted.dedup();
        assert_eq!(before, sorted.len(), "duplicate route path declared");
        assert!(paths.contains(&"/v1/inference"));
        assert!(paths.contains(&"/graphql"));
        assert!(paths.contains(&"/models/load"));
        // The OpenAI-compatible surface is part of the documented route table.
        assert!(paths.contains(&"/v1/chat/completions"));
        assert!(paths.contains(&"/v1/completions"));
        assert!(paths.contains(&"/v1/embeddings"));
        assert!(paths.contains(&"/v1/models"));
    }
}

#[cfg(test)]
mod system_health_info_tests {
    use super::*;

    /// Regression: before the background `HostSampler` has completed its
    /// first measurement, `system_health_info` must report the unmeasured
    /// fields as `None` (serialized as JSON `null`), never a sentinel like
    /// `0.0`/`-1.0` that a client could mistake for a real reading. A fresh
    /// server's sampler cannot have completed `measure_host`'s two
    /// `sysinfo::MINIMUM_CPU_UPDATE_INTERVAL`-apart CPU samples in the time
    /// between construction and this immediate call, so the cold window is
    /// genuinely exercised here, not simulated.
    #[tokio::test]
    async fn reports_none_before_the_first_sample_lands() {
        let server = TrustformerServer::new(ServerConfig::default());
        let info = server.system_health_info().await;
        assert!(
            info.cpu_usage.is_none(),
            "cpu_usage must be None before any background sample has completed"
        );
        assert!(
            info.memory_usage.is_none(),
            "memory_usage must be None before any background sample has completed"
        );
        assert!(
            info.disk_usage.is_none(),
            "disk_usage must be None before any background sample has completed"
        );
    }
}
