//! GraphQL API Implementation
//!
//! Provides a GraphQL interface for flexible queries and mutations
//! on the inference server's capabilities.

use async_graphql::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{
    batching::{
        aggregator::{ProcessingOutput, Request, RequestId, RequestInput},
        config::Priority,
    },
    health::HealthStatus,
    model_management::ModelStatus,
    server::{StreamState, SystemHealthInfo},
    TrustformerServer,
};

/// GraphQL schema context
#[derive(Clone)]
pub struct GraphQLContext {
    pub server: Arc<TrustformerServer>,
}

/// Health information for GraphQL responses
#[derive(SimpleObject, Debug, Serialize)]
pub struct HealthInfo {
    pub status: String,
    pub timestamp: String,
    pub version: String,
    pub uptime_seconds: f64,
}

/// Detailed health information
#[derive(SimpleObject, Debug, Serialize)]
pub struct DetailedHealthInfo {
    pub status: String,
    pub timestamp: String,
    pub version: String,
    pub uptime_seconds: f64,
    pub system_health: SystemHealthInfo,
    pub services: ServiceHealthInfo,
}

/// Per-subsystem health, each field derived from that subsystem's own state.
///
/// None of these are constants. Every value is computed by
/// [`service_health`] from a live reading, and a subsystem that cannot serve
/// says so rather than reporting `"healthy"`.
#[derive(SimpleObject, Debug, Serialize)]
pub struct ServiceHealthInfo {
    /// `"healthy"` when a model is wired into the batching executor and no
    /// batch has failed, `"degraded"` when batches have failed, `"no_model"`
    /// when no executor is installed and inference cannot be served at all.
    pub batching: String,
    /// `"healthy"` when the caching service returns its statistics,
    /// `"unhealthy"` when collecting them fails.
    pub caching: String,
    /// `"healthy"` when no streaming request has failed, `"degraded"` when at
    /// least one recorded stream ended in the failed state.
    pub streaming: String,
    /// The high-availability service's own verdict on system health.
    pub failover: String,
}

/// Read the live state of each subsystem and label it.
///
/// Kept as a free function so the labelling rules are testable without an
/// executing GraphQL schema.
async fn service_health(server: &TrustformerServer, ha_status: HealthStatus) -> ServiceHealthInfo {
    let batching = if !server.batching_service().has_model() {
        "no_model"
    } else if server.batching_service().get_stats().await.processor_stats.failed_batches > 0 {
        "degraded"
    } else {
        "healthy"
    };

    let caching = match server.caching_service().get_stats().await {
        Ok(_) => "healthy",
        Err(_) => "unhealthy",
    };

    let streaming = if server.stream_store().count_in_state(StreamState::Failed) > 0 {
        "degraded"
    } else {
        "healthy"
    };

    let failover = match ha_status {
        HealthStatus::Healthy => "healthy",
        HealthStatus::Degraded => "degraded",
        HealthStatus::Unhealthy => "unhealthy",
    };

    ServiceHealthInfo {
        batching: batching.to_string(),
        caching: caching.to_string(),
        streaming: streaming.to_string(),
        failover: failover.to_string(),
    }
}

/// Inference request input
#[derive(InputObject, Debug, Deserialize)]
pub struct InferenceInput {
    pub text: String,
    pub max_length: Option<i32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
}

/// Inference response
#[derive(SimpleObject, Debug, Serialize)]
pub struct InferenceResult {
    pub request_id: String,
    pub text: String,
    pub tokens: Vec<String>,
    pub processing_time_ms: f64,
}

/// Batch inference input
#[derive(InputObject, Debug, Deserialize)]
pub struct BatchInferenceInput {
    pub requests: Vec<InferenceInput>,
}

/// Batch inference response
#[derive(SimpleObject, Debug, Serialize)]
pub struct BatchInferenceResult {
    pub batch_id: String,
    pub responses: Vec<InferenceResult>,
    pub total_processing_time_ms: f64,
}

/// Statistics information
#[derive(SimpleObject, Debug, Serialize)]
pub struct StatsInfo {
    pub batching_stats: String,
    pub caching_stats: String,
    pub streaming_stats: String,
    pub ha_stats: String,
}

/// One model that is genuinely resident in this process.
///
/// Every field is read from the [`ModelManager`](crate::model_management::ModelManager)'s
/// record of what was actually loaded; a deployment with nothing loaded reports
/// an empty list rather than a stand-in entry.
#[derive(SimpleObject, Debug, Serialize)]
pub struct ModelInfo {
    /// Registered model name.
    pub name: String,
    /// Registered model version.
    pub version: String,
    /// Lifecycle state recorded for the model (`loading`, `active`, `standby`,
    /// `draining`, `unloaded`, or `failed: <error>`).
    pub status: String,
    /// Wall-clock load time in RFC 3339, reconstructed from the monotonic clock
    /// the load was timed against. `"unknown"` only if that reconstruction
    /// overflows, which cannot happen for any realistic uptime.
    pub loaded_at: String,
    /// Measured resident size of the model's weights, **in bytes**, summed over
    /// the tensors actually parsed out of the checkpoint. Never an estimate.
    pub memory_usage: f64,
}

/// Render a [`ModelStatus`] as the string the GraphQL surface reports.
fn model_status_label(status: &ModelStatus) -> String {
    match status {
        ModelStatus::Loading => "loading".to_string(),
        ModelStatus::Active => "active".to_string(),
        ModelStatus::Standby => "standby".to_string(),
        ModelStatus::Draining => "draining".to_string(),
        ModelStatus::Unloaded => "unloaded".to_string(),
        ModelStatus::Failed { error } => format!("failed: {error}"),
    }
}

/// Reconstruct the wall-clock instant a model was loaded from the monotonic
/// duration since that load.
///
/// The manager times loads against [`std::time::Instant`], which carries no
/// calendar information; subtracting the measured elapsed time from the current
/// wall clock recovers the load time to within the clock's own drift. Nothing
/// is invented: if the conversion cannot be represented the caller is told so.
fn loaded_at_rfc3339(elapsed: std::time::Duration) -> String {
    chrono::Duration::from_std(elapsed)
        .ok()
        .and_then(|delta| chrono::Utc::now().checked_sub_signed(delta))
        .map(|when| when.to_rfc3339())
        .unwrap_or_else(|| "unknown".to_string())
}

/// GraphQL Query root
pub struct QueryRoot;

#[Object]
impl QueryRoot {
    /// Get basic health information
    async fn health(&self, ctx: &Context<'_>) -> Result<HealthInfo> {
        let context = ctx.data::<GraphQLContext>()?;
        let system_health = context.server.ha_service().get_system_health().await;

        let status = match system_health.status {
            HealthStatus::Healthy => "healthy",
            HealthStatus::Degraded => "degraded",
            HealthStatus::Unhealthy => "unhealthy",
        };

        Ok(HealthInfo {
            status: status.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            version: crate::VERSION.to_string(),
            uptime_seconds: ctx.data::<GraphQLContext>()?.server.uptime_seconds(),
        })
    }

    /// Get detailed health information
    async fn detailed_health(&self, ctx: &Context<'_>) -> Result<DetailedHealthInfo> {
        let context = ctx.data::<GraphQLContext>()?;
        let system_health = context.server.ha_service().get_system_health().await;

        let status = match system_health.status {
            HealthStatus::Healthy => "healthy",
            HealthStatus::Degraded => "degraded",
            HealthStatus::Unhealthy => "unhealthy",
        };

        Ok(DetailedHealthInfo {
            status: status.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            version: crate::VERSION.to_string(),
            uptime_seconds: ctx.data::<GraphQLContext>()?.server.uptime_seconds(),
            // Reads the same background `HostSampler` cache `/health/detailed`
            // uses (`TrustformerServer::system_health_info`), instead of the
            // blocking two-sysinfo-sample `measure_host()` this used to call
            // synchronously from an async resolver.
            system_health: context.server.system_health_info().await,
            services: service_health(&context.server, system_health.status.clone()).await,
        })
    }

    /// Get system statistics
    async fn stats(&self, ctx: &Context<'_>) -> Result<StatsInfo> {
        let context = ctx.data::<GraphQLContext>()?;

        // Get actual stats from services
        let batching_stats = context.server.batching_service().get_stats().await;
        let caching_stats = context.server.caching_service().get_stats().await;
        let streaming_stats = context.server.streaming_service().get_stats().await;
        let ha_stats = context.server.ha_service().get_stats().await;

        Ok(StatsInfo {
            batching_stats: serde_json::to_string(&batching_stats).unwrap_or_default(),
            caching_stats: match caching_stats {
                Ok(stats) => serde_json::to_string(&stats).unwrap_or_default(),
                Err(_) => "Error getting caching stats".to_string(),
            },
            streaming_stats: serde_json::to_string(&streaming_stats).unwrap_or_default(),
            ha_stats: serde_json::to_string(&ha_stats).unwrap_or_default(),
        })
    }

    /// The models that are genuinely resident in this process.
    ///
    /// Enumerated from the live [`ModelManager`](crate::model_management::ModelManager);
    /// an empty list is the truthful answer for a server that has loaded
    /// nothing, and is returned instead of a stand-in entry. The lookup does not
    /// refresh the models' LRU stamps, so reporting state cannot change
    /// unloading decisions.
    async fn models(&self, ctx: &Context<'_>) -> Result<Vec<ModelInfo>> {
        let context = ctx.data::<GraphQLContext>()?;
        let manager = context.server.model_manager();

        let mut model_infos = Vec::new();
        for model_id in manager.list_loaded_models() {
            // A model unloaded between the listing and the lookup is simply no
            // longer resident; reporting the stale entry would be a lie.
            let Some(loaded) = manager.peek_loaded_model(&model_id) else {
                continue;
            };
            model_infos.push(ModelInfo {
                name: loaded.metadata.name.clone(),
                version: loaded.metadata.version.clone(),
                status: model_status_label(&loaded.metadata.status),
                loaded_at: loaded_at_rfc3339(loaded.loaded_at.elapsed()),
                memory_usage: loaded.memory_usage as f64,
            });
        }

        // Stable ordering: the manager's map iteration order is arbitrary, and a
        // status query that reshuffles between calls is needlessly confusing.
        model_infos.sort_by(|a, b| (&a.name, &a.version).cmp(&(&b.name, &b.version)));
        Ok(model_infos)
    }
}

/// GraphQL Mutation root
pub struct MutationRoot;

#[Object]
impl MutationRoot {
    /// Perform inference
    async fn inference(&self, ctx: &Context<'_>, input: InferenceInput) -> Result<InferenceResult> {
        let context = ctx.data::<GraphQLContext>()?;
        let _request_id = uuid::Uuid::new_v4().to_string();

        // Implement actual inference logic using batching service
        let start_time = std::time::Instant::now();

        // Create batching request
        let request_id = RequestId::new();
        let batching_request = Request {
            id: request_id.clone(),
            input: RequestInput::Text {
                text: input.text,
                max_length: input.max_length.map(|ml| ml as usize),
            },
            priority: Priority::Normal,
            submitted_at: std::time::Instant::now(),
            deadline: None,
            metadata: std::collections::HashMap::new(),
        };

        // Submit to batching service
        let batching_service = context.server.batching_service();
        let result = batching_service.submit_request(batching_request).await;

        let response = match result {
            Ok(output) => {
                let (text, tokens) = match &output.output {
                    ProcessingOutput::Text(t) => (
                        t.clone(),
                        t.split_whitespace().map(|s| s.to_string()).collect(),
                    ),
                    ProcessingOutput::Tokens(token_ids) => {
                        let text = format!("{:?}", token_ids);
                        (text.clone(), vec![text])
                    },
                    ProcessingOutput::Error(e) => (e.clone(), vec!["<error>".to_string()]),
                    _ => (
                        "Unsupported output type".to_string(),
                        vec!["<unsupported>".to_string()],
                    ),
                };
                InferenceResult {
                    request_id: request_id.to_string(),
                    text,
                    tokens,
                    processing_time_ms: output.latency_ms as f64,
                }
            },
            Err(e) => {
                tracing::error!("Inference failed: {}", e);
                // Fallback response
                InferenceResult {
                    request_id: request_id.to_string(),
                    text: format!("Error: {}", e),
                    tokens: vec!["<error>".to_string()],
                    processing_time_ms: start_time.elapsed().as_millis() as f64,
                }
            },
        };

        Ok(response)
    }

    /// Perform batch inference
    async fn batch_inference(
        &self,
        ctx: &Context<'_>,
        input: BatchInferenceInput,
    ) -> Result<BatchInferenceResult> {
        let context = ctx.data::<GraphQLContext>()?;
        let batch_id = uuid::Uuid::new_v4().to_string();
        let start_time = std::time::Instant::now();

        // Implement actual batch inference logic using batching service
        let mut responses = Vec::new();
        let batching_service = context.server.batching_service();

        // Convert GraphQL requests to batching requests
        let mut batch_requests = Vec::new();
        for req in input.requests {
            let request_id = RequestId::new();
            batch_requests.push(Request {
                id: request_id,
                input: RequestInput::Text {
                    text: req.text,
                    max_length: req.max_length.map(|ml| ml as usize),
                },
                priority: Priority::Normal,
                submitted_at: std::time::Instant::now(),
                deadline: None,
                metadata: std::collections::HashMap::new(),
            });
        }

        // Submit batch to batching service
        for req in batch_requests {
            let request_id = req.id.clone();
            let _req_text = match &req.input {
                RequestInput::Text { text, .. } => text.clone(),
                _ => "".to_string(),
            };

            match batching_service.submit_request(req).await {
                Ok(output) => {
                    let (text, tokens) = match &output.output {
                        ProcessingOutput::Text(t) => (
                            t.clone(),
                            t.split_whitespace().map(|s| s.to_string()).collect(),
                        ),
                        ProcessingOutput::Tokens(token_ids) => {
                            let text = format!("{:?}", token_ids);
                            (text.clone(), vec![text])
                        },
                        ProcessingOutput::Error(e) => (e.clone(), vec!["<error>".to_string()]),
                        _ => (
                            "Unsupported output type".to_string(),
                            vec!["<unsupported>".to_string()],
                        ),
                    };
                    responses.push(InferenceResult {
                        request_id: request_id.to_string(),
                        text,
                        tokens,
                        processing_time_ms: output.latency_ms as f64,
                    });
                },
                Err(e) => {
                    tracing::error!("Batch inference failed for request {}: {}", request_id, e);
                    responses.push(InferenceResult {
                        request_id: request_id.to_string(),
                        text: format!("Error: {}", e),
                        tokens: vec!["<error>".to_string()],
                        processing_time_ms: 0.0,
                    });
                },
            }
        }

        Ok(BatchInferenceResult {
            batch_id,
            responses,
            total_processing_time_ms: start_time.elapsed().as_millis() as f64,
        })
    }

    /// Fail over to `target_node`.
    ///
    /// Delegates to the real failover manager behind the high-availability
    /// service — the same path `POST /admin/failover` uses. An unregistered or
    /// unhealthy target is reported as a GraphQL error carrying the manager's
    /// reason; `true` is returned only when the switch actually happened.
    async fn force_failover(&self, ctx: &Context<'_>, target_node: String) -> Result<bool> {
        let context = ctx.data::<GraphQLContext>()?;

        let target = target_node.trim();
        if target.is_empty() {
            return Err(Error::new("target_node must not be empty"));
        }

        match context.server.ha_service().trigger_failover(target).await {
            Ok(outcome) => {
                tracing::info!(
                    "failover to {} completed: previous={:?} active={:?}",
                    target,
                    outcome.previous_node,
                    outcome.active_node
                );
                Ok(true)
            },
            Err(e) => {
                tracing::error!("failover to {} failed: {}", target, e);
                Err(Error::new(format!("failover to {target} failed: {e}")))
            },
        }
    }
}

/// Create GraphQL schema
pub fn create_schema() -> Schema<QueryRoot, MutationRoot, EmptySubscription> {
    Schema::build(QueryRoot, MutationRoot, EmptySubscription).finish()
}

/// Create GraphQL context
pub fn create_context(server: Arc<TrustformerServer>) -> GraphQLContext {
    GraphQLContext { server }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_health_info(status: &str) -> HealthInfo {
        HealthInfo {
            status: status.to_string(),
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            version: "1.0.0".to_string(),
            uptime_seconds: 42.0,
        }
    }

    #[test]
    fn test_health_info_fields() {
        let info = make_health_info("healthy");
        assert_eq!(info.status, "healthy");
        assert_eq!(info.version, "1.0.0");
        assert!((info.uptime_seconds - 42.0).abs() < 1e-9);
        assert!(!info.timestamp.is_empty());
    }

    #[test]
    fn test_health_info_degraded_status() {
        let info = make_health_info("degraded");
        assert_eq!(info.status, "degraded");
    }

    #[test]
    fn test_health_info_unhealthy_status() {
        let info = make_health_info("unhealthy");
        assert_eq!(info.status, "unhealthy");
    }

    #[test]
    fn test_inference_result_fields() {
        let result = InferenceResult {
            request_id: "req-123".to_string(),
            text: "hello world".to_string(),
            tokens: vec!["hello".to_string(), "world".to_string()],
            processing_time_ms: 55.5,
        };
        assert_eq!(result.request_id, "req-123");
        assert_eq!(result.text, "hello world");
        assert_eq!(result.tokens.len(), 2);
        assert!((result.processing_time_ms - 55.5).abs() < 1e-9);
    }

    #[test]
    fn test_inference_result_empty_tokens() {
        let result = InferenceResult {
            request_id: "r".to_string(),
            text: "".to_string(),
            tokens: vec![],
            processing_time_ms: 0.0,
        };
        assert!(result.tokens.is_empty());
    }

    #[test]
    fn test_batch_inference_result_fields() {
        let batch_result = BatchInferenceResult {
            batch_id: "batch-7".to_string(),
            responses: vec![
                InferenceResult {
                    request_id: "r1".to_string(),
                    text: "a".to_string(),
                    tokens: vec![],
                    processing_time_ms: 10.0,
                },
                InferenceResult {
                    request_id: "r2".to_string(),
                    text: "b".to_string(),
                    tokens: vec![],
                    processing_time_ms: 15.0,
                },
            ],
            total_processing_time_ms: 25.0,
        };
        assert_eq!(batch_result.batch_id, "batch-7");
        assert_eq!(batch_result.responses.len(), 2);
        assert!((batch_result.total_processing_time_ms - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_model_info_fields() {
        let info = ModelInfo {
            name: "llama-3".to_string(),
            version: "1.0.0".to_string(),
            status: "active".to_string(),
            loaded_at: "2026-01-01T00:00:00Z".to_string(),
            memory_usage: 1024.0,
        };
        assert_eq!(info.name, "llama-3");
        assert_eq!(info.version, "1.0.0");
        assert_eq!(info.status, "active");
        assert!((info.memory_usage - 1024.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_info_fields() {
        let stats = StatsInfo {
            batching_stats: "{}".to_string(),
            caching_stats: "{}".to_string(),
            streaming_stats: "{}".to_string(),
            ha_stats: "{}".to_string(),
        };
        assert_eq!(stats.batching_stats, "{}");
        assert_eq!(stats.caching_stats, "{}");
    }

    #[test]
    fn test_service_health_info_fields() {
        let service_health = ServiceHealthInfo {
            batching: "healthy".to_string(),
            caching: "healthy".to_string(),
            streaming: "degraded".to_string(),
            failover: "healthy".to_string(),
        };
        assert_eq!(service_health.batching, "healthy");
        assert_eq!(service_health.streaming, "degraded");
    }

    #[test]
    fn test_detailed_health_info_has_system_health() {
        use crate::server::SystemHealthInfo;
        let detailed = DetailedHealthInfo {
            status: "healthy".to_string(),
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            version: "1.0.0".to_string(),
            uptime_seconds: 100.0,
            system_health: SystemHealthInfo {
                cpu_usage: Some(0.0),
                memory_usage: Some(0.0),
                disk_usage: Some(0.0),
                active_connections: 0,
            },
            services: ServiceHealthInfo {
                batching: "healthy".to_string(),
                caching: "healthy".to_string(),
                streaming: "healthy".to_string(),
                failover: "healthy".to_string(),
            },
        };
        assert_eq!(detailed.status, "healthy");
        assert!((detailed.uptime_seconds - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_create_schema_succeeds() {
        // Schema creation should not panic
        let _schema = create_schema();
    }

    #[test]
    fn test_inference_result_error_token() {
        let result = InferenceResult {
            request_id: "err-req".to_string(),
            text: "Error: something went wrong".to_string(),
            tokens: vec!["<error>".to_string()],
            processing_time_ms: 5.0,
        };
        assert!(result.text.starts_with("Error:"));
        assert_eq!(result.tokens[0], "<error>");
    }

    #[test]
    fn test_inference_result_zero_processing_time() {
        let result = InferenceResult {
            request_id: "fast".to_string(),
            text: "fast response".to_string(),
            tokens: vec!["fast".to_string(), "response".to_string()],
            processing_time_ms: 0.0,
        };
        assert!((result.processing_time_ms - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_model_info_loaded_at_field() {
        let info = ModelInfo {
            name: "gpt".to_string(),
            version: "2.0".to_string(),
            status: "active".to_string(),
            loaded_at: "2026-03-24T12:00:00Z".to_string(),
            memory_usage: 2048.5,
        };
        assert!(info.loaded_at.contains("2026"));
    }

    #[test]
    fn test_batch_inference_result_empty_responses() {
        let batch_result = BatchInferenceResult {
            batch_id: "empty-batch".to_string(),
            responses: vec![],
            total_processing_time_ms: 0.0,
        };
        assert!(batch_result.responses.is_empty());
        assert_eq!(batch_result.batch_id, "empty-batch");
    }

    #[test]
    fn test_health_info_uptime_zero() {
        let info = HealthInfo {
            status: "healthy".to_string(),
            timestamp: "now".to_string(),
            version: "0.1.1".to_string(),
            uptime_seconds: 0.0,
        };
        assert!((info.uptime_seconds - 0.0).abs() < 1e-9);
    }

    // ── Regression tests: the resolvers below used to answer with invented data ──

    use crate::ServerConfig;

    fn bare_server() -> Arc<TrustformerServer> {
        Arc::new(TrustformerServer::new(ServerConfig::default()))
    }

    async fn run_query(server: Arc<TrustformerServer>, query: &str) -> async_graphql::Response {
        create_schema()
            .execute(async_graphql::Request::new(query).data(create_context(server)))
            .await
    }

    /// Regression: the `models` resolver pushed a hardcoded `default` / `1.0.0`
    /// / `active` entry claiming 1 GB of memory, no matter what was loaded. A
    /// server that has loaded nothing must report an empty list.
    #[tokio::test]
    async fn models_query_reports_no_model_when_none_is_resident() {
        let server = bare_server();
        assert!(
            server.model_manager().list_loaded_models().is_empty(),
            "precondition: nothing is loaded"
        );

        let response = run_query(server, "{ models { name version status memoryUsage } }").await;
        assert!(
            response.errors.is_empty(),
            "query failed: {:?}",
            response.errors
        );

        let json = response.data.into_json().expect("response is JSON");
        let models = json["models"].as_array().expect("models is a list");
        assert!(
            models.is_empty(),
            "a server with no resident model must report none, got {models:?}"
        );
    }

    /// Regression: `ServiceHealthInfo` was four hardcoded `"healthy"` strings,
    /// so a server with no inference executor at all still reported the batching
    /// subsystem as healthy.
    #[tokio::test]
    async fn service_health_reports_no_model_rather_than_healthy() {
        let server = TrustformerServer::new(ServerConfig::default());
        assert!(
            !server.has_model(),
            "precondition: no executor is installed"
        );

        let health = service_health(&server, HealthStatus::Healthy).await;
        assert_eq!(health.batching, "no_model");
        assert_eq!(health.caching, "healthy");
        assert_eq!(health.streaming, "healthy");
        assert_eq!(health.failover, "healthy");
    }

    /// The failover label tracks the high-availability service's own verdict
    /// instead of being pinned to `"healthy"`.
    #[tokio::test]
    async fn service_health_failover_tracks_the_ha_verdict() {
        let server = TrustformerServer::new(ServerConfig::default());

        let degraded = service_health(&server, HealthStatus::Degraded).await;
        assert_eq!(degraded.failover, "degraded");

        let unhealthy = service_health(&server, HealthStatus::Unhealthy).await;
        assert_eq!(unhealthy.failover, "unhealthy");
    }

    /// Regression: `forceFailover` returned `true` for every target that was
    /// not the literal string `""` or `"invalid"`, without touching the
    /// failover manager. An unregistered node must be an error.
    #[tokio::test]
    async fn force_failover_to_an_unregistered_node_is_an_error() {
        let server = bare_server();
        let response = run_query(
            server,
            r#"mutation { forceFailover(targetNode: "node-a") }"#,
        )
        .await;

        assert!(
            !response.errors.is_empty(),
            "failing over to a node that was never registered must not report success"
        );
        assert!(
            response.errors[0].message.contains("node-a"),
            "the error must name the target: {}",
            response.errors[0].message
        );
    }

    /// And a genuinely registered, healthy node really does become primary.
    #[tokio::test]
    async fn force_failover_to_a_registered_node_switches_the_primary() {
        let server = bare_server();
        server
            .ha_service()
            .register_node("node-a".to_string(), "http://127.0.0.1:1".to_string())
            .await
            .expect("registration succeeds");
        server
            .ha_service()
            .register_node("node-b".to_string(), "http://127.0.0.1:2".to_string())
            .await
            .expect("registration succeeds");
        assert_eq!(
            server.ha_service().primary_node().await.as_deref(),
            Some("node-a")
        );

        let response = run_query(
            Arc::clone(&server),
            r#"mutation { forceFailover(targetNode: "node-b") }"#,
        )
        .await;
        assert!(
            response.errors.is_empty(),
            "mutation failed: {:?}",
            response.errors
        );

        let json = response.data.into_json().expect("response is JSON");
        assert_eq!(json["forceFailover"], serde_json::json!(true));
        assert_eq!(
            server.ha_service().primary_node().await.as_deref(),
            Some("node-b"),
            "the mutation must have moved the real primary, not just returned true"
        );
    }

    /// An empty target is rejected outright rather than silently accepted.
    #[tokio::test]
    async fn force_failover_rejects_an_empty_target() {
        let response = run_query(
            bare_server(),
            r#"mutation { forceFailover(targetNode: "  ") }"#,
        )
        .await;
        assert!(!response.errors.is_empty());
        assert!(response.errors[0].message.contains("must not be empty"));
    }

    /// The fixed resolvers must be reachable over the real HTTP route, not just
    /// through a schema built in a test. `/graphql` is mounted in the server's
    /// single route table, and a query through it must see the same honest
    /// answers.
    #[tokio::test]
    async fn graphql_route_serves_the_real_resolvers() {
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let router = TrustformerServer::new(ServerConfig::default()).create_test_router().await;
        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/graphql")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"query":"{ models { name } detailedHealth { services { batching } } }"}"#,
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
        assert!(body["errors"].is_null(), "query errored: {}", body);
        assert_eq!(
            body["data"]["models"],
            serde_json::json!([]),
            "a server with nothing loaded must not report a stand-in model"
        );
        assert_eq!(
            body["data"]["detailedHealth"]["services"]["batching"],
            serde_json::json!("no_model"),
            "a server with no executor must not claim its batching stack is healthy"
        );
    }

    #[test]
    fn model_status_labels_are_distinct_and_carry_the_failure_reason() {
        assert_eq!(model_status_label(&ModelStatus::Active), "active");
        assert_eq!(model_status_label(&ModelStatus::Unloaded), "unloaded");
        assert_eq!(
            model_status_label(&ModelStatus::Failed {
                error: "bad header".to_string()
            }),
            "failed: bad header"
        );
    }

    #[test]
    fn loaded_at_is_reconstructed_from_the_measured_elapsed_time() {
        let before = chrono::Utc::now();
        let rendered = loaded_at_rfc3339(std::time::Duration::from_secs(60));
        let parsed = chrono::DateTime::parse_from_rfc3339(&rendered)
            .expect("a real RFC 3339 timestamp")
            .with_timezone(&chrono::Utc);

        // Loaded a minute ago: the reconstructed instant must sit about 60s in
        // the past, not at "now" as the old hardcoded `Utc::now()` reported.
        let age = before - parsed;
        assert!(
            age.num_seconds() >= 59 && age.num_seconds() <= 62,
            "reconstructed load time is {age:?} behind now"
        );
    }
}
