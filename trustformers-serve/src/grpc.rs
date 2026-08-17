use crate::batching::aggregator::{ProcessingOutput, RequestInput};
use tonic::{Request, Response, Status};
use uuid::Uuid;

/// Current system memory usage as a ratio in `[0, 1]`.
///
/// Returns `None` when the platform does not report a total, so callers can
/// report "unknown" instead of assuming a number.
fn get_memory_usage() -> Option<f64> {
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    let total = system.total_memory();
    if total == 0 {
        return None;
    }
    Some(system.used_memory() as f64 / total as f64)
}

pub mod inference {
    tonic::include_proto!("trustformers.serve.v1");
}

use inference::{
    inference_service_server::{InferenceService, InferenceServiceServer},
    *,
};

use crate::{batching::DynamicBatchingService, ServerConfig};

pub struct InferenceServiceImpl {
    batching_service: DynamicBatchingService,
    _config: ServerConfig,
}

impl InferenceServiceImpl {
    pub fn new(batching_service: DynamicBatchingService, config: ServerConfig) -> Self {
        Self {
            batching_service,
            _config: config,
        }
    }

    pub fn into_service(self) -> InferenceServiceServer<Self> {
        InferenceServiceServer::new(self)
    }
}

#[tonic::async_trait]
impl InferenceService for InferenceServiceImpl {
    async fn predict(
        &self,
        request: Request<PredictRequest>,
    ) -> Result<Response<PredictResponse>, Status> {
        let req = request.into_inner();
        let request_id = Uuid::new_v4().to_string();

        let start_time = std::time::Instant::now();

        // Convert gRPC request to internal request format
        let internal_request = crate::batching::Request {
            id: crate::batching::RequestId::new(),
            input: RequestInput::Text {
                text: req.inputs.join(" "),
                max_length: None,
            },
            priority: crate::batching::config::Priority::Normal,
            submitted_at: std::time::Instant::now(),
            deadline: None,
            metadata: req.parameters,
        };

        // Process through batching service
        match self.batching_service.submit_request(internal_request).await {
            Ok(result) => {
                let latency_ms = start_time.elapsed().as_millis() as i64;

                let outputs = match &result.output {
                    ProcessingOutput::Text(text) => vec![text.clone()],
                    ProcessingOutput::Tokens(tokens) => {
                        vec![tokens
                            .iter()
                            .map(|t| t.to_string())
                            .collect::<Vec<String>>()
                            .join(" ")]
                    },
                    ProcessingOutput::Embeddings(embeddings) => {
                        vec![format!("embeddings: {} values", embeddings.len())]
                    },
                    ProcessingOutput::Classification(classes) => classes
                        .iter()
                        .map(|(class, score)| format!("{}: {:.4}", class, score))
                        .collect(),
                    ProcessingOutput::Error(error) => vec![format!("Error: {}", error)],
                };

                let response = PredictResponse {
                    outputs,
                    metadata: Some(PredictionMetadata {
                        latency_ms,
                        request_id,
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs() as i64,
                    }),
                };

                Ok(Response::new(response))
            },
            Err(e) => Err(Status::internal(format!("Processing failed: {}", e))),
        }
    }

    async fn batch_predict(
        &self,
        request: Request<BatchPredictRequest>,
    ) -> Result<Response<BatchPredictResponse>, Status> {
        let req = request.into_inner();
        let batch_id = Uuid::new_v4().to_string();
        let start_time = std::time::Instant::now();

        let batch_size = req.requests.len();

        // Submit every sub-request concurrently. Awaiting them one at a time
        // would serialise the batch and guarantee the aggregator never sees a
        // batch larger than one, defeating the endpoint.
        let futures = req.requests.into_iter().map(|predict_req| {
            let batching_service = self.batching_service.clone();
            async move {
                let internal_request = crate::batching::Request {
                    id: crate::batching::RequestId::new(),
                    input: RequestInput::Text {
                        text: predict_req.inputs.join(" "),
                        max_length: None,
                    },
                    priority: crate::batching::config::Priority::Normal,
                    submitted_at: std::time::Instant::now(),
                    deadline: None,
                    metadata: predict_req.parameters,
                };
                // Each sub-request measures its own latency.
                let submitted_at = std::time::Instant::now();
                let result = batching_service.submit_request(internal_request).await;
                (result, submitted_at.elapsed())
            }
        });

        let outcomes = futures::future::join_all(futures).await;

        let mut responses = Vec::with_capacity(batch_size);
        for (result, elapsed) in outcomes {
            match result {
                Ok(result) => {
                    let outputs = match &result.output {
                        ProcessingOutput::Text(text) => vec![text.clone()],
                        ProcessingOutput::Tokens(tokens) => {
                            vec![tokens
                                .iter()
                                .map(|t| t.to_string())
                                .collect::<Vec<String>>()
                                .join(" ")]
                        },
                        ProcessingOutput::Embeddings(embeddings) => {
                            vec![format!("embeddings: {} values", embeddings.len())]
                        },
                        ProcessingOutput::Classification(classes) => classes
                            .iter()
                            .map(|(class, score)| format!("{}: {:.4}", class, score))
                            .collect(),
                        ProcessingOutput::Error(error) => vec![format!("Error: {}", error)],
                    };

                    responses.push(PredictResponse {
                        outputs,
                        metadata: Some(PredictionMetadata {
                            latency_ms: elapsed.as_millis() as i64,
                            request_id: Uuid::new_v4().to_string(),
                            timestamp: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs() as i64,
                        }),
                    });
                },
                Err(e) => {
                    return Err(Status::internal(format!("Batch processing failed: {}", e)));
                },
            }
        }

        let total_latency_ms = start_time.elapsed().as_millis() as i64;

        let response = BatchPredictResponse {
            responses,
            batch_metadata: Some(BatchMetadata {
                batch_size: batch_size as i32,
                total_latency_ms,
                batch_id,
            }),
        };

        Ok(Response::new(response))
    }

    async fn get_model_info(
        &self,
        _request: Request<GetModelInfoRequest>,
    ) -> Result<Response<GetModelInfoResponse>, Status> {
        // Get actual model info from configuration and batching service
        let batching_config = &self._config.batching_config;
        let validation_config = &self._config.validation_config;

        // Determine device based on configuration
        let device =
            if self._config.gpu_scheduler_config.enabled { "gpu" } else { "cpu" }.to_string();

        // Get model name from configuration or default
        let model_name = self._config.model_config.model_name.clone();

        // Get version from environment or default
        let version = std::env::var("TRUSTFORMERS_VERSION").unwrap_or_else(|_| "1.0.0".to_string());

        // Create comprehensive model description
        let mut description_parts = vec!["TrustformeRS transformer model".to_string()];

        if self._config.gpu_scheduler_config.enabled {
            let gpu_config = &self._config.gpu_scheduler_config;
            description_parts.push(format!(
                "GPU scheduling enabled with {} algorithm",
                match gpu_config.scheduling_algorithm {
                    crate::gpu_scheduler::SchedulingAlgorithm::FirstFit => "First-Fit",
                    crate::gpu_scheduler::SchedulingAlgorithm::BestFit => "Best-Fit",
                    crate::gpu_scheduler::SchedulingAlgorithm::WorstFit => "Worst-Fit",
                    crate::gpu_scheduler::SchedulingAlgorithm::RoundRobin => "Round-Robin",
                    crate::gpu_scheduler::SchedulingAlgorithm::Priority => "Priority-based",
                    crate::gpu_scheduler::SchedulingAlgorithm::LoadBalanced => "Load-Balanced",
                }
            ));
        }

        if batching_config.enable_adaptive_batching {
            description_parts.push(format!(
                "Dynamic batching enabled (max size: {})",
                batching_config.max_batch_size
            ));
        }

        // Check if caching is configured with meaningful settings
        let caching = &self._config.caching_config;
        if caching.result_cache.max_entries > 0 {
            description_parts.push("Result caching enabled".to_string());
        }

        let description = description_parts.join(". ");

        // Determine max sequence length from validation config or default
        let max_sequence_length = validation_config.max_text_length as i32;

        let model_info = ModelInfo {
            name: model_name,
            version,
            description,
            max_sequence_length,
            device,
        };

        let response = GetModelInfoResponse {
            model_info: Some(model_info),
        };

        Ok(Response::new(response))
    }

    async fn health_check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        // Perform comprehensive health checks
        let mut health_issues = Vec::new();
        let mut overall_status = HealthStatus::Serving;

        // The decisive signal: can this process actually answer an inference
        // request? A batching stack without a model cannot, whatever its
        // counters say.
        if !self.batching_service.has_model() {
            health_issues.push("No model is configured on the batch executor".to_string());
            overall_status = HealthStatus::NotServing;
        }

        // Check batching service health. An idle server is healthy; only a
        // saturated queue is not.
        let stats = self.batching_service.get_stats().await;
        if stats.aggregator_stats.pending_requests > 1000 {
            health_issues.push(format!(
                "High pending request count: {}",
                stats.aggregator_stats.pending_requests
            ));
            overall_status = HealthStatus::NotServing;
        }
        if stats.processor_stats.total_batches > 0 && stats.processor_stats.success_rate < 0.5 {
            health_issues.push(format!(
                "Batch success rate is {:.0}%",
                stats.processor_stats.success_rate * 100.0
            ));
            overall_status = HealthStatus::NotServing;
        }

        // Check system resources. An unavailable measurement is reported as
        // such rather than being assumed healthy or unhealthy.
        match get_memory_usage() {
            Some(memory_usage) if memory_usage > 0.9 => {
                health_issues.push(format!("High memory usage: {:.0}%", memory_usage * 100.0));
                if overall_status == HealthStatus::Serving {
                    overall_status = HealthStatus::NotServing;
                }
            },
            Some(_) => {},
            None => {
                health_issues.push("System memory usage is unavailable".to_string());
            },
        }

        // Prepare health check response
        let message = if health_issues.is_empty() {
            "Service is healthy and operating normally".to_string()
        } else {
            format!("Health issues detected: {}", health_issues.join(", "))
        };

        let response = HealthCheckResponse {
            status: overall_status as i32,
            message,
        };

        Ok(Response::new(response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_memory_usage_is_measured_or_absent() {
        match get_memory_usage() {
            Some(usage) => {
                assert!(usage.is_finite(), "memory usage should be a finite number");
                assert!(!usage.is_nan(), "memory usage should not be NaN");
                assert!(
                    (0.0..=1.0).contains(&usage),
                    "memory usage should be a ratio, got {usage}"
                );
            },
            None => {
                // Platform reports no total memory; reporting "unknown" is the
                // honest answer and is explicitly allowed.
            },
        }
    }

    /// Regression: the probe must not fall back to a hardcoded 0.5.
    #[test]
    fn test_get_memory_usage_is_not_a_constant_placeholder() {
        let usage = get_memory_usage().expect("this platform reports total memory");
        let mut system = sysinfo::System::new();
        system.refresh_memory();
        let expected = system.used_memory() as f64 / system.total_memory() as f64;
        // Sampled a moment apart, so allow drift but not a fixed constant.
        assert!(
            (usage - expected).abs() < 0.2,
            "reported {usage} is unrelated to the measured {expected}"
        );
    }

    #[test]
    fn test_health_issue_message_format() {
        // Verify the health message format logic
        let health_issues: Vec<String> = vec![];
        let message = if health_issues.is_empty() {
            "Service is healthy and operating normally".to_string()
        } else {
            format!("Health issues detected: {}", health_issues.join(", "))
        };
        assert_eq!(message, "Service is healthy and operating normally");
    }

    #[test]
    fn test_health_issue_message_with_issues() {
        let health_issues = vec![
            "High pending request count".to_string(),
            "High memory usage".to_string(),
        ];
        let message = if health_issues.is_empty() {
            "Service is healthy and operating normally".to_string()
        } else {
            format!("Health issues detected: {}", health_issues.join(", "))
        };
        assert!(message.starts_with("Health issues detected:"));
        assert!(message.contains("High pending request count"));
        assert!(message.contains("High memory usage"));
    }

    #[test]
    fn test_health_status_enum_values_match_proto() {
        // proto/inference.proto defines the wire values explicitly:
        //   UNKNOWN = 0; SERVING = 1; NOT_SERVING = 2; SERVICE_UNKNOWN = 3;
        assert_eq!(HealthStatus::Unknown as i32, 0);
        assert_eq!(HealthStatus::Serving as i32, 1);
        assert_eq!(HealthStatus::NotServing as i32, 2);
        assert_eq!(HealthStatus::ServiceUnknown as i32, 3);
    }

    #[test]
    fn test_health_status_not_serving_is_nonzero() {
        let status = HealthStatus::NotServing;
        assert_ne!(status as i32, 0);
    }

    #[test]
    fn test_health_status_unknown_distinct_from_serving() {
        let serving = HealthStatus::Serving as i32;
        let unknown = HealthStatus::Unknown as i32;
        assert_ne!(serving, unknown);
    }

    #[test]
    fn test_memory_threshold_check_low_usage() {
        // Usage below 0.9 should not add health issue
        let usage = 0.5_f64;
        let mut issues: Vec<String> = Vec::new();
        if usage > 0.9 {
            issues.push("High memory usage".to_string());
        }
        assert!(issues.is_empty());
    }

    #[test]
    fn test_memory_threshold_check_high_usage() {
        // Usage above 0.9 should add health issue
        let usage = 0.95_f64;
        let mut issues: Vec<String> = Vec::new();
        if usage > 0.9 {
            issues.push("High memory usage".to_string());
        }
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0], "High memory usage");
    }

    #[test]
    fn test_memory_threshold_at_exactly_09() {
        // Usage at exactly 0.9 should NOT add the issue (condition is > 0.9)
        let usage = 0.9_f64;
        let mut issues: Vec<String> = Vec::new();
        if usage > 0.9 {
            issues.push("High memory usage".to_string());
        }
        assert!(issues.is_empty());
    }

    #[test]
    fn test_pending_requests_threshold_check_below_1000() {
        let pending: u64 = 999;
        let mut issues: Vec<String> = Vec::new();
        if pending > 1000 {
            issues.push("High pending request count".to_string());
        }
        assert!(issues.is_empty());
    }

    #[test]
    fn test_pending_requests_threshold_check_above_1000() {
        let pending: u64 = 1001;
        let mut issues: Vec<String> = Vec::new();
        if pending > 1000 {
            issues.push("High pending request count".to_string());
        }
        assert_eq!(issues.len(), 1);
    }

    #[test]
    fn test_output_format_text_join() {
        // Test how token outputs are joined (as done in predict handler)
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let text = tokens.iter().map(|t| t.to_string()).collect::<Vec<String>>().join(" ");
        assert_eq!(text, "1 2 3 4 5");
    }

    #[test]
    fn test_output_format_empty_tokens() {
        let tokens: Vec<u32> = vec![];
        let text = tokens.iter().map(|t| t.to_string()).collect::<Vec<String>>().join(" ");
        assert_eq!(text, "");
    }

    #[test]
    fn test_inference_service_impl_construction() {
        use crate::batching::DynamicBatchingService;
        use crate::ServerConfig;

        let config = ServerConfig::default();
        let batching_service = DynamicBatchingService::new(config.batching_config.clone());
        let _service = InferenceServiceImpl::new(batching_service, config);
        // Construction should not panic
    }

    #[test]
    fn test_inference_service_into_service() {
        use crate::batching::DynamicBatchingService;
        use crate::ServerConfig;

        let config = ServerConfig::default();
        let batching_service = DynamicBatchingService::new(config.batching_config.clone());
        let service = InferenceServiceImpl::new(batching_service, config);
        let _server = service.into_service();
        // into_service should not panic
    }

    /// Regression: `health_check` used to compare the absolute Unix timestamp
    /// against 300 and therefore reported `NOT_SERVING` on every call. It must
    /// now reflect the real serving state: `NOT_SERVING` only while no model is
    /// configured, `SERVING` once one is.
    #[tokio::test]
    async fn health_check_reflects_real_serving_state() {
        use crate::batching::model_executor::{ByteTokenizer, Gpt2BatchModel};
        use crate::batching::{DynamicBatchingService, ModelBatchExecutor};
        use crate::ServerConfig;
        use std::sync::Arc;
        use trustformers_models::gpt2::Gpt2Config;

        let config = ServerConfig::default();

        // Without a model the service must say so.
        let without_model = InferenceServiceImpl::new(
            DynamicBatchingService::new(config.batching_config.clone()),
            config.clone(),
        );
        let response = without_model
            .health_check(Request::new(HealthCheckRequest::default()))
            .await
            .expect("health check must answer")
            .into_inner();
        assert_eq!(response.status, HealthStatus::NotServing as i32);
        assert!(response.message.contains("No model is configured"));

        // With a real model wired in, an idle server is healthy.
        let gpt2_config = Gpt2Config {
            vocab_size: ByteTokenizer::VOCAB_SIZE,
            n_positions: 16,
            n_embd: 8,
            n_layer: 1,
            n_head: 2,
            n_inner: Some(16),
            resid_pdrop: 0.0,
            embd_pdrop: 0.0,
            attn_pdrop: 0.0,
            ..Gpt2Config::default()
        };
        let model = Arc::new(Gpt2BatchModel::untrained(gpt2_config).expect("model builds"));
        let executor =
            Arc::new(ModelBatchExecutor::new(model).with_tokenizer(Arc::new(ByteTokenizer)));
        let with_model = InferenceServiceImpl::new(
            DynamicBatchingService::with_executor(config.batching_config.clone(), executor),
            config,
        );
        let response = with_model
            .health_check(Request::new(HealthCheckRequest::default()))
            .await
            .expect("health check must answer")
            .into_inner();
        assert_eq!(
            response.status,
            HealthStatus::Serving as i32,
            "an idle but model-backed server must report SERVING, got: {}",
            response.message
        );
    }
}
