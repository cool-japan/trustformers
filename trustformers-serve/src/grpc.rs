use crate::batching::aggregator::{ProcessingOutput, RequestInput};
use tonic::{Request, Response, Status};
use uuid::Uuid;

/// Get current memory usage as a ratio (0.0 to 1.0)
fn get_memory_usage() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
            let mut total_memory = 0u64;
            let mut available_memory = 0u64;

            for line in contents.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        total_memory = value.parse().unwrap_or(0);
                    }
                } else if line.starts_with("MemAvailable:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        available_memory = value.parse().unwrap_or(0);
                    }
                }
            }

            if total_memory > 0 && available_memory <= total_memory {
                return (total_memory - available_memory) as f64 / total_memory as f64;
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("vm_stat").output() {
            if let Ok(vm_stat) = String::from_utf8(output.stdout) {
                let mut pages_free = 0u64;
                let mut pages_wired = 0u64;
                let mut pages_active = 0u64;
                let mut pages_inactive = 0u64;

                for line in vm_stat.lines() {
                    if line.contains("Pages free:") {
                        if let Some(value) = line.split(':').nth(1) {
                            pages_free = value.trim().trim_end_matches('.').parse().unwrap_or(0);
                        }
                    } else if line.contains("Pages wired down:") {
                        if let Some(value) = line.split(':').nth(1) {
                            pages_wired = value.trim().trim_end_matches('.').parse().unwrap_or(0);
                        }
                    } else if line.contains("Pages active:") {
                        if let Some(value) = line.split(':').nth(1) {
                            pages_active = value.trim().trim_end_matches('.').parse().unwrap_or(0);
                        }
                    } else if line.contains("Pages inactive:") {
                        if let Some(value) = line.split(':').nth(1) {
                            pages_inactive =
                                value.trim().trim_end_matches('.').parse().unwrap_or(0);
                        }
                    }
                }

                let total_pages = pages_free + pages_wired + pages_active + pages_inactive;
                if total_pages > 0 {
                    return (pages_wired + pages_active + pages_inactive) as f64
                        / total_pages as f64;
                }
            }
        }
    }

    // Fallback: assume moderate memory usage
    0.5
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

        let mut responses = Vec::new();

        let batch_size = req.requests.len();
        for predict_req in req.requests {
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

            match self.batching_service.submit_request(internal_request).await {
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
                            latency_ms: 0, // Individual latency not tracked in batch
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
            let ref gpu_config = self._config.gpu_scheduler_config;
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
        let ref caching = self._config.caching_config;
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

        // Check batching service health
        let stats = self.batching_service.get_stats().await;
        {
            // Check if batching service has critical issues
            if stats.aggregator_stats.pending_requests > 1000 {
                health_issues.push("High pending request count".to_string());
                overall_status = HealthStatus::NotServing;
            }
            if stats.processor_stats.total_requests == 0
                && std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
                    > 300
            // Service running for more than 5 minutes
            {
                health_issues.push("No requests processed recently".to_string());
                overall_status = HealthStatus::NotServing;
            }
        }

        // Check system resources
        let memory_usage = get_memory_usage();
        if memory_usage > 0.9 {
            // More than 90% memory usage
            health_issues.push("High memory usage".to_string());
            if overall_status == HealthStatus::Serving {
                overall_status = HealthStatus::NotServing;
            }
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
